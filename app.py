import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, field_validator, ValidationInfo
from dotenv import load_dotenv
from io import BytesIO
from fastai.vision.all import *
import warnings

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DATABASE_PATH = os.getenv("DATABASE_PATH", "./app.db")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# --- DB helpers --------------------------------------------------------------

def get_db_conn():
    """
    Returns a sqlite3.Connection with Row factory and foreign keys enabled.
    Use `with get_db_conn() as conn:` to auto-commit/close.
    """
    conn = sqlite3.connect(DATABASE_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_db_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT UNIQUE,
                hashed_password TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                state TEXT,
                city TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS crops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_crops (
                user_id INTEGER NOT NULL,
                crop_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, crop_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (crop_id) REFERENCES crops(id) ON DELETE CASCADE
            );
            """
        )

init_db()

# --- Pydantic schemas -------------------------------------------------------

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    state: Optional[str] = None
    city: Optional[str] = None
    phone: Optional[str] = None
    crop_preferences: Optional[List[str]] = []

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v):
        if v is None or v == "":
            return None
        # Remove spaces, parentheses, hyphens; keep optional leading '+'
        v = str(v).strip()
        keep_plus = v.startswith("+")
        digits = "".join(ch for ch in v if ch.isdigit())
        if not (7 <= len(digits) <= 15):
            raise ValueError("phone number must contain 7 to 15 digits")
        return ("+" + digits) if keep_plus else digits

    @field_validator("crop_preferences", mode="before")
    @classmethod
    def normalize_crops(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # allow comma-separated string
            lst = [s.strip() for s in v.split(",") if s.strip()]
            return list(dict.fromkeys(lst))  # preserve order, dedupe
        if isinstance(v, list):
            cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
            return list(dict.fromkeys(cleaned))
        raise ValueError("crop_preferences must be a list of strings or comma-separated string")

class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_active: bool
    state: Optional[str] = None
    city: Optional[str] = None
    phone: Optional[str] = None
    crop_preferences: List[str] = []

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- Auth helpers -----------------------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            return None
        return TokenData(username=username)
    except JWTError:
        return None

# --- Crop / user helpers ---------------------------------------------------

def read_imagefile(file) -> PILImage:
    return PILImage.create(BytesIO(file))

def get_or_create_crop(conn: sqlite3.Connection, name: str) -> Optional[int]:
    """
    Returns crop.id. Normalizes name to lower-case trimmed.
    """
    name_clean = name.strip().lower()
    if not name_clean:
        return None
    cur = conn.execute("SELECT id FROM crops WHERE name = ?", (name_clean,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO crops (name) VALUES (?)", (name_clean,))
    return cur.lastrowid

def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_db_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cur.fetchone()

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_db_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        return cur.fetchone()

def get_user_by_phone(phone: str) -> Optional[sqlite3.Row]:
    with get_db_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,))
        return cur.fetchone()

def get_user_with_prefs(username: str) -> Optional[Tuple[sqlite3.Row, List[str]]]:
    with get_db_conn() as conn:
        user_row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not user_row:
            return None
        prefs = [r["name"] for r in conn.execute(
            "SELECT c.name FROM crops c JOIN user_crops uc ON c.id = uc.crop_id WHERE uc.user_id = ? ORDER BY c.name",
            (user_row["id"],)
        ).fetchall()]
        return user_row, prefs

def create_user(username: str, email: str, password: str,
                state: Optional[str], city: Optional[str], phone: Optional[str],
                crop_preferences: List[str]) -> int:
    hashed_pw = get_password_hash(password)
    with get_db_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, email, phone, hashed_password, state, city) VALUES (?, ?, ?, ?, ?, ?)",
                (username, email, phone, hashed_pw, state, city),
            )
            user_id = cur.lastrowid
            # Insert crop preferences
            crops = []
            for crop in crop_preferences or []:
                if not crop or not crop.strip():
                    continue
                crop_id = get_or_create_crop(conn, crop)
                if crop_id:
                    crops.append(crop_id)
            # Link crops
            for cid in set(crops):
                conn.execute("INSERT OR IGNORE INTO user_crops (user_id, crop_id) VALUES (?, ?)", (user_id, cid))
            conn.commit()
            return user_id
        except sqlite3.IntegrityError as e:
            # bubble up a clearer message
            raise HTTPException(status_code=400, detail=f"Database integrity error: {e}")

# --- FastAPI app ------------------------------------------------------------

app = FastAPI(title="AgroAI - Crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
"*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Load model with warning handling
def load_model_safely():
    """Load the FastAI model with proper warning handling"""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*load_learner.*")
            learn = load_learner("crop_disease_model.pkl")
            return learn
    except FileNotFoundError:
        print("Warning: Model file 'crop_disease_model.pkl' not found. Please ensure the model file exists.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

learn = load_model_safely()
disease_info = {
    # ---------- WHEAT ----------
    "Wheat__Healthy": {
        "crop": "Wheat",
        "description": "Healthy wheat plant without any visible disease symptoms.",
        "symptoms": [],
        "treatment": [],
        "prevention": [
            "Maintain balanced soil fertility",
            "Ensure proper irrigation",
            "Use certified disease-free seeds"
        ]
    },
    "Wheat__Brown_Rust": {
        "crop": "Wheat",
        "description": "Brown rust is caused by Puccinia triticina, a major foliar disease of wheat.",
        "symptoms": [
            "Small, circular to oval brown pustules on leaves",
            "Yellowing around lesions",
            "Reduced grain yield and quality"
        ],
        "treatment": [
            "Apply fungicides such as Propiconazole or Tebuconazole",
            "Early spray during initial infection"
        ],
        "prevention": [
            "Plant resistant wheat varieties",
            "Destroy volunteer wheat plants (green bridge)",
            "Rotate crops to break pathogen cycle"
        ]
    },

    # ---------- RICE ----------
    "Rice__Healthy": {
        "crop": "Rice",
        "description": "Healthy rice plant with no disease or insect damage.",
        "symptoms": [],
        "treatment": [],
        "prevention": [
            "Use disease-free seeds",
            "Ensure balanced fertilizer application",
            "Maintain proper water management"
        ]
    },
    "Rice__Leaf_Blast": {
        "crop": "Rice",
        "description": "Leaf blast caused by Magnaporthe oryzae affects rice leaves.",
        "symptoms": [
            "Diamond-shaped lesions on leaves",
            "Grayish centers with brown borders",
            "Rapid drying of leaves in severe cases"
        ],
        "treatment": [
            "Apply fungicides such as Tricyclazole or Isoprothiolane",
            "Remove infected plants to reduce spread"
        ],
        "prevention": [
            "Grow resistant rice varieties",
            "Avoid excess nitrogen fertilizer",
            "Ensure proper spacing to reduce humidity"
        ]
    },
    "Rice__Neck_Blast": {
        "crop": "Rice",
        "description": "Neck blast affects the panicle neck, caused by Magnaporthe oryzae.",
        "symptoms": [
            "Blackening and rotting at panicle neck",
            "Panicles dry prematurely",
            "Grain filling is reduced or absent"
        ],
        "treatment": [
            "Spray fungicides at heading stage (Tricyclazole, Carbendazim)",
            "Destroy infected residues"
        ],
        "prevention": [
            "Cultivate resistant varieties",
            "Balanced fertilizer use",
            "Crop rotation with non-host crops"
        ]
    },

    # ---------- POTATO ----------
    "Potato__Healthy": {
        "crop": "Potato",
        "description": "Healthy potato leaf without disease or pest damage.",
        "symptoms": [],
        "treatment": [],
        "prevention": [
            "Plant certified seed potatoes",
            "Follow crop rotation",
            "Maintain field hygiene"
        ]
    },
    "Potato__Late_Blight": {
        "crop": "Potato",
        "description": "Late blight, caused by Phytophthora infestans, is a devastating potato disease.",
        "symptoms": [
            "Dark brown to black lesions on leaves with pale green halos",
            "White fungal growth under moist conditions",
            "Tuber rot in severe cases"
        ],
        "treatment": [
            "Apply fungicides such as Mancozeb or Metalaxyl",
            "Remove and destroy infected plants"
        ],
        "prevention": [
            "Plant resistant potato varieties",
            "Avoid overhead irrigation",
            "Ensure proper crop rotation"
        ]
    },
    "Potato__Early_Blight": {
        "crop": "Potato",
        "description": "Early blight, caused by Alternaria solani, is a common fungal disease of potato.",
        "symptoms": [
            "Small dark brown spots with concentric rings on leaves",
            "Premature leaf drop",
            "Reduced tuber yield"
        ],
        "treatment": [
            "Apply fungicides like Chlorothalonil or Copper-based sprays",
            "Remove infected leaves early"
        ],
        "prevention": [
            "Use resistant varieties",
            "Practice crop rotation",
            "Avoid excessive nitrogen fertilizer"
        ]
    }
}


@app.get("/")
def read_root():
    return {
        "message": "Welcome to AgroAI - Crop Disease Detection API",
        "status": "running",
        "model_loaded": learn is not None
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if learn is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check server configuration."
        )
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and process the image
        image_data = await file.read()
        img = read_imagefile(image_data)

        # Make prediction
        pred, pred_idx, probs = learn.predict(img)
        pred_str = str(pred)
        info = disease_info.get(pred_str, {
            "crop": "",
            "description": "",
            "symptoms": [],
            "treatment": [],
            "prevention": []
        })

        return {
            "prediction": pred_str,
            "confidence": float(probs[pred_idx]),
            "filename": file.filename,
            **info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/register", response_model=UserOut)
def register(user_in: UserCreate):
    # uniqueness checks
    if get_user_by_username(user_in.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if get_user_by_email(user_in.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if user_in.phone and get_user_by_phone(user_in.phone):
        raise HTTPException(status_code=400, detail="Phone number already registered")

    user_id = create_user(
        username=user_in.username,
        email=user_in.email,
        password=user_in.password,
        state=user_in.state,
        city=user_in.city,
        phone=user_in.phone,
        crop_preferences=user_in.crop_preferences,
    )

    # fetch created user + prefs to return
    row, prefs = get_user_with_prefs(user_in.username)
    return UserOut(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        is_active=bool(row["is_active"]),
        state=row["state"],
        city=row["city"],
        phone=row["phone"],
        crop_preferences=prefs,
    )

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    row = get_user_by_username(form_data.username)
    print(form_data.username, form_data.password,  row)
    if not row:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not verify_password(form_data.password, row["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": row["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)) -> Tuple[sqlite3.Row, List[str]]:
    token_data = decode_access_token(token)
    if not token_data or not token_data.username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication")
    up = get_user_with_prefs(token_data.username)
    if not up:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return up  # (row, prefs)

@app.get("/users/me", response_model=UserOut)
def read_users_me(current_user=Depends(get_current_user)):
    row, prefs = current_user
    return UserOut(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        is_active=bool(row["is_active"]),
        state=row["state"],
        city=row["city"],
        phone=row["phone"],
        crop_preferences=prefs,
    )

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_status": "loaded" if learn is not None else "not_loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)