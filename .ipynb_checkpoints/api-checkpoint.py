from fastapi import FastAPI, UploadFile, File
from fastai.vision.all import *
import uvicorn
from io import BytesIO
from PIL import Image

# Load model at startup
learn = load_learner("crop_disease_model.pkl")

app = FastAPI()

def read_imagefile(file) -> PILImage:
    return PILImage.create(BytesIO(file))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = read_imagefile(await file.read())

    # Prediction
    pred, pred_idx, probs = learn.predict(img)

    return {
        "prediction": str(pred),
        "confidence": float(probs[pred_idx])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
