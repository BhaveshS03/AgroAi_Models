from fastai.vision.all import *

# Create dummy DataLoaders
path = Path("../ModelTraining/Crop___Disease")


dls = ImageDataLoaders.from_folder(
    path,
    valid_pct=0.2,     # 80% train, 20% validation
    seed=42,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224),
    bs=32,
    num_workers=0 
)
dls.show_batch(max_n=9, figsize=(8,8))

learn = vision_learner(dls, resnet50, metrics=accuracy)

learn.fine_tune(5)