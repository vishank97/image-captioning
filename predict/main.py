from typing  import List
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from predict.predict import predict_step

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Image Captioning model inference!"}

class ImageCaption(BaseModel):
    images: list
    is_url: bool

@app.post("/predict")

async def predict(ic: ImageCaption):
    return predict_step(ic.images, ic.is_url)


async def predict_upload(files: List[UploadFile]):
    return predict_step([file.filename for file in files],False)