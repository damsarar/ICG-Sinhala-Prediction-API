import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from keras.preprocessing.image import load_img
from keras.utils.data_utils import get_file
from process_image import encode
from generate_caption import generate_caption


app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get("/")
async def root():
    return {"message": "Welcome to Image Caption Generator for Sinhala"}


@app.post("/caption/generate")
async def generate_caption(image_url: str = ""):
    if image_url == "":
        return {"message": "No URL provided"}

    img_path = get_file(origin=image_url)

    img = encode(img_path).reshape((1, 2048))
    ret = generate_caption(img, 42)

    print(ret)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
