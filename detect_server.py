import io, base64
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch

from dbnet_text_detector import DBNet, DBDetector  # <-- from your Step1 script

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = DBNet().to(device)
model.eval()
detector = DBDetector(model, device=device)

app = FastAPI()

def read_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil = read_image(img_bytes)
    results = detector.detect(pil, max_side=1024, bin_thresh=0.4, unclip=1.5)
    return JSONResponse({"detections": results})
