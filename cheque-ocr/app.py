"""
FastAPI server:  POST /predict  →  JSON with amount_words, amount, date, payee

ENV:
  YOLO_WEIGHTS   absolute path or URL to your .pt/.ptm file
"""
import os, io, re
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request           # +Request
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dateparser import parse as dp

from fastapi.responses import HTMLResponse                      # new
from fastapi.staticfiles import StaticFiles                     # new
from fastapi.templating import Jinja2Templates                  # new

# ---------- 1.  load models once -------------------------------------------
WEIGHTS = os.getenv("YOLO_WEIGHTS", "/weights/best_yolo11n.pt")
yolo    = YOLO(WEIGHTS)

proc    = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr   = (VisionEncoderDecoderModel
           .from_pretrained("microsoft/trocr-base-handwritten")
           .to("cpu").eval())                     # GPU later if Lambda not used

FIELD_NAMES     = ["amount", "amount_words", "date", "payee"]
KEEP_CLASS_IDS  = [1, 2, 3, 4]                    # maps 1→amount, etc.

# ---------- 2.  tiny helpers -------------------------------------------------
def to_pil(img_cv: np.ndarray) -> Image.Image:
    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_cv)

def ocr_trocr(img_cv: np.ndarray) -> str:
    with torch.no_grad():
        pv  = proc(to_pil(img_cv), return_tensors="pt").pixel_values
        ids = trocr.generate(pv, max_length=64)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return txt.strip()

_num_re = re.compile(r"[^0-9.,]")
def clean_amount(raw: str) -> str:
    txt = _num_re.sub("", raw).replace(",", "").replace(" ", "")
    if txt and "." not in txt:
        txt += ".00"
    return txt or ""

def clean_date(raw: str) -> str:
    txt = re.sub(r"[^\dA-Za-z/.\- ]", " ", raw).strip()
    dt  = dp(txt, settings={"DATE_ORDER": "DMY"})
    return dt.strftime("%Y-%m-%d") if dt else txt

# ---------- 3.  FastAPI ------------------------------------------------------
app = FastAPI(title="Cheque-OCR Service")

# serve static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)                       # UI page
def ui(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request": request})

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    content = await file.read()
    buf     = np.frombuffer(content, np.uint8)
    img     = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    res = {k: "" for k in FIELD_NAMES}

    for box in yolo(img, conf=0.2, classes=KEEP_CLASS_IDS)[0].boxes:
        cls   = int(box.cls[0])
        field = FIELD_NAMES[KEEP_CLASS_IDS.index(cls)]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        crop  = img[y1:y2, x1:x2]

        txt = ocr_trocr(crop)
        if field == "amount":
            txt = clean_amount(txt)
        elif field == "date":
            txt = clean_date(txt)
        res[field] = txt

    return {"file": file.filename, **res}
