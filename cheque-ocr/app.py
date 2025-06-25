"""
Cheque-OCR micro-service
────────────────────────
* YOLOv8 detects the 4 key fields on a cheque image
* TrOCR (base) reads the cropped text
* FastAPI  /predict  → JSON  (text + crops as base-64 PNGs)
* Gradio UI at `/`   → shows: annotated cheque, crop gallery, JSON

Run locally:   python app.py
Docker:        see Dockerfile below
"""

import os, re, io, base64, json, cv2, torch, uvicorn, numpy as np
from typing import Dict
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dateparser import parse as date_parse
from fastapi import FastAPI, UploadFile, File
import gradio as gr

# ──────────────────────────────────────────────────────────────────────────
# 0.  Load models once (cold start: TrOCR checkpoint ~370 MB)
# ──────────────────────────────────────────────────────────────────────────
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "/weights/best_yolo11n.pt")
yolo  = YOLO(YOLO_WEIGHTS)

proc  = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten").to("cpu").eval()

FIELDS = ["amount", "amount_words", "date", "payee"]
KEEP   = [1, 2, 3, 4]                       # YOLO class → field index

# ──────────────────────────────────────────────────────────────────────────
# 1.  Utility helpers
# ──────────────────────────────────────────────────────────────────────────
def np_bgr_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)

def ocr_trocr(arr_bgr: np.ndarray) -> str:
    with torch.no_grad():
        ids = trocr.generate(
            proc(np_bgr_to_pil(arr_bgr), return_tensors="pt").pixel_values,
            max_length=64)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

def to_b64_png(arr_bgr: np.ndarray) -> str:
    _, png = cv2.imencode(".png", arr_bgr)
    return ("data:image/png;base64," +
            base64.b64encode(png).decode("utf-8"))

_num_re = re.compile(r"[^0-9.,]")
def clean_amount(raw: str) -> str:
    t = _num_re.sub("", raw).replace(",", "").replace(" ", "")
    return (t + ".00") if (t and "." not in t) else t

def clean_date(raw: str) -> str:
    txt = re.sub(r"[^\dA-Za-z/.\- ]", " ", raw).strip()
    dt  = date_parse(txt, settings={"DATE_ORDER": "DMY"})
    return dt.strftime("%Y-%m-%d") if dt else txt

# ──────────────────────────────────────────────────────────────────────────
# 2.  Core pipeline – returns text, crops (b64 & rgb), annotated cheque
# ──────────────────────────────────────────────────────────────────────────
def cheque_ocr(img_bgr: np.ndarray):
    preds: Dict[str, str] = {k: "" for k in FIELDS}
    crops_b64: Dict[str, str] = {}
    crops_rgb  = []
    vis = img_bgr.copy()

    for box in yolo(img_bgr, conf=0.2, classes=KEEP)[0].boxes:
        field = FIELDS[KEEP.index(int(box.cls[0]))]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_bgr[y1:y2, x1:x2]

        txt = ocr_trocr(crop)
        if field == "amount": txt = clean_amount(txt)
        if field == "date":   txt = clean_date(txt)

        preds[field] = txt
        crops_b64[field] = to_b64_png(crop)
        crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # draw on vis
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(vis, field, (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    return preds, crops_b64, crops_rgb, vis_rgb

# ──────────────────────────────────────────────────────────────────────────
# 3.  FastAPI JSON endpoint
# ──────────────────────────────────────────────────────────────────────────
api = FastAPI(title="Cheque-OCR")

@api.post("/predict")
async def predict(file: UploadFile = File(...)):
    buf = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    preds, crops_b64, _, _ = cheque_ocr(img)
    return {"file": file.filename, "predictions": preds, "crops": crops_b64}

# ──────────────────────────────────────────────────────────────────────────
# 4.  Gradio UI  (mounted at “/”)
# ──────────────────────────────────────────────────────────────────────────
def gradio_fn(img_rgb):
    if img_rgb is None:
        return None, None, None
    preds, _, crops_rgb, vis_rgb = cheque_ocr(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    gallery = crops_rgb          # list[ndarray]
    return vis_rgb, gallery, preds

demo = gr.Interface(
    fn=gradio_fn,
    inputs=gr.Image(type="numpy", label="Upload cheque"),
    outputs=[
        gr.Image(label="Detected fields"),
        gr.Gallery(label="Crops").style(grid=4),
        gr.JSON(label="Text")
    ],
    title="Cheque-OCR",
    description="YOLOv11n crops + OCR text extraction")

api = gr.mount_gradio_app(app=api, blocks=demo, path="/")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8080)
