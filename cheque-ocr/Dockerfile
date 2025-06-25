FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (optional) pre-download TrOCR weights to avoid first-run delay
# RUN python - <<'PY'
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
# PY

COPY app.py /app/

ENV YOLO_WEIGHTS=/weights/best_yolo11n.pt
EXPOSE 8080
CMD ["python", "app.py"]
