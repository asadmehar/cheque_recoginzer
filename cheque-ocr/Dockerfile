# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app
# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY app.py ./app.py
# # copy YOLO weights separately so you can swap without rebuild
# COPY weights /weights

# COPY templates /app/templates
# COPY static     /app/static


# EXPOSE 8080
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]



###########################
# ── 1) base layer (deps) ─
###########################
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- add missing shared libs for OpenCV ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1        \
        libglib2.0-0  \
    && rm -rf /var/lib/apt/lists/*

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
RUN mkdir -p $YOLO_CONFIG_DIR


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

##############################
# ── 2) prod runtime image  ──
##############################
FROM base AS prod
COPY app.py templates static /app/
# copy weights only if you prefer them *inside* the image
# COPY weights /weights
ENV YOLO_WEIGHTS=/weights/best_yolo11n.pt
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

##############################
# ── 3) dev runtime image  ──
##############################
FROM base AS dev
# nothing copied here – source will be bind-mounted
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
