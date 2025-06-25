######################################################################
# Cheque-OCR  |  CUDA 12.1 + PyTorch 2.2  |  Works on Lambda URL
######################################################################

######## 1 – build layer #############################################
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS build

# System libs + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + YOLO weights
COPY app.py .
COPY weights/best_yolo11n.pt /weights/best_yolo11n.pt

######## 2 – final runtime layer #####################################
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Copy everything built above
COPY --from=build /app /app
COPY --from=build /weights /weights

# AWS Lambda Web Adapter v0.9.1 (just drop into /opt/extensions)
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 \
     /lambda-adapter /opt/extensions/lambda-adapter

# Runtime env
ENV YOLO_WEIGHTS=/weights/best_yolo11n.pt \
    PORT=8080

WORKDIR /app
EXPOSE 8080

# DO NOT change ENTRYPOINT; keep the base image’s default runtime
CMD ["python", "app.py"]
