


==========================================================
Try it locally

docker build -t cheque-ocr:gradio .
docker run -p 8080:8080 \
           -v "$(pwd)/weights:/weights:ro" \
           -e YOLO_WEIGHTS=/weights/best_yolo11n.pt \
           cheque-ocr:gradio

=========================================================

Build & run

docker build -t cheque-ocr:gradio .
docker run --rm -p 8080:8080 \
           -v "$(pwd)/weights:/weights:ro" \
           -e YOLO_WEIGHTS=/weights/best_yolo11n.pt \
           cheque-ocr:gradio
=========================================================

Use it

curl -X POST -F file=@sample.jpg http://localhost:8080/predict
==========================================================
Uvicorn running on http://0.0.0.0:8080, you’re live.

Browser → http://localhost:8080

====================================================