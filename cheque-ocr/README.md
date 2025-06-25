


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
(base) asad@asad:~/ITU/Projects/cheque_recoginzer/cheque-ocr$ docker build -t cheque-ocr-final:latest .

AWS:
#Bake the weights into the image (one-time)
COPY weights/best_yolo11n.pt /weights/best_yolo11n.pt
ENV YOLO_WEIGHTS=/weights/best_yolo11n.pt

Re-build:
docker build -t cheque-ocr:latest .


export REGION=us-east-1
export AWS_ACC=$(aws sts get-caller-identity --query Account --output text)
export REPO=cheque-ocr
export URI=$AWS_ACC.dkr.ecr.$REGION.amazonaws.com/$REPO:latest

aws ecr create-repository --repository-name $REPO --region $REGION || true

aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $AWS_ACC.dkr.ecr.$REGION.amazonaws.com

docker tag cheque-ocr:latest $URI
docker push $URI 


====================

Step-2
#2 Create a minimal IAM role for Lambda (once)

aws iam create-role \
  --role-name LambdaChequeOCR \
  --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"lambda.amazonaws.com"},
        "Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
  --role-name LambdaChequeOCR \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

=====================================

step-3
#3 Create the Lambda function from your container (1 command)
aws lambda create-function \
  --function-name cheque-ocr \
  --package-type Image \
  --code ImageUri=$URI \
  --role arn:aws:iam::$AWS_ACC:role/LambdaChequeOCR \
  --memory-size 2048 \
  --timeout 900


======================================

4 Attach a Function URL (public HTTPS)
(base) asad@asad:~/ITU/Projects/cheque_recoginzer/cheque-ocr$ aws lambda create-function-url-config \
  --function-name cheque-ocr \
  --auth-type AWS_IAM \
  --cors '{
      "AllowOrigins":["*"],
      "AllowMethods":["*"],
      "AllowHeaders":["*"]
  }' \
  --region $REGION

Output:
NONE    2025-06-25T08:47:26.684359286Z  arn:aws:lambda:us-east-1:582939842902:function:cheque-ocr       https://dc2rzzdctdvwxttr76kcqhtve40fbsjf.lambda-url.us-east-1.on.aws/
ALLOWHEADERS    *
ALLOWMETHODS    *
ALLOWORIGINS    *
(base) asad@asad:~/ITU/Projects/cheque_recoginzer/cheque-ocr$ 

=================================================



curl -X POST -F file=@sample.jpg \
     https://dc2rzzdctdvwxttr76kcqhtve40fbsjf.lambda-url.us-east-1.on.aws/predict


aws lambda get-function-url-config \
  --function-name cheque-ocr \
  --query https://dc2rzzdctdvwxttr76kcqhtve40fbsjf.lambda-url.us-east-1.on.aws/predict \
  --output text

