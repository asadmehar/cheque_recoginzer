#!/usr/bin/env bash
set -euo pipefail

REGION=us-east-1
REPO=cheque-ocr
AWS_ACC=$(aws sts get-caller-identity --query Account --output text)
URI=${AWS_ACC}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest
ROLE_NAME=AppRunnerECRAccessRole
ROLE_ARN=arn:aws:iam::${AWS_ACC}:role/${ROLE_NAME}

# ── 0. create IAM role once ─────────────────────────────────────────
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "Creating IAM role $ROLE_NAME…"
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
        "Version":"2012-10-17",
        "Statement":[{
          "Effect":"Allow",
          "Principal":{"Service":"build.apprunner.amazonaws.com"},
          "Action":"sts:AssumeRole"
        }]
    }'
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
fi

# ── 1. build & push image ───────────────────────────────────────────
docker build -t ${REPO}:latest .
aws ecr describe-repositories --repository-names ${REPO} --region ${REGION} \
  >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name ${REPO} --region ${REGION}

aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${AWS_ACC}.dkr.ecr.${REGION}.amazonaws.com
docker tag ${REPO}:latest ${URI}
docker push ${URI}

# ── 2. create or update App Runner service ─────────────────────────-
SERVICE_ARN=$(aws apprunner list-services \
  --query "ServiceSummaryList[?ServiceName=='cheque-ocr'].ServiceArn" \
  --output text || true)

SRC_CFG="ImageRepository={ImageIdentifier=${URI},ImageRepositoryType=ECR,\
ImageConfiguration={Port=8080}},\
AuthenticationConfiguration={AccessRoleArn=${ROLE_ARN}},\
AutoDeploymentsEnabled=false"

if [[ -z "$SERVICE_ARN" || "$SERVICE_ARN" == "None" ]]; then
  echo "Creating App Runner service…"
  aws apprunner create-service \
    --service-name cheque-ocr \
    --source-configuration "$SRC_CFG" \
    --instance-configuration "Cpu=1 vCPU,Memory=2 GB"
else
  echo "Updating existing App Runner service…"
  aws apprunner update-service \
    --service-arn "$SERVICE_ARN" \
    --source-configuration "$SRC_CFG"
fi
