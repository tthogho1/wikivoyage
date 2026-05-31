#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   AWS_ACCOUNT_ID=123... AWS_REGION=us-east-1 REPO_NAME=wikivoyage ./scripts/publish_ecr.sh
# or rely on AWS CLI config for account/region and pass REPO_NAME

AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-}
AWS_REGION=${AWS_REGION:-$(aws configure get region || echo us-east-1)}
REPO_NAME=${REPO_NAME:-wikivoyage}
IMAGE_TAG=${IMAGE_TAG:-latest}

if [ -z "$AWS_ACCOUNT_ID" ]; then
  echo "AWS_ACCOUNT_ID environment variable must be set (or modify script to infer)."
  exit 1
fi

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME"

echo "Building Docker image..."
docker build -t $REPO_NAME:$IMAGE_TAG .

echo "Ensuring repository exists in ECR..."
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION" >/dev/null

echo "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "Tagging and pushing image to $ECR_URI:$IMAGE_TAG"
docker tag $REPO_NAME:$IMAGE_TAG $ECR_URI:$IMAGE_TAG
docker push $ECR_URI:$IMAGE_TAG

echo "Done. Image available at: $ECR_URI:$IMAGE_TAG"
