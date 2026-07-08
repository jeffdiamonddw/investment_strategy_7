#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- CONFIGURATION ---
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID="129861351772"
REPO_NAME="optimization"
IMAGE_TAG="latest"
DOCKERFILE_PATH="docker/optimization.dockerfile"
FULL_REPO_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "🚀 Starting daemonless deployment for ${REPO_NAME}..."

# 1. Ensure the ECR repository exists
echo "🔍 Checking ECR repository..."
aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION} > /dev/null 2>&1 || echo "✅ Repository already exists."

# 2. Build the image using buildah
# Note: --arch arm64 handles the platform requirement
# Replace the build step in your script with this:
echo "🛠️ Building ARM64 image using podman..."
podman build --storage-driver vfs --arch arm64 -f ${DOCKERFILE_PATH} -t ${FULL_REPO_URL}:${IMAGE_TAG} .

# 3. Authenticate to ECR
echo "🔑 Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | buildah login --username AWS --password-stdin ${FULL_REPO_URL}

# 4. Push directly to ECR
echo "📤 Pushing to ECR..."
buildah push ${FULL_REPO_URL}:${IMAGE_TAG}

echo "---"
echo "✅ Success! Simulation image is now at:"
echo "${FULL_REPO_URL}:${IMAGE_TAG}"