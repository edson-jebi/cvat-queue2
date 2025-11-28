#!/bin/bash

# CVAT Queue Manager - Docker Build and Push Script
# Usage: ./docker-build.sh [dockerhub_username] [tag]

# Configuration
DOCKERHUB_USER="${1:-YOUR_DOCKERHUB_USERNAME}"
TAG="${2:-latest}"
IMAGE_NAME="cvat-queue"

echo "========================================"
echo "CVAT Queue Manager - Docker Build"
echo "========================================"
echo "Docker Hub User: $DOCKERHUB_USER"
echo "Image: $IMAGE_NAME:$TAG"
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build the image
echo ""
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo ""
echo "Build successful!"

# Tag for Docker Hub
FULL_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"
echo ""
echo "Tagging as $FULL_IMAGE..."
docker tag $IMAGE_NAME:$TAG $FULL_IMAGE

# Push to Docker Hub
echo ""
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to Docker Hub..."
    docker login

    echo ""
    echo "Pushing $FULL_IMAGE..."
    docker push $FULL_IMAGE

    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Successfully pushed to Docker Hub!"
        echo "Image: $FULL_IMAGE"
        echo ""
        echo "To pull and run:"
        echo "  docker pull $FULL_IMAGE"
        echo "  docker run -d -p 8000:8000 -v ./data:/app/data -v ./backups:/app/backups $FULL_IMAGE"
        echo "========================================"
    else
        echo "Error: Failed to push to Docker Hub"
        exit 1
    fi
else
    echo ""
    echo "Skipped Docker Hub push."
    echo ""
    echo "To run locally:"
    echo "  docker run -d -p 8000:8000 -v ./data:/app/data -v ./backups:/app/backups $IMAGE_NAME:$TAG"
fi
