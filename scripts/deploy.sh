#!/bin/bash
# Deployment script
echo "Deploying application..."
docker-compose up -d --build
