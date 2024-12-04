#!/bin/bash

# Directories to clean
echo "Cleaning up training results directories..."
rm -rf sb3_results/*
rm -rf ray_results/*
rm -rf recordings/*

# Optional: Clean up Python cache files
echo "Cleaning up Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "Cleanup complete!"