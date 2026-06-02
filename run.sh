#!/bin/bash
set -e

echo "Starting ML Pipeline..."
python src/cleaning.py
python src/features.py
python src/train.py
python src/evaluate.py
echo "Pipeline complete."
