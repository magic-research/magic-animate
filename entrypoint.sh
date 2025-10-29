#!/bin/bash

ENV_NAME="manimate"

echo "Activating Conda environment $ENV_NAME..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Running manimate..."
python3 -m demo.gradio_animate

