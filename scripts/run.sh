!/usr/bin/env bash

#################################################
# Script for setting up MagicAnimate environment with model downloads
#################################################

use_venv=1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
install_dir="$SCRIPT_DIR"
clone_dir="magic-animate"
venv_dir="${install_dir}/${clone_dir}/venv"
python_cmd="python3"
GIT="git"
LAUNCH_SCRIPT="demo.gradio_animate"

# Pretty print
delimiter="################################################################"
printf "
%s
" "${delimiter}"
printf "\e[1m\e[32mSetup Script for MagicAnimate
\e[0m"
printf "
%s
" "${delimiter}"

# Check if install_dir ends with "magic-animate/scripts" and adjust paths accordingly
if [[ "${install_dir}" == *"/magic-animate/scripts" ]]; then
    printf "\e[1m\e[32mAdjusting the work directory...
\e[0m"
    work_dir="${install_dir%"/scripts"}"
    venv_dir="${install_dir%"/scripts"}/venv"
else
    work_dir="${install_dir}/${clone_dir}"
fi

# Clone the MagicAnimate repository if it does not exist
if [[ -d "${work_dir}" ]]; then
    printf "\e[1m\e[32mMagicAnimate directory exists. Using existing directory.
\e[0m"
else
    printf "\e[1m\e[32mCloning MagicAnimate repository...
\e[0m"
    "${GIT}" clone https://github.com/magic-research/magic-animate.git "${work_dir}"
fi

# Navigate to the MagicAnimate directory
cd "${work_dir}" || { printf "\e[1m\e[31mERROR: Can't cd to %s, aborting...\e[0m
" "${work_dir}"; exit 1; }

# Setting up Python virtual environment
if [[ $use_venv -eq 1 ]]; then
    if [[ ! -d "${venv_dir}" ]]; then
        printf "\e[1m\e[32mCreating Python virtual environment...
\e[0m"
        "${python_cmd}" -m venv "${venv_dir}"
    fi
    printf "\e[1m\e[32mActivating Python virtual environment...
\e[0m"
    source "${venv_dir}/bin/activate"
fi

# Install Python dependencies
printf "\e[1m\e[32mInstalling Python dependencies...
\e[0m"
pip3 install -r requirements.txt

# Model download
printf "\e[1m\e[32mDownloading models...
\e[0m"
mkdir -p pretrained_models
cd pretrained_models

# Use Git LFS to clone the repository for MagicAnimate models
if [ ! -d "MagicAnimate" ]; then
    printf "\e[1m\e[32mCloning MagicAnimate models repository...
\e[0m"
    git lfs clone https://huggingface.co/zcxu-eric/MagicAnimate
fi

# sd-vae-ft-mse model
mkdir -p sd-vae-ft-mse
cd sd-vae-ft-mse
wget -nc https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
wget -nc https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
wget -nc https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
cd ..

# stable-diffusion-v1-5 model
mkdir -p stable-diffusion-v1-5
cd stable-diffusion-v1-5
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/model_index.json

# Create necessary directories under stable-diffusion-v1-5
mkdir -p tokenizer
cd tokenizer
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_config.json
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/special_tokens_map.json
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt
cd ..

# text_encoder directory
mkdir -p text_encoder
cd text_encoder
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/pytorch_model.bin
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/text_encoder/config.json
cd ..

# unet directory
mkdir -p unet
cd unet
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/config.json
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/unet/diffusion_pytorch_model.bin
cd ..

# scheduler directory
mkdir -p scheduler
cd scheduler
wget -nc https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/scheduler/scheduler_config.json
cd ../../..
    
# Run the launch script
printf "\e[1m\e[32mRunning MagicAnimate...
\e[0m"
"${python_cmd}" -m "${LAUNCH_SCRIPT}"