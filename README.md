<!-- # magic-edit.github.io -->

<p align="center">

  <h2 align="center">MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=-4iADzMAAAAJ&hl=en"><strong>Zhongcong Xu</strong></a>
    ·
    <a href="http://jeff95.me/"><strong>Jianfeng Zhang</strong></a>
    ·
    <a href="https://scholar.google.com.sg/citations?user=8gm-CYYAAAAJ&hl=en"><strong>Jun Hao Liew</strong></a>
    ·
    <a href="https://hanshuyan.github.io/"><strong>Hanshu Yan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=stQQf7wAAAAJ&hl=en"><strong>Jia-Wei Liu</strong></a>
    ·
    <a href="https://zhangchenxu528.github.io/"><strong>Chenxu Zhang</strong></a>
    ·
    <a href="https://sites.google.com/site/jshfeng/home"><strong>Jiashi Feng</strong></a>
    ·
    <a href="https://sites.google.com/view/showlab"><strong>Mike Zheng Shou</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2311.16498"><img src='https://img.shields.io/badge/arXiv-MagicAnimate-red' alt='Paper PDF'></a>
        <a href='https://showlab.github.io/magicanimate'><img src='https://img.shields.io/badge/Project_Page-MagicAnimate-green' alt='Project Page'></a>
        <a href='https://huggingface.co/spaces/zcxu-eric/magicanimate'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
    <br>
    <b>National University of Singapore &nbsp; | &nbsp;  ByteDance</b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/teaser/t4.gif">
    </td>
    <td>
      <img src="assets/teaser/t2.gif">
    </td>
    </tr>
  </table>

## 📢 News
* **[2023.12.4]** Release inference code and gradio demo. We are working to improve MagicAnimate, stay tuned!
* **[2023.11.23]** Release MagicAnimate paper and project page.

## 🏃‍♂️ Getting Started
Download the pretrained base models for [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [MSE-finetuned VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse).

Download our MagicAnimate [checkpoints](https://huggingface.co/zcxu-eric/MagicAnimate).

Please follow the huggingface download instructions to download the above models and checkpoints, `git lfs` is recommended.

Place the based models and checkpoints as follows:
```bash
magic-animate
|----pretrained_models
  |----MagicAnimate
    |----appearance_encoder
      |----diffusion_pytorch_model.safetensors
      |----config.json
    |----densepose_controlnet
      |----diffusion_pytorch_model.safetensors
      |----config.json
    |----temporal_attention
      |----temporal_attention.ckpt
  |----sd-vae-ft-mse
    |----config.json
    |----diffusion_pytorch_model.safetensors
  |----stable-diffusion-v1-5
    |----scheduler
       |----scheduler_config.json
    |----text_encoder
       |----config.json
       |----pytorch_model.bin
    |----tokenizer (all)
    |----unet
       |----diffusion_pytorch_model.bin
       |----config.json
    |----v1-5-pruned-emaonly.safetensors
|----...
```

## ⚒️ Installation
prerequisites: `python>=3.8`, `CUDA>=11.3`, and `ffmpeg`.

Install with `conda`: 
```bash
conda env create -f environment.yaml
conda activate manimate
```
or `pip`:
```bash
pip3 install -r requirements.txt
```

## 💃 Inference
Run inference on single GPU:
```bash
bash scripts/animate.sh
```
Run inference with multiple GPUs:
```bash
bash scripts/animate_dist.sh
```

## 🎨 Gradio Demo 

#### Online Gradio Demo:
Try our [online gradio demo](https://huggingface.co/spaces/zcxu-eric/magicanimate) quickly.

#### Local Gradio Demo:
Launch local gradio demo on single GPU:
```bash
python3 -m demo.gradio_animate
```
Launch local gradio demo if you have multiple GPUs:
```bash
python3 -m demo.gradio_animate_dist
```
Then open gradio demo in local browser.

## 🐳 Docker

Running code in environments other than Linux can be challenging. Docker is a great solution for this. Install [Docker](https://docs.docker.com/engine/install/) and [Nvidia-container-runtime](https://nvidia.github.io/nvidia-container-runtime/). Place pretrained_models as expected. Docker will ignore models in the building stage.

### Building and Running the Docker Image

Run the following command to build and start the Docker image/container::

```bash
docker-compose up -d
```

Open a terminal and create an SSH tunnel to the Docker container:

```bash
ssh -L 7860:127.0.0.1:7860 -p 2222 models@localhost
```

Password: `root`

Execute `./entrypoint.sh` to run the Gradio demo. The Gradio demo will be available at [http://localhost:7860](http://localhost:7860).

The SSH tunnel needs to remain active at all times for the Gradio demo to work. Additionally, you need to be inside the Docker container to execute commands.

Before running commands like:

```bash
python3 -m magicanimate.pipelines.animation --config configs/prompts/animation.yaml
```

or any other command that is not a script that automatically activates the environment, make sure to have the conda environment active:

```bash
conda activate manimate
```

A volume is mounted at the root directory and all changes, files, etc. will be saved in the root directory and accessible from the Docker container.

## 🙏 Acknowledgements
We would like to thank [AK(@_akhaliq)](https://twitter.com/_akhaliq?lang=en) and huggingface team for the help of setting up oneline gradio demo.

## 🎓 Citation
If you find this codebase useful for your research, please use the following entry.
```BibTeX
@inproceedings{xu2023magicanimate,
    author    = {Xu, Zhongcong and Zhang, Jianfeng and Liew, Jun Hao and Yan, Hanshu and Liu, Jia-Wei and Zhang, Chenxu and Feng, Jiashi and Shou, Mike Zheng},
    title     = {MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model},
    booktitle = {arXiv},
    year      = {2023}
}
```

