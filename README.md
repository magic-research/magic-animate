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
Please download the pretrained base models for [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [MSE-finetuned VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse).

Download our MagicAnimate [checkpoints](https://huggingface.co/zcxu-eric/MagicAnimate).

**Place them as follows:**
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
    |----...
  |----stable-diffusion-v1-5
    |----...
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

