# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import os

import gradio as gr
import imageio
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from demo.animate import MagicAnimate
from demo.paths import config_path, script_path, models_path, magic_models_path, source_images_path, \
    motion_sequences_path

animator = None


def list_checkpoints():
    checkpoint_dir = os.path.join(models_path, "checkpoints")
    # Recursively find all .ckpt and .safetensors files
    checkpoints = [""]
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".ckpt") or file.endswith(".safetensors"):
                checkpoints.append(os.path.join(root, file))
    return checkpoints


# source_image, motion_sequence, random_seed, step, guidance_scale, size=512, checkpoint=None
def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale, size, checkpoint=None):
    return animator(reference_image, motion_sequence_state, seed, steps, guidance_scale, size, checkpoint)


with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/magic-research/magic-animate" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model</h1>
            <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;>
                <a href="https://arxiv.org/abs/2311.16498"><img src="https://img.shields.io/badge/Arxiv-2311.16498-red"></a>
                <a href='https://showlab.github.io/magicanimate'><img src='https://img.shields.io/badge/Project_Page-MagicAnimate-green' alt='Project Page'></a>
                <a href='https://github.com/magic-research/magic-animate'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
            </div>
        </div>
        </div>
        """)
    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)
    with gr.Row():
        checkpoint = gr.Dropdown(label="Checkpoint", choices=list_checkpoints())
    with gr.Row():
        reference_image = gr.Image(label="Reference Image")
        motion_sequence = gr.Video(format="mp4", label="Motion Sequence")

        with gr.Column():
            size = gr.Slider(label="Size", value=512, min=256, max=1024, step=256, info="default: 512", visible=False)
            random_seed = gr.Slider(label="Random seed", value=1, info="default: -1")
            sampling_steps = gr.Slider(label="Sampling steps", value=25, info="default: 25")
            guidance_scale = gr.Slider(label="Guidance scale", value=7.5, info="default: 7.5", step=0.1)
            submit = gr.Button("Animate")


    def read_video(video):
        reader = imageio.get_reader(video)
        fps = reader.get_meta_data()['fps']
        return video


    def read_image(image, size=512):
        return np.array(Image.fromarray(image).resize((size, size)))


    # when user uploads a new video
    motion_sequence.upload(
        read_video,
        motion_sequence,
        motion_sequence
    )
    # when `first_frame` is updated
    reference_image.upload(
        read_image,
        reference_image,
        reference_image
    )
    # when the `submit` button is clicked
    #source_image, motion_sequence, random_seed, step, guidance_scale, size = 512, checkpoint = None
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale, size, checkpoint],
        animation
    )

    # source_images_path = os.path.join(script_path, "inputs", "applications", "source_image")
    # motion_sequences_path = os.path.join(script_path, "inputs", "applications", "driving", "densepose")
    #Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            [f"{source_images_path}/monalisa.png", f"{motion_sequences_path}/running.mp4"],
            [f"{source_images_path}/demo4.png", f"{motion_sequences_path}/demo4.mp4"],
            [f"{source_images_path}/dalle2.jpeg", f"{motion_sequences_path}/running2.mp4"],
            [f"{source_images_path}/dalle8.jpeg", f"{motion_sequences_path}/dancing2.mp4"],
            [f"{source_images_path}/multi1_source.png",
             f"{motion_sequences_path}/multi_dancing.mp4"],
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )

if __name__ == '__main__':
    if not os.path.exists(models_path):
        os.mkdir(models_path)

    if not os.path.exists(os.path.join(models_path, "checkpoints")):
        os.mkdir(os.path.join(models_path, "checkpoints"))

    if not os.path.exists(magic_models_path):
        # git lfs clone https://huggingface.co/zcxu-eric/MagicAnimate, not hf_hub_download
        git_lfs_path = os.path.join(models_path, "MagicAnimate")
        if not os.path.exists(git_lfs_path):
            os.system(f"git clone https://huggingface.co/zcxu-eric/MagicAnimate {git_lfs_path}")
        else:
            print(f"MagicAnimate already exists at {git_lfs_path}")

    animator = MagicAnimate()

    demo.launch(share=True)
