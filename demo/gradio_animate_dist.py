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
import argparse
import imageio
import os, datetime
import numpy as np
import gradio as gr
from PIL import Image
from subprocess import PIPE, run

os.makedirs("./demo/tmp", exist_ok=True)
savedir = f"demo/outputs"
os.makedirs(savedir, exist_ok=True)

def animate(reference_image, motion_sequence, seed, steps, guidance_scale):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    animation_path = f"{savedir}/{time_str}.mp4"
    save_path = "./demo/tmp/input_reference_image.png"
    Image.fromarray(reference_image).save(save_path)
    command = "python -m demo.animate_dist --reference_image {} --motion_sequence {} --random_seed {} --step {} --guidance_scale {} --save_path {}".format(
        save_path,
        motion_sequence,
        seed,
        steps,
        guidance_scale,
        animation_path
    )
    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return animation_path

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
        reference_image  = gr.Image(label="Reference Image")
        motion_sequence  = gr.Video(format="mp4", label="Motion Sequence")
        
        with gr.Column():
            random_seed         = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps      = gr.Textbox(label="Sampling steps", value=25, info="default: 25")
            guidance_scale      = gr.Textbox(label="Guidance scale", value=7.5, info="default: 7.5")
            submit              = gr.Button("Animate")

    def read_video(video, size=512):
        size = int(size)
        reader = imageio.get_reader(video)
        # fps = reader.get_meta_data()['fps']
        frames = []
        for img in reader:
            frames.append(np.array(Image.fromarray(img).resize((size, size))))
        save_path = "./demo/tmp/input_motion_sequence.mp4"
        imageio.mimwrite(save_path, frames, fps=25)
        return save_path
    
    def read_image(image, size=512):
        img = np.array(Image.fromarray(image).resize((size, size)))
        return img
        
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
    submit.click(
        animate,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale], 
        animation
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            ["inputs/applications/source_image/monalisa.png", "inputs/applications/driving/densepose/running.mp4"], 
            ["inputs/applications/source_image/demo4.png", "inputs/applications/driving/densepose/demo4.mp4"],
            ["inputs/applications/source_image/dalle2.jpeg", "inputs/applications/driving/densepose/running2.mp4"],
            ["inputs/applications/source_image/dalle8.jpeg", "inputs/applications/driving/densepose/dancing2.mp4"],
            ["inputs/applications/source_image/multi1_source.png", "inputs/applications/driving/densepose/multi_dancing.mp4"],
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )

# demo.queue(max_size=10)
demo.launch(share=True)