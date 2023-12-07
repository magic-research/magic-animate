import argparse
import numpy as np
from PIL import Image
import imageio

from demo.animate import MagicAnimate

def read_video(video_path):
    reader = imageio.get_reader(video_path)
    return video_path

def read_image(image_path, size=512):
    image = Image.open(image_path)
    return np.array(image.resize((size, size)))

def main(reference_image_path, motion_sequence_path, seed, steps, guidance_scale):
    animator = MagicAnimate()

    reference_image = read_image(reference_image_path)
    motion_sequence = read_video(motion_sequence_path)

    animation = animator(reference_image, motion_sequence, seed, steps, guidance_scale)
    return animation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate images using MagicAnimate.")
    parser.add_argument("reference_image", help="Path to the reference image")
    parser.add_argument("motion_sequence", help="Path to the motion sequence video")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--steps", type=int, default=25, help="Sampling steps (default: 25)")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (default: 7.5)")

    args = parser.parse_args()

    main(args.reference_image, args.motion_sequence, args.seed, args.steps, args.guidance_scale)
