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
import datetime
import inspect
import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from einops import rearrange
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer

from demo.paths import config_path, inference_config_path, pretrained_encoder_path, pretrained_controlnet_path, \
    motion_module, pretrained_model_path, pretrained_vae_path, output_path
from magicanimate.models.appearance_encoder import AppearanceEncoderModel
from magicanimate.models.controlnet import ControlNetModel
from magicanimate.models.mutual_self_attention import ReferenceAttentionControl
from magicanimate.models.unet_controlnet import UNet3DConditionModel
from magicanimate.pipelines.pipeline_animation import AnimationPipeline
from magicanimate.utils.util import save_videos_grid
from magicanimate.utils.videoreader import VideoReader


def xformerify(obj):
    try:
        import xformers
        obj.enable_xformers_memory_efficient_attention

    except ImportError:
        obj.set_attn_processor(AttnProcessor2_0())


class MagicAnimate:
    def __init__(self) -> None:
        config = OmegaConf.load(config_path)

        print("Initializing MagicAnimate Pipeline...")
        *_, func_args = inspect.getargvalues(inspect.currentframe())
        self.func_args = dict(func_args)

        self.config = config

        inference_config = OmegaConf.load(inference_config_path)
        self.inference_config = inference_config

        ### Load controlnet and appearance encoder
        self.controlnet = None
        self.appearance_encoder = None
        self.pipeline = None
        self.reference_control_writer = None
        self.reference_control_reader = None
        self.L = config.L

        print("Initialization Done!")

    def __call__(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512, half_precision=False, checkpoint=None):
        self.load_pipeline(half_precision, checkpoint)
        prompt = n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        if motion_sequence.endswith('.mp4'):
            control = VideoReader(motion_sequence).read()
            if control[0].shape[0] != size:
                control = [np.array(Image.fromarray(c).resize((size, size))) for c in control]
            control = np.array(control)
        else:
            control = np.load(motion_sequence)
            if control.shape[1] != size:
                control = np.array([np.array(Image.fromarray(c).resize((size, size))) for c in control])
            control = np.array(control)

        if source_image.shape[0] != size:
            source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        H, W, C = source_image.shape

        init_latents = None
        original_length = control.shape[0]
        if control.shape[0] % self.L > 0:
            control = np.pad(control, ((0, self.L - control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)), mode='edge')
        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=len(control),
            controlnet_condition=control,
            init_latents=init_latents,
            generator=generator,
            appearance_encoder=self.appearance_encoder,
            reference_control_writer=self.reference_control_writer,
            reference_control_reader=self.reference_control_reader,
            source_image=source_image,
        ).videos

        source_images = np.array([source_image] * original_length)
        source_images = rearrange(torch.from_numpy(source_images), "t h w c -> 1 c t h w") / 255.0
        samples_per_video.append(source_images)

        control = control / 255.0
        control = rearrange(control, "t h w c -> 1 c t h w")
        control = torch.from_numpy(control)
        samples_per_video.append(control[:, :, :original_length])

        samples_per_video.append(sample[:, :, :original_length])

        samples_per_video = torch.cat(samples_per_video)

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = output_path
        animation_path = os.path.join(savedir, f"{time_str}.mp4")

        os.makedirs(savedir, exist_ok=True)
        save_videos_grid(samples_per_video, animation_path)

        return animation_path

    def load_pipeline(self, half_precision=False, model_path=None):
        if self.pipeline is not None:
            del self.pipeline

        if self.appearance_encoder is not None:
            del self.appearance_encoder

        if self.controlnet is not None:
            del self.controlnet
        torch.cuda.empty_cache()

        self.appearance_encoder = AppearanceEncoderModel.from_pretrained(pretrained_encoder_path,
                                                                         subfolder="appearance_encoder").cuda()

        self.controlnet = ControlNetModel.from_pretrained(pretrained_controlnet_path)
        if half_precision:
            self.controlnet.to(torch.float16)
            self.appearance_encoder.to(torch.float16)
        xformerify(self.controlnet)
        xformerify(self.appearance_encoder)

        config = self.config
        inference_config = self.inference_config
        vae = None
        print(f"Loading pipeline from {model_path}")
        if not model_path:
            model_path = pretrained_model_path
            unet_path = model_path
        else:
            unet_path = model_path
        if "safetensors" in model_path or "ckpt" in model_path:
            temp_pipeline = StableDiffusionPipeline.from_single_file(model_path)
            tokenizer = temp_pipeline.tokenizer
            text_encoder = temp_pipeline.text_encoder
            unet_2d = temp_pipeline.unet
            unet = UNet3DConditionModel.from_2d_unet(unet_2d, inference_config.unet_additional_kwargs)
            try:
                vae = temp_pipeline.vae
            except:
                print("No VAE found in ckpt, using default VAE")
        else:
            tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
            unet = UNet3DConditionModel.from_pretrained_2d(model_path, subfolder="unet",
                                                           unet_additional_kwargs=OmegaConf.to_container(
                                                               inference_config.unet_additional_kwargs))

        if vae is None:
            if os.path.exists(pretrained_vae_path):
                vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path)
            else:
                vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

        self.reference_control_writer = ReferenceAttentionControl(self.appearance_encoder,
                                                                  do_classifier_free_guidance=True, mode='write',
                                                                  fusion_blocks=config.fusion_blocks)
        self.reference_control_reader = ReferenceAttentionControl(unet, do_classifier_free_guidance=True, mode='read',
                                                                  fusion_blocks=config.fusion_blocks)

        if half_precision:
            vae.to(torch.float16)
            unet.to(torch.float16)
            text_encoder.to(torch.float16)

        xformerify(unet)
        xformerify(vae)

        self.pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=self.controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            # NOTE: UniPCMultistepScheduler
        ).to("cuda")

        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict: self.func_args.update(
            {"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = motion_module_state_dict[
            'state_dict'] if 'state_dict' in motion_module_state_dict else motion_module_state_dict
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = self.pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0
        except:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split('unet.')[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
            assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

        self.pipeline.to("cuda")
        self.L = config.L
