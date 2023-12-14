import os

script_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
models_path = os.path.join(script_path, "pretrained_models")
magic_models_path = os.path.join(models_path, "MagicAnimate")

pretrained_model_path = os.path.join(models_path, "stable-diffusion-v1-5")
pretrained_vae_path = os.path.join(models_path, "pretrained_vae")

pretrained_controlnet_path = os.path.join(magic_models_path, "densepose_controlnet")
pretrained_encoder_path = os.path.join(magic_models_path, "appearance_encoder")
pretrained_motion_module_path = os.path.join(magic_models_path, "temporal_attention")
motion_module = os.path.join(pretrained_motion_module_path, "temporal_attention.ckpt")

pretrained_unet_path = ""

config_path = os.path.join(script_path, "configs", "prompts", "animation.yaml")
inference_config_path = os.path.join(script_path, "configs", "inference", "inference.yaml")

