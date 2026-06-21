"""
Diffusion Model Inference

    python -m train.diff_model_train --run_name <name>
"""
from __future__ import annotations


import argparse
from argparse import Namespace
import numpy as np

import torch
from monai.inferers.inferer import SlidingWindowInferer
from monai.utils import set_determinism
from monai.transforms import ResizeWithPadOrCrop

from models.autoencoder import load_autoencoder
from models.diff_model import load_diff_model
from utils.diff_model_setting import load_configs, setup_logging
from utils.utils import (
    define_instance,
    dynamic_infer,
    save_image,
    ReconModel,
)

@torch.inference_mode()
def diff_model_infer(
    unet: torch.nn.Module,
    recon_model: torch.nn.Module,
    device: torch.device,
    args: Namespace,
) -> np.ndarray:
    """
    Run the inference to generate synthetic images.

    Args:
        unet (torch.nn.Module): UNet model.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to run inference on.
        args (Namespace): Configuration arguments.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    unet.eval()
    recon_model.eval()
    
    logger = setup_logging("diff_model_infer")
    logger.info("Running inference")

    if args.trained_diff_model_path is not None:
        logger.info(f"[config] trained_diff_model_path -> {args.trained_diff_model_path}.")

    divisor = 2 ** (len(args.autoencoder_def["num_channels"]) - 1)
    latent_shape = (
        1,
        #args.latent_channels,
        args.latent_channels * 3,
        args.training_img_dim[0] // divisor,
        args.training_img_dim[1] // divisor,
        args.training_img_dim[2] // divisor,
    )
    noise = torch.randn(latent_shape, device=device)
    logger.info(f"noise: {noise.device}, {noise.dtype}, {noise.shape}, {type(noise)}")

    img = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(
        num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
        input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
    )

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in zip(all_timesteps, all_next_timesteps):
            unet_inputs = {
                "x": img,
                "timesteps": torch.Tensor((t,)).to(device),
            }
            model_output = unet(**unet_inputs)

            img, _ = noise_scheduler.step(model_output, t, img, next_t)

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )

        
        logger.info(f"Reconstructing latent")

        # Split 12 channels back into 3 modalities (4 channels each)
        imgs = torch.chunk(img, 3, dim=1)  # list of 3 tensors [1, 4, ...]
        outputs = []
        for img_mod in imgs:
            out = dynamic_infer(inferer, recon_model, img_mod)
            out = ResizeWithPadOrCrop(spatial_size=args.raw_img_dim)(out.squeeze(0))
            out = out.squeeze().cpu().detach().numpy()
            out = np.clip(out, 0, 1)
            outputs.append(out)

        return outputs  # list of 3 reconstructed modalities [T1, T1c, FLAIR]


def main(args: Namespace) -> torch.Tensor:
    """
    Main function to run the diffusion model inference.

    Args:
        args (Namespace): Configuration arguments.
    """
    random_seed = args.diffusion_unet_inference["random_seed"]
    set_determinism(random_seed)

    device = torch.device("cuda")

    unet = load_diff_model(
        trained_diff_model_path=args.trained_diff_model_path,
        model_def_args=args,
        device=device,
    )
    autoencoder = load_autoencoder(
        trained_autoencoder_path=args.trained_autoencoder_path,
        model_def_args=args,
        device=device,
    )
    recon_model = ReconModel(autoencoder)

    img = diff_model_infer(
        unet=unet,
        recon_model=recon_model,
        device=device,
        args=args,
    )

    save_image(img, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument("--env_config", type=str, default="./configs/env_config_diff_model_infer.json")
    parser.add_argument("--model_config", type=str, default="./configs/model_config_diff_model_train.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_maisi3d-rflow.json")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    load_configs(args, [args.env_config, args.model_config, args.model_def])
    
    main(args)
