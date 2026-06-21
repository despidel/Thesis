from argparse import Namespace

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi

from utils.utils import define_instance

def load_diff_model(trained_diff_model_path: str, model_def_args: Namespace, device: torch.device) -> DiffusionModelUNetMaisi:
    """
    Load the diffusion UNet model.
    
    Note: This function only loads the model. DDP/SyncBatchNorm wrapping should be
    applied in training scripts for models that require gradient synchronization.
    """
    unet = define_instance(model_def_args, "diffusion_unet_def").to(device)

    if trained_diff_model_path is not None:
        checkpoint_unet = torch.load(f"{trained_diff_model_path}", map_location=device, weights_only=False)
        unet.load_state_dict(checkpoint_unet, strict=True)

    return unet