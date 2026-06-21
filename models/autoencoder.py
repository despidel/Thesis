from argparse import Namespace
from pathlib import Path

import torch
from monai.apps import download_url
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from utils.utils import define_instance

def load_autoencoder(trained_autoencoder_path: str, model_def_args: Namespace, device: torch.device) -> AutoencoderKlMaisi:
    autoencoder = define_instance(model_def_args, "autoencoder_def").to(device)
    
    if not Path(trained_autoencoder_path).exists():
        Path(trained_autoencoder_path).parent.mkdir(parents=True, exist_ok=True)

        download_url(
            url="https://developer.download.nvidia.com/assets/Clara/monai/tutorials/model_zoo/model_maisi_autoencoder_epoch273_alternative.pt",
            filepath=trained_autoencoder_path
        )
    
    checkpoint_autoencoder = torch.load(trained_autoencoder_path, weights_only=True) 
    autoencoder.load_state_dict(checkpoint_autoencoder)

    return autoencoder