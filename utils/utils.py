import json
import math
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import wandb
from monai.bundle import ConfigParser
from torch.nn.parallel import DistributedDataParallel
from typing import Optional

def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a model from DistributedDataParallel if wrapped."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def define_instance(args: Namespace, instance_def_key: str) -> Any:
    """
    Define and instantiate an object based on the provided arguments and instance definition key.

    Args:
        args: An object containing the arguments to be parsed.
        instance_def_key: The key used to retrieve the instance definition from the parsed content.

    Returns:
        The instantiated object as defined by the instance_def_key in the parsed configuration.
    """
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def dynamic_infer(inferer, model, images):
    """
    Perform dynamic inference using a model and an inferer, typically a MONAI SlidingWindowInferer.

    Args:
        inferer: An inference object, typically a MONAI SlidingWindowInferer.
        model: The model used for inference.
        images: The input images, shape [N,C,H,W,D] or [N,C,H,W].

    Returns:
        The output from the model or the inferer, depending on the input size.
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    
    spatial_dims = images.shape[2:]
    orig_roi = inferer.roi_size

    if len(orig_roi) != len(spatial_dims):
        raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")

    adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
    inferer.roi_size = adjusted_roi
    output = inferer(network=model, inputs=images)
    inferer.roi_size = orig_roi
    return output


def add_timestamp(run_name: Optional[str] = None) -> str:
    """Add a timestamp prefix to a run name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{run_name}" if run_name else timestamp


def setup_wandb_run(project_name: str, run_name: str, config: Namespace) -> wandb.Run:
    """Initialize a wandb run with the given name."""
    run = wandb.init(project=project_name, name=run_name, config=config)
    run.config.update(config)
    return run


def save_checkpoint(model: torch.nn.Module, output_path: Path) -> None:
    """Save a model checkpoint (handles DDP automatically)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.module.state_dict() if dist.is_initialized() else model.state_dict()
    torch.save(state_dict, output_path)


def save_config(args: Namespace, run_dir: Path) -> None:
    """Save configuration to run directory as config.json."""
    config_path = run_dir / "config.json"
    config_dict = {}
    for key, value in vars(args).items():
        try:
            json.dumps(value)
            config_dict[key] = value
        except (TypeError, ValueError):
            config_dict[key] = str(value)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)


def run_dist(f, args: Namespace):
    """Run a function with distributed training support."""
    if dist.is_torchelastic_launched():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        local_rank = 0

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    f(args, rank, device)

    if dist.is_initialized():
        dist.destroy_process_group()


def save_image(data: np.ndarray, output_path: str, affine: np.ndarray = np.eye(4)) -> None:
    """Save numpy array as NIfTI image."""
    new_image = nib.Nifti1Image(data.astype(np.float32), affine=affine)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_image, output_path)


class ReconModel(torch.nn.Module):
    """A PyTorch module for reconstructing images from latent representations."""

    def __init__(self, autoencoder: torch.nn.Module, scale_factor: float = 1.0):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the input latent representation to an image."""
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def get_pred_path(sample: dict, output_dir: Path, mod_name: str) -> Path:
    """Get the prediction output path for a sample."""
    first_key = next(k for k in sample.keys() if k not in ["dose", "roi_mask", "fu_time"])
    img_path = Path(sample[first_key].meta["filename_or_obj"][0])
    
    anat_path = img_path.parent
   
    ses_path = anat_path.parent
    
    sub_path = ses_path.parent
   
    return output_dir / sub_path.name / ses_path.name / anat_path.name / f"{mod_name}.nii.gz"