from argparse import Namespace
import subprocess
import os
import torch.distributed as dist

import pandas as pd
from monai.transforms import (
    LoadImaged,
    ClipIntensityPercentilesd,
    ScaleIntensityd,
    Compose,
    ResizeWithPadOrCrop,
)

def load_data(paths: dict, args: Namespace) -> dict:
    """Loads images with MONAI transforms, and CSVs with pandas.
    - Image keys: LoadImaged returns MetaTensor (torch.float32 by default)
    - "gt" key gets preprocessing: ClipIntensityPercentiles + ScaleIntensity
    - CSV keys (paths ending in .csv): loaded as pandas DataFrames
    """
    # Separate CSV paths from image paths
    csv_paths = {k: v for k, v in paths.items() if v.endswith(".csv")}
    image_paths = {k: v for k, v in paths.items() if not v.endswith(".csv")}
    
    # Load images with MONAI
    data = {}
    if image_paths:
        keys = list(image_paths.keys())
        transforms = [LoadImaged(keys=keys, ensure_channel_first=True)]
        if "gt" in keys:
            transforms.extend([
                ClipIntensityPercentilesd(keys=["gt"], lower=0.1, upper=99.9),
                ScaleIntensityd(keys=["gt"], minv=0.0, maxv=1.0),
            ])
        data = Compose(transforms)(image_paths)
    
    # Load CSVs with pandas
    for key, path in csv_paths.items():
        data[key] = pd.read_csv(path)
    
    return data





def flatten_metric_input_paths(sample: dict, metrics: list, mod_name: str) -> dict:
    """Flattens the metric_input_paths of all metrics for a specific modality."""
    paths = {}
    for metric in metrics:
        paths.update(sample[metric.name][mod_name]["metric_input_paths"])
    return paths

def gather_results(results: list[dict]) -> list[dict]:
    results_list = [None] * dist.get_world_size()
    dist.all_gather_object(results_list, results)

    gathered = []
    for r in results_list:
        gathered.extend(r)

    return gathered


def run_apptainer_cmd(cmd: list[str], logger=None):
    """Runs `apptainer exec $NEURO_CONTAINER_PATH <cmd>`.
    
    Environment variables:
        NEURO_CONTAINER_PATH: Path to the apptainer/singularity container (required).
        APPTAINER_BINDPATH: Comma-separated paths to bind into container (optional).
    """
    container_path = os.getenv("NEURO_CONTAINER_PATH")
    if not container_path:
        raise EnvironmentError(
            "NEURO_CONTAINER_PATH environment variable is not set. "
            "See README for container installation."
        )
    
    full_cmd = ["apptainer", "exec", "--env", "CUDA_VISIBLE_DEVICES=0"]
    
    

    bind_path = os.getenv("APPTAINER_BINDPATH", "")
    extra_binds = "/projects"
    combined_bind = f"{bind_path},{extra_binds}" if bind_path else extra_binds
    full_cmd.extend(["--bind", combined_bind])
    
    full_cmd.extend([container_path] + cmd)
    
    if logger:
        logger.info(f"Running: {' '.join(full_cmd)}")
    
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = f"Command failed with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        if logger:
            logger.error(error_msg)
        raise subprocess.CalledProcessError(result.returncode, full_cmd, result.stdout, result.stderr)


def run_tumorsynth_cmd(cmd: list[str], logger=None):
    """Runs `apptainer exec $TUMORSYNTH_CONTAINER_PATH <cmd>`.

    Environment variables:
        TUMORSYNTH_CONTAINER_PATH: Path to the TumorSynth apptainer container (required).
        APPTAINER_BINDPATH: Comma-separated paths to bind into container (optional).
    """
    container_path = os.getenv("TUMORSYNTH_CONTAINER_PATH")
    if not container_path:
        raise EnvironmentError(
            "TUMORSYNTH_CONTAINER_PATH environment variable is not set. "
            "See README for container installation."
        )

    full_cmd = ["apptainer", "exec", "--nv", "--env", "CUDA_VISIBLE_DEVICES=0"]

    bind_path = os.getenv("APPTAINER_BINDPATH")
    extra_binds = "/projects"

    combined_bind = f"{bind_path},{extra_binds}" if bind_path else extra_binds

    full_cmd.extend(["--bind", combined_bind])
    full_cmd.extend([container_path] + cmd)

    if logger:
        logger.info(f"Running: {' '.join(full_cmd)}")

    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = f"Command failed with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        if logger:
            logger.error(error_msg)
        raise subprocess.CalledProcessError(result.returncode, full_cmd, result.stdout, result.stderr)



def get_transforms_3d(args: Namespace) -> Compose:
    """
    Returns transforms for 3D method evaluation.
    Evaluates at the computed cropping resolution used for 3D training,
    with no further slice range constraints.
    """
    dim_3d = args.training_img_dim
    return Compose([
        ResizeWithPadOrCrop(spatial_size=dim_3d),
        lambda x: x.contiguous()
    ])


def get_transforms_2d(args: Namespace) -> Compose:
    """
    Returns transforms for 2D slice-based method evaluation.
    This involves cropping/padding to the UNet 2D spatial size
    in the first two dimensions, and cropping the depth dimension
    to a specific slice range.
    """
    dim_2d = args.controlnet_infer["metrics_2d_resolution"]
    slice_start, slice_end = args.controlnet_infer["metrics_2d_slice_range"]
    
    return Compose([
        # Regular slicing along depth dimension (Z)
        lambda x: x[..., slice_start:slice_end],
        # ResizeWithPadOrCrop specifying -1 for Z means it won't be altered
        ResizeWithPadOrCrop(spatial_size=(*dim_2d, -1)),
        lambda x: x.contiguous()
    ])