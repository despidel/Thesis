"""Configure image dimensions and compute normalization statistics.

Usage: python -m data.configure_dims_and_stats
"""

import argparse
import json
import math

import numpy as np
import torch
from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, ResizeWithPadOrCropd, ClipIntensityPercentilesd, ScaleIntensityRangePercentilesd
from tqdm import tqdm


def compute_dims(datalist_path: str, model_def: dict, batch_size: int, num_workers: int) -> tuple[list[int], list[int]]:
    # Compute alignment multiple from model architecture
    # Training image dimensions must be divisible by this value
    
    autoencoder_levels = len(model_def["autoencoder_def"]["num_channels"])
    unet_levels = len(model_def["diffusion_unet_def"]["num_channels"])
    multiple = 2 ** ((autoencoder_levels - 1) + (unet_levels - 1))
    print(f"Alignment multiple: {multiple}")

    # Load datalist JSON and collect image paths 
    with open(datalist_path) as f:
        datalist = json.load(f)

    
    
    samples = [{"image": s["T1"]} for split in datalist.values() if isinstance(split, list) for s in split]
    print(f"Processing {len(samples)} images...")

    loader = DataLoader(
        Dataset(samples, Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        ])),
        batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x: x,
    )

    # Read shape from first image 
    # All images are assumed to have the same shape (registered/resampled data)
    # No need to iterate all samples — shape is consistent across the dataset
    shape = None
    for batch in tqdm(loader, desc="Computing dims"):
        for item in batch:
            data = item["image"].squeeze(0).numpy()
            if shape is None:
                shape = np.array(data.shape)
                break  # stop after first image
        if shape is not None:
            break

    # Align size to nearest multiple of alignment factor 
    # Ensures compatibility with autoencoder + UNet downsampling levels
    raw_size = shape
    aligned_size = np.ceil(raw_size / multiple).astype(int) * multiple

    print(f"  raw_img_dim: {shape.tolist()}, training_img_dim: {aligned_size.tolist()}")
    return aligned_size.tolist(), shape.tolist()

def compute_normalization_stats(datalist_path: str, img_dim: list[int], batch_size: int, num_workers: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(datalist_path) as f:
        full_train_files = json.load(f)["train"]
    
    # Define all keys to normalize
    # Include dose and all three baseline modalities
    keys = ["dose", "baseline_T1", "baseline_T1C", "baseline_FLAIR"]
    
    # 2. Build the train_files list using the new keys
    train_files = [
        {
            "dose": s["dose"], 
            "baseline_T1": s["baseline_T1"],
            "baseline_T1C": s["baseline_T1C"],
            "baseline_FLAIR": s["baseline_FLAIR"]
        } for s in full_train_files
    ]
    print(f"Processing {len(train_files)} training samples for keys: {keys}")

   
    loader = DataLoader(
    CacheDataset(train_files, Compose([
        # Load all modality images and ensure channel-first format (C, H, W, D)
        LoadImaged(keys=keys, ensure_channel_first=True),
        # Resize/pad/crop all images to the target training dimensions
        ResizeWithPadOrCropd(keys=keys, spatial_size=img_dim),
        # Clip extreme intensity values (top/bottom 0.5%) to remove outliers
        ClipIntensityPercentilesd(keys=keys, lower=0.5, upper=99.5),
        # Rescale intensities to [0, 1] based on the clipped percentile range
        ScaleIntensityRangePercentilesd(keys=keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
    ]), cache_rate=0, num_workers=num_workers),
    num_workers=num_workers, batch_size=batch_size, shuffle=False,
)
    # Mean and Std loops will loop through all 4 keys and compute stats for each one.
    sums = {k: torch.tensor(0.0, device=device) for k in keys}
    sq_diffs = {k: torch.tensor(0.0, device=device) for k in keys}
    counts = {k: torch.tensor(0, dtype=torch.int64, device=device) for k in keys}

    for batch in tqdm(loader, desc="Mean"):
        for k in keys:
            data = batch[k].to(device)
            sums[k] += data.sum()
            counts[k] += data.numel()

    means = {k: (sums[k] / counts[k]).item() for k in keys}

    for batch in tqdm(loader, desc="Std"):
        for k in keys:
            data = batch[k].to(device)
            sq_diffs[k] += ((data - means[k]) ** 2).sum()

    stats = {k: {"mean": means[k], "std": math.sqrt((sq_diffs[k] / counts[k]).item())} for k in keys}
    for k in keys:
        print(f"  {k}: mean={stats[k]['mean']:.4f}, std={stats[k]['std']:.4f}")
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config_diff_model", default="configs/env_config_diff_model_train.json")
    parser.add_argument("--env_config_controlnet", default="configs/env_config_controlnet_train.json")
    parser.add_argument("--model_def", default="./configs/config_maisi3d-rflow.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--method", type=str, choices=["baseline", "domainRand"], required=True)
    #parser.add_argument("--method", type=str, choices=["baseline", "domainRand"])
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.model_def) as f:
        model_def = json.load(f)

    

    with open(args.env_config_diff_model) as f:
        env_config_diff_model = json.load(f)
        if args.method == "domainRand":
            dims_datalist = env_config_diff_model["json_domainrand_datalist_path"]
        else:
            dims_datalist = env_config_diff_model["json_datalist_path"]

    with open(args.env_config_controlnet) as f:
        env_config_controlnet = json.load(f)
        if args.method == "domainRand":
            norm_datalist = env_config_controlnet["json_domainrand_datalist_path"]
        else:
            norm_datalist = env_config_controlnet["json_datalist_path"]
    
    print("\n=== Computing image dimensions ===")
    print(f"dims_datalist: {dims_datalist}")
    training_dim, raw_dim = compute_dims(dims_datalist, model_def, args.batch_size, args.num_workers)
    model_def["raw_img_dim"], model_def["training_img_dim"] = raw_dim, training_dim
    with open(args.model_def, "w") as f:
        json.dump(model_def, f, indent=4)
    print(f"Updated {args.model_def}")

    print("\n=== Computing normalization stats ===")
    print(f"norm_datalist: {norm_datalist}")
    stats = compute_normalization_stats(norm_datalist, training_dim, args.batch_size, args.num_workers)
    stats_path = norm_datalist.replace(".json", "_normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
