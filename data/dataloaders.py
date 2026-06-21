"""Dataloader creation functions for diffusion model and ControlNet training/inference.

This module provides centralized dataloader creation with consistent transforms
and normalization across training, validation, and inference.
"""

import json
import logging
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.transforms import Compose, Lambdad, LoadImaged, ResizeWithPadOrCropd
import torchio as tio


def load_datalist(json_datalist_path: Path) -> dict:
    """Load datalist from JSON file.
    
    Args:
        json_datalist_path: Path to the datalist JSON file.
        
    Returns:
        Dict with keys 'train', 'val', 'test' containing sample lists.
    """
    with open(json_datalist_path, "r") as f:
        return json.load(f)


def get_diff_model_dataloader(
    json_datalist_path: Path,
    args: Namespace,
    rank: int,
    logger: logging.Logger,
) -> DataLoader:
    """
    Get dataloader for diffusion model training.
    
    Args:
        json_datalist_path: Path to the datalist JSON file.
        args: Configuration arguments.
        rank: Process rank for distributed training.
        logger: Logger instance.
        
    Returns:
        Training dataloader.
    """
    with open(json_datalist_path, "r") as f:
        datalist = json.load(f)

    train_files = datalist["train"][:args.num_samples] if args.num_samples else datalist["train"]

    if rank == 0:
        logger.info(f"train: {len(train_files)} file(s)")

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[rank]

    """
    train_transforms = Compose([
        LoadImaged(keys=["image_emb"], ensure_channel_first=True),
    ])
    """
    
    train_transforms = Compose([LoadImaged(keys=["T1_emb", "T1C_emb", "FLAIR_emb"], ensure_channel_first=True),])


    batch_size = args.diffusion_unet_train["batch_size"]
    cache_rate = args.diffusion_unet_train["cache_rate"]

    dataset = CacheDataset(train_files, train_transforms, cache_rate=cache_rate, num_workers=4)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)

    return dataloader


def get_controlnet_dataloaders(
    json_datalist_path: Path,
    args: Namespace,
    rank: int,
    logger: logging.Logger,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for ControlNet training/inference.
    
    Includes normalization transforms for dose and baseline using pre-computed stats.
    
    Args:
        json_datalist_path: Path to the datalist JSON file.
        args: Configuration arguments.
        rank: Process rank for distributed training.
        logger: Logger instance.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    with open(json_datalist_path, "r") as f:
        datalist = json.load(f)

    train_files = datalist["train"][:args.num_samples] if args.num_samples else datalist["train"]
    val_files = datalist["train"][:args.num_samples] if args.num_samples else datalist["val"]
    test_files = datalist["test"][:args.num_samples] if args.num_samples else datalist["test"]
    
    if rank == 0:
        logger.info(f"train: {len(train_files)} file(s)")
        logger.info(f"val: {len(val_files)} file(s)")
        logger.info(f"test: {len(test_files)} file(s)")
    
    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[rank]
        val_files = partition_dataset(
            data=val_files,
            num_partitions=dist.get_world_size(),
            even_divisible=False,
        )[rank]
        test_files = partition_dataset(
            data=test_files,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[rank]
    
    # Load normalization stats (must exist)
   
    datalist_dir = Path(json_datalist_path).parent
    stats_candidates = list(datalist_dir.glob("*_normalization_stats.json"))

    if len(stats_candidates) == 0:
        raise FileNotFoundError(
            f"No normalization stats file found in {datalist_dir}\n"
            "Run 'python -m data.configure_dims_and_stats' first."
        )
    elif len(stats_candidates) > 1:
        raise FileNotFoundError(
            f"Multiple normalization stats files found in {datalist_dir}:\n"
            + "\n".join(str(p) for p in stats_candidates)
            + "\nPlease ensure only one *_normalization_stats.json exists."
        )

  
    
    stats_path = stats_candidates[0]
    logger.info(f"Loading normalization stats from {stats_path}")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    """
    for key in ["dose", "baseline"]:
        logger.info(f"{key} stats: mean={stats[key]['mean']:.4f}, std={stats[key]['std']:.4f}")
    
    dose_mean, dose_std = stats["dose"]["mean"], stats["dose"]["std"]
    baseline_mean, baseline_std = stats["baseline"]["mean"], stats["baseline"]["std"]
    """

    if not hasattr(args, 'img_filename'):
        args.img_filename = {"T1": "T1.nii.gz", "T1C": "T1C.nii.gz", "FLAIR": "FLAIR.nii.gz"}

    for key in ["dose"] + [f"baseline_{mod}" for mod in args.img_filename]:
        logger.info(f"{key} stats: mean={stats[key]['mean']:.4f}, std={stats[key]['std']:.4f}")
    # Extract stats
    dose_mean, dose_std = stats["dose"]["mean"], stats["dose"]["std"]
    baseline_stats = {
        mod: (stats[f"baseline_{mod}"]["mean"], stats[f"baseline_{mod}"]["std"])
        for mod in args.img_filename
        }
    

    modality_keys     = list(args.img_filename.keys())                    # ["fT1", "fT1c", "fFLAIR"]
    modality_emb_keys = [f"{mod}_emb" for mod in modality_keys]           # ["fT1_emb", ...]
    baseline_keys     = [f"baseline_{mod}" for mod in modality_keys]      # ["baseline_fT1", ...]
    baseline_emb_keys = [f"baseline_{mod}_emb" for mod in modality_keys]  # ["baseline_fT1_emb", ...]

    # Shared normalization lambdas (reused in both transforms)
    baseline_norm_transforms = [
        Lambdad(
            keys=[f"baseline_{mod}"],
            func=lambda x, m=mean, s=std: (x - m) / s
        )
        for mod, (mean, std) in baseline_stats.items()
    ]

    # Train transforms (no raw image needed)
    train_transforms = Compose([
        LoadImaged(
            #keys=modality_emb_keys + baseline_keys + baseline_emb_keys + ["dose", "roi_mask"],
            keys=modality_emb_keys + baseline_keys + baseline_emb_keys + ["dose"],
            ensure_channel_first=True),

        ResizeWithPadOrCropd(keys=["dose"] + baseline_keys, spatial_size=args.training_img_dim),
        Lambdad(keys=["fu_time"], func=lambda x: torch.tensor(x, dtype=torch.float32)),
        Lambdad(keys=["dose"], func=lambda x: (x - dose_mean) / dose_std),
        *baseline_norm_transforms,
        ])

    # Val/Test transforms (include raw images for metrics)
    val_test_transforms = Compose([
        LoadImaged(
            keys=modality_keys + modality_emb_keys + baseline_keys + baseline_emb_keys + ["dose"],
            ensure_channel_first=True),
        ResizeWithPadOrCropd(keys=["dose"] + baseline_keys, spatial_size=args.training_img_dim),
        Lambdad(keys=["fu_time"], func=lambda x: torch.tensor(x, dtype=torch.float32)),
        Lambdad(keys=["dose"], func=lambda x: (x - dose_mean) / dose_std),
        *baseline_norm_transforms,
        ])


    

    batch_size = args.controlnet_train["batch_size"]
    cache_rate = args.controlnet_train["cache_rate"]
    num_workers = args.controlnet_train["num_workers"]
    train_shuffle = False if args.num_samples else True
    
    train_dataset = CacheDataset(train_files, train_transforms, cache_rate=cache_rate, num_workers=num_workers)

    
    
    val_dataset = CacheDataset(val_files, val_test_transforms, cache_rate=cache_rate, num_workers=num_workers)
    
    
    test_dataset = CacheDataset(test_files, val_test_transforms, cache_rate=0, num_workers=num_workers)
    
    loader_kwargs = {
    "num_workers": num_workers,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
