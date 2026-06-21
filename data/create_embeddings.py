"""Create embeddings for training diffusion model and ControlNet.

This script processes images through an autoencoder to create latent embeddings
that are used for training.

Usage:
    python -m data.create_embeddings --num_samples 4 --with_recon --method <baseline or domainRand>
    python -m data.create_embeddings --run_name embedding_creation --method baseline
    python -m data.create_embeddings --run_name embedding_creation --method domainRand
"""

import argparse
from argparse import Namespace
import json
import logging
from pathlib import Path

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ResizeWithPadOrCropd,
    ResizeWithPadOrCrop,
    ClipIntensityPercentilesd,
    ScaleIntensityd,
)

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist #PyTorch's distributed computing module, which enables training across multiple GPUs or multiple machines in parallel.
from monai.data import DataLoader, CacheDataset, partition_dataset
from monai.transforms import Compose

from models.autoencoder import load_autoencoder
from utils.diff_model_setting import load_configs, setup_logging
from utils.utils import setup_wandb_run, add_timestamp, run_dist
from utils.visualizer import plot_nifti_columns, get_roi_mask_center_indices


def get_items(datalist: dict, args: Namespace) -> list:
    items = []
    num_samples = 0

    first_sample =  datalist[list(datalist.keys())[0]][0]
    emb_keys = [key for key in first_sample if "_emb" in key]
    
    for split_data in datalist.values():
        for sample in split_data:    
            for emb_key in emb_keys:
                emb_path = sample[emb_key]
                img_key = emb_key.split("_emb")[0]

                items.append({
                    "image": sample[img_key],
                    "image_emb": emb_path,
                    #"roi_mask": sample["roi_mask"],
                })

            num_samples += 1
            if args.num_samples is not None and num_samples >= args.num_samples:
                return items
    
    return items


def prepare_data(
    json_datalist_path: Path,
    logger: logging.Logger,
    rank: int,
    args: Namespace,
) -> DataLoader:
    logger.info(f"Loading datalist json: {json_datalist_path}")
    with open(json_datalist_path, "r") as file:
        datalist = json.load(file)
    
    items = get_items(datalist, args)

    if rank == 0:
        logger.info(f"Creating {len(items)} new embedding(s)")

    if dist.is_initialized():
        dataset = partition_dataset(
            data=items,
            shuffle=False,
            num_partitions=dist.get_world_size(),
            even_divisible=False,
        )[rank]
    else:
        dataset = items

    # pre-processing pipeline
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=args.training_img_dim),
        ClipIntensityPercentilesd(keys=["image"], lower=0.1, upper=99.9),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    ])

    cache_rate = args.create_embeddings["cache_rate"]

    dataset = CacheDataset(dataset, transforms, cache_rate=cache_rate, num_workers=4)
    loader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)
    
    return loader


def save_reconstruction_and_comparison_plot(
    recon: torch.Tensor,
    input_img_path: str,
    emb_path: str,
    raw_img_dim: list[int],
    logger: logging.Logger,
) -> None:
    affine = nib.load(input_img_path).affine

    recon = ResizeWithPadOrCrop(spatial_size=raw_img_dim)(recon.squeeze(0))
    recon = recon.squeeze().float().cpu().detach().numpy()
    recon = nib.Nifti1Image(recon, affine=affine)
    recon_path = emb_path.replace("_emb.nii.gz", "_recon.nii.gz")
    nib.save(recon, recon_path)
    logger.info(f"Saved reconstructed image to {recon_path}")

    # Use center slice of the image as the comparison slice
    k = recon.shape[-1] // 2

    # Create comparison graph for original image vs reconstructed image
    plot_nifti_columns(
        nifti_paths_dict={
            "Original": input_img_path,
            "Reconstructed": recon_path,
        },
        vmin_vmax=((None, None), (0.0, 1.0)),
        output_path=recon_path.replace(".nii.gz", ".png"),
        slice_indices={"axial": k},
        font_size=24,
        logger=logger,
    )

def process_file(
    sample: dict,
    autoencoder: torch.nn.Module,
    with_recon: bool,
    raw_img_dim: list[int],
    device: torch.device,
    logger: logging.Logger,
) -> None:
    
    emb_path = sample["image_emb"][0]
    
    # Skip if embedding already exists
    if Path(emb_path).exists():
        logger.info(f"Embedding already exists, skipping: {emb_path}")
        return

    img = sample["image"].to(device)                # (1, 1, H, W, D)
    
    Path(emb_path).parent.mkdir(parents=True, exist_ok=True)

    with torch.amp.autocast("cuda"):
        z = autoencoder.encode_stage_2_inputs(img)  # (1, 4, h, w, d)
        
        emb = z.squeeze(0).cpu().detach().numpy().transpose(1, 2, 3, 0)
        emb = nib.Nifti1Image(np.float32(emb), affine=None)
        nib.save(emb, emb_path)
        logger.info(f"Saved embedding z ({z.size()}, {z.dtype}) to {emb_path}")

        if with_recon:
            recon = autoencoder.decode_stage_2_outputs(z) # decode the latent embedding z back into image space
            save_reconstruction_and_comparison_plot(
                recon=recon,
                #roi_mask=sample["roi_mask"],
                input_img_path=img.meta["filename_or_obj"][0],
                emb_path=emb_path,
                raw_img_dim=raw_img_dim,
                logger=logger,
            )


@torch.inference_mode() #everything inside this function is for inference only, not training. (No gradients tracked)
def create_embeddings(
    with_recon: bool,
    run_name: str,
    rank: int,
    device: torch.device,
    args: Namespace,
) -> None:
    logger = setup_logging("create_embeddings")
    if run_name and rank == 0:
        run = setup_wandb_run("maisi-diffusion-model", run_name, args)

    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)
    
    # preprocess data
    loader = prepare_data(
        json_datalist_path=Path(args.json_datalist_path),
        logger=logger,
        rank=rank,
        args=args,
    )

    # Load pretrained autoencoder
    if len(loader.dataset) > 0:
        autoencoder = load_autoencoder(args.trained_autoencoder_path, args, device)
        autoencoder.eval()

    for i, sample in enumerate(loader):

        logger.info(f"Processing item {i + 1}/{len(loader.dataset)} on rank {rank}")
        
        # create and save embeddings, save reconstructed image, save comparison graph
        process_file(
            sample=sample,
            autoencoder=autoencoder,
            with_recon=with_recon,
            raw_img_dim=args.raw_img_dim,
            device=device,
            logger=logger,
        )
    
    logger.info("Done creating embeddings")
        
    if run_name and rank == 0:
        run.finish()

    if dist.is_initialized():
        dist.barrier()


def main(args, rank, device):
    create_embeddings(
        with_recon=args.with_recon,
        run_name=args.run_name,
        rank=rank,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding creation")
    parser.add_argument("--env_config", type=str, default="./configs/env_config_create_emb.json")
    parser.add_argument("--model_config", type=str, default="./configs/model_config_diff_model_train.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_maisi3d-rflow.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--with_recon", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--method", type=str, choices=["baseline", "domainRand"])
    args = parser.parse_args()
    
    load_configs(args, [args.env_config, args.model_config, args.model_def])

    if args.method == "domainRand":
        args.json_datalist_path = args.json_domainrand_datalist_path
    run_dist(main, args)
