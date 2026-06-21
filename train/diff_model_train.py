from __future__ import annotations

import argparse
from argparse import Namespace
import logging
import wandb
from pathlib import Path

import torch
import torch.distributed as dist
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils.visualizer import np_img_to_slices

from utils.diff_model_setting import load_configs, setup_logging
from infer.diff_model_infer import diff_model_infer
from utils.utils import (
    ReconModel,
    define_instance,
    setup_wandb_run,
    add_timestamp,
    run_dist,
    save_config,
    save_checkpoint,
    save_image,
    unwrap_ddp,
)
from data.dataloaders import get_diff_model_dataloader
from models.autoencoder import load_autoencoder
from models.diff_model import load_diff_model



def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_fn: torch.nn.L1Loss,
    grad_scaler: GradScaler,
    noise_scheduler: torch.nn.Module,
    amp: bool,
    logger: logging.Logger,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    if rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}, lr {current_lr}.")

    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    unet.train()

    for train_data in train_loader:
        
        #images = train_data["image_emb"].to(device)
        images = torch.cat([
        train_data["T1_emb"].to(device),
        train_data["T1C_emb"].to(device),
        train_data["FLAIR_emb"].to(device),
        ], dim=1)  # dim=1 is the channel dimension
        
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn_like(images)
            timesteps = noise_scheduler.sample_timesteps(images)
            noisy_latents = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            unet_inputs = {
                "x": noisy_latents,
                "timesteps": timesteps,
            }

            model_output = unet(**unet_inputs)
            model_gt = images - noise
            loss = loss_fn(model_output.float(), model_gt.float())

        if amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def diff_model_train(args: Namespace, rank: int, device: torch.device) -> None:
    """
    Main function to train a diffusion model.

    Args:
        args (Namespace): Configuration arguments.
        rank (int): Process rank.
        device (torch.device): Device to use for training.
    """
    set_determinism(seed=args.diffusion_unet_train["random_seed"])
    run_dir = Path(args.output_dir) / args.run_name
    
    if rank == 0:
        run = setup_wandb_run(
            project_name="maisi-diff-model",
            run_name=args.run_name,
            config=args,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(args, run_dir)

    logger = setup_logging("diff_model_train")

    if rank == 0:
        logger.info(f"[config] data_base_dir -> {args.embedding_base_dir}.")
        logger.info(f"[config] json_datalist_path -> {args.json_datalist_path}.")
        logger.info(f"[config] trained_diff_model_path -> {args.trained_diff_model_path}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] batch_size -> {args.diffusion_unet_train['batch_size']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")

    unet = load_diff_model(
        trained_diff_model_path=args.trained_diff_model_path,
        model_def_args=args,
        device=device
    )
    
    if dist.is_initialized():
        unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
        unet = DDP(unet, device_ids=[device], find_unused_parameters=False)
    
    autoencoder = load_autoencoder(
        trained_autoencoder_path=args.trained_autoencoder_path,
        model_def_args=args,
        device=device,
    )
    
    train_loader = get_diff_model_dataloader(args.json_datalist_path, args, rank, logger)
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_unet_train["lr"])
    noise_scheduler = define_instance(args, "noise_scheduler")
    loss_fn = torch.nn.L1Loss()
    grad_scaler = GradScaler("cuda")
    batch_size = args.diffusion_unet_train["batch_size"]
    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / batch_size
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(1, args.diffusion_unet_train["n_epochs"] + 1):
        loss_torch = train_one_epoch(
            epoch=epoch,
            unet=unet,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            grad_scaler=grad_scaler,
            noise_scheduler=noise_scheduler,
            amp=args.amp,
            logger=logger,
            rank=rank,
            device=device,
        )

        loss_torch = loss_torch.tolist()

        if rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch} average loss: {loss_torch_epoch:.4f}.")
            run.log({
                "train loss": loss_torch_epoch,
                "learning rate": optimizer.param_groups[0]["lr"],
                "optimization steps": epoch * len(train_loader.dataset) / batch_size,
            }, step=epoch)

        if epoch == 1 or epoch % 10 == 0:
            if dist.is_initialized():
                dist.barrier()

            if rank == 0:
                logger.info(f"Saving checkpoint for epoch {epoch}...")

                save_checkpoint(
                    model=unet,
                    output_path=run_dir / f"unet_epoch_{epoch}.pt",
                )

                pred_path = f"{run_dir}/pred_epoch_{epoch}.nii.gz"
                recon_model = ReconModel(autoencoder).to(device)

                img = diff_model_infer(
                    unet=unwrap_ddp(unet),
                    recon_model=recon_model,
                    device=device,
                    args=args
                )
                
               
                modality_names = ["T1", "T1C", "FLAIR"]
                for i, (mod_img, mod_name) in enumerate(zip(img, modality_names)):
                    mod_pred_path = f"{run_dir}/pred_epoch_{epoch}_{mod_name}.nii.gz"
                    save_image(mod_img, mod_pred_path)
                    logger.info(f"Saved inference output: {mod_pred_path}")
                    
                    axial, coronal, sagittal = np_img_to_slices(mod_img)
                    
                    run.log({
                        f"prediction/{mod_name}/axial": wandb.Image(axial, caption=f"Prediction {mod_name} - Epoch {epoch}"),
                        f"prediction/{mod_name}/coronal": wandb.Image(coronal, caption=f"Prediction {mod_name} - Epoch {epoch}"),
                        f"prediction/{mod_name}/sagittal": wandb.Image(sagittal, caption=f"Prediction {mod_name} - Epoch {epoch}"),
                        },
                    step=epoch,
                    )


    if rank == 0:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument("--env_config", type=str, default="./configs/env_config_diff_model_train.json")
    parser.add_argument("--model_config", type=str, default="./configs/model_config_diff_model_train.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_maisi3d-rflow.json")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    load_configs(args, [args.env_config, args.model_config, args.model_def])
    args.run_name = add_timestamp(args.run_name)
    
    run_dist(diff_model_train, args)
