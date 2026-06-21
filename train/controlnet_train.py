from __future__ import annotations

import argparse
from argparse import Namespace
import logging
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from eval.controlnet_eval import save_metrics
from eval.controlnet_eval import evaluate_predictions, get_metrics_means
from plot.plot import plot_predictions

from utils.diff_model_setting import load_configs, setup_logging
from infer.controlnet_infer import controlnet_infer_dataset
from models.autoencoder import load_autoencoder
from models.diff_model import load_diff_model
from models.controlnet import load_controlnet
from torch.utils.checkpoint import checkpoint
from utils.utils import (
    ReconModel,
    define_instance,
    setup_wandb_run,
    add_timestamp,
    run_dist,
    save_checkpoint,
    save_config,
    unwrap_ddp,
)
from data.dataloaders import get_controlnet_dataloaders


import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    mode: str,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_fn: torch.nn.Module,
    grad_scaler: GradScaler,
    noise_scheduler: torch.nn.Module,
    amp: bool,
    logger: logging.Logger,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Train for one epoch.
    
    Args:
        epoch: Current epoch number (1-indexed).
        unet: Frozen UNet model.
        controlnet: ControlNet model being trained.
        mode: Conditioning mode for ControlNet.
        train_loader: Training data loader.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        grad_scaler: Gradient scaler for AMP.
        noise_scheduler: Noise scheduler for diffusion.
        logger: Logger instance.
        device: Training device.
        rank: Process rank.
        amp: Whether to use automatic mixed precision.
        
    Returns:
        Tensor with accumulated loss information.
    """
    # Log current learning rate at the start of each epoch
    if rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 37)
        logger.info(f"======== Epoch {epoch}, lr {current_lr}. ========")
        logger.info("=" * 37)
        #logger.info(f"Epoch {epoch}, lr {current_lr}.")

    # Reset loss tracking at the start of each epoch
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    # Set model modes. ControlNet learns, U-Net stays frozen and deterministic.
    controlnet.train()
    unet.eval()

    for train_data in train_loader:
        """
        fu_embs = train_data["image_emb"].to(device)                # [B, C, H, W, D] - Follow-up embeddings 
        baseline_embs = train_data["baseline_emb"].to(device)       # [B, C, H, W, D] - Baseline embeddings
        baselines = train_data["baseline"].to(device)               # [B, C, H, W, D] - Raw baseline images
        doses = train_data["dose"].to(device)                       # [B, C, H, W, D] - Dose maps
        fu_times = train_data["fu_time"].to(device).unsqueeze(-1)   # [B, 1] - Follow-up times
        """
        
        # Load each modality's embedding separately
        fu_embs_t1    = train_data["T1_emb"].to(device)      # (B, 4, h, w, d)
        fu_embs_t1c   = train_data["T1C_emb"].to(device)     # (B, 4, h, w, d)
        fu_embs_flair = train_data["FLAIR_emb"].to(device)   # (B, 4, h, w, d)

        # Concatenate along channel dim
        fu_embs = torch.cat([fu_embs_t1, fu_embs_t1c, fu_embs_flair], dim=1)  # (B, 12, h, w, d)

        # Same for baseline
        baseline_embs_t1    = train_data["baseline_T1_emb"].to(device)
        baseline_embs_t1c   = train_data["baseline_T1C_emb"].to(device)
        baseline_embs_flair = train_data["baseline_FLAIR_emb"].to(device)

        baseline_embs = torch.cat([baseline_embs_t1, baseline_embs_t1c, baseline_embs_flair], dim=1)  # (B, 12, h, w, d)


        # Dose and fu_times unchanged
        doses    = train_data["dose"].to(device)
        fu_times = train_data["fu_time"].to(device).unsqueeze(-1)
        
        baseline_t1     =  train_data["baseline_T1"].to(device)
        baseline_t1c    =  train_data["baseline_T1C"].to(device)
        baseline_Flair  =  train_data["baseline_FLAIR"].to(device)
        
        baselines   =   torch.cat([baseline_t1, baseline_t1c, baseline_Flair], dim=1)

        baseline_inputs = baselines if mode == "concat" else baseline_embs
        
        optimizer.zero_grad(set_to_none=True)       # reset gradients

        #with autocast("cuda", enabled=amp):
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp):

            # Add noise
            noise = torch.randn_like(fu_embs)
            timesteps = noise_scheduler.sample_timesteps(fu_embs)
            noisy_latents = noise_scheduler.add_noise(original_samples=fu_embs, noise=noise, timesteps=timesteps)

            # Control net forward pass - Processes the conditioning information (dose, baseline, fu_time)
            controlnet_inputs = {
                "x": noisy_latents,
                "timesteps": timesteps,
                "dose": doses,
                "fu_times": fu_times,
                "baseline": baseline_inputs,
            }
            
            #down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)
            down_block_res_samples, mid_block_res_sample = checkpoint(lambda: controlnet(**controlnet_inputs),use_reentrant=False)

            # U-Net forward pass - Receives the ControlNet residuals and uses them to steer its predictions
            unet_inputs = {
                "x": noisy_latents,
                "timesteps": timesteps,
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
            }

            #model_output = unet(**unet_inputs)
            model_output = checkpoint(lambda: unet(**unet_inputs),use_reentrant=False)
            
            # Train U-Net to predict velocity x_0 - epsilon 
            model_gt = fu_embs - noise
            loss = loss_fn(model_output.float(), model_gt.float())
            del noisy_latents, down_block_res_samples, mid_block_res_sample, model_output
            torch.cuda.empty_cache()

        # Automatic Mixed Precision
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


def controlnet_train(args: Namespace, rank: int, device: torch.device) -> None:
    """
    Main function to train ControlNet.

    Args:
        args (Namespace): Configuration arguments.
        rank (int): Process rank.
        device (torch.device): Device to use for training.
    """
    set_determinism(seed=args.controlnet_train["random_seed"])
    run_dir = Path(args.output_dir) / args.run_name
    
    if args.disable_wandb:
        run = None
    elif rank == 0:
        run = setup_wandb_run(
            project_name="maisi-controlnet",
            run_name=args.run_name,
            config=args,
        )
    else:
        run = None
    
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(args, run_dir)

    logger = setup_logging("controlnet_train")

    if rank == 0:
        logger.info(f"[config] data_base_dir -> {args.data_base_dir}.")
        logger.info(f"[config] json_datalist_path -> {args.json_datalist_path}.")
        logger.info(f"[config] json_domainrand_datalist_path -> {args.json_domainrand_datalist_path}.")
        logger.info(f"[config] trained_autoencoder_path -> {args.trained_autoencoder_path}.")
        logger.info(f"[config] trained_diff_model_path -> {args.trained_diff_model_path}.")
        logger.info(f"[config] trained_controlnet_path -> {args.trained_controlnet_path}.")
        logger.info(f"[config] mode -> {args.mode}.")
        logger.info(f"[config] lr -> {args.controlnet_train['lr']}.")
        logger.info(f"[config] batch_size -> {args.controlnet_train['batch_size']}.")
        logger.info(f"[config] num_epochs -> {args.controlnet_train['n_epochs']}.")

    args.controlnet_def["conditioning_embedding_in_channels"] = 1 if args.mode == "embed" else 4  # 1 channel for dose and 3 channels for modalities

    autoencoder = load_autoencoder(
        trained_autoencoder_path=args.trained_autoencoder_path,
        model_def_args=args,
        device=device,
    )

    unet = load_diff_model(
        trained_diff_model_path=args.trained_diff_model_path,
        model_def_args=args,
        device=device,
    )
    
    controlnet = load_controlnet(
        trained_controlnet_path=args.trained_controlnet_path,
        unet=unet,
        model_def_args=args,
        device=device,
    )


    for p in unet.parameters():
        p.requires_grad = False

    recon_model = ReconModel(autoencoder).to(device)
    
    if dist.is_initialized():
        controlnet = DDP(controlnet, device_ids=[device], output_device=rank, find_unused_parameters=True)
    

    if args.method == "baseline":
        datalist_path = args.json_datalist_path          # conditional datalist
    elif args.method == "domainRand":
        datalist_path = args.json_domainrand_datalist_path  # domainRand datalist
    else:
        raise ValueError(f"Unknown method: {args.method}")


    train_loader, val_loader, _ = get_controlnet_dataloaders(datalist_path, args, rank, logger)
    
    
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=args.controlnet_train["lr"])
    noise_scheduler = define_instance(args, "noise_scheduler")
    loss_fn = torch.nn.L1Loss()
    grad_scaler = GradScaler("cuda")
    batch_size = args.controlnet_train["batch_size"]
    total_steps = (args.controlnet_train["n_epochs"] * len(train_loader.dataset)) / batch_size
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=1.0)

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(1, args.controlnet_train["n_epochs"] + 1):
        loss_torch = train_one_epoch(
            epoch=epoch,
            unet=unet,
            controlnet=controlnet,
            mode=args.mode,
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
            loss_torch_epoch = loss_torch[0] / loss_torch[1]    # Compute and log average loss
            logger.info(f"epoch {epoch} average loss: {loss_torch_epoch:.4f}.")
            
            if run is not None:
                run.log({
                    "train loss": loss_torch_epoch,
                    "learning rate": optimizer.param_groups[0]["lr"],
                    "optimization steps": epoch * len(train_loader.dataset) / batch_size,
                }, step=epoch)

        val_interval = args.controlnet_train["val_interval"]
        if epoch == 1 or epoch % val_interval == 0:
            if rank == 0:
                logger.info(f"Saving checkpoint for epoch {epoch}...")
                save_checkpoint(
                    model=controlnet,
                    output_path=run_dir / f"controlnet_epoch_{epoch}.pt",
                )
                logger.info(f"Running validation for epoch {epoch}")
            
            controlnet_infer_dataset(
                controlnet=unwrap_ddp(controlnet),
                unet=unet,
                recon_model=recon_model,
                dataloader=val_loader,
                noise_scheduler=noise_scheduler,
                output_dir=run_dir / "predictions",
                device=device,
                args=args,
            )
            
          
            metrics_df = evaluate_predictions(
                dataloader=val_loader,
                run_dir=str(run_dir / "predictions"),  # <-- add /predictions
                metrics=args.controlnet_infer["val_metrics"],
                prev_run_dir=None,
                args=args,
                logger=logger,
                mode="train"
                )
            
            if rank == 0:
                save_metrics(metrics_df, run_dir / "metrics.csv")
                logger.info(f"Metrics saved to {run_dir / 'metrics.csv'}")

                metrics_means = get_metrics_means(metrics_df)
                prediction_plot_paths = plot_predictions(
                    dataloader=val_loader,
                    run_dir=str(run_dir),
                    args=args,
                    num_samples=1,
                    mode = "train",
                    logger=logger,
                    )
                
                

                if run is not None:
                    run.log({f"val_metrics/{k}": v for k, v in metrics_means.items()}, step=epoch)
                    logger.info(f"DEBUG logged val_metrics to WandB: {list(metrics_means.keys())}")

                    if prediction_plot_paths:
                        wandb_images = {}
                        for plot_paths in prediction_plot_paths:
                            logger.info(f"DEBUG plot_paths entry: {plot_paths}")
                            for mod_name in ["T1", "T1C", "FLAIR"]:
                                if mod_name in plot_paths.get("middle_slices", ""):
                                    path = plot_paths["middle_slices"]
                                    exists = Path(path).exists()
                                    logger.info(f"DEBUG {mod_name} middle_slices path={path} exists={exists}")
                                    if exists:
                                        wandb_images[f"val_plots/{mod_name}"] = wandb.Image(path)
                                    break

                        logger.info(f"DEBUG wandb_images keys: {list(wandb_images.keys())}")
                        if wandb_images:
                            run.log(wandb_images, step=epoch)
                            logger.info("DEBUG successfully logged val_plots to WandB")
                        else:
                            logger.warning("DEBUG wandb_images is empty, nothing logged to WandB")
                    else:
                        logger.warning("DEBUG prediction_plot_paths is empty or None")

            torch.cuda.empty_cache()

    if rank == 0 and run is not None:
        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Training")
    parser.add_argument("--env_config", type=str, default="./configs/env_config_controlnet_train.json")
    parser.add_argument("--model_config", type=str, default="./configs/model_config_controlnet_train.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_maisi3d-rflow.json")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--disable_wandb", action="store_true")
    #parser.add_argument("--mode", type=str, required=True, choices=["embed", "concat"])
    parser.add_argument("--mode", type=str, default="concat", choices=["embed", "concat"])
    parser.add_argument("--method", type=str, required=True, choices=["baseline", "domainRand"])
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    
    load_configs(args, [args.env_config, args.model_config, args.model_def])
    args.run_name = add_timestamp(args.run_name)
    
    run_dist(controlnet_train, args)


