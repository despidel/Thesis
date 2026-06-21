"""
ControlNet Inference
Inference for Baseline model and Domain Randomisation Model
    python -m infer.controlnet_infer --mode concat --max_follow_ups <number of maximum follow ups per baseline image> --num_subjects <number of subjects> --run_name <name>
    

"""
from __future__ import annotations
from monai.data.dataloader import DataLoader

import argparse
from argparse import Namespace
import gc
import nibabel as nib
import numpy as np
from pathlib import Path
import time
import torch

from monai.inferers import SlidingWindowInferer
from monai.transforms import ResizeWithPadOrCrop
from monai.utils import set_determinism
from collections import defaultdict
from utils.diff_model_setting import load_configs, setup_logging
from data.dataloaders import get_controlnet_dataloaders
from eval.controlnet_eval import evaluate_predictions, save_metrics
from models.autoencoder import load_autoencoder
from models.diff_model import load_diff_model
from models.controlnet import load_controlnet
from plot.plot import plot_predictions
from plot.plot_longitudinal import plot_longitudinal_predictions
from plot.metrics_summary_plot import plot_all_metrics_summary, plot_dice_by_time, plot_image_quality_by_time, plot_metrics_by_time_multilevel, plot_volumes_by_time, plot_volumes_combined
from utils.utils import (
    ReconModel,
    add_timestamp,
    define_instance,
    dynamic_infer,
    save_image,
    save_config,
    run_dist,
    get_pred_path,
)
from data.Transforms import apply_transform

def get_filtered_dataloader(
    dataloader: DataLoader,
    max_follow_ups: int = None,
    num_subjects: int = None,  # max number of unique subjects to include
) -> DataLoader:
    """
    Returns a new DataLoader containing only the subset of samples that pass
    the max_follow_ups and num_subjects filters — matching the logic previously
    inline in controlnet_infer_dataset, so that evaluate_predictions sees the
    exact same samples as inference.
    """
    patient_follow_ups = defaultdict(int)
    seen_subjects = set()   # tracks unique subjects for num_subjects cap
    filtered_indices = []

    for idx, sample in enumerate(dataloader):
        # Extract subject ID (e.g. sub-IM0318) from baseline T1 file path
        subject = Path(sample["baseline_T1"].meta["filename_or_obj"][0]).parts[-4]
        # Skip new subjects once we've reached the subject cap
        if num_subjects is not None and subject not in seen_subjects and len(seen_subjects) >= num_subjects:
            continue
        # Skip follow-ups beyond the per-patient limit
        if max_follow_ups is not None and patient_follow_ups[subject] >= max_follow_ups:
            continue
        seen_subjects.add(subject)
        patient_follow_ups[subject] += 1
        filtered_indices.append(idx)

    # Wrap the accepted indices in a Subset and return a new DataLoader
    subset = torch.utils.data.Subset(dataloader.dataset, filtered_indices)
    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
    )

@torch.inference_mode()
def controlnet_infer(
    sample: dict,
    controlnet: torch.nn.Module,
    unet: torch.nn.Module,
    recon_model: torch.nn.Module,
    noise_scheduler: torch.nn.Module,
    device: torch.device,
    args: Namespace,
    logger,
    transform_level = None,
) -> np.ndarray:
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        sample: The sample dict from the dataloader.
        controlnet: The controlnet model.
        unet: The diffusion U-Net model.
        recon_model: The reconstruction model.
        noise_scheduler: The noise scheduler for the diffusion process.
        device: The device to run the computation on.
        args: Configuration arguments.
        logger: Logger instance.

    Returns:
        Generated synthetic image as numpy array.
    """
    """
    dose = sample["dose"].to(device)                                # [1, C, H, W, D] - Dose maps
    baseline_emb = sample["baseline_emb"].to(device)                # [1, C, H, W, D] - Baseline embeddings
    baseline = sample["baseline"].to(device)                        # [1, C, H, W, D] - Raw baseline images
    fu_time = sample["fu_time"].to(device).unsqueeze(-1)            # [1, 1] - Follow-up time
    """

    # Save original file path before transforms, as they corrupt tensor metadata
    baseline_embs_t1    = sample["baseline_T1_emb"].to(device)
    baseline_embs_t1c   = sample["baseline_T1C_emb"].to(device)
    baseline_embs_flair = sample["baseline_FLAIR_emb"].to(device)

    

    baseline_emb = torch.cat([baseline_embs_t1, baseline_embs_t1c, baseline_embs_flair], dim=1)  # (B, 12, h, w, d)

    # Dose and fu_times unchanged
    dose    = sample["dose"].to(device)
    fu_time = sample["fu_time"].to(device).unsqueeze(-1)
    

    baseline_t1    = sample["baseline_T1"].to(device)
    baseline_t1c   = sample["baseline_T1C"].to(device)
    baseline_Flair = sample["baseline_FLAIR"].to(device)


    # Save filename before transform corrupts metadata
    baseline_filename = baseline_t1.meta["filename_or_obj"][0]
    # Load original affine from file before any transform
    baseline_affine = nib.load(baseline_filename).affine

    transformed_baselines = None
    if transform_level is not None:
        baseline_t1    = apply_transform(baseline_t1, transform_level)
        baseline_t1c   = apply_transform(baseline_t1c, transform_level)
        baseline_Flair = apply_transform(baseline_Flair, transform_level)

        # Store transformed baseline arrays keyed by modality name
        transformed_baselines = {
            "T1":    baseline_t1.squeeze().cpu().detach().numpy(),
            "T1C":   baseline_t1c.squeeze().cpu().detach().numpy(),
            "FLAIR": baseline_Flair.squeeze().cpu().detach().numpy(),
        }
    baseline = torch.cat([baseline_t1, baseline_t1c, baseline_Flair], dim=1)
    

    logger.info(f"Baseline: {baseline_filename}")
    logger.info(f"Dose: {dose.meta['filename_or_obj'][0]}")
    
    baseline_input = baseline if args.mode == "concat" else baseline_emb

    noise = torch.randn_like(baseline_emb)
    logger.info(f"noise: {noise.device}, {noise.dtype}, {noise.shape}, {type(noise)}")

    noise_scheduler.set_timesteps(
        num_inference_steps=args.controlnet_infer["num_inference_steps"],
        input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
    )
    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))

    img = noise
    start_time = time.time()

    denoising_start = start_time

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in zip(all_timesteps, all_next_timesteps):
            controlnet_inputs = {
                "x": img,
                "timesteps": torch.Tensor((t,)).to(device),
                "dose": dose,
                "baseline": baseline_input,
                "fu_times": fu_time,
            }
            down_block_res_samples, mid_block_res_sample = controlnet(**controlnet_inputs)

            unet_inputs = {
                "x": img,
                "timesteps": torch.Tensor((t,)).to(device),
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual": mid_block_res_sample,
            }
            model_output = unet(**unet_inputs)

            img, _ = noise_scheduler.step(model_output, t, img, next_t)

        denoising_end = time.time()

        del (
            unet_inputs,
            controlnet_inputs,
            model_output,
            down_block_res_samples,
            mid_block_res_sample,
        )
        gc.collect()
        torch.cuda.empty_cache()

        # Decode embedding
        decoding_start = time.time()

        inferer = SlidingWindowInferer(
            roi_size=args.controlnet_infer["autoencoder_sliding_window_infer_size"],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=args.controlnet_infer["autoencoder_sliding_window_infer_overlap"],
            sw_device=device,
            device=torch.device("cpu"),
        )

        # Split 12-channel latent into 3 modalities (T1, T1c, FLAIR) and decode each
        imgs = torch.chunk(img, 3, dim=1)  # 3 × (B, 4, H, W, D)
        modality_outputs = []
        for img_mod in imgs:
            out = dynamic_infer(inferer, recon_model, img_mod)
            out = ResizeWithPadOrCrop(spatial_size=args.raw_img_dim)(out.squeeze(0))
            out = out.squeeze().cpu().detach().numpy()
            out = np.clip(out, 0, 1)
            modality_outputs.append(out)
        output = modality_outputs  # list of 3 numpy arrays [T1, T1c, FLAIR]

        """
        output = dynamic_infer(inferer, recon_model, img)
        output = ResizeWithPadOrCrop(spatial_size=args.raw_img_dim)(output.squeeze(0))
        output = output.squeeze().cpu().detach().numpy()
        output = np.clip(output, 0, 1)
        """


        end_time = time.time()
        denoising_time = denoising_end - denoising_start
        decoding_time = end_time - decoding_start
        total_time = end_time - start_time
        logger.info(f"Inference: total={total_time:.2f}s | denoising={denoising_time:.2f}s | decoding={decoding_time:.2f}s")

        torch.cuda.empty_cache()
    
    # Use saved path to load the affine matrix from the original NIfTI file.
    baseline_affine = nib.load(baseline_filename).affine

    #return output, baseline_affine
    return output, baseline_affine, transformed_baselines



def controlnet_infer_dataset(
    # Only used for inference, NOT during training
    controlnet: torch.nn.Module,
    unet: torch.nn.Module,
    recon_model: torch.nn.Module,
    dataloader: DataLoader,
    noise_scheduler: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    args: Namespace,
    transform_level: int = None,
    max_follow_ups: int = None,
) -> None:
    controlnet.eval()
    unet.eval()
    recon_model.eval()

    logger = setup_logging("controlnet_infer")
    """
    for idx, sample in enumerate(dataloader):
        logger.info(f"Processing sample {idx + 1}/{len(dataloader)}")

        output, affine, transformed_baseline = controlnet_infer(
            sample=sample,
            controlnet=controlnet,
            unet=unet,
            recon_model=recon_model,
            noise_scheduler=noise_scheduler,
            device=device,
            args=args,
            logger=logger,
            transform_level=transform_level,
        )


        modality_names = ["T1", "T1C", "FLAIR"]
        for mod_output, mod_name in zip(output, modality_names):
            output_path = get_pred_path(sample, output_dir, mod_name)
            output_path = output_path.parent / f"{mod_name}.nii.gz"
            save_image(mod_output, str(output_path), affine)
            logger.info(f"Saved output to {output_path}")

            # Save transformed baseline if transform was applied
            if transformed_affine is not None:
                transformed_path = get_pred_path(sample, output_dir / "baseline_transformed", mod_name)
                transformed_path = transformed_path.parent / f"{mod_name}.nii.gz"
                transformed_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(mod_output, str(transformed_path), transformed_affine)
                logger.info(f"Saved transformed baseline to {transformed_path}")
        """
    for idx, sample in enumerate(dataloader):
        logger.info(f"Processing sample {idx + 1}/{len(dataloader)}")

        output, baseline_affine, transformed_baselines = controlnet_infer(
            sample=sample,
            controlnet=controlnet,
            unet=unet,
            recon_model=recon_model,
            noise_scheduler=noise_scheduler,
            device=device,
            args=args,
            logger=logger,
            transform_level=transform_level,
        )
        
        modality_names = ["T1", "T1C", "FLAIR"]
        for mod_output, mod_name in zip(output, modality_names):
            output_path = get_pred_path(sample, output_dir, mod_name)
            output_path = output_path.parent / f"{mod_name}.nii.gz"
            save_image(mod_output, str(output_path), baseline_affine)
            logger.info(f"Saved output to {output_path}")



            if transformed_baselines is not None:
                transformed_path = get_pred_path(sample, output_dir / "baseline_transformed", mod_name)
                transformed_path = transformed_path.parent / f"{mod_name}.nii.gz"
                transformed_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(transformed_baselines[mod_name], str(transformed_path), baseline_affine)
                logger.info(f"Saved transformed baseline to {transformed_path}")


def main(args: Namespace, rank: int, device: torch.device) -> None:
    """
    Main function to run ControlNet inference.

    Args:
        args: Configuration arguments.
        rank: Process rank.
        device: Device to run inference on.
    """
    logger = setup_logging("controlnet_infer")
    random_seed = args.controlnet_infer["random_seed"]
    set_determinism(random_seed)

    if args.prev_run_dir:
        run_dir = Path(args.prev_run_dir)
        logger.info(f"Using existing run directory: {run_dir}")
    else:
        run_dir = Path(args.output_dir) / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            save_config(args, run_dir)

    _, _, test_dataloader = get_controlnet_dataloaders(
        json_datalist_path=args.json_datalist_path,
        args=args,
        rank=rank,
        logger=logger,
    )
    

    if not args.prev_run_dir:
        args.controlnet_def["conditioning_embedding_in_channels"] = 1 if args.mode == "embed" else 4

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

        autoencoder = load_autoencoder(
            trained_autoencoder_path=args.trained_autoencoder_path,
            model_def_args=args,
            device=device,
        )

        recon_model = ReconModel(autoencoder).to(device)
        noise_scheduler = define_instance(args, "noise_scheduler")
        subject_counts = defaultdict(int)
        for sample in test_dataloader:
            subject = Path(sample["baseline_T1"].meta["filename_or_obj"][0]).parts[-4]
            subject_counts[subject] += 1
        print(f"Total subjects: {len(subject_counts)}")
        print(f"Total samples: {sum(subject_counts.values())}")
        #print(f"Follow-ups per subject: {dict(subject_counts)}")

        filtered_dataloader = get_filtered_dataloader(
            dataloader=test_dataloader,
            max_follow_ups=getattr(args, "max_follow_ups", None),
            num_subjects=getattr(args, "num_subjects", None),  # cap on number of unique subjects
            )

        print("=== AFTER FILTERING === ")
        print(f"Filtered subjects: {len(set(Path(s['baseline_T1'].meta['filename_or_obj'][0]).parts[-4] for s in filtered_dataloader))}")
        print(f"Filtered samples: {len(filtered_dataloader.dataset)}")
        # Loop over transform levels: None = original, 1,2 = increasing corruption

        for level in [None, 1, 2]:
            level_name = f"level_{level}" if level is not None else "original"
            logger.info(f"Running inference with transform level: {level_name}")
            level_dir = run_dir / "predictions" / level_name
            level_dir.mkdir(parents=True, exist_ok=True)

            controlnet_infer_dataset(
                controlnet=controlnet,
                unet=unet,
                recon_model=recon_model,
                dataloader=filtered_dataloader,
                noise_scheduler=noise_scheduler,
                output_dir=level_dir,
                device=device,
                args=args,
                transform_level=level,
                max_follow_ups=getattr(args, "max_follow_ups", None), 
        )
            
    all_results = {}    
    
    #for level in [None]:
    for level in [None, 1, 2]:
        level_name = f"level_{level}" if level is not None else "original"
        print("====================")
        print(" LEVEL ", level_name)
        level_run_dir = run_dir / "predictions" / level_name

        print("Requested metrics:", args.controlnet_infer["test_metrics"])
        filtered_dataloader.dataset.data = [test_dataloader.dataset.data[i] for i in filtered_dataloader.dataset.indices]
    
        results_df = evaluate_predictions(
            dataloader=filtered_dataloader,
            run_dir=str(level_run_dir),
            metrics=args.controlnet_infer["test_metrics"],
            prev_run_dir=args.prev_run_dir,
            args=args,
            logger=logger,
            mode = "test"
            )
       
        all_results[level_name] = results_df

        if rank == 0:
            save_metrics(results_df, run_dir / f"metrics_{level_name}.csv")


            logger.info(f"Metrics saved to {run_dir / 'metrics.csv'}")

            
            plots_dir = run_dir / "plots" / level_name
            plots_dir.mkdir(exist_ok=True, parents=True)
            plot_all_metrics_summary(results_df, str(plots_dir))
            if 'fu_time' in results_df.columns:
                plot_dice_by_time(results_df, str(plots_dir), level=level_name)
                plot_image_quality_by_time(results_df, str(plots_dir), level=level_name)
                #plot_hippo_vol_by_time(results_df, str(plots_dir))
                #plot_cwm_by_time(results_df, str(plots_dir))
                plot_volumes_by_time(results_df, str(plots_dir), level=level_name)

            plot_predictions(
                dataloader=filtered_dataloader,
                run_dir=str(run_dir),
                args=args,
                mode = "test",
                logger=logger,
            )

            plot_longitudinal_predictions(
                #dataloader=test_dataloader,
                dataloader=filtered_dataloader,
                run_dir=str(run_dir),
                args=args,
                logger=logger,
            )
        
    if rank == 0:
   
        combined_plots_dir = run_dir / "plots" / "combined"
        combined_plots_dir.mkdir(exist_ok=True, parents=True)

        dice_metrics = [m for m in next(iter(all_results.values())).columns
                if 'dice' in m and 'mean' not in m and 'std' not in m]
        quality_metrics = [m for m in next(iter(all_results.values())).columns
                    if any(q in m for q in ['mse', 'psnr', 'ssim'])]
        volume_metrics = [m for m in next(iter(all_results.values())).columns
            if 'volume' in m and 'mean' not in m and 'std' not in m]

        if dice_metrics:
            plot_metrics_by_time_multilevel(
                all_results, dice_metrics,
                output_path=str(combined_plots_dir / "dice_by_time_combined.png"),
                title="Dice Coefficient by Follow-up Time",
            )
        if quality_metrics:
            plot_metrics_by_time_multilevel(
                all_results, quality_metrics,
                output_path=str(combined_plots_dir / "image_quality_by_time_combined.png"),
                title="Image Quality Metrics by Follow-up Time",
            )
        if volume_metrics:
            plot_metrics_by_time_multilevel(
            all_results, volume_metrics,
            output_path=str(combined_plots_dir / "volume_by_time_combined.png"),
            title="Volume Metrics by Follow-up Time",
        )
        if all(k in all_results for k in ['original', 'level_1', 'level_2']):
            plot_volumes_combined(all_results, str(combined_plots_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ControlNet Inference")
    parser.add_argument("--env_config", type=str, default="./configs/env_config_controlnet_infer.json")
    parser.add_argument("--model_config", type=str, default="./configs/model_config_controlnet_train.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_maisi3d-rflow.json")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--prev_run_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, required=True, choices=["embed", "concat"])
    parser.add_argument("--max_follow_ups", type=int, default=None) # maximum number of follow ups 
    parser.add_argument("--num_subjects", type=int, default=None)  # maximum number of unique subjects
    parser.add_argument("--run_name", type=str)
    args = parser.parse_args()
    
    load_configs(args, [args.env_config, args.model_config, args.model_def])
    args.run_name = add_timestamp(args.run_name)
    
    run_dist(main, args)