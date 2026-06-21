"""ControlNet evaluation module.

This module provides the main evaluation pipeline for ControlNet predictions,
supporting both validation (during training) and testing workflows.
"""

from logging import Logger
from pathlib import Path

import pandas as pd
import torch.distributed as dist
import numpy as np
from typing import Optional

from .metrics import Metric
from .metrics_inputs import get_eval_datalist, compute_metrics_inputs
from .utils import load_data, gather_results, flatten_metric_input_paths, get_transforms_2d, get_transforms_3d


def get_metrics_means(results_df: pd.DataFrame) -> dict[str, float]:
    """
    Returns e.g. `{'mse': 0.001, 'psnr': 0.9}`, corresponding to the mean row
    (which is the second to last row of `results_df`)
    """
    mean_row = results_df.iloc[-2]  # Second to last row is mean
    return {col: mean_row[col] for col in mean_row.index if col != "pred_path"}


def save_metrics(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save metrics DataFrame to csv"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, float_format="%.4f")


def evaluate_predictions(
# mode = train / eval
    dataloader,
    run_dir: str,
    metrics: list[str],
    prev_run_dir: Optional[str],
    args,
    logger: Logger,
    mode: str
) -> pd.DataFrame:
    """Run the full evaluation pipeline with distributed metrics computation.
    
    Args:
        dataloader: DataLoader with dataset.data as the datalist (already partitioned).
        run_dir: Run directory containing predictions.
        metrics: List of metric names to compute.
        prev_run_dir: If provided, skip computing metric inputs (use existing).
        args: Namespace with data_base_dir.
        logger: Logger instance.
        
    Returns:
        DataFrame with computed metrics, including mean and std rows.
    """
    eval_datalist = get_eval_datalist(
        datalist=dataloader.dataset.data,
        data_base_dir=args.data_base_dir,
        run_dir=run_dir if not prev_run_dir else prev_run_dir,
        metric_names=metrics,
        mode = mode
    )
    
    if not prev_run_dir:
        compute_metrics_inputs(eval_datalist, metrics, logger)
        
        if dist.is_initialized():
            dist.barrier()
    
    logger.info(f"Computing metrics for {len(eval_datalist)} samples")
    results = compute_metrics(eval_datalist, metrics, args, logger)
    
    if dist.is_initialized():
        results = gather_results(results)
    
    results = add_mean_std_rows(results)
    
    return results


def compute_metrics(
    datalist: list[dict],
    metric_names: list[str],
    args,
    logger: Logger,
) -> list[dict]:
    """Compute metrics for all samples using registered metrics.
    
    Loads all required files once per sample, then passes loaded data to each metric.
    """
    metrics = [Metric.get(name)() for name in metric_names]
    
    transforms = {
        "3d": get_transforms_3d(args),
        "2d": get_transforms_2d(args),
    }
   
    
    results = []
    for idx, sample in enumerate(datalist):
        logger.info(f"Computing metrics for sample {idx + 1}/{len(datalist)}")
    
        for mod_name in ["T1", "T1C", "FLAIR"]:
            paths = flatten_metric_input_paths(sample, metrics, mod_name)
            data = load_data(paths, args)
            
            result = {"pred_path": paths.get("pred", list(paths.values())[0]), "modality": mod_name}
        
            if "fu_time" in sample:
                result["fu_time"] = sample["fu_time"]
        
            for metric in metrics:
                result.update(metric.compute(data, transforms))
        
            results.append(result)
    return results

def add_mean_std_rows(results: list[dict]) -> pd.DataFrame:
    """
    Add mean and std per modality (Add rows)
    """
    rows = list(results)
    modalities = ["T1", "T1C", "FLAIR"]

    for mod in modalities:
        mod_results = [r for r in results if r.get("modality") == mod]
        if not mod_results:
            continue

        mean_dict = {"pred_path": f"mean_{mod}", "modality": mod}
        std_dict  = {"pred_path": f"std_{mod}",  "modality": mod}

        for key in mod_results[0].keys():
            if key not in ["pred_path", "modality"]:
                vals = [r[key] for r in mod_results if key in r and r[key] is not None]
                if vals:
                    try:
                        mean_dict[key] = np.mean(vals)
                        std_dict[key]  = np.std(vals)
                    except TypeError:
                        pass

        rows += [mean_dict, std_dict]

    return pd.DataFrame(rows)


