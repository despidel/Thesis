"""Metrics input computation module for ControlNet evaluation.

This module provides functionality to:
1. Build a datalist with all paths needed for metric computation
2. Compute metric inputs (segmentations, jacobians) using external tools
"""

from logging import Logger

from .metrics import Metric


def get_eval_datalist(
# ADDED mode = train, eval 
# if mode = train : get_paths -> run_dir/sub-10/ses-02/T1.nii.gz
# if mode = test  : get_paths -> run_dir/predictions/original/sub-10/ses-02/T1.nii.gz
    datalist: list[dict],
    data_base_dir: str,
    run_dir: str,
    metric_names: list[str],
    mode: str
) -> list[dict]:
    """
    Returns list of dicts where each dict has format:
        {
    "SSIM": {
        "T1":    {"pred_and_data_paths": {...}, "metric_input_paths": {...}},
        "T1c":   {"pred_and_data_paths": {...}, "metric_input_paths": {...}},
        "FLAIR": {"pred_and_data_paths": {...}, "metric_input_paths": {...}},
    },
    "fu_time": 6.0
    }
    """
    metrics = [Metric.get(name)() for name in metric_names]
    metrics_datalist = []
    
    
    for sample in datalist:
        new_sample = {}
        
        modality_names = ["T1", "T1C", "FLAIR"]

        for metric in metrics:
            new_sample[metric.name] = {
                mod_name: metric.get_paths(sample, run_dir, data_base_dir, mod_name, mode)
                #mod_name: metric.get_paths(sample, run_dir, data_base_dir, mod_name.upper() if mod_name == "Flair" else mod_name)
                for mod_name in modality_names
            }
        # Preserve fu_time for time-based analysis
        new_sample["fu_time"] = sample.get("fu_time")
        metrics_datalist.append(new_sample)
    
        

    return metrics_datalist


def compute_metrics_inputs(
    datalist: list[dict],
    metric_names: list[str],
    logger: Logger,
) -> None:
    """Compute metric inputs (segmentations, jacobians, etc.) for all samples."""
    metrics = [Metric.get(name)() for name in metric_names]

    for sample in datalist:
        for metric in metrics:
            for mod_name, paths in sample[metric.name].items():
                metric.compute_inputs(
                    paths["pred_and_data_paths"],
                    paths["metric_input_paths"],
                    logger,
                )