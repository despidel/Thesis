"""Utility to configure data and output paths in environment configs.

Usage example:

python -m data.configure_paths \
    --data_base_dir <data_base_dir> \
    --output_dir <output_dir> \
    --img_filename T1.nii.gz \
    --brain_mask_filename BrainExtractionMask.nii.gz \
    --roi_mask_filename ContrastEnhancedMask-ONCO.nii.gz \
    --dose_filename DoseMap.nii.gz
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

# Default environment config files
ENV_CONFIG_PATHS = [
    "./configs/env_config_controlnet_infer.json",
    "./configs/env_config_controlnet_train.json",
    "./configs/env_config_create_emb.json",
    "./configs/env_config_datalist.json",
    "./configs/env_config_diff_model_infer.json",
    "./configs/env_config_diff_model_train.json",
]


def configure_paths(
    data_base_dir: str,
    output_dir: str,
    img_filename: dict,
    brain_mask_filename: str,
    roi_mask_filename: str,
    dose_filename: str,
    #env_config_paths: list[str] | None = None,
    env_config_paths: Optional[List[str]] = None,
) -> None:
    """
    Update environment config files with new data and output paths.
    
    Args:
        data_base_dir: Base directory where the dataset is stored.
        output_dir: Directory where runs should be saved (for future use).
        img_filename: Name of the image file.
        brain_mask_filename: Name of the brain mask file.
        roi_mask_filename: Name of the ROI mask file.
        dose_filename: Name of the dose file.
        env_config_paths: List of environment config file paths to update.
                         Defaults to all environment configs in configs/.
    """
    if env_config_paths is None:
        env_config_paths = ENV_CONFIG_PATHS

    config_vars = {
        "data_base_dir": Path(data_base_dir).resolve(),
        "embedding_base_dir": Path(data_base_dir).parent / f"{Path(data_base_dir).name}_emb",
        "output_dir": Path(output_dir).resolve(),
        "img_filename": img_filename,
        "brain_mask_filename": brain_mask_filename,
        "roi_mask_filename": roi_mask_filename,
        "dose_filename": dose_filename,
    }
    
    for config_path in env_config_paths:
        if not Path(config_path).exists():
            print(f"Warning: Config file not found: {config_path}")
            continue
        
        with open(config_path, "r") as f:
            original_config = json.load(f)
        
        config = original_config.copy()
        
        for k, v in config_vars.items():
            if k in config:
                #config[k] = str(v)
                config[k] = v if isinstance(v, dict) else str(v)

        """
        if "json_datalist_path" in config and config["json_datalist_path"]:
            old_path = Path(config["json_datalist_path"])
            config["json_datalist_path"] = str(config_vars["data_base_dir"] / old_path.name)
        """
        if "json_datalist_path" in config:
            current_val = str(config["json_datalist_path"])
            if "\\" in current_val:
                filename = current_val.split("\\")[-1]
            elif "/" in current_val:
                filename = current_val.split("/")[-1]
            else:
                filename = current_val if current_val else "datalist_controlnet.json"
            config["json_datalist_path"] = str(Path(config_vars["data_base_dir"]) / filename)
        
        if original_config != config:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Updated config file: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure data and output paths")
    parser.add_argument("--data_base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_filename", type=str, required=True)
    parser.add_argument("--brain_mask_filename", type=str, required=True)
    parser.add_argument("--roi_mask_filename", type=str, required=True)
    parser.add_argument("--dose_filename", type=str, required=True)
    parser.add_argument("--env_config_paths", type=str, nargs="*", default=None)
    args = parser.parse_args()

    configure_paths(
        data_base_dir=args.data_base_dir,
        output_dir=args.output_dir,
        env_config_paths=args.env_config_paths,
        img_filename=args.img_filename,
        brain_mask_filename=args.brain_mask_filename,
        roi_mask_filename=args.roi_mask_filename,
        dose_filename=args.dose_filename,
    )
