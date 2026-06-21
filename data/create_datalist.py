"""Create datalist JSON files for training.

This script creates datalist JSON files for:
- Unconditional diffusion model training (all sessions in train split)
- Conditional Baseline ControlNet training (baseline -> follow-up pairs) 
- Conditional Domain Randomisation ControlNet training (baseline -> follow-up pairs, tranformed baseline -> follow-up pairs)

Usage:
    python -m data.create_datalist --mode unconditional --output_filename datalist_diff_model.json
    python -m data.create_datalist --mode conditional --output_filename datalist_controlnet.json
    python -m data.create_datalist --mode domainRand --output_filename datalist_domainRand.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
#from logging import Logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.diff_model_setting import load_configs
from data.Transforms import domain_rand_transform
from data.parse_excel import parse_excel_and_build_patient_data, get_filenames_for_session, require_exists

import pandas as pd
import numpy as np
from pathlib import Path
import random
import pickle

"""
ENV_CONFIG_PATHS = {
    "conditional": [
        "./configs/env_config_controlnet_train.json",
        "./configs/env_config_controlnet_infer.json",
    ],
    "unconditional": [
        "./configs/env_config_diff_model_train.json",
        "./configs/env_config_diff_model_infer.json",
        "./configs/env_config_create_emb.json",
    ],
}
"""
ENV_CONFIG_PATHS = {
    "conditional": [
        "./configs/env_config_controlnet_train.json",
        "./configs/env_config_controlnet_infer.json",
    ],
    "unconditional": [
        "./configs/env_config_diff_model_train.json",
        "./configs/env_config_diff_model_infer.json",
        "./configs/env_config_create_emb.json",
    ],
    "domainRand": [
        "./configs/env_config_controlnet_train.json",
        "./configs/env_config_controlnet_infer.json",
        "./configs/env_config_create_emb.json",
    ],
}

def require_exists(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not path.exists():
            raise ValueError(f"File not found: {path}")

def update_config(config_path: str, json_datalist_path: str, config_key: str = "json_datalist_path") -> None:
    with open(config_path, "r") as f:
        config = json.load(f)

    config[config_key] = json_datalist_path

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Updated {config_key} in config file: {config_path}")
    


def create_unconditional_datalist(patient_data, args):
    """Create unconditional datalist (diff_model datalist): all sessions merged into train split.
    
    For unconditional generation, we don't need val/test splits - all sessions are used for training.
    
    Returns:
        tuple: (datalist, breakdown) where breakdown maps split -> patient_id -> list of session names
    """
    datalist = {"train": []}
    breakdown = {"train": {}}
    
    for split, patients in patient_data.items():
        for patient_id, patient_info in patients.items():
            session_dirs = patient_info["session_dirs"]
            exclude = patient_info["exclude"]
            
            included_sessions = []
            for session_dir, should_exclude in zip(session_dirs, exclude):
                if should_exclude:
                    continue
                
                included_sessions.append(session_dir.name)
                    
                mod_files = get_filenames_for_session(session_dir)
                if mod_files is None:
                    print(f"Skipping session {session_dir.name}: missing modality.")
                    continue

                img_paths = {mod: session_dir / "anat" / filename for mod, filename in mod_files.items()}
               
                require_exists([*img_paths.values()])
                
                
                # create embedding path for each modality
                emb_paths = {
                    mod: Path(str(p).replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", "_emb.nii.gz"))
                    for mod, p in img_paths.items()
                    }

                datalist["train"].append({
                    # image paths per modality
                    **{mod: str(p) for mod, p in img_paths.items()},
                    # embedding paths per modality
                    **{f"{mod}_emb": str(p) for mod, p in emb_paths.items()},
                    })
            
            if included_sessions:
                breakdown["train"][patient_id] = included_sessions
        
    return datalist, breakdown


def create_conditional_datalist(patient_data, args):
    """Create conditional datalist for controlnet training.
    
    Args:
        patient_data: dict from get_patient_data_by_split
        args: namespace with img_filename, data_base_dir, embedding_base_dir, 
              dose_filename
    
    Returns:
        tuple: (datalist, breakdown) where:
            - datalist has keys "train", "val", "test", each containing sample dicts
            - breakdown maps split -> patient_id -> list of session names
    """
    datalist = {"train": [], "val": [], "test": []}
    breakdown = {"train": {}, "val": {}, "test": {}}
    
    for split, patients in patient_data.items():
       
        for patient_id, patient_info in patients.items():
            session_dirs = patient_info["session_dirs"]
            fu_times = patient_info["fu_times"]
            exclude = patient_info["exclude"]
            no_of_fracts = patient_info["no_of_fracts"]
            patient_dir = patient_info["patient_dir"]

            if exclude[0]:
                continue
            
            dose_path = patient_info["dose_dir"]
            #mod_files = get_filenames_for_session(session_dirs)
            #baseline_paths = {mod: session_dirs[0] / filename for mod, filename in args.img_filename.items()} # multiple paths
            baseline_paths = get_filenames_for_session(session_dirs[0])
           
            require_exists([dose_path, *baseline_paths.values()])
            
            # Create path for baseline embeddings
            baseline_emb_paths = {
                mod: Path(str(p).replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", "_emb.nii.gz"))
                for mod, p in baseline_paths.items()
                }
            
            included_sessions = []
            for session_dir, fu_time, should_exclude in zip(session_dirs[1:], fu_times[1:], exclude[1:]):
                if should_exclude:
                    continue
                
                included_sessions.append(session_dir.name)
                    
               
                #fu_paths = {mod: session_dir / filename for mod, filename in args.img_filename.items()}
                #fu_paths = {mod: session_dir / "anat" / filename for mod, filename in mod_files.items()}
                fu_paths = get_filenames_for_session(session_dir)

                # Create path for follow-up embeddings
                fu_emb_paths = {
                     mod: Path(str(p).replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", "_emb.nii.gz"))
                     for mod, p in fu_paths.items()
                     }

                datalist[split].append({
                # follow-up image paths per modality
                **{mod: str(p) for mod, p in fu_paths.items()},
                # follow-up embedding paths per modality
                **{f"{mod}_emb": str(p) for mod, p in fu_emb_paths.items()},
                # baseline paths per modality 
                **{f"baseline_{mod}": str(p) for mod, p in baseline_paths.items()},
                **{f"baseline_{mod}_emb": str(p) for mod, p in baseline_emb_paths.items()},
                "dose":        str(dose_path),
                "fu_time":     fu_time,
                "no_of_fracts": no_of_fracts,
                })


            if included_sessions:
                breakdown[split][patient_id] = included_sessions
        
    
    return datalist, breakdown


def create_domainrand_datalist(patient_data, args):
    """Create datalist for domainRand controlnet training.
    
    Same as create_conditional_datalist but each sample appears twice:
    once with the original baseline and once with the domain-randomized baseline.
    Augmented baselines are generated on-the-fly and saved as <name>_aug.nii.gz.
    Each augmented sample has its own embedding path (<name>_aug_emb.nii.gz).
    All other fields (fu, dose, fu_time, roi_mask, brain_mask) are shared but
    the model treats each sample as fully independent.

    Args:
        patient_data: dict from get_patient_data_by_split
        args: namespace with img_filename, data_base_dir, embedding_base_dir,
              dose_filename, roi_mask_filename, brain_mask_filename

    Returns:
        tuple: (datalist, breakdown)
    """
    datalist = {"train": [], "val": [], "test": []}
    breakdown = {"train": {}, "val": {}, "test": {}}

    for split, patients in patient_data.items():
        for patient_id, patient_info in patients.items():
            session_dirs = patient_info["session_dirs"]
            fu_times     = patient_info["fu_times"]
            exclude      = patient_info["exclude"]
            no_of_fracts = patient_info["no_of_fracts"]
            patient_dir  = patient_info["patient_dir"]

            if exclude[0]:
                continue

            #dose_path      = patient_dir / args.dose_filename
            dose_path = patient_info["dose_dir"]
            #baseline_paths = {mod: session_dirs[0] / filename for mod, filename in args.img_filename.items()}
            baseline_paths = get_filenames_for_session(session_dirs[0])
            require_exists([dose_path, *baseline_paths.values()])
            

            baseline_emb_paths = {
                mod: Path(str(p).replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", "_emb.nii.gz"))
                for mod, p in baseline_paths.items()
            }

            # Generate augmented baseline per modality (on-the-fly, skip if exists)
            aug_baseline_paths = {}
            aug_baseline_emb_paths = {}
            for mod, p in baseline_paths.items():
                aug_path = p.parent / p.name.replace(".nii.gz", "_aug.nii.gz")
                if not aug_path.exists():
                    domain_rand_transform(str(p), str(aug_path)) # Generate augmented image
                aug_baseline_paths[mod] = aug_path
                aug_baseline_emb_paths[mod] = Path(
                    str(aug_path).replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", "_emb.nii.gz")
                )

            included_sessions = []
            for session_dir, fu_time, should_exclude in zip(session_dirs[1:], fu_times[1:], exclude[1:]):
                if should_exclude:
                    continue

                included_sessions.append(session_dir.name)

                #fu_paths = {mod: session_dir / filename for mod, filename in args.img_filename.items()}
                fu_paths = get_filenames_for_session(session_dir)
                logger.info(f"baseline_paths: {baseline_paths}")
                logger.info(f"fu_paths: {fu_paths}")
                fu_emb_paths = {
                    mod: Path(str(p).replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", "_emb.nii.gz"))
                    for mod, p in fu_paths.items()
                }

                # Shared fields (fu, dose, fu_time, roi_mask, brain_mask)
                shared_fields = {
                    **{mod: str(p) for mod, p in fu_paths.items()},
                    **{f"{mod}_emb": str(p) for mod, p in fu_emb_paths.items()},
                    "dose":       str(dose_path),
                    "fu_time":    fu_time,
                    "no_of_fracts": no_of_fracts
                }

                # Original baseline sample
                datalist[split].append({
                    **shared_fields,
                    **{f"baseline_{mod}": str(p) for mod, p in baseline_paths.items()},
                    **{f"baseline_{mod}_emb": str(p) for mod, p in baseline_emb_paths.items()},
                })

                # Augmented baseline sample (fully independent — own image + own embedding)
                datalist[split].append({
                    **shared_fields,
                    **{f"baseline_{mod}": str(p) for mod, p in aug_baseline_paths.items()},
                    **{f"baseline_{mod}_emb": str(p) for mod, p in aug_baseline_emb_paths.items()},
                })

            if included_sessions:
                breakdown[split][patient_id] = included_sessions
        
    return datalist, breakdown




def create_datalist_json(args):
    """
    Creates a datalist JSON for a dataset in the format described in the README.md file.
    """
    #patient_data = parse_excel_and_build_patient_data(args.data_base_dir, args.source_excel, args.quality_excel, logger= logger)
    with open(args.patient_data, "rb") as f:
        patient_data = pickle.load(f)  # restores full Python dict with Path objects intact
    
    if args.mode == "unconditional":
        datalist, breakdown = create_unconditional_datalist(patient_data, args)
    elif args.mode == "conditional":
        datalist, breakdown = create_conditional_datalist(patient_data, args)
    elif args.mode == "domainRand":
        datalist, breakdown = create_domainrand_datalist(patient_data, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    datalist_file_path = Path(args.data_base_dir) / args.output_filename
    with open(datalist_file_path, "w") as f:
        json.dump(datalist, f, indent=4)
    
    print("\nBreakdown by patient:")
    for split, patients in breakdown.items():
        if not patients:
            continue
        print(f"  {split}:")
        for patient_id in sorted(patients.keys()):
            sessions = patients[patient_id]
            session_nums = [int(s.split("-")[1]) for s in sessions]
            print(f"    patient {patient_id}: {len(sessions)} sessions {session_nums}")

    print(f"Datalist JSON file saved at {datalist_file_path}")
    for split, samples in datalist.items():
        print(f"  {split}: {len(samples)} samples")
    

    
    for config_path in ENV_CONFIG_PATHS[args.mode]:
        if args.mode == "domainRand":
            config_key = "json_domainrand_datalist_path"
        else:
            config_key = "json_datalist_path"

        #print(f"DEBUG: Updating {config_path} with {datalist_file_path} (key: {config_key})")
        update_config(config_path, str(datalist_file_path), config_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create datalist JSON for MAISI dataset")
    parser.add_argument("--env_config",       type=str, default="./configs/env_config_datalist.json")
    parser.add_argument("--mode",             type=str, required=True, choices=["unconditional", "conditional", "domainRand"])  
    parser.add_argument("--output_filename",  type=str, required=True)
    parser.add_argument("--patient_data", default="dataset_split.pkl", help="Path to dataset_split.pkl")
    args = parser.parse_args()

    load_configs(args, args.env_config)

    

    create_datalist_json(args)
