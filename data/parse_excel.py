"""Parse Excel file with data information

This script processes Excel file and store information for each subject and session. 

Usage:
    python -m data.parse_excel --data_base_dir /path/to/data --source_excel /path/to/excelfile.xlsx --quality_excel /path/to/exxcelfile_image_quality_info.csv


"""
import argparse
import json
import numpy as np
from pathlib import Path
import logging
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import random

FIRST_FU = 2
LAST_FU = 30


def require_exists(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not path.exists():
            raise ValueError(f"File not found: {path}")

def check_path(path):
    if path.exists():
        return str(path)
    print(f"Warning: File not found: {path}")
    return None


MODALITIES = ["T1", "T1C", "FLAIR"]
counter_2d = {"T1": 0, "T1C": 0, "FLAIR": 0}


def get_filenames_for_session(session_dir):
    """
    For each modality, prefer 3D over 2D.
    Returns dict {mod: Path} or None if any modality is missing.
    """
    mod_filenames = {
        "T1":    ["3D_T1.nii.gz",    "2D_T1.nii.gz"],
        "T1C":   ["3D_T1C.nii.gz",   "2D_T1C.nii.gz"],
        "FLAIR": ["3DFLAIR.nii.gz",  "2DFLAIR.nii.gz"]
    }
    result = {}
    used_2d = []  # track which modalities fell back to 2D
    for mod, candidates in mod_filenames.items():
        found = None
        for fname in candidates:  # 3D checked first
            fpath = session_dir / "anat" / fname
            if fpath.exists():
                found = fpath
                if "2D" in fname:
                    used_2d.append(mod)  # flag this modality as 2D fallback
                    counter_2d[mod] += 1 
                break
        if found is None:
            return None  # missing modality → exclude session
        result[mod] = found

    return result

def get_dose_for_subject(subject_dir):
    
    #Returns Path or None if not found.
    
    dose_filename = "Dose.nii.gz"

    matches = list(subject_dir.glob(f"ses-*_RT/other/{dose_filename}"))

    if not matches:
        return None  # not found → exclude subject

    return matches[0]  # return first match

def get_session_dirs_and_fu_times(
    subject_id: str,
    row,
    all_session_dirs: list,
    FIRST_FU: int,
    LAST_FU: int,
    df_columns,
) -> tuple[list, list, bool]:
    """
    Returns (session_dirs, fu_times, skip_subject).
    session_dirs[0] = baseline folder, session_dirs[1:] = follow-up folders.
    skip_subject=True means this subject should be skipped.
    """
    session_dirs = []
    
    fu_times = [0]  # baseline has fu_time = 0

   
    # If this first follow-up MRI column is missing or empty, there are no follow-ups → skip subject
    mr2_col = f"MR{FIRST_FU}"
    mr2_val = row.get(mr2_col, None)

    if mr2_col not in df_columns or pd.isna(mr2_val) or str(mr2_val).strip() == "":
        print(f"Skipping {subject_id}: No follow-up MRI available.")
        return [], [], True

    fu_cols = []
    for i in range(FIRST_FU, LAST_FU):
        col = f"MR{i}"
        if col not in df_columns:
            break

        val = row.get(col, None)

        if pd.isna(val) or str(val).strip() == "":
            break

        # Store (column name, follow-up time in days)
        fu_cols.append((col, int(val)))

    no_of_fu   = len(fu_cols)       # number of follow-up MRI columns found
    timepoints = int(row["timepoints"]) 
    no_folders_in_disk = len(all_session_dirs) # number of folders in disk (baseline + follow ups)

    # Determine which folder is the baseline
    # Expected: timepoints = no_of_fu + 1 (1 baseline + N follow-ups)
    # If there is 1 extra folder (timepoints = no_of_fu + 2), skip the earliest and use the second earliest as baseline

    if len(all_session_dirs) == no_of_fu + 1:
        baseline_folder_idx = 0

    elif len(all_session_dirs) == no_of_fu + 2:
        baseline_folder_idx = 1
        print(
            f"{subject_id}: timepoints == no_of_fu + 2 "
            f"({timepoints} folders, {no_of_fu} FU MR columns). "
            f"Using second earliest folder as baseline: {all_session_dirs[baseline_folder_idx].name}"
        )

    else :
        # Mismatch between folders on disk and MRI columns in Excel → skip subject
        print(
            f"Skipping {subject_id}: mismatch between folders and FU MR columns "
            f"({no_folders_in_disk} folders, {no_of_fu} FU MR columns)."
        )
        return [], [], True

    #Add baseline folder as first session 
    session_dirs.append(all_session_dirs[baseline_folder_idx])

    # Add follow-up session folders in order
    # Start from the folder immediately after baseline
    fu_folder_idx = baseline_folder_idx + 1

    for col, val in fu_cols:
        if fu_folder_idx >= len(all_session_dirs):
            print(f"Warning {subject_id}: more MRI columns than folders, stopping.")
            break

        session_dirs.append(all_session_dirs[fu_folder_idx])
        fu_times.append(val)
        fu_folder_idx += 1

    return session_dirs, fu_times, False


def parse_excel_and_build_patient_data(data_base_dir, source_excel, quality_excel, seed=42, logger=None):
    """
    Parse Excel to extract valid subjects and their sessions/fu_times.
    Builds patient_data dict and updates global splits (70-15-15).

    Excel structure:
        - First column ("sub"): subject ID, format IMXXXX
        - Columns MRI1..MRI23: integer days from treatment (can be negative)

    Logic:
        - Find first follow-up MRI ses-02
        - Find baseline
        - fu_times are raw Excel values (zero for baseline, positive for follow-ups)
        - Exclude subject if:
             No follow up available
             Fewer than 2 valid sessions (baseline and 1 follow-up) after modality check

    Returns:
        patient_data: dict[split][patient_id] = {
            "session_dirs": list of Path (renamed ses-01, ses-02, ...),
            "dose_dir":     path for dose map
            "fu_times":     list of int (raw Excel values),
            "no_of_fracts": number of fractions for radiotherapy
            "exclude":      list of bool (all False, excluded subjects are dropped),
            "patient_dir":  Path,
        }
    """
    global splits

    # Load source and quality Excel/CSV files 
    print(f"Reading Excel from: {source_excel}")
    print(f"File exists: {Path(source_excel).exists()}")
    print(f"File size: {Path(source_excel).stat().st_size} bytes")
    print(f"File for Quality check exists: {Path(quality_excel).exists()}")

    df         = pd.read_excel(source_excel, engine="openpyxl")
    quality_df = pd.read_csv(quality_excel)

    # Build set of "Good" file paths from quality CSV
    good_subjects = set(
        quality_df[quality_df["Timestamp"] == "Good"]["File Path"].astype(str).str.strip()
    )

    #  Collect all follow-up MRI column names present in source Excel 
    mri_cols = [f"MR{i}" for i in range(FIRST_FU, LAST_FU) if f"MR{i}" in df.columns]

    valid_subjects = []

    # Iterate over each row (subject) in the source Excel 
    for _, row in df.iterrows():
        subject_id = str(row["sub"]).strip()

        # Skip rows that are not subject IDs (expected format: sub-IMXXXX)
        if not subject_id.startswith("sub-IM"):
            continue

        # Skip if subject directory does not exist on disk
        patient_dir = Path(data_base_dir) / subject_id
        if not patient_dir.exists():
            print(f"Skipping {subject_id}: patient directory not found.")
            continue

        # Get all session folders sorted chronologically (exclude RT folders)
        all_session_dirs = sorted(
            [d for d in patient_dir.glob("ses-*") if d.is_dir() and not d.name.endswith("_RT")],
            key=lambda d: d.name  # ses-YYYYMMDD sorts correctly as string
        )
        if not all_session_dirs:
            print(f"Skipping {subject_id}: no session folders found.")
            continue

        # Determine baseline and follow-up session folders from Excel MR columns
        session_dirs, fu_times, skip_subject = get_session_dirs_and_fu_times(
            subject_id, row, all_session_dirs, FIRST_FU, LAST_FU, mri_cols
        )

        if skip_subject:
            continue

        # Validate each session: must have all modalities + pass quality check 
        valid_session_dirs = []
        valid_fu_times     = []

        for s_dir, ft in zip(session_dirs, fu_times):

            # Skip session if any required modality file is missing
            mod_files = get_filenames_for_session(s_dir)
            if mod_files is None:
                print(f"  Skipping session {s_dir.name} for {subject_id}: missing modality.")
                continue

            
            session_name = s_dir.name  # e.g. "20220613"
       
            subject_id_stripped = subject_id.replace("sub-", "")
            session_name_stripped = s_dir.name.replace("ses-", "")

            if not any(subject_id_stripped in gs and session_name_stripped in gs for gs in good_subjects):
                print(f"  Skipping session {s_dir.name} for {subject_id}: not marked as Good.")
                continue

            valid_session_dirs.append(s_dir)
            valid_fu_times.append(ft)

        # Need at least baseline + 1 follow-up to be useful
        if len(valid_session_dirs) < 2:
            print(f"Skipping {subject_id}: fewer than 2 valid sessions after modality check.")
            continue

        # Get dose file path (filesystem lookup, independent of Excel) 
        dose_dir = get_dose_for_subject(patient_dir)
        if dose_dir is None:
            print(f"Skipping {subject_id}: dose file not found.")
            continue

        # Get number of RT fractions from Excel 
        no_of_fracts = int(row["Number_of_fractions"])

        # Store all valid subject data for later splitting
        valid_subjects.append({
            "patient_id":   subject_id,
            "patient_dir":  patient_dir,
            "session_dirs": valid_session_dirs,
            "fu_times":     valid_fu_times,
            "dose_dir":     dose_dir,
            "no_of_fracts": no_of_fracts,
        })

    # Summary stats 
    N  = len(valid_subjects)
    Ns = sum(len(s["session_dirs"]) for s in valid_subjects)
    Ni = 3 * Ns  # 3 modalities per session

    if N == 0:
        raise ValueError("No valid subjects found.")


    print(f"Number of subjects: {N}")
    print(f"Number of sessions: {Ns}")
    print(f"Number of images:   {Ni}")


    # Split into train / val / test (70 - 15 - 15) 
    random.seed(seed)
    indices = list(range(N))
    random.shuffle(indices)

    val_size   = int(N * 0.15)
    test_size  = int(N * 0.15)
    val_indices   = indices[:val_size]
    test_indices  = indices[val_size:val_size + test_size]
    train_indices = indices[val_size + test_size:]

    def build_split(idx_list):
        """Build patient_data dict for a given list of subject indices."""
        split_data = {}
        for idx in idx_list:
            s   = valid_subjects[idx]
            pid = s["patient_id"]
            split_data[pid] = {
                "session_dirs":  s["session_dirs"],
                "fu_times":      s["fu_times"],
                "exclude":       [False] * len(s["session_dirs"]),  
                "patient_dir":   s["patient_dir"],
                "dose_dir":      s["dose_dir"],
                "no_of_fracts":  s["no_of_fracts"],
            }
        return split_data

    patient_data = {
        "train": build_split(train_indices),
        "val":   build_split(val_indices),
        "test":  build_split(test_indices),
    }

    # ── Update global splits (used elsewhere in the pipeline) ────────────────
    splits = {
        "train": [valid_subjects[i]["patient_id"] for i in train_indices],
        "val":   [valid_subjects[i]["patient_id"] for i in val_indices],
        "test":  [valid_subjects[i]["patient_id"] for i in test_indices],
    }

    logger.info(
        f"Split summary: {N} subjects → "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    return patient_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_base_dir",   required=True)
    parser.add_argument("--source_excel",    required=True)
    parser.add_argument("--quality_excel",   required=True)
    parser.add_argument("--output_filename", default="dataset_split", help="Output pickle filename (without extension)")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    patient_data = parse_excel_and_build_patient_data(
        data_base_dir=args.data_base_dir,
        source_excel=args.source_excel,
        quality_excel=args.quality_excel,
        seed=args.seed,
        logger=logger,
    )
    print(f"\n2D fallback summary: {counter_2d}")

    # Save as pickle (preserves Path objects)
    output_path = Path(args.output_filename).with_suffix(".pkl")
    with open(output_path, "wb") as f:
        pickle.dump(patient_data, f)

    logger.info(f"Saved patient_data to {output_path}")

if __name__ == "__main__":
    main()