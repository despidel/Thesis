from __future__ import annotations
from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import ClassVar
import subprocess
import nibabel as nib
import numpy as np
import torch

from eval.utils import run_apptainer_cmd, run_tumorsynth_cmd

from torchmetrics.functional import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
)
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
from torchmetrics.functional.segmentation import dice_score as dice





def ensure_masked(input_path: str, mask_path: str, output_path: str, logger: Logger) -> None:
    img = nib.load(input_path)
    mask = nib.load(mask_path)
    
    masked_data = img.get_fdata() * (mask.get_fdata() > 0)
    masked_img = nib.Nifti1Image(masked_data.astype(np.float32), img.affine)
    
    nib.save(masked_img, output_path)
    logger.info(f"Saved masked image: {output_path}")

def is_empty(nii_path):
    data = nib.load(nii_path).get_fdata()
    return np.sum(np.isfinite(data) & (data != 0)) == 0

def save_empty_nii(reference_path, output_path):
    """Save an empty (all-zeros) NIfTI with same shape/affine as reference."""
    ref = nib.load(reference_path)
    empty = nib.Nifti1Image(np.zeros(ref.shape, dtype=np.float32), ref.affine, ref.header)
    nib.save(empty, output_path)

def segment_image(key,  seg_dir,input_str, output_path_whole, output_path_inner, logger):
    """
    Use TumorSynth to segment images
    """

    mod_path = input_str.split(",")[0]
    #  Whole tumor

    if is_empty(input_str):
        logger.warning(f"Skipping {key} — empty image: {input_str}")
        save_empty_nii(mod_path, output_path_whole)
        save_empty_nii(mod_path, output_path_inner)
        return

    run_tumorsynth_cmd([
        "mri_TumorSynth",
        "--i", input_str,
        "--o", output_path_whole,
        "--wholetumor"
    ], logger)
    logger.info(f"Saved Wholetumor Segmentation {key}_seg to {output_path_whole}")


    # Extract tumor ROI 

    # check if image is empty
    if is_empty(output_path_whole):
        logger.warning(f"Skipping {key} — empty whole tumor segmentation: {input_str}")
        save_empty_nii(mod_path, output_path_inner)
        return

    
    mod_tumor_roi = str(seg_dir / f"{key}_tumor_roi.nii.gz")

    logger.info(f"Extracting tumor ROI for {key}...")
    subprocess.run([
        "fslmaths", output_path_whole,
        "-thr", "17.5", "-uthr", "18.5",
        "-bin",
        "-mul", mod_path,
        mod_tumor_roi
    ], check=True)
    logger.info(f"Saved tumor ROI to {mod_tumor_roi}")

    # Inner tumor substructure segmentation

    if is_empty(mod_tumor_roi):
        logger.warning(f"Skipping {key} — empty tumor ROI: {mod_tumor_roi}")
        save_empty_nii(mod_path, output_path_inner)
        return

    run_tumorsynth_cmd([
        "mri_TumorSynth",
        "--i", mod_tumor_roi,
        "--o", output_path_inner,
        "--innertumor"
    ], logger)
    logger.info(f"Saved Inner Tumor Segmentation {key}_seg to {output_path_inner}")

class Metric(ABC):
    """Base class for metrics.
    
    Each metric defines:
    - name: Identifier used in config (e.g., "mse", "dice")
    - output_dir: Subdirectory for computed files (e.g., "segmentations")
    - get_paths(): Returns pred_and_data_paths and metric_input_paths
    - compute_inputs(): Computes metric inputs from pred/gt (default: no-op)
    - compute(): Computes metric from loaded data
    
    Subclasses are automatically registered via __init_subclass__.
    """
    _registry: ClassVar[dict[str, type[Metric]]] = {} # automatically tracks all metric subclasses:
    
    name: str
    output_dir: str | None = None
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            Metric._registry[cls.name] = cls
    
    @classmethod
    def get(cls, name: str) -> type[Metric]:
        """Get a registered metric class by name."""
        return cls._registry[name]
    
   
    def get_paths(self, sample: dict, run_dir: str, data_base_dir: str, mod_name: str, mode: str = "train") -> dict:
        MOD_SAMPLE_KEY_MAP = {"T1": "T1", "T1C": "T1C", "FLAIR": "FLAIR"}
        MOD_FILE_MAP       = {"T1": "T1", "T1C": "T1C", "FLAIR": "FLAIR"}

        sample_key = MOD_SAMPLE_KEY_MAP.get(mod_name, mod_name)
        gt_path   = Path(sample[sample_key])   
        anat_path = gt_path.parent             
        ses_path  = anat_path.parent           
        sub_path  = ses_path.parent            

        
        pred_rel_path = Path(sub_path.name) / ses_path.name / anat_path.name / f"{mod_name}.nii.gz"
       

        if mode == "train":
            pred_path = str(Path(run_dir) / pred_rel_path)
        elif mode in ["eval", "test"]:
            #pred_path = str(Path(run_dir) / "predictions" / "original" / pred_rel_path)
            pred_path = str(Path(run_dir) / pred_rel_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {
            "pred_and_data_paths": {},
            "metric_input_paths": {
                "pred": pred_path,
                "gt": str(gt_path),
            },
        }


    def compute_inputs(self, pred_and_data_paths: dict, metric_input_paths: dict, logger: Logger) -> None:
        """Compute metric input files from pred/gt files. Default: no-op."""
        pass
    
    @abstractmethod
    def compute(self, data: dict, transforms: dict) -> dict:
        """Compute metric from loaded data and return {metric_name: value}."""


class MSE(Metric):
    name = "mse"
    
    def compute(self, data, transforms):
        pred_3d = transforms["3d"](data["pred"])
        gt_3d = transforms["3d"](data["gt"])
        val_3d = mse(pred_3d, gt_3d).item()
        
        pred_2d = transforms["2d"](data["pred"])
        gt_2d = transforms["2d"](data["gt"])
        val_2d = mse(pred_2d, gt_2d).item()
        
        return {"mse_2d": val_2d, "mse_3d": val_3d}


class PSNR(Metric):
    name = "psnr"
    
    def compute(self, data, transforms):
        pred_3d = transforms["3d"](data["pred"])
        gt_3d = transforms["3d"](data["gt"])
        val_3d = psnr(pred_3d, gt_3d, data_range=1.0).item()
        
        pred_2d = transforms["2d"](data["pred"])
        gt_2d = transforms["2d"](data["gt"])
        val_2d = psnr(pred_2d, gt_2d, data_range=1.0).item()
        
        return {"psnr_2d": val_2d, "psnr_3d": val_3d}


class SSIM(Metric):
    name = "ssim"
    
    def compute(self, data, transforms):
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
       
        pred_3d_tf = transforms["3d"](data["pred"])
        gt_3d_tf = transforms["3d"](data["gt"])
        
        pred_3d = pred_3d_tf.unsqueeze(0).to(device)
        gt_3d = gt_3d_tf.unsqueeze(0).to(device)
        
        with torch.inference_mode():
            val_3d = ssim(pred_3d, gt_3d).item()
            
        # --- 2D SSIM ---
        # Crop to 2D model dimensions
        pred_2d = transforms["2d"](data["pred"])
        gt_2d = transforms["2d"](data["gt"])

        # For 2D: (1, H, W, D) -> (D, 1, H, W), treat depth slices as batch
        pred_2d = pred_2d.permute(3, 0, 1, 2).to(device)
        gt_2d = gt_2d.permute(3, 0, 1, 2).to(device)
        
        with torch.inference_mode():
            val_2d = ssim(pred_2d, gt_2d).item()
            
        return {"ssim_2d": val_2d, "ssim_3d": val_3d}



class Dice(Metric):
    name = "dice"
    output_dir = "segmentations"

    def get_paths(self, sample, run_dir, data_base_dir, mod_name,mode):
       
        base = super().get_paths(sample, run_dir, data_base_dir, mod_name, mode=mode)
        

        MOD_SAMPLE_KEY_MAP = {"T1": "T1", "T1C": "T1C", "FLAIR": "FLAIR"}
        sample_key = MOD_SAMPLE_KEY_MAP.get(mod_name, mod_name)
        gt_path = Path(sample[sample_key])
        

        anat_path = gt_path.parent           # .../anat
        ses_path  = anat_path.parent         # .../ses-XXXXXXXX
        sub_path  = ses_path.parent          # .../sub-XXXXXX

        rel_parts = Path(sub_path.name) / ses_path.name / anat_path.name
        stem = gt_path.stem.replace(".nii", "")

        seg_dir  = Path(run_dir) / self.output_dir / rel_parts
        pred_dir = Path(run_dir) / rel_parts

        return {
            "pred_and_data_paths": {
                **base["metric_input_paths"],
                
                "pred_masked": str(pred_dir / f"{stem}_masked.nii.gz"),
                
                "all_modalities": {
                    m: sample[m] for m in ["T1", "T1C",  "FLAIR"]
                    if m in sample and m != mod_name
                },
            },
            "metric_input_paths": {
                "pred_seg": str(seg_dir / f"{stem}_pred_tumorsynth.nii.gz"),
                "gt_seg":   str(seg_dir / f"{stem}_gt_tumorsynth.nii.gz"),
                "gt_seg_whole":   str(seg_dir / f"{stem}_gt_tumorsynth_whole.nii.gz"),
                "pred_seg_whole":   str(seg_dir / f"{stem}_pred_tumorsynth_whole.nii.gz"),
            },
        }
    
    def compute_inputs(self, pred_and_data_paths, metric_input_paths, logger):
        #Mask pred, then run TumorSynth wholetumor segmentation on pred and gt.
        seg_dir = Path(metric_input_paths["pred_seg"]).parent
        seg_dir.mkdir(parents=True, exist_ok=True)

       

        logger.info(f"COMPUTING DICE SCORE")
        
        extra_mods = pred_and_data_paths.get("all_modalities", {})

      
        inputs = {
            "pred": pred_and_data_paths["pred"],  # raw, no mask
            "gt":   pred_and_data_paths["gt"],
            }

        # Run whole-tumor segmentation for both prediction and ground truth
        for key in ["pred", "gt"]:
            output_path_inner = metric_input_paths[f"{key}_seg"]
            output_path_whole = metric_input_paths[f"{key}_seg_whole"]
            if Path(output_path_inner).exists():
                logger.info(f"{key}_seg already exists, skipping: {output_path_inner}")
                continue

           
            input_str = inputs[key]

            # segmentation using tumorSynth
            segment_image(key,  seg_dir,input_str, output_path_whole, output_path_inner, logger)
            
    
    def compute(self, data, transforms):
        # Inner tumor segmentation (--innertumor output)
        pred_seg_inner = transforms["3d"](data["pred_seg"]).long()
        gt_seg_inner   = transforms["3d"](data["gt_seg"]).long()

        # Whole tumor segmentation (--wholetumor output)
        pred_seg_whole = transforms["3d"](data["pred_seg_whole"]).long()
        gt_seg_whole   = transforms["3d"](data["gt_seg_whole"]).long()

        # Dice for whole tumor (19 classes — brain structures + whole tumor label)
        score_whole = dice(
            pred_seg_whole.unsqueeze(0),
            gt_seg_whole.unsqueeze(0),
            num_classes=19,
            input_format="index",
            average="none",
        ).squeeze().tolist()

        # Dice for inner tumor (3 classes: 0=bg, 1=NCR, 2=ET)
        score_inner = dice(
            pred_seg_inner.unsqueeze(0),
            gt_seg_inner.unsqueeze(0),
            num_classes=4,
            input_format="index",
            average="none",
        ).squeeze().tolist()

        return {
            "dice_bg":                   score_whole[0],   # Unknown
            "dice_CWM":                  score_whole[1],   # Cerebral-White-Matter
            "dice_Cere_Cor":             score_whole[2],   # Cerebral-Cortex
            "dice_Lateral_Vent":         score_whole[3],   # Lateral-Ventricle
            "dice_Infe_Lateral_Vent":    score_whole[4],   # Inferior-Lateral-Ventricle
            "dice_Cerebellum_WM":        score_whole[5],   # Cerebellum-White-Matter
            "dice_Cerebellum_Cortex":    score_whole[6],   # Cerebellum-Cortex
            "dice_Thalamus":             score_whole[7],   # Thalamus
            "dice_Caudate":              score_whole[8],   # Caudate
            "dice_Putamen":              score_whole[9],   # Putamen
            "dice_Pallidum":             score_whole[10],  # Pallidum
            "dice_Ventricle_3":          score_whole[11],  # 3rd-Ventricle
            "dice_Ventricle_4":          score_whole[12],  # 4th-Ventricle
            "dice_Brain_Stem":           score_whole[13],  # Brain-Stem
            "dice_Hippocampus":          score_whole[14],  # Hippocampus
            "dice_Amygdala":             score_whole[15],  # Amygdala
            "dice_Accumbens_Area":       score_whole[16],  # Accumbens-Area
            "dice_Ventral_DC":           score_whole[17],  # Ventral-DC
            "dice_whole_tumor":          score_whole[18],  # Whole-Tumor
        }

class JacobianMAE(Metric):
    """Jacobian MAE: MAE between Jacobian determinants of baseline→pred and baseline→gt registrations."""
    name = "jac_mae"
    output_dir = "jacobians"

   
    def get_paths(self, sample, run_dir, data_base_dir, mod_name,mode):
        #base = super().get_paths(sample, run_dir, data_base_dir, mod_name)
        base = super().get_paths(sample, run_dir, data_base_dir, mod_name, mode=mode)
        #gt_path = Path(sample[mod_name])

        
        MOD_SAMPLE_KEY_MAP = {"T1": "T1", "T1C": "T1C", "FLAIR": "FLAIR"}
        sample_key = MOD_SAMPLE_KEY_MAP.get(mod_name, mod_name)
        gt_path = Path(sample[sample_key])
        #relative_path = gt_path.relative_to(Path(data_base_dir))
        #stem = relative_path.stem.replace(".nii", "")
        #jac_dir = Path(run_dir) / self.output_dir / relative_path.parent
        #pred_dir = Path(run_dir) / "predictions" / relative_path.parent
        
        anat_path = gt_path.parent
        ses_path  = anat_path.parent
        sub_path  = ses_path.parent

        rel_parts = Path(sub_path.name) / ses_path.name / anat_path.name
        stem = gt_path.stem.replace(".nii", "")

        jac_dir  = Path(run_dir) / self.output_dir / rel_parts
        pred_dir = Path(run_dir) / "predictions" / rel_parts

        MOD_SAMPLE_KEY_MAP = {"T1": "T1", "T1C": "T1C", "FLAIR": "FLAIR"}
        sample_key = MOD_SAMPLE_KEY_MAP.get(mod_name, mod_name)
        return {
            "pred_and_data_paths": {
                **base["metric_input_paths"],
                #"baseline": sample[f"baseline_{mod_name}"],
                
                "baseline": sample[f"baseline_{sample_key}"],
                #"brain_mask": sample["brain_mask"],
                "pred_masked": str(pred_dir / f"{stem}_masked.nii.gz"),
                # pass all available modalities for multi-sequence support
            },
            "metric_input_paths": {
                "pred_jac": str(jac_dir / f"{stem}_pred_jac.nii.gz"),
                "gt_jac":   str(jac_dir / f"{stem}_gt_jac.nii.gz"),
                },
        }

    def compute_inputs(self, pred_and_data_paths, metric_input_paths, logger):
        """Create masked pred, then run registration and compute Jacobians."""
        logger.info(f"COMPUTING JACOBIAN")
        baseline = pred_and_data_paths["baseline"]
        #jac_dir = Path(metric_input_paths["pred_jac"]).parent
        #jac_dir.mkdir(parents=True, exist_ok=True)
        jac_dir = Path(metric_input_paths["pred_jac"]).parent
        jac_dir.mkdir(parents=True, exist_ok=True)
        pred_masked_path = Path(pred_and_data_paths["pred_masked"])
        pred_masked_path.parent.mkdir(parents=True, exist_ok=True)

       

        stem = Path(metric_input_paths["pred_jac"]).stem.replace("_pred_jac", "")

        inputs = {
            "pred": pred_and_data_paths["pred"],
            "gt":   pred_and_data_paths["gt"],
        }
        for key in ["pred", "gt"]:
            jac_path = metric_input_paths[f"{key}_jac"]
            disp_path = str(jac_dir / f"{stem}_{key}_d.nii.gz")

            if Path(jac_path).exists():
                logger.info(f"{key}_jac already exists, skipping: {jac_path}")
                continue

            # Guard against empty/NaN images
            if is_empty(inputs[key]):
                logger.warning(f"Skipping {key}_jac for {stem} — empty/NaN image: {inputs[key]}")
                save_empty_nii(inputs[key], jac_path)
                continue

            cmds = [
                ["mri_synthmorph", "-t", disp_path, "--threads", "10", baseline, inputs[key]],
                ["CreateJacobianDeterminantImage", "3", disp_path, jac_path, "1"],
                #["fslcpgeom", baseline, jac_path, "-d"],
                ["/opt/fsl-6.0.7.1/bin/fslcpgeom", baseline, jac_path, "-d"],
            ]
            for cmd in cmds:
                run_apptainer_cmd(cmd, logger)

            jac_img = nib.load(jac_path)
            nib.save(nib.Nifti1Image(jac_img.get_fdata().astype(np.float32), jac_img.affine), jac_path)

            Path(disp_path).unlink(missing_ok=True)
            logger.info(f"Saved {key}_jac to {jac_path} (float32), removed displacement field")

    def compute(self, data, transforms):
        pred_jac = transforms["3d"](data["pred_jac"])
        gt_jac   = transforms["3d"](data["gt_jac"])
        return {"jac_mae": mae(pred_jac, gt_jac).item()}


class TumorVolume(Metric):
    """cwm volume comparison from TumorSynth for predicted and ground-truth scans."""
    name = "cwm_vol"
    output_dir = "segmentations"

    def get_paths(self, sample, run_dir, data_base_dir, mod_name,mode):
        #base = super().get_paths(sample, run_dir, data_base_dir, mod_name)
        base = super().get_paths(sample, run_dir, data_base_dir, mod_name, mode=mode)
        #gt_path = Path(sample[mod_name])

        MOD_SAMPLE_KEY_MAP = {"T1": "T1", "T1C": "T1C", "FLAIR": "FLAIR"}
        sample_key = MOD_SAMPLE_KEY_MAP.get(mod_name, mod_name)
        gt_path = Path(sample[sample_key])

        #relative_path = gt_path.relative_to(Path(data_base_dir))
        #stem = relative_path.stem.replace(".nii", "")
        #seg_dir = Path(run_dir) / self.output_dir / relative_path.parent

        anat_path = gt_path.parent
        ses_path  = anat_path.parent
        sub_path  = ses_path.parent

        rel_parts = Path(sub_path.name) / ses_path.name / anat_path.name
        stem = gt_path.stem.replace(".nii", "")

        seg_dir = Path(run_dir) / self.output_dir / rel_parts

        return {
            "pred_and_data_paths": {
                **base["metric_input_paths"],
                #"brain_mask": sample["brain_mask"],  
                "pred_masked": str(seg_dir / f"{stem}_pred_masked.nii.gz"),
                "pred_seg":    str(seg_dir / f"{stem}_pred_tumorsynth.nii.gz"),
                "gt_seg":      str(seg_dir / f"{stem}_gt_tumorsynth.nii.gz"),
            },
            "metric_input_paths": {
                "pred_seg": str(seg_dir / f"{stem}_pred_tumorsynth.nii.gz"),
                "gt_seg":   str(seg_dir / f"{stem}_gt_tumorsynth.nii.gz"),
                "gt_seg_whole":   str(seg_dir / f"{stem}_gt_tumorsynth_whole.nii.gz"),
                "pred_seg_whole":   str(seg_dir / f"{stem}_pred_tumorsynth_whole.nii.gz"),
            },
        }

    def compute_inputs(self, pred_and_data_paths, metric_input_paths, logger):
        """Mask pred, then run TumorSynth wholetumor segmentation on pred and gt."""
        logger.info(f"COMPUTING TUMOR")
        seg_dir = Path(metric_input_paths["pred_seg"]).parent
        seg_dir.mkdir(parents=True, exist_ok=True)

      

        extra_mods = pred_and_data_paths.get("all_modalities", {})

        inputs = {
            "pred": pred_and_data_paths["pred"],
            "gt":   pred_and_data_paths["gt"],
        }

        for key in ["pred", "gt"]:
            output_path_inner = metric_input_paths[f"{key}_seg"]
            output_path_whole = metric_input_paths[f"{key}_seg_whole"]
            if Path(output_path_inner).exists():
                logger.info(f"{key}_seg already exists, skipping: {output_path_inner}")
                continue

            input_str = inputs[key]

            segment_image(key,  seg_dir,input_str, output_path_whole, output_path_inner, logger)

    
    def compute_metric(self, metric_input_paths, logger):
        #Compute Celebral White Matter (in mL) for pred and gt, return absolute difference.
        results = {}

        for key in ["pred", "gt"]:
            seg_path = metric_input_paths[f"{key}_seg"]
            img = nib.load(seg_path)
            data = img.get_fdata()

            # Voxel volume in mL (mm^3 -> mL: divide by 1000)
            #voxel_vol_ml = np.prod(img.header.get_zooms()) / 1000.0
            
            _pred_seg_nib = nib.load(data["pred_seg"])
            voxel_vol_ml = np.prod(_pred_seg_nib.header.get_zooms()[:3]) / 1000.0

            
            cwm_voxels = np.sum(data == 1)
            volume_ml = cwm_voxels * voxel_vol_ml

            results[f"{key}_cwm_vol_ml"] = round(float(volume_ml), 4)
            logger.info(f"{key} CWM volume: {volume_ml:.4f} mL")

        # Absolute volume difference
        results["cwm_vol_diff_ml"] = round(
            abs(results["pred_cwm_vol_ml"] - results["gt_cwm_vol_ml"]), 4
        )


        return results
    
    def compute(self, data, transforms):
        # Inner tumor segmentation
        pred_seg_inner = transforms["3d"](data["pred_seg"])
        gt_seg_inner   = transforms["3d"](data["gt_seg"])

        # Whole tumor segmentation
        pred_seg_whole = transforms["3d"](data["pred_seg_whole"])
        gt_seg_whole   = transforms["3d"](data["gt_seg_whole"])

        # Voxel volume in mL from pred_seg_whole metadata
        pixdim = pred_seg_whole.meta["pixdim"][1:4]
        voxel_vol_ml = float(np.prod(pixdim)) / 1000.0

        # Convert to numpy
        pred_inner_np = pred_seg_inner.detach().cpu().numpy()
        gt_inner_np   = gt_seg_inner.detach().cpu().numpy()
        pred_whole_np = pred_seg_whole.detach().cpu().numpy()
        gt_whole_np   = gt_seg_whole.detach().cpu().numpy()

        # Inner tumor labels (0=bg, 1=NCR, 2=ET) — verify with np.unique()
        INNER_LABELS = {
            "inner_bg":          0,
            "necrotic_core":     1,
            "enhancing_tumor":   2,
        }

        # Whole tumor labels — verify with np.unique()
        WHOLE_LABELS = {
            "whole_bg":          0,
            "CWM":          1,
            "Cere_Cor":      2,
            "Lateral_Vent": 3,
            "Infe_Lateral_Vent":               4,
            "Cerebellum_WM": 5,
            "Cerebellum_Cortex":          6,
            "Thalamus":           7,
            "Caudate":           8,
            "Putamen":          9,
            "Pallidum":         10,
            "Ventricle_3":       11,
            "Ventricle_4":       12,
            "Brain_Stem":        13,
            "Hippocampus":       14,
            "Amygdala":          15,
            "Accumbens_Area":    16,
            "Ventral_DC":        17,
            "whole_tumor":       18,
        }

        results = {}

        # Compute volumes for inner tumor segments
        for label_name, label_idx in INNER_LABELS.items():
            pred_vol = round(float(np.sum(pred_inner_np == label_idx) * voxel_vol_ml), 4)
            gt_vol   = round(float(np.sum(gt_inner_np   == label_idx) * voxel_vol_ml), 4)
            results[f"pred_{label_name}_vol_ml"] = pred_vol
            results[f"gt_{label_name}_vol_ml"]   = gt_vol
            results[f"{label_name}_vol_diff_ml"] = round(abs(pred_vol - gt_vol), 4)

        # Compute volumes for whole tumor segments
        for label_name, label_idx in WHOLE_LABELS.items():
            pred_vol = round(float(np.sum(pred_whole_np == label_idx) * voxel_vol_ml), 4)
            gt_vol   = round(float(np.sum(gt_whole_np   == label_idx) * voxel_vol_ml), 4)
            results[f"pred_{label_name}_vol_ml"] = pred_vol
            results[f"gt_{label_name}_vol_ml"]   = gt_vol
            results[f"{label_name}_vol_diff_ml"] = round(abs(pred_vol - gt_vol), 4)

        return results
    
    