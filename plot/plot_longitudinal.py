"""Plot longitudinal scans (real vs predicted) for all subjects in the test set.

The core layout is identical to the reference plot_longitudinal.py:
  - Dark background figure
  - Outer GridSpec with one row-group per `pairs_per_row` time-points
  - Each row-group: 2 sub-rows (Real / Predicted) × `pairs_per_row` columns
  - Column title = "Day {fu_time}"

Cropping mirrors the logic in plot.plot_prediction:
  - Brain mask is loaded once per subject (from the first follow-up entry)
  - Bounding box is computed from the mask (with padding=10) and applied
    identically to all scans for that subject -> no wasted whitespace, consistent framing.

Public API used by the inference script:
    from plot.plot_longitudinal import plot_longitudinal_predictions
    plot_longitudinal_predictions(dataloader, run_dir, args, logger)

Can also be run standalone (see __main__ below).
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Cropping helpers (mirrors plot.plot_prediction logic)
# ---------------------------------------------------------------------------

def _compute_crop_box(mask: np.ndarray, pad: int = 10) -> tuple[int, int, int, int, int, int]:
    """Return (h0, h1, w0, w1, d0, d1) bounding box from a binary mask."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2])
    h0, w0, d0 = coords.min(axis=0)
    h1, w1, d1 = coords.max(axis=0)
    H, W, D = mask.shape
    h0, h1 = max(0, h0 - pad), min(H, h1 + pad + 1)
    w0, w1 = max(0, w0 - pad), min(W, w1 + pad + 1)
    d0, d1 = max(0, d0 - pad), min(D, d1 + pad + 1)
    return h0, h1, w0, w1, d0, d1


def _load_and_crop_axial(nii_path: str, h0, h1, w0, w1, d0, d1) -> np.ndarray | None:
    """Load a NIfTI, apply the pre-computed crop, return middle axial slice (rot90)."""
    if not os.path.exists(nii_path):
        return None
    data = nib.load(nii_path).get_fdata()
    data = data[h0:h1, w0:w1, d0:d1]
    mid = data.shape[2] // 2
    return np.rot90(data[:, :, mid])


def _group_plot_datalist_by_subject(
    plot_datalist: list[dict],
    run_dir: str,
) -> dict[str, list[dict]]:
    """Group plot_datalist entries (from get_plot_datalist) by subject ID.

    Each entry already has resolved 'pred', 'gt', 'brain_mask', 'fu_time'.
    Subject ID is inferred from the structure of the pred path:
        <run_dir>/predictions/<subject_id>/...

    Returns:
        dict: subject_id -> list of entries sorted by fu_time
    """
    pred_root = Path(run_dir) / "predictions"
    raw: dict[str, list] = defaultdict(list)

    for entry in plot_datalist:
        pred_path = Path(entry["pred"])
        # First component after pred_root is the subject directory (e.g. sub-13)
        subject_id = pred_path.relative_to(pred_root).parts[0]
        raw[subject_id].append(entry)

    return {
        sid: sorted(entries, key=lambda e: int(e.get("fu_time", 0)))
        for sid, entries in raw.items()
    }


def _plot_subject(
    subject_id: str,
    entries: list[dict],
    output_dir: str,
    pairs_per_row: int = 6,
    font_size: int = 20,
    logger: logging.Logger | None = None,
) -> None:
    """Plot longitudinal real vs predicted scans for one subject.

    Crop bounding box is computed from the brain_mask of the *first* follow-up
    and applied identically to every scan, ensuring consistent framing.
    """
    n_timepoints = len(entries)
    if n_timepoints == 0:
        return


    first_mask_path = entries[0].get("brain_mask", "")
    if first_mask_path and os.path.exists(first_mask_path):
        mask_data = nib.load(first_mask_path).get_fdata()
    else:
        # Fall back: load first real scan and use its full extent
        first_gt = entries[0].get("gt", "")
        if first_gt and os.path.exists(first_gt):
            mask_data = nib.load(first_gt).get_fdata()
            mask_data = (mask_data > 0).astype(np.float32)
        else:
            mask_data = None

    if mask_data is not None:
        h0, h1, w0, w1, d0, d1 = _compute_crop_box(mask_data, pad=10)
    else:
        # Can't crop; will try to load without crop
        h0 = h1 = w0 = w1 = d0 = d1 = None

    def get_axial_slice(path: str) -> np.ndarray | None:
        if h0 is None:
            # No crop available – load and take middle slice of the full volume
            if not path or not os.path.exists(path):
                return None
            data = nib.load(path).get_fdata()
            mid = data.shape[2] // 2
            return np.rot90(data[:, :, mid])
        return _load_and_crop_axial(path, h0, h1, w0, w1, d0, d1)


    n_rows = (n_timepoints + pairs_per_row - 1) // pairs_per_row
    cols = n_timepoints // n_rows
    if cols == 0:
        cols = 1

    plt.style.use("dark_background")
    fig_width = max(6.0, cols * 3.0)
    fig = plt.figure(figsize=(fig_width, 9.0 * n_rows))
    fig.patch.set_facecolor("black")

    outer_gs = GridSpec(n_rows, 1, figure=fig, hspace=0.2)

    for row_group in range(n_rows):
        inner_gs = outer_gs[row_group].subgridspec(2, cols, hspace=0, wspace=0)

        for col in range(cols):
            idx = row_group * cols + col

            real_ax = fig.add_subplot(inner_gs[0, col])
            pred_ax = fig.add_subplot(inner_gs[1, col])

            # Row labels on the leftmost column of each row-group
            if col == 0:
                real_ax.text(
                    -0.15, 0.5, "Real Follow-Up",
                    transform=real_ax.transAxes,
                    rotation=90, verticalalignment="center",
                    fontsize=font_size, color="cyan",
                )
                pred_ax.text(
                    -0.15, 0.5, "Predicted Follow-Up",
                    transform=pred_ax.transAxes,
                    rotation=90, verticalalignment="center",
                    fontsize=font_size, color="orange",
                )

            entry = entries[idx]
            fu_time = int(entry.get("fu_time", 0))
            real_path = entry.get("gt", "")
            pred_path = entry.get("pred", "")

            # Real Follow-Up
            real_slice = get_axial_slice(real_path)
            if real_slice is not None:
                real_ax.imshow(real_slice, cmap="gray")
            else:
                real_ax.text(0.5, 0.5, "Real image\nnot found",
                             ha="center", va="center",
                             transform=real_ax.transAxes, color="white")
                real_ax.set_facecolor("black")
            real_ax.set_title(f"Day {fu_time}", fontsize=font_size, color="white", pad=20)
            real_ax.axis("off")

            # Predicted Follow-Up
            pred_slice = get_axial_slice(pred_path)
            if pred_slice is not None:
                pred_ax.imshow(pred_slice, cmap="gray")
            else:
                pred_ax.text(0.5, 0.5, "Predicted image\nnot found",
                             ha="center", va="center",
                             transform=pred_ax.transAxes, color="white")
                pred_ax.set_facecolor("black")
            pred_ax.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}_longitudinal.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    if logger:
        logger.info(f"Saved longitudinal plot: {output_path}")
    else:
        print(f"Saved: {output_path}")
    plt.close(fig)



def plot_longitudinal_predictions(
    dataloader,
    run_dir: str,
    args: Namespace,
    logger: logging.Logger,
    pairs_per_row: int = 6,
    font_size: int = 20,
) -> None:
    """Generate one longitudinal plot per subject in the test dataloader.

    Mirrors the signature of plot.plot_predictions so it slots in cleanly
    next to the existing per-sample plots in controlnet_infer.py.

    Args:
        dataloader: Test DataLoader (dataset.data contains the raw datalist).
        run_dir:    Run directory; predictions live at <run_dir>/predictions/.
        args:       Inference args namespace (needs data_base_dir, brain_mask_filename).
        logger:     Logger instance.
        pairs_per_row: Time-points per row group (default 6).
        font_size:  Font size for labels/titles (default 14).
    """
    from plot.plot import get_plot_datalist  # local import to avoid circular deps

    datalist = dataloader.dataset.data
    plot_datalist = get_plot_datalist(datalist, run_dir, args)

    subjects = _group_plot_datalist_by_subject(plot_datalist, run_dir)

    output_dir = str(Path(run_dir) / "plots" / "longitudinal")

    logger.info(
        f"Generating longitudinal plots for {len(subjects)} subject(s): "
        f"{sorted(subjects.keys())}"
    )

    for subject_id, entries in sorted(subjects.items()):
        logger.info(
            f"  {subject_id}: {len(entries)} follow-up(s) "
            f"(fu_times: {[e.get('fu_time') for e in entries]})"
        )
        _plot_subject(
            subject_id=subject_id,
            entries=entries,
            output_dir=output_dir,
            pairs_per_row=pairs_per_row,
            font_size=font_size,
            logger=logger,
        )
