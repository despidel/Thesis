from argparse import Namespace
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils.visualizer import get_roi_mask_center_indices
from monai.transforms import LoadImage

def _wrap_title(title: str, max_chars: int = 12) -> str:
    """Wraps title into multiple lines if it exceeds max_chars."""
    if len(title) <= max_chars:
        return title
    
    words = title.split(' ')
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + (1 if current_line else 0) > max_chars:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                # Word itself is longer than max_chars
                lines.append(word)
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)
            
    if current_line:
        lines.append(' '.join(current_line))
        
    return '\n'.join(lines)

def plot_prediction(
    file_paths_dict: dict[str, str],
    mask_path: str,
    output_path: str,
    logger: logging.Logger,
    slice_indices: dict[str, int] = None,
    font_size: int = 20,
    scan_cmap: str = 'gray',
    dose_cmap: str = 'jet',
    background: str = 'black',
    vmin_vmax_scans: dict[str, tuple[float, float]] = None,
    colorbar: bool = True,
    clip_range: tuple[float, float] = None,
    mod_name: str = None,
):
    """
    Plots medical images across 3 rows (axial, sagittal, coronal) per modality:
    Baseline , Dose Map (with colorbar), Real Follow-up, Predicted Follow-up.
    Uses GridSpec for precise layout control with colorbar next to dose map.
    Applies mask to dose map to remove background.
    """

   
    # Expected order of images
    expected_keys = ['Dose Map', f"Baseline Scan ({mod_name})",  f"Real Follow-up ({mod_name})", f"Predicted Follow-up ({mod_name})"]

    
    for key in expected_keys:
        if key not in file_paths_dict:
            raise ValueError(f"Missing required key: {key}")

    # Load images
    images = {}
    for key in expected_keys:
        if os.path.exists(file_paths_dict[key]):
            images[key] = nib.load(file_paths_dict[key]).get_fdata()
        else:
            raise FileNotFoundError(f"File not found for {key}: {file_paths_dict[key]}")

    # Load mask
    if os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata()
    else:
        mask = np.ones_like(list(images.values())[0])
        

    # Crop arrays tightly around ROI (prioritize brain mask, fallback to baseline)
    ref_for_crop = mask
    if np.all(mask == 1) and 'Baseline Scan' in images:
        ref_for_crop = images['Baseline Scan']
    
    coords = np.argwhere(ref_for_crop > 0)
    if len(coords) > 0:
        h0, w0, d0 = coords.min(axis=0)
        h1, w1, d1 = coords.max(axis=0)
        
        pad = 10 
        orig_H, orig_W, orig_D = mask.shape
        h0, h1 = max(0, h0 - pad), min(orig_H, h1 + pad + 1)
        w0, w1 = max(0, w0 - pad), min(orig_W, w1 + pad + 1)
        d0, d1 = max(0, d0 - pad), min(orig_D, d1 + pad + 1)
        
        mask = mask[h0:h1, w0:w1, d0:d1]
        for key in images:
            images[key] = images[key][h0:h1, w0:w1, d0:d1]
            
        if slice_indices:
            if 'axial' in slice_indices:
                slice_indices['axial'] = max(0, min(d1-d0-1, slice_indices['axial'] - d0))
            if 'sagittal' in slice_indices:
                slice_indices['sagittal'] = max(0, min(w1-w0-1, slice_indices['sagittal'] - w0))
            if 'coronal' in slice_indices:
                slice_indices['coronal'] = max(0, min(h1-h0-1, slice_indices['coronal'] - h0))

    # Assume all images have the same shape
    H, W, D = list(images.values())[0].shape

    # All 3 orientations
    orientations = ['axial', 'sagittal', 'coronal']

    # Default to middle slices if not specified (in case mask was empty or slice_indices wasn't fully populated)
    if slice_indices is None:
        slice_indices = {}
    if 'axial' not in slice_indices:
        slice_indices['axial'] = D // 2
    if 'sagittal' not in slice_indices:
        slice_indices['sagittal'] = W // 2
    if 'coronal' not in slice_indices:
        slice_indices['coronal'] = H // 2

    # Helper to extract slice
    def get_slice(data, orientation, idx):
        if orientation == 'axial':
            return np.rot90(data[:, :, idx])
        elif orientation == 'sagittal':
            return np.rot90(data[:, idx, :])
        elif orientation == 'coronal':
            return np.rot90(data[idx, :, :])
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

    # Compute slice dimensions for each orientation (after rot90)
    # rot90 swaps rows/cols, so shape (R,C) -> (C,R)
    slice_dims = {}  # orientation -> (height_px, width_px)
    for o in orientations:
        if o == 'axial':      # data[:,:,idx] -> (H,W) -> rot90 -> (W,H)
            slice_dims[o] = (W, H)
        elif o == 'sagittal':  # data[:,idx,:] -> (H,D) -> rot90 -> (D,H)
            slice_dims[o] = (D, H)
        elif o == 'coronal':   # data[idx,:,:] -> (W,D) -> rot90 -> (D,W)
            slice_dims[o] = (D, W)

    # All columns show same-shaped images per row, so column width = max width across rows
    max_width_px = max(sd[1] for sd in slice_dims.values())

    # Row heights in pixels (the height of each row's images)
    row_height_px = [slice_dims[o][0] for o in orientations]

    # Figure dimensions
    scale = 0.015  # inches per pixel
    col_width = max_width_px * scale
    fig_width = 4 * col_width

    # Figure height: sum of row heights (in inches) + zoom row + colorbar + minimal margins
    zoom_row_height_px = row_height_px[-1]  # Zoom the sagittal row
    cbar_height_inches = 1.2  # absolute height for colorbar strip, increased for more Y margins
    top_margin_inches = 0.75   # Enough space so 2-line title isn't cut off
    bottom_margin_inches = 0.75  # Enough space for zoom labels beneath
    grid_height = sum(rh * scale for rh in row_height_px) + (zoom_row_height_px * scale)
    fig_height = grid_height + cbar_height_inches + top_margin_inches + bottom_margin_inches

    top = 1 - top_margin_inches / fig_height
    bottom = bottom_margin_inches / fig_height

    # Height ratios: proportional to pixel heights for image rows, colorbar is fixed
    cbar_ratio_in_px = cbar_height_inches / scale
    height_ratios = row_height_px + [cbar_ratio_in_px, zoom_row_height_px]

    # Create figure with specified background
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=background)

    # Create GridSpec: 5 rows (3 anatomy + 1 colorbar + 1 zoom), 4 columns
    gs = gridspec.GridSpec(5, 4, figure=fig,
                          width_ratios=[1, 1, 1, 1],
                          height_ratios=height_ratios,
                          wspace=0,
                          hspace=0,
                          left=0.0, right=1.0,
                          top=top, bottom=bottom)

    text_color = 'white' if background == 'black' else 'black'


    dose_im = None  # Store dose image for colorbar
    source_axes = {}  # Store axes for drawing connection lines

    # Zoom window computation (based on the third row view)
    zoom_orientation = orientations[2]
    z_idx = slice_indices[zoom_orientation]
    m_slice = get_slice(mask, zoom_orientation, z_idx)
    mask_coords = np.argwhere(m_slice > 0)
    if len(mask_coords) > 0:
        ymin, xmin = mask_coords.min(axis=0)
        ymax, xmax = mask_coords.max(axis=0)
        cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
    else:
        cy, cx = m_slice.shape[0] // 2, m_slice.shape[1] // 2
    
    # Define zoom crop (e.g. 80x80 box for common brain sizes)
    crop_size = int(0.4 * min(m_slice.shape))
    y0, y1 = cy - crop_size // 2, cy + crop_size // 2
    x0, x1 = cx - crop_size // 2, cx + crop_size // 2
    y0, y1 = max(0, y0), min(m_slice.shape[0], y1)
    x0, x1 = max(0, x0), min(m_slice.shape[1], x1)

    for row_idx, orientation in enumerate(orientations):
        idx = slice_indices[orientation]
        mask_slice = get_slice(mask, orientation, idx)

        for i, key in enumerate(expected_keys):
            ax = fig.add_subplot(gs[row_idx, i])
            data = images[key]
            slice_ = get_slice(data, orientation, idx)

            # Apply clipping if specified
            if clip_range is not None:
                slice_ = np.clip(slice_, clip_range[0], clip_range[1])

            # Display image based on type
            if key == 'Dose Map':
                masked_slice = slice_.copy().astype(float)
                masked_slice[mask_slice == 0] = np.nan

                valid_values = masked_slice[~np.isnan(masked_slice)]
                # Fix vmin and vmax for dose map
                vmin, vmax = 0, 64

                dose_im = ax.imshow(masked_slice, cmap=dose_cmap, aspect='equal', vmin=vmin, vmax=vmax)
            else:
                vmin, vmax = (None, None)
                if vmin_vmax_scans and key in vmin_vmax_scans:
                    vmin, vmax = vmin_vmax_scans[key]
                
                # Mask predicted scan to remove background noise
                if key == 'Predicted Follow-up Scan':
                    slice_ = slice_ * (mask_slice > 0)

                ax.imshow(slice_, cmap=scan_cmap, aspect='equal', vmin=vmin, vmax=vmax)
            
            # Add red rectangle on source row (the 3rd anatomical row) for scans
            if row_idx == 2 and key != 'Dose Map':
                rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                source_axes[key] = ax

            # Column titles only on first row
            if row_idx == 0:
                ax.set_title(_wrap_title(key.replace(' Scan', '')), fontsize=font_size, color=text_color, pad=3)

            ax.set_facecolor(background)
            ax.axis('off')

    # Add horizontal colorbar below the dose map column (Row 3, column 0)
    if colorbar and dose_im is not None:
        cbar_area_ax = fig.add_subplot(gs[3, 0])  # Row 3, column 0 (dose column)
        cbar_area_ax.axis('off')
        
        # Use inset_axes to make the colorbar thinner and leave room for ticks below it
        cbar_ax = inset_axes(cbar_area_ax,
                             width="80%", 
                             height=0.1, # Explicit thin height in inches
                             loc='center', # Center it to allow margin above and below
                             borderpad=0)
                             
        cbar = plt.colorbar(dose_im, cax=cbar_ax, orientation='horizontal')

        cbar.ax.tick_params(colors=text_color, labelsize=font_size-5, width=0.5, length=3, pad=2)
        
        # Put colorbar on top of connection lines
        cbar_ax.set_zorder(10)
        
        # Add a background rectangle to cover the colorbar and its ticks cleanly
        bg_rect = Rectangle((-0.05, -2.5), 1.1, 4.0, transform=cbar_ax.transAxes, 
                            facecolor=background, edgecolor='none', zorder=-1, clip_on=False)
        cbar_ax.add_patch(bg_rect)
        
        cbar.set_ticks([0, 64])
        cbar.set_ticklabels(['0 Gy', '64 Gy'])
        # Adjust horizontal alignment to prevent cut-off at the edges
        tick_labels = cbar.ax.get_xticklabels()
        if len(tick_labels) >= 2:
            tick_labels[0].set_ha('left')
            tick_labels[-1].set_ha('right')

    # Add middle-centered zoomed row (Row 4)
    # Keys to zoom: Baseline, Real, Predicted. These are indices 1, 2, 3 in expected_keys.
    zoom_keys = [f"Baseline Scan ({mod_name})",  f"Real Follow-up ({mod_name})", f"Predicted Follow-up ({mod_name})"]
    short_titles = ['Baseline', 'Real Follow-up', 'Predicted Follow-up']

    for i, (key, title) in enumerate(zip(zoom_keys, short_titles)):
        ax_zoom = fig.add_subplot(gs[4, i + 1])  # Use columns 1, 2, 3
        data = images[key]
        slice_full = get_slice(data, zoom_orientation, z_idx)
        slice_crop = slice_full[y0:y1, x0:x1]
        
        # Apply clipping to crop too
        if clip_range is not None:
            slice_crop = np.clip(slice_crop, clip_range[0], clip_range[1])
            
        vmin, vmax = (None, None)
        if vmin_vmax_scans and key in vmin_vmax_scans:
            vmin, vmax = vmin_vmax_scans[key]
        
        ax_zoom.imshow(slice_crop, cmap=scan_cmap, aspect='equal', vmin=vmin, vmax=vmax)
        
        # Red border around zoom
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
            spine.set_visible(True)
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        
        # Title below
        ax_zoom.set_xlabel(_wrap_title(title), color=text_color, fontsize=font_size-2, labelpad=5)
        ax_zoom.xaxis.set_label_position('bottom')
        
            # Draw connection lines from source to zoom
        if key in source_axes:
            src_ax = source_axes[key]
            # Bottom-left of rect to top-left of zoom
            con1 = ConnectionPatch(xyA=(x0, y1), xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                                  axesA=src_ax, axesB=ax_zoom, color="red", linestyle="--", linewidth=0.8)
            # Bottom-right of rect to top-right of zoom
            con2 = ConnectionPatch(xyA=(x1, y1), xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                                  axesA=src_ax, axesB=ax_zoom, color="red", linestyle="--", linewidth=0.8)
            
            fig.add_artist(con1)
            fig.add_artist(con2)

    # Save figure
    plt.savefig(output_path, facecolor=background, dpi=150, pad_inches=0)
    logger.info(f"Saved prediction figure to {output_path}")
    plt.close(fig)


def plot_jacobian(
    file_paths_dict: dict[str, str],
    mask_path: str,
    output_path: str,
    logger: logging.Logger,
    slice_indices: dict[str, int] = None,
    font_size: int = 18,
    scan_cmap: str = 'gray',
    jac_cmap: str = 'jet',
    background: str = 'black',
    jac_vmin: float = -1.0,
    jac_vmax: float = 1.0,
    jac_threshold: float = 0.33,
):
    """
    Plots 3 medical images across 3 rows (axial, sagittal, coronal):
    Baseline, Real Follow-up, Predicted Follow-up.
    Jacobian determinants are overlaid on Real and Predicted Follow-up images.
    
    Args:
        file_paths_dict: Dict with keys 'Baseline Scan', 'Real Follow-up Scan', 
                        'Predicted Follow-up Scan', 'Real Follow-up Jacobian', 'Predicted Follow-up Jacobian'
        mask_path: Path to brain mask
        output_path: Output file path
        logger: Logger instance
        slice_indices: Dict with orientation -> slice index (axial, sagittal, coronal)
        font_size: Font size for titles
        scan_cmap: Colormap for scans
        jac_cmap: Colormap for Jacobian overlay
        background: Background color
        jac_vmin: Jacobian colormap minimum (0 = shrunk to nothing)
        jac_vmax: Jacobian colormap maximum (2 = doubled size)
        jac_threshold: Values within [1-threshold, 1+threshold] are masked (33% = 0.33)
    """
    
    # 3 columns: Baseline, Real Follow-up, Predicted Follow-up (no Dose Map)
    expected_keys = ['Baseline Scan', 'Real Follow-up Scan', 'Predicted Follow-up Scan']
    jacobian_keys = ['Real Follow-up Jacobian', 'Predicted Follow-up Jacobian']
    
    # Verify all required keys are present
    for key in expected_keys + jacobian_keys:
        if key not in file_paths_dict:
            raise ValueError(f"Missing required key: {key}")

    # Load images
    images = {}
    for key in expected_keys:
        if os.path.exists(file_paths_dict[key]):
            images[key] = nib.load(file_paths_dict[key]).get_fdata()
        else:
            raise FileNotFoundError(f"File not found for {key}: {file_paths_dict[key]}")
    
    # Load Jacobians
    jacobians = {}
    for key in jacobian_keys:
        if os.path.exists(file_paths_dict[key]):
            data = nib.load(file_paths_dict[key]).get_fdata()
            jacobians[key] = np.clip(data, jac_vmin, jac_vmax)
        else:
            raise FileNotFoundError(f"File not found for {key}: {file_paths_dict[key]}")
    
    # Load mask
    if os.path.exists(mask_path):
        mask = nib.load(mask_path).get_fdata()
    else:
        mask = np.ones_like(list(images.values())[0])

    # Crop arrays tightly around ROI (prioritize brain mask, fallback to baseline)
    ref_for_crop = mask
    if np.all(mask == 1) and 'Baseline Scan' in images:
        ref_for_crop = images['Baseline Scan']
    
    coords = np.argwhere(ref_for_crop > 0)
    if len(coords) > 0:
        h0, w0, d0 = coords.min(axis=0)
        h1, w1, d1 = coords.max(axis=0)
        
        pad = 10 
        orig_H, orig_W, orig_D = mask.shape
        h0, h1 = max(0, h0 - pad), min(orig_H, h1 + pad + 1)
        w0, w1 = max(0, w0 - pad), min(orig_W, w1 + pad + 1)
        d0, d1 = max(0, d0 - pad), min(orig_D, d1 + pad + 1)
        
        mask = mask[h0:h1, w0:w1, d0:d1]
        for key in images:
            images[key] = images[key][h0:h1, w0:w1, d0:d1]
        for key in jacobians:
            jacobians[key] = jacobians[key][h0:h1, w0:w1, d0:d1]
            
        if slice_indices:
            if 'axial' in slice_indices:
                slice_indices['axial'] = max(0, min(d1-d0-1, slice_indices.get('axial', orig_D//2) - d0))
            if 'sagittal' in slice_indices:
                slice_indices['sagittal'] = max(0, min(w1-w0-1, slice_indices.get('sagittal', orig_W//2) - w0))
            if 'coronal' in slice_indices:
                slice_indices['coronal'] = max(0, min(h1-h0-1, slice_indices.get('coronal', orig_H//2) - h0))
    H, W, D = list(images.values())[0].shape

    # All 3 orientations
    orientations = ['axial', 'sagittal', 'coronal']

    # Default to middle slices if not specified
    if slice_indices is None:
        slice_indices = {}
    if 'axial' not in slice_indices:
        slice_indices['axial'] = D // 2
    if 'sagittal' not in slice_indices:
        slice_indices['sagittal'] = W // 2
    if 'coronal' not in slice_indices:
        slice_indices['coronal'] = H // 2

    # Helper to extract slice
    def get_slice(data, orientation, idx):
        if orientation == 'axial':
            return np.rot90(data[:, :, idx])
        elif orientation == 'sagittal':
            return np.rot90(data[:, idx, :])
        elif orientation == 'coronal':
            return np.rot90(data[idx, :, :])
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

    # Compute slice dimensions for each orientation (after rot90)
    slice_dims = {}  # orientation -> (height_px, width_px)
    for o in orientations:
        if o == 'axial':      # data[:,:,idx] -> (H,W) -> rot90 -> (W,H)
            slice_dims[o] = (W, H)
        elif o == 'sagittal':  # data[:,idx,:] -> (H,D) -> rot90 -> (D,H)
            slice_dims[o] = (D, H)
        elif o == 'coronal':   # data[idx,:,:] -> (W,D) -> rot90 -> (D,W)
            slice_dims[o] = (D, W)

    # All columns show same-shaped images per row, so column width = max width across rows
    max_width_px = max(sd[1] for sd in slice_dims.values())

    # Row heights in pixels (the height of each row's images)
    row_height_px = [slice_dims[o][0] for o in orientations]

    # Figure width: 3 columns (tight, no gaps), plus colorbar width
    scale = 0.015  # inches per pixel
    col_width = max_width_px * scale
    cbar_width_inches = 0.8  # Room for vertical colorbar
    right_margin_inches = 0.6  # Room for colorbar labels to not get cut off
    fig_width = 3 * col_width + cbar_width_inches + right_margin_inches

    # Figure height: sum of row heights (in inches) + minimal margins
    top_margin_inches = 0.70   # Tight top margin for multiline titles
    bottom_margin_inches = 0.35  # Tight bottom margin
    grid_height = sum(rh * scale for rh in row_height_px)
    fig_height = grid_height + top_margin_inches + bottom_margin_inches

    top = 1 - top_margin_inches / fig_height
    bottom = bottom_margin_inches / fig_height
    right = 1 - right_margin_inches / fig_width

    # Height ratios: proportional to pixel heights for image rows
    height_ratios = row_height_px

    # Create figure with specified background
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=background)

    # Create GridSpec: 3 rows (image), 4 columns (3 images + 1 colorbar)
    gs = gridspec.GridSpec(3, 4, figure=fig,
                          width_ratios=[col_width, col_width, col_width, cbar_width_inches],
                          height_ratios=height_ratios,
                          wspace=0,
                          hspace=0,
                          left=0.0, right=right,
                          top=top, bottom=bottom)

    text_color = 'white' if background == 'black' else 'black'

    # Jacobian colormap normalization
    jac_norm = mcolors.Normalize(vmin=jac_vmin, vmax=jac_vmax)
    
    # Map follow-up images to their corresponding Jacobians
    jac_mapping = {
        'Real Follow-up Scan': 'Real Follow-up Jacobian',
        'Predicted Follow-up Scan': 'Predicted Follow-up Jacobian',
    }
    
    
    jac_im = None  # Store jacobian image for colorbar

    for row_idx, orientation in enumerate(orientations):
        idx = slice_indices[orientation]
        mask_slice = get_slice(mask, orientation, idx)

        for i, key in enumerate(expected_keys):
            ax = fig.add_subplot(gs[row_idx, i])
            data = images[key]
            slice_ = get_slice(data, orientation, idx)

            # Mask predicted scan to remove background noise
            if key == 'Predicted Follow-up Scan':
                slice_ = slice_ * (mask_slice > 0)

            # Show grayscale background
            ax.imshow(slice_, cmap=scan_cmap, aspect='equal')
            
            # Overlay Jacobian if this is a follow-up image
            if key in jac_mapping:
                jac_key = jac_mapping[key]
                jac_data = jacobians[jac_key]
                jac_slice = get_slice(jac_data, orientation, idx)
                
                # Mask values near middle (no change)
                middle_val = (jac_vmax + jac_vmin) / 2
                masked_jac = np.ma.masked_inside(jac_slice, middle_val - jac_threshold, middle_val + jac_threshold)
                
                # Also mask outside the brain
                masked_jac = np.ma.masked_where(mask_slice == 0, masked_jac)
                
                # Overlay Jacobian with high alpha for vivid colors
                jac_im = ax.imshow(masked_jac, cmap=jac_cmap, norm=jac_norm, alpha=0.9)

            # Column titles only on first row
            if row_idx == 0:
                ax.set_title(_wrap_title(key.replace(' Scan', '')), fontsize=font_size, color=text_color, pad=3)
            ax.set_facecolor(background)
            ax.axis('off')



    # Add narrow vertical colorbar on the right side spanning most of the height
    if jac_im is not None:
        # Create invisible dummy axis spanning all rows in the last column
        dummy_ax = fig.add_subplot(gs[:, 3])
        dummy_ax.axis('off')
        
        # Create narrow inset colorbar centered in the dummy axis
        cbar_ax = inset_axes(dummy_ax,
                            width=0.15,  # thin width in inches
                            height="50%",  # mostly fills the height
                            loc='center',  # Center it
                            borderpad=0)
        
        cbar = plt.colorbar(jac_im, cax=cbar_ax, orientation='vertical')
        cbar.ax.tick_params(colors=text_color, labelsize=font_size-5, width=0.5, length=3, pad=2)
        middle_val = (jac_vmax + jac_vmin) / 2
        ticks = [jac_vmin, middle_val - jac_threshold, middle_val, middle_val + jac_threshold, jac_vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:g}" for t in ticks])
        cbar.set_label('log jacobian', fontsize=font_size-4, color=text_color, labelpad=2)

    # Save figure
    plt.savefig(output_path, facecolor=background, dpi=150, pad_inches=0)
    logger.info(f"Saved Jacobian overlay figure to {output_path}")
    plt.close(fig)


def plot_segmentation(
    file_paths_dict: dict[str, str],
    output_path: str,
    logger: logging.Logger,
    slice_indices: dict[str, int] = None,
    font_size: int = 16,
    background: str = 'black',
):
    """
    Plots 2 segmentation images across 3 rows (axial, sagittal, coronal):
    GT Segmentation, Predicted Segmentation.
    No colorbar.
    
    Args:
        file_paths_dict: Dict with keys 'GT Segmentation', 'Predicted Segmentation'
        output_path: Output file path
        logger: Logger instance
        slice_indices: Dict with orientation -> slice index (axial, sagittal, coronal)
        font_size: Font size for titles
        background: Background color
    """
    
    expected_keys = ['GT Segmentation', 'Predicted Segmentation', 'Error Map']
    file_keys = ['GT Segmentation', 'Predicted Segmentation']
    
    # Verify all required files are present
    for key in file_keys:
        if key not in file_paths_dict:
            raise ValueError(f"Missing required key: {key}")

    # Load images
    images = {}
    for key in file_keys:
        if os.path.exists(file_paths_dict[key]):
            images[key] = nib.load(file_paths_dict[key]).get_fdata()
        else:
            raise FileNotFoundError(f"File not found for {key}: {file_paths_dict[key]}")
    
    # Crop arrays tightly around GT Segmentation
    coords = np.argwhere(images['GT Segmentation'] > 0)
    if len(coords) > 0:
        h0, w0, d0 = coords.min(axis=0)
        h1, w1, d1 = coords.max(axis=0)
        
        pad = 10 
        orig_H, orig_W, orig_D = list(images.values())[0].shape
        h0, h1 = max(0, h0 - pad), min(orig_H, h1 + pad + 1)
        w0, w1 = max(0, w0 - pad), min(orig_W, w1 + pad + 1)
        d0, d1 = max(0, d0 - pad), min(orig_D, d1 + pad + 1)
        
        for key in images:
            images[key] = images[key][h0:h1, w0:w1, d0:d1]
            
        if slice_indices:
            if 'axial' in slice_indices:
                slice_indices['axial'] = max(0, min(d1-d0-1, slice_indices.get('axial', orig_D//2) - d0))
            if 'sagittal' in slice_indices:
                slice_indices['sagittal'] = max(0, min(w1-w0-1, slice_indices.get('sagittal', orig_W//2) - w0))
            if 'coronal' in slice_indices:
                slice_indices['coronal'] = max(0, min(h1-h0-1, slice_indices.get('coronal', orig_H//2) - h0))
    H, W, D = list(images.values())[0].shape

    # All 3 orientations
    orientations = ['axial', 'sagittal', 'coronal']

    # Default to middle slices if not specified
    if slice_indices is None:
        slice_indices = {}
    if 'axial' not in slice_indices:
        slice_indices['axial'] = D // 2
    if 'sagittal' not in slice_indices:
        slice_indices['sagittal'] = W // 2
    if 'coronal' not in slice_indices:
        slice_indices['coronal'] = H // 2

    # Helper to extract slice
    def get_slice(data, orientation, idx):
        if orientation == 'axial':
            return np.rot90(data[:, :, idx])
        elif orientation == 'sagittal':
            return np.rot90(data[:, idx, :])
        elif orientation == 'coronal':
            return np.rot90(data[idx, :, :])
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

    # Compute slice dimensions for each orientation (after rot90)
    slice_dims = {}  # orientation -> (height_px, width_px)
    for o in orientations:
        if o == 'axial':      # data[:,:,idx] -> (H,W) -> rot90 -> (W,H)
            slice_dims[o] = (W, H)
        elif o == 'sagittal':  # data[:,idx,:] -> (H,D) -> rot90 -> (D,H)
            slice_dims[o] = (D, H)
        elif o == 'coronal':   # data[idx,:,:] -> (W,D) -> rot90 -> (D,W)
            slice_dims[o] = (D, W)

    # All columns show same-shaped images per row, so column width = max width across rows
    max_width_px = max(sd[1] for sd in slice_dims.values())

    # Row heights in pixels (the height of each row's images)
    row_height_px = [slice_dims[o][0] for o in orientations]

    # Figure width: 3 columns (segmentation), each col_width inches
    scale = 0.015  # inches per pixel
    col_width = max_width_px * scale
    fig_width = 3 * col_width

    # Figure height: sum of row heights (in inches) + minimal margins (no colorbar)
    top_margin_inches = 0.60   # Tight top margin for multiline titles
    bottom_margin_inches = 0.05  # Minimal bottom margin for tight fit
    grid_height = sum(rh * scale for rh in row_height_px)
    fig_height = grid_height + top_margin_inches + bottom_margin_inches

    top = 1 - top_margin_inches / fig_height
    bottom = bottom_margin_inches / fig_height

    # Height ratios: proportional to pixel heights for image rows
    height_ratios = row_height_px

    # Create figure with specified background
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=background)

    # Create GridSpec: 3 rows, 3 columns (no colorbar row)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                          width_ratios=[1, 1, 1],
                          height_ratios=height_ratios,
                          wspace=0,
                          hspace=0,
                          left=0.0, right=1.0,
                          top=top, bottom=bottom)

    text_color = 'white' if background == 'black' else 'black'
    
    # Colormap for segmentation (black to white)
    seg_cmap = 'gray'

    for row_idx, orientation in enumerate(orientations):
        idx = slice_indices[orientation]

        for i, key in enumerate(expected_keys):
            ax = fig.add_subplot(gs[row_idx, i])
            if key == 'Error Map':
                gt_slice = get_slice(images['GT Segmentation'], orientation, idx)
                pred_slice = get_slice(images['Predicted Segmentation'], orientation, idx)
                
                diff_slice = np.zeros_like(gt_slice)
                correct_mask = (gt_slice == pred_slice) & (gt_slice > 0)
                error_mask = (gt_slice != pred_slice) & ((gt_slice > 0) | (pred_slice > 0))
                
                diff_slice[correct_mask] = 1 # Black for correct
                diff_slice[error_mask] = 2   # Red for error
                
                diff_cmap = mcolors.ListedColormap(['black', 'black', '#FF0000'])
                ax.imshow(diff_slice, cmap=diff_cmap, aspect='equal', vmin=0, vmax=2)
            else:
                data = images[key]
                slice_ = get_slice(data, orientation, idx)

                # Mask predicted segmentation to remove background noise
                if key == 'Predicted Segmentation':
                    gt_slice = get_slice(images['GT Segmentation'], orientation, idx)
                    slice_ = slice_ * (gt_slice > 0)

                # Display segmentation
                ax.imshow(slice_, cmap=seg_cmap, aspect='equal', vmin=0, vmax=3)

            # Column titles only on first row
            if row_idx == 0:
                title = 'Real Follow-up' if key == 'GT Segmentation' else 'Predicted Follow-up' if key == 'Predicted Segmentation' else key
                ax.set_title(_wrap_title(title), fontsize=font_size, color=text_color, pad=3)
            ax.set_facecolor(background)
            ax.axis('off')



    # Save figure
    plt.savefig(output_path, facecolor=background, dpi=150, pad_inches=0)
    logger.info(f"Saved segmentation figure to {output_path}")
    plt.close(fig)

def get_plot_datalist(
    datalist: list[dict],
    run_dir: str,
    args: Namespace,
    mode: str = "test",
) -> list[dict]:
    if mode == "train":
        level = ""
    else:
        level = getattr(args, "level", "original")

    pred_dir = Path(run_dir) / "predictions" / level
    jac_dir  = Path(run_dir) / "predictions" / level / "jacobians"
    seg_dir  = Path(run_dir) / "predictions" / level / "segmentations"
    plot_datalist = []
    for sample in datalist:
        session_dir = Path(sample["T1"]).parent
        patient_dir = session_dir.parent

        for mod_name in ["T1", "T1C", "FLAIR"]:
            if mod_name not in sample:
                continue

            gt_path = Path(sample[mod_name])
            #relative_path = gt_path.relative_to(Path(args.data_base_dir))
            relative_path = gt_path.resolve().relative_to(Path(args.data_base_dir).resolve())
            #pred_path = pred_dir / relative_path
            pred_path = pred_dir / relative_path.parent / f"{mod_name}.nii.gz"

            stem = relative_path.stem.replace(".nii", "")
            jac_subdir = jac_dir / relative_path.parent
            seg_subdir = seg_dir / relative_path.parent


            plot_datalist.append({
                "mod_name": mod_name,
                "pred": str(pred_path),
                "gt": str(gt_path),
                f"baseline_{mod_name}": sample[f"baseline_{mod_name}"],
                #"baseline_T1c": sample["baseline_T1c"],
                #"baseline_FLAIR": sample["baseline_FLAIR"],
                "dose": sample["dose"],
                "fu_time": sample.get("fu_time", 0),
                #"brain_mask": str(session_dir / args.brain_mask_filename),
                #"brain_mask": None,
                "brain_mask": str(session_dir / "brain_mask.nii.gz"),
                #"roi_mask": sample["roi_mask"],
                "gt_jac": str(jac_subdir / f"{stem}_gt_jac.nii.gz"),
                "pred_jac": str(jac_subdir / f"{stem}_pred_jac.nii.gz"),
                "gt_seg": str(seg_subdir / f"{stem}_gt_seg.nii.gz"),
                "pred_seg": str(seg_subdir / f"{stem}_pred_seg.nii.gz"),
            })

    return plot_datalist


def plot_predictions(
    dataloader,
    run_dir: str,
    args: Namespace,
    num_samples: int = None,
    mode: str = "train",
    logger: logging.Logger = None,
) -> list[dict[str, str]]:
    datalist = dataloader.dataset.data
    if num_samples:
        datalist = datalist[:num_samples]

    plot_datalist = get_plot_datalist(datalist, run_dir, args, mode = mode)
    run_dir = Path(run_dir)
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Making plots for {len(plot_datalist)} predictions")

    # train: predictions/sub-.../ses-.../anat/T1.nii.gz
    # test:  predictions/original/sub-.../ses-.../anat/T1.nii.gz
    pred_root = run_dir / "predictions" 
    plot_paths_list = []
    for i, sample in enumerate(plot_datalist):
        #for mod_name in ["T1", "T1c", "FLAIR"]:

        mod_name = sample["mod_name"]
        logger.info(f"Making plots for prediction {mod_name}... ({i + 1}/{len(plot_datalist)})")

        pred_path = Path(sample["pred"])
        
        #relative_path = pred_path.relative_to(pred_dir := Path(run_dir) / "predictions")
        try:
            relative_path = pred_path.relative_to(pred_root)
        except ValueError:
            raise ValueError(
                f"Predicted path is not under expected predictions root.\n"
                f"pred_path: {pred_path}\n"
                f"pred_root: {pred_root}\n"
                f"mode: {mode}"
            )
        plot_subdir = plot_dir / relative_path.parent
        plot_subdir.mkdir(exist_ok=True, parents=True)

        # Include modality in filename to avoid overwriting
        base_name = str(plot_dir / relative_path).replace(".nii.gz", "")
        plot_paths = {
            "middle_slices": f"{base_name}_{mod_name}.png",
            "tumor_slices":  f"{base_name}_{mod_name}_t.png",
        }


        file_paths_dict = {
            f"Baseline Scan ({mod_name})": sample[f"baseline_{mod_name}"],
            "Dose Map": sample["dose"],
            f"Real Follow-up ({mod_name})": sample["gt"],
            f"Predicted Follow-up ({mod_name})": sample["pred"],
        }

   
        plot_prediction(
            file_paths_dict=file_paths_dict,
            mask_path=sample["brain_mask"],
            #mask_path = sample["brain_mask"] if sample.get("brain_mask") else None,
            output_path=plot_paths["middle_slices"],
            logger=logger,
            mod_name=mod_name
        )

       
        plot_prediction(
            file_paths_dict=file_paths_dict,
            mask_path=sample["brain_mask"],
            #mask_path=sample["brain_mask"] if sample.get("brain_mask") else None,
            output_path=plot_paths["tumor_slices"],
            #slice_indices={"axial": k},
            logger=logger,
            mod_name=mod_name
    )

        # Jacobian plots
        if Path(sample["gt_jac"]).exists() and Path(sample["pred_jac"]).exists():
            logger.info("Jacobian files found, generating Jacobian plots...")
            plot_paths["middle_slices_jac"] = f"{base_name}_{mod_name}_jac.png"
            plot_paths["tumor_slices_jac"]  = f"{base_name}_{mod_name}_t_jac.png"

            """
            jac_file_paths_dict = {
                **file_paths_dict,
                "Real Follow-up Jacobian": sample["gt_jac"],
                "Predicted Follow-up Jacobian": sample["pred_jac"],
            }

            """

            jac_file_paths_dict = {
                "Baseline Scan": sample[f"baseline_{mod_name}"],
                "Real Follow-up Scan": sample["gt"],
                "Predicted Follow-up Scan": sample["pred"],
                "Real Follow-up Jacobian": sample["gt_jac"],
                "Predicted Follow-up Jacobian": sample["pred_jac"],
            }


            plot_jacobian(file_paths_dict=jac_file_paths_dict, mask_path=sample["brain_mask"],
                          output_path=plot_paths["middle_slices_jac"], logger=logger)
            plot_jacobian(file_paths_dict=jac_file_paths_dict, mask_path=sample["brain_mask"],
                          output_path=plot_paths["tumor_slices_jac"], #slice_indices={"axial": k},
                            logger=logger)
        else:
            logger.info("Jacobian files not found, skipping Jacobian plots")

        # Segmentation plots
        if Path(sample["gt_seg"]).exists() and Path(sample["pred_seg"]).exists():
            logger.info("Segmentation files found, generating segmentation plots...")
            plot_paths["middle_slices_seg"] = f"{base_name}_{mod_name}_seg.png"
            plot_paths["tumor_slices_seg"]  = f"{base_name}_{mod_name}_t_seg.png"

            seg_file_paths_dict = {
                "GT Segmentation": sample["gt_seg"],
                "Predicted Segmentation": sample["pred_seg"],
            }

            plot_segmentation(file_paths_dict=seg_file_paths_dict,
                              output_path=plot_paths["middle_slices_seg"], logger=logger)
            plot_segmentation(file_paths_dict=seg_file_paths_dict,
                              output_path=plot_paths["tumor_slices_seg"], #slice_indices={"axial": k}, 
                              logger=logger)
        else:
            logger.info("Segmentation files not found, skipping segmentation plots")

        plot_paths_list.append(plot_paths)

    return plot_paths_list    

