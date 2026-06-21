import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import AxesGrid


def np_img_to_slices(np_img):
    """
    Convert np array to three np arrays, the axial, coronal, and sagittal views (tuple in that order)
    np_img: np array (H, W, D)
    """
    d, h, w = np_img.shape
    coronal = np.rot90(np_img[d // 2, :, :])
    sagittal = np.rot90(np_img[:, h // 2, :])
    axial = np.rot90(np_img[:, :, w // 2])
    return axial, coronal, sagittal


def get_roi_mask_center_indices(roi_mask: torch.Tensor) -> list[int, int, int]:
    """
    Get the center indices of a 3D mask.
    
    Args:
        roi_mask (torch.Tensor): A 3D torch tensor representing the mask.
        
    Returns:
        tuple[int, int, int]: The center indices (z, y, x).
    """
    roi_mask = roi_mask.cpu().numpy()
    mask_coords = np.argwhere(roi_mask > 0)
    if len(mask_coords) == 0:
        print("WARNING! The mask is empty, no non-zero elements found.")
        return list(np.array(roi_mask.shape) // 2)
    
    center_of_mass = mask_coords.mean(axis=0).astype(int)
    return center_of_mass.tolist()


def plot_nifti_columns(
    nifti_paths_dict: dict[str, str],
    vmin_vmax: tuple[tuple[float, float]],
    output_path: str,
    slice_indices: dict[str, int] = None,
    font_size: int = 12,
    cmap: str = 'gray',
    background: str = 'black',
    clip_range: tuple[float, float] = None,
    colorbar: bool = False,
    logger=None,
):
    """
    Plot each NIfTI file as a column of slices based on specified orientations.
    
    Args:
        nifti_paths_dict: e.g. {"Original": "path/to/gt.nii", "Reconstructed": "path/to/pred.nii"}
        vmin_vmax: tuple of (vmin, vmax) for image intensity scaling e.g. ((None, None), (0.2, 1.8))
        output_path: output figure path
        slice_indices: dict with keys from {'axial', 'sagittal', 'coronal'} specifying slice indices
                     If None, defaults to all three orientations with middle slices
        font_size: font size for titles
        cmap: colormap for the images
        background: background color for the figure
        clip_range: optional (min, max) to clip image values
        colorbar: if True, adds colorbar at the end of each row
        logger: optional logger instance
    """
    n_cols = len(nifti_paths_dict)
    text_color = 'white' if background == 'black' else 'black'
    
    # Load images
    data_list = [nib.load(p).get_fdata() for p in nifti_paths_dict.values()]
    H, W, D = data_list[0].shape
    
    # Default to all orientations if not specified
    if slice_indices is None:
        slice_indices = {'axial': D // 2, 'sagittal': W // 2, 'coronal': H // 2}
    else:
        defaults = {'axial': D // 2, 'sagittal': W // 2, 'coronal': H // 2}
        for k, v in slice_indices.items():
            if v is None:
                slice_indices[k] = defaults[k]
    
    orientations = list(slice_indices.keys())
    n_rows = len(orientations)
    
    # Figure size
    scale = 0.03
    max_slice_width = max([H, W, D])
    fig_width = n_cols * max_slice_width * scale
    fig_height = n_rows * max_slice_width * scale + 2
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=background)
    
    # Create AxesGrid
    grid_kwargs = {'nrows_ncols': (n_rows, n_cols), 'axes_pad': 0.05}
    if colorbar:
        grid_kwargs.update({'cbar_location': 'right', 'cbar_mode': 'edge', 
                           'cbar_size': '5%', 'cbar_pad': '2%'})
    grid = AxesGrid(fig, 111, **grid_kwargs)
    
    # Plot images
    for col, data in enumerate(data_list):
        col_label = list(nifti_paths_dict.keys())[col]
        grid[col].set_title(col_label, color=text_color, pad=10, fontsize=font_size)
        
        for row, orientation in enumerate(orientations):
            ax_idx = row * n_cols + col
            
            if orientation == 'axial':
                slice_ = np.rot90(data[:, :, slice_indices[orientation]])
            elif orientation == 'sagittal':
                slice_ = np.rot90(data[:, slice_indices[orientation], :])
            elif orientation == 'coronal':
                slice_ = np.rot90(data[slice_indices[orientation], :, :])
            else:
                raise ValueError(f"Unknown orientation: {orientation}")
            
            if clip_range is not None:
                slice_ = np.clip(slice_, clip_range[0], clip_range[1])

            vmin, vmax = vmin_vmax[col]
            im = grid[ax_idx].imshow(slice_, cmap=cmap, vmin=vmin, vmax=vmax)
            grid[ax_idx].set_facecolor(background)
            grid[ax_idx].axis('off')
            
            # Add colorbar for the last column of each row
            if colorbar and col == n_cols - 1:
                tick_vals = np.linspace(vmin, vmax, 5).tolist()
                cbar = grid.cbar_axes[row].colorbar(im, ticks=tick_vals)
                cbar.ax.set_yticklabels([f"{val:.1f}" for val in tick_vals], 
                                        color=text_color, fontsize=font_size)
    
    plt.savefig(output_path, facecolor=background, bbox_inches='tight')
    if logger:
        logger.info(f"Saved NIfTI columns plot to {output_path})")
    else:
        print(f"Saved NIfTI columns plot to {output_path}")
    plt.close(fig)