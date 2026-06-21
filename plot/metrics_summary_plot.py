"""
Box plot generation for metrics comparison.

Usage:
    python -m plot.metrics_summary_plot --metrics_csv /path/to/metrics.csv --output_dir /path/to/output
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

def style_boxplot(bp, color: str = '#1E3A8A'):
    """Apply consistent styling to a boxplot."""
    for box in bp['boxes']:
        box.set_facecolor(color)
        box.set_alpha(0.85)
        box.set_edgecolor('black')
        box.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    for mean in bp['means']:
        mean.set_color('black')
        mean.set_linewidth(1.5)
        mean.set_linestyle('--')
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='gray', alpha=0.5, markersize=4)


def style_axis(ax):
    """Apply consistent styling to an axis."""
    ax.grid(True, alpha=0.4, linewidth=0.8)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove mean/std summary rows from dataframe."""
    #return df[~df['pred_path'].isin(['mean', 'std'])]
    return df[~df['pred_path'].str.contains('mean|std', na=False)]


def bin_data_by_time(df: pd.DataFrame, metric: str, bin_edges: np.ndarray) -> list:
    """Bin data by follow-up time."""
    binned_data = []
    for i in range(len(bin_edges) - 1):
        is_last = i == len(bin_edges) - 2
        if is_last:
            mask = (df['fu_time'] >= bin_edges[i]) & (df['fu_time'] <= bin_edges[i+1])
        else:
            mask = (df['fu_time'] >= bin_edges[i]) & (df['fu_time'] < bin_edges[i+1])
        binned_data.append(df[mask][metric].values)
    return binned_data


def plot_metrics_by_time(
    metrics_df: pd.DataFrame,
    metric_names: list[str],
    output_path: str,
    title: str = "Metrics by Follow-up Time",
    color: str = '#1E3A8A',
):
    """Create box plots of metrics binned by follow-up time. One image per metric."""
    plt.style.use('seaborn-v0_8-whitegrid')
    metrics_df = filter_df(metrics_df)

    # Automatic binning by fu_time
    all_fu_times = metrics_df['fu_time'].values
    n_bins = min(4, max(3, len(all_fu_times) // 25))
    

    bin_edges = np.unique(np.quantile(all_fu_times, np.linspace(0, 1, n_bins + 1)))
    if len(bin_edges) < 2:
        bin_edges = np.array([all_fu_times.min(), all_fu_times.max()])
    bin_edges[0]  -= 1e-6
    bin_edges[-1] += 1e-6
    bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(bin_edges) - 1)]

    output_base = Path(output_path).with_suffix('')  # strip extension

    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(7, 8))
        fig.patch.set_facecolor('white')
        fig.suptitle(f"{title} — {metric.upper().replace('_', ' ')}", fontsize=16, fontweight='bold', y=0.98)

        #binned_data = bin_data_by_time(metrics_df, metric, bin_edges)
        tmp = metrics_df.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric])
        binned_data = bin_data_by_time(tmp, metric, bin_edges)

        valid_bins = [(j, data) for j, data in enumerate(binned_data) if len(data) > 0]
        if not valid_bins:
            print(f"No data for metric: {metric}, skipping.")
            plt.close()
            continue

        valid_indices, valid_data = zip(*valid_bins)
        valid_labels = [bin_labels[j] for j in valid_indices]
        positions = np.arange(len(valid_data)) * 2.0 + 0.75

        bp = ax.boxplot(valid_data, positions=positions, widths=0.8,
                        patch_artist=True, showmeans=True, meanline=True)
        style_boxplot(bp, color)

        ax.set_title(metric.upper().replace('_', ' '), fontweight='bold', fontsize=16, pad=20)
        ax.set_xticks(positions)
        ax.set_xticklabels(valid_labels, fontsize=14, fontweight='bold')
        ax.set_xlabel('Days after treatment', fontsize=14, fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', labelsize=14)
        style_axis(ax)

        plt.tight_layout()
        metric_output_path = f"{output_base}_{metric}.png"
        plt.savefig(metric_output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved plot to {metric_output_path}")

def plot_dice_by_time(metrics_df: pd.DataFrame, output_dir: str, color: str = '#1E3A8A', level: str = None):
    """Plot Dice metrics by follow-up time, one plot per modality."""
    available = [m for m in metrics_df.columns if 'dice' in m and 'mean' not in m and 'std' not in m]
    if not available:
        print("No Dice metrics found in DataFrame")
        return
    level_str = f" — {level}" if level else "Original"

    for mod in metrics_df['modality'].unique():
        mod_df = metrics_df[metrics_df['modality'] == mod]
        plot_metrics_by_time(
            metrics_df=mod_df,
            metric_names=available,
            output_path=str(Path(output_dir) / f'dice_by_time_{mod}{"_" + level if level else ""}.png'),
            title=f'Dice Coefficient by Follow-up Time{level_str} — {mod}',
            color=color,
        )


def plot_image_quality_by_time(metrics_df: pd.DataFrame, output_dir: str, color: str = '#1E3A8A', level: str = None):
    """Plot image quality metrics by follow-up time, one plot per modality."""
    available = [m for m in metrics_df.columns if any(q in m for q in ['mse', 'psnr', 'ssim'])]
    if not available:
        print("No image quality metrics found in DataFrame")
        return
    level_str = f" — {level}" if level else "Original"

    for mod in metrics_df['modality'].unique():
        mod_df = metrics_df[metrics_df['modality'] == mod]
        plot_metrics_by_time(
            metrics_df=mod_df,
            metric_names=available,
            output_path=str(Path(output_dir) / f'image_quality_by_time_{mod}{"_" + level if level else ""}.png'),
            title=f'Image Quality Metrics by Follow-up Time{level_str} — {mod}',
            color=color,
        )

def plot_cwm_by_time(metrics_df: pd.DataFrame, output_dir: str):
    """Plot CWM volume (GT vs predicted) against follow-up time.
    
    Creates one plot per patient. Each plot shows left (green) and right (red)
    hippocampus volumes, with solid lines for GT and dashed lines for predicted.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    df = filter_df(metrics_df)
    print("--------------")
    print("DF")
    print(metrics_df)
    print("--------------")
    print("--------------")
    print("FILTERED DF")
    print(df)
    print("--------------")
    
    # Check required columns exist
    required = ['fu_time', 'pred_cwm_vol_ml', 'gt_cwm_vol_ml']
    available = [c for c in required if c in df.columns]
    print("--------------")
    print(" AVAILABLE")
    print(available)
    print("--------------")
    if len(available) < len(required):
        missing = set(required) - set(available)
        print(f"Missing columns for cwm volume plot: {missing}")
        return

    # Extract patient ID and run dir from pred_path
    df = df.copy()
    
    def get_info(p):
        path = Path(p)
        print("==================--")
        print("PATH")
        print(path)
        print("====================")
        try:
            pred_idx = path.parts.index('predictions')
            run_dir = Path(*path.parts[:pred_idx])
            patient_id = path.parts[pred_idx + 1]
            return pd.Series([patient_id, run_dir])
        except ValueError:
            return pd.Series([path.parent.parent.name, path.parents[3]])
            
    df[['patient', 'run_dir']] = df['pred_path'].apply(get_info)
      

    
    error_summary = []
    
    for patient_id, patient_df in df.groupby('patient'):
        patient_df = patient_df.sort_values('fu_time')
        fu = patient_df['fu_time'].values
        
        # Calculate errors
        error = patient_df['pred_cwm_vol_ml'] - patient_df['gt_cwm_vol_ml']
        #err_right = patient_df['hippo_vol_right_pred'] - patient_df['hippo_vol_right_gt']
        rel_err = (error / patient_df['gt_cwm_vol_ml']) * 100
        #rel_err_right = (err_right / patient_df['hippo_vol_right_gt']) * 100
        patient_errors = pd.DataFrame({
            'patient': patient_id,
            'fu_time': fu,
            'abs_diff': error.abs(),
            'rel_diff_%': rel_err.abs(),
  
        })
        error_summary.append(patient_errors)
        
        run_dir = patient_df['run_dir'].iloc[0]
        #left_txt = run_dir / 'segmentations' / patient_id / 'hippo-vol-left.txt'
        #right_txt = run_dir / 'segmentations' / patient_id / 'hippo-vol-right.txt'
        txt = run_dir / 'segmentations' / patient_id / 'cwm_vol.txt'
        
        #base_left, base_right = None, None
        base = None
        #if txt.exists() and right_txt.exists():
        if txt.exists():
            base = float(txt.read_text().strip())
            #base_right = float(right_txt.read_text().strip())
            
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'CWM Volumes — {patient_id}', fontsize=16, fontweight='bold')
        
        # GT Arrays
        # ['fu_time', 'pred_cwm_vol_ml', 'gt_cwm_vol_ml']
        if base is not None:
            fu_gt = np.insert(fu, 0, 0)
            vol_gt = np.insert(patient_df['gt_cwm_vol_ml'].values, 0, base)
            #vol_right_gt = np.insert(patient_df['hippo_vol_right_gt'].values, 0, base_right)
        else:
            fu_gt = fu
            vol_gt = patient_df['gt_cwm_vol_ml'].values
            #vol_right_gt = patient_df['hippo_vol_right_gt'].values
        
   
        ax.plot(fu_gt, vol_gt,
                linestyle='-', linewidth=2, marker='o', markersize=5, label='GT')
        ax.plot(fu, patient_df['pred_cwm_vol_ml'].values,
                linestyle='--', linewidth=2, marker='x', markersize=6, label='Pred')
        
       
        ax.set_xlabel('Days after treatment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volume (mm³)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, framealpha=0.9)
        ax.set_xlim(left=0)
        ax.tick_params(labelsize=11)
        style_axis(ax)
        
        plt.tight_layout()
        output_path = str(Path(output_dir) / f'cwm_vol_{patient_id}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved CWM volume plot to {output_path}")

    # Save error summary table
    if error_summary:
        error_df = pd.concat(error_summary)
        error_df = error_df[['patient', 'fu_time', 'abs_diff', 'rel_diff_%']]
        err_out = Path(output_dir) / 'cwm_volume_differences.csv'
        error_df.to_csv(err_out, index=False, float_format="%.2f")
        print(f"Saved CWM volume differences summary to {err_out}")

        # Save mean error summary table per patient
        mean_error_df = error_df.drop(columns=['fu_time']).groupby('patient').mean().reset_index()
        mean_err_out = Path(output_dir) / 'cwm_volume_differences_mean.csv'
        mean_error_df.to_csv(mean_err_out, index=False, float_format="%.2f")
        print(f"Saved mean CWM volume differences summary to {mean_err_out}")


def plot_volumes_by_time(metrics_df: pd.DataFrame, output_dir: str, level: str = None):
    """Plot volumes (GT vs predicted) per segment per patient, one plot per modality."""
    plt.style.use('seaborn-v0_8-whitegrid')

    for mod in metrics_df['modality'].unique():
        df = filter_df(metrics_df[metrics_df['modality'] == mod])

        pred_cols = [c for c in df.columns if c.startswith('pred_') and c.endswith('_vol_ml')]
        segments  = [c.replace('pred_', '').replace('_vol_ml', '') for c in pred_cols]
        segments  = [s for s in segments if f'gt_{s}_vol_ml' in df.columns]

        if not segments:
            print(f"No volume columns found for modality {mod}")
            continue

        print(f"Found segments for {mod}: {segments}")

        df = df.copy()

        def get_info(p):
            path = Path(p)
            try:
                pred_idx = path.parts.index('predictions')
                run_dir = Path(*path.parts[:pred_idx])
                patient_id = path.parts[pred_idx + 1]
                return pd.Series([patient_id, run_dir])
            except ValueError:
                return pd.Series([path.parent.parent.name, path.parents[3]])

        df[['patient', 'run_dir']] = df['pred_path'].apply(get_info)

        for patient_id, patient_df in df.groupby('patient'):
            patient_df = patient_df.sort_values('fu_time')
            fu = patient_df['fu_time'].values

            for seg in segments:
                pred_col = f'pred_{seg}_vol_ml'
                gt_col   = f'gt_{seg}_vol_ml'

                gt_vals   = patient_df[gt_col].values
                pred_vals = patient_df[pred_col].values

                txt = Path(patient_df['run_dir'].iloc[0]) / 'segmentations' / patient_id / f'{seg}_vol.txt'
                if txt.exists():
                    base   = float(txt.read_text().strip())
                    fu_gt  = np.insert(fu, 0, 0)
                    vol_gt = np.insert(gt_vals, 0, base)
                else:
                    fu_gt  = fu
                    vol_gt = gt_vals

                fig, ax = plt.subplots(figsize=(7, 5))
                fig.patch.set_facecolor('white')
                fig.suptitle(f'{seg.replace("_", " ").title()} Volume — {patient_id} ({mod})',
                             fontsize=14, fontweight='bold')

                
                # Discrete Points
                ax.plot(fu_gt, vol_gt, linestyle='None', linewidth=2, marker='o', markersize=5, label='GT', color='blue')
                ax.plot(fu, pred_vals, linestyle='None', linewidth=2, marker='x', markersize=6, label='Pred', color='red')



                ax.set_xlabel('Days after treatment', fontsize=10)
                ax.set_ylabel('Volume (mL)', fontsize=10)
                ax.legend(fontsize=10, framealpha=0.9)
                ax.set_xlim(left=0)
                ax.tick_params(labelsize=10)
                style_axis(ax)

                plt.tight_layout()
                output_path = str(Path(output_dir) / f'volumes_{patient_id}_{seg}_{mod}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"Saved volume plot to {output_path}")

def plot_volumes_combined(all_results: dict, output_dir: str):
    """
    Plot GT (once) + predicted volumes for each level on the same graph.
    One plot per modality per patient per segment — mirrors plot_volumes_by_time structure.

    all_results: dict mapping level_name -> results_df
                 e.g. {'original': df, 'level_1': df, 'level_2': df}
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    LEVEL_COLORS = {
        'original': 'red',
        'level_1':  'green',
        'level_2':  'orange',
    }
    LEVEL_LABELS = {
        'original': 'Pred (original)',
        'level_1':  'Pred (level 1)',
        'level_2':  'Pred (level 2)',
    }

    def get_info(p):
        path = Path(p)
        try:
            pred_idx   = path.parts.index('predictions')
            run_dir    = Path(*path.parts[:pred_idx])
            patient_id = path.parts[pred_idx + 1]
            return pd.Series([patient_id, run_dir])
        except ValueError:
            return pd.Series([path.parent.parent.name, path.parents[3]])

    first_df_raw = next(iter(all_results.values()))

    # loop per modality, same as plot_volumes_by_time
    for mod in first_df_raw['modality'].unique():

        # Build per-level filtered dfs for this modality
        level_dfs = {}
        for level_name, df_raw in all_results.items():
            df = filter_df(df_raw[df_raw['modality'] == mod]).copy()
            df[['patient', 'run_dir']] = df['pred_path'].apply(get_info)
            level_dfs[level_name] = df

        first_df  = next(iter(level_dfs.values()))
        pred_cols = [c for c in first_df.columns if c.startswith('pred_') and c.endswith('_vol_ml')]
        segments  = [c.replace('pred_', '').replace('_vol_ml', '') for c in pred_cols]
        segments  = [s for s in segments if f'gt_{s}_vol_ml' in first_df.columns]

        if not segments:
            print(f"[combined] No volume columns for modality {mod}")
            continue

        print(f"[combined] Found segments for {mod}: {segments}")

        all_patients = first_df['patient'].unique()

        # loop per patient per segment, same as plot_volumes_by_time
        for patient_id in all_patients:
            for seg in segments:
                pred_col = f'pred_{seg}_vol_ml'
                gt_col   = f'gt_{seg}_vol_ml'

                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('white')
                fig.suptitle(
                    f'{seg.replace("_", " ").title()} Volume — {patient_id} ({mod}) — All Levels',
                    fontsize=13, fontweight='bold'
                )

                gt_plotted = False

                for level_name, df in level_dfs.items():
                    patient_df = df[df['patient'] == patient_id].sort_values('fu_time')
                    if patient_df.empty or pred_col not in patient_df.columns:
                        continue

                    fu        = patient_df['fu_time'].values
                    pred_vals = patient_df[pred_col].values
                    gt_vals   = patient_df[gt_col].values

                    # GT plotted only once (identical across levels)
                    if not gt_plotted:
                        txt = Path(patient_df['run_dir'].iloc[0]) / 'segmentations' / patient_id / f'{seg}_vol.txt'
                        if txt.exists():
                            base   = float(txt.read_text().strip())
                            fu_gt  = np.insert(fu, 0, 0)
                            vol_gt = np.insert(gt_vals, 0, base)
                        else:
                            fu_gt  = fu
                            vol_gt = gt_vals

                        ax.plot(fu_gt, vol_gt, linestyle='None', marker='o',
                                markersize=6, label='GT', color='blue', zorder=5)
                        gt_plotted = True

                    color = LEVEL_COLORS.get(level_name, 'gray')
                    label = LEVEL_LABELS.get(level_name, f'Pred ({level_name})')
                    ax.plot(fu, pred_vals, linestyle='None', marker='x',
                            markersize=7, label=label, color=color)

                ax.set_xlabel('Days after treatment', fontsize=10)
                ax.set_ylabel('Volume (mL)', fontsize=10)
                ax.legend(fontsize=9, framealpha=0.9)
                ax.set_xlim(left=0)
                ax.tick_params(labelsize=10)
                style_axis(ax)

                plt.tight_layout()
                out = Path(output_dir) / f'volumes_combined_{patient_id}_{seg}_{mod}.png'
                plt.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"Saved combined volume plot to {out}")



def plot_all_metrics_summary(metrics_df: pd.DataFrame, output_dir: str, color: str = '#1E3A8A'):
    """Plot summary box plots for all metrics (not binned by time)."""
    plt.style.use('seaborn-v0_8-whitegrid')
    metrics_df = filter_df(metrics_df)
    
    # Get all numeric columns except fu_time and pred_path
    metric_cols = [c for c in metrics_df.columns 
                   if c not in ['pred_path', 'fu_time'] and metrics_df[c].dtype in ['float64', 'int64']]
    if not metric_cols:
        print("No metric columns found")
        return
    
    n_metrics = len(metric_cols)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor('white')
    fig.suptitle('Metrics Summary', fontsize=20, fontweight='bold', y=1.02)
    
    # Normalize axes to 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        data = metrics_df[metric].dropna().values
        if len(data) == 0:
            continue
        
        bp = ax.boxplot([data], positions=[1], widths=0.6, 
                        patch_artist=True, showmeans=True, meanline=True)
        style_boxplot(bp, color)
        
        mean_val = np.mean(data)
        ax.text(1, mean_val, f'{mean_val:.4f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        ax.set_title(metric.upper().replace('_', ' '), fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=10)
        style_axis(ax)
    
    # Hide unused axes
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / 'metrics_summary.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved summary plot to {Path(output_dir) / 'metrics_summary.png'}")


def plot_metrics_by_time_multilevel(
    metrics_by_level: dict,
    metric_names: list,
    output_path: str,
    title: str = "Metrics by Follow-up Time",
):
    plt.style.use('seaborn-v0_8-whitegrid')

    LEVEL_COLORS = {
        "gt":       "#2ca02c",
        "original": "#1E3A8A",
        "level_1":  "#4e79a7",
        "level_2":  "#f28e2b",
        "level_3":  "#e15759",
        "level_4":  "#9467bd",
    }

    # Get all modalities from the first df
    first_df_all = filter_df(next(iter(metrics_by_level.values())))
    modalities = first_df_all['modality'].unique()

    output_base = Path(output_path).with_suffix('')
    level_names = list(metrics_by_level.keys())
    n_levels = len(level_names)
    width = 0.6
    group_spacing = n_levels * width + 1.0

    for mod in modalities:
        # Filter all levels to this modality
        mod_by_level = {
            lvl: df[df['modality'] == mod]
            for lvl, df in metrics_by_level.items()
        }

        # Compute bins from this modality's data
        first_df = filter_df(next(iter(mod_by_level.values())))
        all_fu_times = first_df['fu_time'].values
        n_bins = min(4, max(3, len(all_fu_times) // 25))
        n_bins = min(n_bins, len(np.unique(all_fu_times)))
        bin_edges = np.unique(np.quantile(all_fu_times, np.linspace(0, 1, n_bins + 1)).astype(int))
        if len(bin_edges) < 2:
            bin_edges = np.array([int(all_fu_times.min()), int(all_fu_times.max()) + 1])
        bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges) - 1)]

        for metric in metric_names:
            fig, ax = plt.subplots(figsize=(max(10, n_levels * len(bin_labels) * 1.2), 7))
            fig.patch.set_facecolor('white')
            fig.suptitle(f"{title} — {mod} — {metric.upper().replace('_', ' ')}",
                         fontsize=16, fontweight='bold', y=1.01)

            tick_positions = []

            for bin_idx in range(len(bin_labels)):
                group_start = bin_idx * group_spacing
                bin_positions = []

                for lvl_idx, level_name in enumerate(level_names):
                    df = filter_df(mod_by_level[level_name])
                    binned = bin_data_by_time(df, metric, bin_edges)
                    data = binned[bin_idx] if bin_idx < len(binned) else []

                    pos = group_start + lvl_idx * width
                    bin_positions.append(pos)

                    if len(data) == 0:
                        continue

                    color = LEVEL_COLORS.get(level_name, "#aaaaaa")
                    bp = ax.boxplot(
                        [data], positions=[pos], widths=width * 0.8,
                        patch_artist=True, showmeans=True, meanline=True,
                        manage_ticks=False,
                    )
                    style_boxplot(bp, color)

                tick_positions.append(np.mean(bin_positions))

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(bin_labels, fontsize=13, fontweight='bold')
            ax.set_xlabel('Days after treatment', fontsize=13, fontweight='bold', labelpad=10)
            ax.set_ylabel(metric.upper().replace('_', ' '), fontsize=13)
            ax.tick_params(axis='y', labelsize=12)
            style_axis(ax)

            legend_handles = [
                mpatches.Patch(color=LEVEL_COLORS.get(lvl, "#aaaaaa"), label=lvl)
                for lvl in level_names
            ]
            ax.legend(handles=legend_handles, fontsize=11, framealpha=0.9,
                      loc='upper right', title='Level', title_fontsize=11)

            plt.tight_layout()
            out = f"{output_base}_{mod}_{metric}.png"
            plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved multilevel plot to {out}")




def main():
    parser = argparse.ArgumentParser(description='Generate box plots from metrics CSV')
    parser.add_argument('--metrics_csv', type=str, required=True, help='Path to metrics CSV file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    parser.add_argument('--color', type=str, default='#1E3A8A', help='Color for box plots')
    parser.add_argument('--level', type=str, default=None, help='Level')
    args = parser.parse_args()
    
    metrics_df = pd.read_csv(args.metrics_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_all_metrics_summary(metrics_df, str(output_dir), args.color)
    
    if 'fu_time' in metrics_df.columns:
        plot_dice_by_time(metrics_df, str(output_dir), args.color, args.level)
        plot_image_quality_by_time(metrics_df, str(output_dir), args.color, args.level)
        plot_cwm_by_time(metrics_df, str(output_dir))
        plot_volumes_by_time(metrics_df, str(output_dir),args.level)
        #plot_volumes_combined()
    else:
        print("Warning: fu_time column not found, skipping time-based plots")


if __name__ == '__main__':
    main()
