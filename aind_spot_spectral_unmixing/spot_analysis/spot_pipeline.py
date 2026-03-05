"""
Refactored spot pipeline with modular components for batch experimentation.

This module separates data loading, configuration, and execution logic to enable
easy parameter tuning and batch experiments.
"""

# --- Standard library imports ---
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# --- Third-party imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- aind-hcr-data-loader imports ---
import aind_hcr_data_loader.filters as hcr_filters
from aind_hcr_data_loader.hcr_dataset import HCRDataset

# --- aind-hcr-qc imports ---
import aind_hcr_qc.viz as viz
import aind_hcr_qc.viz.spectral_unmixing as su
from aind_hcr_qc.utils.utils import saveable_plot

# --- spot_analysis imports (sibling modules) ---
from aind_spot_spectral_unmixing.spot_analysis import clean_spots
from aind_spot_spectral_unmixing.spot_analysis import config, ratio_calculator
from aind_spot_spectral_unmixing.spot_analysis.cell_by_gene_table import cell_by_gene_processor
from aind_spot_spectral_unmixing.spot_analysis.spot_processor import SpotProcessor
from aind_spot_spectral_unmixing.spot_analysis.unmixer import SpotUnmixer


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PipelineData:
    """Container for all data loaded for the pipeline."""
    ds: Any  # HCRDataset object
    mouse_id: str
    mouse_name: str
    round_key: str
    spots_df: pd.DataFrame
    filtered_cell_ids: List[int]
    cell_info: pd.DataFrame
    channels: List[str]
    dye_to_label: Dict[int, str]
    processing_manifest: Dict[str, Any]


@dataclass
class SpotPipelineConfig:
    """Configuration for spot pipeline execution."""
    
    # Experiment metadata
    expt_key: str
    output_base_folder: Path = Path('/root/capsule/results')
    
    # Spot filtering parameters
    do_clean_spots: bool = True
    clean_spots_params: Optional[Dict[str, Any]] = None
    
    # Config parameters (spot_capsule config)
    cent_cutoff: float = 1.0
    corr_cutoff: float = 0.3
    dist_cutoff: float = 1.0  #
    
    # TUNABLE: Distance parameters for batch experiments
    min_distances: List[int] = field(default_factory=lambda: [3])
    
    # TUNABLE: Unmixing method
    unmixing_method: str = 'reassignment'  # 'reassignment' or 'pairwise'
    
    # TUNABLE: Reassignment mode (DEPRECATED - use unmixing_method instead)
    reassignment: bool = True  # True = reassign based on spectral fit, False = spatial dedup only
    
    # TUNABLE: Channel pairs for pairwise unmixing
    channel_pairs: Optional[List[Tuple[str, str]]] = None
    
    # TUNABLE: Spatial scale for anisotropic distance (z, y, x) in um/pixel
    spatial_scale: Tuple[float, float, float] = (1.0, 0.24, 0.24)
    
    # Ratio calculation parameters
    ratio_spot_filter_method: str = "95percentile"
    
    # Visualization parameters
    plot_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default values for optional parameters."""
        if self.clean_spots_params is None:
            self.clean_spots_params = {
                'voxel_size_xyz': (.24, .24, 1.0),
                'R_iso': 0.1,
                'r_in': 1.0,
                'r_out': 3.0,
                'max_neighbors': 5,
                'min_top_frac': 0.2,
                'min_top_to_second': 0.3,
                'r_min': 0.2,
                'group_cols': None
            }
        
        if self.plot_params is None:
            self.plot_params = {
                'xlims': (-50, 500),
                'ylims': (-50, 500),
                'xlim_dye': (0, 1500),
                'ylim_dye': (0, 1500),
                'sample_per_channel': 10000,
                'dist_xlims': (0, 15),
                'dist_bins': 50,
                'figsize_dye': (15, 15),
                'alpha_points': 0.5
            }


# ============================================================================
# HELPER FUNCTIONS FOR CHANNEL DETECTION
# ============================================================================

def get_spot_channels_from_dataset(dataset: HCRDataset, round_key: str) -> List[str]:
    """
    Extract spot channels from a dataset's processing manifest.
    
    Parameters
    ----------
    dataset : HCRDataset
        The HCR dataset object
    round_key : str
        Round key (e.g., 'R1', 'R2')
    
    Returns
    -------
    List[str]
        Sorted list of spot channel wavelengths as strings
    """
    if round_key not in dataset.rounds:
        raise ValueError(f"Round {round_key} not found in dataset {dataset.mouse_id}")
    
    round_obj = dataset.rounds[round_key]
    
    if round_obj.processing_manifest is None:
        raise ValueError(f"No processing_manifest found for {dataset.mouse_id} - {round_key}")
    
    spot_channels = round_obj.processing_manifest.get('spot_channels')
    
    if spot_channels is None:
        raise ValueError(f"No 'spot_channels' key in processing_manifest for {dataset.mouse_id} - {round_key}")
    
    # Convert to strings and sort
    spot_channels = sorted([str(ch) for ch in spot_channels])
    
    return spot_channels


def get_default_channel_pairs(spot_channels: List[str]) -> List[Tuple[str, str]]:
    """
    Generate default channel pairs based on number of spot channels.
    
    Rules:
    - 2 channels (['488', '561']): [('488', '561')]
    - 3 channels (['488', '561', '638']): [('488', '561'), ('561', '638')]
    - 5 channels (['488', '514', '561', '594', '638']): 
        [('488', '514'), ('514', '561'), ('561', '594'), ('594', '638'), ('488', '594')]
    
    Parameters
    ----------
    spot_channels : List[str]
        List of spot channel wavelengths as strings
    
    Returns
    -------
    List[Tuple[str, str]]
        List of channel pairs for pairwise unmixing
    
    Raises
    ------
    ValueError
        If channel count is not 2, 3, or 5, or if expected channels are missing
    """
    channels_set = set(spot_channels)
    n_channels = len(spot_channels)
    
    if n_channels == 2:
        # Must have 488 and 561
        if {'488', '561'}.issubset(channels_set):
            return [('488', '561')]
        else:
            raise ValueError(f"2-channel case expects ['488', '561'], got {spot_channels}")
    
    elif n_channels == 3:
        # Must have 488, 561, 638
        if {'488', '561', '638'}.issubset(channels_set):
            return [('488', '561'), ('561', '638')]
        else:
            raise ValueError(f"3-channel case expects ['488', '561', '638'], got {spot_channels}")
    
    elif n_channels == 5:
        # Must have 488, 514, 561, 594, 638
        if {'488', '514', '561', '594', '638'}.issubset(channels_set):
            return [('488', '514'), ('514', '561'), ('561', '594'), ('594', '638'), ('488', '594')]
        else:
            raise ValueError(f"5-channel case expects ['488', '514', '561', '594', '638'], got {spot_channels}")
    
    else:
        raise ValueError(f"Unsupported number of channels: {n_channels}. Expected 2, 3, or 5.")


# ============================================================================
# PHASE 1: DATA LOADING
# ============================================================================

def load_pipeline_data(
    datasets: Dict[str, Any],
    mouse_id: str,
    round_key: str,
    use_soma_overlap_filter: bool = True
) -> PipelineData:
    """
    Load and prepare all data needed for the spot pipeline.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of HCRDataset objects keyed by mouse_id
    mouse_id : str
        Mouse identifier
    round_key : str
        Round key (e.g., "R3")
    use_soma_overlap_filter : bool, default=True
        If True, uses advanced ROI filtering for soma and overlap cells
    
    Returns
    -------
    PipelineData
        Container with all loaded data
    """
    # Get dataset
    ds = datasets[mouse_id]
    mouse_name = ds.metadata.get("nickname", ds.mouse_id)
    
    print(f"Loading data for {mouse_name} - {round_key}")
    
    # Get cell info
    cell_info = ds.rounds[round_key].get_cell_info(source="mixed_cxg")
    
    # IMPORTANT: cell_info has 'cell_id' as a COLUMN, not the index.
    # The index is just leftover row numbers from drop_duplicates() on the CSV.
    # We must use cell_info['cell_id'] everywhere, NOT cell_info.index.
    all_cell_ids = set(cell_info['cell_id'].values)
    
    # ---- DIAGNOSTIC: Confirm cell_id is a column, not the index ----
    print(f"  [DIAG] cell_info shape: {cell_info.shape}")
    print(f"  [DIAG] cell_info.index (first 5): {cell_info.index[:5].tolist()}  (these are ROW numbers, NOT cell IDs)")
    print(f"  [DIAG] cell_info['cell_id'] (first 5): {cell_info['cell_id'].values[:5].tolist()}  (these are REAL cell IDs)")
    print(f"  [DIAG] cell_info.index.max()={cell_info.index.max()}, cell_info['cell_id'].max()={cell_info['cell_id'].max()}")
    if cell_info.index.max() != cell_info['cell_id'].max():
        print(f"  [DIAG] *** CONFIRMED: index != cell_id — the old code was using the WRONG values")
    
    # Filter cells
    if use_soma_overlap_filter:
        filter_results = hcr_filters.roi_filter_comprehensive(ds)
        combined_ids = set(filter_results['filtered_ids'])
        roi_classifier_df = filter_results['soma_classifier_df']
        roi_upscale_df = filter_results['metrics_df']
        # FIX: Use cell_info['cell_id'] (the actual cell IDs), NOT cell_info.index (row numbers)
        filtered_cell_ids = [c for c in all_cell_ids if c not in combined_ids]
        filter_type = "soma and overlap"
        
        # ---- DIAGNOSTIC: Show the fix impact ----
        old_filtered = [c for c in cell_info.index if c not in combined_ids]
        print(f"  [DIAG] OLD (buggy) filtered_cell_ids count: {len(old_filtered)} "
              f"(was using row indices 0..{cell_info.index.max()})")
        print(f"  [DIAG] NEW (fixed) filtered_cell_ids count: {len(filtered_cell_ids)} "
              f"(using real cell_ids {min(all_cell_ids)}..{max(all_cell_ids)})")
        n_actually_filtered = len(all_cell_ids) - len(filtered_cell_ids)
        print(f"  [DIAG] ROI filter now actually removes {n_actually_filtered} cells "
              f"(was removing {len(cell_info) - len(old_filtered)} before)")
    else:
        # Basic filter - keep all cells
        # FIX: Use cell_info['cell_id'], NOT cell_info.index
        filtered_cell_ids = list(all_cell_ids)
        filter_type = "none"
    
    per_remain = len(filtered_cell_ids) / len(cell_info)
    print(f"Filtered {filter_type} cells: {len(filtered_cell_ids)}/{len(cell_info)} ({per_remain:.2%})")
    
    # Load spots with filtered cell IDs
    spots_df = ds.rounds[round_key].load_spots(table_type="mixed", filter_cell_ids=filtered_cell_ids)
    print(f"Loaded {len(spots_df)} spots")
    
    # Get channel information
    pm = ds.rounds[round_key].processing_manifest
    channels = pm["spot_channels"]
    dye_to_label = {0: '488', 1: '514', 2: '561', 3: '594', 4: '638'}
    
    return PipelineData(
        ds=ds,
        mouse_id=mouse_id,
        mouse_name=mouse_name,
        round_key=round_key,
        spots_df=spots_df,
        filtered_cell_ids=filtered_cell_ids,
        cell_info=cell_info,
        channels=channels,
        dye_to_label=dye_to_label,
        processing_manifest=pm
    )


# ============================================================================
# PHASE 2: SPOT CLEANING
# ============================================================================

def apply_spot_cleaning(
    spots_df: pd.DataFrame,
    pipeline_config: SpotPipelineConfig
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply spot cleaning filters.
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        Input spots dataframe
    pipeline_config : SpotPipelineConfig
        Configuration with cleaning parameters
    
    Returns
    -------
    cleaned_spots : pd.DataFrame
        Cleaned spots dataframe
    clean_mask : np.ndarray
        Boolean mask indicating which spots were kept
    """
    if not pipeline_config.do_clean_spots:
        print("Skipping spot cleaning")
        return spots_df.copy(), np.ones(len(spots_df), dtype=bool)
    
    print("Applying spot cleaning...")
    clean_mask = clean_spots.compute_clean_mask_kdtree(
        spots_df,
        **pipeline_config.clean_spots_params
    )
    
    cleaned_spots = spots_df[clean_mask].copy()
    print(f"Kept {clean_mask.sum()} / {len(spots_df)} spots after cleaning ({clean_mask.sum()/len(spots_df):.2%})")
    
    return cleaned_spots, clean_mask


# ============================================================================
# PHASE 3: RATIO CALCULATION
# ============================================================================

def calculate_ratios(
    spots_df: pd.DataFrame,
    pipeline_data: PipelineData,
    output_folder: Path,
    pipeline_config: SpotPipelineConfig
) -> Tuple[np.ndarray, Dict[str, Any], Path, Any]:
    """
    Calculate spectral unmixing ratios.
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        Spots dataframe (typically cleaned spots for optimization)
    pipeline_data : PipelineData
        Pipeline data container
    output_folder : Path
        Output directory
    pipeline_config : SpotPipelineConfig
        Configuration with ratio calculation parameters
    
    Returns
    -------
    ratios : np.ndarray
        Calculated ratio matrix
    loss_history : dict
        Optimization loss history
    ratio_path : Path
        Path to saved ratios file
    ds_config : config.Config
        spot_capsule config object
    """
    print("Calculating spectral unmixing ratios...")
    
    # Create spot_capsule config
    ds_config = config.Config(dataset_folder=pipeline_data.ds.rounds[pipeline_data.round_key].name)
    ds_config.SCRATCH_FOLDER = output_folder
    ds_config.OUTPUT_FOLDER = output_folder
    ds_config.ROUND_N = str(pipeline_data.round_key[1:])
    ds_config.CENT_CUTOFF = pipeline_config.cent_cutoff
    ds_config.CORR_CUTOFF = pipeline_config.corr_cutoff
    ds_config.DIST_CUTOFF = pipeline_config.dist_cutoff
    
    # Extract intensity array
    intensity_cols = [col for col in spots_df.columns if col.endswith('intensity')]
    intensity_array = spots_df[intensity_cols].to_numpy()
    print(f"Intensity array shape: {intensity_array.shape}")
    
    # Calculate ratios
    ratio_path = output_folder / f"{pipeline_data.round_key}_ratios.txt"
    rc = ratio_calculator.RatioCalculator(
        dataset_folder=pipeline_data.ds.rounds[pipeline_data.round_key].name,
        config=ds_config
    )
    
    ratios, loss_history = rc.calculate_ratios_by_channel_loss(
        intensity_array,
        ratio_path,
        detection_channels=spots_df["chan"].values,
        spot_filter_method=pipeline_config.ratio_spot_filter_method,
        return_loss_history=True
    )
    
    # Save loss history
    loss_history_path = output_folder / f"{pipeline_data.round_key}_loss_history.pkl"
    with open(loss_history_path, "wb") as f:
        pickle.dump(loss_history, f)
    
    print(f"Ratios calculated and saved to {ratio_path}")
    
    return ratios, loss_history, ratio_path, ds_config


# ============================================================================
# PHASE 4: UNMIXING AND PROCESSING
# ============================================================================

def unmix_and_process_spots(
    spots_df: pd.DataFrame,
    ratios: np.ndarray,
    pipeline_data: PipelineData,
    ds_config: Any,
    pipeline_config: SpotPipelineConfig,
    output_folder: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Unmix spots and generate cell-by-gene tables.
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        Spots dataframe to unmix (typically full dataset, not just cleaned)
    ratios : np.ndarray
        Ratio matrix from calculate_ratios
    pipeline_data : PipelineData
        Pipeline data container
    ds_config : config.Config
        spot_capsule config object
    pipeline_config : SpotPipelineConfig
        Configuration with min_distances and other parameters
    output_folder : Path
        Output directory for saving reassignment matrix
    
    Returns
    -------
    unmixed_df : pd.DataFrame
        Unmixed spots dataframe with unmixed_chan column
    stats_df : pd.DataFrame
        Distance statistics for spots
    results : dict
        Results from process_multiple_distances (dict of {min_dist: (unmixed_df, stats)})
    unmixed_results : pd.DataFrame
        Unmixed cell-by-gene counts
    mixed_results : pd.DataFrame
        Mixed cell-by-gene counts
    reassignment_matrix : pd.DataFrame
        Matrix showing how spots are reassigned from chan to unmixed_chan
    spot_fate_matrix : pd.DataFrame
        Matrix showing spot fate: fixed, reassigned, or removed
    """
    print("Unmixing and processing spots...")
    
    # ---- DIAGNOSTIC: Input state ----
    print(f"  [DIAG] Input spots_df: shape={spots_df.shape}, "
          f"index dtype={spots_df.index.dtype}, "
          f"index range=[{spots_df.index.min()}, {spots_df.index.max()}], "
          f"is_contiguous={spots_df.index.is_monotonic_increasing and len(spots_df.index) == spots_df.index.max() - spots_df.index.min() + 1}")
    print(f"  [DIAG] 'spot_id' already in columns: {'spot_id' in spots_df.columns}")
    print(f"  [DIAG] 'chan' dtype: {spots_df['chan'].dtype}, unique values: {sorted(spots_df['chan'].unique())}")
    
    # Prepare spots_df with required index
    spots_df_work = spots_df.copy()
    spots_df_work.reset_index(inplace=True)
    if 'index' in spots_df_work.columns:
        spots_df_work.rename(columns={'index': 'spot_id'}, inplace=True)
    
    # ---- DIAGNOSTIC: After reset_index ----
    print(f"  [DIAG] After reset_index: shape={spots_df_work.shape}, "
          f"index range=[{spots_df_work.index.min()}, {spots_df_work.index.max()}]")
    assert spots_df_work.index.is_monotonic_increasing, \
        "BUG: spots_df_work index is not monotonic after reset_index"
    assert len(spots_df_work.index) == spots_df_work.index.max() + 1, \
        f"BUG: spots_df_work index is not 0-based contiguous: len={len(spots_df_work)}, max={spots_df_work.index.max()}"
    
    # ---- DIAGNOSTIC: Check intensity column order consistency ----
    intensity_cols_work = [col for col in spots_df_work.columns if col.endswith('intensity')]
    print(f"  [DIAG] Intensity columns in spots_df_work: {intensity_cols_work}")
    config_channels = ds_config.get_round_spot_channels()
    expected_int_cols = [f'chan_{ch}_intensity' for ch in config_channels]
    print(f"  [DIAG] Config spot channels: {config_channels}")
    print(f"  [DIAG] Expected intensity cols from config: {expected_int_cols}")
    missing_int_cols = set(expected_int_cols) - set(intensity_cols_work)
    extra_int_cols = set(intensity_cols_work) - set(expected_int_cols)
    if missing_int_cols:
        print(f"  [DIAG] *** WARNING: Missing intensity columns: {missing_int_cols}")
    if extra_int_cols:
        print(f"  [DIAG] *** NOTE: Extra intensity columns (not in config): {extra_int_cols}")
    
    # Save spots — normalize chan to str for consistent dtype in downstream pickle consumers
    spots_df_work['chan'] = spots_df_work['chan'].astype(str)
    spots_df_work.to_pickle(
        ds_config.SCRATCH_FOLDER / f'mixed_spots_R{ds_config.ROUND_N}.pkl'
    )
    
    # Initialize unmixer and processor
    unmixer = SpotUnmixer(
        dataset_folder=pipeline_data.ds.rounds[pipeline_data.round_key].name,
        config=ds_config,
        channel_pairs=pipeline_config.channel_pairs,
        spatial_scale=pipeline_config.spatial_scale
    )
    processor = SpotProcessor()
    
    # Calculate distances
    stats_df = unmixer.calculate_distances(spots_df_work, ratios)
    
    # ---- DIAGNOSTIC: stats_df alignment with spots_df_work ----
    print(f"  [DIAG] stats_df: shape={stats_df.shape}, index range=[{stats_df.index.min()}, {stats_df.index.max()}]")
    assert len(stats_df) == len(spots_df_work), \
        f"BUG: stats_df length ({len(stats_df)}) != spots_df_work length ({len(spots_df_work)})"
    
    # Apply QC filters
    spots_df_filtered = processor.apply_qc_filters(spots_df_work, stats_df)
    print(f"After QC filters: {len(spots_df_filtered)} / {len(spots_df_work)} spots")
    
    # ---- DIAGNOSTIC: QC filter output ----
    # apply_qc_filters calls reset_index(drop=True) internally on BOTH dfs,
    # so the returned df has a NEW 0-based index that does NOT correspond to spots_df_work's index.
    # This is fine IF downstream code only uses positional alignment (not index-based joins).
    print(f"  [DIAG] spots_df_filtered after QC: shape={spots_df_filtered.shape}, "
          f"index range=[{spots_df_filtered.index.min()}, {spots_df_filtered.index.max()}], "
          f"is_contiguous={(spots_df_filtered.index.max() + 1 == len(spots_df_filtered)) if len(spots_df_filtered) > 0 else True}")
    print(f"  [DIAG] 'valid_spot' value_counts: {spots_df_filtered['valid_spot'].value_counts().to_dict()}")
    
    # ---- DIAGNOSTIC: Check that valid_spot filtering was applied ----
    n_valid = spots_df_filtered['valid_spot'].sum()
    n_total = len(spots_df_filtered)
    print(f"  [DIAG] *** NOTE: apply_qc_filters returns ALL rows with valid_spot column set. "
          f"valid={n_valid}, invalid={n_total - n_valid}. "
          f"Downstream code must check: does process_multiple_distances filter on valid_spot?")
    
    # ---- DIAGNOSTIC: Check chan dtype consistency ----
    chan_dtype_filtered = spots_df_filtered['chan'].dtype
    print(f"  [DIAG] Filtered 'chan' dtype: {chan_dtype_filtered}, "
          f"unique values: {sorted(spots_df_filtered['chan'].unique())}")
    
    # Recalculate distances for filtered spots
    all_chans_filt_stats = unmixer.calculate_distances(spots_df_filtered, ratios)
    
    # ---- DIAGNOSTIC: Filtered stats alignment ----
    print(f"  [DIAG] all_chans_filt_stats: shape={all_chans_filt_stats.shape}, "
          f"index range=[{all_chans_filt_stats.index.min()}, {all_chans_filt_stats.index.max()}]")
    assert len(all_chans_filt_stats) == len(spots_df_filtered), \
        f"BUG: all_chans_filt_stats length ({len(all_chans_filt_stats)}) != spots_df_filtered length ({len(spots_df_filtered)})"
    # Check index alignment — this is the critical hypothesis #2 check
    if not spots_df_filtered.index.equals(all_chans_filt_stats.index):
        print(f"  [DIAG] *** WARNING: Index mismatch between spots_df_filtered and all_chans_filt_stats!")
        print(f"         spots_df_filtered.index[:5] = {spots_df_filtered.index[:5].tolist()}")
        print(f"         all_chans_filt_stats.index[:5] = {all_chans_filt_stats.index[:5].tolist()}")
    else:
        print(f"  [DIAG] ✓ spots_df_filtered and all_chans_filt_stats indices match")
    
    # Process multiple distances
    print(f"Processing with min_distances: {pipeline_config.min_distances}")
    print(f"Unmixing method: {pipeline_config.unmixing_method}")
    if pipeline_config.unmixing_method == 'pairwise':
        print(f"Channel pairs: {pipeline_config.channel_pairs}")
        print(f"Spatial scale: {pipeline_config.spatial_scale}")
    
    results = unmixer.process_multiple_distances(
        spots_df_filtered,
        all_chans_filt_stats,
        pipeline_config.min_distances,
        unmixing_method=pipeline_config.unmixing_method,
        channel_pairs=pipeline_config.channel_pairs
    )
    
    # Extract unmixed dataframe from results (use first min_distance)
    # Results is a dict: {min_dist: (unmixed_df, channel_stats)}
    first_min_dist = pipeline_config.min_distances[0]
    unmixed_df, channel_stats = results[first_min_dist]
    print(f"Unmixed {len(unmixed_df)} spots with min_distance={first_min_dist}")
    
    # Normalize chan/unmixed_chan to str to prevent category vs object comparison bugs
    unmixed_df['chan'] = unmixed_df['chan'].astype(str)
    unmixed_df['unmixed_chan'] = unmixed_df['unmixed_chan'].astype(str)
    
    # ---- DIAGNOSTIC: Unmixed output ----
    print(f"  [DIAG] unmixed_df: shape={unmixed_df.shape}, "
          f"index range=[{unmixed_df.index.min()}, {unmixed_df.index.max()}]")
    if 'unmixed_chan' in unmixed_df.columns:
        print(f"  [DIAG] unmixed_chan unique values: {sorted(unmixed_df['unmixed_chan'].unique())}")
        print(f"  [DIAG] unmixed_chan dtype: {unmixed_df['unmixed_chan'].dtype}")
        print(f"  [DIAG] chan dtype: {unmixed_df['chan'].dtype}")
        # Check: are chan and unmixed_chan comparable types?
        if unmixed_df['chan'].dtype != unmixed_df['unmixed_chan'].dtype:
            print(f"  [DIAG] *** WARNING: dtype mismatch between chan ({unmixed_df['chan'].dtype}) "
                  f"and unmixed_chan ({unmixed_df['unmixed_chan'].dtype})")
    if 'valid_spot' in unmixed_df.columns:
        print(f"  [DIAG] unmixed_df valid_spot counts: {unmixed_df['valid_spot'].value_counts().to_dict()}")
    
    # IMPORTANT: Set min_dist in config for cell_by_gene_processor
    ds_config.min_dist = int(first_min_dist)
    
    # ---- DIAGNOSTIC: Verify file paths for cell_by_gene_processor ----
    mixed_pkl_path = ds_config.SCRATCH_FOLDER / f'mixed_spots_R{ds_config.ROUND_N}.pkl'
    unmixed_pkl_path = ds_config.SCRATCH_FOLDER / f'unmixed_spots_R{ds_config.ROUND_N}_minDist_{int(first_min_dist)}.pkl'
    print(f"  [DIAG] cell_by_gene_processor will read:")
    print(f"         Mixed:   {mixed_pkl_path} (exists={mixed_pkl_path.exists()})")
    print(f"         Unmixed: {unmixed_pkl_path} (exists={unmixed_pkl_path.exists()})")
    
    # ---- DIAGNOSTIC: Verify saved pickle matches in-memory unmixed_df ----
    if unmixed_pkl_path.exists():
        import pickle as _pkl
        with open(unmixed_pkl_path, 'rb') as _f:
            unmixed_from_disk = _pkl.load(_f)
        print(f"  [DIAG] unmixed_df on disk: shape={unmixed_from_disk.shape}")
        if len(unmixed_from_disk) != len(unmixed_df):
            print(f"  [DIAG] *** WARNING: Disk unmixed ({len(unmixed_from_disk)}) != "
                  f"in-memory unmixed ({len(unmixed_df)})")
        if 'unmixed_chan' in unmixed_from_disk.columns and 'chan' in unmixed_from_disk.columns:
            n_same = (unmixed_from_disk['chan'] == unmixed_from_disk['unmixed_chan']).sum()
            n_reassigned = (unmixed_from_disk['chan'] != unmixed_from_disk['unmixed_chan']).sum()
            print(f"  [DIAG] On-disk unmixed: {n_same} kept original chan, {n_reassigned} reassigned")
    
    # ---- DIAGNOSTIC: Verify gene_dict mapping ----
    gene_dict = ds_config.GENE_DICT
    print(f"  [DIAG] GENE_DICT for round {ds_config.ROUND_N}: {gene_dict.get(str(ds_config.ROUND_N), 'NOT FOUND')}")
    config_chan_keys = list(gene_dict.get(str(ds_config.ROUND_N), {}).keys())
    spots_chan_unique = sorted(map(str, unmixed_df['chan'].unique())) if 'chan' in unmixed_df.columns else []
    unmixed_chan_unique = sorted(map(str, unmixed_df['unmixed_chan'].unique())) if 'unmixed_chan' in unmixed_df.columns else []
    print(f"  [DIAG] GENE_DICT channel keys: {config_chan_keys}")
    print(f"  [DIAG] Spots 'chan' unique (str): {spots_chan_unique}")
    print(f"  [DIAG] Spots 'unmixed_chan' unique (str): {unmixed_chan_unique}")
    unmatched_gene_chans = set(spots_chan_unique) - set(config_chan_keys)
    if unmatched_gene_chans:
        print(f"  [DIAG] *** WARNING: chan values {unmatched_gene_chans} have NO gene mapping in GENE_DICT!")
    
    # Generate cell-by-gene tables
    cbg_processor = cell_by_gene_processor(
        dataset_folder=pipeline_data.ds.rounds[pipeline_data.round_key].name,
        config=ds_config
    )
    unmixed_results, mixed_results = cbg_processor.process_pipeline([ds_config.ROUND_N])
    
    print(f"Generated cell-by-gene tables: {len(unmixed_results)} unmixed, {len(mixed_results)} mixed")
    
    # ---- DIAGNOSTIC: cell-by-gene output sanity checks ----
    if 'gene' in unmixed_results.columns:
        print(f"  [DIAG] Unmixed CxG genes: {sorted(unmixed_results['gene'].unique())}")
        print(f"  [DIAG] Mixed CxG genes: {sorted(mixed_results['gene'].unique())}")
        # Check gene names match GENE_DICT values
        expected_genes = set(gene_dict.get(str(ds_config.ROUND_N), {}).values())
        actual_unmixed_genes = set(unmixed_results['gene'].unique())
        actual_mixed_genes = set(mixed_results['gene'].unique())
        if expected_genes != actual_unmixed_genes:
            print(f"  [DIAG] *** WARNING: Unmixed gene mismatch! "
                  f"Expected: {expected_genes}, Got: {actual_unmixed_genes}, "
                  f"Missing: {expected_genes - actual_unmixed_genes}, "
                  f"Extra: {actual_unmixed_genes - expected_genes}")
    if 'spot_count' in unmixed_results.columns:
        print(f"  [DIAG] Unmixed total spot_count: {unmixed_results['spot_count'].sum()}")
        print(f"  [DIAG] Mixed total spot_count: {mixed_results['spot_count'].sum()}")
    
    # Create reassignment matrix using the unmixed dataframe
    print("Creating reassignment matrix...")
    reassignment_matrix = su.create_reassignment_matrix(unmixed_df, normalize=False)
    
    # Save reassignment matrix
    reassignment_path = output_folder / f"{pipeline_data.round_key}_reassignment_matrix.csv"
    reassignment_matrix.to_csv(reassignment_path)
    print(f"Reassignment matrix saved to {reassignment_path}")
    
    # Also save normalized versions
    reassignment_matrix_rows = su.create_reassignment_matrix(unmixed_df, normalize='rows')
    reassignment_path_rows = output_folder / f"{pipeline_data.round_key}_reassignment_matrix_norm_rows.csv"
    reassignment_matrix_rows.to_csv(reassignment_path_rows)
    
    # Create spot fate matrix (tracks fixed/reassigned/removed spots)
    print("Creating spot fate matrix...")
    # Guard: analyze_spot_fate requires chan_spot_id for unique spot identity
    for _label, _df in [("spots_df_work", spots_df_work), ("unmixed_df", unmixed_df)]:
        assert 'chan_spot_id' in _df.columns, \
            f"BUG: '{_label}' is missing 'chan_spot_id' column (needed by analyze_spot_fate). Columns: {list(_df.columns)}"
    spot_fate_matrix = su.analyze_spot_fate(spots_df_work, unmixed_df)
    
    # Save spot fate matrix
    spot_fate_path = output_folder / f"{pipeline_data.round_key}_spot_fate_matrix.csv"
    spot_fate_matrix.to_csv(spot_fate_path, index=False)
    print(f"Spot fate matrix saved to {spot_fate_path}")

    # I think cx gene already saved
    # cxg_df_mixed = mixed_results[["cell_id", "gene", "spot_count"]].copy()
    # cxg_df_mixed = cxg_df_mixed.pivot(index="cell_id", columns="gene", values="spot_count")

    # cxg_df_unmixed = unmixed_results[["cell_id", "gene", "spot_count"]].copy()
    # cxg_df_unmixed = cxg_df_unmixed.pivot(index="cell_id", columns="gene", values="spot_count")

    return unmixed_df, stats_df, results, unmixed_results, mixed_results, reassignment_matrix, spot_fate_matrix


# ============================================================================
# PHASE 5: VISUALIZATION
# ============================================================================

def generate_all_visualizations(
    pipeline_data: PipelineData,
    spots_df_full: pd.DataFrame,
    spots_df_cleaned: pd.DataFrame,
    ratios: np.ndarray,
    mixed_results: pd.DataFrame,
    unmixed_results: pd.DataFrame,
    output_folder: Path,
    pipeline_config: SpotPipelineConfig,
    ds_config: Any,
    color_map: Optional[Dict] = None
):
    """
    Generate all visualization plots for the pipeline.
    
    Parameters
    ----------
    pipeline_data : PipelineData
        Pipeline data container
    spots_df_full : pd.DataFrame
        Full spots dataframe (after QC)
    spots_df_cleaned : pd.DataFrame
        Cleaned spots used for optimization
    ratios : np.ndarray
        Ratio matrix
    mixed_results : pd.DataFrame
        Mixed cell-by-gene results
    unmixed_results : pd.DataFrame
        Unmixed cell-by-gene results
    output_folder : Path
        Output directory for plots
    pipeline_config : SpotPipelineConfig
        Configuration with plotting parameters
    ds_config : config.Config
        spot_capsule config object
    color_map : dict, optional
        Color map for plotting
    """
    print("Generating visualizations...")
    
    pp = pipeline_config.plot_params
    
    # 1. Plot cleaned mixed spots (pairwise intensities)
    print("  - Plotting pairwise intensities (cleaned spots)...")
    viz.plot_filtered_intensities(
        spots_df_cleaned,
        title=f"{pipeline_data.mouse_name} - {pipeline_data.round_key} - Cleaned Mixed Spots",
        plot_cell_ids=list(range(1, 100000)),
        channel_label="chan",
        scale="linear",
        xlims=pp['xlims'],
        ylims=pp['ylims'],
        save=True,
        output_dir=output_folder,
        filename="pairwise_intensities_mixed",
        show=False
    )
    
    # 2. Plot dye lines for optimized spots
    print("  - Plotting dye lines (optimized spots)...")
    intensity_cols = [col for col in spots_df_cleaned.columns if col.endswith('intensity')]
    intensity_array_opt = spots_df_cleaned[intensity_cols].to_numpy()
    
    for plot_spots in [True, False]:
        filename = f"{pipeline_data.mouse_id}_{pipeline_data.round_key}_pairwise_dye_lines_spots={plot_spots}_optimized"
        viz.plot_dye_lines_pairwise(
            intensity_array_opt,
            pipeline_data.channels,
            ratios,
            spots_df_cleaned["chan"].values,
            color_map=color_map,
            dye_to_label=pipeline_data.dye_to_label,
            sample_per_channel=pp['sample_per_channel'],
            plot_only_pair_dyes=True,
            plot_dye_lines=True,
            plot_spots=plot_spots,
            xlim=(0, pp['xlim_dye'][1]),
            ylim=(0, pp['ylim_dye'][1]),
            figsize=pp['figsize_dye'],
            alpha_points=pp['alpha_points'],
            save=True,
            output_dir=output_folder,
            filename=filename,
            show=False
        )
    
    # 3. Plot dye lines for all spots
    print("  - Plotting dye lines (all spots)...")
    intensity_cols_full = [col for col in spots_df_full.columns if col.endswith('intensity')]
    intensity_array_full = spots_df_full[intensity_cols_full].to_numpy()
    detected_channels_full = spots_df_full["chan"].values
    
    for plot_spots in [True, False]:
        filename = f"{pipeline_data.mouse_id}_{pipeline_data.round_key}_pairwise_dye_lines_spots={plot_spots}"
        viz.plot_dye_lines_pairwise(
            intensity_array_full,
            pipeline_data.channels,
            ratios,
            detected_channels_full,
            color_map=color_map,
            dye_to_label=pipeline_data.dye_to_label,
            sample_per_channel=pp['sample_per_channel'],
            plot_only_pair_dyes=True,
            plot_dye_lines=True,
            plot_spots=plot_spots,
            xlim=(0, pp['xlim_dye'][1]),
            ylim=(0, pp['ylim_dye'][1]),
            figsize=pp['figsize_dye'],
            alpha_points=pp['alpha_points'],
            save=True,
            output_dir=output_folder,
            filename=filename,
            show=False
        )
    
    # 4. Plot channel distance distributions
    try:
        print("  - Plotting channel distance distributions...")
        su.plot_channel_distributions(
            spots_df_full,
            xlims=(pp['dist_xlims'][0], pp['dist_xlims'][1]),
            bins=pp['dist_bins'],
            threshold=ds_config.DIST_CUTOFF,
            kde=False,
            save=True,
            output_dir=output_folder,
            filename="dist_ratio_by_channel",
            show=False
        )
    except (ImportError, NameError, AttributeError):
        print("  - Skipping channel distributions (plot_channel_distributions not available)")
    
    # 5. Plot mixed vs unmixed comparison
    try:
        from capsule_scratch_utils import plot_mixed_unmixed_comparison
        print("  - Plotting mixed vs unmixed comparison...")
        plot_mixed_unmixed_comparison(
            mixed_results,
            unmixed_results,
            k=6,
            save=True,
            output_dir=output_folder,
            filename="mixed_vs_unmixed_comparison",
            show=False
        )
    except (ImportError, NameError):
        print("  - Skipping mixed/unmixed comparison (plot function not available)")
    
    print("Visualizations complete!")


# ============================================================================
# PHASE 6: MAIN PIPELINE FUNCTION
# ============================================================================

def run_spot_pipeline_v2(
    datasets: Dict[str, Any],
    mouse_id: str,
    round_key: str,
    pipeline_config: SpotPipelineConfig,
    skip_visualizations: bool = False,
    color_map: Optional[Dict] = None,
    return_results = False,
) -> Dict[str, Any]:
    """
    Modular spot pipeline with configurable parameters.
    
    This is the main entry point for the refactored pipeline. It orchestrates
    data loading, spot cleaning, ratio calculation, unmixing, and visualization.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of HCRDataset objects keyed by mouse_id
    mouse_id : str
        Mouse identifier
    round_key : str
        Round key (e.g., "R3")
    pipeline_config : SpotPipelineConfig
        Configuration object with all tunable parameters
    skip_visualizations : bool, default=False
        Skip generating plots (useful for batch experiments)
    color_map : dict, optional
        Color map for plotting
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - pipeline_data: PipelineData
        - spots_df_full: final processed DataFrame
        - spots_df_cleaned: cleaned spots used for optimization
        - clean_mask: boolean mask for cleaning
        - ratios: ratio array
        - loss_history: optimization history
        - mixed_results: DataFrame
        - unmixed_results: DataFrame
        - output_folder: Path
        - ds_config: spot_capsule config object
    """
    print("=" * 80)
    print(f"RUNNING SPOT PIPELINE V2")
    print(f"Mouse: {mouse_id}, Round: {round_key}, Experiment: {pipeline_config.expt_key}")
    print("=" * 80)
    
    # Create output folder
    output_folder = pipeline_config.output_base_folder / pipeline_config.expt_key / f"{mouse_id}_{round_key}"
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Phase 1: Load data
    print("\n[1/5] Loading data...")
    pipeline_data = load_pipeline_data(datasets, mouse_id, round_key)
    
    # Phase 2: Clean spots
    print("\n[2/5] Cleaning spots...")
    spots_df_cleaned, clean_mask = apply_spot_cleaning(pipeline_data.spots_df, pipeline_config)
    
    # Phase 3: Calculate ratios (using cleaned spots)
    print("\n[3/5] Calculating ratios...")
    ratios, loss_history, ratio_path, ds_config = calculate_ratios(
        spots_df_cleaned,
        pipeline_data,
        output_folder,
        pipeline_config
    )
    
    # Phase 4: Unmix and process (using full spots_df, not just cleaned)
    print("\n[4/5] Unmixing and processing...")
    spots_df_processed, stats_df, unmix_results, unmixed_results, mixed_results, reassignment_matrix, spot_fate_matrix = unmix_and_process_spots(
        pipeline_data.spots_df,  # Use full dataset for unmixing
        ratios,
        pipeline_data,
        ds_config,
        pipeline_config,
        output_folder
    )
    
    # Phase 5: Visualizations
    if not skip_visualizations:
        print("\n[5/5] Generating visualizations...")
        generate_all_visualizations(
            pipeline_data,
            spots_df_processed,
            spots_df_cleaned,
            ratios,
            mixed_results,
            unmixed_results,
            output_folder,
            pipeline_config,
            ds_config,
            color_map
        )
    else:
        print("\n[5/5] Skipping visualizations")
    
    # Save ratios matrix as CSV for easy inspection
    ratios_csv_path = output_folder / f"{pipeline_data.round_key}_ratios_matrix.csv"
    ratios_df = pd.DataFrame(ratios)
    ratios_df.to_csv(ratios_csv_path, index=False)
    print(f"Ratios matrix saved to {ratios_csv_path}")

    # save config
    config_path = output_folder / f"unmixing_config.json"
    with open(config_path, "w") as f:
        json.dump(pipeline_config.__dict__, f, indent=4)
    print(f"Pipeline config saved to {config_path}")

    # ds_config
    ds_config_path = output_folder / f"ds_config.json"
    with open(ds_config_path, "w") as f:
        json.dump(ds_config.__dict__, f, indent=4)
    print(f"Dataset config saved to {ds_config_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    # Return all results
    if return_results:
        return {
            'pipeline_data': pipeline_data,
            'spots_df_full': spots_df_processed,
            'spots_df_cleaned': spots_df_cleaned,
            'clean_mask': clean_mask,
            'ratios': ratios,
            'ratios_matrix': ratios_df,
            'reassignment_matrix': reassignment_matrix,
            'spot_fate_matrix': spot_fate_matrix,
            'loss_history': loss_history,
            'ratio_path': ratio_path,
            'mixed_results': mixed_results,
            'unmixed_results': unmixed_results,
            'stats_df': stats_df,
            'unmix_results': unmix_results,
            'output_folder': output_folder,
            'ds_config': ds_config,
            'config': pipeline_config
        }


# ============================================================================
# LEGACY WRAPPER FOR BACKWARDS COMPATIBILITY
# ============================================================================

def run_spot_pipeline(
    datasets: Dict[str, Any],
    mouse_id: str,
    round_key: str,
    expt_key: str,
    do_clean_spots: bool
) -> Dict[str, Any]:
    """
    Legacy wrapper for backwards compatibility with original function signature.
    
    Parameters
    ----------
    datasets : dict
        Dataset dictionary
    mouse_id : str
        Mouse identifier
    round_key : str
        Round key (e.g., "R3")
    expt_key : str
        Experiment key
    do_clean_spots : bool
        Whether to clean spots
    
    Returns
    -------
    results : dict
        Results from run_spot_pipeline_v2
    """
    pipeline_config = SpotPipelineConfig(
        expt_key=expt_key,
        do_clean_spots=do_clean_spots,
        min_distances=[3]
    )
    return run_spot_pipeline_v2(datasets, mouse_id, round_key, pipeline_config)


# ============================================================================
# PHASE 7: BATCH EXPERIMENT UTILITIES
# ============================================================================

def run_batch_experiment(
    datasets: Dict[str, Any],
    mouse_id: str,
    round_key: str,
    base_config: SpotPipelineConfig,
    param_grid: Dict[str, List[Any]],
    output_key: str = "batch_experiment"
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run pipeline with different parameter combinations for experimentation.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of HCRDataset objects
    mouse_id : str
        Mouse identifier
    round_key : str
        Round key (e.g., "R3")
    base_config : SpotPipelineConfig
        Base configuration to use as starting point
    param_grid : dict
        Dictionary of parameter names to lists of values.
        Example: {
            'min_distances': [[1], [2], [3], [5]],
            'dist_cutoff': [0.5, 1.0, 1.5],
            'corr_cutoff': [0.2, 0.3, 0.4]
        }
    output_key : str, default="batch_experiment"
        Key to append to experiment name
    
    Returns
    -------
    summary_df : pd.DataFrame
        Summary of results for each parameter combination with metrics
    all_results : list of dict
        Full results for each run
    """
    from itertools import product
    from copy import deepcopy
    
    print("=" * 80)
    print("RUNNING BATCH EXPERIMENT")
    print(f"Mouse: {mouse_id}, Round: {round_key}")
    print(f"Parameter grid: {param_grid}")
    print("=" * 80)
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nTotal combinations to test: {len(combinations)}\n")
    
    all_results = []
    summary_records = []
    
    for idx, combo in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"Combination {idx}/{len(combinations)}")
        print(f"{'='*80}")
        
        # Create config for this combination
        config = deepcopy(base_config)
        config.expt_key = f"{base_config.expt_key}_{output_key}"
        
        # Set parameters
        param_dict = dict(zip(param_names, combo))
        print(f"Parameters: {param_dict}")
        
        for param_name, param_value in param_dict.items():
            setattr(config, param_name, param_value)
        
        # Update experiment key to include parameters
        param_str = "_".join([f"{k}={v}" for k, v in param_dict.items()])
        config.expt_key = f"{config.expt_key}/{param_str}"
        
        # Run pipeline (skip visualizations for speed)
        try:
            results = run_spot_pipeline_v2(
                datasets,
                mouse_id,
                round_key,
                pipeline_config=config,
                skip_visualizations=True
            )
            
            # Extract summary metrics
            summary = {
                'combination_id': idx,
                'mouse_id': mouse_id,
                'round_key': round_key,
                **param_dict,
                'n_spots_total': len(results['pipeline_data'].spots_df),
                'n_spots_cleaned': len(results['spots_df_cleaned']),
                'n_spots_final': len(results['spots_df_full']),
                'n_cells': len(results['pipeline_data'].cell_info),
                'unmixed_total_counts': results['unmixed_results']['spot_count'].sum(),
                'mixed_total_counts': results['mixed_results']['spot_count'].sum(),
                'output_folder': str(results['output_folder']),
                'ratio_path': str(results['ratio_path']),
                'reassignment_matrix_path': str(results['output_folder'] / f"{round_key}_reassignment_matrix.csv"),
                'spot_fate_matrix_path': str(results['output_folder'] / f"{round_key}_spot_fate_matrix.csv"),
                'ratios_matrix_path': str(results['output_folder'] / f"{round_key}_ratios_matrix.csv"),
                'success': True,
                'error': None
            }
            
            # Save individual matrices for this combination
            batch_matrices_folder = base_config.output_base_folder / f"{base_config.expt_key}_{output_key}" / "matrices"
            batch_matrices_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy matrices to batch folder with parameter info
            combo_name = "_".join([f"{k}={v}" for k, v in param_dict.items()])
            
            # Save ratios matrix
            ratios_batch_path = batch_matrices_folder / f"{mouse_id}_{round_key}_{combo_name}_ratios.csv"
            results['ratios_matrix'].to_csv(ratios_batch_path, index=False)
            
            # Save reassignment matrix
            reassign_batch_path = batch_matrices_folder / f"{mouse_id}_{round_key}_{combo_name}_reassignment.csv"
            results['reassignment_matrix'].to_csv(reassign_batch_path)
            
            # Save spot fate matrix
            spot_fate_batch_path = batch_matrices_folder / f"{mouse_id}_{round_key}_{combo_name}_spot_fate.csv"
            results['spot_fate_matrix'].to_csv(spot_fate_batch_path, index=False)
            
            summary['batch_ratios_path'] = str(ratios_batch_path)
            summary['batch_reassignment_path'] = str(reassign_batch_path)
            summary['batch_spot_fate_path'] = str(spot_fate_batch_path)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"ERROR in combination {idx}: {e}")
            summary = {
                'combination_id': idx,
                'mouse_id': mouse_id,
                'round_key': round_key,
                **param_dict,
                'success': False,
                'error': str(e)
            }
        
        summary_records.append(summary)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_records)
    
    # Save summary
    summary_path = base_config.output_base_folder / f"{base_config.expt_key}_{output_key}" / "batch_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "=" * 80)
    print("BATCH EXPERIMENT COMPLETE!")
    print(f"Summary saved to: {summary_path}")
    print(f"Successful runs: {summary_df['success'].sum()}/{len(summary_df)}")
    print("=" * 80)
    
    return summary_df, all_results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_pipeline_results(
    results_list: List[Dict[str, Any]],
    metric: str = 'unmixed_total_counts'
) -> pd.DataFrame:
    """
    Compare results from multiple pipeline runs.
    
    Parameters
    ----------
    results_list : list of dict
        List of results dictionaries from run_spot_pipeline_v2
    metric : str
        Metric to compare (default: 'unmixed_total_counts')
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table
    """
    records = []
    for idx, result in enumerate(results_list):
        config = result['config']
        record = {
            'run_id': idx,
            'expt_key': config.expt_key,
            'min_distances': str(config.min_distances),
            'dist_cutoff': config.dist_cutoff,
            'corr_cutoff': config.corr_cutoff,
            'cent_cutoff': config.cent_cutoff,
            'do_clean_spots': config.do_clean_spots,
            'n_spots_cleaned': len(result['spots_df_cleaned']),
            'n_spots_final': len(result['spots_df_full']),
            'unmixed_total': result['unmixed_results']['spot_count'].sum(),
            'mixed_total': result['mixed_results']['spot_count'].sum(),
            'output_folder': str(result['output_folder']),
            'has_reassignment_matrix': result['reassignment_matrix'] is not None,
        }
        records.append(record)
    
    return pd.DataFrame(records)


def load_batch_matrices(
    batch_summary_df: pd.DataFrame,
    matrix_type: str = 'ratios'
) -> Dict[int, pd.DataFrame]:
    """
    Load matrices from a batch experiment.
    
    Parameters
    ----------
    batch_summary_df : pd.DataFrame
        Summary dataframe from run_batch_experiment or run_multi_dataset_batch_experiment
    matrix_type : str
        Type of matrix to load: 'ratios', 'reassignment', or 'spot_fate'
    
    Returns
    -------
    matrices : dict
        Dictionary mapping run_id to matrix DataFrame
    """
    if matrix_type == 'ratios':
        path_col = 'batch_ratios_path'
    elif matrix_type == 'reassignment':
        path_col = 'batch_reassignment_path'
    elif matrix_type == 'spot_fate':
        path_col = 'batch_spot_fate_path'
    else:
        raise ValueError("matrix_type must be 'ratios', 'reassignment', or 'spot_fate'")
    
    matrices = {}
    for idx, row in batch_summary_df.iterrows():
        if row['success'] and path_col in row:
            matrix_path = Path(row[path_col])
            if matrix_path.exists():
                if matrix_type in ['ratios', 'spot_fate']:
                    matrices[row['run_id']] = pd.read_csv(matrix_path)
                else:
                    matrices[row['run_id']] = pd.read_csv(matrix_path, index_col=0)
    
    return matrices


def analyze_multi_batch_results(
    summary_df: pd.DataFrame,
    group_by: List[str] = None
) -> pd.DataFrame:
    """
    Analyze results from multi-dataset batch experiment.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe from run_multi_dataset_batch_experiment
    group_by : list of str, optional
        Columns to group by for aggregation (e.g., ['mouse_id', 'min_distances'])
        If None, groups by mouse_id and round_key
    
    Returns
    -------
    analysis_df : pd.DataFrame
        Aggregated analysis results
    """
    if group_by is None:
        group_by = ['mouse_id', 'round_key']
    
    # Filter successful runs
    success_df = summary_df[summary_df['success']].copy()
    
    if len(success_df) == 0:
        print("WARNING: No successful runs to analyze")
        return pd.DataFrame()
    
    # Aggregate metrics
    agg_dict = {
        'n_spots_total': ['mean', 'std', 'min', 'max'],
        'n_spots_final': ['mean', 'std', 'min', 'max'],
        'unmixed_total_counts': ['mean', 'std', 'min', 'max'],
        'mixed_total_counts': ['mean', 'std', 'min', 'max'],
        'run_id': 'count'
    }
    
    analysis_df = success_df.groupby(group_by).agg(agg_dict).round(2)
    analysis_df.columns = ['_'.join(col).strip() for col in analysis_df.columns.values]
    analysis_df.rename(columns={'run_id_count': 'n_runs'}, inplace=True)
    
    return analysis_df.reset_index()


def get_failed_runs_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed summary of failed runs.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe from batch experiment
    
    Returns
    -------
    failed_df : pd.DataFrame
        DataFrame with failed runs and error information
    """
    failed_df = summary_df[~summary_df['success']].copy()
    
    if len(failed_df) == 0:
        print("No failed runs!")
        return pd.DataFrame()
    
    # Select relevant columns
    cols = ['run_id', 'mouse_id', 'round_key', 'error', 'retry_count']
    
    # Add parameter columns if they exist
    param_cols = [c for c in failed_df.columns if c not in cols and c not in [
        'success', 'error_trace', 'n_spots_total', 'n_spots_cleaned', 
        'n_spots_final', 'n_cells', 'unmixed_total_counts', 'mixed_total_counts',
        'output_folder', 'ratio_path', 'reassignment_matrix_path', 
        'ratios_matrix_path', 'batch_ratios_path', 'batch_reassignment_path'
    ]]
    
    display_cols = cols + param_cols
    display_cols = [c for c in display_cols if c in failed_df.columns]
    
    return failed_df[display_cols]


def run_multi_dataset_batch_experiment(
    datasets: Dict[str, Any],
    mouse_ids: List[str],
    round_keys: List[str],
    base_config: SpotPipelineConfig,
    param_grid: Any,  # Can be Dict[str, List] or List[Dict]
    output_key: str = "multi_batch",
    skip_on_error: bool = True,
    max_retries: int = 0,
    return_results: bool = False
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run batch experiments across multiple mice and rounds with error handling.
    
    Parameters
    ----------
    datasets : dict
        Dictionary of HCRDataset objects
    mouse_ids : list of str
        List of mouse identifiers to process
    round_keys : list of str
        List of round keys to process (e.g., ["R2", "R3", "R4", "R5"])
    base_config : SpotPipelineConfig
        Base configuration to use as starting point
    param_grid : dict or list of dict
        If dict: Dictionary of parameter names to lists of values (creates cross-product)
        If list of dict: Each dict is a separate parameter combination (no cross-product)
    output_key : str, default="multi_batch"
        Key to append to experiment name
    skip_on_error : bool, default=True
        If True, skip failed datasets and continue. If False, raise exception.
    max_retries : int, default=0
        Number of times to retry failed runs
    return_results : bool, default=False
        If True, return full results dictionaries. If False, only return summary_df.
        Set to False for large batch experiments to save memory.
    
    Returns
    -------
    summary_df : pd.DataFrame
        Summary of results for all datasets and parameter combinations
    all_results : list of dict
        Full results for successful runs (empty list if return_results=False)
    
    Examples
    --------
    >>> # Dict format: creates cross-product of parameters
    >>> param_grid = {
    ...     'min_distances': [[1], [2], [3]],
    ...     'dist_cutoff': [0.5, 1.0, 1.5]
    ... }
    
    >>> # List of dict format: each dict is a separate combination (no cross-product)
    >>> param_grid = [
    ...     {'unmixing_method': 'reassignment', 'min_distances': [3]},
    ...     {'unmixing_method': 'pairwise', 'min_distances': [1]},
    ... ]
    
    >>> summary_df, results = run_multi_dataset_batch_experiment(
    ...     datasets, mouse_ids, round_keys, base_config, param_grid
    ... )
    """
    from itertools import product
    from copy import deepcopy
    import traceback
    
    print("=" * 80)
    print("RUNNING MULTI-DATASET BATCH EXPERIMENT")
    print(f"Mice: {mouse_ids}")
    print(f"Rounds: {round_keys}")
    print(f"Parameter grid: {param_grid}")
    print(f"Skip on error: {skip_on_error}")
    print(f"Return full results: {return_results}")
    print("=" * 80)
    
    # Handle both dict and list of dicts formats
    if isinstance(param_grid, dict):
        # Dict format: create cross-product of all parameter values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]
        print(f"\nParam grid format: dict (cross-product)")
    elif isinstance(param_grid, list):
        # List of dicts format: each dict is a separate combination
        param_combinations = param_grid
        print(f"\nParam grid format: list of dicts (no cross-product)")
    else:
        raise ValueError(f"param_grid must be dict or list of dicts, got {type(param_grid)}")
    
    # Total runs = mice × rounds × param_combinations
    total_runs = len(mouse_ids) * len(round_keys) * len(param_combinations)
    print(f"Total runs: {total_runs} ({len(mouse_ids)} mice × {len(round_keys)} rounds × {len(param_combinations)} param combos)\n")
    
    if not return_results:
        print("NOTE: Full results will NOT be returned (return_results=False). Only summary DataFrame will be returned.")
        print("      This saves memory for large batch experiments. All outputs are saved to disk.\n")
    
    all_results = [] if return_results else None
    summary_records = []
    run_id = 0
    
    # Pre-detect channel pairs for pairwise unmixing method
    # This allows different datasets to have different channel counts
    dataset_channel_pairs = {}
    
    for mouse_id in mouse_ids:
        if mouse_id not in datasets:
            continue
        for round_key in round_keys:
            try:
                spot_channels = get_spot_channels_from_dataset(datasets[mouse_id], round_key)
                default_pairs = get_default_channel_pairs(spot_channels)
                dataset_channel_pairs[(mouse_id, round_key)] = default_pairs
                print(f"Pre-detected channel pairs for {mouse_id} - {round_key}: {default_pairs}")
            except Exception as e:
                print(f"Warning: Could not detect channel pairs for {mouse_id} - {round_key}: {e}")
                dataset_channel_pairs[(mouse_id, round_key)] = None
    
    print("\n" + "="*80)
    print("Starting batch runs...")
    print("="*80 + "\n")
    
    for mouse_id in mouse_ids:
        for round_key in round_keys:
            # Check if dataset exists
            if mouse_id not in datasets:
                print(f"\n{'='*80}")
                print(f"WARNING: Mouse {mouse_id} not found in datasets. Skipping.")
                print(f"{'='*80}\n")
                continue
            
            for param_idx, param_dict in enumerate(param_combinations, 1):
                run_id += 1
                print(f"\n{'='*80}")
                print(f"Run {run_id}/{total_runs}: Mouse={mouse_id}, Round={round_key}, Params={param_idx}/{len(param_combinations)}")
                print(f"{'='*80}")
                
                # Create config for this run
                config = deepcopy(base_config)
                config.expt_key = f"{base_config.expt_key}_{output_key}"
                
                # Set parameters
                print(f"Parameters: {param_dict}")
                
                for param_name, param_value in param_dict.items():
                    setattr(config, param_name, param_value)
                
                # Auto-detect channel pairs for pairwise unmixing if not explicitly set
                if config.unmixing_method == 'pairwise' and config.channel_pairs is None:
                    detected_pairs = dataset_channel_pairs.get((mouse_id, round_key))
                    if detected_pairs is None:
                        error_msg = f"Could not detect channel pairs for pairwise unmixing (round may not exist)"
                        print(f"  ✗ SKIPPING: {error_msg}")
                        
                        # Record the skip
                        summary = {
                            'run_id': run_id,
                            'mouse_id': mouse_id,
                            'round_key': round_key,
                            **param_dict,
                            'success': False,
                            'error': error_msg,
                            'retry_count': 0
                        }
                        summary_records.append(summary)
                        continue  # Skip this run
                    
                    config.channel_pairs = detected_pairs
                    print(f"  Auto-set channel_pairs for pairwise: {config.channel_pairs}")
                
                # Update experiment key to include dataset and parameters
                param_str = "_".join([f"{k}={v}" for k, v in param_dict.items()])
                config.expt_key = f"{config.expt_key}/{mouse_id}_{round_key}/{param_str}"
                
                # Try running with retries
                success = False
                error_msg = None
                retry_count = 0
                
                while not success and retry_count <= max_retries:
                    try:
                        if retry_count > 0:
                            print(f"  Retry attempt {retry_count}/{max_retries}")
                        
                        results = run_spot_pipeline_v2(
                            datasets,
                            mouse_id,
                            round_key,
                            pipeline_config=config,
                            skip_visualizations=True
                        )
                        
                        # Extract summary metrics
                        summary = {
                            'run_id': run_id,
                            'mouse_id': mouse_id,
                            'round_key': round_key,
                            **param_dict,
                            'n_spots_total': len(results['pipeline_data'].spots_df),
                            'n_spots_cleaned': len(results['spots_df_cleaned']),
                            'n_spots_final': len(results['spots_df_full']),
                            'n_cells': len(results['pipeline_data'].cell_info),
                            'unmixed_total_counts': results['unmixed_results']['spot_count'].sum(),
                            'mixed_total_counts': results['mixed_results']['spot_count'].sum(),
                            'output_folder': str(results['output_folder']),
                            'ratio_path': str(results['ratio_path']),
                            'reassignment_matrix_path': str(results['output_folder'] / f"{round_key}_reassignment_matrix.csv"),
                            'ratios_matrix_path': str(results['output_folder'] / f"{round_key}_ratios_matrix.csv"),
                            'success': True,
                            'error': None,
                            'retry_count': retry_count
                        }
                        
                        # Save individual matrices for this run
                        batch_matrices_folder = base_config.output_base_folder / f"{base_config.expt_key}_{output_key}" / "matrices"
                        batch_matrices_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Copy matrices to batch folder
                        combo_name = f"{mouse_id}_{round_key}_{param_str}"
                        
                        # Save ratios matrix
                        ratios_batch_path = batch_matrices_folder / f"{combo_name}_ratios.csv"
                        results['ratios_matrix'].to_csv(ratios_batch_path, index=False)
                        
                        # Save reassignment matrix
                        reassign_batch_path = batch_matrices_folder / f"{combo_name}_reassignment.csv"
                        results['reassignment_matrix'].to_csv(reassign_batch_path)
                        
                        summary['batch_ratios_path'] = str(ratios_batch_path)
                        summary['batch_reassignment_path'] = str(reassign_batch_path)
                        
                        # Only append results if requested
                        if return_results:
                            all_results.append(results)
                        
                        success = True
                        print(f"✓ SUCCESS")
                        
                    except Exception as e:
                        error_msg = str(e)
                        error_trace = traceback.format_exc()
                        print(f"✗ ERROR: {error_msg}")
                        
                        if retry_count < max_retries:
                            retry_count += 1
                        else:
                            if skip_on_error:
                                print(f"  Skipping after {retry_count} retries...")
                                summary = {
                                    'run_id': run_id,
                                    'mouse_id': mouse_id,
                                    'round_key': round_key,
                                    **param_dict,
                                    'success': False,
                                    'error': error_msg,
                                    'error_trace': error_trace,
                                    'retry_count': retry_count
                                }
                                success = True  # Exit retry loop
                            else:
                                print(f"  Raising exception (skip_on_error=False)")
                                raise
                
                summary_records.append(summary)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_records)
    
    # Save summary
    summary_path = base_config.output_base_folder / f"{base_config.expt_key}_{output_key}" / "multi_batch_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    # Print final statistics
    n_success = summary_df['success'].sum()
    n_failed = len(summary_df) - n_success
    
    print("\n" + "=" * 80)
    print("MULTI-DATASET BATCH EXPERIMENT COMPLETE!")
    print(f"Summary saved to: {summary_path}")
    print(f"Total runs: {len(summary_df)}")
    print(f"Successful: {n_success} ({n_success/len(summary_df)*100:.1f}%)")
    print(f"Failed: {n_failed} ({n_failed/len(summary_df)*100:.1f}%)")
    
    if n_failed > 0:
        print("\nFailed runs:")
        failed_df = summary_df[~summary_df['success']][['run_id', 'mouse_id', 'round_key', 'error']]
        print(failed_df.to_string())
    
    if not return_results:
        print("\nNOTE: Full results not returned. All outputs saved to disk.")
        print(f"      Matrices folder: {base_config.output_base_folder / f'{base_config.expt_key}_{output_key}' / 'matrices'}")
    
    print("=" * 80)
    
    return summary_df, (all_results if return_results else [])


if __name__ == "__main__":
    print("spot_pipeline_refactored.py loaded successfully!")
    print("\nExample usage:")
    print("""
    # Single run
    config = SpotPipelineConfig(
        expt_key="test_run",
        min_distances=[2, 3, 5],
        dist_cutoff=1.5
    )
    results = run_spot_pipeline_v2(datasets, "mouse123", "R3", config)
    
    # Access matrices
    print(results['ratios_matrix'])
    print(results['reassignment_matrix'])
    
    # Single mouse/round batch experiment
    param_grid = {
        'min_distances': [[1], [2], [3], [5]],
        'dist_cutoff': [0.5, 1.0, 1.5]
    }
    summary, all_results = run_batch_experiment(
        datasets, "mouse123", "R3", config, param_grid
    )
    
    # Multi-dataset batch experiment (NEW!)
    mouse_ids = ["754803", "767018", "767022"]
    round_keys = ["R3", "R4", "R5"]
    param_grid = {
        'min_distances': [[1], [2], [3]],
        'dist_cutoff': [0.5, 1.0, 1.5]
    }
    summary_df, all_results = run_multi_dataset_batch_experiment(
        datasets, mouse_ids, round_keys, base_config, param_grid,
        skip_on_error=True  # Skip failed datasets
    )
    
    # Load all reassignment matrices from batch
    reassignment_matrices = load_batch_matrices(summary_df, 'reassignment')
    ratios_matrices = load_batch_matrices(summary_df, 'ratios')
    """)
