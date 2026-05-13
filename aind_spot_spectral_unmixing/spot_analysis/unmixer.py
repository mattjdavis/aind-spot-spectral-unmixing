import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List, Dict, Tuple
from .config import Config

class SpotUnmixer:
    def __init__(self, dataset_folder: str, config, channel_pairs=None, spatial_scale=None):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Pairwise unmixing configuration
        self.channel_pairs = channel_pairs if channel_pairs is not None else []
        self.spatial_scale = spatial_scale if spatial_scale is not None else np.array([1.0, 0.24, 0.24])  # (z, y, x) um/pixel
        # Ellipsoidal NN search radii (µm) — read from config if available, else fall back to defaults
        self.r_xy_um = getattr(config, 'R_XY_UM', 0.5)   # lateral semi-axis
        self.r_z_um  = getattr(config, 'R_Z_UM',  1.0)   # axial semi-axis  (PSF is ~2x wider in z)
    
    def calculate_distances(
        self,
        spots_df: pd.DataFrame,
        ratios: np.ndarray
    ) -> pd.DataFrame:
        """Calculate distances between spots and ratio lines"""
        intensity_cols = [
            f'chan_{ch}_intensity'
            for ch in self.config.get_round_spot_channels()
        ]
        
        # Convert to GPU tensors
        data_gpu = torch.from_numpy(
            np.array(spots_df[intensity_cols])
        ).to(self.device).double()
        ratios_gpu = torch.from_numpy(
            ratios / np.linalg.norm(ratios, axis=0)
        ).to(self.device).double()
        
        # Calculate fits and distances
        n_cam = len(self.config.get_round_spot_channels())
        fit = torch.tile(data_gpu @ ratios_gpu, (n_cam, 1, 1))
        fit *= torch.tile(
            torch.unsqueeze(ratios_gpu, 1),
            (1, len(spots_df), 1)
        )
        
        data_gpu = torch.tile(
            torch.unsqueeze(data_gpu.T, 2),
            (1, 1, n_cam)
        )
        
        distances = torch.norm(fit - data_gpu, dim=0).cpu().numpy()
        
        stats_df = self._create_stats_dataframe(distances)

        # Stamp spot_uid_int as a data column so downstream code can safely
        # merge stats_df back onto any filtered / re-indexed spots dataframe
        # without relying on positional index alignment.
        if 'spot_uid_int' in spots_df.columns:
            stats_df['spot_uid_int'] = spots_df['spot_uid_int'].values
        else:
            stats_df['spot_uid_int'] = spots_df.index.values

        return stats_df
    
    def _create_stats_dataframe(self, distances: np.ndarray) -> pd.DataFrame:
        """Create statistics dataframe from distances"""
        dist_rank = np.argsort(distances)[:, :2]
        small0 = distances[np.arange(len(dist_rank)), dist_rank[:, 0]]
        small1 = distances[np.arange(len(dist_rank)), dist_rank[:, 1]]
        r = small1 / small0
        
        stats = np.vstack((dist_rank.T, small0, small1, r)).T.astype(np.float32)
        stats_df = pd.DataFrame(
            stats,
            columns=['dist_rank_1', 'dist_rank_2', 'small0', 'small1', 'dist_r']
        )
        
        # Add full distance columns for pairwise unmixing
        for idx, channel in enumerate(self.config.get_round_spot_channels()):
            stats_df[f'dist_to_chan_{channel}'] = distances[:, idx]
        
        return stats_df
    
    def _filter_spatial_matches(
        self,
        spots: pd.DataFrame,
        stats: pd.DataFrame,
        intensities: pd.DataFrame,
        channel_idx: int,
        min_dist: float
    ) -> np.ndarray:
        """Filter spots based on spatial proximity"""
        # Use column NAME instead of positional index to avoid column-order bugs
        channels = self.config.get_round_spot_channels()
        channel_name = str(channels[channel_idx])
        intensity_col = f'chan_{channel_name}_intensity'
        if intensity_col in intensities.columns:
            keep = np.array(intensities[intensity_col] > 0)
        else:
            # Fallback to positional (original behavior) with warning
            print(f"WARNING: '{intensity_col}' not found in intensities columns {list(intensities.columns)}. "
                  f"Falling back to positional index {channel_idx}.")
            keep = np.array(intensities.iloc[:, channel_idx] > 0)
        
        # Build KD-tree for spatial matching
        spatial_matches = cKDTree(
            spots[['z', 'y', 'x']]
        ).query_pairs(min_dist)
        # debug
        print("n_pairs:", len(spatial_matches), "min_dist:", min_dist)

        same_cell = 0
        both_keep = 0
        for a,b in spatial_matches:
            if spots.iloc[a]["cell_id"] == spots.iloc[b]["cell_id"]:
                same_cell += 1
                if keep[a] and keep[b]:
                    both_keep += 1
        print("pairs total:", len(spatial_matches), "same_cell:", same_cell, "both_keep:", both_keep)
        # debug

        # Process spatial matches
        for a, b in spatial_matches:
            # Skip if spots are from different cells
            if spots.iloc[a]['cell_id'] != spots.iloc[b]['cell_id']:
                continue
                
            # Skip if either point is already removed
            if not keep[a] or not keep[b]:
                continue
                
            # Remove point with worse ratio match
            if stats.iloc[a]['dist_r'] > stats.iloc[b]['dist_r']:
                keep[b] = 0
                #print("remove 1", spots.iloc[b]['cell_id']) # debug
            else:
                #print("remove 2", spots.iloc[a]['cell_id'])
                keep[a] = 0
                
        return keep
    
    def _process_channel(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        channel_idx: int,
        channel: str,
        min_dist: float,
        reassignment: bool
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Process spots for a single channel"""
        # Get channel matching spots

        if reassignment:
            #TODO link stats_df to spots_df here.... currently stats_df is unfiltered by qc (intensity, shape, etc)
            dist_rank_chan_match = np.array(stats_df['dist_rank_1'] == channel_idx)
            chan_spots = spots_df.loc[dist_rank_chan_match].copy()
            chan_stats = stats_df.loc[dist_rank_chan_match].copy()
        else:
            # Pure spatial de-dupe within ORIGINAL (detected) channel.
            # Assumes stats_df rows align 1:1 with spots_df rows (same index / same filtering).
            # If stats_df is unfiltered while spots_df is QC-filtered, fix upstream by reindexing/merging.
            chan_match = np.array(spots_df['chan'].astype(str) == str(channel))
            chan_spots = spots_df.loc[chan_match].copy()
            chan_stats = stats_df.loc[chan_match].copy()

            # debug
            print(f"DEBUG Processing channel: {channel} (index {channel_idx}), n spots = {len(chan_spots)}")
            print("unique spots_df['chan']:", sorted(map(str, spots_df["chan"].unique()))[:20])
            print("config channels:", list(map(str, self.config.get_round_spot_channels())))
            
        # Get intensities for the channel
        intensity_cols = [
            f'chan_{ch}_intensity'
            for ch in self.config.get_round_spot_channels()
        ]

        if reassignment:
            chan_intensities = spots_df.loc[dist_rank_chan_match, intensity_cols].copy()
        else:
            chan_intensities = spots_df.loc[chan_match, intensity_cols].copy()

        # Filter spots based on spatial proximity
        keep = self._filter_spatial_matches(
            chan_spots,
            chan_stats,
            chan_intensities,
            channel_idx,
            min_dist
        )
        
        # Apply filtering
        filtered_spots = chan_spots.loc[keep].copy()
        filtered_spots['unmixed_chan'] = channel
        
        # Calculate statistics
        stats = {
            'total_spots': len(chan_spots),
            'kept_spots': keep.sum(),
            'reassigned_spots': len(filtered_spots) - len(filtered_spots.loc[filtered_spots['chan'] == channel])
        }
        
        return filtered_spots, stats
    
    def unmix_spots(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        min_dist: float,
        unmixing_method: str = 'reassignment',
        channel_pairs: List[Tuple[str, str]] = None,
        r_xy_um: float = None,
        r_z_um: float = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
        """
        Unmix spots based on calculated statistics
        
        Args:
            spots_df: DataFrame containing spot data
            stats_df: DataFrame containing distance statistics
            min_dist: Minimum distance between spots for spatial filtering (in physical units, um)
            unmixing_method: 'reassignment' (default) or 'pairwise'
            channel_pairs: List of (chanA, chanB) tuples for pairwise method
            
        Returns:
            Tuple containing:
            - DataFrame of unmixed spots
            - List of dictionaries containing statistics for each channel/pair

        Note: 
        MJD suggests approach where we _don't_ iterate over spots by/within channels in _filter_spatial_matches(), 
        but instead perform our _spatial_ analysis on more systematic level, such as running (multiscale? like spotsweeper)
        KNN on all spots, and then globally removing neighbours that are too close, and/or (following spotsweeper) running clustering (PCA) on those results. 
        """
        
        if unmixing_method == 'reassignment':
            return self._unmix_spots_reassignment(spots_df, stats_df, min_dist)
        elif unmixing_method == 'pairwise':
            if channel_pairs is None:
                channel_pairs = self.channel_pairs
            return self._unmix_spots_pairwise(
                spots_df, stats_df, min_dist, channel_pairs,
                r_xy_um=r_xy_um if r_xy_um is not None else self.r_xy_um,
                r_z_um=r_z_um if r_z_um is not None else self.r_z_um,
            )
        else:
            raise ValueError(f"Unknown unmixing_method: {unmixing_method}. Use 'reassignment' or 'pairwise'")
    
    def _unmix_spots_reassignment(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        min_dist: float
    ) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
        """
        Original unmixing: reassign spots to best-fit channel, then remove spatial overlaps
        
        Args:
            spots_df: DataFrame containing spot data
            stats_df: DataFrame containing distance statistics
            min_dist: Minimum distance between spots for spatial filtering
            
        Returns:
            Tuple containing:
            - DataFrame of unmixed spots
            - List of dictionaries containing statistics for each channel
        """
        unmixed_spots = []
        channel_stats = []
        
        # Process each channel
        for idx, channel in enumerate(self.config.get_round_spot_channels()):
            channel = str(channel)
            gene = self.config.get_round_channels()[channel]
            
            # Process spots for this channel
            channel_spots, stats = self._process_channel(
                spots_df,
                stats_df,
                idx,
                channel,
                min_dist,
                reassignment=True  # Always use reassignment in this method
            )
            
            # Add gene name to stats
            stats['gene'] = gene
            stats['channel'] = channel
            
            # Store results
            unmixed_spots.append(channel_spots)
            channel_stats.append(stats)
            
        # Combine results
        unmixed_df = pd.concat(unmixed_spots, ignore_index=True)
        
        # Normalize chan/unmixed_chan to str to prevent category vs object mismatches downstream
        unmixed_df['chan'] = unmixed_df['chan'].astype(str)
        unmixed_df['unmixed_chan'] = unmixed_df['unmixed_chan'].astype(str)
        
        self._save_results(unmixed_df, min_dist)
            
        return unmixed_df, channel_stats
    
    def _unmix_spots_pairwise(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        min_dist: float,
        channel_pairs: List[Tuple[str, str]],
        r_xy_um: float = None,
        r_z_um: float = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
        """
        Remove crosstalk by comparing spatially overlapping spots between channel pairs
        
        Args:
            spots_df: Spot locations and detected channels
            stats_df: Distance to each dye line (must align with spots_df index)
            min_dist: Minimum spatial distance in physical units (um) — used only for
                      reassignment path; pairwise uses r_xy_um / r_z_um ellipsoid.
            channel_pairs: List of (chanA, chanB) tuples to check for crosstalk
            r_xy_um: Lateral semi-axis of the ellipsoidal search volume (µm).
            r_z_um:  Axial semi-axis of the ellipsoidal search volume (µm).
        
        Returns:
            Tuple containing:
            - DataFrame of filtered spots
            - List of dictionaries containing statistics for each pair
        """
        # Validate alignment
        assert len(spots_df) == len(stats_df), "spots_df and stats_df must have same length"
        assert spots_df.index.equals(stats_df.index), "Indices must match"
        
        # Initialize global keep mask
        keep = np.ones(len(spots_df), dtype=bool)
        
        # Track statistics
        pair_stats = []
        
        _r_xy = r_xy_um if r_xy_um is not None else self.r_xy_um
        _r_z  = r_z_um  if r_z_um  is not None else self.r_z_um

        print(f"  [pairwise] Ellipsoidal NN search active: "
              f"r_xy={_r_xy} µm, r_z={_r_z} µm  "
              f"(axial/lateral ratio = {_r_z/_r_xy:.2f}x)")

        # Process each channel pair
        for chanA, chanB in channel_pairs:
            print(f"Processing channel pair: {chanA} - {chanB}")
            stats = self._filter_pairwise_crosstalk(
                spots_df,
                stats_df,
                chanA,
                chanB,
                _r_xy,
                _r_z,
                keep  # Modified in-place
            )
            pair_stats.append(stats)
            print(f"  Total overlaps: {stats['total_overlaps']}, Same cell: {stats['same_cell_overlaps']}")
            print(f"  Removed from {chanA}: {stats['removed_from_A']}, from {chanB}: {stats['removed_from_B']}")
        
        # Apply final filtering
        filtered_spots = spots_df[keep].copy()
        filtered_spots['unmixed_chan'] = filtered_spots['chan']  # Keep original channel
        
        self._save_results(filtered_spots, _r_xy)
        
        return filtered_spots, pair_stats
    
    def _filter_pairwise_crosstalk(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        chanA: str,
        chanB: str,
        r_xy_um: float,
        r_z_um: float,
        keep: np.ndarray
    ) -> Dict[str, int]:
        """
        Filter crosstalk between a specific channel pair using an ellipsoidal
        NN search volume that matches the microscope PSF anisotropy.

        Args:
            chanA, chanB: Channel identifiers (e.g., '488', '514')
            r_xy_um: Lateral semi-axis of the search ellipsoid (µm)
            r_z_um:  Axial semi-axis of the search ellipsoid (µm)
            keep: Global boolean mask (modified in-place)

        Returns:
            Statistics dict for this pair
        """
        # Get channel indices
        channels = [str(ch) for ch in self.config.get_round_spot_channels()]
        try:
            idxA = channels.index(str(chanA))
            idxB = channels.index(str(chanB))
        except ValueError as e:
            print(f"Warning: Channel not found in config. Available channels: {channels}")
            raise e
        
        # Extract spots detected in each channel (that haven't been removed yet)
        maskA = (spots_df['chan'].astype(str) == str(chanA)) & keep
        maskB = (spots_df['chan'].astype(str) == str(chanB)) & keep
        
        spotsA = spots_df[maskA]
        spotsB = spots_df[maskB]
        
        if len(spotsA) == 0 or len(spotsB) == 0:
            return {
                'pair': f'{chanA}-{chanB}',
                'total_overlaps': 0,
                'same_cell_overlaps': 0,
                'removed_from_A': 0,
                'removed_from_B': 0,
                'kept_A': maskA.sum(),
                'kept_B': maskB.sum()
            }
        
        # Convert pixels -> µm, then normalise to unit-ellipsoid coordinates:
        #   (z_um/r_z, y_um/r_xy, x_um/r_xy).  A pair is a candidate iff its
        #   normalised Euclidean distance <= 1.0.
        spatial_scale = np.array(self.spatial_scale)  # (z, y, x) µm/px
        ellipsoid_scale = np.array([1.0 / r_z_um, 1.0 / r_xy_um, 1.0 / r_xy_um])  # (z, y, x)
        norm_scale = spatial_scale * ellipsoid_scale

        posA = spotsA[['z', 'y', 'x']].values * norm_scale
        posB = spotsB[['z', 'y', 'x']].values * norm_scale

        # Build KD-tree for channel B (unit-ellipsoid space)
        treeB = cKDTree(posB)

        # Find spatial overlaps: threshold = 1.0 in normalised space
        pairs = treeB.query_ball_point(posA, 1.0)  # Returns list of lists
        
        # Track removals
        removed_A = 0
        removed_B = 0
        total_overlaps = 0
        same_cell_overlaps = 0
        
        # Process each spot in A
        for i, neighbors_in_B in enumerate(pairs):
            if len(neighbors_in_B) == 0:
                continue
                
            spot_a_idx = spotsA.index[i]  # Original index in spots_df
            
            # Check each neighbor in B
            for j in neighbors_in_B:
                spot_b_idx = spotsB.index[j]
                
                total_overlaps += 1
                
                # Only process if same cell
                if spotsA.iloc[i]['cell_id'] != spotsB.iloc[j]['cell_id']:
                    continue
                
                same_cell_overlaps += 1
                
                # Skip if either already removed
                if not keep[spot_a_idx] or not keep[spot_b_idx]:
                    continue
                
                # Get distances to RESPECTIVE dye lines
                distA = stats_df.loc[spot_a_idx, f'dist_to_chan_{chanA}']
                distB = stats_df.loc[spot_b_idx, f'dist_to_chan_{chanB}']
                
                # Remove spot with WORSE fit to its own dye line
                if distA > distB:
                    keep[spot_a_idx] = False
                    removed_A += 1
                else:
                    keep[spot_b_idx] = False
                    removed_B += 1
        
        return {
            'pair': f'{chanA}-{chanB}',
            'total_overlaps': total_overlaps,
            'same_cell_overlaps': same_cell_overlaps,
            'removed_from_A': removed_A,
            'removed_from_B': removed_B,
            'kept_A': maskA.sum() - removed_A,
            'kept_B': maskB.sum() - removed_B
        }
    
    def build_removed_spots(
        self,
        spots_df_mixed: pd.DataFrame,
        unmixed_df: pd.DataFrame,
        min_dist: float,
    ) -> pd.DataFrame:
        """
        Identify spots present in the mixed table that were removed during unmixing.

        Removal happens for two reasons:
        1. Spatial de-duplication (two spots in the same cell within min_dist).
        2. Channel re-assignment followed by conflict resolution.

        The returned DataFrame is the **reference population** used to calculate
        ``z_intensity_vs_removed`` in ``compute_crosstalk_scores``.  It is also
        saved as ``removed_spots_R{N}_minDist_{d}.pkl`` for offline QC.

        Parameters
        ----------
        spots_df_mixed : pd.DataFrame
            All spots entering the unmixer (post-QC geometric filters, pre-spatial dedup).
            Must have a ``spot_uid_int`` column.
        unmixed_df : pd.DataFrame
            Spots that survived unmixing.  Must have a ``spot_uid_int`` column.
        min_dist : float
            Minimum distance used for this unmixing pass (used only for file naming).

        Returns
        -------
        pd.DataFrame
            Rows from ``spots_df_mixed`` whose ``spot_uid_int`` is absent in
            ``unmixed_df``.  An extra boolean column ``removed=True`` is added.
        """
        survived_uids = set(unmixed_df['spot_uid_int'].values)
        removed_mask = ~spots_df_mixed['spot_uid_int'].isin(survived_uids)
        removed_df = spots_df_mixed[removed_mask].copy()
        removed_df['removed'] = True

        n_removed = len(removed_df)
        n_total = len(spots_df_mixed)
        print(f"  [crosstalk] Removed spots: {n_removed}/{n_total} "
              f"({100 * n_removed / max(n_total, 1):.1f}%) for min_dist={min_dist}")

        # Save to disk
        out_path = self.config.OUTPUT_FOLDER / (
            f'removed_spots_R{self.config.ROUND_N}_minDist_{int(min_dist)}.pkl'
        )
        scratch_path = self.config.SCRATCH_FOLDER / (
            f'removed_spots_R{self.config.ROUND_N}_minDist_{int(min_dist)}.pkl'
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        removed_df.to_pickle(out_path)
        if scratch_path != out_path:
            scratch_path.parent.mkdir(parents=True, exist_ok=True)
            removed_df.to_pickle(scratch_path)

        return removed_df

    def compute_crosstalk_scores(
        self,
        unmixed_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        removed_df: pd.DataFrame,
        chan_order: List[str] = None,
        z_threshold: float = None,
    ) -> pd.DataFrame:
        """
        Annotate unmixed spots with spectral crosstalk quality metrics.

        Two complementary spectral metrics are computed:

        ``dye_line_dist_ratio``
            Already present on spots after ``apply_qc_filters`` (alias of ``dist_r``
            from ``stats_df``).  Global spectral ambiguity — channel-agnostic,
            computed *before* assignment: ``d_2nd_closest / d_closest``.

        ``d_assignment_ratio``
            **New** post-assignment metric.  Uses the assigned channel
            (``unmixed_chan``) specifically:
            ``dist_to_assigned_chan / min(dist_to_all_other_chans)``.
            Values > 1 mean the spot fits another channel's dye line better than
            its own — a direct indicator of crosstalk.

        ``z_intensity_vs_removed``
            Robust z-score of the spot's own-channel intensity vs the *removed*
            population for the same channel.  Positive → brighter than removed
            (likely real signal); negative → dimmer (suspect).

        ``crosstalk_score``
            Combined metric:
            ``d_assignment_ratio × (1 + max(0, −z_intensity_vs_removed))``.
            Penalises spectrally ambiguous spots that are also *dimmer* than the
            typical removed spot.

        ``z_vetoed``
            Boolean.  True when ``z_intensity_vs_removed > z_threshold``; the spot
            is unconditionally kept (``crosstalk_score`` forced to 0) regardless
            of spectral purity.

        Parameters
        ----------
        unmixed_df : pd.DataFrame
            Output of the unmixer; must have ``spot_uid_int`` and ``unmixed_chan``.
        stats_df : pd.DataFrame
            Output of ``calculate_distances``; must have ``spot_uid_int`` and
            ``dist_to_chan_{ch}`` columns for every spot channel.
        removed_df : pd.DataFrame
            Output of ``build_removed_spots``; used as the reference intensity
            population.  Must have intensity columns ``chan_{ch}_intensity`` and a
            ``chan`` column.
        chan_order : list of str, optional
            Ordered list of spot channels.  Defaults to
            ``config.get_round_spot_channels()``.
        z_threshold : float, optional
            Brightness veto threshold.  Defaults to
            ``config.CROSSTALK_Z_THRESHOLD``.

        Returns
        -------
        pd.DataFrame
            Copy of ``unmixed_df`` with additional columns:
            ``d_assignment_ratio``, ``d_assign_neighbor_ratio_1``,
            ``d_assign_neighbor_ratio_2``, ``z_intensity_vs_removed``,
            ``crosstalk_score``, ``z_vetoed``.
            ``dye_line_dist_ratio`` is added/refreshed from ``stats_df`` if not
            already present.

            ``d_assign_neighbor_ratio_1`` — ratio of ``dist_to_assigned_chan``
            to the closer of the two spectrally adjacent channels in
            ``chan_order`` (i.e. ``chan_order[i-1]`` or ``chan_order[i+1]``).

            ``d_assign_neighbor_ratio_2`` — same but for the farther adjacent
            channel.  ``NaN`` for boundary channels (first or last in
            ``chan_order``) that have only one neighbor.
        """
        if chan_order is None:
            chan_order = [str(ch) for ch in self.config.get_round_spot_channels()]
        if z_threshold is None:
            z_threshold = self.config.CROSSTALK_Z_THRESHOLD

        # ── Merge stats columns onto unmixed_df via spot_uid_int ─────────────
        dist_cols = [f'dist_to_chan_{ch}' for ch in chan_order
                     if f'dist_to_chan_{ch}' in stats_df.columns]
        merge_cols = ['spot_uid_int', 'dist_r'] + dist_cols
        # keep only columns that actually exist in stats_df
        merge_cols = [c for c in merge_cols if c in stats_df.columns]

        out = unmixed_df.copy()
        out = out.merge(
            stats_df[merge_cols],
            on='spot_uid_int',
            how='left',
            suffixes=('', '_stats'),
        )

        # Refresh dye_line_dist_ratio from the freshly merged dist_r
        if 'dist_r' in out.columns:
            out['dye_line_dist_ratio'] = out['dist_r']

        # ── d_assignment_ratio (vectorised) ──────────────────────────────────
        d_ratio = np.full(len(out), np.nan)
        if dist_cols:
            chan_labels = [c.replace('dist_to_chan_', '') for c in dist_cols]
            chan_to_idx = {ch: i for i, ch in enumerate(chan_labels)}
            d_matrix = out[dist_cols].values.astype(np.float64)  # (n_spots, n_ch)
            assigned_idx = out['unmixed_chan'].astype(str).map(chan_to_idx)
            valid = assigned_idx.notna().values
            row_idx = np.where(valid)[0]
            col_idx = assigned_idx.dropna().astype(int).values
            if len(row_idx):
                d_assigned = d_matrix[row_idx, col_idx]
                d_others = d_matrix[row_idx].copy()
                d_others[np.arange(len(row_idx)), col_idx] = np.inf
                d_min_other = d_others.min(axis=1)
                d_min_other[d_min_other == 0] = np.nan
                d_ratio[row_idx] = d_assigned / d_min_other
        out['d_assignment_ratio'] = d_ratio

        # ── d_assign_neighbor_ratio_1 & _2 (spectrally adjacent channels) ────
        # For a spot assigned to chan_order[i]:
        #   neighbor distances = [d_matrix[r, i-1], d_matrix[r, i+1]] (where valid)
        #   ratio_1 = d_assigned / closer neighbor
        #   ratio_2 = d_assigned / farther neighbor  (NaN if only one neighbor)
        d_n1 = np.full(len(out), np.nan)
        d_n2 = np.full(len(out), np.nan)
        if dist_cols and len(row_idx):
            n_ch = len(chan_labels)
            # Build left/right neighbor distances; mask invalid (boundary) with NaN
            has_left  = col_idx > 0
            has_right = col_idx < n_ch - 1
            left_col  = np.clip(col_idx - 1, 0, n_ch - 1)
            right_col = np.clip(col_idx + 1, 0, n_ch - 1)

            d_left  = np.where(has_left,  d_matrix[row_idx, left_col],  np.nan)
            d_right = np.where(has_right, d_matrix[row_idx, right_col], np.nan)

            # Stack and sort so neighbor_1 is always the closer one
            neighbor_stack = np.stack([d_left, d_right], axis=1)  # (n_valid, 2)
            neighbor_stack = np.sort(neighbor_stack, axis=1)       # ascending

            n1 = neighbor_stack[:, 0]  # closer neighbor distance
            n2 = neighbor_stack[:, 1]  # farther neighbor distance

            # Only compute ratio where neighbor exists and is non-zero
            with np.errstate(invalid='ignore', divide='ignore'):
                r1 = np.where((~np.isnan(n1)) & (n1 > 0), d_assigned / n1, np.nan)
                r2 = np.where((~np.isnan(n2)) & (n2 > 0), d_assigned / n2, np.nan)

            d_n1[row_idx] = r1
            d_n2[row_idx] = r2
        out['d_assign_neighbor_ratio_1'] = d_n1
        out['d_assign_neighbor_ratio_2'] = d_n2

        # ── z_intensity_vs_removed (per channel) ─────────────────────────────
        z_arr = np.full(len(out), np.nan)
        unmixed_chan_vals = out['unmixed_chan'].astype(str).values

        for ch in chan_order:
            int_col = f'chan_{ch}_intensity'
            if int_col not in out.columns or int_col not in removed_df.columns:
                continue
            row_idx = np.where(unmixed_chan_vals == ch)[0]
            if len(row_idx) == 0:
                continue

            ref_vals = removed_df.loc[
                removed_df['chan'].astype(str) == ch, int_col
            ].dropna()
            if len(ref_vals) < 5:
                continue

            ref_med = ref_vals.median()
            ref_mad = (ref_vals - ref_med).abs().median()
            if ref_mad < 1e-10:
                ref_mad = ref_vals.std()
            if ref_mad < 1e-10:
                continue  # degenerate distribution — skip

            spot_int = out[int_col].values[row_idx].astype(np.float64)
            z_arr[row_idx] = (spot_int - ref_med) / (ref_mad * 1.4826)

        out['z_intensity_vs_removed'] = z_arr

        # ── crosstalk_score + brightness veto ────────────────────────────────
        dim_penalty = np.maximum(0.0, -out['z_intensity_vs_removed'].fillna(0).values)
        score = out['d_assignment_ratio'].values * (1.0 + dim_penalty)

        z_vetoed = out['z_intensity_vs_removed'].values > z_threshold
        score = np.where(z_vetoed, 0.0, score)
        out['z_vetoed'] = z_vetoed
        out['crosstalk_score'] = score

        n_vetoed = int(z_vetoed.sum())
        if n_vetoed:
            print(f"  [crosstalk] z_threshold={z_threshold}: "
                  f"{n_vetoed} spots brightness-vetoed (score → 0)")

        # Summary per channel
        print(f"\n  {'ch':>5}  {'n':>6}  {'med_d_ratio':>12}  "
              f"{'med_z':>8}  {'med_score':>10}  {'%>thresh':>8}")
        for ch in chan_order:
            sub = out[out['unmixed_chan'].astype(str) == ch]
            if len(sub) == 0:
                continue
            print(f"  {ch:>5}  {len(sub):>6}"
                  f"  {sub['d_assignment_ratio'].median():>12.3f}"
                  f"  {sub['z_intensity_vs_removed'].median():>8.2f}"
                  f"  {sub['crosstalk_score'].median():>10.3f}"
                  f"  {100*(sub['crosstalk_score'] > self.config.CROSSTALK_SCORE_THRESHOLD).mean():>7.1f}%")

        return out

    def apply_crosstalk_filter(
        self,
        unmixed_df_scored: pd.DataFrame,
        score_threshold: float = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Mark spots with high crosstalk scores as invalid in the ``valid_spot``
        column, making it the single comprehensive QC gate for downstream tables.

        A spot is flagged (``valid_spot = False``) when:
          * ``crosstalk_score > score_threshold``, AND
          * ``z_vetoed == False``  (brightness-vetoed spots are always kept)

        Parameters
        ----------
        unmixed_df_scored : pd.DataFrame
            Output of ``compute_crosstalk_scores``; must have ``crosstalk_score``,
            ``z_vetoed``, and ``valid_spot`` columns.
        score_threshold : float, optional
            Defaults to ``config.CROSSTALK_SCORE_THRESHOLD``.

        Returns
        -------
        filtered_df : pd.DataFrame
            Copy of input with ``valid_spot`` updated.
        summary : dict
            Per-channel and total removal counts.
        """
        if score_threshold is None:
            score_threshold = self.config.CROSSTALK_SCORE_THRESHOLD

        filtered_df = unmixed_df_scored.copy()

        # Ensure valid_spot column exists (may be absent on pairwise-only runs)
        if 'valid_spot' not in filtered_df.columns:
            filtered_df['valid_spot'] = True

        crosstalk_fail = (
            (filtered_df['crosstalk_score'] > score_threshold) &
            (~filtered_df['z_vetoed'].fillna(False))
        )
        # Only flip True → False; never resurrect spots already failed by geometry
        filtered_df.loc[crosstalk_fail, 'valid_spot'] = False

        n_flagged = int(crosstalk_fail.sum())
        n_total = len(filtered_df)
        print(f"  [crosstalk] score_threshold={score_threshold}: "
              f"flagged {n_flagged}/{n_total} additional spots "
              f"({100*n_flagged/max(n_total,1):.1f}%)")

        # Build per-channel summary
        chan_order = [str(ch) for ch in self.config.get_round_spot_channels()]
        summary: Dict = {'total_flagged': n_flagged, 'total_spots': n_total, 'channels': {}}
        for ch in chan_order:
            mask_ch = filtered_df['unmixed_chan'].astype(str) == ch
            n_ch = int(mask_ch.sum())
            n_ch_flagged = int((crosstalk_fail & mask_ch).sum())
            summary['channels'][ch] = {'n_spots': n_ch, 'n_flagged': n_ch_flagged}

        return filtered_df, summary

    def compute_cross_channel_nn_density(
        self,
        unmixed_df: pd.DataFrame,
        chan_order: "list[str] | None" = None,
        k: int = 5,
    ) -> pd.DataFrame:
        """
        Compute the k-th nearest-neighbour distance from each spot to same-channel
        and cross-channel spots within the same cell.

        For each spot and each channel, the distance to the k-th nearest neighbour
        (in physical space, using ``self.spatial_scale``) among all spots of that
        channel in the same cell is recorded.  This distance is the standard input
        to a kNN density estimator:

        .. math::

            \\hat{\\rho} \\propto \\frac{k}{\\frac{4}{3}\\pi\\, r_k^3}

        Spots in cells that have fewer than 2 spots in a given channel receive
        ``NaN`` for that channel's column (a single-spot cell has no meaningful
        neighbour).  If fewer than ``k`` spots exist the distance to the
        available furthest neighbour is used instead.

        Parameters
        ----------
        unmixed_df : pd.DataFrame
            Must have columns ``cell_id``, ``chan``, ``z``, ``y``, ``x``.
            Typically the output of ``apply_crosstalk_filter``.
        chan_order : list of str, optional
            Ordered spot channels.  Defaults to
            ``config.get_round_spot_channels()``.
        k : int, default 5
            Neighbour rank to use as the density proxy.  k=5 gives a good
            bias/variance tradeoff for typical HCR spot densities (~5-15
            spots/cell/channel).

        Returns
        -------
        pd.DataFrame
            Copy of ``unmixed_df`` with new float32 columns
            ``nn{k}_dist_{ch}`` for each channel in ``chan_order``.
        """
        if chan_order is None:
            chan_order = [str(ch) for ch in self.config.get_round_spot_channels()]

        spatial_scale = np.array(self.spatial_scale, dtype=np.float64)  # (z, y, x) µm/px
        coord_cols = ['z', 'y', 'x']

        out = unmixed_df.copy()
        n_spots = len(out)

        # Pre-scale all coordinates once
        coords_scaled = out[coord_cols].values.astype(np.float64) * spatial_scale  # (n, 3)

        # Normalise chan dtype for matching
        chan_vals = out['chan'].astype(str).values
        cell_vals = out['cell_id'].values

        # Build {cell_id: row_indices} lookup once — shared across channels
        cell_to_rows: Dict = {}
        for i, cid in enumerate(cell_vals):
            cell_to_rows.setdefault(cid, []).append(i)

        for ch in chan_order:
            col_name = f'nn{k}_dist_{ch}'
            result = np.full(n_spots, np.nan, dtype=np.float32)

            # Rows belonging to this channel
            ch_mask = chan_vals == ch
            ch_rows = np.where(ch_mask)[0]

            if len(ch_rows) == 0:
                out[col_name] = result
                continue

            ch_cell_vals = cell_vals[ch_rows]

            # Group channel rows by cell
            cell_to_ch_rows: Dict = {}
            for row_i, cid in zip(ch_rows, ch_cell_vals):
                cell_to_ch_rows.setdefault(cid, []).append(row_i)

            for cid, rows_in_cell in cell_to_ch_rows.items():
                n_cell_ch = len(rows_in_cell)
                if n_cell_ch < 2:
                    # Single spot — no meaningful neighbour
                    continue

                pts = coords_scaled[rows_in_cell]  # (n_cell_ch, 3)
                # Query k+1 neighbours (first hit is self at dist 0)
                k_query = min(k + 1, n_cell_ch)
                tree = cKDTree(pts)
                dists, _ = tree.query(pts, k=k_query, workers=1)
                # dists[:, 0] == 0 (self); take the last column as the k-th neighbour
                # (or the furthest available if n_cell_ch <= k)
                kth_dists = dists[:, -1].astype(np.float32)
                for local_i, row_i in enumerate(rows_in_cell):
                    result[row_i] = kth_dists[local_i]

            out[col_name] = result
            n_valid = int(np.sum(~np.isnan(result)))
            print(f"  [nn_density] chan={ch}: {n_valid}/{n_spots} spots have nn{k}_dist")

        return out

    def _save_results(self, unmixed_df: pd.DataFrame, min_dist: float, suffix: str = '') -> None:
        """Save unmixed spots to file"""
        suffix_str = f'_{suffix}' if suffix else ''
        output_path = (
            self.config.OUTPUT_FOLDER /
            f'unmixed_spots_R{self.config.ROUND_N}_minDist_{int(min_dist)}{suffix_str}.pkl'
        )
        scratch_path = (
            self.config.SCRATCH_FOLDER /
            f'unmixed_spots_R{self.config.ROUND_N}_minDist_{int(min_dist)}{suffix_str}.pkl'
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        unmixed_df.to_pickle(output_path)
        if scratch_path != output_path:
            scratch_path.parent.mkdir(parents=True, exist_ok=True)
            unmixed_df.to_pickle(scratch_path)
        
    def process_multiple_distances(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        min_distances: List[float],
        unmixing_method: str = 'reassignment',
        channel_pairs: List[Tuple[str, str]] = None
    ) -> Dict[float, Tuple[pd.DataFrame, List[Dict[str, int]]]]:
        """
        Process spots with multiple minimum distance values
        
        Args:
            spots_df: DataFrame containing spot data
            stats_df: DataFrame containing distance statistics
            min_distances: List of minimum distances to try
            unmixing_method: 'reassignment' (default) or 'pairwise'
            channel_pairs: List of (chanA, chanB) tuples for pairwise method
            
        Returns:
            Dictionary mapping distances to results tuples
        """
        results = {}
        
        for min_dist in min_distances:
            print(f'Processing minimum distance: {min_dist}')
            results[min_dist] = self.unmix_spots(
                spots_df,
                stats_df,
                min_dist,
                unmixing_method=unmixing_method,
                channel_pairs=channel_pairs
            )
            
            # Print statistics for this distance
            _, stats = results[min_dist]
            
            if unmixing_method == 'reassignment':
                for channel_stat in stats:
                    print(
                        f"Channel {channel_stat['channel']} ({channel_stat['gene']}): "
                        f"Kept {channel_stat['kept_spots']} of {channel_stat['total_spots']} spots "
                        f"({channel_stat['reassigned_spots']} reassigned)"
                    )
            elif unmixing_method == 'pairwise':
                for pair_stat in stats:
                    print(
                        f"Pair {pair_stat['pair']}: "
                        f"Overlaps {pair_stat['same_cell_overlaps']} (total: {pair_stat['total_overlaps']}), "
                        f"Removed from A: {pair_stat['removed_from_A']}, B: {pair_stat['removed_from_B']}"
                    )
                
        return results