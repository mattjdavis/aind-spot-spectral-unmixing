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
        
        return self._create_stats_dataframe(distances)
    
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
        # Initial filtering
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
                print("remove 1", spots.iloc[b]['cell_id']) # debug
            else:
                print("remove 2", spots.iloc[a]['cell_id'])
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
        channel_pairs: List[Tuple[str, str]] = None
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
            return self._unmix_spots_pairwise(spots_df, stats_df, min_dist, channel_pairs)
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
        
        self._save_results(unmixed_df, min_dist)
            
        return unmixed_df, channel_stats
    
    def _unmix_spots_pairwise(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        min_dist: float,
        channel_pairs: List[Tuple[str, str]]
    ) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
        """
        Remove crosstalk by comparing spatially overlapping spots between channel pairs
        
        Args:
            spots_df: Spot locations and detected channels
            stats_df: Distance to each dye line (must align with spots_df index)
            min_dist: Minimum spatial distance in physical units (um)
            channel_pairs: List of (chanA, chanB) tuples to check for crosstalk
        
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
        
        # Process each channel pair
        for chanA, chanB in channel_pairs:
            print(f"Processing channel pair: {chanA} - {chanB}")
            stats = self._filter_pairwise_crosstalk(
                spots_df,
                stats_df,
                chanA,
                chanB,
                min_dist,
                keep  # Modified in-place
            )
            pair_stats.append(stats)
            print(f"  Total overlaps: {stats['total_overlaps']}, Same cell: {stats['same_cell_overlaps']}")
            print(f"  Removed from {chanA}: {stats['removed_from_A']}, from {chanB}: {stats['removed_from_B']}")
        
        # Apply final filtering
        filtered_spots = spots_df[keep].copy()
        filtered_spots['unmixed_chan'] = filtered_spots['chan']  # Keep original channel
        
        self._save_results(filtered_spots, min_dist)
        
        return filtered_spots, pair_stats
    
    def _filter_pairwise_crosstalk(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        chanA: str,
        chanB: str,
        min_dist: float,
        keep: np.ndarray
    ) -> Dict[str, int]:
        """
        Filter crosstalk between a specific channel pair
        
        Args:
            chanA, chanB: Channel identifiers (e.g., '488', '514')
            min_dist: Spatial threshold in physical units (um)
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
        
        # Scale positions by anisotropic factor
        spatial_scale = np.array(self.spatial_scale)  # (z, y, x)
        posA = spotsA[['z', 'y', 'x']].values * spatial_scale
        posB = spotsB[['z', 'y', 'x']].values * spatial_scale
        
        # Build KD-tree for channel B
        treeB = cKDTree(posB)
        
        # Find spatial overlaps: for each spot in A, find neighbors in B
        pairs = treeB.query_ball_point(posA, min_dist)  # Returns list of lists
        
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