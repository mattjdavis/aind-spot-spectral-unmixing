import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List, Dict, Tuple
from .config import Config

class SpotUnmixer:
    def __init__(self, dataset_folder: str, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
        return pd.DataFrame(
            stats,
            columns=['dist_rank_1', 'dist_rank_2', 'small0', 'small1', 'dist_r']
        )
    
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
            else:
                keep[a] = 0
                
        return keep
    
    def _process_channel(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        channel_idx: int,
        channel: str,
        min_dist: float
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Process spots for a single channel"""
        # Get channel matching spots

        #TODO link stats_df to spots_df here.... currently stats_df is unfiltered by qc (intensity, shape, etc)
        dist_rank_chan_match = np.array(stats_df['dist_rank_1'] == channel_idx)
        chan_spots = spots_df.loc[dist_rank_chan_match].copy()
        chan_stats = stats_df.loc[dist_rank_chan_match].copy()
        
        # Get intensities for the channel
        intensity_cols = [
            f'chan_{ch}_intensity'
            for ch in self.config.get_round_spot_channels()
        ]
        chan_intensities = spots_df.loc[dist_rank_chan_match, intensity_cols].copy()
        
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
        min_dist: float
    ) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
        """
        Unmix spots based on calculated statistics
        
        Args:
            spots_df: DataFrame containing spot data
            stats_df: DataFrame containing distance statistics
            min_dist: Minimum distance between spots for spatial filtering
            
        Returns:
            Tuple containing:
            - DataFrame of unmixed spots
            - List of dictionaries containing statistics for each channel


        Note: 
        MJD suggests approach where we _don't_ iterate over spots by/within channels in _filter_spatial_matches(), 
        but instead perform our _spatial_ analysis on more systematic level, such as running (multiscale? like spotsweeper)
        KNN on all spots, and then globally removing neighbours that are too close, and/or (following spotsweeper) running clustering (PCA) on those results. 

        
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
                min_dist
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
    
    def _save_results(self, unmixed_df: pd.DataFrame, min_dist: float) -> None:
        """Save unmixed spots to file"""
        output_path = (
            self.config.OUTPUT_FOLDER /
            f'unmixed_spots_R{self.config.ROUND_N}_minDist_{int(min_dist)}.pkl'
        )
        scratch_path = (
            self.config.SCRATCH_FOLDER /
            f'unmixed_spots_R{self.config.ROUND_N}_minDist_{int(min_dist)}.pkl'
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        unmixed_df.to_pickle(output_path)
        unmixed_df.to_pickle(scratch_path)
        
    def process_multiple_distances(
        self,
        spots_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        min_distances: List[float]
    ) -> Dict[float, Tuple[pd.DataFrame, List[Dict[str, int]]]]:
        """
        Process spots with multiple minimum distance values
        
        Args:
            spots_df: DataFrame containing spot data
            stats_df: DataFrame containing distance statistics
            min_distances: List of minimum distances to try
            
        Returns:
            Dictionary mapping distances to results tuples
        """
        results = {}
        
        for min_dist in min_distances:
            print(f'Processing minimum distance: {min_dist}')
            results[min_dist] = self.unmix_spots(
                spots_df,
                stats_df,
                min_dist
            )
            
            # Print statistics for this distance
            _, stats = results[min_dist]
            for channel_stat in stats:
                print(
                    f"Channel {channel_stat['channel']} ({channel_stat['gene']}): "
                    f"Kept {channel_stat['kept_spots']} of {channel_stat['total_spots']} spots "
                    f"({channel_stat['reassigned_spots']} reassigned)"
                )
                
        return results