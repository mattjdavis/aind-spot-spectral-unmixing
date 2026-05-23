import numpy as np
import pandas as pd
from typing import Tuple
from .config import Config

class SpotProcessor:
    def __init__(self):
        self.config = Config
    
    def calculate_intensities(self, spots_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate intensities for each channel"""
        channels = self.config.get_round_spot_channels()
        for channel in channels:
            spots_df[f'chan_{channel}_intensity'] = (
                spots_df[f'chan_{channel}_fg'] - spots_df[f'chan_{channel}_bg']
            )
        return spots_df
    
    def filter_by_threshold(self, spots_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Filter spots based on intensity threshold"""
        intensity_cols = [f'chan_{ch}_intensity' for ch in self.config.get_round_spot_channels()]
        
        # Find spots over threshold
        spots_over_thresh = spots_df.loc[
            np.any(spots_df[intensity_cols] > np.percentile(
                spots_df[intensity_cols],
                self.config.PERCENTILE, 0
            ), 1)
        ]['spot_id'].values
        
        # Mark spots over threshold
        spots_df['over_thresh'] = False
        spots_df.loc[spots_df['spot_id'].isin(spots_over_thresh), 'over_thresh'] = True
        
        return spots_df, spots_over_thresh
        
    def new_filter_by_threshold(self, spots_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Filter spots based on:
        1. Having sufficient intensity in their detection channel
        2. Overall intensity threshold across channels
        """
        # Calculate thresholds for each channel
        channels = self.config.get_round_spot_channels()
        channel_thresholds = {}
        for channel in channels:
            channel_intensities = spots_df[f'chan_{channel}_intensity']
            threshold = np.percentile(channel_intensities, self.config.PERCENTILE)
            channel_thresholds[channel] = threshold
            print(f"Channel {channel} threshold: {threshold}")
        
        spots_df = spots_df.copy()
        spots_df['over_thresh'] = False
        spots_over_thresh = []
        
        # Create a mask for spots to keep
        keep_spots = np.ones(len(spots_df), dtype=bool)
        
        # Step 1: Filter based on detection channel intensity
        for channel in channels:
            # Get spots detected in this channel
            channel_mask = spots_df['chan'] == channel
            channel_spots = spots_df[channel_mask]
            
            # Mark spots that don't meet their detection channel threshold
            failed_spots = channel_spots[
                channel_spots[f'chan_{channel}_intensity'] <= channel_thresholds[channel]
            ].index
            keep_spots[failed_spots] = False
        
        # Apply first filter
        spots_df = spots_df[keep_spots]
        
        # Step 2: Apply the original threshold across all channels
        intensity_cols = [f'chan_{ch}_intensity' for ch in channels]
        
        # For each spot, check if it exceeds threshold in any channel
        for channel in channels:
            channel_spots = spots_df[spots_df['chan'] == channel]
            if len(channel_spots) == 0:
                continue
                
            # Check each intensity column against its threshold
            any_over_thresh = False
            for col, thresh in zip(intensity_cols, [channel_thresholds[ch] for ch in channels]):
                if any_over_thresh:
                    break
                any_over_thresh |= (channel_spots[col] > thresh).any()
            
            if any_over_thresh:
                spots_over_thresh.extend(channel_spots['spot_id'].values)
        
        # Update over_thresh flag
        spots_over_thresh = np.array(spots_over_thresh)
        spots_df.loc[spots_df['spot_id'].isin(spots_over_thresh), 'over_thresh'] = True
        
        # Print summary statistics
        print("\nAfter filtering:")
        for channel in channels:
            chan_spots = spots_df[
                (spots_df['chan'] == channel) & 
                spots_df['over_thresh']
            ]
            print(f"Channel {channel} retained spots: {len(chan_spots)}")
        
        return spots_df, spots_over_thresh
    
    def apply_qc_filters(self, spots_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Alias for apply_geometric_qc — kept for backwards compatibility."""
        return self.apply_geometric_qc(spots_df, stats_df)

    def apply_geometric_qc(self, spots_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 1 QC: annotate valid_spot based on geometric spot-shape metrics.

        Filters applied (all three must pass):
          - ``dist < CENT_CUTOFF``  — spot is well-centred inside its cell
          - ``r > CORR_CUTOFF``    — spot profile matches a Gaussian PSF
          - ``dist_r > DIST_CUTOFF`` — spot is spectrally unambiguous (dye-line ratio)

        All rows are kept; only ``valid_spot`` is annotated.
        """

        spots_df['valid_spot'] = False
        spots_df = spots_df.copy()
        stats_df = stats_df.copy()

        spots_df = spots_df.reset_index(drop = True)
        stats_df = stats_df.reset_index(drop = True)

        filters = (
            spots_df['dist'] < self.config.CENT_CUTOFF,
            spots_df['r'] > self.config.CORR_CUTOFF,
            stats_df['dist_r'] > self.config.DIST_CUTOFF
        )
        
        all_filters = np.all(np.vstack(filters), 0)
        
        spots_df.loc[all_filters, 'valid_spot'] = True
        try: 
            spots_df['dye_line_dist_ratio'] = stats_df.loc[spots_df.index, 'dist_r']
        except: 
            print(f'Failed to add dist_r to spots_df')
        return spots_df