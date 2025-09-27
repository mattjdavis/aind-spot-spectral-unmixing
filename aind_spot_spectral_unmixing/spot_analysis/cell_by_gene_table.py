
import numpy as np
import pandas as pd
import pathlib
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from .config import Config


@dataclass
# class SpotProcessingConfig:
#     """Configuration for spot processing pipeline"""
#     gene_dict: Dict[str, Dict[str, str]]
#     scratch_folder: pathlib.Path
#     data_folder: pathlib.Path
#     min_dist: int
#     volume_quantiles: List[float] = (0.08, 0.5, 0.95)
    
class cell_by_gene_processor:
    def __init__(self):
        self.config = Config
        self.spots_df = pd.DataFrame()
        self.segmentation_df = pd.DataFrame()
        
    @staticmethod
    def find_centroid(bbox: Tuple[float, ...]) -> List[float]:
        """Calculate centroid from bounding box coordinates"""
        zmin, ymin, xmin, zmax, ymax, xmax = bbox
        return [(zmin + zmax)/2, (ymin + ymax)/2, (xmin + xmax)/2]
    
    def load_spots(self, rounds: List[int], unmixed: bool = True) -> pd.DataFrame:
        """Load and process spots data for given rounds"""
        spots_df = pd.DataFrame()
        
        for rn in rounds:
            round_chans = list(self.config.GENE_DICT[str(rn)].keys())
            
            file_suffix = 'unmixed_spots' if unmixed else 'mixed_spots'
            file_name = f'{file_suffix}_R{rn}_minDist_{self.config.min_dist}.pkl' if unmixed else f'{file_suffix}_R{rn}.pkl'
            file_location = self.config.SCRATCH_FOLDER / file_name
            
            with open(file_location, 'rb') as file:
                ch_spots_df = pickle.load(file)
                
            for ch in round_chans:
                ch_spots_df.loc[ch_spots_df['chan']==ch, 'gene'] = self.config.GENE_DICT[str(rn)][ch]
            
            if unmixed:
                ch_spots_df = ch_spots_df.loc[ch_spots_df['chan']==ch_spots_df['unmixed_chan']]
                
            spots_df = pd.concat([spots_df, ch_spots_df])
            
        spots_df['cell_id'] = spots_df['cell_id'].astype('int')
        return spots_df
    
    def apply_spot_filters(self, spots_df): 
        """ Only returns the spots with 'valid_spot' column == True """

        return spots_df.loc[spots_df['valid_spot']==True]

    def load_segmentation(self, rounds: List[int]) -> pd.DataFrame:
        """Load segmentation data for given rounds"""
        segmentation_df = pd.DataFrame()
        
        for rn in rounds: #TODO update this file name with the standard filename
        
            file_location = list(pathlib.Path(self.config.DATA_FOLDER).glob('metrics.pickle')) #pipeline
            
            if len(file_location)>0:
                file_loc = file_location[0]
            else:
                file_location = list(pathlib.Path(self.config.DATA_FOLDER).glob('*/metrics.pickle')) #capsule
                file_loc = file_location[0]

            with open(file_loc, 'rb') as file:
                round_data = pickle.load(file)
                
            round_df = pd.DataFrame(round_data).T
            round_df['round'] = rn
            round_df['cell_id'] = round_df.index
            round_df['centroid'] = round_df['global_bbox'].apply(self.find_centroid)
            
            segmentation_df = pd.concat([segmentation_df, round_df])
            
        return segmentation_df
    
    def process_cell_annotations(self, spots_df: pd.DataFrame, segmentation_df: pd.DataFrame) -> pd.DataFrame:
        """Process cell annotations and create counts dataframe"""
        merged_df = pd.merge(segmentation_df, spots_df, on='cell_id', how='inner')
        
        cell_anno_counts = merged_df.groupby(['cell_id', 'gene']).agg({
            'spot_id': list,
            'volume': 'first',
            'centroid': 'first'
        }).reset_index()
        
        # Process centroids and coordinates
        cell_anno_counts[['z_centroid', 'y_centroid', 'x_centroid']] = pd.DataFrame(
            cell_anno_counts['centroid'].tolist(), 
            index=cell_anno_counts.index
        ).astype(int)
        
        cell_anno_counts['spot_count'] = cell_anno_counts['spot_id'].apply(len)
        
        # Calculate median spot coordinates
        med_coords = spots_df.groupby('cell_id')[['z', 'y', 'x']].median().reset_index()
        cell_anno_counts = pd.merge(cell_anno_counts, med_coords, on='cell_id', how='left')
        cell_anno_counts[['z_spots', 'y_spots', 'x_spots']] = cell_anno_counts[['z', 'y', 'x']].astype(int)
        
        return cell_anno_counts
    
    def filter_by_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe based on volume quantiles"""
        volume_quantiles = df['volume'].quantile(self.config.volume_quantiles)
        return df[
            (df['volume'] >= volume_quantiles[self.config.volume_quantiles[0]]) & 
            (df['volume'] <= volume_quantiles[self.config.volume_quantiles[2]])
        ]
    
    def plot_volume_distribution(self, df: pd.DataFrame) -> None:
        """Plot volume distribution with quantiles"""
        plt.figure(figsize=(17, 5))
        
        # Convert volume to numeric
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        quantiles = df['volume'].quantile(self.config.volume_quantiles)
        
        # Plot original distribution
        plt.subplot(1, 3, 1)
        self._plot_volume_histogram(df['volume'], quantiles, "Distribution of Volumes with Quantiles")
        
        # Plot ≤ 95th percentile
        filtered_95 = df[df['volume'] <= quantiles[self.config.volume_quantiles[2]]]
        plt.subplot(1, 3, 2)
        self._plot_volume_histogram(filtered_95['volume'], quantiles, "Volumes ≤ 95th Percentile")
        
        # Plot between 5th and 95th percentile
        filtered_5_95 = filtered_95[filtered_95['volume'] >= quantiles[self.config.volume_quantiles[0]]]
        plt.subplot(1, 3, 3)
        self._plot_volume_histogram(filtered_5_95['volume'], quantiles, "Volumes Between 5th and 95th Percentile")
        
        plt.tight_layout()
        plt.show()
        fig_loc = self.config.OUTPUT_FOLDER / 'volume_distribution.png'
        plt.savefig(fig_loc)
        plt.close()
    
    @staticmethod
    def _plot_volume_histogram(data: pd.Series, quantiles: pd.Series, title: str) -> None:
        """Helper method for plotting volume histograms"""
        sns.histplot(data.values, bins=30, kde=False)
        for q in quantiles:
            plt.axvline(q, color='r', linestyle='--')
        plt.xlabel('Volume')
        plt.ylabel('Frequency')
        plt.title(title)
    
    def process_pipeline(self, rounds: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete processing pipeline"""
        # Process unmixed spots
        unmixed_spots = self.load_spots(rounds, unmixed=True)
        unmixed_spots_filtered = self.apply_spot_filters(unmixed_spots)
        segmentation = self.load_segmentation(rounds)
        unmixed_annotations = self.process_cell_annotations(unmixed_spots_filtered, segmentation)
        #filtered_unmixed = self.filter_by_volume(unmixed_annotations)
        
        # Process mixed spots
        mixed_spots = self.load_spots(rounds, unmixed=False)
        mixed_annotations = self.process_cell_annotations(mixed_spots, segmentation)
        #filtered_mixed = self.filter_by_volume(mixed_annotations)
        
        # Save results
        unmixed_annotations.to_pickle(self.config.SCRATCH_FOLDER / 'unmixed_cell_by_gene.pkl')
        mixed_annotations.to_pickle(self.config.SCRATCH_FOLDER / 'mixed_cell_by_gene.pkl')

        unmixed_annotations.to_csv(self.config.OUTPUT_FOLDER / 'unmixed_cell_by_gene.csv')
        mixed_annotations.to_csv(self.config.OUTPUT_FOLDER / 'mixed_cell_by_gene.csv')
        
        return unmixed_annotations, mixed_annotations

# Example usage
if __name__ == "__main__":
    # config = SpotProcessingConfig(
    #     gene_dict={...},  # Your gene dictionary here
    #     scratch_folder=pathlib.Path('/scratch/'),
    #     data_folder=pathlib.Path('/data/'),
    #     min_dist=1
    # )
    
    processor = cell_by_gene_processor()
    unmixed_results, mixed_results = processor.process_pipeline([0, 13])
    
    # Plot volume distributions
    processor.plot_volume_distribution(unmixed_results)
