#!/usr/bin/env python3
"""
Main script for processing multichannel spot data.
This script demonstrates how to use the spot_analysis package to process
a complete round of multichannel data.
"""

import logging
from pathlib import Path
from typing import List, Optional
import argparse

import numpy as np
import pandas as pd

from spot_analysis.config import Config
from spot_analysis.data_loader import SpotDataLoader
from spot_analysis.spot_processor import SpotProcessor
from spot_analysis.ratio_calculator import RatioCalculator
from spot_analysis.unmixer import SpotUnmixer
from spot_analysis.cell_by_gene_table import cell_by_gene_processor


class SpotAnalysisPipeline:
    def __init__(
        self,
        #round_number: int,
        spots_folder: Path = Path('/data/'),
        output_folder: Path = Path('/results/'),
        min_distances: Optional[List[float]] = None,
        crosstalk_z_threshold: float = None,
        crosstalk_score_threshold: float = None,
    ):
        """
        Initialize the spot analysis pipeline.
        
        Args:
            round_number: The round number to process
            spots_folder: Path to the folder containing spot data
            output_folder: Path to save results
            min_distances: List of minimum distances to try for unmixing
            crosstalk_z_threshold: Brightness veto z-score (default from Config)
            crosstalk_score_threshold: Max allowed crosstalk score (default from Config)
        """
        # Update configuration
        #Config.ROUND_N = round_number
        #Config = Config()
        self.config = Config()
        Config.SPOTS_FOLDER = spots_folder
        Config.OUTPUT_FOLDER = output_folder
        
        # Set default min distances if not provided
        self.min_distances = min_distances or [3.0, 4.0, 5.0]

        # Crosstalk QC thresholds — fall back to Config class-level defaults
        self.crosstalk_z_threshold = (
            crosstalk_z_threshold
            if crosstalk_z_threshold is not None
            else Config.CROSSTALK_Z_THRESHOLD
        )
        self.crosstalk_score_threshold = (
            crosstalk_score_threshold
            if crosstalk_score_threshold is not None
            else Config.CROSSTALK_SCORE_THRESHOLD
        )
        
        # Initialize components
        self.data_loader = SpotDataLoader()
        self.processor = SpotProcessor()
        self.ratio_calculator = RatioCalculator()
        self.unmixer = SpotUnmixer()
        self.cell_by_gene_processor = cell_by_gene_processor()
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(
            Config.OUTPUT_FOLDER / f'round_{Config.ROUND_N}_processing.log'
        )
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def run(self):
        """Run the complete spot analysis pipeline"""
        self.logger.info(f"Starting analysis for Round {Config.ROUND_N}")
        
        try:
            # 1. Load Data
            self.logger.info("Loading spot data...")
            spots_df = self.data_loader.load_all_spots()
            self.logger.info(f"Loaded {len(spots_df)} total spots")
            
            # 2. Calculate intensities and apply threshold filtering
            self.logger.info("Processing spots...")
            spots_df = self.processor.calculate_intensities(spots_df)
            spots_df, spots_over_thresh = self.processor.filter_by_threshold(spots_df)
            
            self.logger.info(
                f"Found {len(spots_over_thresh)} spots over threshold "
                f"({len(spots_over_thresh)/len(spots_df)*100:.1f}%)"
            )
            
            # Save intermediate results
            spots_df.to_pickle(
                Config.OUTPUT_FOLDER / 
                f'mixed_spots_R{Config.ROUND_N}.pkl'
            )

            spots_df.to_pickle(
                Config.SCRATCH_FOLDER / 
                f'mixed_spots_R{Config.ROUND_N}.pkl'
            )
            
            # 3. Calculate ratios
            self.logger.info("Calculating channel ratios...")
            ratio_path = Config.OUTPUT_FOLDER / f'r{Config.ROUND_N}_ratios.txt'
            intensity_cols = [
                f'chan_{ch}_intensity'
                for ch in Config.get_round_spot_channels()
            ]
            # ratios = self.ratio_calculator.calculate_ratios( #tim's old way
            #     spots_df[intensity_cols].values,
            #     ratio_path
            # )
            ratios = self.ratio_calculator.calculate_ratios_by_channel( #trying a weighted avg
                spots_df[intensity_cols].values,
                ratio_path, 
                spots_df['chan'].values
            )

            # 4. Calculate distances and create stats
            self.logger.info("Calculating distances between spots...")
            #thresh_spots_df = spots_df[spots_df['over_thresh']].copy()
            stats_df = self.unmixer.calculate_distances(spots_df, ratios)

            #save stats_df
            stats_df_csv_name = '/results/spot_unmixing_stats.csv'
            stats_df.to_csv(stats_df_csv_name)
            
            # 5. Application of QC filters has been moved to interactive capsule
            self.logger.info("Applying QC filters...")
            spots_df = self.processor.apply_qc_filters(
                spots_df,
                stats_df
            )
            
            #self.logger.info(
            #    f"Kept {len(filtered_spots_df)} spots after QC "
            #    f"({len(filtered_spots_df)/len(thresh_spots_df)*100:.1f}%)"
            #)
            all_chans_filt_stats = self.unmixer.calculate_distances(spots_df, ratios)
            
            # 6. Process multiple minimum distances
            self.logger.info("Running unmixing with multiple minimum distances...")
            results = self.unmixer.process_multiple_distances(
                spots_df,
                all_chans_filt_stats,
                self.min_distances
            )

            # 6b. Crosstalk QC — run independently for each min_dist
            # all_chans_filt_stats has spot_uid_int stamped so merges are index-safe.
            # spots_df (post-geometric-QC, pre-dedup) is the mixed reference.
            self.logger.info("Applying crosstalk QC filters...")
            for min_dist, (unmixed_df_md, _ch_stats) in results.items():
                unmixed_df_md = unmixed_df_md.copy()
                unmixed_df_md['chan'] = unmixed_df_md['chan'].astype(str)
                unmixed_df_md['unmixed_chan'] = unmixed_df_md['unmixed_chan'].astype(str)

                removed_df = self.unmixer.build_removed_spots(
                    spots_df, unmixed_df_md, min_dist
                )
                unmixed_df_md = self.unmixer.compute_crosstalk_scores(
                    unmixed_df_md,
                    all_chans_filt_stats,
                    removed_df,
                    z_threshold=self.crosstalk_z_threshold,
                )
                unmixed_df_md, ct_summary = self.unmixer.apply_crosstalk_filter(
                    unmixed_df_md,
                    score_threshold=self.crosstalk_score_threshold,
                )
                self.logger.info(
                    f"min_dist={min_dist}: crosstalk flagged "
                    f"{ct_summary['total_flagged']}/{ct_summary['total_spots']} spots"
                )

                # Re-save unmixed pkl so cell_by_gene_processor reads updated valid_spot
                unmixed_pkl = (
                    Config.SCRATCH_FOLDER
                    / f'unmixed_spots_R{Config.ROUND_N}_minDist_{int(min_dist)}.pkl'
                )
                unmixed_df_md.to_pickle(unmixed_pkl)

                results[min_dist] = (unmixed_df_md, _ch_stats)

            # 7. Save summary statistics
            self._save_summary_statistics(results)

            # 8. Generate cleaned up tables for analysis
            processor = cell_by_gene_processor()
            unmixed_results, mixed_results = processor.process_pipeline([Config.ROUND_N])
            # Save final results
            #unmixed_results.to_csv(
            #    Config.OUTPUT_FOLDER / 
            #    f'unmixed_spots_R{Config.ROUND_N}.csv'
            #)
            #unmixed_results.to_pickle(
            #    Config.OUTPUT_FOLDER / 
            #    f'unmixed_spots_R{Config.ROUND_N}.pkl'
            #)
            #mixed_results.to_csv(
            #    Config.OUTPUT_FOLDER / 
            #    f'mixed_spots_R{Config.ROUND_N}.csv'
            #)
            self.logger.info("Processing completed successfully")

            return results
            
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}", exc_info=True)
            raise
            
    def _save_summary_statistics(self, results):
        """Save summary statistics for all processing runs"""
        summary_stats = []
        
        for min_dist, (unmixed_df, stats) in results.items():
            for channel_stat in stats:
                stat_dict = {
                    'min_dist': min_dist,
                    'round': Config.ROUND_N,
                    **channel_stat
                }
                summary_stats.append(stat_dict)
        
        # Create summary DataFrame and save
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(
            Config.OUTPUT_FOLDER / 
            f'round_{Config.ROUND_N}_summary_stats.csv',
            index=False
        )


def main():
    """Main entry point"""
    # Example usage
    pipeline = SpotAnalysisPipeline(
       round_number=13,
       spots_folder=Path('/data/'),
       output_folder=Path('/results/'),
       min_distances=[3.0, 4.0, 5.0]
    )

    # parser = argparse.ArgumentParser()
    # parser.add_argument("round", type=str, help="Dataset round until procedures.json is finished")
    # args = parser.parse_args()
    # round = int(args.round)


    # pipeline = SpotAnalysisPipeline(round_number = round, min_distances = [3])
    
    results = pipeline.run()
    
    # Print final statistics
    print("\nFinal Results Summary:")
    for min_dist, (unmixed_df, stats) in results.items():
        print(f"\nResults for minimum distance {min_dist}:")
        for channel_stat in stats:
            print(
                f"Channel {channel_stat['channel']} ({channel_stat['gene']}): "
                f"Kept {channel_stat['kept_spots']} of {channel_stat['total_spots']} spots "
                f"({channel_stat['kept_spots']/channel_stat['total_spots']*100:.1f}%)"
            )


if __name__ == '__main__':
    main()