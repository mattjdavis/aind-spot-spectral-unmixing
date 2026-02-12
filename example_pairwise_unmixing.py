"""
Example script demonstrating pairwise channel crosstalk removal

This script shows how to use the new pairwise unmixing method to remove 
crosstalk between specific channel pairs based on spatial proximity and 
spectral fit quality.
"""

import numpy as np
from aind_spot_spectral_unmixing.spot_analysis.unmixer import SpotUnmixer
from aind_spot_spectral_unmixing.spot_analysis.config import Config

# Example usage
def run_pairwise_unmixing(dataset_folder, round_num=0):
    """
    Run pairwise unmixing on a dataset
    
    Args:
        dataset_folder: Path to the dataset
        round_num: Round number to process
    """
    
    # Initialize config
    config = Config(dataset_folder)
    
    # Define channel pairs to check for crosstalk
    # These are typically adjacent or overlapping spectral channels
    channel_pairs = [
        ('488', '514'),  # Check crosstalk between 488 and 514
        ('514', '561'),  # Check crosstalk between 514 and 561
        ('561', '594'),  # Check crosstalk between 561 and 594
    ]
    
    # Define anisotropic spatial scaling (z, y, x) in um/pixel
    # This accounts for different resolution in z vs xy
    spatial_scale = np.array([1.0, 0.24, 0.24])
    
    # Initialize unmixer with pairwise configuration
    unmixer = SpotUnmixer(
        dataset_folder=dataset_folder,
        config=config,
        channel_pairs=channel_pairs,
        spatial_scale=spatial_scale
    )
    
    # Load your spots and calculate distances
    # (This part depends on your data loading pipeline)
    # spots_df = load_spots(...)  # Load spot data
    # stats_df = unmixer.calculate_distances(spots_df, ratios)
    
    # Define minimum distance threshold in physical units (micrometers)
    min_dist = 2.0  # 2 um threshold
    
    # Option 1: Use pairwise unmixing
    print("\n=== Running Pairwise Unmixing ===")
    filtered_spots_pairwise, pair_stats = unmixer.unmix_spots(
        spots_df=spots_df,
        stats_df=stats_df,
        min_dist=min_dist,
        unmixing_method='pairwise',
        channel_pairs=channel_pairs
    )
    
    print("\nPairwise unmixing results:")
    for stat in pair_stats:
        print(f"\n  Pair {stat['pair']}:")
        print(f"    Total overlaps: {stat['total_overlaps']}")
        print(f"    Same-cell overlaps: {stat['same_cell_overlaps']}")
        print(f"    Removed from channel A: {stat['removed_from_A']}")
        print(f"    Removed from channel B: {stat['removed_from_B']}")
        print(f"    Remaining in A: {stat['kept_A']}")
        print(f"    Remaining in B: {stat['kept_B']}")
    
    # Option 2: Use traditional reassignment method (for comparison)
    print("\n=== Running Reassignment Unmixing (for comparison) ===")
    filtered_spots_reassignment, channel_stats = unmixer.unmix_spots(
        spots_df=spots_df,
        stats_df=stats_df,
        min_dist=min_dist,
        unmixing_method='reassignment'
    )
    
    print("\nReassignment unmixing results:")
    for stat in channel_stats:
        print(f"\n  Channel {stat['channel']} ({stat['gene']}):")
        print(f"    Total spots: {stat['total_spots']}")
        print(f"    Kept spots: {stat['kept_spots']}")
        print(f"    Reassigned spots: {stat['reassigned_spots']}")
    
    # Option 3: Process multiple distance thresholds
    print("\n=== Processing Multiple Distances ===")
    min_distances = [1.5, 2.0, 2.5, 3.0]  # in micrometers
    
    results = unmixer.process_multiple_distances(
        spots_df=spots_df,
        stats_df=stats_df,
        min_distances=min_distances,
        unmixing_method='pairwise',
        channel_pairs=channel_pairs
    )
    
    # Analyze results across distances
    print("\nSpot counts across different distance thresholds:")
    for dist in min_distances:
        filtered_df, _ = results[dist]
        print(f"  min_dist={dist} um: {len(filtered_df)} total spots")
    
    return filtered_spots_pairwise, pair_stats


# Key differences between methods:
#
# REASSIGNMENT METHOD (original):
# - Reassigns all spots to their best-fit spectral channel
# - Then removes spatial duplicates within each reassigned channel
# - Spots can change channels based on spectral fit
#
# PAIRWISE METHOD (new):
# - Keeps spots in their originally detected channels
# - Only checks for spatial overlaps between specific channel pairs
# - When overlap found, removes the spot with worse fit to its OWN dye line
# - Better for identifying and removing crosstalk while preserving true signals
#
# PARAMETERS:
# - min_dist: Spatial threshold in micrometers (physical units)
# - spatial_scale: (z, y, x) scaling to convert pixels to micrometers
# - channel_pairs: Which channel combinations to check for crosstalk
#
# OUTPUT:
# - Filtered spots dataframe with 'unmixed_chan' column
# - Statistics about overlaps and removals for each channel pair


if __name__ == "__main__":
    # Example: Run on a specific dataset
    # dataset_folder = "HCR_754803_2025-09-18_13-00-00_processed_2025-09-20_22-57-09"
    # run_pairwise_unmixing(dataset_folder)
    
    print("This is an example script. Modify with your dataset path to run.")
