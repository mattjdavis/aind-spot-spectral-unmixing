# Pairwise Channel Crosstalk Removal - Implementation Summary

## Overview

Implemented a new unmixing method that identifies and removes spectral crosstalk by comparing spatially overlapping spots between specific channel pairs, rather than globally reassigning all spots.

## Key Changes

### 1. **Updated `SpotUnmixer.__init__()` **
- Added `channel_pairs` parameter: List of tuples defining which channel combinations to check
- Added `spatial_scale` parameter: (z, y, x) anisotropic scaling in um/pixel (default: [1.0, 0.24, 0.24])

### 2. **Enhanced `_create_stats_dataframe()`**
- Now stores full distance matrix for all channels
- Adds columns: `dist_to_chan_{channel}` for each channel
- Required for pairwise method to compare fit quality to respective dye lines

### 3. **Refactored `unmix_spots()`**
- New parameter `unmixing_method`: 'reassignment' or 'pairwise'
- Routes to appropriate method based on selection
- Backwards compatible with existing code (defaults to 'reassignment')

### 4. **New Method: `_unmix_spots_pairwise()`**
- Main entry point for pairwise unmixing
- Validates spots_df and stats_df alignment
- Iterates over channel pairs
- Returns filtered spots with original channel assignments

### 5. **New Method: `_filter_pairwise_crosstalk()`**
- Core logic for removing crosstalk between a single channel pair
- Process:
  1. Extract spots detected in channels A and B
  2. Apply anisotropic spatial scaling to coordinates
  3. Build KDTree for channel B spots
  4. Find spatial overlaps using `query_ball_point()` with min_dist threshold
  5. Filter to same cell_id only
  6. For each overlap, compare distance to respective dye lines
  7. Remove spot with worse spectral fit
- Modifies global `keep` mask in-place
- Returns detailed statistics per pair

### 6. **Updated `_save_results()`**
- Added `suffix` parameter to differentiate output files
- Pairwise results saved with '_pairwise' suffix

### 7. **Updated `process_multiple_distances()`**
- Now accepts `unmixing_method` and `channel_pairs` parameters
- Prints appropriate statistics based on method used

## Algorithm Details

### Spatial Matching with Anisotropy

```python
# Apply anisotropic scaling before distance calculation
spatial_scale = np.array([1.0, 0.24, 0.24])  # z, y, x in um/pixel
positions_scaled = spots[['z', 'y', 'x']].values * spatial_scale

# Build KDTree on scaled coordinates
tree = cKDTree(positions_scaled)

# Query with min_dist in physical units (um)
pairs = tree.query_ball_point(positions_A, min_dist)
```

### Spectral Fit Comparison

```python
# Get distance to each spot's OWN detected channel
distA = stats_df.loc[spot_a_idx, f'dist_to_chan_{chanA}']
distB = stats_df.loc[spot_b_idx, f'dist_to_chan_{chanB}']

# Remove spot with worse fit to its own dye line
if distA > distB:
    keep[spot_a_idx] = False  # Remove A
else:
    keep[spot_b_idx] = False  # Remove B
```

## Usage Examples

### Basic Pairwise Unmixing

```python
from aind_spot_spectral_unmixing.spot_analysis.unmixer import SpotUnmixer
from aind_spot_spectral_unmixing.spot_analysis.config import Config

# Initialize with configuration
config = Config(dataset_folder)
channel_pairs = [('488', '514'), ('514', '561'), ('561', '594')]
spatial_scale = np.array([1.0, 0.24, 0.24])

unmixer = SpotUnmixer(
    dataset_folder=dataset_folder,
    config=config,
    channel_pairs=channel_pairs,
    spatial_scale=spatial_scale
)

# Run pairwise unmixing
filtered_spots, pair_stats = unmixer.unmix_spots(
    spots_df=spots_df,
    stats_df=stats_df,
    min_dist=2.0,  # in micrometers
    unmixing_method='pairwise',
    channel_pairs=channel_pairs
)
```

### Compare Methods

```python
# Method 1: Pairwise (new)
spots_pairwise, stats_pairwise = unmixer.unmix_spots(
    spots_df, stats_df, min_dist=2.0,
    unmixing_method='pairwise',
    channel_pairs=[('488', '514'), ('514', '561')]
)

# Method 2: Reassignment (original)
spots_reassign, stats_reassign = unmixer.unmix_spots(
    spots_df, stats_df, min_dist=2.0,
    unmixing_method='reassignment'
)
```

### Process Multiple Distances

```python
results = unmixer.process_multiple_distances(
    spots_df=spots_df,
    stats_df=stats_df,
    min_distances=[1.5, 2.0, 2.5, 3.0],
    unmixing_method='pairwise',
    channel_pairs=channel_pairs
)
```

## Statistics Output

### Pairwise Method
Each channel pair returns:
- `pair`: Channel pair identifier (e.g., "488-514")
- `total_overlaps`: All spatial overlaps found
- `same_cell_overlaps`: Overlaps within same cell
- `removed_from_A`: Spots removed from first channel
- `removed_from_B`: Spots removed from second channel
- `kept_A`: Remaining spots in first channel
- `kept_B`: Remaining spots in second channel

### Reassignment Method (unchanged)
Each channel returns:
- `channel`: Channel identifier
- `gene`: Gene name for this channel
- `total_spots`: Initial spot count
- `kept_spots`: Final spot count
- `reassigned_spots`: Spots reassigned from other channels

## Key Advantages of Pairwise Method

1. **Preserves Original Detection**: Spots stay in their detected channels
2. **Targeted Crosstalk Removal**: Only checks specified channel pairs
3. **Spectral Quality Based**: Compares fit to respective dye lines, not global best fit
4. **Anisotropic Distance**: Accounts for z-axis vs xy-axis resolution differences
5. **Interpretable**: Clear statistics on crosstalk between specific pairs

## Implementation Notes

- **Distance Metric**: Uses absolute distance to dye line (`dist_to_chan_X`), not ratio (`dist_r`)
- **min_dist Units**: Always in physical units (micrometers), automatically handled via spatial_scale
- **KDTree Strategy**: Per-pair trees (simpler, sufficient for typical channel counts)
- **Index Alignment**: Requires spots_df and stats_df to have matching indices (validated at runtime)
- **Output Files**: Saved with '_pairwise' suffix to distinguish from reassignment results

## Backward Compatibility

All existing code continues to work unchanged:
- Default `unmixing_method='reassignment'` maintains original behavior
- Original API signatures still supported
- No breaking changes to existing workflows

## Testing Recommendations

1. Compare spot counts between methods at same min_dist
2. Visualize spatial distribution of removed vs. kept spots
3. Check spectral profiles of removed spots vs. kept spots
4. Validate that high-confidence spots are preserved
5. Confirm crosstalk removal in known problematic channel pairs
