""" top level run script """

from process_round import SpotAnalysisPipeline
from pathlib import Path

def run():
    #TODO add argparse for round number, spots folder, output folder, min distances

    #example   
    # parser = argparse.ArgumentParser()
    # parser.add_argument("round", type=str, help="Dataset round until procedures.json is finished")
    # args = parser.parse_args()
    # round = int(args.round)


    pipeline = SpotAnalysisPipeline(min_distances = [3])
    
    #pipeline = SpotAnalysisPipeline(
    #    round_number=13,
    #    spots_folder=Path('/data/'),
    #     output_folder=Path('/results/'),
    #    min_distances=[3.0, 4.0, 5.0] #can probably set default of 3 pixels and forget this
    #)
    
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

if __name__ == "__main__": run()