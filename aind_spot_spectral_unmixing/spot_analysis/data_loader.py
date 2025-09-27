import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from .config import Config

class SpotDataLoader:
    def __init__(self):
        self.config = Config
        self.spot_col_order = [
            'spot_id', 'chan', 'chan_spot_id', 'cell_id', 'round',
            'z', 'y', 'x', 'z_center', 'y_center', 'x_center', 'dist', 'r'
        ]
    
    """def load_channel_spots(self, channel: str) -> pd.DataFrame:
    #depreciated

        spot_cols = [
            'z', 'y', 'x', 'z_center', 'y_center', 'x_center',
            'dist', 'r', f'chan_{channel}_fg', f'chan_{channel}_bg', 'cell_id'
        ]
        
        spots_path = self.config.SPOTS_FOLDER / self.config.get_folder_paths()['spots_folders'][channel]

        spots_data = pd.DataFrame(
            np.load(spots_path),
            columns=spot_cols
        )

        
        spots_data['round'] = str(self.config.ROUND_N)
        spots_data['chan'] = channel
        spots_data['spot_id'] = range(1, len(spots_data) + 1)
        spots_data['chan_spot_id'] = range(1, len(spots_data) + 1)
        
        return spots_data[list(self.spot_col_order) + list(np.setdiff1d(spots_data.columns, self.spot_col_order))]
    """

    def load_multichannel_data(self, ch: str, m_ch: str) -> pd.DataFrame:
        
        
        round_n = self.config.ROUND_N
        multichan_folders = self.config.get_folder_paths()['multichan_folders']
        spot_col_order = ['spot_id','chan','chan_spot_id','cell_id','round','z','y','x','z_center','y_center','x_center','dist','r'] 
        # spot_cols = ['Z','Y','X','Z_center','Y_center','X_center','dist','r','SEG_ID','FG','BG'] #what comes from CSV 
        #spot_cols = ['z','y','x','z_center','y_center','x_center','dist','r','chan_'+str(ch)+'_fg','chan_'+str(ch)+'_bg','cell_id']
        # spots = pd.DataFrame(np.load(self.config.SPOTS_FOLDER.joinpath(spots_folders[str(ch)])),columns=spot_cols)
        spots = pd.read_csv(self.config.SPOTS_FOLDER.joinpath(multichan_folders[str(ch)][str(m_ch)]))
        spots = spots.rename(columns = {'Z':'z',
                                 'Y': 'y',
                                 'X': 'x', 
                                 'Z_center': 'z_center', 
                                 'Y_center': 'y_center', 
                                 'X_center': 'x_center', 
                                 'SEG_ID': 'cell_id', 
                                 'FG': f'chan_{ch}_fg', 
                                 'BG': f'chan_{ch}_bg'})
        spots['round']=str(round_n)
        spots['chan']=str(ch)
        spots['spot_id']=range(1, len(spots)+1)
        spots['chan_spot_id']=range(1, len(spots)+1)
        spots = spots[list(spot_col_order)+list(np.setdiff1d(spots.columns,spot_col_order))]
        return spots

    def load_detected_spots_for_channel(self, ch):
        round_n = self.config.ROUND_N
        spots_folders = self.config.get_folder_paths()['spots_folders']
        spot_col_order = ['spot_id','chan','chan_spot_id','cell_id','round','z','y','x','z_center','y_center','x_center','dist','r'] 
        # spot_cols = ['Z','Y','X','Z_center','Y_center','X_center','dist','r','SEG_ID','FG','BG'] #what comes from CSV 
        spots = pd.read_csv(self.config.SPOTS_FOLDER.joinpath(spots_folders[str(ch)]))
        spots = spots.rename(columns = {'Z':'z',
                                 'Y': 'y',
                                 'X': 'x', 
                                 'Z_center': 'z_center', 
                                 'Y_center': 'y_center', 
                                 'X_center': 'x_center', 
                                 'SEG_ID': 'cell_id', 
                                 'FG': f'chan_{ch}_fg', 
                                 'BG': f'chan_{ch}_bg'})
        spots['round']=str(round_n)
        spots['chan']=str(ch)
        spots['spot_id']=range(1, len(spots)+1)
        spots['chan_spot_id']=range(1, len(spots)+1)
        spots = spots[list(spot_col_order)+list(np.setdiff1d(spots.columns,spot_col_order))]
        return spots


    def load_all_spots(self) -> pd.DataFrame:
        """Load and combine all spot data"""


        channels = self.config.get_round_spot_channels()
        channels = [str(i) for i in list(channels) if i!= '405']

        round_n = self.config.ROUND_N
        multichan_folders = self.config.get_folder_paths()['multichan_folders']

        channel_spots = {}
        #get detected spots
        for ch in channels:
            channel_spots[ch] = self.load_detected_spots_for_channel(ch)


        for ch in channels:
            # Luminance from loop channel at channel 3's spot locations
            if len(ch) ==1:
                # check if ch is single digit or 
                if str(ch) != '1':
                    m_ch = 1
                    multi_spot_cols = ['z','y','x','chan_'+str(ch)+'_fg','chan_'+str(ch)+'_bg']
                    chan_multichan_df = pd.DataFrame(np.load(self.config.SPOTS_FOLDER.joinpath(multichan_folders[str(ch)][str(m_ch)])),columns=multi_spot_cols)
                    channel_spots[str(m_ch)] = channel_spots[str(m_ch)].merge(chan_multichan_df, on=['z', 'y', 'x'], how='inner')
                # Luminance from loop channel at channel 2's spot locations
                if str(ch) != '2':
                    m_ch = 2
                    multi_spot_cols = ['z','y','x','chan_'+str(ch)+'_fg','chan_'+str(ch)+'_bg']
                    chan_multichan_df = pd.DataFrame(np.load(self.config.SPOTS_FOLDER.joinpath(multichan_folders[str(ch)][str(m_ch)])),columns=multi_spot_cols)
                    channel_spots[str(m_ch)] = channel_spots[str(m_ch)].merge(chan_multichan_df, on=['z', 'y', 'x'], how='inner')
                # Luminance from loop channel at channel 3's spot locations
                if (str(ch) != '3') & (round_n != 0):
                    m_ch = 3
                    multi_spot_cols = ['z','y','x','chan_'+str(ch)+'_fg','chan_'+str(ch)+'_bg']
                    chan_multichan_df = pd.DataFrame(np.load(self.config.SPOTS_FOLDER.joinpath(multichan_folders[str(ch)][str(m_ch)])),columns=multi_spot_cols)
                    channel_spots[str(m_ch)] = channel_spots[str(m_ch)].merge(chan_multichan_df, on=['z', 'y', 'x'], how='inner')
                # Luminance from loop channel at channel 4's spot locations
                if str(ch) != '4':
                    m_ch = 4
                    multi_spot_cols = ['z','y','x','chan_'+str(ch)+'_fg','chan_'+str(ch)+'_bg']
                    chan_multichan_df = pd.DataFrame(np.load(self.config.SPOTS_FOLDER.joinpath(multichan_folders[str(ch)][str(m_ch)])),columns=multi_spot_cols)
                    channel_spots[str(m_ch)] = channel_spots[str(m_ch)].merge(chan_multichan_df, on=['z', 'y', 'x'], how='inner')
            else: 

                active_channel_list = channels
                active_channel_list_without_current_ch = [chan for chan in channels if chan!=ch]
                for m_ch in active_channel_list_without_current_ch: 
                    
                    try: 
                        multi_spot_cols = ['z','y','x','chan_'+str(ch)+'_fg','chan_'+str(ch)+'_bg']
                        chan_multichan_df = self.load_multichannel_data(ch, m_ch)
                        test_multichan = chan_multichan_df.drop(['spot_id', 'chan', 'chan_spot_id'], axis = 1)

                        channel_spots[str(m_ch)] = channel_spots[m_ch].merge(test_multichan, on = ['z', 'y', 'x', 'cell_id', 'round', 'z_center', 'y_center', 'x_center', 'dist', 'r'], how = 'inner')
                    except Exception as e: 
                        print(f"Error processing channel {ch} at measurement channel {m_ch}: {str(e)}")
                        continue

        mixed_spots_df = pd.DataFrame()
        #convert dict of channel_spots to list
        spots_list = list(channel_spots.values())
        for i, ch in enumerate(channels):
            mixed_spots_df = pd.concat([mixed_spots_df,spots_list[i] ])
        for ch in channels:
            mixed_spots_df['chan_'+str(ch)+'_intensity'] = mixed_spots_df['chan_'+str(ch)+'_fg']-mixed_spots_df['chan_'+str(ch)+'_bg']
        return mixed_spots_df    