import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import os
from .config import Config

class RatioCalculator:
    def __init__(self, dataset_folder):
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable CUDA launch blocking
        os.environ['TORCH_USE_CUDA_DSA'] = '1'

        self.config = Config(dataset_folder=dataset_folder)
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')#
        self.channels = self.config.get_round_spot_channels()

        self.channels = ['488', '514', '561', '594', '638'] # MJD HACK
        
    # def objective_fn(self, r: torch.Tensor, subset: torch.Tensor, L1: float) -> torch.Tensor:
    #     """Calculate objective function for ratio optimization"""
    #     r = r / torch.norm(r, dim=0)
    #     n_cam = len(self.config.get_round_channels())
        
    #     dot_products = torch.tile(subset @ r, (n_cam, 1, 1))
    #     ys = torch.tile(torch.unsqueeze(r, 1), (1, subset.shape[0], 1))
    #     xs = torch.tile(
    #         torch.unsqueeze(torch.transpose(subset, 0, 1), 2),
    #         (1, 1, n_cam)
    #     )
        
    #     return (
    #         torch.sum(torch.min(torch.norm(dot_products * ys - xs, dim=0), dim=1)[0]) +
    #         L1 * torch.sum(torch.abs(r))
    #     )
    
    # def calculate_ratios(self, intensity_data: np.ndarray, ratio_path: Path) -> np.ndarray:
    #     """Calculate or load channel ratios"""
    #     os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable CUDA launch blocking
    #     os.environ['TORCH_USE_CUDA_DSA'] = '1'
    #     torch.backends.cuda.matmul.allow_tf32 = False  # Disable TensorFloat-32 (TF32) for better error reporting

    #     if ratio_path.exists():
    #         return np.loadtxt(ratio_path).T
        
    #     # Initialize parameters
    #     n_cam = len(self.config.get_round_channels())
    #     initial = np.eye(len(self.channels))
    #     n_sub = int(self.config.N_SUBSET * self.config.FRAC_SAMPLED)
        
    #     # Setup tensors
    #     r_gpu = torch.from_numpy(initial).cuda(0).requires_grad_()
    #     # data_gpu = torch.from_numpy(intensity_data).to(self.device).double()
    #     data_gpu = torch.from_numpy(np.array(intensity_data)).cuda(0).double()
        
    #     # Optimization loop
    #     loss_hist = torch.zeros(self.config.EPOCHS)
    #     r_hist = torch.zeros((self.config.EPOCHS, n_cam, n_cam))
        
    #     for i in range(self.config.EPOCHS):
    #         if not i % self.config.RESAMPLE_ITER:
    #             sub_gpu = data_gpu[np.random.choice(
    #                 self.config.N_SUBSET,
    #                 n_sub,
    #                 replace=False
    #             )]
            
    #         loss = self.objective_fn(r_gpu.double(), sub_gpu, self.config.L1)
    #         loss.backward()
            
    #         loss_hist[i] = self.objective_fn(
    #             r_gpu.clone().double(),
    #             data_gpu,
    #             self.config.L1
    #         ).data
    #         r_hist[i] = r_gpu.clone().double()
            
    #         r_gpu.data -= self.config.LEARNING_RATE * r_gpu.grad.data
    #         r_gpu.data = torch.div(r_gpu.data, torch.norm(r_gpu.data))
    #         r_gpu.grad = None
        
    #     # Get optimized ratios
    #     optimized = r_hist[np.argmin(loss_hist)].detach().numpy()
    #     optimized = optimized / np.linalg.norm(optimized, axis=0)
        
    #     # Save ratios
    #     np.savetxt(
    #         ratio_path,
    #         100 * optimized.T / optimized.max(0)[..., None],
    #         delimiter='\t',
    #         fmt='%d'
    #     )
        
    #     return optimized

    def subset_spots_df(self, thresh_spots):
        if len(thresh_spots)> self.config.N_SUBSET:
            thresh_spots_subsetted = thresh_spots[::int(len(thresh_spots)/self.config.N_SUBSET)]
        else: 
            thresh_spots_subsetted = thresh_spots
        self.config.N_SUBSET = len(thresh_spots_subsetted)
        return thresh_spots_subsetted

    def calculate_ratios(self, intensity_data: np.ndarray, ratio_location: Path) -> np.ndarray:
        if not os.path.exists(ratio_location):
            # Max normalizing the ratios
            def norm(ratios): 
                return (100 * ratios / ratios.max()).astype(int)

            # Linear least squares objective function
            def objective_fn(r, subset, L1): 
                n_cam = len(self.config.get_round_spot_channels())
                r = r / torch.norm(r, dim=0)
                dot_products = torch.tile(subset @ r, (n_cam, 1, 1))
                ys = torch.tile(torch.unsqueeze(r,1), (1, subset.shape[0], 1))
                xs = torch.tile(torch.unsqueeze(torch.transpose(subset, 0, 1),2), (1, 1, n_cam))
                return torch.sum(torch.min(torch.norm(dot_products * ys - xs, dim=0), dim=1)[0]) + self.config.L1 * torch.sum(torch.abs(r)) 
            n_cam = len(self.config.get_round_spot_channels())
            initial = np.eye((n_cam)) # Initial ratios matrices for each channel

            n_sub = np.int32(self.config.N_SUBSET * self.config.FRAC_SAMPLED)
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable CUDA launch blocking
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            torch.backends.cuda.matmul.allow_tf32 = False  # Disable TensorFloat-32 (TF32) for better error reporting
            # initial = np.random.rand(n_cam, n_dye) # for checking that initialized performance is much better than with random lines
            r_gpu = torch.from_numpy(initial).cuda(0).requires_grad_()
            thresh_spots_df_subsetted = self.subset_spots_df(intensity_data)
            if thresh_spots_df_subsetted.shape[0] < self.config.N_SUBSET:
                self.config.N_SUBSET = thresh_spots_df_subsetted.shape[0]
            data_gpu = torch.from_numpy(np.array(thresh_spots_df_subsetted)).cuda(0).double()

            loss_hist = torch.zeros(self.config.EPOCHS)
            r_hist = torch.zeros((self.config.EPOCHS, n_cam, n_cam))

            for i in range(self.config.EPOCHS):
                if not i%self.config.RESAMPLE_ITER:
                    sub_gpu = data_gpu[np.random.choice(self.config.N_SUBSET, n_sub, replace=False)]
                if not i % 1000 and i or i == 1: print(i, loss_hist[i-1], flush=True)
                loss = objective_fn(r_gpu.double(), sub_gpu, self.config.L1)
                loss.backward()
                loss_hist[i] = objective_fn(r_gpu.clone().double(), data_gpu, self.config.L1).data
                # loss_hist[i] = objective_fn(r_gpu.clone().double(), sub_gpu, self.config.L1).data

                r_hist[i] = r_gpu.clone().double()
                r_gpu.data -= self.config.LEARNING_RATE * r_gpu.grad.data
                r_gpu.data = torch.div(r_gpu.data, torch.norm(r_gpu.data))
                r_gpu.grad = None

            optimized = r_hist[np.argmin(loss_hist)].detach().numpy()
            optimized = optimized/np.linalg.norm(optimized, axis=0)
            np.savetxt(ratio_location, 100 * optimized.T / optimized.max(0)[..., None], delimiter='\t', fmt='%d')
            print('original')
            for x in initial.T: print(norm(x))
            print('optimized')
            for x in optimized.T: print(norm(x))
            print('error', loss_hist.min()/len(intensity_data))
        ratios = np.loadtxt(ratio_location).T
        return ratios

    def subset_spots_for_ratio(self, intensity_data: np.ndarray, detection_channels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subset spots for ratio calculation with emphasis on high-intensity spots in their detection channels
        """
        channels = self.config.get_round_spot_channels()
        selected_indices = []
        
        # For each channel, prioritize its high-intensity spots
        for ch in channels:
            # Get spots detected in this channel
            channel_mask = detection_channels == ch
            channel_intensities = intensity_data[channel_mask]
            channel_indices = np.where(channel_mask)[0]
            
            if len(channel_indices) > 0:
                # Find high-intensity spots (top 20%)
                intensity_threshold = np.percentile(
                    channel_intensities[:, channels.index(ch)],  # Look at intensity in detection channel
                    80
                )
                high_intensity_mask = channel_intensities[:, channels.index(ch)] > intensity_threshold
                high_intensity_indices = channel_indices[high_intensity_mask]
                
                # Take more spots from high-intensity regions
                if len(high_intensity_indices) > 0:
                    n_high = min(len(high_intensity_indices), self.config.N_SUBSET // (2 * len(channels)))
                    selected_indices.extend(
                        np.random.choice(high_intensity_indices, n_high, replace=False)
                    )
        
        # Fill remaining spots proportionally if needed
        if len(selected_indices) < self.config.N_SUBSET:
            remaining_spots = self.config.N_SUBSET - len(selected_indices)
            remaining_indices = list(set(range(len(detection_channels))) - set(selected_indices))
            
            if remaining_indices:
                additional_indices = np.random.choice(remaining_indices, 
                                                min(remaining_spots, len(remaining_indices)), 
                                                replace=False)
                selected_indices.extend(additional_indices)
        
        selected_indices = np.array(selected_indices)
        return intensity_data[selected_indices], detection_channels[selected_indices]

    def calculate_ratios_weighted(self, intensity_data: np.ndarray, ratio_location: Path, detection_channels: np.ndarray) -> np.ndarray:
        if not os.path.exists(ratio_location):

            def norm(ratios): 
                return (100 * ratios / ratios.max()).astype(int)

            def objective_fn(r, subset, subset_weights, L1): 
                n_cam = len(self.config.get_round_spot_channels())
                r = r / torch.norm(r, dim=0)
                dot_products = torch.tile(subset @ r, (n_cam, 1, 1))
                ys = torch.tile(torch.unsqueeze(r,1), (1, subset.shape[0], 1))
                xs = torch.tile(torch.unsqueeze(torch.transpose(subset, 0, 1),2), (1, 1, n_cam))
                
                errors = torch.norm(dot_products * ys - xs, dim=0)
                # Ensure weights are properly shaped for broadcasting
                weighted_errors = errors * subset_weights.view(-1, 1)
                
                return torch.sum(torch.min(weighted_errors, dim=1)[0]) + L1 * torch.sum(torch.abs(r))

            n_cam = len(self.config.get_round_spot_channels())
            initial = np.eye((n_cam))
            n_sub = np.int32(self.config.N_SUBSET * self.config.FRAC_SAMPLED)
            
            # Setup CUDA
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
            torch.backends.cuda.matmul.allow_tf32 = False
            
            r_gpu = torch.from_numpy(initial).cuda(0).requires_grad_()
            
            # Subset the data while maintaining channel information
            subsetted_data, subsetted_channels = self.subset_spots_for_ratio(
                intensity_data, 
                detection_channels
            )
            self.config.N_SUBSET = len(subsetted_data)
            
            data_gpu = torch.from_numpy(subsetted_data).cuda(0).double()
            
            
            # Calculate weights with stronger emphasis on detection channel
            detection_weights = np.ones(len(subsetted_channels))
            channels = self.config.get_round_spot_channels()
            
            for i, channel in enumerate(channels):
                channel_mask = subsetted_channels == channel
                channel_intensities = subsetted_data[channel_mask]
                
                if len(channel_intensities) > 0:
                    # Base weights
                    detection_weights[channel_mask] = 10.0  # Higher base weight for detection channel
                    detection_weights[~channel_mask] = 0.1  # Much lower weight for other channels
                    
                    # Additional weight based on intensity in detection channel
                    if len(channel_intensities) > 0:
                        intensity_percentiles = np.percentile(
                            channel_intensities[:, i],
                            [25, 50, 75, 90]
                        )
                        
                        # Progressive weighting based on intensity
                        mask = channel_mask & (subsetted_data[:, i] > intensity_percentiles[3])
                        detection_weights[mask] *= 4.0  # Very high intensity spots
                        
                        mask = channel_mask & (subsetted_data[:, i] > intensity_percentiles[2])
                        detection_weights[mask] *= 2.0  # High intensity spots
                        
                        mask = channel_mask & (subsetted_data[:, i] < intensity_percentiles[0])
                        detection_weights[mask] *= 0.5  # Low intensity spots
            
            weights_gpu = torch.from_numpy(detection_weights).cuda(0).double()

            loss_hist = torch.zeros(self.config.EPOCHS)
            r_hist = torch.zeros((self.config.EPOCHS, n_cam, n_cam))

            for i in range(self.config.EPOCHS):
                if not i % self.config.RESAMPLE_ITER:
                    indices = np.random.choice(self.config.N_SUBSET, n_sub, replace=False)
                    sub_gpu = data_gpu[indices]
                    sub_weights = weights_gpu[indices]
                    
                if not i % 1000 and i or i == 1:
                    print(i, loss_hist[i-1], flush=True)
                    
                loss = objective_fn(r_gpu.double(), sub_gpu, sub_weights, self.config.L1)
                loss.backward()
                
                # Use full dataset for loss history
                loss_hist[i] = objective_fn(r_gpu.clone().double(), data_gpu, weights_gpu, self.config.L1).data
                r_hist[i] = r_gpu.clone().double()
                
                r_gpu.data -= self.config.LEARNING_RATE * r_gpu.grad.data
                r_gpu.data = torch.div(r_gpu.data, torch.norm(r_gpu.data))
                r_gpu.grad = None

            optimized = r_hist[np.argmin(loss_hist)].detach().numpy()
            optimized = optimized/np.linalg.norm(optimized, axis=0)
            
            np.savetxt(ratio_location, 100 * optimized.T / optimized.max(0)[..., None], 
                    delimiter='\t', fmt='%d')
            
            print('original')
            for x in initial.T: print(norm(x))
            print('optimized')
            for x in optimized.T: print(norm(x))
            print('error', loss_hist.min()/len(intensity_data))
            
        ratios = np.loadtxt(ratio_location).T
        return ratios
    
    def calculate_ratios_by_channel(self, intensity_data: np.ndarray, ratio_location: Path, detection_channels: np.ndarray) -> np.ndarray:
        """Calculate ratios using only spots detected in their primary channels"""
        if not os.path.exists(ratio_location):
            def norm(ratios): 
                return (100 * ratios / ratios.max()).astype(int)

            def objective_fn(r, subset, L1): 
                n_cam = len(self.channels) # MJD
                r = r / torch.norm(r, dim=0)
                dot_products = torch.tile(subset @ r, (n_cam, 1, 1))
                ys = torch.tile(torch.unsqueeze(r,1), (1, subset.shape[0], 1))
                xs = torch.tile(torch.unsqueeze(torch.transpose(subset, 0, 1),2), (1, 1, n_cam))
                return torch.sum(torch.min(torch.norm(dot_products * ys - xs, dim=0), dim=1)[0]) + L1 * torch.sum(torch.abs(r))

            n_cam = len(self.channels) # MJD
            channels = self.channels # MJD
            initial = np.eye((n_cam))
            
            # Select spots for each channel
            selected_spots = []
            for ch_idx, channel in enumerate(channels):
                # Get spots detected in this channel
                try: 
                    channel_mask = detection_channels == channel
                    channel_spots = intensity_data[channel_mask]
                    
                    if len(channel_spots) > 0:
                        # Filter based on intensity in detection channel
                        intensity_threshold = np.percentile(channel_spots[:, ch_idx], self.config.PERCENTILE)  # More stringent threshold
                        print(f" Intensity threshold: {intensity_threshold} for channel: {channel}")
                        high_intensity_mask = channel_spots[:, ch_idx] > intensity_threshold
                        high_intensity_spots = channel_spots[high_intensity_mask]
                        print(f" Number of high intensity spots: {len(high_intensity_spots)} in channel: {channel}")
                        
                        # Sample spots if we have too many
                        target_spots = min(len(high_intensity_spots), self.config.N_SUBSET // n_cam)
                        if target_spots==0: 
                            print(f'no spots found in channel {channel}')
                            continue
                        if len(high_intensity_spots) > target_spots:
                            selected_indices = np.random.choice(len(high_intensity_spots), target_spots, replace=False)
                            selected_spots.append(high_intensity_spots[selected_indices])
                        else:
                            selected_spots.append(high_intensity_spots)
                except Exception as e: 
                    print(f'Error {e} in sampling spots in channel {channel} for calculating dye line')
            selected_data = np.vstack(selected_spots)
            # Combine selected spots
            if len(selected_data)<1000:
                print("Warning: Less than 1000 spots met the selection criteria.")
                print(f"This is generally due to there being very few spots detected.")
                print("Saving default identity matrix as ratio between channel intensity.")
                optimized = initial
            else: 
                                
                # Setup CUDA and tensors
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['TORCH_USE_CUDA_DSA'] = '1'
                torch.backends.cuda.matmul.allow_tf32 = False
                
                r_gpu = torch.from_numpy(initial).cuda(0).requires_grad_()
                data_gpu = torch.from_numpy(selected_data).cuda(0).double()
                
                # Optimization
                n_sub = np.int32(len(selected_data) * self.config.FRAC_SAMPLED)
                loss_hist = torch.zeros(self.config.EPOCHS)
                r_hist = torch.zeros((self.config.EPOCHS, n_cam, n_cam))

                for i in range(self.config.EPOCHS):
                    if not i % self.config.RESAMPLE_ITER:
                        sub_gpu = data_gpu[np.random.choice(len(selected_data), n_sub, replace=False)]
                        
                    if not i % 1000 and i or i == 1:
                        print(i, loss_hist[i-1], flush=True)
                        
                    loss = objective_fn(r_gpu.double(), sub_gpu, self.config.L1)
                    loss.backward()
                    
                    loss_hist[i] = objective_fn(r_gpu.clone().double(), data_gpu, self.config.L1).data
                    r_hist[i] = r_gpu.clone().double()
                    
                    r_gpu.data -= self.config.LEARNING_RATE * r_gpu.grad.data
                    r_gpu.data = torch.div(r_gpu.data, torch.norm(r_gpu.data))
                    r_gpu.grad = None

                optimized = r_hist[np.argmin(loss_hist)].detach().numpy()
                optimized = optimized/np.linalg.norm(optimized, axis=0)
                print('error', loss_hist.min()/len(selected_data))

            
            np.savetxt(ratio_location, 100 * optimized.T / optimized.max(0)[..., None], 
                    delimiter='\t', fmt='%d')
            
            print('original')
            for x in initial.T: print(norm(x))
            print('optimized')
            for x in optimized.T: print(norm(x))
            
        ratios = np.loadtxt(ratio_location).T
        return ratios