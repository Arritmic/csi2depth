# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2025, Constantino Álvarez (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------


import os
import sys
import yaml
import cv2
import time
import numpy as np
import logging

import random
import copy
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image

# Add Main_Folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mmfi.mmfi_lib.mmfi import make_dataset, make_dataloader
from src.csi2depth.csi2depth import CSI2Depth_Generator, CSI2Depth_Discriminator

name_output_folder = ""
torch.cuda.empty_cache()
device = torch.device("cpu")
# Initialize CUDA explicitly
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.init()
#     torch.tensor([0.0], device=device)  # Forces context creation
# else:
#     device = torch.device("cpu")

loss_fn_simm = SSIM(win_size=7, win_sigma=1.2, data_range=1.0, size_average=True, channel=1)

# Load VGG16 for perceptual loss
from torchvision.models import vgg16, VGG16_Weights, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor

vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
features = create_feature_extractor(vgg, {'features.3': 'relu1_2', 'features.8': 'relu2_2'}).to(device)
features.eval()

alpha = 5
beta = 5
gpw = 10


###############################
#
#      DA FUNCTIONS
#
###############################

def add_gaussian_noise(csi_data, magnitude_std=0.01, phase_std=0.01):
    # Separate magnitude and phase
    magnitude = csi_data[:, :, 0, :]  # Shape: [A, S, T]
    phase = csi_data[:, :, 1, :]  # Shape: [A, S, T]

    # Add noise
    magnitude_noise = magnitude + torch.randn_like(magnitude) * magnitude_std
    phase_noise = phase + torch.randn_like(phase) * phase_std

    # Combine back
    csi_noisy = torch.stack((magnitude_noise, phase_noise), dim=2)  # Correct stacking
    return csi_noisy  # Shape: [A, S, 2, T]


def apply_phase_shift(csi_data, max_shift=np.pi):
    # Generate random phase shifts
    phase_shift = torch.rand(1) * 2 * max_shift - max_shift  # Scalar shift
    csi_data_shifted = csi_data.clone()
    csi_data_shifted[:, :, 1, :] += phase_shift  # Shift phase component
    return csi_data_shifted


def scale_magnitude(csi_data, scale_range=(0.9, 1.1)):
    # Generate random scaling factor
    scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
    csi_data_scaled = csi_data.clone()
    csi_data_scaled[:, :, 0, :] *= scale  # Scale magnitude component
    return csi_data_scaled


def shift_time(csi_data, max_shift=2):
    shift = np.random.randint(-max_shift, max_shift + 1)
    T = csi_data.shape[3]  # Correct index for time dimension
    shift = shift % T  # Ensure shift is within valid range
    csi_data_shifted = torch.roll(csi_data, shifts=shift, dims=3)  # Correct dim
    return csi_data_shifted  # Shape: [A, S, 2, T]


def simulate_multipath(csi_data, num_paths=2, delay_max=5, attenuation_range=(0.5, 1.0)):
    csi_augmented = csi_data.clone()
    T = csi_data.shape[3]  # Time dimension index

    for _ in range(num_paths):
        # Generate random delay and attenuation
        delay = torch.randint(1, min(delay_max + 1, T), (1,)).item()
        attenuation = torch.rand(1).item() * (attenuation_range[1] - attenuation_range[0]) + attenuation_range[0]

        # Create delayed and attenuated version
        delayed_csi = csi_data[:, :, :, :-delay]  # Shape: [A, S, 2, T - delay]

        # Pad the time dimension at the beginning to maintain shape
        pad_size = delay
        padding = torch.zeros((csi_augmented.shape[0], csi_augmented.shape[1], csi_augmented.shape[2], pad_size), dtype=csi_augmented.dtype, device=csi_augmented.device)
        delayed_csi_padded = torch.cat((padding, delayed_csi), dim=3)  # Shape: [A, S, 2, T]

        # Apply attenuation
        delayed_csi_padded *= attenuation

        # Combine with original CSI
        csi_augmented += delayed_csi_padded

    return csi_augmented  # Shape: [A, S, 2, T]


def augment_csi_data(csi_data, number=0):
    num_augm = random.randint(0, 3)
    # Apply one augmentation based on the number
    if num_augm == 0:
        csi_data = add_gaussian_noise(csi_data)
    elif num_augm == 1:
        csi_data = apply_phase_shift(csi_data)
    elif num_augm == 2:
        csi_data = scale_magnitude(csi_data)
    elif num_augm == 3:
        csi_data = shift_time(csi_data)
    elif num_augm == 4:
        csi_data = simulate_multipath(csi_data)
    return csi_data


def create_augmented_dataset(csi_dataset, num_augmentations=3):
    augmented_data = []

    index = 0
    amount_data = len(csi_dataset)
    for sample in csi_dataset:
        augmented_data.append(sample)  # Add the original sample

        csi_sample = sample['input_wifi-csi']
        gt_point_cloud = sample['input_depth']

        # Apply augmentations and add to the dataset
        for idx in range(num_augmentations):
            csi_augmented = augment_csi_data(csi_sample)
            sample_cop = copy.deepcopy(sample)
            sample_cop['input_wifi-csi'] = csi_augmented
            # 'input_lidar' remains the same
            augmented_data.append(sample_cop)
        if index % 100 == 0:
            print(f"  >> Data augmentation. Sample {index}/{amount_data}")
        index += 1

    return augmented_data


class AugmentedMMFiDataset(torch.utils.data.Dataset):
    def __init__(self, augmented_data):
        self.data = augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_padd2(batch):
    '''
    Pads batch of variable length if necessary and collates the batch.
    Assumes that all 'wifi-csi' samples have consistent shapes [A, S, 2, T].
    '''

    batch_data = {
        'modality': batch[0]['modality'],
        'scene': [sample['scene'] for sample in batch],
        'subject': [sample['subject'] for sample in batch],
        'action': [sample['action'] for sample in batch],
        'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
    }

    # Collate 'output'
    _output = [sample['output'] for sample in batch]  # Each is [Npoints, 3]
    _output = torch.stack(_output, dim=0)  # Shape: [B, Npoints, 3]
    batch_data['output'] = _output

    for mod in batch_data['modality']:
        if mod in ['mmwave', 'lidar']:
            # Assuming 'mmwave' and 'lidar' have their own handling
            _input = [torch.Tensor(sample['input_' + mod]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input, batch_first=True)
            batch_data['input_' + mod] = _input
        else:
            # Assuming 'wifi-csi' has shape [A, S, 2, T]
            _input = [torch.FloatTensor(sample['input_' + mod]) for sample in batch]
            _input = torch.stack(_input, dim=0)  # Shape: [B, A, S, 2, T]
            batch_data['input_' + mod] = _input

    return batch_data


###############################
#
#      LOSS FUNCTIONS: ARCH
#
###############################
# class PerceptualLoss(nn.Module):
#     def __init__(self, layers=('3', '8', '15'), weight=1.0):
#         """
#         layers: indices of VGG layers at which to extract features.
#         weight: weight of the perceptual loss.
#         """
#         super(PerceptualLoss, self).__init__()
#         vgg = models.vgg16(pretrained=True).features
#         self.layers = layers
#         self.vgg_layers = vgg.eval()
#         self.weight = weight
#         for param in self.vgg_layers.parameters():
#             param.requires_grad = False
#
#     def forward(self, generated, target):
#         # If images are single channel, replicate to 3 channels
#         if generated.size(1) == 1:
#             generated = generated.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#
#         loss = 0.0
#         x_gen = generated
#         x_tar = target
#         for i, layer in self.vgg_layers._modules.items():
#             x_gen = layer(x_gen)
#             x_tar = layer(x_tar)
#             if i in self.layers:
#                 loss += F.l1_loss(x_gen, x_tar)
#         return self.weight * loss

class PerceptualLoss(nn.Module):
    def __init__(self, resize=True, layers=None, weights=VGG19_Weights.IMAGENET1K_V1, loss_fn=nn.L1Loss()):  # Add layers and weights
        super(PerceptualLoss, self).__init__()

        if layers is None:  # Default layers if not specified
            layers = [2, 7, 12, 21, 30]  # relu1_2, relu2_2, relu3_2, relu4_2, relu5_2

        vgg = models.vgg19(weights=weights)  # Use weights argument
        features = list(vgg.features)

        self.vgg_layers = nn.ModuleList(features).eval()
        self.vgg_layers.requires_grad = False
        self.layers = layers
        self.criterion = loss_fn  # nn.L1Loss() or nn.MSELoss()
        self.resize = resize

    def forward(self, generated_image, real_image):
        if self.resize:
            generated_image = nn.functional.interpolate(generated_image, size=(224, 224), mode='bilinear', align_corners=False)
            real_image = nn.functional.interpolate(real_image, size=(224, 224), mode='bilinear', align_corners=False)

        generated_image = generated_image.repeat(1, 3, 1, 1)  # Repeat channel
        real_image = real_image.repeat(1, 3, 1, 1)  # Repeat channel

        generated_features = []
        real_features = []

        for i, layer in enumerate(self.vgg_layers):
            generated_image = layer(generated_image)
            real_image = layer(real_image)
            if i + 1 in self.layers:  # Check if it's the layer we want
                generated_features.append(generated_image)
                real_features.append(real_image)

        loss = 0
        for gen_feat, real_feat in zip(generated_features, real_features):
            loss += self.criterion(gen_feat, real_feat)

        return loss


# # Compute perceptual loss
# def perceptual_loss(gen_images, gt_images):
#     """
#     Compute perceptual loss between generated and ground truth images.
#
#     Args:
#         gen_images (Tensor): Generated images [B, 1, H, W].
#         gt_images (Tensor): Ground truth images [B, 1, H, W].
#
#     Returns:
#         Tensor: Perceptual loss value.
#     """
#     # Repeat the single-channel depth images across 3 channels
#     gen_images_3ch = gen_images.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
#     gt_images_3ch = gt_images.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
#
#     # Extract features
#     gen_features = features(gen_images_3ch)
#     gt_features = features(gt_images_3ch)
#
#     # Compute perceptual loss
#     loss = F.mse_loss(gen_features['relu2_2'], gt_features['relu2_2'])
#     return loss


###############################
#
#      MODEL FUNCTIONS: ARCH
#
###############################

###############################
#
#      MODEL FUNCTIONS
#
###############################


###############################
#
#      MAIN FUNCTIONS
#
###############################
# Training Function

import numpy as np


def calculate_depth_metrics2(gt_depth, predicted_depth):
    """
    Calculates standard depth estimation metrics.

    Args:
        gt_depth (numpy.ndarray): Ground truth depth image (height x width).
        predicted_depth (numpy.ndarray): Predicted depth image (height x width).

    Returns:
        dict: A dictionary containing the calculated metrics:
              - 'MAE': Mean Absolute Error
              - 'RMSE': Root Mean Squared Error
              - 'REL': Relative Absolute Error
              - 'delta_1.25': Threshold Accuracy with t=1.25
              - 'delta_1.25^2': Threshold Accuracy with t=1.25^2
              - 'delta_1.25^3': Threshold Accuracy with t=1.25^3
    """

    # Ensure valid depth values
    mask = (gt_depth > 0) & (predicted_depth > 0)  # Avoid invalid depth values (zero or negative)

    if np.sum(mask) == 0:
        raise ValueError("No valid depth values found for evaluation.")

    gt_depth = gt_depth[mask]
    predicted_depth = predicted_depth[mask]

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted_depth - gt_depth))

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((predicted_depth - gt_depth) ** 2))

    # Relative Absolute Error (REL)
    rel = np.mean(np.abs(predicted_depth - gt_depth) / gt_depth)

    # Threshold Accuracy (delta)
    threshold = np.maximum(gt_depth / predicted_depth, predicted_depth / gt_depth)

    delta_1_25 = np.mean(threshold < 1.25)
    delta_1_25_2 = np.mean(threshold < (1.25 ** 2))
    delta_1_25_3 = np.mean(threshold < (1.25 ** 3))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "REL": rel,
        "delta_1": delta_1_25,
        "delta_2": delta_1_25_2,
        "delta_3": delta_1_25_3
    }


def calculate_depth_metrics(gt_depth, predicted_depth):
    """
    Calculates standard depth estimation metrics.

    Args:
        gt_depth (numpy.ndarray): Ground truth depth image (height x width x 1 or height x width).
        predicted_depth (numpy.ndarray): Predicted depth image (height x width x 1 or height x width).

    Returns:
        dict: A dictionary containing the calculated metrics:
              - 'MAE': Mean Absolute Error
              - 'RMSE': Root Mean Squared Error
              - 'REL': Relative Absolute Error
              - 'delta_1.25': Threshold Accuracy with t=1.25
              - 'delta_1.25^2': Threshold Accuracy with t=1.25^2
              - 'delta_1.25^3': Threshold Accuracy with t=1.25^3
    """

    # # Ensure depth images are flattened for metric calculation
    # gt_depth_flat = gt_depth.flatten()
    # predicted_depth_flat = predicted_depth.flatten()
    #
    # # Remove invalid pixels (e.g., where ground truth depth is zero or invalid)
    # valid_mask = gt_depth_flat > 0 and predicted_depth_flat > 0  # Assuming zero or negative depth is invalid
    # gt_depth_valid = gt_depth_flat[valid_mask]
    # predicted_depth_valid = predicted_depth_flat[valid_mask]

    gt_depth_flat = gt_depth.flatten()
    predicted_depth_flat = predicted_depth.flatten()



    # Remove invalid pixels (e.g., where ground truth depth is zero or invalid)
    # AND remove predicted depth values that are zero or negative
    # valid_mask = (gt_depth_flat > 0) & (predicted_depth_flat > 0) # Added condition for predicted depth
    valid_mask = gt_depth_flat > 0   # Added condition for predicted depth
    gt_depth_valid = gt_depth_flat[valid_mask]
    predicted_depth_valid = predicted_depth_flat[valid_mask]
    # predicted_depth_valid[predicted_depth_valid == 0] = 1


    # # predicted_depth_valid[predicted_depth_valid > 255] = 255
    # # print(gt_depth_valid)
    # # print(predicted_depth_valid)
    # gt_depth_valid = gt_depth_flat
    # predicted_depth_valid = predicted_depth_flat
    # # gt_depth_valid[gt_depth_valid == 0] = 0.01
    # # predicted_depth_valid[predicted_depth_valid == 0] = 0.01


    if gt_depth_valid.size == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'REL': np.nan,
            'delta_1.25': np.nan,
            'delta_1.25^2': np.nan,
            'delta_1.25^3': np.nan
        }


    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(gt_depth_valid - predicted_depth_valid))

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((gt_depth_valid - predicted_depth_valid)**2))

    # Relative Absolute Error (REL)
    # rel = np.mean(np.abs(gt_depth_valid - predicted_depth_valid) / (gt_depth_valid + 1e-6)) # Added small epsilon to avoid division by zero
    rel = np.mean(np.abs(gt_depth_valid - predicted_depth_valid) / (gt_depth_valid + 1e-12)) # Added small epsilon to avoid division by zero


    # Threshold Accuracy (delta_t)
    deltas = {}
    thresholds = [1.25, 1.25**2, 1.25**3]
    indx_d = 1
    for t in thresholds:
        max_ratio = np.maximum(predicted_depth_valid / (gt_depth_valid + 1e-12), gt_depth_valid / (predicted_depth_valid + 1e-12)) # Added small epsilon
        delta_t = np.mean(max_ratio < t)
        deltas[f'delta_{indx_d}'] = delta_t # Formatted key for clarity, e.g., delta_1.250
        indx_d += 1

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'REL': rel,
        'delta_1': deltas['delta_1'],
        'delta_2': deltas['delta_2'], # 1.25^2 = 1.5625, using formatted value for key access
        'delta_3': deltas['delta_3']  # 1.25^3 = 1.953125, using formatted value for key access
    }

    # if metrics["REL"] > 5:
    #     print(metrics)
    #     print(gt_depth_flat)
    #     print("fdfkjfdkfkfdkfjdkf")
    #     print(predicted_depth_flat)
    #     input("sdfkjslfdölsjflösdjfl")

    # input("sdfkjslfdölsjflösdjfl")
    #


    return metrics


def figure_to_array(fig, target_size=None, dpi=400):
    """
    Convert a Matplotlib figure to a high-resolution NumPy array (RGB).
    Optionally resize to a target size (width, height) using Pillow.
    """
    fig.set_dpi(dpi)
    fig.canvas.draw()

    # Use buffer_rgba for rendering
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    img = buf.reshape(height, width, 4)  # RGBA

    # Convert RGBA to RGB using Pillow
    img = Image.fromarray(img).convert("RGB")

    # Resize to target size if specified
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)  # Updated method for resizing

    return np.array(img)

def visualize_wifi_csi2(wifi_csi_frame, frame_index):
    """
    Visualize the magnitude and phase of the CSI data for all 3 antennas.

    Upper row: Magnitude for each antenna
    Bottom row: Phase for each antenna (unwrapped phase).
    """
    # wifi_csi_frame shape: [num_antennas, num_subcarriers, 2 (magnitude and phase), num_time_slices]
    num_antennas, num_subcarriers, _, num_time_slices = wifi_csi_frame.shape

    # Extract magnitude and phase
    csi_magnitude = wifi_csi_frame[:, :, 0, :]  # Shape: [num_antennas, num_subcarriers, num_time_slices]
    csi_phase = wifi_csi_frame[:, :, 1, :]  # Shape: [num_antennas, num_subcarriers, num_time_slices]

    # Unwrap the phase to avoid discontinuities between subcarriers
    csi_phase_unwrapped = np.unwrap(csi_phase, axis=1)  # Unwrap along subcarriers

    # Create a figure with 2 rows (Magnitude and Phase), 3 columns (one for each antenna)
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))

    # Titles for the plots
    titles = [f"Antenna {i + 1}" for i in range(num_antennas)]

    # Plot magnitude in the upper row
    for ant in range(2):
        ax_mag = axes[0, ant]
        im_mag = ax_mag.imshow(csi_magnitude[ant], aspect='auto', cmap='viridis', extent=[0, num_time_slices, 0, num_subcarriers])
        ax_mag.set_title(f'{titles[ant]} Magnitude', fontsize=16)
        ax_mag.set_xlabel('Time Slices', fontsize=14)
        ax_mag.set_ylabel('Subcarriers', fontsize=14)
        fig.colorbar(im_mag, ax=ax_mag, shrink=0.6)

    # Plot phase in the bottom row (with unwrapped phase)
    for ant in range(2):
        ax_phase = axes[1, ant]
        im_phase = ax_phase.imshow(csi_phase_unwrapped[ant], aspect='auto', cmap='twilight', extent=[0, num_time_slices, 0, num_subcarriers])
        ax_phase.set_title(f'{titles[ant]} Phase (Unwrapped)', fontsize=16)
        ax_phase.set_xlabel('Time Slices', fontsize=14)
        ax_phase.set_ylabel('Subcarriers', fontsize=14)
        fig.colorbar(im_phase, ax=ax_phase, shrink=0.6)

    plt.tight_layout()
    # Convert the Matplotlib figure to a NumPy image array
    fig.canvas.draw()
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format (if needed)

    plt.close(fig)  # Close the figure to free memory

    return image_bgr  # Return the image in BGR format (OpenCV compatible)


def visualize_wifi_csi_antenna1_spectrograms_smooth(wifi_csi_frame, target_resolution=(640, 480)):
    """
    Visualize SMOOTHED magnitude and phase spectrograms for ANTENNA 1 of CSI data.

    Args:
        wifi_csi_frame (numpy.ndarray): CSI data, shape: [num_antennas, num_subcarriers, 2 (magnitude/phase), num_time_slices]
        target_resolution (tuple): Desired resolution (width, height) of spectrogram images.

    Returns:
        tuple: (magnitude_spectrogram_image, phase_spectrogram_image) - OpenCV BGR images, smoothed.
    """
    num_antennas, num_subcarriers, _, num_time_slices = wifi_csi_frame.shape

    # Extract magnitude and phase for ANTENNA 1 ONLY
    csi_magnitude_ant1 = wifi_csi_frame[0, :, 0, :]  # Shape: [num_subcarriers, num_time_slices] (Antenna 1, Magnitude)
    csi_phase_ant1 = wifi_csi_frame[0, :, 1, :]      # Shape: [num_subcarriers, num_time_slices] (Antenna 1, Phase)

    # Unwrap the phase to avoid discontinuities between subcarriers
    csi_phase_unwrapped_ant1 = np.unwrap(csi_phase_ant1, axis=0)  # Unwrap along subcarriers (axis=0 because now subcarriers is the first dimension)

    # --- Magnitude Spectrogram (Antenna 1) ---
    fig_mag, ax_mag = plt.subplots(figsize=(6.4, 4.8)) # Adjust figsize for desired output image size
    im_mag = ax_mag.imshow(csi_magnitude_ant1, aspect='auto', cmap='viridis', extent=[0, num_time_slices, 0, num_subcarriers], interpolation='none') # interpolation='none' for initial heatmap
    ax_mag.set_title('Antenna 1 Magnitude Spectrogram', fontsize=10) # Reduced fontsize
    ax_mag.set_xlabel('Time Slices', fontsize=8) # Reduced fontsize
    ax_mag.set_ylabel('Subcarriers', fontsize=8) # Reduced fontsize
    fig_mag.colorbar(im_mag, ax=ax_mag, shrink=0.7) # Reduced colorbar shrink
    fig_mag.canvas.draw()
    mag_spectrogram_np = np.frombuffer(fig_mag.canvas.tostring_rgb(), dtype=np.uint8)
    mag_spectrogram_image_raw = mag_spectrogram_np.reshape(fig_mag.canvas.get_width_height()[::-1] + (3,))
    mag_spectrogram_image_raw_bgr = cv2.cvtColor(mag_spectrogram_image_raw, cv2.COLOR_RGB2BGR) # To BGR
    plt.close(fig_mag) # Close figure

    # --- Phase Spectrogram (Antenna 1) ---
    fig_phase, ax_phase = plt.subplots(figsize=(6.4, 4.8)) # Adjust figsize
    im_phase = ax_phase.imshow(csi_phase_unwrapped_ant1, aspect='auto', cmap='twilight', extent=[0, num_time_slices, 0, num_subcarriers], interpolation='none') # interpolation='none' for initial heatmap
    ax_phase.set_title('Antenna 1 Phase Spectrogram (Unwrapped)', fontsize=10) # Reduced fontsize
    ax_phase.set_xlabel('Time Slices', fontsize=8) # Reduced fontsize
    ax_phase.set_ylabel('Subcarriers', fontsize=8) # Reduced fontsize
    fig_phase.colorbar(im_phase, ax=ax_phase, shrink=0.7) # Reduced colorbar shrink
    fig_phase.canvas.draw()
    phase_spectrogram_np = np.frombuffer(fig_phase.canvas.tostring_rgb(), dtype=np.uint8)
    phase_spectrogram_image_raw = phase_spectrogram_np.reshape(fig_phase.canvas.get_width_height()[::-1] + (3,))
    phase_spectrogram_image_raw_bgr = cv2.cvtColor(phase_spectrogram_image_raw, cv2.COLOR_RGB2BGR) # To BGR
    plt.close(fig_phase) # Close figure


    # --- Interpolate to Smooth ---
    target_width, target_height = target_resolution
    mag_spectrogram_image_smooth = cv2.resize(mag_spectrogram_image_raw_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR) # Bilinear interpolation
    phase_spectrogram_image_smooth = cv2.resize(phase_spectrogram_image_raw_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR) # Bilinear interpolation

    return mag_spectrogram_image_smooth, phase_spectrogram_image_smooth






def test_epoch(generator, discriminator, device, dataloader, loss, dataset_type="Validation"):
    generator.eval()
    discriminator.eval()
    g_loss_values = []
    d_loss_values = []

    # Define loss functions
    perceptual_loss = loss
    reconstruction_loss = nn.MSELoss()

    index_total_time = 0
    total_time = 0

    # Setup video writer: 15 FPS, composite frame resolution 1280x960 (each subvideo 640x480)
    output_video_path = "test_epoch_output2.avi"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 960))
    # Initialize a global frame counter
    frame_counter = 0

    # Define font parameters for overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # white
    thickness = 2

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            wifi_csi_frame = batch['input_wifi-csi'].to(device)
            gt_depth_images = batch['input_depth'].to(device)
            wifi_csi_frame = (wifi_csi_frame - wifi_csi_frame.mean(dim=(2, 3), keepdim=True)) / (wifi_csi_frame.std(dim=(2, 3), keepdim=True) + 1e-8)

            # Normalize and permute RGB images
            gt_depth_images = gt_depth_images.unsqueeze(1)
            gt_depth_images = F.interpolate(gt_depth_images, size=(120, 160), mode='bilinear', align_corners=False)
            gt_depth_images = (gt_depth_images / 127.5) - 1.0  # Normalize to [-1, 1]

            batch_size = gt_depth_images.size(0)

            # Generate images
            t0 = time.time()
            gen_images = generator(wifi_csi_frame)
            total_time += time.time() - t0
            index_total_time += len(batch['input_depth'])

            # Discriminator outputs
            real_validity = discriminator(gt_depth_images, wifi_csi_frame)
            fake_validity = discriminator(gen_images, wifi_csi_frame)

            # Generator loss
            g_adv_loss = -torch.mean(fake_validity)  # WGAN-GP generator loss
            g_recon_loss = reconstruction_loss(gen_images, gt_depth_images)
            g_perceptual_loss = perceptual_loss(gen_images, gt_depth_images)

            g_loss = g_adv_loss + alpha * g_recon_loss + beta * g_perceptual_loss
            g_loss_values.append(g_loss.item())

            # Discriminator loss (real and fake components)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            d_loss_values.append(d_loss.item())

            # Metric comparison


            # Save real and generated images every few epochs
            save_images = True
            if save_images:
                for frame_idx in range(wifi_csi_frame.size(0)):
                    frame_counter += 1
                    subject = batch['subject'][frame_idx]
                    scene = batch['scene'][frame_idx]
                    action = batch['action'][frame_idx]
                    print(f"     >> Saving: Frame {frame_idx} in batch {batch_idx}/{len(dataloader)}. Info: {subject} {scene} {action}")

                    # Prepare real and generated depth images
                    real_image = gt_depth_images[frame_idx].cpu().numpy().transpose(1, 2, 0)
                    gen_image = gen_images[frame_idx].cpu().numpy().transpose(1, 2, 0)
                    real_image = ((real_image + 1.0) * 127.5).astype(np.uint8)
                    gen_image = ((gen_image + 1.0) * 127.5).astype(np.uint8)
                    real_image_resized = cv2.resize(real_image, (640, 480))
                    gen_image_resized = cv2.resize(gen_image, (640, 480))
                    if real_image_resized.ndim == 2 or real_image_resized.shape[2] == 1:
                        real_image_resized = cv2.cvtColor(real_image_resized, cv2.COLOR_GRAY2BGR)
                    else:
                        real_image_resized = cv2.cvtColor(real_image_resized, cv2.COLOR_RGB2BGR)
                    if gen_image_resized.ndim == 2 or gen_image_resized.shape[2] == 1:
                        gen_image_resized = cv2.cvtColor(gen_image_resized, cv2.COLOR_GRAY2BGR)
                    else:
                        gen_image_resized = cv2.cvtColor(gen_image_resized, cv2.COLOR_RGB2BGR)

                    # Overlay "REAL" and "GEN" labels on the respective depth images
                    cv2.putText(real_image_resized, "REAL", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
                    cv2.putText(real_image_resized, f" Frame {frame_idx}. User: {subject}, Room: {scene}, Action: {action}", (10, 460), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(gen_image_resized, "GEN", (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

                    # Prepare CSI spectrogram images
                    wifi_csi_np = batch['input_wifi-csi'][frame_idx].cpu().numpy()  # Get the entire CSI frame [antennas, subcarriers, 2, time_slices]
                    # mag_spectrogram_image, phase_spectrogram_image = visualize_wifi_csi_antenna1_spectrograms_smooth(wifi_csi_np)  # Pass the full CSI frame
                    mag_spectrogram_image = visualize_wifi_csi2(wifi_csi_np, 0)  # Pass the full CSI frame

                    resized_mag_csi = cv2.resize(mag_spectrogram_image, (1280, 960))
                    # resized_phase_csi = cv2.resize(phase_spectrogram_image, (640, 480))

                    # Create composite frame: top row (real left, generated right), bottom row (CSI magnitude left, CSI phase right)
                    top_row = np.vstack((real_image_resized, gen_image_resized))
                    # bottom_row = np.hstack((resized_mag_csi, resized_phase_csi))
                    composite_frame = np.hstack((top_row, resized_mag_csi))

                    # Write composite frame to video
                    video_writer.write(composite_frame)

                    # Optionally display the composite frame
                    cv2.imshow("Composite", composite_frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        video_writer.release()
        cv2.destroyAllWindows()
        # Average losses
        avg_d_loss = np.mean(d_loss_values)
        avg_g_loss = np.mean(g_loss_values)
        print(f"    >> {dataset_type} >> D Loss: {avg_d_loss:.6f}, G Loss: {avg_g_loss:.6f}")

        print(f"     >> Total time: {total_time:.2f}")
        print(f"     >> Total images testing: {index_total_time}")
        print(f"     >> Average time per estimation: {(total_time / index_total_time) * 1000:.4f}")

        return avg_d_loss, avg_g_loss


def save_checkpoint_last(model, optimizer, epoch, loss, checkpoint_dir, name_model="default"):
    """
    Save a checkpoint of the model state, optimizer state, epoch, and loss, along with model parameters.

    :param model: Model to save.
    :param optimizer: Optimizer to save.
    :param epoch: Current epoch.
    :param loss: Loss value at this epoch.
    :param checkpoint_dir: Directory to save checkpoints.
    :param name_model: Name for the checkpoint file.
    """
    import os
    import torch

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_last_{name_model}.pth")

    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Dynamically add model parameters if the encoder exists
    if hasattr(model, 'csi_encoder'):
        csi_encoder = model.csi_encoder
        checkpoint.update({
            'embedding_dim': getattr(csi_encoder, 'embedding_dim', None),
            'num_heads': getattr(csi_encoder, 'num_heads', None),
            'num_encoder_layers': getattr(csi_encoder, 'num_encoder_layers', None),
            'num_points': getattr(csi_encoder, 'num_points', None),
            'num_antennas': getattr(csi_encoder, 'num_antennas', None),
            'num_subcarriers': getattr(csi_encoder, 'num_subcarriers', None),
            'num_time_slices': getattr(csi_encoder, 'num_time_slices', None),
        })

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"     >> Checkpoint saved at: {checkpoint_path}")


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, name_model="default"):
    """
    Save a checkpoint of the model state, optimizer state, epoch, and loss, along with model parameters.

    :param model: Model to save.
    :param optimizer: Optimizer to save.
    :param epoch: Current epoch.
    :param loss: Loss value at this epoch.
    :param checkpoint_dir: Directory to save checkpoints.
    :param name_model: Name for the checkpoint file.
    """
    import os
    import torch

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define the checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}_{name_model}.pth")

    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Dynamically add model parameters if the encoder exists
    if hasattr(model, 'csi_encoder'):
        csi_encoder = model.csi_encoder
        checkpoint.update({
            'embedding_dim': getattr(csi_encoder, 'embedding_dim', None),
            'num_heads': getattr(csi_encoder, 'num_heads', None),
            'num_encoder_layers': getattr(csi_encoder, 'num_encoder_layers', None),
            'num_points': getattr(csi_encoder, 'num_points', None),
            'num_antennas': getattr(csi_encoder, 'num_antennas', None),
            'num_subcarriers': getattr(csi_encoder, 'num_subcarriers', None),
            'num_time_slices': getattr(csi_encoder, 'num_time_slices', None),
        })

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"     >> Checkpoint saved at: {checkpoint_path}")


# def load_checkpoint(model, optimizer, checkpoint_path):
#     """
#     Load model and optimizer state from a checkpoint, with parameter checks.

#     Args:
#         model (nn.Module): The model instance.
#         optimizer (torch.optim.Optimizer): The optimizer instance.
#         checkpoint_path (str): Path to the checkpoint file.

#     Returns:
#         int: The epoch to resume training from.
#     """
#     if not os.path.isfile(checkpoint_path):
#         print(f"     >> No checkpoint found at: {checkpoint_path}")
#         return 0  # Start from epoch 0 if no checkpoint is available

#     print(f"Loading checkpoint from {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']
#     print(f"     >> Resumed training from epoch {start_epoch}")

#     # Validate model parameters with assertions if attributes exist
#     if hasattr(model, 'csi_encoder'):
#         assert model.csi_encoder.embedding_dim == checkpoint['embedding_dim'], "Mismatch in embedding dimensions!"
#         assert model.csi_encoder.num_heads == checkpoint['num_heads'], "Mismatch in number of heads!"
#         assert model.csi_encoder.num_antennas == checkpoint['num_antennas'], "Mismatch in number of antennas!"
#         assert model.csi_encoder.num_subcarriers == checkpoint['num_subcarriers'], "Mismatch in number of subcarriers!"
#         assert model.csi_encoder.num_points == checkpoint['num_points'], "Mismatch in number of cloud points!"
#         assert model.csi_encoder.num_time_slices == checkpoint['num_time_slices'], "Mismatch in number of time slices!"

#     return start_epoch


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from a checkpoint, with parameter checks.

    Args:
        model (nn.Module): The model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        int: The epoch to resume training from.
        float: The loss value from the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"     >> No checkpoint found at: {checkpoint_path}")
        return 0, float('inf')  # Start from epoch 0, return high loss if checkpoint is missing

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))  # Default to a high value if 'loss' key is missing

    print(f"     >> Resumed training from epoch {start_epoch}, with checkpoint loss: {loss:.6f}")

    # Validate model parameters with assertions if attributes exist
    if hasattr(model, 'csi_encoder'):
        assert model.csi_encoder.embedding_dim == checkpoint['embedding_dim'], "Mismatch in embedding dimensions!"
        assert model.csi_encoder.num_heads == checkpoint['num_heads'], "Mismatch in number of heads!"
        assert model.csi_encoder.num_antennas == checkpoint['num_antennas'], "Mismatch in number of antennas!"
        assert model.csi_encoder.num_subcarriers == checkpoint['num_subcarriers'], "Mismatch in number of subcarriers!"
        assert model.csi_encoder.num_points == checkpoint['num_points'], "Mismatch in number of cloud points!"
        assert model.csi_encoder.num_time_slices == checkpoint['num_time_slices'], "Mismatch in number of time slices!"

    return start_epoch, loss


def save_final_model(generator, discriminator, model_version, num_epochs, learning_rate, optimizer_name, save_dir="../models/"):
    """
    Save the final generator and discriminator models after training.

    Args:
        generator (nn.Module): Trained generator model.
        discriminator (nn.Module): Trained discriminator model.
        model_version (str): Version name for the model.
        num_epochs (int): Number of epochs the model was trained for.
        learning_rate (float): Learning rate used in training.
        optimizer_name (str): Name of the optimizer.
        save_dir (str): Directory to save the final models.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get the current date for the filename
    date_str = time.strftime("%d%m%y")

    # Create metadata to save with the model
    metadata = {
        'model_version': model_version,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'optimizer_name': optimizer_name,
        'date_saved': date_str,
    }

    # Save the generator
    generator_path = os.path.join(
        save_dir,
        f"csi2depth_generator_{date_str}_{model_version}_ep{num_epochs}_lr{learning_rate:.0e}_{optimizer_name}.pth"
    )
    torch.save({'model_state_dict': generator.state_dict(), 'metadata': metadata}, generator_path)
    print(f" >> Generator model saved: {generator_path}")

    # Save the discriminator
    discriminator_path = os.path.join(
        save_dir,
        f"csi2depth_discriminator_{date_str}_{model_version}_ep{num_epochs}_lr{learning_rate:.0e}_{optimizer_name}.pth"
    )
    torch.save({'model_state_dict': discriminator.state_dict(), 'metadata': metadata}, discriminator_path)
    print(f" >> Discriminator model saved: {discriminator_path}")


def load_checkpoint_for_transfer(model, checkpoint_path):
    """
    Load model weights from a checkpoint for transfer learning.

    Args:
        model (nn.Module): The model instance.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        None
    """
    if not os.path.isfile(checkpoint_path):
        print(f"     >> No checkpoint found at: {checkpoint_path}")
        return

    print(f"Loading checkpoint weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)  # Set to False for legacy checkpoints
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"     >> Model weights loaded for transfer learning")

    # Optional: Validate model parameters with assertions
    if hasattr(model, 'csi_encoder'):
        assert model.csi_encoder.embedding_dim == checkpoint.get('embedding_dim', model.csi_encoder.embedding_dim), "Mismatch in embedding dimensions!"
        assert model.csi_encoder.num_heads == checkpoint.get('num_heads', model.csi_encoder.num_heads), "Mismatch in number of heads!"
        assert model.csi_encoder.num_antennas == checkpoint.get('num_antennas', model.csi_encoder.num_antennas), "Mismatch in number of antennas!"
        assert model.csi_encoder.num_subcarriers == checkpoint.get('num_subcarriers', model.csi_encoder.num_subcarriers), "Mismatch in number of subcarriers!"
        # assert model.csi_encoder.num_points == checkpoint.get('num_points', model.csi_encoder.num_points), "Mismatch in number of cloud points!"
        assert model.csi_encoder.num_time_slices == checkpoint.get('num_time_slices', model.csi_encoder.num_time_slices), "Mismatch"


###############################
#
#      MAIN FUNCTION
#
###############################
def main():
    print('\n')
    print('*******************************************************************************************')
    print('         MMSLAB CSI2DEPTH: Starting inference test to visualize the model results         *')
    print('*******************************************************************************************')
    print('\n')

    #############################
    #       LOAD DATASET        #
    #############################
    yaml_config_file = "../data/configurations/config_mmfi_csi2depth.yaml"
    dataset_root = "/media/arritmic/MMST003/DATABASES/Joint_Comm_and_Sensing/MMFI-Dataset/data3"
    # dataset_root = "/path/to/the/MMFI-Dataset/data"
    print(f"  * Config file: {yaml_config_file}")
    print(f"  * Dataset path: {dataset_root}")
    print('\n')


    with open(yaml_config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # print(val_loader)

    print(f"[TRAINING]")
    # batch_size = config['train_loader']['batch_size']
    # print(f"  * Batch size: {batch_size}")
    print(f"    >> [CSI2DEPTH Train] Trainset samples: {len(train_loader)}. Batch size: {config['train_loader']['batch_size']}")
    print(f"    >> [CSI2DEPTH Train] Testset samples: {len(val_loader)}")
    # input("paksdsdfmn")


    print(val_loader)
    # for batch in val_loader:
    #     print(batch)
    #     input("para")

    #############################
    #        MODEL CONFIG       #
    #############################
    # Initialize device
    torch.cuda.empty_cache()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    logging.info("----- Model and Training Information -----")
    logging.info(f"    >> [CSI2DEPTH Train] Device: {device}")
    logging.info(f"\n")

    # Model Parameters
    embedding_dim = 256
    num_heads = 8
    num_encoder_layers = 4
    num_antennas = 3
    num_subcarriers = 114
    num_time_slices = 10

    # Initialize Generator
    generator = CSI2Depth_Generator(
        csi_embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_antennas=num_antennas,
        num_subcarriers=num_subcarriers,
        num_time_slices=num_time_slices,
        output_height=120,
        output_width=160
    ).to(device)

    # Initialize Discriminator with the same AdvancedCSIEncoder configuration
    discriminator = CSI2Depth_Discriminator(
        csi_embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_antennas=num_antennas,
        num_subcarriers=num_subcarriers,
        num_time_slices=num_time_slices
    ).to(device)


    checkpoint_path_generator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004af_generator.pth" # Paper SCIA
    checkpoint_path_discriminator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004af_discriminator.pth" # Paper SCIA

    checkpoint_path_generator = "../models/good_ones/checkpoint_epoch_130_dcsi2depth25_gan_arch004ab3_generator.pth"
    checkpoint_path_discriminator = "../models/good_ones/checkpoint_epoch_130_dcsi2depth25_gan_arch004ab3_discriminator.pth"

    checkpoint_path_generator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004ab3d_generator.pth"
    checkpoint_path_discriminator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004ab3d_discriminator.pth"

    checkpoint_path_generator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004ad2_generator.pth"
    checkpoint_path_discriminator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004ad2_discriminator.pth"

    # checkpoint_path_generator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004ab3_generatorb.pth"
    # checkpoint_path_discriminator = "../models/good_ones/checkpoint_epoch_last_dcsi2depth25_gan_arch004ab3_discriminatorb.pth"

    load_checkpoint_for_transfer(generator, checkpoint_path_generator)
    load_checkpoint_for_transfer(discriminator, checkpoint_path_discriminator)



    global alpha
    global beta
    global gpw
    alpha = 10
    beta = 15
    gpw = 20



    t0 = time.time()

    logging.info(f"\n############################")
    logging.info(f"\n#     Training Process     #")
    logging.info(f"\n############################")

    perceptual_loss_fn = PerceptualLoss(layers=[2, 7, 12, 21, 30], weights=VGG19_Weights.IMAGENET1K_V1).to(device)


    # Validation
    val_d_loss, val_g_loss = test_epoch(
        generator=generator,
        discriminator=discriminator,
        device=device,
        dataloader=val_loader,
        loss=perceptual_loss_fn,
        dataset_type="Validation"
    )


    print(f"Total training time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
