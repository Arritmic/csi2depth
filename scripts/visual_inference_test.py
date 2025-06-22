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
from src.loss.mmslab_loss import PerceptualLoss

name_output_folder = ""
torch.cuda.empty_cache()
device = torch.device("cpu")

alpha = 5
beta = 5
gpw = 10


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
#      VISUALIZATION FUNCTIONS
#
###############################

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


def visualize_wifi_csi(wifi_csi_frame, frame_index):
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


###############################
#
#      MAIN FUNCTIONS
#
###############################
def test_epoch(generator, discriminator, input_device, dataloader, loss, dataset_type="Validation"):
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
    # output_video_path = "test_epoch_output.avi"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = 15
    # video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 960))

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

                    mag_spectrogram_image = visualize_wifi_csi(wifi_csi_np, 0)  # Pass the full CSI frame

                    resized_mag_csi = cv2.resize(mag_spectrogram_image, (1280, 960))
                    # resized_phase_csi = cv2.resize(phase_spectrogram_image, (640, 480))

                    # Create composite frame: top row (real left, generated right), bottom row (CSI magnitude left, CSI phase right)
                    top_row = np.vstack((real_image_resized, gen_image_resized))
                    # bottom_row = np.hstack((resized_mag_csi, resized_phase_csi))
                    composite_frame = np.hstack((top_row, resized_mag_csi))

                    # Write composite frame to video
                    # video_writer.write(composite_frame)

                    # Optionally display the composite frame
                    cv2.imshow("Composite", composite_frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        # video_writer.release()
        # cv2.destroyAllWindows()
        # Average losses
        avg_d_loss = np.mean(d_loss_values)
        avg_g_loss = np.mean(g_loss_values)
        print(f"    >> {dataset_type} >> D Loss: {avg_d_loss:.6f}, G Loss: {avg_g_loss:.6f}")

        print(f"     >> Total time: {total_time:.2f}")
        print(f"     >> Total images testing: {index_total_time}")
        print(f"     >> Average time per estimation: {(total_time / index_total_time) * 1000:.4f}")

        return avg_d_loss, avg_g_loss


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

    train_dataset, val_dataset = make_dataset(dataset_root, config, subsampling=1, frame_limit=50)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    print(f"[TESTING]")
    print(f"    >> [CSI2Depth Test] Testset samples: {len(val_loader)}")
    print(f"    >> [CSI2Depth Test] Selected data split: {config['split_to_use']}")

    #############################
    #        MODEL CONFIG       #
    #############################
    torch.cuda.empty_cache()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

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

    checkpoint_path_generator = "../models/checkpoints/checkpoint_epoch_last_dcsi2depth25_gan_arch004af_generator.pth"  # Paper SCIA
    checkpoint_path_discriminator = "../models/checkpoints/checkpoint_epoch_last_dcsi2depth25_gan_arch004af_discriminator.pth"  # Paper SCIA

    checkpoint_path_generator = "../models/checkpoints/checkpoint_epoch_last_dcsi2depth25_gan_arch004ad2_generator.pth"
    checkpoint_path_discriminator = "../models/checkpoints/checkpoint_epoch_last_dcsi2depth25_gan_arch004ad2_discriminator.pth"

    load_checkpoint_for_transfer(generator, checkpoint_path_generator)
    load_checkpoint_for_transfer(discriminator, checkpoint_path_discriminator)

    global alpha
    global beta
    global gpw
    alpha = 10
    beta = 15
    gpw = 20

    t0 = time.time()

    perceptual_loss_fn = PerceptualLoss(layers=[2, 7, 12, 21, 30]).to(device)

    # Validation
    val_d_loss, val_g_loss = test_epoch(
        generator=generator,
        discriminator=discriminator,
        input_device=device,
        dataloader=val_loader,
        loss=perceptual_loss_fn,
        dataset_type="Validation"
    )

    print(f"Total training time: {(time.time() - t0) / 60:.2f} minutes")


if __name__ == '__main__':
    main()
