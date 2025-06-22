# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2025, Constantino Álvarez Casado, Manuel Lage Cañellas, Janne Mustaniemi, Matteo Pedone, Olli Silvén, Miguel Bordallo López  (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerEncoderCSI2Depth(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(TransformerEncoderCSI2Depth, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src shape: [batch_size, sequence_length, embedding_dim]
        output = self.transformer_encoder(src)
        return output  # Shape: [batch_size, sequence_length, embedding_dim]


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [batch_size * num_features, in_channels, sequence_length]
        x = self.conv(x)  # Shape: [batch_size * num_features, out_channels, sequence_length]
        x = F.relu(x)  # Apply activation
        x = x.mean(dim=2)  # Average over the sequence_length dimension
        # Now x has shape: [batch_size * num_features, out_channels]
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        """
        query: Tensor of shape [batch_size, seq_len, embed_dim]
        key: Tensor of shape [batch_size, seq_len, embed_dim]
        value: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        attention_output, _ = self.cross_attention(query, key, value)
        return attention_output


class AdvancedCSIEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers,
                 num_antennas=3, num_subcarriers=114, num_time_slices=10):
        super(AdvancedCSIEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.num_time_slices = num_time_slices
        self.num_heads = num_heads

        # Input Encoding
        # self.linear_proj = nn.Linear(num_time_slices * 2, embedding_dim)
        self.temporal_encoder = TemporalEncoder(in_channels=2, out_channels=embedding_dim)

        # Positional Encodings
        self.antenna_embeddings = nn.Embedding(num_antennas, embedding_dim)
        self.subcarrier_embeddings = nn.Embedding(num_subcarriers, embedding_dim)

        # Transformer Encoder
        self.encoder = TransformerEncoderCSI2Depth(embedding_dim, num_heads, num_encoder_layers)

    def forward(self, wifi_csi_frame):
        # wifi_csi_frame shape: [batch_size, num_antennas, num_subcarriers, 2, num_time_slices]
        batch_size = wifi_csi_frame.size(0)
        num_antennas = self.num_antennas
        num_subcarriers = self.num_subcarriers
        num_time_slices = self.num_time_slices

        num_features = self.num_antennas * self.num_subcarriers

        csi_data = wifi_csi_frame.permute(0, 1, 2, 4, 3)

        csi_data = csi_data.reshape(batch_size, num_features, self.num_time_slices, 2)

        csi_data = csi_data.permute(0, 1, 3, 2)

        # Reshape to merge batch_size and num_features
        csi_data = csi_data.reshape(batch_size * num_features, 2, self.num_time_slices)

        # Apply Temporal Encoder
        temporal_features = self.temporal_encoder(csi_data)

        embeddings = temporal_features.view(batch_size, num_features, self.embedding_dim)  # [batch_size, 342, embedding_dim]

        device = wifi_csi_frame.device

        # Antenna and subcarrier positional encodings
        antenna_indices = torch.arange(num_antennas, device=device).unsqueeze(1).expand(-1, num_subcarriers).reshape(-1)
        subcarrier_indices = torch.arange(num_subcarriers, device=device).unsqueeze(0).expand(num_antennas, -1).reshape(-1)

        antenna_encodings = self.antenna_embeddings(antenna_indices)
        subcarrier_encodings = self.subcarrier_embeddings(subcarrier_indices)

        # Sum antenna and subcarrier encodings
        positional_encodings = antenna_encodings + subcarrier_encodings  # Shape: [342, embedding_dim]

        # Expand positional encodings to match batch size
        positional_encodings = positional_encodings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, 342, embedding_dim]

        # Add positional encodings to embeddings
        embeddings = embeddings + positional_encodings  # Shape: [batch_size, 342, embedding_dim]

        # Transformer Encoder
        encoder_output = self.encoder(embeddings)  # Shape: [batch_size, 342, embedding_dim]

        return encoder_output


class CSI2Depth_Generator(nn.Module):
    def __init__(self, csi_embedding_dim, num_heads, num_encoder_layers,
                 num_antennas, num_subcarriers, num_time_slices,
                 output_height=120, output_width=160):
        super(CSI2Depth_Generator, self).__init__()
        self.output_height = output_height
        self.output_width = output_width

        # CSI Encoder
        self.csi_encoder = AdvancedCSIEncoder(
            embedding_dim=csi_embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_antennas=num_antennas,
            num_subcarriers=num_subcarriers,
            num_time_slices=num_time_slices
        )

        # Calculate input channels to the first ConvTranspose2d layer
        self.gen_input_channels = num_antennas * num_subcarriers * csi_embedding_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(self.gen_input_channels, 2048),  # Adjust output channels as needed
            nn.ReLU(inplace=True)
        )

        # Transposed Convolution Layers for Depth Image Generation
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.2),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.3),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # Final depth image layer
            # nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, csi_data):
        # Encode CSI data
        csi_features = self.csi_encoder(csi_data)  # Expected shape: [B, tokens, csi_embedding_dim]

        batch_size = csi_features.size(0)
        flattened_features = csi_features.permute(0, 2, 1).reshape(batch_size, -1)  # Fixed reshaping

        # gen_input = flattened_features.unsqueeze(2).unsqueeze(3)  # [B, 32832, 1, 1]
        compressed_features = self.bottleneck(flattened_features).unsqueeze(2).unsqueeze(3)
        generated_image = self.generator(compressed_features)

        generated_image = F.interpolate(generated_image, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)
        return generated_image


class CSI2Depth_Discriminator(nn.Module):
    def __init__(self, csi_embedding_dim=192, num_heads=8, num_encoder_layers=4,
                 num_antennas=3, num_subcarriers=114, num_time_slices=10):
        super(CSI2Depth_Discriminator, self).__init__()

        # CSI Encoder
        self.csi_encoder = AdvancedCSIEncoder(
            embedding_dim=csi_embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_antennas=num_antennas,
            num_subcarriers=num_subcarriers,
            num_time_slices=num_time_slices
        )

        # Cross-Attention Module
        self.cross_attention = CrossAttention(embed_dim=csi_embedding_dim, num_heads=num_heads)

        # Feature Extractor for Depth Image
        self.image_feature_extractor = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)),  # [B, 64, 60, 80]
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # [B, 128, 30, 40]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(p=0.3),  # Dropout added to prevent overfitting
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # [B, 256, 15, 20]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),  # [B, 512, 8, 10]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global feature extraction -> [B, 512, 1, 1]
        )

        # **Projection Layer to Match CSI Embedding Size**
        self.feature_projection = nn.Linear(512, csi_embedding_dim)  # Ensure matching size

        # Residual Connection to Enhance Feature Learning
        self.residual_fc = nn.Sequential(
            nn.Linear(csi_embedding_dim, csi_embedding_dim),
            nn.LayerNorm(csi_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(csi_embedding_dim, csi_embedding_dim),
            nn.LayerNorm(csi_embedding_dim)
        )

        # Final Classification Block
        self.classifier = nn.Sequential(
            nn.Linear(csi_embedding_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)  # Binary classification: Real/Fake
        )

    def forward(self, img, csi_data):
        """
        img: Depth image -> [B, 1, 120, 160]
        csi_data: CSI input -> [B, num_antennas, num_subcarriers, 2, num_time_slices]
        """

        # Encode CSI data
        csi_features = self.csi_encoder(csi_data)  # [B, num_points, embedding_dim]

        # Extract features from depth image
        img_features = self.image_feature_extractor(img)  # [B, 512, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)  # [B, 512]

        # **Project Image Features to Match CSI Embedding Dimension**
        img_features = self.feature_projection(img_features)  # [B, csi_embedding_dim]
        img_features = img_features.unsqueeze(1)  # Reshape to [B, 1, csi_embedding_dim]

        # Apply Cross-Attention: CSI (Query) attends to Depth Image Features (Key/Value)
        attended_csi_features = self.cross_attention(csi_features, img_features, img_features)  # [B, num_points, csi_embedding_dim]

        # Aggregate Features for Classification
        fused_features = attended_csi_features.mean(dim=1)  # Global pooling: [B, embedding_dim]

        # Apply Residual Connection
        fused_features = fused_features + self.residual_fc(fused_features)

        # Normalize before classification
        validity = self.classifier(fused_features)  # [B, 1]

        return validity


def compute_gradient_penalty(discriminator, real_data, generated_data, csi_data, input_device, gp_weight=10.0):
    """
    Compute the gradient penalty for WGAN-GP.

    Args:
        input_device:
        discriminator (nn.Module): The discriminator model.
        real_data (torch.Tensor): Batch of real images. Shape: [B, C, H, W]
        generated_data (torch.Tensor): Batch of generated images. Shape: [B, C, H, W]
        device (torch.device): Torch device (cuda or cpu).
        gp_weight (float): Gradient penalty coefficient (default = 10.0).

    Returns:
        torch.Tensor: Gradient penalty scalar value.
    """
    batch_size = real_data.size(0)

    # Random interpolation factor α sampled from U(0,1)
    alpha = torch.rand(batch_size, 1, 1, 1, device=input_device)

    # Interpolate between real and generated images
    interpolated = (alpha * real_data + (1 - alpha) * generated_data).requires_grad_(True)

    # Compute the discriminator's prediction on the interpolated samples
    prob_interpolated = discriminator(interpolated, csi_data)

    # Compute gradients of the outputs w.r.t. the interpolated inputs
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated, device=input_device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten the gradients
    gradients = gradients.view(batch_size, -1)

    # Compute L2 norm of gradients
    gradients_norm = torch.norm(gradients, p=2, dim=1)

    # Compute gradient penalty loss
    gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()

    return gradient_penalty
