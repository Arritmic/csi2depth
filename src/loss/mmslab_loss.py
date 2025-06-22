# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2024, Constantino Álvarez  (CMVS - University of Oulu)
# All rights reserved.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# -----------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import torchvision.models as models

# Load VGG16 for perceptual loss
from torchvision.models import vgg16, VGG16_Weights, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor

torch.cuda.empty_cache()
device = torch.device("cpu")
vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
features = create_feature_extractor(vgg, {'features.3': 'relu1_2', 'features.8': 'relu2_2'}).to(device)
features.eval()


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

def chamfer_distance(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chamfer Distance between predicted and ground truth point clouds.

    Args:
        pred_points (torch.Tensor): Predicted points of shape [batch_size, num_points, 3].
        gt_points (torch.Tensor): Ground truth points of shape [batch_size, num_points, 3].

    Returns:
        torch.Tensor: Chamfer distance between the predicted and ground truth point clouds.
    """
    # Compute pairwise distances between predicted and ground truth points
    dist1 = torch.cdist(pred_points, gt_points)
    dist2 = torch.cdist(gt_points, pred_points)

    # Find nearest neighbors and compute the loss
    loss1 = dist1.min(dim=2)[0].mean(dim=1)
    loss2 = dist2.min(dim=2)[0].mean(dim=1)
    loss = loss1 + loss2

    return loss.mean()


def chamfer_loss_function(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chamfer Loss between predicted and target point clouds.

    Args:
        pred (torch.Tensor): Predicted points of shape [batch_size, num_points, 3].
        target (torch.Tensor): Target points of shape [batch_size, num_points, 3].

    Returns:
        torch.Tensor: Chamfer loss between the predicted and target point clouds.
    """
    # Compute pairwise distances between predicted and target points
    dist1 = torch.cdist(pred, target)

    # Compute the minimum distances for each point
    min_dist1, _ = torch.min(dist1, dim=2)  # Min distance to target for each predicted point
    min_dist2, _ = torch.min(dist1, dim=1)  # Min distance to prediction for each ground truth point

    # Return the average Chamfer loss over the batch
    return (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)).mean()


def feature_transform_regularizer(trans):
    """
    Regularization loss for the feature transformation matrix.

    Args:
        trans (torch.Tensor): Transformation matrix, [B, K, K]

    Returns:
        torch.Tensor: Regularization loss
    """
    K = trans.size(1)
    batchsize = trans.size(0)
    device = trans.device
    # Normalize the transformation matrix along the last dimension (K)
    trans_normalized = F.normalize(trans, p=2, dim=2)  # L2 normalization

    I = torch.eye(K, device=device).unsqueeze(0).repeat(batchsize, 1, 1)  # [B, K, K]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

    # K = trans.size(1)
    # batchsize = trans.size(0)
    # device = trans.device
    # # Normalize the transformation matrix along the last dimension (K)
    # trans_normalized = F.normalize(trans, p=2, dim=2)  # L2 normalization
    #
    # I = torch.eye(K, device=device).unsqueeze(0).repeat(batchsize, 1, 1)  # [B, K, K]
    # loss = torch.mean(torch.norm(torch.bmm(trans_normalized, trans_normalized.transpose(2, 1)) - I, dim=(1, 2))) * 0.001
    # return loss

    K = trans.size(1)
    batchsize = trans.size(0)
    device = trans.device

    I = torch.eye(K, device=device).unsqueeze(0).repeat(batchsize, 1, 1)  # Identity matrix [B, K, K]

    trans_norm = trans / (torch.norm(trans, dim=(1, 2), keepdim=True) + 1e-6)  # Normalize by Frobenius norm

    loss = torch.mean(torch.norm(torch.bmm(trans_norm, trans_norm.transpose(2, 1)) - I, dim=(1, 2))) * 1e-2
    return loss


class CSI2PointCloudLoss(nn.Module):
    def __init__(self, chamfer_weight=1.0, regularization_weight=0.1):
        super(CSI2PointCloudLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.regularization_weight = regularization_weight

    def forward(self, predicted_points, ground_truth_points, trans_feat=None):
        # Chamfer Distance Loss
        # chamfer_loss = chamfer_loss_function(predicted_points, ground_truth_points)
        chamfer_loss = chamfer_distance(predicted_points, ground_truth_points)

        if trans_feat is not None:

            # Regularization Loss
            # print(trans_feat)
            # print(torch.max(trans_feat))
            # print(torch.min(trans_feat))
            # input("stop1")
            regularization_loss = feature_transform_regularizer(trans_feat)
            # Adaptive weighting based on ratio
            scale_factor = chamfer_loss.detach() / (regularization_loss.detach() + 1e-6)
            adapted_reg_weight = self.regularization_weight * scale_factor
            print("-----***----")
            print(regularization_loss)
            print(adapted_reg_weight)
            print(chamfer_loss)
            print(adapted_reg_weight * regularization_loss)
            print("----------")
            # input("stop2")
            # Combine Chamfer loss with regularization loss
            total_loss = self.chamfer_weight * chamfer_loss + adapted_reg_weight * regularization_loss
        else:
            total_loss = chamfer_loss

        return total_loss
