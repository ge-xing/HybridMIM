# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.nn import functional as F

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrain_models.utils import patchify

def forward_constrast_loss(x_i, x_j, temp=0.5):
    device = x_i.device
    batch_size = x_i.shape[0]
    temp = torch.tensor(temp).to(device)
    neg_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float()
    z_i = F.normalize(x_i, dim=1)
    z_j = F.normalize(x_j, dim=1)
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1),
                              z.unsqueeze(0),
                              dim=2)
    sim_ij = torch.diag(sim, batch_size)
    sim_ji = torch.diag(sim, -batch_size)
    pos = torch.cat([sim_ij, sim_ji], dim=0)
    nom = torch.exp(pos / temp)
    denom = neg_mask * torch.exp(sim / temp)
    return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * batch_size)

def forward_loss_reconstruct_mask(pred, labels, mask_image, mask_value=0.0):
    # pred (b c d w h)
    # pred = torch.einsum("")
    mask = (mask_image == mask_value).float()

    loss = (pred - labels) ** 2

    # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

def forward_loss_reconstruct(pred, labels):
    loss = (pred - labels) ** 2
    loss = loss.mean()  # [N, L], mean loss per patch
    return loss


def forward_loss_similarity(pred_1, pred_2):
    loss_fct = nn.CrossEntropyLoss()
    device = pred_1.device
    cos = nn.CosineSimilarity(dim=-1)
    sim = cos(pred_1.unsqueeze(dim=1), pred_2.unsqueeze(dim=0))
    labels = torch.arange(sim.shape[0], dtype=torch.long).to(device)

    loss = loss_fct(sim, labels)

    return loss

def forward_loss_mask_region(pred_bottom_feature, mask_labels):

    # pred_bottom_feature = einops.rearrange(pred_bottom_feature, "b d w h c->b (d w h) c")
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(pred_bottom_feature.reshape(-1, pred_bottom_feature.shape[-1]), mask_labels.reshape(-1))
    return loss

def forward_loss_mask(pred_bottom_feature, mask_labels):
    mask_labels = mask_labels.long()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(pred_bottom_feature.reshape(-1, pred_bottom_feature.shape[-1]), mask_labels.reshape(-1))
    return loss

def forward_loss_mask_region_patch(pred_bottom_feature_patch, mask_labels_patch):

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(pred_bottom_feature_patch.reshape(-1, pred_bottom_feature_patch.shape[-1]), mask_labels_patch.reshape(-1))
    return loss


def forward_loss_mask_region_multi_label(pred_bottom_feature, mask_labels):
    loss_fct = nn.BCEWithLogitsLoss()
    mask_labels = mask_labels.float()
    loss = loss_fct(pred_bottom_feature, mask_labels)
    return loss


def forward_loss_mask_position(pred_bottom_feature, mask_labels):
    mask_labels = mask_labels.float()
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(pred_bottom_feature, mask_labels)
    return loss


class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss().cuda()
        self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size).cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)
