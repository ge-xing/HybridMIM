# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from einops import rearrange

from multiprocessing import pool
from typing import Sequence, Union
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from .utils import mask_func
from .utils import get_mask_labels, get_mask_labelsv2
import copy

class TwoConv(nn.Sequential):
    """two convolutions."""
    def __init__(
            self,
            dim: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        conv_1 = Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)

class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
            self,
            dim: int,
            in_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            dropout: Union[float, tuple] = 0.0,
            pool_size=(2, 2, 2)
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()

        max_pooling = Pool["MAX", dim](kernel_size=pool_size)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            dim: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: Union[str, tuple],
            norm: Union[str, tuple],
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            halves: bool = True,
            pool_size = (2, 2, 2)
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, pool_size, mode=upsample)
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

        x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x

class DeepUNet(nn.Module):

    def cons_stages(self, pools, region):
        stage = [(copy.deepcopy(region[0]), copy.deepcopy(region[1]))]
        for pool in reversed(pools):
            for i, r in enumerate(region):
                region[i][0] = region[i][0] * pool[0]
                region[i][1] = region[i][1] * pool[1]
                region[i][2] = region[i][2] * pool[2]
            stage.append((copy.deepcopy(region[0]), copy.deepcopy(region[1])))

        return stage

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 2,
            features: Sequence[int] = (32, 32, 64, 128, 256),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
            pool_size = ((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
            select_reconstruct_region=[[0, 0, 0], [8, 8, 8]], # 重构范围,
            first_level_region = (64, 64, 64),
            two_level_region = (32, 32, 32),
            pretrain=False,
    ):
        super().__init__()
        deepth = len(pool_size)
        self.deepth = deepth
        self.in_channels = in_channels
        fea = features
        print(f"BasicUNet features: {fea}.")
        self.select_reconstruct_region = select_reconstruct_region
        self.stages = self.cons_stages(pool_size, select_reconstruct_region)
        print(f"self.stages is {self.stages}")
        self.pool_size_all = self.get_pool_size_all(pool_size)
        self.window_size = torch.tensor(first_level_region) // torch.tensor(self.pool_size_all)
        print(f"window size is {self.window_size}")
        self.pretrain = pretrain
        ## get patches of region
        self.drop = nn.Dropout()
        self.conv_0 = TwoConv(3, in_channels, features[0], act, norm, dropout)

        self.downs = nn.ModuleList([])

        for d in range(deepth):
            self.downs.append(Down(3, fea[d], fea[d+1], act=act, norm=norm, pool_size=pool_size[d]))

        self.ups = nn.ModuleList([])
        for d in range(deepth):
            self.ups.append(UpCat(3, fea[deepth-d], fea[deepth-d-1], fea[deepth-d-1], act, norm, dropout, pool_size=pool_size[deepth-d-1], upsample=upsample))

        self.decoder_pred = nn.Conv3d(fea[0], out_channels, 1, 1)

        if pretrain:
            bottom_feature = features[-1]
            self.pred_mask_region = nn.Linear(bottom_feature, 65)# 一个region 4个 patch
            self.contrast_learning_head = nn.Linear(bottom_feature, 384)
            self.pred_mask_region_position = nn.Linear(bottom_feature, 64)

    def get_pool_size_all(self, pool_size):
        p_all = [1, 1, 1]
        for p in pool_size:
            p_all[0] = p_all[0] * p[0]
            p_all[1] = p_all[1] * p[1]
            p_all[2] = p_all[2] * p[2]
        return p_all 

    def wrap_feature_selection(self, feature, region_box):
        # feature: b, c, d, w, h
        return feature[..., region_box[0][0]:region_box[1][0], region_box[0][1]:region_box[1][1], region_box[0][2]:region_box[1][2]]

    def get_local_images(self, images):
        images = self.wrap_feature_selection(images, region_box=self.stages[-1])
        return images

    def forward_encoder(self, x):
        x = self.conv_0(x)
        x_downs = [x]
        for d in range(self.deepth):
            x = self.downs[d](x)
            x_downs.append(x)
        return x_downs

    def forward_decoder(self, x_downs):
        x = self.wrap_feature_selection(x_downs[-1], self.stages[0])

        for d in range(self.deepth):
            x = self.ups[d](x, self.wrap_feature_selection(x_downs[self.deepth-d-1], self.stages[d+1]))
        logits = self.decoder_pred(x)
        return logits

    def forward(self, x):
        device = x.device
        images = x.detach()
        local_images = self.get_local_images(images)
        if self.pretrain:
            # mask_ratio = torch.clamp(torch.rand(1), 0.4, 0.75)
            mask_ratio = 0.4
            x, mask = mask_func(x, self.in_channels, mask_ratio, (32, 32, 32), (4, 4, 4), mask_value=0.0)
            region_mask_labels = get_mask_labels(x.shape[0], 2*2*2, mask, 4*4*4, device)
            region_mask_position = get_mask_labelsv2(x.shape[0], 2*2*2, mask, 4*4*4, device=device)

            x_mask = self.wrap_feature_selection(x, region_box=self.stages[-1])

        hidden_states_out = self.forward_encoder(x)
        logits = self.forward_decoder(hidden_states_out)  

        if self.pretrain:
            # print(hidden_states_out.shape)
            classifier_hidden_states = rearrange(hidden_states_out[-1], "b c (d m) (w n) (h l) -> b c d w h (m n l)", m=self.window_size[0], n=self.window_size[1], l=self.window_size[2])
            classifier_hidden_states = classifier_hidden_states.mean(dim=-1)
            with torch.no_grad():
                hidden_states_out_2 = self.forward_encoder(x)
            encode_feature = hidden_states_out[-1]
            encode_feature_2 = hidden_states_out_2[-1]

            x4_reshape = encode_feature.flatten(start_dim=2, end_dim=4)
            x4_reshape = x4_reshape.transpose(1, 2)

            x4_reshape_2 = encode_feature_2.flatten(start_dim=2, end_dim=4)
            x4_reshape_2 = x4_reshape_2.transpose(1, 2)

            contrast_pred = self.contrast_learning_head(x4_reshape.mean(dim=1))
            contrast_pred_2 = self.contrast_learning_head(x4_reshape_2.mean(dim=1))

            pred_mask_feature = classifier_hidden_states.flatten(start_dim=2, end_dim=4)
            pred_mask_feature = pred_mask_feature.transpose(1, 2)
            mask_region_pred = self.pred_mask_region(pred_mask_feature)

            pred_mask_feature_position = classifier_hidden_states.flatten(start_dim=2, end_dim=4)
            pred_mask_feature_position = pred_mask_feature_position.transpose(1, 2)
            mask_region_position_pred = self.pred_mask_region_position(pred_mask_feature_position)

            return {
                "logits": logits,
                'images': local_images,
                "pred_mask_region": mask_region_pred,
                "pred_mask_region_position": mask_region_position_pred,
                "mask_position_lables": region_mask_position,
                "mask": mask,
                "x_mask": x_mask,
                "mask_labels": region_mask_labels,
                "contrast_pred_1": contrast_pred,
                "contrast_pred_2": contrast_pred_2,
            }
        else :
            return logits

