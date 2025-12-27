import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type
# from src.vector_quantizer import EMAQuantizer, VectorQuantizer
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from collections.abc import Sequence
# from generative.networks.layers.vector_quantizer import EMAQuantizer, VectorQuantizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import torch.nn as nn


from typing import Tuple
from torch.distributions.normal import Normal
from torch.nn.modules.loss import _Loss

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.losses.spatial_mask import MaskedLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, deprecated_arg, look_up_option, pytorch_after
# Create an instance of MSELoss



class Normalization(nn.Module):
    def __init__(self, num_features: int, norm_type: str, num_groups: int = 8) -> None:
        super(Normalization, self).__init__()
        if norm_type == 'batch':
            self.norm = nn.BatchNorm3d(num_features)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(num_features)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, num_features)
        else:
            raise ValueError(f"Unsupported normalization: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)




class Convolution(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: int,
        kernel_size: int,
        padding: int,
        norm_type: Optional[str] = None,
        num_groups: int = 8,
        act: Optional[str] = None,
        dropout: Optional[float] = None,
        transpose: bool = False,
        with_conv: bool = False,  # Add with_conv parameter
        use_bias: bool = True  # Use bias parameter
    ) -> None:
        super(Convolution, self).__init__()
        self.with_conv = with_conv
        
        if transpose:
            # Transposed Convolution
            self.conv = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=strides,
                padding=padding,
                output_padding=0,  # Default value; adjust if needed
                bias=use_bias
            )
            # Additional convolution after transposed conv if with_conv is True
            if with_conv:
                self.additional_conv = nn.Conv3d(
                    out_channels,  # The output from transpose conv
                    out_channels // 2,  # Reduce channels with regular conv
                    kernel_size=1,  # Typically 1x1 kernel size for this layer
                    stride=1,  # Keep stride as 1 for this layer
                    padding=0,  # No padding needed for this conv layer
                    bias=use_bias
                )
                out_channels = out_channels // 2
            if with_conv and out_channels==8:
                self.additional_adj_conv = nn.Conv3d(
                    out_channels,  # The output from transpose conv
                    4,  # Reduce channels with regular conv
                    kernel_size=1,  # Typically 1x1 kernel size for this layer
                    stride=1,  # Keep stride as 1 for this layer
                    padding=0,  # No padding needed for this conv layer
                    bias=use_bias
                )
                out_channels = out_channels // 2
        else:
            # Regular Convolution
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=strides,
                padding=padding,
                bias=use_bias
            )
            # self.additional_conv = None  # No additional conv if no transpose
        
        self.norm = Normalization(out_channels, norm_type, num_groups) if norm_type else None
        self.act = self._get_activation(act) if act else None
        self.dropout = nn.Dropout3d(p=dropout) if dropout is not None else None

    def _get_activation(self, act: str) -> nn.Module:
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
        elif act == 'selu':
            return nn.SELU(inplace=True)
        elif act == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        # Apply additional convolution if needed (for transpose case)
        if self.with_conv and hasattr(self, 'additional_conv') and self.additional_conv:
            print("with_conv_processed")
            x = self.additional_conv(x)
            if self.with_conv and hasattr(self, 'additional_adj_conv') and self.additional_adj_conv:
                print("with_conv_processed and_adj")
                x=self.additional_adj_conv(x)
                 
            
        
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        if self.dropout:
            x = self.dropout(x)
        return x



# Define ResidualUnit class
class ResidualUnit(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_channels: int,
        act: Optional[str] = None,
        dropout: Optional[float] = None
    ) -> None:
        super(ResidualUnit, self).__init__()
        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_res_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            act=act,
            dropout=dropout
        )
        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=num_res_channels,
            out_channels=in_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            act=None,  # No activation after the final convolution in the residual block
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = F.relu(x)
        # x = self.conv2(x)
        return x

# Define VQVAEEncoder class
class VQVAEEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: List[int],
        num_res_channels: List[int],
        num_res_layers: int,
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        embedding_dim: int,
        dropout: float,
        act: str
    ) -> None:
        super(VQVAEEncoder, self).__init__()
        blocks = []

        for i in range(len(num_channels)):
            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else num_channels[i - 1],
                    out_channels=num_channels[i],
                    strides=downsample_parameters[i][0],  # Downsampling
                    kernel_size=downsample_parameters[i][1],
                    padding=downsample_parameters[i][2],
                    norm_type='batch',  # Using batch normalization
                    act=act,
                    dropout=dropout
                )
            )

            for _ in range(num_res_layers):
                blocks.append(
                    ResidualUnit(
                        spatial_dims=spatial_dims,
                        in_channels=num_channels[i],
                        num_res_channels=num_res_channels[i],
                        act=act,
                        dropout=dropout
                    )
                )

        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=64,  # Output channels for embedding dimension
                strides=1,
                kernel_size=3,
                padding=1,
                norm_type=None,  # No normalization after final convolution
                act=act
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
            print("Condition latent size is", x.shape)
        return x





class cond_latent_encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: List[int],
        num_res_channels: List[int],
        num_res_layers: int,
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: str,
        ddp_sync: bool
    ) -> None:
        super(cond_latent_encoder, self).__init__()


        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_res_channels = num_res_channels
        self.num_res_layers = num_res_layers
        self.dropout = dropout
        self.act = act

        
        # Define model components
        self.encoder = VQVAEEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            num_res_channels=num_res_channels,
            num_res_layers=num_res_layers,
            downsample_parameters=downsample_parameters,
            embedding_dim=64,
            dropout=dropout,
            act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encoder(x)
        print("conditional encoder shape is", z.shape)
        return z

    def encode_cond_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z


