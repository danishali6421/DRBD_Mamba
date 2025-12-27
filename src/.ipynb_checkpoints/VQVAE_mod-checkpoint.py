# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from src.convolutions import Convolution
# from monai.networks.blocks import Convolution
from monai.networks.layers import Act
from monai.utils.misc import ensure_tuple_rep
import monai

from src.vector_quantizer import EMAQuantizer, VectorQuantizer
import numpy as np
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



__all__ = ["VQVAE"]

criterion = nn.BCEWithLogitsLoss()

class SegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(SegmentationModel, self).__init__()
        
        # Define 3D convolution layers
        self.conv1 = nn.Conv3d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(1, 1, kernel_size=1)  # Output channels should match number of classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through convolutional layers with ReLU activation
        x = self.conv1(x)
        x = F.relu(x)  # Applying ReLU activation
        x = self.conv2(x)
        x = F.relu(x)  # Applying ReLU activation
        
        # Final convolution layer (no activation here)
        x = self.conv3(x)
        probabilities = torch.sigmoid(x)  # Shape: (batch_size, num_classes, D, H, W)
        
        # Convert probabilities to binary mask using a threshold (e.g., 0.5)
        threshold = 0.5
        binary_mask = (probabilities > threshold).float()  # Shape: (batch_size, num_classes, D, H, W)
        
        return binary_mask
        
        # # Apply softmax along the channel dimension to get probabilities
        # output_probabilities = F.softmax(x, dim=1)  # Shape: (batch_size, num_classes, H, W, D)
        # print("Output probabilities:", (output_probabilities.shape[1]))

        
        # # Use argmax to convert probabilities to class labels
        # segmentation_mask = torch.argmax(output_probabilities, dim=1)  # Shape: (batch_size, H, W, D)
        # print("s_mask", segmentation_mask.shape)
        # s_mask = segmentation_mask.detach().cpu().numpy()
        # print("s_mask", s_mask.shape)
        # print("Unique labels are:", np.unique(s_mask))

        # # Create specific masks for NCR, ED, and ET
        # # Assume segmentation_mask contains values 0 (background), 1 (NCR), 2 (ED), 3 (ET)
        # # We need to map these to the required channels

        # # First channel: Combination of NCR (1) and ET (3)
        # first_channel = (segmentation_mask == 1) | (segmentation_mask == 3)  # 1 for NCR and 3 for ET

        # # Second channel: Combination of all three (NCR, ED, ET)
        # second_channel = (segmentation_mask == 1) | (segmentation_mask == 2) | (segmentation_mask == 3)  # All three

        # # Third channel: Only ET (3)
        # third_channel = (segmentation_mask == 3)  # 3 for ET only

        # # Stack these masks into a new segmentation mask with 3 channels
        # final_segmentation_mask = torch.stack([first_channel, second_channel, third_channel], dim=1).float()  # Shape: (batch_size, 3, H, W, D)

        # return x

class VQVAEResidualUnit(nn.Module):
    """
    Implementation of the ResidualLayer used in the VQVAE network as originally used in Morphology-preserving
    Autoregressive 3D Generative Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf) and
    the original implementation that can be found at
    https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L150.

    Args:
        spatial_dims: number of spatial spatial_dims of the input data.
        num_channels: number of input channels.
        num_res_channels: number of channels in the residual layers.
        act: activation type and arguments. Defaults to RELU.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_res_channels: int,
        act: tuple | str | None = Act.RELU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_res_channels = num_res_channels
        self.act = act
        self.dropout = dropout
        self.bias = bias

        self.conv1 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_channels,
            out_channels=self.num_res_channels,
            adn_ordering="DA",
            act=self.act,
            dropout=self.dropout,
            bias=self.bias,
        )

        self.conv2 = Convolution(
            spatial_dims=self.spatial_dims,
            in_channels=self.num_res_channels,
            out_channels=self.num_channels,
            bias=self.bias,
            conv_only=True,
        )

    def forward(self, x):
        return torch.nn.functional.relu(x + self.conv2(self.conv1(x)), True)


class Encoder(nn.Module):
    """
    Encoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of channels in the latent space (embedding_dim).
        num_channels: number of channels at each level.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: tuple | str | None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.downsample_parameters = downsample_parameters
        self.dropout = dropout
        self.act = act

        blocks = []

        for i in range(len(self.num_channels)):
            blocks.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=self.in_channels if i == 0 else self.num_channels[i - 1],
                    out_channels=self.num_channels[i],
                    strides=self.downsample_parameters[i][0],
                    kernel_size=self.downsample_parameters[i][1],
                    adn_ordering="DA",
                    act=self.act,
                    dropout=None if i == 0 else self.dropout,
                    dropout_dim=1,
                    dilation=self.downsample_parameters[i][2],
                    padding=self.downsample_parameters[i][3],
                )
            )

            for _ in range(self.num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        num_channels=self.num_channels[i],
                        num_res_channels=self.num_res_channels[i],
                        act=self.act,
                        dropout=self.dropout,
                    )
                )

        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.num_channels[len(self.num_channels) - 1],
                out_channels=self.out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    """
    Decoder module for VQ-VAE.

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of channels in the latent space (embedding_dim).
        out_channels: number of output channels.
        num_channels: number of channels at each level.
        num_res_layers: number of sequential residual layers at each level.
        num_res_channels: number of channels in the residual layers at each level.
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        dropout: dropout ratio.
        act: activation type and arguments.
        output_act: activation type and arguments for the output.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int],
        num_res_layers: int,
        num_res_channels: Sequence[int],
        upsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: tuple | str | None,
        output_act: tuple | str | None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels
        self.upsample_parameters = upsample_parameters
        self.dropout = dropout
        self.act = act
        self.output_act = output_act

        reversed_num_channels = list(reversed(self.num_channels))

        blocks = []
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=reversed_num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        reversed_num_res_channels = list(reversed(self.num_res_channels))
        for i in range(len(self.num_channels)):
            for _ in range(self.num_res_layers):
                blocks.append(
                    VQVAEResidualUnit(
                        spatial_dims=self.spatial_dims,
                        num_channels=reversed_num_channels[i],
                        num_res_channels=reversed_num_res_channels[i],
                        act=self.act,
                        dropout=self.dropout,
                    )
                )

            blocks.append(
                Convolution(
                    spatial_dims=self.spatial_dims,
                    in_channels=reversed_num_channels[i],
                    out_channels=self.out_channels if i == len(self.num_channels) - 1 else reversed_num_channels[i + 1],
                    strides=self.upsample_parameters[i][0],
                    kernel_size=self.upsample_parameters[i][1],
                    adn_ordering="DA",
                    act=self.act,
                    dropout=self.dropout if i != len(self.num_channels) - 1 else None,
                    norm=None,
                    dilation=self.upsample_parameters[i][2],
                    conv_only=i == len(self.num_channels) - 1,
                    is_transposed=True,
                    padding=self.upsample_parameters[i][3],
                    output_padding=self.upsample_parameters[i][4],
                )
            )

        if self.output_act:
            blocks.append(Act[self.output_act]())

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VQVAE(nn.Module):
    """
    Vector-Quantised Variational Autoencoder (VQ-VAE) used in Morphology-preserving Autoregressive 3D Generative
    Modelling of the Brain by Tudosiu et al. (https://arxiv.org/pdf/2209.03177.pdf) and the original implementation
    that can be found at https://github.com/AmigoLab/SynthAnatomy/blob/main/src/networks/vqvae/baseline.py#L163/

    Args:
        spatial_dims: number of spatial spatial_dims.
        in_channels: number of input channels.
        out_channels: number of output channels.
        downsample_parameters: A Tuple of Tuples for defining the downsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int) and padding (int).
        upsample_parameters: A Tuple of Tuples for defining the upsampling convolutions. Each Tuple should hold the
            following information stride (int), kernel_size (int), dilation (int), padding (int), output_padding (int).
        num_res_layers: number of sequential residual layers at each level.
        num_channels: number of channels at each level.
        num_res_channels: number of channels in the residual layers at each level.
        num_embeddings: VectorQuantization number of atomic elements in the codebook.
        embedding_dim: VectorQuantization number of channels of the input and atomic elements.
        commitment_cost: VectorQuantization commitment_cost.
        decay: VectorQuantization decay.
        epsilon: VectorQuantization epsilon.
        act: activation type and arguments.
        dropout: dropout ratio.
        output_act: activation type and arguments for the output.
        ddp_sync: whether to synchronize the codebook across processes.
        use_checkpointing if True, use activation checkpointing to save memory.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_channels: Sequence[int] | int = (96, 96, 192),
        num_res_layers: int = 3,
        num_res_channels: Sequence[int] | int = (96, 96, 192),
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...]
        | Sequence[int, int, int, int] = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters: Sequence[Sequence[int, int, int, int, int], ...]
        | Sequence[int, int, int, int] = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings: int = 32,
        embedding_dim: int = 64,
        embedding_init: str = "normal",
        # mode: str="unified"
        commitment_cost: float = 0.25,
        decay: float = 0.5,
        epsilon: float = 1e-5,
        dropout: float = 0.0,
        act: tuple | str | None = Act.RELU,
        output_act: tuple | str | None = None,
        ddp_sync: bool = True,
        use_checkpointing: bool = False,

    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_checkpointing = use_checkpointing
        # self.mode=mode

        if isinstance(num_res_channels, int):
            num_res_channels = ensure_tuple_rep(num_res_channels, len(num_channels))

        if len(num_res_channels) != len(num_channels):
            raise ValueError(
                "`num_res_channels` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if not all(isinstance(values, (int, Sequence)) for values in downsample_parameters):
            raise ValueError("`downsample_parameters` should be a single tuple of integer or a tuple of tuples.")

        if not all(isinstance(values, (int, Sequence)) for values in upsample_parameters):
            raise ValueError("`upsample_parameters` should be a single tuple of integer or a tuple of tuples.")

        if all(isinstance(values, int) for values in upsample_parameters):
            upsample_parameters = (upsample_parameters,) * len(num_channels)

        if all(isinstance(values, int) for values in downsample_parameters):
            downsample_parameters = (downsample_parameters,) * len(num_channels)

        for parameter in downsample_parameters:
            if len(parameter) != 4:
                raise ValueError("`downsample_parameters` should be a tuple of tuples with 4 integers.")

        for parameter in upsample_parameters:
            if len(parameter) != 5:
                raise ValueError("`upsample_parameters` should be a tuple of tuples with 5 integers.")

        if len(downsample_parameters) != len(num_channels):
            raise ValueError(
                "`downsample_parameters` should be a tuple of tuples with the same length as `num_channels`."
            )

        if len(upsample_parameters) != len(num_channels):
            raise ValueError(
                "`upsample_parameters` should be a tuple of tuples with the same length as `num_channels`."
            )

        self.num_res_layers = num_res_layers
        self.num_res_channels = num_res_channels

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=embedding_dim,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            downsample_parameters=downsample_parameters,
            dropout=dropout,
            act=act,
        )

        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            in_channels=embedding_dim,
            out_channels=out_channels,
            num_channels=num_channels,
            num_res_layers=num_res_layers,
            num_res_channels=num_res_channels,
            upsample_parameters=upsample_parameters,
            dropout=dropout,
            act=act,
            output_act=output_act,
        )

        # self.quantizer = VectorQuantizer(
        #     quantizer=EMAQuantizer(
        #         spatial_dims=spatial_dims,
        #         num_embeddings=num_embeddings,
        #         embedding_dim=embedding_dim,
        #         commitment_cost=commitment_cost,
        #         decay=decay,
        #         epsilon=epsilon,
        #         embedding_init=embedding_init,
        #         ddp_sync=ddp_sync,
        #     )
        # )
        self.segmentation=SegmentationModel(out_channels, out_channels, out_channels)
        self.quantizer_unified = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=spatial_dims,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
                embedding_init=embedding_init,
                ddp_sync=ddp_sync,
            )
        )
        
        # Miss quantizer
        self.quantizer_miss = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=spatial_dims,
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
                embedding_init=embedding_init,
                ddp_sync=ddp_sync,
            )
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.encoder, images, use_reentrant=False)
        else:
            return self.encoder(images)

    # def quantize(self, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     x_loss, x = self.quantizer(encodings)
    #     return x, x_loss
    def quantize(self, encodings: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == "unified":
            x_loss, x = self.quantizer_unified(encodings)
        elif mode == "miss":  # "miss"
            x_loss, x = self.quantizer_miss(encodings)
        return x, x_loss

    def decode(self, quantizations: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.decoder, quantizations, use_reentrant=False)
        else:
            recons_img=self.decoder(quantizations)
            print("type of recons_img", (recons_img.shape))
            # mask_gen=SegmentationModel(recons_img.shape[1], recons_img.shape[1], recons_img.shape[1])
            # fin_seg=self.segmentation(recons_img)
            # print("fin_seg", type(fin_seg))
            
            
            return recons_img

    def index_quantize(self, images: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(self.encode(images=images))

    def decode_samples(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.decode(self.quantizer.embed(embedding_indices))

    # def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     quantizations, quantization_losses = self.quantize(self.encode(images))
    #     # print("quantization_losses", quantization_losses)
    #     reconstruction = self.decode(quantizations)

    #     return reconstruction, quantization_losses

    def forward(self, images: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        if mode=="unified":
            print("Full Modality Training Started")
            quantizations, quantization_losses = self.quantize(self.encode(images), mode)
            print("Quantization shape is", quantizations.shape)
            reconstruction = self.decode(quantizations)
            self.unified_quantization_losses = quantization_losses
            #return reconstruction, quantization_losses+torch.tensor(0.0)
        elif mode == "miss":
            print("Missing Modality Training Started")
            quantizations_miss, quantization_losses_miss = self.quantize(self.encode(images), mode)
            print("Quantization shape is", quantizations_miss.shape)
            reconstruction_miss = self.decode(quantizations_miss)
            if self.unified_quantization_losses is not None:
                print("Full Modality Q loss is", self.unified_quantization_losses)
                print("Miss Modality Q loss is", quantization_losses_miss)
                loss_difference = torch.abs(self.unified_quantization_losses - quantization_losses_miss)
            else:
                loss_difference = torch.tensor(0.0)
             #return reconstruction_miss, quantization_losses_miss+loss_difference
        
        return (reconstruction, quantization_losses+torch.tensor(0.0)) if mode == "unified" else (reconstruction_miss, quantization_losses_miss+loss_difference)


        # segmentation_mask = torch.argmax(reconstruction, dim=1)  # Shape: [128, 128, 128]
    
        # # One-hot encode the segmentation mask to convert it into 3 channels
        # segmentation_mask_one_hot = torch.nn.functional.one_hot(segmentation_mask, num_classes=3)  # Shape: [128, 128, 128, 3]
        
        # # Permute to get the shape [3, 128, 128, 128]
        # segmentation_mask_one_hot = segmentation_mask_one_hot.permute(0, 4, 1, 2, 3).float()
        

       

    # def encode_stage_2_inputs(self, x: torch.Tensor, quantized: bool = True) -> torch.Tensor:
    #     z = self.encode(x)
    #     e, _ = self.quantize(z)
    #     if quantized:
    #         return e
    #     return z

    def encode_stage_2_inputs(self, x: torch.Tensor, md: str, quantized: bool = True) -> torch.Tensor:
              
        if md == "unified":
            z1 = self.encode(x)
            e1, _ = self.quantize(z1, md)
        elif md == "miss":
            z2 = self.encode(x)
            e2, quantization_losses = self.quantize(z2, md)
            print("miss mod latent shape is", e2.shape)
            print("quantization_losses", quantization_losses)
             
        if quantized and md == "unified":
            return e1
        elif quantized and md == "miss":
            return e2
        return (z1, md), (z2, md)
    

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        e, _ = self.quantize(z)
        image = self.decode(e)
        return image




#Create the model and apply the initialization
# def initialize_weights(m):
#         if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#             print("Weights Initialized")
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.constant_(m.bias, 0)
# model = VQVAE(
#     spatial_dims=3,
#     in_channels=3,
#     out_channels=3,
#     num_channels=(256, 256),
#     num_res_channels=256,
#     num_res_layers=2,
#     downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
#     upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
#     num_embeddings=256,
#     embedding_dim=32,
# )
# model.apply(initialize_weights)
# print("length of model weights", (model))
