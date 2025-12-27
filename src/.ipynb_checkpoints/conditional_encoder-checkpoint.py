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
mse_loss = nn.MSELoss()




class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.NONE,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        # print("n_pred_ch", n_pred_ch)
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        target = F.softmax(target, dim=1)
        target = torch.argmax(target, axis=1)
        target = torch.unsqueeze(target, 1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                # print("target shape is", target.shape)
                target = one_hot(target, num_classes=n_pred_ch)
                # print("target shape is", target.shape)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                # print("target shape is", target.shape)
                # print("input shape is", input.shape)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o
        # print(f"Intersection: {intersection}")
        # print(f"Denominator: {denominator}")

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        dice = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        # print(f"Dice: {dice}")

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            f = f * self.class_weight.to(f)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f



dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

class EMAQuantizer(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
    Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
    that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
    58d9a2746493717a7c9252938da7efa6006f3739.

    This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
    to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
    on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

    Args:
        spatial_dims :  number of spatial spatial_dims.
        num_embeddings: number of atomic elements in the codebook.
        embedding_dim: number of channels of the input and atomic elements.
        commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
        decay: EMA decay. Defaults to 0.99.
        epsilon: epsilon value. Defaults to 1e-5.
        embedding_init: initialization method for the codebook. Defaults to "normal".
        ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
        ddp_sync: bool = True,
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings

        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )

        self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        if embedding_init == "normal":
            # Initialization is passed since the default one is normal inside the nn.Embedding
            pass
        elif embedding_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
        self.embedding.weight.requires_grad = False

        self.commitment_cost: float = commitment_cost

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

        self.decay: float = decay
        self.epsilon: float = epsilon

        self.ddp_sync: bool = ddp_sync

        # Precalculating required permutation shapes
        self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
            range(1, self.spatial_dims + 1)
        )

    def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

        Args:
            inputs: Encoding space tensors

        Returns:
            torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
            torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
            torch.Tensor: Quantization indices of shape [B,D,H,W,1]

        """
        encoding_indices_view = list(inputs.shape)
        del encoding_indices_view[1]

        with torch.cuda.amp.autocast(enabled=False):
            inputs = inputs.float()

            # Converting to channel last format
            flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

            # Calculate Euclidean distances
            distances = (
                (flat_input**2).sum(dim=1, keepdim=True)
                + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
                - 2 * torch.mm(flat_input, self.embedding.weight.t())
            )

            # Mapping distances to indexes
            encoding_indices = torch.max(-distances, dim=1)[1]
            encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

            # Quantize and reshape
            encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
        [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
        decoder.

        Args:
            embedding_indices: Tensor in channel last format which holds indices referencing atomic
                elements from self.embedding

        Returns:
            torch.Tensor: Quantize space representation of encoding_indices in channel first format.
        """
        with torch.cuda.amp.autocast(enabled=False):
            return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

    @torch.jit.unused
    def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
        """
        TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
        example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

        Args:
            encodings_sum: The summation of one hot representation of what encoding was used for each
                position.
            dw: The multiplication of the one hot representation of what encoding was used for each
                position with the flattened input.

        Returns:
            None
        """
        if self.ddp_sync and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
        else:
            pass

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            print("EMA Training Started")
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                dw = torch.mm(encodings.t(), flat_input)

                if self.ddp_sync:
                    self.distributed_synchronization(encodings_sum, dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
                self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
        else:
            encodings_sum=torch.zeros(256)

        # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
        print("quantized ema shape is", quantized.shape)
        print("inputs ema shape is", inputs.shape)
        # Encoding Loss
        
        loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
        loss = loss*10

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data


class VectorQuantizer(torch.nn.Module):
    """
    Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
    the quantization in their own class.

    Args:
        quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
            based quantized representation. Defaults to None
    """

    def __init__(self, quantizer: torch.nn.Module = None):
        super().__init__()

        self.quantizer: torch.nn.Module = quantizer

        self.perplexity: torch.Tensor = torch.rand(1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

        # Perplexity calculations
        avg_probs = (
            torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
            .float()
            .div(encoding_indices.numel())
        )

        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, encodings_sum, embedding

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, _, encoding_indices = self.quantizer(encodings)

        return encoding_indices






class Normalization(nn.Module):
    def __init__(self, num_features: int, norm_type: str, num_groups: int = 8) -> None:
        super(Normalization, self).__init__()
        if norm_type == 'batch':
            self.norm = nn.BatchNorm3d(num_features, eps=1e-5)
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
        elif act == 'tanh':
            return nn.Tanh()
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
            norm_type='batch',
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
            norm_type='batch',
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
                # norm_type=None,  # No normalization after final convolution
                act=act,
                dropout=dropout
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_con = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            print("latent size is", x.shape)
            if (i+1)%2==0 or i==(len(self.blocks)-1):
                skip_con.append(x)
        return x, skip_con






class VQVAEDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: List[int],
        num_res_channels: List[int],
        num_res_layers: int,
        upsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: str
    ) -> None:
        super(VQVAEDecoder, self).__init__()
        blocks = []

        for i in range(len(num_channels)):
            # blocks.append(
            #     nn.Upsample(scale_factor=upsample_parameters[i][0], mode='trilinear', align_corners=True)
            blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=in_channels if i == 0 else num_channels[i - 1],
                    out_channels=num_channels[i],
                    strides=upsample_parameters[i][0],  # Upsampling
                    kernel_size=upsample_parameters[i][1],
                    padding=upsample_parameters[i][2],
                    norm_type='batch',  # Using batch normalization
                    act=act,
                    dropout=dropout,
                    transpose=True  # Use transpose convolution
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
                out_channels=1,  # Output channels for reconstruction
                strides=1,
                kernel_size=3,
                padding=1,
                # norm_type='batch',  # No normalization after final convolution
                act=act,
                dropout=dropout,
                transpose=False  # Final layer is a regular convolution
            )
        )

        self.blocks = nn.ModuleList(blocks)
        self.convs = nn.ModuleList([Convolution(
                spatial_dims=spatial_dims,
                in_channels=128,
                out_channels=64,  # Output channels for reconstruction
                strides=1,
                kernel_size=3,
                padding=1,
                norm_type='batch',  # No normalization after final convolution
                act=None,
                # dropout=dropout,
                transpose=False  # Final layer is a regular convolution
            ), Convolution(
                spatial_dims=spatial_dims,
                in_channels=64,
                out_channels=32,  # Output channels for reconstruction
                strides=1,
                kernel_size=3,
                padding=1,
                norm_type='batch',  # No normalization after final convolution
                act=None,
                # dropout=dropout,
                transpose=False  # Final layer is a regular convolution
            ), Convolution(
                spatial_dims=spatial_dims,
                in_channels=32,
                out_channels=16,  # Output channels for reconstruction
                strides=1,
                kernel_size=3,
                padding=1,
                norm_type='batch',  # No normalization after final convolution
                act=None,
                # dropout=dropout,
                transpose=False  # Final layer is a regular convolution
            ), Convolution(
                spatial_dims=spatial_dims,
                in_channels=16,
                out_channels=8,  # Output channels for reconstruction
                strides=1,
                kernel_size=3,
                padding=1,
                norm_type='batch',  # No normalization after final convolution
                act=None,
                # dropout=dropout,
                transpose=False  # Final layer is a regular convolution
            )
        ])

    def forward(self, x: torch.Tensor, skip_con: list) -> torch.Tensor:
        skip_con = list(reversed(skip_con[0:-1]))
        print("len of skp connection ", len(skip_con))
        skip_counter = 0
        for i, block in enumerate(self.blocks):
            print(len(self.blocks))
            # print("decoder size is", x.shape)
            print("iiiiiiiiiiiiiiiii", i)
            if i>=1 and i%2==0:
                print("Using skip_con[{}]:", skip_con[skip_counter].shape)  # Debugging output
        
                x = torch.cat([x, skip_con[skip_counter]], dim=1)  # Concatenate tensors
                print("before concatenation conv process:", x.shape)  # Debugging output
                
                x = self.convs[skip_counter](x)  # Apply convolution
                print("x.shape after concatenation post-process:", x.shape)  # Debugging output
                print(f"skip_con{i}", skip_con[skip_counter].shape)
                
                skip_counter += 1  # Increment the counter for the next match

                
                
            x = block(x)
            print("decoder size is", x.shape)

        return x





class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv3d(4, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(4, 4, kernel_size=3, padding=1)
        self.conv3 = Convolution(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,  # Output channels for reconstruction
                strides=1,
                kernel_size=1,
                padding=0,
                # norm_type=None,  # No normalization after final convolution
                act=None,
                transpose=False  # Final layer is a regular convolution
            )
        # self.conv4_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
        # self.conv5_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
        # self.conv6_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("X shape is", x.shape)
        # x = self.conv1(x)
        # x = F.relu(x)
        # # print("X shape is", x.shape)
        # x = self.conv2(x)
        # # print("X shape is", x.shape)
        # x = F.relu(x)
        print("X shape  before is", x.shape)
        segmentation_mask1 = self.conv3(x)
        print("X shape after is", segmentation_mask1.shape)
        
        
        # Compute softmax probabilities over classes
        # output_probabilities = F.softmax(segmentation_mask1, dim=1)
        # segmentation_mask2 = self.conv4_op(x)
        # output_probabilities2 = F.softmax(segmentation_mask2, dim=1)
        # segmentation_mask3 = self.conv5_op(x)
        # output_probabilities3 = F.softmax(segmentation_mask3, dim=1)
        # segmentation_mask4 = self.conv6_op(x)
        # output_probabilities4 = F.softmax(segmentation_mask4, dim=1)
        # output_probabilities = (output_probabilities4+output_probabilities1+output_probabilities2+output_probabilities3) / 4
        # # Compute the maximum probability and its class
        # segmentation_mask = torch.argmax(output_probabilities, dim=1)
        # segmentation_mask = [(segmentation_mask == 0), (segmentation_mask == 1) | (segmentation_mask == 3), (segmentation_mask == 1) | (segmentation_mask == 3) | (segmentation_mask == 2), (segmentation_mask == 3)]
        # result = [(segmentation_mask == 1), (segmentation_mask == 2), (segmentation_mask == 3)]
        # # # print("max_class", max_class)
        # # # # Define a confidence threshold
        # # # confidence_threshold = 0.45
        
        # # # # Create a mask for uncertain pixels
        # # # uncertain_mask = max_prob < confidence_threshold
        
        # # # # Segmentation mask for highest probability class
        # # # segmentation_mask = torch.where(uncertain_mask, torch.tensor(0), max_class)  # Assign to "unknown" class
        # # # print("s_mask", segmentation_mask.shape)
        # s_mask = segmentation_mask.detach().cpu().numpy()
        # print("s_mask", s_mask.shape)
        # print("Unique labels are:", np.unique(s_mask))
        # # Create specific masks for NCR, ED, and ET
        # # Assume segmentation_mask contains values 0 (background), 1 (NCR), 2 (ED), 3 (ET)
        
        # # First channel: Combination of NCR (1) and ET (3)
        # # first_channel = (segmentation_mask == 1) | (segmentation_mask == 3)
        
        # # # Second channel: Combination of all three (NCR, ED, ET)
        # # second_channel = (segmentation_mask == 1) | (segmentation_mask == 2) | (segmentation_mask == 3)
        
        # # # Third channel: Only ET (3)
        # # third_channel = (segmentation_mask ==3)
        
        # # Stack these masks into a new segmentation mask with 3 channels
        # segmentation_mask = torch.stack(segmentation_mask, dim=1).float()  # Shape: (batch_size, 3, H, W, D)
        # print("final segmentation", segmentation_mask.shape)
        return segmentation_mask1


# def freeze_module(module):
#     for param in module.parameters():
#         param.requires_grad = False



class cond(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: List[int],
        num_res_channels: List[int],
        num_res_layers: int,
        downsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        upsample_parameters: Sequence[Sequence[int, int, int, int], ...],
        dropout: float,
        act: str,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float,
        embedding_init: str,
        ddp_sync: bool
    ) -> None:
        super(cond, self).__init__()


        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_res_channels = num_res_channels
        self.num_res_layers = num_res_layers
        self.dropout = dropout
        self.act = act
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding_init = embedding_init

        
        # Define model components
        self.encoder = VQVAEEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            num_res_channels=num_res_channels,
            num_res_layers=num_res_layers,
            downsample_parameters=downsample_parameters,
            embedding_dim=embedding_dim,
            dropout=dropout,
            act=act
        )

        # self.quantizer1 = VectorQuantizer(
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
        # self.quantizer2 = VectorQuantizer(
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
        # self.quantizer3 = VectorQuantizer(
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
        self.quantizer0 = VectorQuantizer(
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

        self.decoder = VQVAEDecoder(
            spatial_dims=spatial_dims,
            in_channels=64,
            num_channels=num_channels[::-1],  # Reverse the channel order for decoding
            num_res_channels=num_res_channels[::-1],
            num_res_layers=num_res_layers,
            upsample_parameters=upsample_parameters,
            dropout=dropout,
            act=act
        )

        self.segmentation=SegmentationModel(4, 4, 4)

        # for block in self.encoder.blocks:
        #     if isinstance(block, ResidualUnit):
        #         freeze_module(block)

        # for block in self.decoder.blocks:
        #     if isinstance(block, ResidualUnit):
        #         freeze_module(block)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z, skip_con = self.encoder(x)
        print("encoder shape is", z.shape)
        # print("unique labels are", len(torch.unique(z)))
      
        # channel_groups = torch.split(z, 16, dim=1)
        # # print("channel_groups[4]", channel_groups[4].shape)
        # # z_5 = torch.cat((channel_groups[4], channel_groups[5]), dim=1)
        # # z_6 = torch.cat((channel_groups[6], channel_groups[7]), dim=1)
        # # # Loop through each group of 16 channels and apply separate quantizer
        # quantization_loss0, z_quantized0, encodings_sum0, embedding0 = self.quantizer0(z)
        # print("z_quantized0.shape", z_quantized0.shape)
    
        # quantization_loss1, z_quantized1, encodings_sum1, embedding1 = self.quantizer1(channel_groups[1])
        # print("z_quantized1.shape", z_quantized1.shape)
    
        # quantization_loss2, z_quantized2, encodings_sum2, embedding2 = self.quantizer2(channel_groups[2])
        # print("z_quantized2.shape", z_quantized2.shape)
    
        # quantization_loss3, z_quantized3, encodings_sum3, embedding3 = self.quantizer3(channel_groups[3])
        # print("z_quantized3.shape", z_quantized3.shape)
        # # # quantization_loss4, z_quantized4, encodings_sum4 = self.quantizer4(z_5)
        # # # print("z_quantized4.shape", z_quantized4.shape)
        # # # quantization_loss5, z_quantized5, encodings_sum5 = self.quantizer5(z_6)
        # # # print("z_quantized5.shape", z_quantized5.shape)
        # # # quantization_loss6, z_quantized6, encodings_sum6 = self.quantizer6(channel_groups[6])
        # # # print("z_quantized6.shape", z_quantized6.shape)
        # # # quantization_loss7, z_quantized7, encodings_sum7 = self.quantizer7(channel_groups[7])
        # # # print("z_quantized7.shape", z_quantized7.shape)

        
        # # # Concatenate all the quantized channel groups back into a single tensor
        # z_quantized_all = torch.cat((z_quantized0, z_quantized1, z_quantized2, z_quantized3), dim=1)

        # print("z_quantized_all", z_quantized_all.shape)
        
        # Pass the concatenated quantized tensor to the decoder
        reconstruction = self.decoder(z, skip_con)
        # # # reconstruction = (reconstruction+1)/2
        reconstruction = self.segmentation(reconstruction)
        
        # # Obtain the segmentation mask
        # # reconstruction = self.conv1_ft(reconstruction)
        # # reconstruction = self.conv2_ft(reconstruction)
        # # reconstruction = self.conv3_ft(reconstruction)
        # # segmentation_mask = self.segmentation2(reconstruction)
        
        
#         # print("segmentation_mask", segmentation_mask.shape)
        
#         # You may want to combine the quantization losses and encodings sums (e.g., by summing them)
#         total_quantization_loss = (
#     torch.mean(quantization_loss0) +
#     torch.mean(quantization_loss1) +
#     torch.mean(quantization_loss2) +
#     torch.mean(quantization_loss3)
# )
        # quantization_loss0 = torch.mean(quantization_loss0)
#         # print("total_quantization_loss", len(total_quantization_loss))
#         print("total_quantization_loss1111111111", (total_quantization_loss))
# #         # print("encodings_sums", (encodings_sums.shape))
# #         # print("len of encodings_sum0", (encodings_sum0.shape))
# #         print("total_quantization_loss", total_quantization_loss.shape)
#         total_quantization_loss = torch.mean(total_quantization_loss)
#         print("total_quantization_loss2222222222222222", (total_quantization_loss))
# #         # print("total_quantization_loss", total_quantization_loss)
#         total_encodings_sum = torch.cat((encodings_sum0, encodings_sum1, encodings_sum2, encodings_sum3), dim=0)  # Adjust if necessary
#         # total_encodings = torch.cat((embedding0, embedding1, embedding2, embedding3), dim=0)  # Adjust if necessary
#         # print("total_encodings_sum", len(total_encodings_sum))

        # return z_quantized_all, segmentation_mask, total_quantization_loss, encodings_sum0, embedding0
        
        return reconstruction

    def encode_stage_2_inputs(self, x: torch.Tensor, quantized: bool = False) -> torch.Tensor:
        z = self.encoder(x)
#         print("encoder shape is", z.shape)
#         # print("unique labels are", len(torch.unique(z)))
      
#         channel_groups = torch.split(z, 16, dim=1)
#         # print("channel_groups[4]", channel_groups[4].shape)
#         # z_5 = torch.cat((channel_groups[4], channel_groups[5]), dim=1)
#         # z_6 = torch.cat((channel_groups[6], channel_groups[7]), dim=1)
#         # # Loop through each group of 16 channels and apply separate quantizer
#         quantization_loss0, z_quantized0, encodings_sum0, embedding0 = self.quantizer0(channel_groups[0])
#         print("z_quantized0.shape", z_quantized0.shape)
    
#         quantization_loss1, z_quantized1, encodings_sum1, embedding1 = self.quantizer1(channel_groups[1])
#         print("z_quantized1.shape", z_quantized1.shape)
    
#         quantization_loss2, z_quantized2, encodings_sum2, embedding2 = self.quantizer2(channel_groups[2])
#         print("z_quantized2.shape", z_quantized2.shape)
    
#         quantization_loss3, z_quantized3, encodings_sum3, embedding3 = self.quantizer3(channel_groups[3])
#         print("z_quantized3.shape", z_quantized3.shape)

        
#         # # Concatenate all the quantized channel groups back into a single tensor
#         z_quantized_all = torch.cat((z_quantized0, z_quantized1, z_quantized2, z_quantized3), dim=1)

#         total_quantization_loss = (
#     torch.mean(quantization_loss0) +
#     torch.mean(quantization_loss1) +
#     torch.mean(quantization_loss2) +
#     torch.mean(quantization_loss3)
# )
        
#         print("total_quantization_loss1", (total_quantization_loss))
#         total_quantization_loss = torch.mean(total_quantization_loss)
#         print("total_quantization_loss2", (total_quantization_loss))

        if quantized:
            return z_quantized_all
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:

        reconstruction = self.decoder(z)
        # segmentation_mask = self.segmentation(reconstruction)
        
        # Obtain the segmentation mask
        # reconstruction = self.conv1_ft(reconstruction)
        # reconstruction = self.conv2_ft(reconstruction)
        # reconstruction = self.conv3_ft(reconstruction)
        # segmentation_mask = self.segmentation2(reconstruction)
        
        
        # print("segmentation_mask", segmentation_mask.shape)
        
        return reconstruction




# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import init
# import math 
# import group_norm
# class conv_block(nn.Module):
#     def __init__(self, ch_in, ch_out, k_size, stride=1, p=1, num_groups=1):
#         super(conv_block, self).__init__()
#         self.conv = nn.Sequential(
#             #nn.GroupNorm(num_groups=num_groups, num_channels=ch_in),
#             # group_norm.GroupNorm3d(num_features=ch_in, num_groups=num_groups),
#             # nn.ReLU(inplace=True), 
#             nn.Conv3d(ch_in, ch_out, kernel_size=k_size, stride=stride, padding=p),  
#             nn.BatchNorm3d(ch_out),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         out = self.conv(x)
#         return out


# class ResNet_block(nn.Module):
#     "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"
#     def __init__(self, ch, k_size, stride=1, p=1, num_groups=1):
#         super(ResNet_block, self).__init__()
#         self.conv = nn.Sequential(
#             #nn.GroupNorm(num_groups=num_groups, num_channels=ch),
#             # group_norm.GroupNorm3d(num_features=ch, num_groups=num_groups),
#             # nn.ReLU(inplace=True), 
#             nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p), 
#             nn.BatchNorm3d(ch),
#             nn.ReLU(inplace=True),

#             #nn.GroupNorm(num_groups=num_groups, num_channels=ch),
#             # group_norm.GroupNorm3d(num_features=ch, num_groups=num_groups),
#             # nn.ReLU(inplace=True), 
#             nn.Conv3d(ch, ch, kernel_size=k_size, stride=stride, padding=p),  
#             nn.BatchNorm3d(ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         out = self.conv(x) + x
#         return out


# class up_conv(nn.Module):
#     "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 3D trilinear upsampling"
#     def __init__(self, ch_in, ch_out, k_size=1, scale=2, align_corners=False):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Conv3d(ch_in, ch_out, kernel_size=k_size),
#             nn.Upsample(scale_factor=scale, mode='trilinear', align_corners=align_corners),
#         )
#     def forward(self, x):
#         return self.up(x)

# class Encoder(nn.Module):
#     """ Encoder module """
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = conv_block(ch_in=4, ch_out=32, k_size=3, num_groups=1)
#         self.res_block1 = ResNet_block(ch=32, k_size=3, num_groups=8)
#         self.MaxPool1 = nn.MaxPool3d(3, stride=2, padding=1)

#         self.conv2 = conv_block(ch_in=32, ch_out=64, k_size=3, num_groups=8)
#         self.res_block2 = ResNet_block(ch=64, k_size=3, num_groups=16)
#         self.MaxPool2 = nn.MaxPool3d(3, stride=2, padding=1)

#         self.conv3 = conv_block(ch_in=64, ch_out=128, k_size=3, num_groups=16)
#         self.res_block3 = ResNet_block(ch=128, k_size=3, num_groups=16)
#         self.MaxPool3 = nn.MaxPool3d(3, stride=2, padding=1)

#         self.conv4 = conv_block(ch_in=128, ch_out=256, k_size=3, num_groups=16)
#         self.res_block4 = ResNet_block(ch=256, k_size=3, num_groups=16)
#         self.MaxPool4 = nn.MaxPool3d(3, stride=2, padding=1)

#     #     self.reset_parameters()
      
#     # def reset_parameters(self):
#     #     for weight in self.parameters():
#     #         stdv = 1.0 / math.sqrt(weight.size(0))
#     #         torch.nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x1 = self.res_block1(x1)
#         x1 = self.MaxPool1(x1) # torch.Size([1, 32, 26, 31, 26])
        
#         x2 = self.conv2(x1)
#         x2 = self.res_block2(x2)
#         x2 = self.MaxPool2(x2) # torch.Size([1, 64, 8, 10, 8])

#         x3 = self.conv3(x2)
#         x3 = self.res_block3(x3)
#         x3 = self.MaxPool3(x3) # torch.Size([1, 128, 2, 3, 2])
        
#         x4 = self.conv4(x3)
#         x4 = self.res_block4(x4) # torch.Size([1, 256, 2, 3, 2])
#         x4 = self.MaxPool4(x4) # torch.Size([1, 256, 1, 1, 1])
#         print("x1 shape: ", x1.shape)
#         print("x2 shape: ", x2.shape)
#         print("x3 shape: ", x3.shape)
#         print("x4 shape: ", x4.shape) 
#         return x4

# class Decoder(nn.Module):
#     """ Decoder Module """
#     def __init__(self, latent_dim):
#         super(Decoder, self).__init__()
#         self.latent_dim = latent_dim
#         self.linear_up = nn.Linear(latent_dim, 256*150)
#         self.relu = nn.ReLU()
#         self.upsize4 = up_conv(ch_in=256, ch_out=128, k_size=1, scale=2)
#         self.res_block4 = ResNet_block(ch=128, k_size=3, num_groups=16)
#         self.upsize3 = up_conv(ch_in=128, ch_out=64, k_size=1, scale=2)
#         self.res_block3 = ResNet_block(ch=64, k_size=3, num_groups=16)        
#         self.upsize2 = up_conv(ch_in=64, ch_out=32, k_size=1, scale=2)
#         self.res_block2 = ResNet_block(ch=32, k_size=3, num_groups=16)   
#         self.upsize1 = up_conv(ch_in=32, ch_out=1, k_size=1, scale=2)
#         self.res_block1 = ResNet_block(ch=1, k_size=3, num_groups=1)   

#     #     self.reset_parameters()
      
#     # def reset_parameters(self):
#     #     for weight in self.parameters():
#     #         stdv = 1.0 / math.sqrt(weight.size(0))
#     #         torch.nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x):
#         # x4_ = self.linear_up(x)
#         # x4_ = self.relu(x4_)

#         # x4_ = x4_.view(-1, 256, 5, 6, 5)
#         # x4_ = self.upsize4(x4_) 
#         # x4_ = self.res_block4(x4_)

#         x3_ = self.upsize3(X) 
#         x3_ = self.res_block3(x3_)

#         x2_ = self.upsize2(x3_) 
#         x2_ = self.res_block2(x2_)

#         x1_ = self.upsize1(x2_) 
#         x1_ = self.res_block1(x1_)
#         # print("x4_ shape: ", x4_.shape)
#         print("x3_ shape: ", x3_.shape)
#         print("x2_ shape: ", x2_.shape)
#         print("x1_ shape: ", x1_.shape) 
#         return x1_


# class VAE(nn.Module):
#     def __init__(self, latent_dim=128):
#         super(VAE, self).__init__()
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         self.latent_dim = latent_dim
#         # self.z_mean = nn.Linear(256*150, latent_dim)
#         # self.z_log_sigma = nn.Linear(256*150, latent_dim)
#         # self.epsilon = torch.normal(size=(1, latent_dim), mean=0, std=1.0, device=self.device)
#         self.encoder = Encoder()
#         self.decoder = Decoder(latent_dim)

#     #     self.reset_parameters()
      
#     # def reset_parameters(self):
#     #     for weight in self.parameters():
#     #         stdv = 1.0 / math.sqrt(weight.size(0))
#     #         torch.nn.init.uniform_(weight, -stdv, stdv)

#     def forward(self, x):
#         x = self.encoder(x)
#         # x = torch.flatten(x, start_dim=1)
#         # z_mean = self.z_mean(x)
#         # z_log_sigma = self.z_log_sigma(x)
#         # z = z_mean + z_log_sigma.exp()*self.epsilon
#         y = self.decoder(x)
#         return y
