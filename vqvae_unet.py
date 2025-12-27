import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from src.mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from src.mamba.mamba_ssm.models.config_mamba import MambaConfig
from config_mamba_l1 import MambaConfigl1
from config_mamba_l2 import MambaConfigl2
from config_mamba_l3 import MambaConfigl3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange







class ConvBlock(nn.Module):
    """A Conv Block with LayerNorm."""
    def __init__(self, embed_dim, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=embed_dim, out_channels=int(4 * embed_dim), kernel_size=3, stride=1, padding=1)
        self.act = GELU2()
        self.conv2 = nn.Conv3d(in_channels=int(4 * embed_dim), out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # b, c, d, h, w = x.shape
        x = rearrange(x, 'b (h w d) c -> b c h w d', h=15, w=15, d=9)
        x = (self.conv2(self.act(self.conv1(x))))
        # x = self.conv4(self.act(self.conv3(x)))
        x = rearrange(x, 'b c h w d -> b (h w d) c')
        return x









#original conv_3d
class ConvBlock3D(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x






mse_loss = nn.MSELoss()

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
        pretrained_embedding: Optional[torch.Tensor] = None,  # Add this parameter
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings
    
        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )
    
        # Initialize embedding
        self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        
        # Load pretrained embedding if provided
        if pretrained_embedding is not None:
            if pretrained_embedding.shape != (self.num_embeddings, self.embedding_dim):
                raise ValueError(
                    f"Pretrained embedding must have shape ({self.num_embeddings}, {self.embedding_dim}), "
                    f"but got {pretrained_embedding.shape}."
                )
            self.embedding.weight.data.copy_(pretrained_embedding)
        elif embedding_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
        # Otherwise, use default initialization (normal)
    
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
            # print("encoding_indices shape issssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss", encoding_indices)
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
            # print("EMA Training Started")
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

       
        
        loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
        loss = loss

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
        # print("self.perplexity", self.perplexity)
        # loss += 0.5 * self.perplexity

        return loss, quantized, encodings_sum, embedding

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(encodings)

        return encoding_indices


#original
class ConvBlock3D_wo_p(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D_wo_p, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x





class ConvBlock3D_won_p(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D_won_p, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
       
        return x


class Encoder3D(nn.Module):
    """Encoder consisting of multiple convolution blocks with increasing feature maps"""
    def __init__(self, in_channels, dropout_prob=0.5):
        super(Encoder3D, self).__init__()
        # self.transmodel = TransformerModeldec(input_shape=[96, 96, 96], embed_dim=16, num_layers=5, num_heads=2)
        self.encoder1 = ConvBlock3D(in_channels, 16, dropout_prob)
        self.encoder2 = ConvBlock3D(16, 32, dropout_prob)
        self.encoder3 = ConvBlock3D(32, 64, dropout_prob)
        # self.encoder3_cat = ConvBlock3D(160, 32, dropout_prob)
        self.encoder4 = ConvBlock3D(64, 128, dropout_prob)
        self.encoder5 = ConvBlock3D(128, 256, dropout_prob)
        # self.res5 = ResidualBlock(128)
        self.encoder6 = ConvBlock3D(256, 512, dropout_prob)
      
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        
        x1 = self.encoder1(x)
        # x1 = self.transmodel(x1)
        # print(f"Encoder1 output shape: {x1.shape}")
        # x1 = self.res1(x1)
        x2 = self.encoder2(self.pool(x1))
       
        x3 = self.encoder3(self.pool(x2))
        
        x4 = self.encoder4(self.pool(x3))
       
        x5 = self.encoder5(self.pool(x4))
        # print(f"Encoder5 output shape: {x5.shape}")
        x6 = self.encoder6((x5))
     
        return x1, x2, x3, x4, x5, x6
        
class BottleneckBlock(nn.Module):
    """Bottleneck block with 128 to 128 features"""
    def __init__(self, in_channels, dropout_prob=0.5):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = ConvBlock3D_won_p(in_channels, in_channels, dropout_prob)
        
    def forward(self, x):
        x = self.bottleneck(x)
        print(f"Bottleneck output shape: {x.shape}")
        return x



class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class SEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Output shape: (batch, channels, 1, 1, 1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_squeezed = self.global_avg_pool(x)  # Global pooling
        x_fc = torch.relu(self.fc1(x_squeezed))  # Fully connected layer 1
        x_fc = self.sigmoid(self.fc2(x_fc))  # Fully connected layer 2 with sigmoid
        return x * x_fc  # Scale input by channel attention weights

class SpatialAttention3D(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=1)  # Single channel output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_pool = torch.mean(x, dim=1, keepdim=True)  # Pool across channels
        spatial_attention = self.sigmoid(self.conv1(x))  # Spatial attention map
        return x * spatial_attention  # Scale input by spatial attention map


class CombinedAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CombinedAttention3D, self).__init__()
        self.channel_attention = SEBlock3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)  # Channel-wise attention
        x = self.spatial_attention(x)  # Spatial attention
        return x

class Decoder3D(nn.Module):
    """Decoder with skip connections and upsampling"""
    def __init__(self, dropout_prob=0.5):
        super(Decoder3D, self).__init__()
        self.upsample_0 = self.upsample_block1(512, dropout_prob)
        self.conv_0 = ConvBlock3D_wo_p(384, 128, dropout_prob)
        self.upsample_1 = self.upsample_block(512, dropout_prob)
        self.upsample_2 = self.upsample_block2(128, dropout_prob)
        # self.conv_1 = nn.Conv3d(96, 128, kernel_size=3, padding=1)
        self.conv_1 = ConvBlock3D_wo_p(128, 64, dropout_prob)
        self.upsample_3 = self.upsample_block2(64, dropout_prob)
        self.conv_2 = ConvBlock3D_wo_p(64, 32, dropout_prob)
        self.upsample_4 = self.upsample_block2(32, dropout_prob)
        self.conv_3 = ConvBlock3D_wo_p(32, 16, dropout_prob)
        # self.conv_4 = nn.Conv3d(8, 4, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv3d(16, 4, kernel_size=1)
        # LDA Top-20 feature indices from analysis
        self.topk_indices = torch.tensor([12, 14, 0, 3, 25, 5, 6, 23, 19, 15])

      
    
    def upsample_block(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
          
            nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)
    def upsample_block1(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            #i will change this after this test experminet just un comment
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # ConvBlock3D_wo_p(in_channels, in_channels, dropout_prob)
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)

    def upsample_block2(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [            
            nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)

    def forward(self, x, x4, x3, x2, x1):
        # print(f"Decoder input (x): {x.shape}")
        x_6 = self.upsample_0(x)
        
        x_6 = self.upsample_1(x_6)
        
        x_6 = self.conv_0(torch.cat((x_6, x4), dim=1))  # Concatenate with encoder3
        # print("conv 0 shape is", x_6.shape)
        x_6 = self.upsample_2(x_6)
        # print("uosample 2 sahpe is", x_6.shape)
        x_6 = self.conv_1(torch.cat((x_6, x3), dim=1))  # Concatenate with encoder3
        # print("conv 1 shape is", x_6.shape)
        x_6 = self.upsample_3(x_6)
        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # x_6 = F.pad(x_6, padding, mode='constant', value=0)
        x_6 = self.conv_2(torch.cat((x_6, x2), dim=1)) 
        x_6 = self.upsample_4(x_6)
        # x_6 = F.pad(x_6, padding, mode='constant', value=0)
        feature = x1   #torch.cat((x_6, x1), dim=1)


        x_6 = torch.cat((x_6, x1), dim=1)
       

        x_6 = self.conv_3(x_6)



        x_6 = self.conv_5(x_6)
        # x_6 = self.conv_5(x_6)
        x_6 = F.softmax(x_6, dim=1)



      
        return x_6, feature



class LatentSpaceMaskReducer(nn.Module):
    def __init__(self, input_channels, reduction_factor=1):
        super(LatentSpaceMaskReducer, self).__init__()
        self.reduction_factor = reduction_factor
        # 1x1 convolution to reduce channels to 1
        self.channel_reduction = nn.Conv3d(input_channels, 1, kernel_size=1)
        # A small network to learn the mask
        self.mask_network = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output is a mask between 0 and 1
        )

    def generate_mask(self, reduced_tensor):
        # Generate a mask based on the reduced tensor
        mask = self.mask_network(reduced_tensor)
        return mask

    def reduce_latent(self, latent_tensor):
        B, C, H, W, D = latent_tensor.shape
        
        # Ensure tensor is on the correct device
        
        # Reduce channels from C to 1 using a 1x1 convolution
        reduced_tensor = self.channel_reduction(latent_tensor)  # Shape: (B, 1, D, H, W)

        # Generate a mask that identifies the background and foreground
        mask = self.generate_mask(reduced_tensor)  # Shape: (B, 1, D, H, W)

        # Mask the original latent tensor
        masked_tensor = latent_tensor * mask  # Element-wise multiplication

        return masked_tensor





class EdgeRefinement3D(torch.nn.Module):
    def __init__(self):
        super(EdgeRefinement3D, self).__init__()
        self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

    def gaussian_filter_3d(self, input_tensor, kernel_size=3, sigma=1.0):
        """
        Apply a Gaussian filter in 3D for smoothing.
        """
        # Create a 3D Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma, input_tensor.device)

        # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
        if input_tensor.ndim != 5:
            raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

        # Apply the Gaussian filter using convolution
        gaussian_output = F.conv3d(input_tensor, kernel, stride=1, padding=kernel_size//2)
        return gaussian_output

    def create_gaussian_kernel(self, kernel_size, sigma, device):
        """
        Create a 3D Gaussian kernel.
        """
        # Create a 1D Gaussian kernel
        kernel_1d = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, device=device)
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)

        # Normalize the kernel
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create a 3D Gaussian kernel by taking the outer product of the 1D kernels
        kernel_3d = kernel_1d.view(1, 1, kernel_size, 1, 1) * kernel_1d.view(1, 1, 1, kernel_size, 1) * kernel_1d.view(1, 1, 1, 1, kernel_size)
        kernel_3d = kernel_3d.expand(1, 1, kernel_size, kernel_size, kernel_size)  # Expand to 3D kernel

        # Ensure the kernel has the correct shape for convolution
        return kernel_3d

    def sobel_3d(self, input_tensor):
        """
        Apply a Sobel filter in 3D to compute gradients in the x, y, and z directions.
        """
        # Sobel kernel for the x direction
        sobel_x = torch.tensor(
            [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
               [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Sobel kernel for the y direction
        sobel_y = torch.tensor(
            [[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
               [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Sobel kernel for the z direction
        sobel_z = torch.tensor(
            [[[[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
        if input_tensor.ndim != 5:
            raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

        # Apply Sobel filters in each direction
        grad_x = F.conv3d(input_tensor, sobel_x, stride=1, padding=1)
        grad_y = F.conv3d(input_tensor, sobel_y, stride=1, padding=1)
        grad_z = F.conv3d(input_tensor, sobel_z, stride=1, padding=1)

        # Combine gradients
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        return gradient_magnitude

    def forward(self, input_tensor):
        """Forward pass for Edge Refinement."""
        # Apply the Gaussian filter to the input tensor first
        x = self.conv(input_tensor)
        smoothed_input = self.gaussian_filter_3d(x)

        # Apply Sobel filter to the smoothed input
        edge_map = self.sobel_3d(smoothed_input)

        # Refine the input tensor using the edge map
        input_tensor_up = edge_map * input_tensor
        return input_tensor_up




class EdgeRefinement3D_lap(torch.nn.Module):
    def __init__(self):
        super(EdgeRefinement3D_lap, self).__init__()
        self.conv = nn.Conv3d(128, 1, kernel_size=3, padding=1)

    def gaussian_3d(self, input_tensor, sigma=1.0):
        """
        Apply a 3D Gaussian filter to smooth the image.
        """
        # Define a 3D Gaussian kernel using the Gaussian function.
        kernel_size = 3  # Size of the kernel (3x3x3 for simplicity)
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(input_tensor.device)

        # Apply the Gaussian filter (convolution).
        smoothed_input = F.conv3d(input_tensor, kernel, stride=1, padding=1)
        return smoothed_input

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create a 3D Gaussian kernel.
        """
        # Create a 1D Gaussian kernel.
        ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(ax, ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

        # Normalize the kernel to ensure the sum is 1.
        kernel = kernel / kernel.sum()

        # Reshape the kernel into a 4D tensor for convolution (out_channels=1, in_channels=1, depth, height, width)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, depth, height, width)
        return kernel

    def laplacian_3d(self, input_tensor):
        """
        Apply a Laplacian filter in 3D to compute second-order derivatives.
        """
        # Laplacian kernel
        laplacian_kernel = torch.tensor(
            [[[[[0, 1, 0], [1, -6, 1], [0, 1, 0]],
               [[1, -6, 1], [-6, 36, -6], [1, -6, 1]],
               [[0, 1, 0], [1, -6, 1], [0, 1, 0]]]]],
            dtype=torch.float32, device=input_tensor.device
        )

        # Ensure input_tensor is 5D: (batch, channel, depth, height, width)
        if input_tensor.ndim != 5:
            raise ValueError("Expected input_tensor to be 5D (batch, channel, depth, height, width), got shape: {}".format(input_tensor.shape))

        # Apply the Laplacian filter
        laplacian_output = F.conv3d(input_tensor, laplacian_kernel, stride=1, padding=1)
        return laplacian_output

    def forward(self, input_tensor):
        """Forward pass for Edge Refinement."""
        # Apply a 3D Gaussian filter for smoothing
        x = self.conv(input_tensor)
        smoothed_tensor = self.gaussian_3d(x)

        # Apply Laplacian filter to the smoothed tensor
        laplacian_map = self.laplacian_3d(smoothed_tensor)

        # Refine the input tensor using the Laplacian map
        input_tensor_up = laplacian_map * input_tensor
        return input_tensor_up

class SPADEGenerator(nn.Module):
    def __init__(self, latent_channels, bottleneck_channels):
        super().__init__()
        self.conv_mean = nn.Conv3d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv_std = nn.Conv3d(latent_channels, bottleneck_channels, kernel_size=3, padding=1)

    def forward(self, autoencoder_latent, x_bottt):
        autoencoder_latent_mean = autoencoder_latent.mean(dim=(2, 3, 4), keepdim=True)
        autoencoder_latent_std = autoencoder_latent.std(dim=(2, 3, 4), keepdim=True)
        mean = self.conv_mean(x_bottt)  # Compute mean
        std = self.conv_std(x_bottt)   # Compute std
        mean = mean.mean(dim=(2, 3, 4), keepdim=True)
        std = std.std(dim=(2, 3, 4), keepdim=True)
        
        return mean, std, autoencoder_latent_mean, autoencoder_latent_std

class SPADELayer(nn.Module):
    def __init__(self, bottleneck_channels):
        super().__init__()
        self.epsilon = 1e-5  # To prevent division by zero

    def forward(self, x_bottt, mean, std):
        # Compute the batch mean and std of x_bottt
        batch_mean = x_bottt.mean(dim=(2, 3, 4), keepdim=True)
        batch_std = x_bottt.std(dim=(2, 3, 4), keepdim=True)

        # Normalize x_bottt using its batch stats
        normalized = (x_bottt - batch_mean) / (batch_std + self.epsilon)

        # Apply spatial conditioning
        output = std * normalized + mean
        return output
# 4 p_k * input_dim) * top_k *



import torch
from einops import rearrange

# ---- Morton/Z-order helpers ----
def _part1by2(n: torch.Tensor):
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8))  & 0x0300F00F
    n = (n | (n << 4))  & 0x030C30C3
    n = (n | (n << 2))  & 0x09249249
    return n

def _morton3d(x, y, z):
    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


@torch.no_grad()
def morton_perm(h: int, w: int, d: int, device: torch.device):
    """
    perm: [L] indices mapping ROW-MAJOR(H,W,D) -> MORTON
    inv : [L] indices mapping MORTON -> ROW-MAJOR(H,W,D)
    """
    x = torch.arange(h, device=device, dtype=torch.int32)  # H
    y = torch.arange(w, device=device, dtype=torch.int32)  # W
    z = torch.arange(d, device=device, dtype=torch.int32)  # D

    # match the flatten order 'h w d'
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')    # [H, W, D]

    keys = _morton3d(xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))
    perm = torch.argsort(keys)                             # row-major(H,W,D) -> morton
    inv = torch.empty_like(perm); inv[perm] = torch.arange(perm.numel(), device=device)
    return perm.long(), inv.long()





class VQVAE_seq_unet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float()):
        super(VQVAE_seq_unet, self).__init__()

        self.dropout_prob = dropout_prob  # Dropout probability

        # Initialize Encoder, Bottleneck, and Decoder as separate modules
        self.encoder = Encoder3D(in_channels, dropout_prob)
     
        self.decoder = Decoder3D(dropout_prob)
   
        self.quantizer0 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=512,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
                pretrained_embedding=None,
            )
        )
        self.quantizer1 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=256,
                embedding_dim=128,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
                pretrained_embedding=None,

            )
        )
        # config = MambaConfig()  # ✅ Creates an instance with all attributes
        # self.mamba = MambaLMHeadModel(config)  # ✅ Pass the instance with attributes
        self.mamba_b1 = MambaLMHeadModel(MambaConfig)
        self.mamba_b2 = MambaLMHeadModel(MambaConfig)
        self.mamba1 = MambaLMHeadModel(MambaConfigl1)
        self.mamba2 = MambaLMHeadModel(MambaConfigl1)

# ffffgggtteewe
       

            
        self.bidir_gate_x4 = nn.Parameter(torch.full((128,), 0.5))  # C at x4 stage
        self.bidir_gate_x6 = nn.Parameter(torch.full((512,), 0.5))  # C at x6 stage


        self._morton_cache = {}

   
    def extract_descriptor(self, vect):
        vect = rearrange(vect, 'b c h w d -> b (h w d) c')
        return vect.mean(dim=1)  # global average pooling


    def add_3d_coords(self, x):  # x: (B, C, H, W, D)
        B, C, H, W, D = x.shape
        z = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        y = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        xg = torch.linspace(-1, 1, D, device=x.device, dtype=x.dtype)
        zz, yy, xx = torch.meshgrid(z, y, xg, indexing='ij')      # (H, W, D) each
        pe = torch.stack([zz, yy, xx], dim=0).expand(B, -1, -1, -1, -1)  # (B, 3, H, W, D)
        return torch.cat([x, pe], dim=1)  # (B, C+3, H, W, D)


    def _get_morton_perm(self, h, w, d, device):
            key = (h, w, d, str(device))
            if key not in self._morton_cache:
                perm, inv = morton_perm(h, w, d, device=device)
                self._morton_cache[key] = (perm, inv)
            return self._morton_cache[key]




    def forward(self, x, mode='val', build_faiss_db=False, mask_list=None, device=None, perform_retrieval=False):

        # Encoder path
        x1, x2, x3, x4, x5, x6 = self.encoder(x)

       
        b, c, h, w, d = x4.shape
        perm, inv = self._get_morton_perm(h, w, d, x4.device)
        
      
        x_seq = rearrange(x4, 'b c h w d -> b (h w d) c')

       

        
       
        x_morton = x_seq[:, perm, :].contiguous()      # (b, L, c)
       
        
        # ---------- forward pass (→) ----------
        y_f = self.mamba1(x_morton)
        y_f = (y_f if isinstance(y_f, torch.Tensor)
               else getattr(y_f, 'last_hidden_state', getattr(y_f, 'hidden_states', getattr(y_f, 'logits', None))))
        assert y_f is not None, "mamba1 forward returned no tensor."
        y_f = y_f.contiguous()                          # (b, L, c)
        
        # ---------- backward pass (←) ----------
        x_rev = torch.flip(x_morton, dims=[1])          # reverse token order
        y_b = self.mamba2(x_rev)
        y_b = (y_b if isinstance(y_b, torch.Tensor)
               else getattr(y_b, 'last_hidden_state', getattr(y_b, 'hidden_states', getattr(y_b, 'logits', None))))
        assert y_b is not None, "mamba1 backward returned no tensor."
        y_b = torch.flip(y_b, dims=[1]).contiguous()    # un-reverse to align with y_f
        
        # ---------- fuse ----------
        # Option A: simple mean (no extra params)
        alpha = torch.sigmoid(self.bidir_gate_x4) 
       
        
        y = y_f * alpha + y_b * (1.0 - alpha)   # (b, L, c)
        
       
        y = y[:, inv, :].contiguous()
        
        # 4) back to grid
        x4 = rearrange(y, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)




        
        quantized_loss1, x4, encodings_sum1, embedding1 = self.quantizer1(x4)
        
       
        b, c, h, w, d = x6.shape
        perm, inv = self._get_morton_perm(h, w, d, x6.device)
        
        # 1) row-major -> sequence, then Morton order
        x_seq = rearrange(x6, 'b c h w d -> b (h w d) c')
        x_morton = x_seq[:, perm, :].contiguous()          # (b, L, c)
        
        # ---- forward (→) ----
        y_f = self.mamba_b1(x_morton)
        y_f = (y_f if isinstance(y_f, torch.Tensor)
               else getattr(y_f, 'last_hidden_state', getattr(y_f, 'hidden_states', getattr(y_f, 'logits', None))))
        assert y_f is not None, "mamba forward returned no tensor."
        y_f = y_f.contiguous()
        
        # ---- backward (←) ----
        x_rev = torch.flip(x_morton, dims=[1])             # reverse token order
        y_b = self.mamba_b2(x_rev)
        y_b = (y_b if isinstance(y_b, torch.Tensor)
               else getattr(y_b, 'last_hidden_state', getattr(y_b, 'hidden_states', getattr(y_b, 'logits', None))))
        assert y_b is not None, "mamba backward returned no tensor."
        y_b = torch.flip(y_b, dims=[1]).contiguous()       # un-reverse to align with y_f
        
        # ---- fuse ----
        # simple mean (no extra params). If you want more capacity, concat+linear like below.
        alpha = torch.sigmoid(self.bidir_gate_x6)  # (c,)
        y = y_f * alpha + y_b * (1.0 - alpha)

        
        # 2) Morton -> row-major
        y = y[:, inv, :].contiguous()
        
        # 3) back to grid
        x8 = rearrange(y, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        
        
        latent = x4
        
        
        quantized_loss, quantized, encodings_sum, embedding = self.quantizer0(x8)
        quantized_loss = quantized_loss+quantized_loss1
        
        # quantized_loss = 0
        # quantized = x8
        
        x8, feature = self.decoder(quantized, x4, x3, x2, x1)
      
        if mode=='train':
            return x8, quantized_loss, feature, feature

        else:
            return x8, feature
        
        # return x8, quantized, segmentataion, quantized_loss
