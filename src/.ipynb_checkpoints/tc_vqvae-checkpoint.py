# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
# import math
# from einops import rearrange
# mse_loss = nn.MSELoss()

# class EMAQuantizer(nn.Module):
#     """
#     Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
#     Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
#     that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
#     58d9a2746493717a7c9252938da7efa6006f3739.

#     This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
#     to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
#     on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

#     Args:
#         spatial_dims :  number of spatial spatial_dims.
#         num_embeddings: number of atomic elements in the codebook.
#         embedding_dim: number of channels of the input and atomic elements.
#         commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
#         decay: EMA decay. Defaults to 0.99.
#         epsilon: epsilon value. Defaults to 1e-5.
#         embedding_init: initialization method for the codebook. Defaults to "normal".
#         ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         num_embeddings: int,
#         embedding_dim: int,
#         commitment_cost: float = 0.25,
#         decay: float = 0.99,
#         epsilon: float = 1e-5,
#         embedding_init: str = "normal",
#         ddp_sync: bool = True,
#     ):
#         super().__init__()
#         self.spatial_dims: int = spatial_dims
#         self.embedding_dim: int = embedding_dim
#         self.num_embeddings: int = num_embeddings

#         assert self.spatial_dims in [2, 3], ValueError(
#             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
#         )

#         self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
#         if embedding_init == "normal":
#             # Initialization is passed since the default one is normal inside the nn.Embedding
#             pass
#         elif embedding_init == "kaiming_uniform":
#             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
#         self.embedding.weight.requires_grad = False

#         self.commitment_cost: float = commitment_cost

#         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
#         self.register_buffer("ema_w", self.embedding.weight.data.clone())

#         self.decay: float = decay
#         self.epsilon: float = epsilon

#         self.ddp_sync: bool = ddp_sync

#         # Precalculating required permutation shapes
#         self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
#         self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
#             range(1, self.spatial_dims + 1)
#         )

#     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

#         Args:
#             inputs: Encoding space tensors

#         Returns:
#             torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
#             torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
#             torch.Tensor: Quantization indices of shape [B,D,H,W,1]

#         """
#         encoding_indices_view = list(inputs.shape)
#         del encoding_indices_view[1]

#         with torch.cuda.amp.autocast(enabled=False):
#             inputs = inputs.float()

#             # Converting to channel last format
#             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

#             # Calculate Euclidean distances
#             distances = (
#                 (flat_input**2).sum(dim=1, keepdim=True)
#                 + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
#                 - 2 * torch.mm(flat_input, self.embedding.weight.t())
#             )

#             # Mapping distances to indexes
#             encoding_indices = torch.max(-distances, dim=1)[1]
#             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()
#             encoding_probabilities = torch.softmax(-distances / 1.0, dim=1)
#             entropy_loss = -torch.sum(encoding_probabilities * torch.log(encoding_probabilities + 1e-8)) / encoding_probabilities.size(0)
#             entropy_loss = 0.01 * entropy_loss 
#             # Quantize and reshape
#             encoding_indices = encoding_indices.view(encoding_indices_view)

#         return flat_input, encodings, encoding_indices, entropy_loss

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         """
#         Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
#         [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
#         decoder.

#         Args:
#             embedding_indices: Tensor in channel last format which holds indices referencing atomic
#                 elements from self.embedding

#         Returns:
#             torch.Tensor: Quantize space representation of encoding_indices in channel first format.
#         """
#         with torch.cuda.amp.autocast(enabled=False):
#             return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

#     @torch.jit.unused
#     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
#         """
#         TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
#         example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

#         Args:
#             encodings_sum: The summation of one hot representation of what encoding was used for each
#                 position.
#             dw: The multiplication of the one hot representation of what encoding was used for each
#                 position with the flattened input.

#         Returns:
#             None
#         """
#         if self.ddp_sync and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
#             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
#         else:
#             pass

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         flat_input, encodings, encoding_indices, entropy_loss = self.quantize(inputs)

#         # print("encoding_indices are", encoding_indices)
#         quantized = self.embed(encoding_indices)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             # print("EMA Training Started")
#             with torch.no_grad():
#                 encodings_sum = encodings.sum(0)
#                 dw = torch.mm(encodings.t(), flat_input)

#                 if self.ddp_sync:
#                     self.distributed_synchronization(encodings_sum, dw)

#                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

#                 # Laplace smoothing of the cluster size
#                 n = self.ema_cluster_size.sum()
#                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
#                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
#                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
#         else:
#             encodings_sum=torch.zeros(256)

#         # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
#         # print("quantized ema shape is", quantized.shape)
#         # print("inputs ema shape is", inputs.shape)
#         # Encoding Loss
        
#         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
#         # loss += entropy_loss

#         # Straight Through Estimator
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data


# class VectorQuantizer(torch.nn.Module):
#     """
#     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
#     the quantization in their own class.

#     Args:
#         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
#             based quantized representation. Defaults to None
#     """

#     def __init__(self, quantizer: torch.nn.Module = None):
#         super().__init__()

#         self.quantizer: torch.nn.Module = quantizer

#         self.perplexity: torch.Tensor = torch.rand(1)

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

#         # Perplexity calculations
#         avg_probs = (
#             torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
#             .float()
#             .div(encoding_indices.numel())
#         )

#         # self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         print("self.perplexity", self.perplexity)
#         # loss += 0.01 * self.perplexity

#         return loss, quantized, encodings_sum, embedding, encoding_indices

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         return self.quantizer.embed(embedding_indices=embedding_indices)

#     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
#         _, _, encoding_indices, _, _ = self.quantizer(encodings)

#         return encoding_indices

# # class ResidualBlock(nn.Module):
# #     def __init__(self, in_channels, dropout_prob=0.3):
# #         super(ResidualBlock, self).__init__()

# #         # First convolution: doubles the channels
# #         self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn1 = nn.BatchNorm3d(in_channels * 2)
# #         self.relu = nn.ReLU(inplace=True)
        
# #         # Dropout layer
# #         self.dropout = nn.Dropout3d(p=dropout_prob)
        
# #         # Second convolution: reduces channels back to in_channels
# #         self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn2 = nn.BatchNorm3d(in_channels)

# #     def forward(self, x):
# #         identity = x  # No downsampling, identity remains unchanged

# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)

# #         # Apply dropout
# #         out = self.dropout(out)

# #         out = self.conv2(out)
# #         out = self.bn2(out)

# #         out += identity
# #         out = self.relu(out)
        
# #         return out





# # class EMAQuantizer(nn.Module):
# #     def __init__(
# #         self,
# #         spatial_dims: int,
# #         num_embeddings: int,
# #         embedding_dim: int,
# #         commitment_cost: float = 0.25,
# #         decay: float = 0.99,
# #         epsilon: float = 1e-5,
# #         embedding_init: str = "normal",
# #         ddp_sync: bool = True,
# #     ):
# #         super().__init__()
# #         self.spatial_dims = spatial_dims
# #         self.embedding_dim = embedding_dim
# #         self.num_embeddings = num_embeddings
# #         self.num_clusters = 4

# #         # Validate spatial dimensions
# #         assert self.spatial_dims in [2, 3], ValueError(
# #             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
# #         )

# #         # Initialize embedding and clusters
# #         self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
# #         # if pretrained_embedding is not None:
# #         #     if pretrained_embedding.shape != (self.num_embeddings, self.embedding_dim):
# #         #         raise ValueError(
# #         #             f"Pretrained embedding must have shape ({self.num_embeddings}, {self.embedding_dim}), "
# #         #             f"but got {pretrained_embedding.shape}."
# #         #         )
# #         #     self.embedding.weight.data.copy_(pretrained_embedding)
# #         if embedding_init == "kaiming_uniform":
# #             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")

# #         self.embedding.weight.requires_grad = False

# #         # Divide embeddings into clusters
# #         self.cluster_size = self.num_embeddings // self.num_clusters
# #         self.cluster_centroids = self.embedding.weight.data[:self.num_clusters]  # Initialize centroids
# #         self.cluster_assignments = torch.arange(self.num_embeddings) // self.cluster_size

# #         self.commitment_cost = commitment_cost

# #         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
# #         self.register_buffer("ema_w", self.embedding.weight.data.clone())

# #         self.decay = decay
# #         self.epsilon = epsilon

# #         self.ddp_sync = ddp_sync

# #         self.flatten_permutation = [0] + list(range(2, self.spatial_dims + 2)) + [1]
# #         self.quantization_permutation = [0, self.spatial_dims + 1] + list(range(1, self.spatial_dims + 1))

# #     def assign_to_cluster(self, flat_input: torch.Tensor) -> torch.Tensor:
# #         # Compute distances to cluster centroids
# #         centroid_distances = torch.cdist(flat_input, self.cluster_centroids.to(flat_input.device), p=2)

# #         nearest_centroids = torch.argmin(centroid_distances, dim=1)

# #         # Assign each vector to the nearest centroid's cluster
# #         cluster_indices = []
# #         for i, centroid_idx in enumerate(nearest_centroids):
# #             start_idx = centroid_idx * self.cluster_size
# #             end_idx = start_idx + self.cluster_size
# #             cluster_vectors = self.embedding.weight[start_idx:end_idx]
# #             distances = torch.cdist(flat_input[i:i+1], cluster_vectors, p=2)
# #             closest_vector_idx = torch.argmin(distances)
# #             cluster_indices.append(start_idx + closest_vector_idx.item())

# #         return torch.tensor(cluster_indices, device=flat_input.device)

# #     def redistribute_unused_vectors(self):
# #         # Identify unused vectors
# #         unused_mask = self.ema_cluster_size == 0
# #         unused_indices = torch.where(unused_mask)[0]

# #         # Identify clusters with high demand
# #         high_demand_clusters = torch.topk(self.ema_cluster_size, self.num_clusters // 2).indices

# #         # Redistribute unused vectors
# #         for unused_idx in unused_indices:
# #             target_cluster = high_demand_clusters[unused_idx % len(high_demand_clusters)]
# #             cluster_start = target_cluster * self.cluster_size
# #             cluster_end = cluster_start + self.cluster_size
# #             self.cluster_assignments[unused_idx] = target_cluster

# #     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         encoding_indices_view = list(inputs.shape)
# #         del encoding_indices_view[1]

# #         with torch.cuda.amp.autocast(enabled=False):
# #             inputs = inputs.float()

# #             # Flatten input
# #             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

# #             # Assign to clusters
# #             encoding_indices = self.assign_to_cluster(flat_input)

# #             # One-hot encodings
# #             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

# #             # Reshape
# #             encoding_indices = encoding_indices.view(encoding_indices_view)

# #         return flat_input, encodings, encoding_indices

# #     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
# #         return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

# #     @torch.jit.unused
# #     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
# #         if self.ddp_sync and torch.distributed.is_initialized():
# #             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
# #             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)

# #     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
# #         flat_input, encodings, encoding_indices = self.quantize(inputs)
# #         quantized = self.embed(encoding_indices)

# #         # EMA Updates
# #         if self.training:
# #             with torch.no_grad():
# #                 encodings_sum = encodings.sum(0)
# #                 dw = torch.mm(encodings.t(), flat_input)

# #                 if self.ddp_sync:
# #                     self.distributed_synchronization(encodings_sum, dw)

# #                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))
# #                 n = self.ema_cluster_size.sum()
# #                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
# #                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
# #                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))

# #                 # Redistribute unused vectors
# #                 self.redistribute_unused_vectors()

# #         else:
# #             encodings_sum=torch.zeros(128)

# #         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
# #         quantized = inputs + (quantized - inputs).detach()

# #         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data



# # class VectorQuantizer(torch.nn.Module):
# #     """
# #     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non-fp16 compatible parts of
# #     the quantization in their own class.

# #     Args:
# #         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
# #             based quantized representation. Defaults to None
# #     """

# #     def __init__(self, quantizer: torch.nn.Module = None):
# #         super().__init__()

# #         self.quantizer: torch.nn.Module = quantizer

# #         self.perplexity: torch.Tensor = torch.rand(1)

# #     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# #         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

# #         # Perplexity calculations (considering clusters)
# #         if hasattr(self.quantizer, "num_clusters"):
# #             avg_probs_per_cluster = []
# #             for cluster_idx in range(self.quantizer.num_clusters):
# #                 # Filter indices belonging to the current cluster
# #                 cluster_mask = (self.quantizer.cluster_assignments == cluster_idx)
# #                 cluster_indices = encoding_indices[cluster_mask]
# #                 if cluster_indices.numel() > 0:
# #                     avg_probs = (
# #                         torch.histc(
# #                             cluster_indices.float(),
# #                             bins=self.quantizer.num_embeddings,
# #                             max=self.quantizer.num_embeddings,
# #                         )
# #                         .float()
# #                         .div(cluster_indices.numel())
# #                     )
# #                     avg_probs_per_cluster.append(avg_probs)
# #             # Calculate perplexity per cluster
# #             self.perplexity = torch.mean(torch.stack([
# #                 torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
# #                 for avg_probs in avg_probs_per_cluster
# #             ]))
# #         else:
# #             # Perplexity for the entire codebook
# #             avg_probs = (
# #                 torch.histc(
# #                     encoding_indices.float(),
# #                     bins=self.quantizer.num_embeddings,
# #                     max=self.quantizer.num_embeddings,
# #                 )
# #                 .float()
# #                 .div(encoding_indices.numel())
# #             )
# #             self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

# #         return loss, quantized, encodings_sum, embedding, encoding_indices

# #     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
# #         return self.quantizer.embed(embedding_indices=embedding_indices)

# #     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
# #         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(encodings)

# #         return encoding_indices







# class PositionalEncoding3D(nn.Module):
#     """Updated 3D Positional Encoding."""
#     def __init__(self, 
#                  num_embed=8192, 
#                  spatial_size=[32, 32, 32], 
#                  embed_dim=3968, 
#                  trainable=True, 
#                  pos_emb_type='embedding'):
#         super().__init__()
        
#         if isinstance(spatial_size, int):
#             spatial_size = [spatial_size, spatial_size, spatial_size]

#         self.spatial_size = spatial_size
#         self.num_embed = num_embed + 1
#         self.embed_dim = embed_dim
#         self.trainable = trainable
#         self.pos_emb_type = pos_emb_type

#         assert self.pos_emb_type in ['embedding', 'parameter']
        
#         self.emb = nn.Conv3d(128, embed_dim, kernel_size=3, padding=1)
#         self.flatten = nn.Flatten(start_dim=2)
#         if self.pos_emb_type == 'embedding':
#             self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim)
#             self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim)
#             self.depth_emb = nn.Embedding(self.spatial_size[2], embed_dim)
#         else:
#             self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim))
#             self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim))
#             self.depth_emb = nn.Parameter(torch.zeros(1, self.spatial_size[2], embed_dim))
        
#         self._set_trainable()

#     def _set_trainable(self):
#         if not self.trainable:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, index, **kwargs):
#         # assert index.dim() == 2  # B x L
#         # index = torch.clamp(index, min=0)  # Ensure indices are valid
#         # print("enc_out enc_out enc_out enc_out flatten emb", index.shape)
#         emb = self.emb(index)
#         emb = self.flatten(emb)  # Shape: (B, embed_dim, L)

#         # print("enc_out enc_out enc_out enc_out flatten emb", emb.shape)
        
#         # Transpose latent embeddings to match the attention input format
#         emb = emb.permute(0, 2, 1) 
#         # print("self.spatial_size[0]", self.spatial_size[0])
#         # print("self.spatial_size[0]", self.spatial_size[1])
#         # print("self.spatial_size[0]", self.spatial_size[2])
#         if emb.shape[1] > 0:
#         # if False:
#             if self.pos_emb_type == 'embedding':
#                 height_emb = self.height_emb(torch.arange(self.spatial_size[0], device=index.device).view(1, self.spatial_size[0])).unsqueeze(2).unsqueeze(3)  # Shape: 1 x H x 1 x 1 x embed_dim
#                 width_emb = self.width_emb(torch.arange(self.spatial_size[1], device=index.device).view(1, self.spatial_size[1])).unsqueeze(1).unsqueeze(3)   # Shape: 1 x 1 x W x 1 x embed_dim
#                 depth_emb = self.depth_emb(torch.arange(self.spatial_size[2], device=index.device).view(1, self.spatial_size[2])).unsqueeze(1).unsqueeze(1)   # Shape: 1 x 1 x 1 x D x embed_dim
#             else:
#                 height_emb = self.height_emb.unsqueeze(2).unsqueeze(3)  # Shape: 1 x H x 1 x 1 x embed_dim
#                 width_emb = self.width_emb.unsqueeze(1).unsqueeze(3)    # Shape: 1 x 1 x W x 1 x embed_dim
#                 depth_emb = self.depth_emb.unsqueeze(1).unsqueeze(1)    # Shape: 1 x 1 x 1 x D x embed_dim

#             pos_emb = (height_emb + width_emb + depth_emb).view(1, self.spatial_size[0] * self.spatial_size[1] * self.spatial_size[2], -1) # 1 x H x W x D -> 1 x L xD
#             emb = emb + pos_emb[:, :emb.shape[1], :]
        
#         return emb



# # class Conv_MLP(nn.Module):
# #     def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
# #         super().__init__()
# #         self.conv1 = nn.Conv3d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
# #         self.act = act
# #         self.conv2 = nn.Conv3d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)

# #         # self.conv3 = nn.Conv3d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
# #         # self.act = act
# #         # self.conv4 = nn.Conv3d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
    
# #         self.dropout = nn.Dropout(resid_pdrop)

# #     def forward(self, x):
# #         n =  x.size()[1]
# #         x = rearrange(x, 'b (h w d) c -> b c h w d', h=8, w=8, d=8)
# #         x = (self.conv2(self.act(self.conv1(x))))
# #         # x = self.conv4(self.act(self.conv3(x)))
# #         x = rearrange(x, 'b c h w d -> b (h w d) c')
# #         return self.dropout(x)


# class ConvBlock(nn.Module):
#     """A Conv Block with LayerNorm."""
#     def __init__(self, embed_dim, kernel_size=3, stride=1, padding=1):
#         super(ConvBlock, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels=embed_dim, out_channels=int(4 * embed_dim), kernel_size=3, stride=1, padding=1)
#         self.act = GELU2()
#         self.conv2 = nn.Conv3d(in_channels=int(4 * embed_dim), out_channels=embed_dim, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         # b, c, d, h, w = x.shape
#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=15, w=15, d=9)
#         x = (self.conv2(self.act(self.conv1(x))))
#         # x = self.conv4(self.act(self.conv3(x)))
#         x = rearrange(x, 'b c h w d -> b (h w d) c')
#         return x


# class FullAttention(nn.Module):
#     """Full Attention Module."""
#     def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0, causal=True):
#         super().__init__()
#         assert n_embd % n_head == 0
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head
#         self.causal = causal

#     def forward(self, x, mask=None):
#         # print("x shape issssssssssssssssssssssssssssssssssssss", x)
#         B, T, C = x.size()
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         # if lay>= 5:
#         #     # print("att being revered")
#         #     att = 1-att
#         if mask is not None:
#             att = att.masked_fill(mask == 0, float('-inf'))

#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         y = self.resid_drop(self.proj(y))
#         return y


# class TransformerBlock(nn.Module):
#     """A Transformer Block with Full Attention and MLP."""
#     def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0, mlp_type = None):
#         super(TransformerBlock, self).__init__()
#         self.attn = FullAttention(embed_dim, num_heads, attn_pdrop=dropout, resid_pdrop=dropout)
#         if mlp_type == 'conv_mlp':
#             self.mlp = ConvBlock(embed_dim)
#         else:
#             self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_ratio * embed_dim),
#             GELU2(),
#             nn.Linear(mlp_ratio * embed_dim, embed_dim),
#             nn.Dropout(dropout)
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#     def forward(self, x, mask=None):
#         attn_out = self.attn(x, mask)
#         x = x + attn_out
#         x = self.norm1(x)

#         mlp_out = self.mlp(x)
#         x = x + mlp_out
#         x = self.norm2(x)

#         return x


# class TransformerModel(nn.Module):
#     """Transformer Model with Full Attention, Conv Blocks, and Positional Encoding."""
#     def __init__(self, input_shape, embed_dim, num_layers, num_heads):
#         super(TransformerModel, self).__init__()
#         h, w, d = input_shape
#         self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
#                                                         spatial_size=[h, w, d])
#         self.emb = nn.Conv3d(512, 128, kernel_size=3, padding=1)
#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 TransformerBlock(embed_dim, num_heads),
#                 # ConvBlock(embed_dim)
#             )
#             for _ in range(num_layers)
#         ])

#     def forward(self, x):
#         b, c, h, w, d = x.shape
#         # x = rearrange(x, 'b c h w d -> b (h w d) c')
#         x = self.positional_encoding(x)
#         # x = rearrange(x, 'b c h w d -> b (h w d) c')
#         for layer in self.layers:
#             x = layer(x)

#         x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
#         x = self.emb(x)
#         return x



# # class TransformerModel(nn.Module):
# #     """Transformer Model with Full Attention, Uncertainty Estimation, and Soft Masking."""
# #     def __init__(self, input_shape, embed_dim, num_layers, num_heads):
# #         super(TransformerModel, self).__init__()
# #         h, w, d = input_shape
# #         self.num_layers = num_layers
# #         self.embed_dim = embed_dim
# #         self.positional_encoding = PositionalEncoding3D(embed_dim=embed_dim, 
# #                                                         spatial_size=[h, w, d])
# #         self.layers = nn.ModuleList([
# #             TransformerBlock(embed_dim, num_heads)  # Remove nn.Sequential wrapper
# #             for _ in range(num_layers)
# #         ])
# #         self.uncertainty_layers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_layers)])

# #     def calculate_uncertainty(self, x, layer_idx):
# #         """
# #         Calculate token-level uncertainty for a specific layer using MC Dropout or learned variance.
# #         """
# #         variance = self.uncertainty_layers[layer_idx](x).sigmoid()  # Values in [0, 1]
# #         return variance

# #     def forward(self, x):
# #         b, c, h, w, d = x.shape
# #         x = self.positional_encoding(x)
# #         # gt_x = rearrange(gt_x, 'b c h w d -> b (h w d) c')
# #         # masked_gt_list = []
# #         # masked_out_list = []
# #         for i, layer in enumerate(self.layers):
# #             x, att = layer(x,i)

# #             # print("at shape is", att.shape)
            
# #             if i >= 5 and i < (self.num_layers - 1):  # After the 6th layer, estimate uncertainty and mask
                
# #                 uncertainty = self.calculate_uncertainty(x, i)
# #                 # print("Uncertainty shape is", uncertainty.shape)
                
# #                 # Calculate certainty mask
# #                 certainty_mask = 1 - uncertainty  # Certainty is the complement of uncertainty
# #                 x = x * certainty_mask  # Apply soft masking to the model's output
                
# #                 # Apply the same mask to the ground truth
# #                 # masked_gt = gt_x * certainty_mask  # Mask the ground truth similarly
# #                 # masked_gt_list.append(masked_gt)  # Store masked GT for this layer
# #                 # masked_out_list.append(x)
            
# #         x = rearrange(x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
# #         # gt_x_up = rearrange(gt_x_up, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
# #         return x



# class GELU2(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return F.relu(x)










# class ConvBlock3D(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x

# class ConvBlock3D_bot(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D_bot, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 2),  stride=(1, 1, 2), padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x

# class UpsampleDepthOnly(nn.Module):
#     """Upsample depth dimension only while keeping height and width unchanged"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(UpsampleDepthOnly, self).__init__()
#         self.conv_transpose = nn.ConvTranspose3d(
#             in_channels=in_channels, 
#             out_channels=out_channels, 
#             kernel_size=(1, 1, 2),  # Upsample only depth (depth kernel size = 2)
#             stride=(1, 1, 2),       # Depth stride = 2, height and width stride = 1
#             padding=(0, 0, 0)       # No padding
#         )
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)

#     def forward(self, x):
#         x = self.conv_transpose(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x


# class Encoder3D(nn.Module):
#     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(Encoder3D, self).__init__()
        
#         self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
#         # self.res1 = ResidualBlock(8)
#         self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
#         # self.res2 = ResidualBlock(16)
#         self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
#         # self.res4 = ResidualBlock(64)
#         self.encoder5 = ConvBlock3D(64, 128, dropout_prob)
#         # self.res5 = ResidualBlock(128)
#         # self.encoder6 = ConvBlock3D_bot(128, 128, dropout_prob)
#         self.encoder7 = ConvBlock3D(128, 128, dropout_prob)
#         self.pool = nn.MaxPool3d(2)
        
#     def forward(self, x):
#         x1 = self.encoder1(x)
#         print(f"Encoder1 output shape: {x1.shape}")
#         # x1 = self.res1(x1)
#         x2 = self.encoder2(self.pool(x1))
#         print(f"Encoder2 output shape: {x2.shape}")
#         # x2 = self.res2(x2)
#         x3 = self.encoder3(self.pool(x2))
#         print(f"Encoder3 output shape: {x3.shape}")
#         # x3 = self.res3(x3)
#         x4 = self.encoder4(self.pool(x3))
#         print(f"Encoder4 output shape: {x4.shape}")
#         # x4 = self.res4(x4)
#         x5 = self.encoder5(x4)
#         print(f"Encoder5 output shape: {x5.shape}")
#         # x5 = self.res5(x5)
#         # x5 = self.encoder6(x5)
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x5 = F.pad(x5, padding, mode='constant', value=0)
#         # print(f"Encoder5 output shape: {x5.shape}")
#         x5 = self.encoder7(x5)
#         return x5


# class BottleneckBlock(nn.Module):
#     """Bottleneck block with 128 to 128 features"""
#     def __init__(self, in_channels, dropout_prob=0.3):
#         super(BottleneckBlock, self).__init__()
#         self.bottleneck = ConvBlock3D(in_channels, in_channels, dropout_prob)
        
#     def forward(self, x):
#         x = self.bottleneck(x)
#         print(f"Bottleneck output shape: {x.shape}")
#         return x


# class Decoder3D(nn.Module):
#     """Decoder with skip connections and upsampling"""
#     def __init__(self, dropout_prob=0.5):
#         super(Decoder3D, self).__init__()
#         self.dec1 = ConvBlock3D(128, 128, dropout_prob)
#         # self.res1 = UpsampleDepthOnly(128, 128, dropout_prob)
        
#         self.upsample1 = self.upsample_block(128, dropout_prob)
#         # self.res2 = ResidualBlock(64)
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         self.upsample2 = self.upsample_block(64, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         self.upsample3 = self.upsample_block(32, dropout_prob)
#         # self.res4 = ResidualBlock(16)
#         # self.conv3 = nn.Conv3d(32,16, kernel_size=3, padding=1)
#         self.upsample4 = ConvBlock3D(16, 8, dropout_prob)
#         # self.res5 = ResidualBlock(8)
#         # self.conv4 = nn.Conv3d(16,8, kernel_size=3, padding=1)
        
#         self.final_conv = nn.Conv3d(8, 4, kernel_size=3, padding=1)  # Assuming segmentation output is single channel
#         # self.convvv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
#         # self.convvv2 = nn.Conv3d(1, 1, kernel_size=1)
    
#     def upsample_block(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)
#     # def upsample_block1(self, in_channels, dropout_prob):
#     #     """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#     #     layers = [
#     #         nn.Upsample(scale_factor=1, mode='nearest'),
#     #         nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#     #         nn.ReLU(inplace=True),
#     #         nn.BatchNorm3d(in_channels // 2),
#     #         nn.Dropout3d(p=dropout_prob),
#     #     ]
#     #     return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.dec1(x)
#         # x = self.res1(x)
#         print(f"Decoder input (x): {x.shape}")
#         x6 = self.upsample1(x)  # First decoder layer
#         print(f"Upsample1 output shape: {x6.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x6 = F.pad(x6, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x6.shape}")
#         # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
#         # print(f"Conv1 output shape after concatenation: {x7.shape}")
#         # x6 = self.res2(x6)
#         x7 = self.upsample2(x6)
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x7 = F.pad(x7, padding, mode='constant', value=0)
#         print(f"Upsample2 output shape: {x7.shape}")
        
#         # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
#         # print(f"Conv2 output shape after concatenation: {x8.shape}")
#         # x7 = self.res3(x7)
#         x8 = self.upsample3(x7)
#         print(f"Upsample3 output shape: {x8.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x8 = F.pad(x8, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x8.shape}")

#         # x6 = self.res2(x6)

#         x9 = self.upsample4(x8)
#         print(f"Upsample3 output shape: {x9.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x9 = F.pad(x9, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x9.shape}")
#         # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
#         # print(f"Conv3 output shape after concatenation: {x9.shape}")
#         out = self.final_conv(x9)  # Final output
#         print(f"Final output shape: {out.shape}")
#         # out=self.convvv1(out)
#         # out=self.convvv2(out)
#         return out


# class SegmentationModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
#         super(SegmentationModel, self).__init__()
#         self.conv1 = nn.Conv3d(4, 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv4_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv5_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv6_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
    
#         # print("X shape  before is", x.shape)
#         # if x.shape[4] <= 100:
#         #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
#         # else:
#         #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
#         x = self.conv1(x)
#         segmentation_mask1 = self.conv2(x)
#         print("segmentation_mask1 shape after is", segmentation_mask1.shape)
        
        
#         # Compute softmax probabilities over classes
#         output_probabilities = F.softmax(segmentation_mask1, dim=1)
#         return output_probabilities





# class VQVAE_seq(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, dropout_prob: float()):
#         super(VQVAE_seq, self).__init__()

#         self.dropout_prob = dropout_prob  # Dropout probability

#         # Initialize Encoder, Bottleneck, and Decoder as separate modules
#         self.encoder = Encoder3D(in_channels, dropout_prob)
#         self.bottleneck = BottleneckBlock(128, dropout_prob)
#         self.decoder = Decoder3D(dropout_prob)
#         self.segmentation=SegmentationModel(4, 4, 4)
#         self.quantizer0 = VectorQuantizer(
#             quantizer=EMAQuantizer(
#                 spatial_dims=3,
#                 num_embeddings=512,
#                 embedding_dim=128,
#                 commitment_cost=0.25,
#                 decay=0.99,
#                 epsilon=1e-5,
#                 embedding_init='uniform',
#                 ddp_sync=False,
#             )
#         )
#         self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
#         # self.conv4 = ConvBlock3D(16, 32, dropout_prob)
#         # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
#         # self.transmodel = TransformerModel(input_shape=[14, 14, 16], embed_dim=512, num_layers=10, num_heads=8)
#     def forward(self, x):
#         # Encoder path
#         x4 = self.encoder(x)
        
#         # Bottleneck
#         x5 = self.bottleneck(x4)
#         # print("x_bot isss  s", x5.dtype)
#         # print("x_bot isss  s", x5.shape)
        
#         # x7 = self.conv1(x5)
        
#         # x5 = self.conv2(x5)
#         # x5 = self.conv3(x5)
#         quantization_loss0, z_quantized0, encodings_sum0, embedding0, encoding_indices = self.quantizer0(x5)

#         # z_quantized0_post = self.conv4(z_quantized0)
#         # z_quantized0_post = self.conv3(z_quantized0)
#         # z_quantized0_post = self.conv4(z_quantized0_post)

#         # Decoder path with skip connections
#         # z_quantized0_post=self.transmodel(z_quantized0_post)
#         reconstruction = self.decoder(z_quantized0)
#         segmentation_mask = self.segmentation(reconstruction)
        
       
        
        
#         # print("segmentation_mask", segmentation_mask.shape)

        
#         total_quantization_loss = torch.mean(quantization_loss0)
#         # print("total_quantization_loss2222222222222222", (total_quantization_loss))
# # #        

#         return z_quantized0, segmentation_mask, total_quantization_loss, encodings_sum0, embedding0






#with 2 by 2 conv


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
# from einops import rearrange
# mse_loss = nn.MSELoss()

# class EMAQuantizer(nn.Module):
#     """
#     Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
#     Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
#     that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
#     58d9a2746493717a7c9252938da7efa6006f3739.

#     This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
#     to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
#     on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

#     Args:
#         spatial_dims :  number of spatial spatial_dims.
#         num_embeddings: number of atomic elements in the codebook.
#         embedding_dim: number of channels of the input and atomic elements.
#         commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
#         decay: EMA decay. Defaults to 0.99.
#         epsilon: epsilon value. Defaults to 1e-5.
#         embedding_init: initialization method for the codebook. Defaults to "normal".
#         ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         num_embeddings: int,
#         embedding_dim: int,
#         commitment_cost: float = 0.25,
#         decay: float = 0.99,
#         epsilon: float = 1e-5,
#         embedding_init: str = "normal",
#         ddp_sync: bool = True,
#     ):
#         super().__init__()
#         self.spatial_dims: int = spatial_dims
#         self.embedding_dim: int = embedding_dim
#         self.num_embeddings: int = num_embeddings

#         assert self.spatial_dims in [2, 3], ValueError(
#             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
#         )

#         self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
#         if embedding_init == "normal":
#             # Initialization is passed since the default one is normal inside the nn.Embedding
#             pass
#         elif embedding_init == "kaiming_uniform":
#             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
#         self.embedding.weight.requires_grad = False

#         self.commitment_cost: float = commitment_cost

#         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
#         self.register_buffer("ema_w", self.embedding.weight.data.clone())

#         self.decay: float = decay
#         self.epsilon: float = epsilon

#         self.ddp_sync: bool = ddp_sync

#         # Precalculating required permutation shapes
#         self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
#         self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
#             range(1, self.spatial_dims + 1)
#         )

#     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

#         Args:
#             inputs: Encoding space tensors

#         Returns:
#             torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
#             torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
#             torch.Tensor: Quantization indices of shape [B,D,H,W,1]

#         """
#         encoding_indices_view = list(inputs.shape)
#         del encoding_indices_view[1]

#         with torch.cuda.amp.autocast(enabled=False):
#             inputs = inputs.float()

#             # Converting to channel last format
#             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

#             # Calculate Euclidean distances
#             distances = (
#                 (flat_input**2).sum(dim=1, keepdim=True)
#                 + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
#                 - 2 * torch.mm(flat_input, self.embedding.weight.t())
#             )

#             # Mapping distances to indexes
#             encoding_indices = torch.max(-distances, dim=1)[1]
#             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()
#             encoding_probabilities = torch.softmax(-distances / 1.0, dim=1)
#             entropy_loss = -torch.sum(encoding_probabilities * torch.log(encoding_probabilities + 1e-8)) / encoding_probabilities.size(0)
#             entropy_loss = 0.01 * entropy_loss 
#             # Quantize and reshape
#             encoding_indices = encoding_indices.view(encoding_indices_view)

#         return flat_input, encodings, encoding_indices, entropy_loss

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         """
#         Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
#         [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
#         decoder.

#         Args:
#             embedding_indices: Tensor in channel last format which holds indices referencing atomic
#                 elements from self.embedding

#         Returns:
#             torch.Tensor: Quantize space representation of encoding_indices in channel first format.
#         """
#         with torch.cuda.amp.autocast(enabled=False):
#             return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

#     @torch.jit.unused
#     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
#         """
#         TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
#         example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

#         Args:
#             encodings_sum: The summation of one hot representation of what encoding was used for each
#                 position.
#             dw: The multiplication of the one hot representation of what encoding was used for each
#                 position with the flattened input.

#         Returns:
#             None
#         """
#         if self.ddp_sync and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
#             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
#         else:
#             pass

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         flat_input, encodings, encoding_indices, entropy_loss = self.quantize(inputs)
#         quantized = self.embed(encoding_indices)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             print("EMA Training Started")
#             with torch.no_grad():
#                 encodings_sum = encodings.sum(0)
#                 dw = torch.mm(encodings.t(), flat_input)

#                 if self.ddp_sync:
#                     self.distributed_synchronization(encodings_sum, dw)

#                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

#                 # Laplace smoothing of the cluster size
#                 n = self.ema_cluster_size.sum()
#                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
#                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
#                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
#         else:
#             encodings_sum=torch.zeros(256)

#         # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
#         print("quantized ema shape is", quantized.shape)
#         print("inputs ema shape is", inputs.shape)
#         # Encoding Loss
        
#         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
#         # loss += entropy_loss

#         # Straight Through Estimator
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data


# class VectorQuantizer(torch.nn.Module):
#     """
#     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
#     the quantization in their own class.

#     Args:
#         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
#             based quantized representation. Defaults to None
#     """

#     def __init__(self, quantizer: torch.nn.Module = None):
#         super().__init__()

#         self.quantizer: torch.nn.Module = quantizer

#         self.perplexity: torch.Tensor = torch.rand(1)

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

#         # Perplexity calculations
#         avg_probs = (
#             torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
#             .float()
#             .div(encoding_indices.numel())
#         )

#         # self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         print("self.perplexity", self.perplexity)
#         # loss += 0.01 * self.perplexity

#         return loss, quantized, encodings_sum, embedding, encoding_indices

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         # print("encoding_indices embed view is", embedding_indices.shape)
#         # x = rearrange(embedding_indices, 'b (h w d) -> b h w d', h=15, w=15, d=9)
#         # # print('x rearrange shape', x.shape)
#         # encoding_indices_view = list(x.shape)
#         # encoding_indices = embedding_indices.view(encoding_indices_view)
#         # print("encoding_indices embed view is", encoding_indices.shape)
#         return self.quantizer.embed(embedding_indices)

#     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
#         _, _, encoding_indices, _, _ = self.quantizer(encodings)

#         return encoding_indices

# # class ResidualBlock(nn.Module):
# #     def __init__(self, in_channels, dropout_prob=0.3):
# #         super(ResidualBlock, self).__init__()

# #         # First convolution: doubles the channels
# #         self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn1 = nn.BatchNorm3d(in_channels * 2)
# #         self.relu = nn.ReLU(inplace=True)
        
# #         # Dropout layer
# #         self.dropout = nn.Dropout3d(p=dropout_prob)
        
# #         # Second convolution: reduces channels back to in_channels
# #         self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn2 = nn.BatchNorm3d(in_channels)

# #     def forward(self, x):
# #         identity = x  # No downsampling, identity remains unchanged

# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)

# #         # Apply dropout
# #         out = self.dropout(out)

# #         out = self.conv2(out)
# #         out = self.bn2(out)

# #         out += identity
# #         out = self.relu(out)
        
# #         return out



# class ConvBlock3D(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x

# class ConvBlock2k3D(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock2k3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x


# class Encoder3D(nn.Module):
#     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(Encoder3D, self).__init__()
        
#         self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
#         # self.res1 = ResidualBlock(8)
#         self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
#         # self.res2 = ResidualBlock(16)
#         self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         self.encoder4 = ConvBlock2k3D(32, 64, dropout_prob)
#         # self.res4 = ResidualBlock(64)
#         self.encoder5 = ConvBlock2k3D(64, 128, dropout_prob)
#         # self.res5 = ResidualBlock(128)
#         self.encoder6 = ConvBlock2k3D(128, 128, dropout_prob)
#         self.pool = nn.MaxPool3d(2)
        
#     def forward(self, x):
#         x1 = self.encoder1(x)
#         print(f"Encoder1 output shape: {x1.shape}")
#         # x1 = self.res1(x1)
#         x2 = self.encoder2(self.pool(x1))
#         print(f"Encoder2 output shape: {x2.shape}")
#         # x2 = self.res2(x2)
#         x3 = self.encoder3(self.pool(x2))
#         print(f"Encoder3 output shape: {x3.shape}")
#         # x3 = self.res3(x3)
#         x4 = self.encoder4(self.pool(x3))
#         padding = (0, 1, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
#         x4 = F.pad(x4, padding, mode='constant', value=0)
#         print(f"Encoder4 output shape: {x4.shape}")
#         # x4 = self.res4(x4)
#         x5 = self.encoder5(self.pool(x4))
#         padding = (0, 1, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
#         x5 = F.pad(x5, padding, mode='constant', value=0)
#         print(f"Encoder5 output shape: {x5.shape}")
#         # x5 = self.res5(x5)
#         x5 = self.encoder6(x5)
#         padding = (0, 1, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
#         x5 = F.pad(x5, padding, mode='constant', value=0)
#         print(f"Encoder5 output shape: {x5.shape}")
#         return x5


# class BottleneckBlock(nn.Module):
#     """Bottleneck block with 128 to 128 features"""
#     def __init__(self, in_channels, dropout_prob=0.3):
#         super(BottleneckBlock, self).__init__()
#         self.bottleneck = ConvBlock3D(in_channels, in_channels, dropout_prob)
        
#     def forward(self, x):
#         x = self.bottleneck(x)
#         print(f"Bottleneck output shape: {x.shape}")
#         return x


# class Decoder3D(nn.Module):
#     """Decoder with skip connections and upsampling"""
#     def __init__(self, dropout_prob=0.5):
#         super(Decoder3D, self).__init__()
#         self.res1 = ConvBlock3D(128, 128, dropout_prob)
        
#         self.upsample1 = self.upsample_block1(128, dropout_prob)
#         # self.res2 = ResidualBlock(64)
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         self.upsample2 = self.upsample_block1(64, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         self.upsample3 = self.upsample_block(32, dropout_prob)
#         # self.res4 = ResidualBlock(16)
#         # self.conv3 = nn.Conv3d(32,16, kernel_size=3, padding=1)
#         self.upsample4 = self.upsample_block(16, dropout_prob)
#         # self.res5 = ResidualBlock(8)
#         # self.conv4 = nn.Conv3d(16,8, kernel_size=3, padding=1)
        
#         self.final_conv = nn.Conv3d(8, 4, kernel_size=3, padding=1)  # Assuming segmentation output is single channel
#         # self.convvv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
#         # self.convvv2 = nn.Conv3d(1, 1, kernel_size=1)
    
#     def upsample_block(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)
#     def upsample_block1(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=2, padding=0),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.res1(x)
#         print(f"Decoder input (x): {x.shape}")
#         x6 = self.upsample1(x)  # First decoder layer
#         print(f"Upsample1 output shape: {x6.shape}")
#         padding = (1, 1, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
#         x6 = F.pad(x6, padding, mode='constant', value=0)
#         print(f"Upsample1 after padin output shape: {x6.shape}")
#         # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
#         # print(f"Conv1 output shape after concatenation: {x7.shape}")
#         # x6 = self.res2(x6)
#         x7 = self.upsample2(x6)
#         print(f"Upsample2 output shape: {x7.shape}")
#         padding = (0, 1, 0, 1, 0, 1)  # (left, right, top, bottom, front, back)
#         x7 = F.pad(x7, padding, mode='constant', value=0)
#         print(f"Upsample2 after padin output shape: {x7.shape}")
#         # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
#         # print(f"Conv2 output shape after concatenation: {x8.shape}")
#         # x7 = self.res3(x7)
#         x8 = self.upsample3(x7)
#         print(f"Upsample3 output shape: {x8.shape}")
#         padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         x8 = F.pad(x8, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x8.shape}")

#         # x6 = self.res2(x6)

#         x9 = self.upsample4(x8)
#         print(f"Upsample3 output shape: {x9.shape}")
#         padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         x9 = F.pad(x9, padding, mode='constant', value=0)
#         print(f"Upsample3 output shape: {x9.shape}")
#         # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
#         # print(f"Conv3 output shape after concatenation: {x9.shape}")
#         out = self.final_conv(x9)  # Final output
#         print(f"Final output shape: {out.shape}")
#         # out=self.convvv1(out)
#         # out=self.convvv2(out)
#         return out


# class SegmentationModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
#         super(SegmentationModel, self).__init__()
#         self.conv1 = nn.Conv3d(4, 2, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(2, 2, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv4_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv5_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv6_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
    
#         print("X shape  before is", x.shape)
#         # if x.shape[4] <= 100:
#         #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
#         # else:
#         #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
#         x = self.conv1(x)
#         segmentation_mask1 = self.conv2(x)
#         print("X shape after is", segmentation_mask1.shape)
        
        
#         # Compute softmax probabilities over classes
#         output_probabilities = F.softmax(segmentation_mask1, dim=1)
#         return output_probabilities





# class VQVAE_seq(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, dropout_prob: float()):
#         super(VQVAE_seq, self).__init__()

#         self.dropout_prob = dropout_prob  # Dropout probability

#         # Initialize Encoder, Bottleneck, and Decoder as separate modules
#         self.encoder = Encoder3D(in_channels, dropout_prob)
#         # self.bottleneck = BottleneckBlock(128, dropout_prob)
#         self.decoder = Decoder3D(dropout_prob)
#         self.segmentation=SegmentationModel(4, 4, 4)
#         self.quantizer0 = VectorQuantizer(
#             quantizer=EMAQuantizer(
#                 spatial_dims=3,
#                 num_embeddings=128,
#                 embedding_dim=128,
#                 commitment_cost=0.25,
#                 decay=0.99,
#                 epsilon=1e-5,
#                 embedding_init='uniform',
#                 ddp_sync=False,
#             )
#         )
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
#         # self.conv4 = ConvBlock3D(16, 32, dropout_prob)
#         # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         # self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
#     def forward(self, x):
#         # Encoder path
#         x4 = self.encoder(x)
        
#         # Bottleneck
#         # x5 = self.bottleneck(x4)
#         # x5 = self.conv1(x5)
#         # x5 = self.conv2(x5)
#         # x5 = self.conv3(x5)
#         quantization_loss0, z_quantized0, encodings_sum0, embedding0, encoding_indices = self.quantizer0(x4)

#         # z_quantized0_post = self.conv4(z_quantized0)
#         # z_quantized0_post = self.conv3(z_quantized0)
#         # z_quantized0_post = self.conv4(z_quantized0)

#         # Decoder path with skip connections
#         reconstruction = self.decoder(z_quantized0)
#         segmentation_mask = self.segmentation(reconstruction)
        
       
        
        
#         print("segmentation_mask", segmentation_mask.shape)

        
#         total_quantization_loss = torch.mean(quantization_loss0)
#         print("total_quantization_loss2222222222222222", (total_quantization_loss))
# # #        

#         return z_quantized0, segmentation_mask, total_quantization_loss, encodings_sum0, embedding0






























# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, List
# from einops import rearrange
# mse_loss = nn.MSELoss()

# class EMAQuantizer(nn.Module):
#     """
#     Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
#     Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
#     that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
#     58d9a2746493717a7c9252938da7efa6006f3739.

#     This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
#     to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
#     on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

#     Args:
#         spatial_dims :  number of spatial spatial_dims.
#         num_embeddings: number of atomic elements in the codebook.
#         embedding_dim: number of channels of the input and atomic elements.
#         commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
#         decay: EMA decay. Defaults to 0.99.
#         epsilon: epsilon value. Defaults to 1e-5.
#         embedding_init: initialization method for the codebook. Defaults to "normal".
#         ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
#     """

#     def __init__(
#         self,
#         spatial_dims: int,
#         num_embeddings: int,
#         embedding_dim: int,
#         commitment_cost: float = 0.25,
#         decay: float = 0.99,
#         epsilon: float = 1e-5,
#         embedding_init: str = "normal",
#         ddp_sync: bool = True,
#     ):
#         super().__init__()
#         self.spatial_dims: int = spatial_dims
#         self.embedding_dim: int = embedding_dim
#         self.num_embeddings: int = num_embeddings

#         assert self.spatial_dims in [2, 3], ValueError(
#             f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
#         )

#         self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
#         if embedding_init == "normal":
#             # Initialization is passed since the default one is normal inside the nn.Embedding
#             pass
#         elif embedding_init == "kaiming_uniform":
#             torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
#         self.embedding.weight.requires_grad = False

#         self.commitment_cost: float = commitment_cost

#         self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
#         self.register_buffer("ema_w", self.embedding.weight.data.clone())

#         self.decay: float = decay
#         self.epsilon: float = epsilon

#         self.ddp_sync: bool = ddp_sync

#         # Precalculating required permutation shapes
#         self.flatten_permutation: Sequence[int] = [0] + list(range(2, self.spatial_dims + 2)) + [1]
#         self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
#             range(1, self.spatial_dims + 1)
#         )

#     def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

#         Args:
#             inputs: Encoding space tensors

#         Returns:
#             torch.Tensor: Flatten version of the input of shape [B*D*H*W, C].
#             torch.Tensor: One-hot representation of the quantization indices of shape [B*D*H*W, self.num_embeddings].
#             torch.Tensor: Quantization indices of shape [B,D,H,W,1]

#         """
#         encoding_indices_view = list(inputs.shape)
#         del encoding_indices_view[1]

#         with torch.cuda.amp.autocast(enabled=False):
#             inputs = inputs.float()

#             # Converting to channel last format
#             flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

#             # Calculate Euclidean distances
#             distances = (
#                 (flat_input**2).sum(dim=1, keepdim=True)
#                 + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
#                 - 2 * torch.mm(flat_input, self.embedding.weight.t())
#             )

#             # Mapping distances to indexes
#             encoding_indices = torch.max(-distances, dim=1)[1]
#             encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()
#             encoding_probabilities = torch.softmax(-distances / 1.0, dim=1)
#             entropy_loss = -torch.sum(encoding_probabilities * torch.log(encoding_probabilities + 1e-8)) / encoding_probabilities.size(0)
#             entropy_loss = 0.01 * entropy_loss 
#             # Quantize and reshape
#             encoding_indices = encoding_indices.view(encoding_indices_view)

#         return flat_input, encodings, encoding_indices, entropy_loss

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         """
#         Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
#         [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
#         decoder.

#         Args:
#             embedding_indices: Tensor in channel last format which holds indices referencing atomic
#                 elements from self.embedding

#         Returns:
#             torch.Tensor: Quantize space representation of encoding_indices in channel first format.
#         """
#         with torch.cuda.amp.autocast(enabled=False):
#             return self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()

#     @torch.jit.unused
#     def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
#         """
#         TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
#         example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

#         Args:
#             encodings_sum: The summation of one hot representation of what encoding was used for each
#                 position.
#             dw: The multiplication of the one hot representation of what encoding was used for each
#                 position with the flattened input.

#         Returns:
#             None
#         """
#         if self.ddp_sync and torch.distributed.is_initialized():
#             torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
#             torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
#         else:
#             pass

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         flat_input, encodings, encoding_indices, entropy_loss = self.quantize(inputs)
#         quantized = self.embed(encoding_indices)

#         # Use EMA to update the embedding vectors
#         if self.training:
#             print("EMA Training Started")
#             with torch.no_grad():
#                 encodings_sum = encodings.sum(0)
#                 dw = torch.mm(encodings.t(), flat_input)

#                 if self.ddp_sync:
#                     self.distributed_synchronization(encodings_sum, dw)

#                 self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

#                 # Laplace smoothing of the cluster size
#                 n = self.ema_cluster_size.sum()
#                 weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
#                 self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
#                 self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))
#         else:
#             encodings_sum=torch.zeros(256)

#         # print("self.embedding.weight.data", (self.embedding.weight.data).shape)
#         print("quantized ema shape is", quantized.shape)
#         print("inputs ema shape is", inputs.shape)
#         # Encoding Loss
        
#         loss = self.commitment_cost * mse_loss(quantized.detach(), inputs)
#         # loss += entropy_loss

#         # Straight Through Estimator
#         quantized = inputs + (quantized - inputs).detach()

#         return quantized, loss, encoding_indices, encodings_sum, self.embedding.weight.data


# class VectorQuantizer(torch.nn.Module):
#     """
#     Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
#     the quantization in their own class.

#     Args:
#         quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
#             based quantized representation. Defaults to None
#     """

#     def __init__(self, quantizer: torch.nn.Module = None):
#         super().__init__()

#         self.quantizer: torch.nn.Module = quantizer

#         self.perplexity: torch.Tensor = torch.rand(1)

#     def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         quantized, loss, encoding_indices, encodings_sum, embedding = self.quantizer(inputs)

#         # Perplexity calculations
#         avg_probs = (
#             torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
#             .float()
#             .div(encoding_indices.numel())
#         )

#         # self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
#         print("self.perplexity", self.perplexity)
#         # loss += 0.01 * self.perplexity

#         return loss, quantized, encodings_sum, embedding, encoding_indices

#     def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
#         # print("encoding_indices embed view is", embedding_indices.shape)
#         # x = rearrange(embedding_indices, 'b (h w d) -> b h w d', h=15, w=15, d=9)
#         # # print('x rearrange shape', x.shape)
#         # encoding_indices_view = list(x.shape)
#         # encoding_indices = embedding_indices.view(encoding_indices_view)
#         # print("encoding_indices embed view is", encoding_indices.shape)
#         return self.quantizer.embed(embedding_indices)

#     def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
#         _, _, encoding_indices, _, _ = self.quantizer(encodings)

#         return encoding_indices

# # class ResidualBlock(nn.Module):
# #     def __init__(self, in_channels, dropout_prob=0.3):
# #         super(ResidualBlock, self).__init__()

# #         # First convolution: doubles the channels
# #         self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn1 = nn.BatchNorm3d(in_channels * 2)
# #         self.relu = nn.ReLU(inplace=True)
        
# #         # Dropout layer
# #         self.dropout = nn.Dropout3d(p=dropout_prob)
        
# #         # Second convolution: reduces channels back to in_channels
# #         self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
# #         self.bn2 = nn.BatchNorm3d(in_channels)

# #     def forward(self, x):
# #         identity = x  # No downsampling, identity remains unchanged

# #         out = self.conv1(x)
# #         out = self.bn1(out)
# #         out = self.relu(out)

# #         # Apply dropout
# #         out = self.dropout(out)

# #         out = self.conv2(out)
# #         out = self.bn2(out)

# #         out += identity
# #         out = self.relu(out)
        
# #         return out



# class ConvBlock3D(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x

# class ConvBlock3D_dim(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D_dim, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x


# class Encoder3D(nn.Module):
#     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(Encoder3D, self).__init__()
        
#         self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
#         # self.res1 = ResidualBlock(8)
#         self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
#         # self.res2 = ResidualBlock(16)
#         self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
#         # self.res4 = ResidualBlock(64)
#         self.encoder5 = ConvBlock3D_dim(64, 128, dropout_prob)
#         # self.res5 = ResidualBlock(128)
#         self.encoder6 = ConvBlock3D_dim(128, 128, dropout_prob)
#         self.pool = nn.MaxPool3d(2)
        
#     def forward(self, x):
#         x1 = self.encoder1(x)
#         print(f"Encoder1 output shape: {x1.shape}")
#         # x1 = self.res1(x1)
#         x2 = self.encoder2(self.pool(x1))
#         print(f"Encoder2 output shape: {x2.shape}")
#         # x2 = self.res2(x2)
#         x3 = self.encoder3(self.pool(x2))
#         print(f"Encoder3 output shape: {x3.shape}")
#         # x3 = self.res3(x3)
#         x4 = self.encoder4(self.pool(x3))
#         print(f"Encoder4 output shape: {x4.shape}")
#         # x4 = self.res4(x4)
#         x5 = self.encoder5((x4))
#         padding = (0, 0, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         x5 = F.pad(x5, padding, mode='constant', value=0)
#         print(f"Encoder5 output shape: {x5.shape}")
#         # x5 = self.res5(x5)
#         x5 = self.encoder6(x5)
#         padding = (0, 0, 1, 1, 1, 1)
#         x5 = F.pad(x5, padding, mode='constant', value=0)
#         print(f"Encoder5 output shape: {x5.shape}")
#         return x5


# class BottleneckBlock(nn.Module):
#     """Bottleneck block with 128 to 128 features"""
#     def __init__(self, in_channels, dropout_prob=0.3):
#         super(BottleneckBlock, self).__init__()
#         self.bottleneck = ConvBlock3D(in_channels, in_channels, dropout_prob)
        
#     def forward(self, x):
#         x = self.bottleneck(x)
#         print(f"Bottleneck output shape: {x.shape}")
#         return x


# class Decoder3D(nn.Module):
#     """Decoder with skip connections and upsampling"""
#     def __init__(self, dropout_prob=0.5):
#         super(Decoder3D, self).__init__()
#         self.res1 = ConvBlock3D_dim(128, 128, dropout_prob)
#         self.res2 = ConvBlock3D_dim(128, 64, dropout_prob)
#         self.upsample1 = self.upsample_block1(64, dropout_prob)
#         # self.res2 = ResidualBlock(64)
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         self.upsample2 = self.upsample_block(32, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         self.upsample3 = self.upsample_block(16, dropout_prob)
#         # self.res4 = ResidualBlock(16)
#         # self.conv3 = nn.Conv3d(32,16, kernel_size=3, padding=1)
#         # self.upsample4 = self.upsample_block(8, dropout_prob)
#         # self.res5 = ResidualBlock(8)
#         # self.conv4 = nn.Conv3d(16,8, kernel_size=3, padding=1)
        
#         self.final_conv = nn.Conv3d(8, 4, kernel_size=3, padding=1)  # Assuming segmentation output is single channel
#         # self.convvv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
#         # self.convvv2 = nn.Conv3d(1, 1, kernel_size=1)
    
    # def upsample_block(self, in_channels, dropout_prob):
    #     """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
    #     layers = [
    #         nn.Upsample(scale_factor=2, mode='nearest'),
    #         nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm3d(in_channels // 2),
    #         nn.Dropout3d(p=dropout_prob),
    #     ]
#         return nn.Sequential(*layers)
#     def upsample_block1(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=1, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.res1(x)
#         padding = (2, 2, 1, 1, 1, 1)
#         x = F.pad(x, padding, mode='constant', value=0)
#         print(f"Encoder5 output shape: {x.shape}")
#         print(f"Decoder input (x): {x.shape}")
#         x = self.res2(x)
#         padding = (2, 2, 1, 1, 1, 1)
#         x = F.pad(x, padding, mode='constant', value=0)
#         print(f"Encoder5 output shape: {x.shape}")
#         print(f"Decoder input (x): {x.shape}")
#         x6 = self.upsample1(x)  # First decoder layer
#         print(f"Upsample1 output shape: {x6.shape}")
#         padding = (0, 0, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         x6 = F.pad(x6, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x6.shape}")
#         # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
#         # print(f"Conv1 output shape after concatenation: {x7.shape}")
#         # x6 = self.res2(x6)
#         x7 = self.upsample2(x6)
#         print(f"Upsample2 output shape: {x7.shape}")
        
#         # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
#         # print(f"Conv2 output shape after concatenation: {x8.shape}")
#         # x7 = self.res3(x7)
#         x8 = self.upsample3(x7)
#         print(f"Upsample3 output shape: {x8.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x8 = F.pad(x8, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x8.shape}")

#         # x6 = self.res2(x6)

#         # x9 = self.upsample4(x8)
#         # print(f"Upsample3 output shape: {x9.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x9 = F.pad(x9, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x9.shape}")
#         # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
#         # print(f"Conv3 output shape after concatenation: {x9.shape}")
#         out = self.final_conv(x8)  # Final output
#         print(f"Final output shape: {out.shape}")
#         # out=self.convvv1(out)
#         # out=self.convvv2(out)
#         return out


# class SegmentationModel(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
#         super(SegmentationModel, self).__init__()
#         self.conv1 = nn.Conv3d(4, 4, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv4_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv5_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#         # self.conv6_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
    
#         print("X shape  before is", x.shape)
#         # if x.shape[4] <= 100:
#         #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
#         # else:
#         #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
#         x = self.conv1(x)
#         segmentation_mask1 = self.conv2(x)
#         print("X shape after is", segmentation_mask1.shape)
        
        
#         # Compute softmax probabilities over classes
#         output_probabilities = F.softmax(segmentation_mask1, dim=1)
#         return output_probabilities





# class VQVAE_seq(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, dropout_prob: float()):
#         super(VQVAE_seq, self).__init__()

#         self.dropout_prob = dropout_prob  # Dropout probability

#         # Initialize Encoder, Bottleneck, and Decoder as separate modules
#         self.encoder = Encoder3D(in_channels, dropout_prob)
#         self.bottleneck = BottleneckBlock(128, dropout_prob)
#         self.decoder = Decoder3D(dropout_prob)
#         self.segmentation=SegmentationModel(4, 4, 4)
#         self.quantizer0 = VectorQuantizer(
#             quantizer=EMAQuantizer(
#                 spatial_dims=3,
#                 num_embeddings=512,
#                 embedding_dim=128,
#                 commitment_cost=0.25,
#                 decay=0.99,
#                 epsilon=1e-5,
#                 embedding_init='uniform',
#                 ddp_sync=False,
#             )
#         )
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
#         # self.conv4 = ConvBlock3D(16, 32, dropout_prob)
#         # self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
#         # self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
#     def forward(self, x):
#         # Encoder path
#         x4 = self.encoder(x)
        
#         # Bottleneck
#         x5 = self.bottleneck(x4)
#         # x5 = self.conv1(x5)
#         # x5 = self.conv2(x5)
#         # x5 = self.conv3(x5)
#         quantization_loss0, z_quantized0, encodings_sum0, embedding0, encoding_indices = self.quantizer0(x5)

#         # z_quantized0_post = self.conv4(z_quantized0)
#         # z_quantized0_post = self.conv3(z_quantized0)
#         # z_quantized0_post = self.conv4(z_quantized0_post)

#         # Decoder path with skip connections
#         reconstruction = self.decoder(z_quantized0)
#         segmentation_mask = self.segmentation(reconstruction)
        
       
        
        
#         print("segmentation_mask", segmentation_mask.shape)

        
#         total_quantization_loss = torch.mean(quantization_loss0)
#         print("total_quantization_loss2222222222222222", (total_quantization_loss))
# # #        

#         return z_quantized0, segmentation_mask, total_quantization_loss, encodings_sum0, embedding0








import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from einops import rearrange
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
            encoding_probabilities = torch.softmax(-distances / 1.0, dim=1)
            entropy_loss = -torch.sum(encoding_probabilities * torch.log(encoding_probabilities + 1e-8)) / encoding_probabilities.size(0)
            entropy_loss = 0.01 * entropy_loss 
            # Quantize and reshape
            encoding_indices = encoding_indices.view(encoding_indices_view)

        return flat_input, encodings, encoding_indices, entropy_loss

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
        flat_input, encodings, encoding_indices, entropy_loss = self.quantize(inputs)
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
        # loss += entropy_loss

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

        # self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        print("self.perplexity", self.perplexity)
        # loss += 0.01 * self.perplexity

        return loss, quantized, encodings_sum, embedding, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        # print("encoding_indices embed view is", embedding_indices.shape)
        # x = rearrange(embedding_indices, 'b (h w d) -> b h w d', h=15, w=15, d=9)
        # # print('x rearrange shape', x.shape)
        # encoding_indices_view = list(x.shape)
        # encoding_indices = embedding_indices.view(encoding_indices_view)
        # print("encoding_indices embed view is", encoding_indices.shape)
        return self.quantizer.embed(embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        _, _, encoding_indices, _, _ = self.quantizer(encodings)

        return encoding_indices

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, dropout_prob=0.3):
#         super(ResidualBlock, self).__init__()

#         # First convolution: doubles the channels
#         self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(in_channels * 2)
#         self.relu = nn.ReLU(inplace=True)
        
#         # Dropout layer
#         self.dropout = nn.Dropout3d(p=dropout_prob)
        
#         # Second convolution: reduces channels back to in_channels
#         self.conv2 = nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(in_channels)

#     def forward(self, x):
#         identity = x  # No downsampling, identity remains unchanged

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         # Apply dropout
#         out = self.dropout(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += identity
#         out = self.relu(out)
        
#         return out



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


# class ConvBlock3D_wopad(nn.Module):
#     """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
#     def __init__(self, in_channels, out_channels, dropout_prob):
#         super(ConvBlock3D_wopad, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3)
#         self.relu = nn.ReLU(inplace=True)
#         self.batch_norm = nn.BatchNorm3d(out_channels)
#         self.dropout = nn.Dropout3d(p=dropout_prob)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.batch_norm(x)
#         x = self.dropout(x)
#         return x


class ConvBlock3D_won(nn.Module):
    """Convolution Block with Conv3d, BatchNorm, ReLU, and Dropout"""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock3D_won, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        return x

class AttentionScalingWithHeads(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(AttentionScalingWithHeads, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for query projection
        self.query_proj = nn.Linear(embed_dim // 2, embed_dim)
        self.query_proj_up = nn.Linear(embed_dim, embed_dim)
        self.query_proj_up_cat = nn.Linear(embed_dim * 2, embed_dim)
        self.softmax = nn.Softmax(dim=2)  # Normalize across sequence length
        self.flatten = nn.Flatten(start_dim=2)
    def forward(self, x, x_up):
        b, c, h, w, d = x_up.shape
        x = self.flatten(x)
        x_up = self.flatten(x_up)
        x_up = x_up.permute(0, 2, 1) 
        print(f"x_up output shape: {x_up.shape}")
        query_up = self.query_proj_up(x_up)  # (batch_size, seq_len, embed_dim)
        print(f"query_up output shape: {query_up.shape}")
        query_up = query_up.permute(0, 2, 1) 
        batch_size_up, embed_dim_up, seq_len_up = query_up.size()
        # query_up = query_up.view(batch_size_up, embed_dim_up, seq_len_up)  # (batch_size, seq_len, num_heads, head_dim)
        # x = rearrange(x, 'b c h w d -> b (h w d) c', h=h, w=w, d=d)
        
        
        

        # Project to multi-head space
        x = x.permute(0, 2, 1) 
        print("size of query ", x.shape)
        query = self.query_proj(x)  # (batch_size, seq_len, embed_dim)
        query = query.permute(0, 2, 1)
        print("size of query ", query.shape)
        batch_size, embed_dim_pre, seq_len_pre = query.size()
        query = query.view(batch_size, embed_dim_pre, seq_len_pre)  
        # Compute attention scores
        # attention_scores = torch.mean(query, dim=-1)  # Average over head_dim
        attention_scores = self.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, num_heads)

        # Reshape to apply scaling
        # attention_scores = attention_scores.view(batch_size, embed_dim_pre, seq_len_pre, 1)
        print("size of liner ", query_up.shape)
        print("size of liner ", attention_scores.shape)
        scaled_x = attention_scores * query_up
        # scaled_x = torch.cat((query_up, attention_scores), dim=1)
        # scaled_x = scaled_x.permute(0, 2, 1)
        # scaled_x = self.query_proj_up_cat(scaled_x)
        print("size of scaled_x ", scaled_x.shape)
        scaled_x = rearrange(scaled_x, 'b (h w d) c -> b c h w d', h=h, w=w, d=d)
        # Flatten back to original embedding dimension
        # scaled_x = scaled_x.view(batch_size, seq_len, embed_dim)
        return scaled_x

class Encoder3D(nn.Module):
    """Encoder consisting of multiple convolution blocks with increasing feature maps"""
    def __init__(self, in_channels, dropout_prob=0.5):
        super(Encoder3D, self).__init__()
        
        self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
        # self.res1 = ResidualBlock(8)
        self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
        # self.res2 = ResidualBlock(16)
        self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
        # self.attention_scaling3 = AttentionScalingWithHeads(32, 8)
        # self.res3 = ResidualBlock(32)
        self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
        # self.attention_scaling4 = AttentionScalingWithHeads(64, 8)
        # self.res4 = ResidualBlock(64)
        self.encoder5 = ConvBlock3D(64, 128, dropout_prob)
        # self.encoder5_pre = ConvBlock3D_wopad(64, 64, dropout_prob)
        # self.attention_scaling5 = AttentionScalingWithHeads(128, 8)
        # self.res5 = ResidualBlock(128)
        self.encoder6 = ConvBlock3D(128, 128, dropout_prob)
        
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        x1 = self.encoder1(x)
        print(f"Encoder1 output shape: {x1.shape}")
        # x1 = self.res1(x1)
        x2 = self.encoder2(self.pool(x1))
        print(f"Encoder2 output shape: {x2.shape}")
        # x2 = self.res2(x2)
        x3 = self.encoder3(self.pool(x2))
        print(f"Encoder3 output shape: {x3.shape}")
        # x3 = self.res3(x3)
        x4 = self.encoder4((x3))
        # x4 = self.attention_scaling4(x3, x4)
        print(f"Encoder4 output shape: {x4.shape}")
        # x4 = self.res4(x4)
        # x4 = self.encoder5_pre(x4)
        # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
        # x4 = F.pad(x4, padding, mode='constant', value=0)
        print(f"Encoder4 output shape: {x4.shape}")
        x5 = self.encoder5((x4))
        print(f"Encoder5 output shape: {x5.shape}")
        # x5 = self.attention_scaling5(x4, x5)
        print(f"Encoder5 output shape: {x5.shape}")
        # x5 = self.res5(x5)
        x5 = self.encoder6(x5)
        return x5


class BottleneckBlock(nn.Module):
    """Bottleneck block with 128 to 128 features"""
    def __init__(self, in_channels, dropout_prob=0.3):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = ConvBlock3D(in_channels, in_channels, dropout_prob)
        
    def forward(self, x):
        x = self.bottleneck(x)
        print(f"Bottleneck output shape: {x.shape}")
        return x


class Decoder3D(nn.Module):
    """Decoder with skip connections and upsampling"""
    def __init__(self, dropout_prob=0.5):
        super(Decoder3D, self).__init__()
        self.res1 = ConvBlock3D(128, 128, dropout_prob)
        
        self.upsample1 = self.upsample_block1(128, dropout_prob)
        # self.res2 = ResidualBlock(64)
        # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.upsample2 = self.upsample_block1(64, dropout_prob)
        # self.res3 = ResidualBlock(32)
        # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.upsample3 = self.upsample_block(32, dropout_prob)
        # self.res4 = ResidualBlock(16)
        # self.conv3 = nn.Conv3d(32,16, kernel_size=3, padding=1)
        self.upsample4 = self.upsample_block(16, dropout_prob)
        # self.res5 = ResidualBlock(8)
        # self.conv4 = nn.Conv3d(16,8, kernel_size=3, padding=1)
        
        self.final_conv = nn.Conv3d(8, 4, kernel_size=3, padding=1)  # Assuming segmentation output is single channel
        # self.convvv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        # self.convvv2 = nn.Conv3d(1, 1, kernel_size=1)
    
    def upsample_block(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)
    def upsample_block1(self, in_channels, dropout_prob):
        """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
        layers = [
            nn.Upsample(scale_factor=1, mode='nearest'),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels // 2),
            nn.Dropout3d(p=dropout_prob),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.res1(x)
        print(f"Decoder input (x): {x.shape}")
        x6 = self.upsample1(x)  # First decoder layer
        print(f"Upsample1 output shape: {x6.shape}")
        # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        # x6 = F.pad(x6, padding, mode='constant', value=0)
        # print(f"Upsample3 output shape: {x6.shape}")
        # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
        # print(f"Conv1 output shape after concatenation: {x7.shape}")
        # x6 = self.res2(x6)
        x7 = self.upsample2(x6)
        print(f"Upsample2 output shape: {x7.shape}")
        
        # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
        # print(f"Conv2 output shape after concatenation: {x8.shape}")
        # x7 = self.res3(x7)
        x8 = self.upsample3(x7)
        print(f"Upsample3 output shape: {x8.shape}")
        padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        x8 = F.pad(x8, padding, mode='constant', value=0)
        # print(f"Upsample3 output shape: {x8.shape}")

        # x6 = self.res2(x6)

        x9 = self.upsample4(x8)
        print(f"Upsample3 output shape: {x9.shape}")
        padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
        x9 = F.pad(x9, padding, mode='constant', value=0)
        print(f"Upsample3 output shape: {x9.shape}")
        # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
        # print(f"Conv3 output shape after concatenation: {x9.shape}")
        out = self.final_conv(x9)  # Final output
        print(f"Final output shape: {out.shape}")
        # out=self.convvv1(out)
        # out=self.convvv2(out)
        return out

#ET Only
# class Encoder3D(nn.Module):
#     """Encoder consisting of multiple convolution blocks with increasing feature maps"""
#     def __init__(self, in_channels, dropout_prob=0.5):
#         super(Encoder3D, self).__init__()
        
#         self.encoder1 = ConvBlock3D(in_channels, 8, dropout_prob)
#         # self.res1 = ResidualBlock(8)
#         self.encoder2 = ConvBlock3D(8, 16, dropout_prob)
#         # self.res2 = ResidualBlock(16)
#         self.encoder3 = ConvBlock3D(16, 32, dropout_prob)
#         # self.attention_scaling3 = AttentionScalingWithHeads(32, 8)
#         # self.res3 = ResidualBlock(32)
#         # self.encoder4 = ConvBlock3D(32, 64, dropout_prob)
#         # # self.attention_scaling4 = AttentionScalingWithHeads(64, 8)
#         # # self.res4 = ResidualBlock(64)
#         # self.encoder5 = ConvBlock3D(64, 128, dropout_prob)
#         # # self.encoder5_pre = ConvBlock3D_wopad(64, 64, dropout_prob)
#         # # self.attention_scaling5 = AttentionScalingWithHeads(128, 8)
#         # # self.res5 = ResidualBlock(128)
#         # self.encoder6 = ConvBlock3D(128, 128, dropout_prob)
        
#         self.pool = nn.MaxPool3d(2)
        
#     def forward(self, x):
#         x1 = self.encoder1(x)
#         print(f"Encoder1 output shape: {x1.shape}")
#         # x1 = self.res1(x1)
#         x2 = self.encoder2(self.pool(x1))
#         print(f"Encoder2 output shape: {x2.shape}")
#         # x2 = self.res2(x2)
#         x3 = self.encoder3((x2))
#         print(f"Encoder3 output shape: {x3.shape}")
#         # x3 = self.res3(x3)
#         # x4 = self.encoder4((x3))
#         # # x4 = self.attention_scaling4(x3, x4)
#         # print(f"Encoder4 output shape: {x4.shape}")
#         # # x4 = self.res4(x4)
#         # # x4 = self.encoder5_pre(x4)
#         # # padding = (0, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back)
#         # # x4 = F.pad(x4, padding, mode='constant', value=0)
#         # print(f"Encoder4 output shape: {x4.shape}")
#         # x5 = self.encoder5((x4))
#         # print(f"Encoder5 output shape: {x5.shape}")
#         # # x5 = self.attention_scaling5(x4, x5)
#         # print(f"Encoder5 output shape: {x5.shape}")
#         # # x5 = self.res5(x5)
#         # x5 = self.encoder6(x5)
#         return x3


# class BottleneckBlock(nn.Module):
#     """Bottleneck block with 128 to 128 features"""
#     def __init__(self, in_channels, dropout_prob=0.3):
#         super(BottleneckBlock, self).__init__()
#         self.bottleneck = ConvBlock3D(in_channels, in_channels, dropout_prob)
        
#     def forward(self, x):
#         x = self.bottleneck(x)
#         print(f"Bottleneck output shape: {x.shape}")
#         return x


# class Decoder3D(nn.Module):
#     """Decoder with skip connections and upsampling"""
#     def __init__(self, dropout_prob=0.5):
#         super(Decoder3D, self).__init__()
#         self.res1 = ConvBlock3D(32, 32, dropout_prob)
        
#         self.upsample1 = self.upsample_block1(32, dropout_prob)
#         # self.res2 = ResidualBlock(64)
#         # self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         self.upsample2 = self.upsample_block(16, dropout_prob)
#         # self.res3 = ResidualBlock(32)
#         # self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
#         # self.upsample3 = self.upsample_block(32, dropout_prob)
#         # # self.res4 = ResidualBlock(16)
#         # # self.conv3 = nn.Conv3d(32,16, kernel_size=3, padding=1)
#         # self.upsample4 = self.upsample_block(16, dropout_prob)
#         # self.res5 = ResidualBlock(8)
#         # self.conv4 = nn.Conv3d(16,8, kernel_size=3, padding=1)
        
#         self.final_conv = nn.Conv3d(8, 4, kernel_size=3, padding=1)  # Assuming segmentation output is single channel
#         # self.convvv1 = nn.Conv3d(1, 1, kernel_size=3, padding=1)
#         # self.convvv2 = nn.Conv3d(1, 1, kernel_size=1)
    
#     def upsample_block(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)
#     def upsample_block1(self, in_channels, dropout_prob):
#         """Create an upsampling block with Conv3d, ReLU, BatchNorm, and Dropout"""
#         layers = [
#             nn.Upsample(scale_factor=1, mode='nearest'),
#             nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm3d(in_channels // 2),
#             nn.Dropout3d(p=dropout_prob),
#         ]
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.res1(x)
#         print(f"Decoder input (x): {x.shape}")
#         x6 = self.upsample1(x)  # First decoder layer
#         print(f"Upsample1 output shape: {x6.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x6 = F.pad(x6, padding, mode='constant', value=0)
#         # print(f"Upsample3 output shape: {x6.shape}")
#         # x7 = self.conv1(torch.cat([x6, x3], dim=1))  # Concatenate with encoder3
#         # print(f"Conv1 output shape after concatenation: {x7.shape}")
#         # x6 = self.res2(x6)
#         x7 = self.upsample2(x6)
#         print(f"Upsample2 output shape: {x7.shape}")
        
#         # x8 = self.conv2(torch.cat([x7, x2], dim=1))  # Concatenate with encoder2
#         # print(f"Conv2 output shape after concatenation: {x8.shape}")
#         # x7 = self.res3(x7)
#         # x8 = self.upsample3(x7)
#         # print(f"Upsample3 output shape: {x8.shape}")
#         # padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         # x8 = F.pad(x8, padding, mode='constant', value=0)
#         # # print(f"Upsample3 output shape: {x8.shape}")

#         # # x6 = self.res2(x6)

#         # x9 = self.upsample4(x8)
#         # print(f"Upsample3 output shape: {x9.shape}")
#         padding = (0, 1, 0, 0, 0, 0)  # (left, right, top, bottom, front, back)
#         x9 = F.pad(x7, padding, mode='constant', value=0)
#         print(f"Upsample3 output shape: {x9.shape}")
#         # x9 = self.conv3(torch.cat([x8, x1], dim=1))  # Concatenate with encoder1
#         # print(f"Conv3 output shape after concatenation: {x9.shape}")
#         out = self.final_conv(x9)  # Final output
#         print(f"Final output shape: {out.shape}")
#         # out=self.convvv1(out)
#         # out=self.convvv2(out)
#         return out

class SegmentationModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super(SegmentationModel, self).__init__()
        self.conv1 = nn.Conv3d(4, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)  # Output channels equal to num_classes
        # self.conv4_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
        # self.conv5_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
        # self.conv6_op = nn.Conv3d(4, 4, kernel_size=1)  # Output channels equal to num_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        print("X shape  before is", x.shape)
        # if x.shape[4] <= 100:
        #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
        # else:
        #     x = F.pad(x, (1, 0, 0, 1, 1, 0), mode='constant', value=0)
        x = self.conv1(x)
        segmentation_mask1 = self.conv2(x)
        # print("X shape after is", segmentation_mask1.shape)
        
        
        # # Compute softmax probabilities over classes
        output_probabilities = F.softmax(segmentation_mask1, dim=1)
        return output_probabilities





class VQVAE_seq_TC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float()):
        super(VQVAE_seq_TC, self).__init__()

        self.dropout_prob = dropout_prob  # Dropout probability

        # Initialize Encoder, Bottleneck, and Decoder as separate modules
        self.encoder = Encoder3D(in_channels, dropout_prob)
        self.bottleneck = BottleneckBlock(128, dropout_prob)
        self.decoder = Decoder3D(dropout_prob)
        self.segmentation=SegmentationModel(4, 4, 4)
        self.quantizer0 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=32,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
            )
        )
        self.quantizer1 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=32,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
            )
        )
        self.quantizer2 = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=512,
                embedding_dim=32,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=1e-5,
                embedding_init='uniform',
                ddp_sync=False,
            )
        )
        self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        # # self.conv3 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        # # self.conv4 = ConvBlock3D(16, 32, dropout_prob)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
    def forward(self, x):
        # Encoder path
        x4 = self.encoder(x)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        x5 = self.conv1(x5)
        x5 = self.conv2(x5)
        # x5 = self.conv3(x5)
        quantization_loss0, z_quantized0, encodings_sum0, embedding0, encoding_indices = self.quantizer0(x5)

        # z_quantized0_post = self.conv4(z_quantized0)
        z_quantized0_post = self.conv3(z_quantized0)
        z_quantized0_post = self.conv4(z_quantized0_post)

        # Decoder path with skip connections
        reconstruction = self.decoder(z_quantized0_post)
        segmentation_mask = self.segmentation(reconstruction)
        
       
        
        
        print("segmentation_mask", reconstruction.shape)

        
        total_quantization_loss = torch.mean(quantization_loss0)
        print("total_quantization_loss2222222222222222", (total_quantization_loss))
# #        

        return z_quantized0, segmentation_mask, total_quantization_loss, encodings_sum0, embedding0























