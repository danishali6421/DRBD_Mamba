import torch
from torch import nn
from torch.nn import functional as F
from src.encoder import VAE_ResidualBlock

# class VAE_AttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.groupnorm = nn.GroupNorm(32, channels)
#         self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=1)  # Adjust as necessary
    
#     def forward(self, x):
#         residue = x
#         n, c, d, h, w = x.shape
#         x = self.groupnorm(x)
#         x = x.view(n, c, d * h * w)
#         x = x.transpose(1, 2)
#         x, _ = self.attention(x, x, x)
#         x = x.transpose(1, 2)
#         x = x.view(n, c, d, h, w)
#         x += residue
#         return x

class VAE_Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Linear(128, in_channels * 16 * 16 * 16)  # Adjust latent dimension
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            VAE_ResidualBlock(128, 128),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            VAE_ResidualBlock(64, 64),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            VAE_ResidualBlock(32, 32),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 16, 16, 16)  # Reshape to match input size
        x = self.model(x)
        return x
