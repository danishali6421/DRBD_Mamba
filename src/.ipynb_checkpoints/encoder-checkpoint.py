import torch
from torch import nn
from torch.nn import functional as F

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

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_chan)
        self.conv_1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        # self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_chan == out_chan:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv3d(in_chan, out_chan, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.groupnorm_1(x)
        x = F.relu(x)
        x = self.conv_1(x)
        # x = self.groupnorm_2(x)
        # x = F.silu(x)
        # x = self.conv_2(x)
        return x + self.residual_layer(residue)

class VAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            VAE_ResidualBlock(32, 32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock(64, 64),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock(128, 128),
            nn.Conv3d(128, out_channels, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock(256, out_channels),
        )
        self.fc_mu = nn.Linear(256 * 16 * 16 * 16, 128)  # Adjust latent dimension
        self.fc_logvar = nn.Linear(256 * 16 * 16 * 16, 128)  # Adjust latent 
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        #logvar = torch.clamp(logvar, -30, 20)
        #print(logvar)
        return mu, logvar

