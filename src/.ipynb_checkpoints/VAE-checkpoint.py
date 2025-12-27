import torch
from torch import nn
from torch.nn import functional as F
from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder

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
# def initialize_weights(m):
#     if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm3d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)





class VAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = VAE_Encoder(self.in_channels, self.out_channels)
        self.decoder = VAE_Decoder(self.out_channels)
        # self.encoder.apply(initialize_weights)
        # self.decoder.apply(initialize_weights)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        # print("mu", mu)
        # print("logvar", logvar)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
