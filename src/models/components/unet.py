import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.swish = nn.SiLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity  # Skip connection
        return self.swish(out)


# UNet with advanced techniques
class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_timesteps, embedding_dim=2, multiplier=4, is_warm_init=False):
        super(UNet, self).__init__()
        self.num_timesteps = num_timesteps

        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(multiplier*in_dim + embedding_dim, multiplier*in_dim),
            nn.SiLU(),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            ResidualBlock(multiplier*in_dim, multiplier*in_dim),
            nn.Dropout(0.1),
            nn.Linear(multiplier*in_dim, out_dim)
        )

        # Define time step embedding layer for decoder
        self.embedding = nn.Embedding(num_timesteps, embedding_dim)

        if is_warm_init:
            self.warm_init()

    def forward(self, x, timestep, enc_sigma=None):
        # Encoder
        if enc_sigma is not None:
            encoded_enc_sigma = self.encoder(enc_sigma)
        else:
            encoded_enc_sigma = 0
        x = self.encoder(x) + encoded_enc_sigma

        # Decoder
        x = self.decoder(torch.hstack((x, self.embedding(timestep))))

        return x

    def warm_init(self):
        # Custom initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)