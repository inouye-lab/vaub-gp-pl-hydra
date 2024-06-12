import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)  # Output: 32x14x14
        self.conv2 = nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1)  # Output: 64x7x7
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)  # Output: 128x4x4
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2_mu = nn.Linear(256, 1 * 16 * 16)
        self.fc2_logvar = nn.Linear(256, 1 * 16 * 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mu = self.fc2_mu(x)
        logvar = self.fc2_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 64 * 7 * 7)  # Adjusted to match the size before reshaping
        self.deconv1 = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(56 * 56, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 64, 7, 7)  # Reshape to match the size before upsampling
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x)).view((x.shape[0], -1))
        x = torch.sigmoid(self.fc1(x))
        return x


class CNN_VAE(nn.Module):
    def __init__(self):
        super(CNN_VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        # print(f'recon: {recon.shape}')
        return recon.view((z.shape[0], -1)), z.view((z.shape[0], -1)), mu.view((z.shape[0], -1)), logvar.view(
            (z.shape[0], -1))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Sigmoid(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), z, mean, logvar

