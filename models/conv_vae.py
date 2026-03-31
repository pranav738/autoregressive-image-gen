import torch
from torch import nn

from experiment_config import ConvVAEConfig


class ConvVAE(nn.Module):
    def __init__(self, config: ConvVAEConfig):
        super().__init__()

        c1, c2, c3 = config.channels
        slope = config.negative_slope

        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )

        self.z_mean = nn.Linear(c3 * 4 * 4, config.latent_dim)
        self.z_log_var = nn.Linear(c3 * 4 * 4, config.latent_dim)

        self.decoder_input = nn.Linear(config.latent_dim, c3 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.ConvTranspose2d(c1, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.final_channels = c3

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_mean)
        return z_mean + eps * torch.exp(0.5 * z_log_var)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(x.size(0), self.final_channels, 4, 4)
        return self.decoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encode(x)
        reconstruction = self.decode(z)
        return z_mean, z_log_var, z, reconstruction
