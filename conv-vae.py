import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from experiment_config import ConvVAEConfig, LossConfig
from models import ConvVAE


def kl_divergence(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(z_mean.pow(2) + z_log_var.exp() - z_log_var - 1)


def reconstruction_loss(
    reconstruction: torch.Tensor, target: torch.Tensor, loss_config: LossConfig
) -> torch.Tensor:
    if loss_config.reconstruction_loss != "bce":
        raise ValueError(f"Unsupported reconstruction loss: {loss_config.reconstruction_loss}")
    if loss_config.reduction != "sum":
        raise ValueError(f"Unsupported reduction: {loss_config.reduction}")
    return F.binary_cross_entropy(reconstruction, target, reduction=loss_config.reduction)


def make_loaders(config: ConvVAEConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    transform = transforms.ToTensor()

    full_training_set = datasets.CIFAR10(
        root=config.data_root,
        train=True,
        transform=transform,
        download=True,
    )
    training_indices, val_indices = torch.utils.data.random_split(
        range(len(full_training_set)),
        [45000, 5000],
        generator=generator,
    )

    train_set = Subset(full_training_set, training_indices.indices)
    val_set = Subset(full_training_set, val_indices.indices)
    test_set = datasets.CIFAR10(
        root=config.data_root,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_epoch(
    model: ConvVAE,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_config: LossConfig,
    kl_weight: float,
) -> tuple[float, float, float]:
    model.train()
    total_recon = 0.0
    total_kl = 0.0
    total_images = 0

    for features, _ in data_loader:
        features = features.to(device)
        z_mean, z_log_var, _, reconstruction = model(features)

        recon = reconstruction_loss(reconstruction, features, loss_config)
        kl = kl_divergence(z_mean, z_log_var)
        loss = (recon + kl_weight * kl) / features.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_recon += recon.item()
        total_kl += kl.item()
        total_images += features.size(0)

    avg_recon = total_recon / total_images
    avg_kl = total_kl / total_images
    return avg_recon, avg_kl, avg_recon + kl_weight * avg_kl


def evaluate(
    model: ConvVAE,
    data_loader: DataLoader,
    device: torch.device,
    loss_config: LossConfig,
    kl_weight: float,
) -> tuple[float, float, float]:
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    total_images = 0

    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            z_mean, z_log_var, _, reconstruction = model(features)

            recon = reconstruction_loss(reconstruction, features, loss_config)
            kl = kl_divergence(z_mean, z_log_var)

            total_recon += recon.item()
            total_kl += kl.item()
            total_images += features.size(0)

    avg_recon = total_recon / total_images
    avg_kl = total_kl / total_images
    return avg_recon, avg_kl, avg_recon + kl_weight * avg_kl


def save_final_reconstructions(
    model: ConvVAE,
    data_loader: DataLoader,
    device: torch.device,
    output_path: Path,
    num_images: int = 5,
) -> None:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_batch, _ = next(iter(data_loader))
    sample_batch = sample_batch[:num_images].to(device)

    with torch.no_grad():
        _, _, _, reconstructions = model(sample_batch)

    comparison = torch.cat([sample_batch.cpu(), reconstructions.cpu()], dim=0)
    grid = make_grid(comparison, nrow=num_images)
    save_image(grid, output_path)


def main() -> None:
    config = ConvVAEConfig()
    loss_config = LossConfig()

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = make_loaders(config)

    model = ConvVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_time = time.time()

    for epoch in range(config.num_epochs):
        if loss_config.use_kl_annealing:
            kl_weight = min(1.0, (epoch + 1) / loss_config.anneal_epochs)
        else:
            kl_weight = 1.0

        train_recon, train_kl, train_total = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_config=loss_config,
            kl_weight=kl_weight,
        )
        val_recon, val_kl, val_total = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            loss_config=loss_config,
            kl_weight=kl_weight,
        )

        print(
            f"epoch={epoch + 1:03d} "
            f"train_total={train_total:.4f} "
            f"train_recon={train_recon:.4f} "
            f"train_kl={train_kl:.4f} "
            f"val_total={val_total:.4f} "
            f"val_recon={val_recon:.4f} "
            f"val_kl={val_kl:.4f} "
            f"kl_weight={kl_weight:.4f}"
        )

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"training_complete elapsed_minutes={elapsed_minutes:.2f}")

    save_final_reconstructions(
        model=model,
        data_loader=val_loader,
        device=device,
        output_path=Path("artifacts") / "conv-vae-final-recon.png",
    )
    print("saved_reconstructions path=artifacts/conv-vae-final-recon.png")


if __name__ == "__main__":
    main()
