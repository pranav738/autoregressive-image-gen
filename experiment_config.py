from dataclasses import dataclass, field


@dataclass
class BaseExperimentConfig:
    seed: int = 123
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 5e-4
    data_root: str = "../../data"


@dataclass
class LossConfig:
    reconstruction_loss: str = "bce"
    reduction: str = "sum"
    use_kl_annealing: bool = False
    anneal_epochs: int = 20


@dataclass
class ConvVAEConfig(BaseExperimentConfig):
    latent_dim: int = 128
    channels: tuple[int, int, int] = field(default_factory=lambda: (32, 64, 128))
    negative_slope: float = 1e-4

