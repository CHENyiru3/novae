import lightning as L
import numpy as np
from torch import Tensor, nn

from .. import utils
from .swav import SwavHead


class DualSwavHead(L.LightningModule):
    """Dual SwAV head with independent codebooks and projection heads."""

    def __init__(
        self,
        mode: utils.Mode,
        output_size: int,
        num_prototypes_c: int,
        num_prototypes_n: int,
        temperature_c: float,
        temperature_n: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.output_size = output_size

        self.projection_head_c = _projection_head(output_size)
        self.projection_head_n = _projection_head(output_size)

        self.swav_c = SwavHead(mode, output_size, num_prototypes_c, temperature_c)
        self.swav_n = SwavHead(mode, output_size, num_prototypes_n, temperature_n)

    def forward(self, z1: Tensor, z2: Tensor, slide_id: str | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute SwAV losses for intrinsic (C) and niche (N) embeddings."""
        u1, u2 = self.projection_head_c(z1), self.projection_head_c(z2)
        v1, v2 = self.projection_head_n(z1), self.projection_head_n(z2)

        loss_c, entropy_c = self.swav_c.forward(u1, u2, slide_id)
        loss_n, entropy_n = self.swav_n.forward(v1, v2, slide_id)

        return loss_c, loss_n, entropy_c, entropy_n

    def projection(self, z: Tensor) -> Tensor:
        return self.swav_c.projection(self.projection_head_c(z))

    def projection_n(self, z: Tensor) -> Tensor:
        return self.swav_n.projection(self.projection_head_n(z))

    def set_min_prototypes(self, min_prototypes_ratio: float) -> None:
        self.swav_c.set_min_prototypes(min_prototypes_ratio)
        self.swav_n.set_min_prototypes(min_prototypes_ratio)

    def init_queue(self, slide_ids: list[str]) -> None:
        self.swav_c.init_queue(slide_ids)
        self.swav_n.init_queue(slide_ids)

    def set_prototypes_requires_grad(self, value: bool) -> None:
        self.swav_c.prototypes.requires_grad_(value)
        self.swav_n.prototypes.requires_grad_(value)

    def init_prototypes(self, latent: np.ndarray) -> None:
        prototypes = self.swav_c.compute_kmeans_prototypes(latent)
        self.swav_c.set_trainable_prototypes(prototypes)
        self.swav_n.set_trainable_prototypes(nn.Parameter(prototypes.clone().detach()))

    def set_kmeans_prototypes(self, latent: np.ndarray) -> None:
        prototypes = self.swav_c.compute_kmeans_prototypes(latent)
        self.swav_c.set_kmeans_prototypes(prototypes)
        self.swav_n.set_kmeans_prototypes(nn.Parameter(prototypes.clone().detach()))

    def reset_clustering(self, only_zero_shot: bool = False) -> None:
        self.swav_c.reset_clustering(only_zero_shot=only_zero_shot)

    @property
    def prototypes(self) -> nn.Parameter:
        return self.swav_c.prototypes

    @property
    def queue(self):
        return self.swav_c.queue

    @property
    def min_prototypes(self) -> int:
        return self.swav_c.min_prototypes

    @property
    def slide_label_encoder(self) -> dict[str, int]:
        return self.swav_c.slide_label_encoder

    @property
    def clustering(self):
        return self.swav_c.clustering

    @property
    def clusters_levels(self):
        return self.swav_c.clusters_levels

    def queue_weights(self):
        return self.swav_c.queue_weights()

    def map_leaves_domains(self, series, level: int):
        return self.swav_c.map_leaves_domains(series, level)

    def find_level(self, leaves_indices: np.ndarray, n_domains: int):
        return self.swav_c.find_level(leaves_indices, n_domains)


def _projection_head(output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(output_size, output_size),
        nn.ReLU(),
        nn.Linear(output_size, output_size),
    )
