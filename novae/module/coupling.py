import torch
import torch.nn.functional as F
from torch import Tensor

from .._constants import Nums


def coupling_loss(
    niche_embeddings: Tensor,
    intrinsic_embeddings: Tensor,
    anchor_indices: Tensor,
    neighbor_indices: Tensor,
    distances: Tensor,
    *,
    sigma: float,
    bidirectional: bool = False,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if anchor_indices.numel() == 0:
        zero = niche_embeddings.new_tensor(0.0)
        return zero, {}

    niche_norm = F.normalize(niche_embeddings, dim=1)
    intrinsic_norm = F.normalize(intrinsic_embeddings, dim=1)

    logits = (niche_norm @ intrinsic_norm.T) / max(temperature, Nums.EPS)
    log_probs = F.log_softmax(logits, dim=1)

    weights = torch.exp(-0.5 * (distances / max(sigma, Nums.EPS)) ** 2)
    pos_log_probs = log_probs[anchor_indices, neighbor_indices]

    loss_n_to_c = _weighted_positive_loss(
        anchor_indices,
        pos_log_probs,
        weights,
        niche_embeddings.size(0),
    )

    metrics = {
        "couple/pos_similarity": (niche_norm[anchor_indices] * intrinsic_norm[neighbor_indices]).sum(dim=1).mean()
    }

    if bidirectional:
        log_probs_t = F.log_softmax(logits.T, dim=1)
        pos_log_probs_t = log_probs_t[neighbor_indices, anchor_indices]
        loss_c_to_n = _weighted_positive_loss(
            neighbor_indices,
            pos_log_probs_t,
            weights,
            intrinsic_embeddings.size(0),
        )
        return loss_n_to_c + loss_c_to_n, metrics

    return loss_n_to_c, metrics


def _weighted_positive_loss(indices: Tensor, pos_log_probs: Tensor, weights: Tensor, dim_size: int) -> Tensor:
    loss_num = torch.zeros(dim_size, device=pos_log_probs.device)
    weight_sum = torch.zeros(dim_size, device=pos_log_probs.device)
    loss_num.index_add_(0, indices, -weights * pos_log_probs)
    weight_sum.index_add_(0, indices, weights)

    valid = weight_sum > 0
    if not valid.any():
        return pos_log_probs.new_tensor(0.0)
    return (loss_num[valid] / weight_sum[valid]).mean()
