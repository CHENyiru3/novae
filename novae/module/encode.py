from typing import Literal

import lightning as L
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn.models import GAT

from . import AttentionAggregation


class GraphEncoder(L.LightningModule):
    """Graph encoder of Novae with intrinsic and niche node embeddings."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        heads: int,
        histo_embedding_size: int,
        use_attention_pooling: bool = True,
    ) -> None:
        """
        Args:
            embedding_size: Size of the embeddings of the genes (`E` in the article).
            hidden_size: The size of the hidden layers in the GAT.
            num_layers: The number of layers in the GAT.
            output_size: Size of the representations, i.e. the encoder outputs (`O` in the article).
            heads: The number of attention heads in the GAT.
        """
        super().__init__()
        self.intrinsic_encoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.context_encoder = GAT(
            embedding_size,
            hidden_channels=hidden_size,
            num_layers=num_layers,
            out_channels=output_size,
            edge_dim=1,
            v2=True,
            heads=heads,
            act="ELU",
        )

        self.node_aggregation = AttentionAggregation(output_size) if use_attention_pooling else None

        self.mlp_fusion = nn.Sequential(
            nn.Linear(histo_embedding_size + output_size, histo_embedding_size + output_size),
            nn.ReLU(),
            nn.Linear(histo_embedding_size + output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, data: Batch) -> tuple[Tensor, Tensor]:
        """Encode the input data into intrinsic and niche node embeddings.

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.

        Returns:
            A tuple `(u, v)` of tensors with shape `(N, O)`, containing the intrinsic
            and niche node embeddings for each node in the batch.
        """
        intrinsic = self.intrinsic_encoder(data.x)
        niche = self.context_encoder(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)

        return intrinsic, niche

    def encode_graph(self, data: Batch, combine: Literal["intrinsic", "niche", "mean"] = "niche") -> Tensor:
        """Encode the input data into a graph-level representation.

        Args:
            data: A Pytorch Geometric `Data` object representing a batch of `B` graphs.
            combine: Strategy to combine intrinsic and niche embeddings before pooling.

        Returns:
            A tensor of shape `(B, O)` containing the encoded graphs.
        """
        if self.node_aggregation is None:
            raise RuntimeError("Graph-level pooling is disabled for this GraphEncoder instance.")

        intrinsic, niche = self.forward(data)

        if combine == "intrinsic":
            node_embeddings = intrinsic
        elif combine == "mean":
            node_embeddings = (intrinsic + niche) / 2
        else:
            node_embeddings = niche

        out = self.node_aggregation(node_embeddings, index=data.batch)

        if hasattr(data, "histo_embeddings"):
            out = self.mlp_fusion(torch.cat([out, data.histo_embeddings], dim=-1))

        return out
