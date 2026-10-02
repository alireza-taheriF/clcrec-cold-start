"""Innovation 5: End-to-End Joint Collaborative-Contrastive Model.

Simultaneously optimizes:
1. Collaborative BPR Loss on user-item implicit interactions.
2. InfoNCE Contrastive Loss aligning content representations with collaborative embeddings.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from recsys.models.torch_encoder import TorchContentEncoder, info_nce_loss, get_device


if HAS_TORCH:
    class JointCLCRec(nn.Module):
        """End-to-end joint model coupling BPR collaborative loss with InfoNCE contrastive alignment."""

        def __init__(
            self,
            n_users: int,
            n_items: int,
            content_dim: int,
            emb_dim: int = 32,
            hidden_dim: int = 64,
            tau: float = 0.1,
            init_collab_emb: Optional[np.ndarray] = None,
        ):
            super().__init__()
            self.n_users = n_users
            self.n_items = n_items
            self.emb_dim = emb_dim

            # Trainable user and item collaborative embeddings
            self.user_emb = nn.Embedding(n_users, emb_dim)
            self.item_collab_emb = nn.Embedding(n_items, emb_dim)

            # Initialize with small random or SVD warm start
            nn.init.xavier_uniform_(self.user_emb.weight)
            if init_collab_emb is not None:
                self.item_collab_emb.weight.data.copy_(
                    torch.from_numpy(init_collab_emb.astype(np.float32))
                )
            else:
                nn.init.xavier_uniform_(self.item_collab_emb.weight)

            # Content encoder
            self.content_encoder = TorchContentEncoder(
                content_dim=content_dim,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                tau=tau,
            )

        def forward_bpr(
            self, u: torch.Tensor, i: torch.Tensor, j: torch.Tensor
        ) -> torch.Tensor:
            """Bayesian Personalized Ranking (BPR) loss."""
            u_vec = F.normalize(self.user_emb(u), p=2, dim=1)
            i_vec = F.normalize(self.item_collab_emb(i), p=2, dim=1)
            j_vec = F.normalize(self.item_collab_emb(j), p=2, dim=1)

            x_ui = (u_vec * i_vec).sum(dim=1)
            x_uj = (u_vec * j_vec).sum(dim=1)
            loss_bpr = -F.logsigmoid(x_ui - x_uj).mean()
            return loss_bpr

        def forward_contrastive(
            self,
            item_indices: torch.Tensor,
            content_x: torch.Tensor,
            hard_negatives: bool = False,
            hard_weight: float = 0.5,
        ) -> torch.Tensor:
            """InfoNCE contrastive loss between content encoder and collaborative embeddings."""
            z_content = self.content_encoder(content_x)
            z_collab = F.normalize(self.item_collab_emb(item_indices), p=2, dim=1)

            return info_nce_loss(
                z_content,
                z_collab,
                self.content_encoder.tau,
                hard_negatives=hard_negatives,
                hard_weight=hard_weight,
                content_features=content_x if hard_negatives else None,
            )

        def encode_items(self, content_matrix: np.ndarray, device: torch.device) -> np.ndarray:
            """Encode all items using content features for inference / evaluation."""
            self.content_encoder.eval()
            out = []
            bs = 512
            with torch.no_grad():
                for start in range(0, len(content_matrix), bs):
                    chunk = content_matrix[start:start + bs]
                    x = torch.as_tensor(np.asarray(chunk, dtype=np.float32), device=device)
                    out.append(self.content_encoder(x).cpu().numpy())
            return np.vstack(out).astype(np.float32)


class JointCLCRecRecommender:
    """Recommender adapter implementing the joint training pipeline."""

    backend = "joint"

    def __init__(self, config):
        self.config = config
        self.device = get_device(getattr(config, "device", "auto"))
        self.model: Optional[JointCLCRec] = None
        self.item_emb: Optional[np.ndarray] = None
        self.history: List[float] = []
        self._data = None

    def fit(self, data) -> "JointCLCRecRecommender":
        from recsys.training.trainer_joint import train_joint
        self.model, self.history = train_joint(data, self.config, device=self.device)
        self.item_emb = self.model.encode_items(data.content_features, device=self.device)
        self._data = data
        return self
