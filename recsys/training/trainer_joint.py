"""Training loop for Joint End-to-End Collaborative + Contrastive Model."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

import torch
from recsys.models.joint_model import JointCLCRec


def train_joint(data, config, device=None) -> Tuple[JointCLCRec, List[float]]:
    """Train the JointCLCRec model coupling BPR collaborative loss with InfoNCE contrastive alignment.

    Parameters
    ----------
    data:
        DataBundle containing interaction graphs and content features.
    config:
        Config dataclass.
    device:
        torch device.

    Returns
    -------
    (model, history)
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if device is None:
        from recsys.models.torch_encoder import get_device
        device = get_device(config.device)

    # Prepare positive interaction triplets (u, pos_i)
    triplets = []
    for u, items in data.user_pos_items.items():
        for i in items:
            triplets.append((u, i))
    triplets = np.array(triplets, dtype=np.int64)

    content_dim = data.content_features.shape[1]
    n_users = data.n_users
    n_movies = data.n_movies
    warm_items = np.array(data.warm_items, dtype=np.int64)

    model = JointCLCRec(
        n_users=n_users,
        n_items=n_movies,
        content_dim=content_dim,
        emb_dim=config.emb_dim,
        hidden_dim=config.torch_hidden_dim,
        tau=config.tau,
        init_collab_emb=data.item_collab_emb,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.l2_reg
    )

    content_tensor = torch.from_numpy(data.content_features).to(device)

    bpr_weight = getattr(config, "bpr_weight", 1.0)
    use_hard = getattr(config, "hard_negatives", False)
    hard_w = getattr(config, "hard_negative_weight", 0.5)

    print(
        f"Training JointCLCRec (End-to-End) | device={device} | epochs={config.epochs} "
        f"| batch={config.batch_size} | bpr_weight={bpr_weight} | hard_negatives={use_hard}"
    )

    history: List[float] = []
    model.train()

    n_triplets = len(triplets)
    batch_size = config.batch_size

    for epoch in range(config.epochs):
        perm_triplets = np.random.permutation(n_triplets)
        np.random.shuffle(warm_items)

        epoch_losses = []
        n_batches = min(n_triplets // batch_size, max(1, len(warm_items) // batch_size))

        for b in range(n_batches):
            # 1. BPR Batch: sample (u, pos_i, neg_j)
            t_idx = perm_triplets[b * batch_size : (b + 1) * batch_size]
            u_b = triplets[t_idx, 0]
            i_b = triplets[t_idx, 1]
            # Negative sampling: random items
            j_b = np.random.randint(0, n_movies, size=len(t_idx))

            u_tensor = torch.from_numpy(u_b).to(device)
            i_tensor = torch.from_numpy(i_b).to(device)
            j_tensor = torch.from_numpy(j_b).to(device)

            loss_bpr = model.forward_bpr(u_tensor, i_tensor, j_tensor)

            # 2. Contrastive Batch: warm items
            w_start = (b * batch_size) % max(1, len(warm_items) - batch_size)
            w_idx = warm_items[w_start : w_start + batch_size]
            if len(w_idx) < 8:
                continue

            w_tensor = torch.from_numpy(w_idx).to(device)
            w_content = content_tensor[w_tensor]

            loss_cl = model.forward_contrastive(
                w_tensor, w_content, hard_negatives=use_hard, hard_weight=hard_w
            )

            # Total joint loss
            loss_total = bpr_weight * loss_bpr + loss_cl

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_losses.append(loss_total.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:3d}/{config.epochs}  |  Joint Loss: {avg_loss:.4f}")

    print("Joint training complete.\n")
    model.eval()
    return model, history
