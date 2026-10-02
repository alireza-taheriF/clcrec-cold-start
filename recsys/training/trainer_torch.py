"""Training loop for the PyTorch content encoder.

Mirrors :mod:`recsys.training.trainer_numpy` but uses autograd and an Adam
optimizer from ``torch.optim``. The objective is the same InfoNCE alignment
against fixed SVD collaborative embeddings.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from recsys.models.torch_encoder import (
    TorchContentEncoder,
    get_device,
    info_nce_loss,
)


def train_torch(data, config, device=None) -> Tuple[TorchContentEncoder, List[float]]:
    """Train the Torch encoder on warm items.

    Parameters
    ----------
    data:
        A :class:`~recsys.data.dataset.DataBundle`.
    config:
        Project :class:`~config.Config` holding hyperparameters.
    device:
        Optional pre-resolved ``torch.device``. Falls back to
        :func:`~recsys.models.torch_encoder.get_device` using ``config.device``.

    Returns
    -------
    (encoder, history)
        The trained module (on CPU, in eval mode) and the per-epoch loss list.
    """
    import torch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if device is None:
        device = get_device(config.device)
    content_dim = data.content_features.shape[1]

    encoder = TorchContentEncoder(
        content_dim=content_dim,
        emb_dim=config.emb_dim,
        hidden_dim=config.torch_hidden_dim,
        tau=config.tau,
        learnable_tau=config.learnable_tau,
    ).to(device)

    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=config.lr, weight_decay=config.l2_reg
    )

    # Move the fixed tensors to the device once.
    content = torch.from_numpy(data.content_features).to(device)
    collab = torch.from_numpy(data.item_collab_emb).to(device)
    train_items = np.array(data.warm_items)

    tau_note = "learnable" if config.learnable_tau else f"{config.tau}"
    print(
        f"Training CLCRec (torch) | device={device} | epochs={config.epochs} "
        f"| batch={config.batch_size} | τ={tau_note}"
    )

    history: List[float] = []
    encoder.train()
    for epoch in range(config.epochs):
        np.random.shuffle(train_items)
        batch_losses = []

        for start in range(0, len(train_items), config.batch_size):
            batch = train_items[start: start + config.batch_size]
            if len(batch) < 8:
                continue

            idx = torch.from_numpy(batch).long().to(device)
            x = content[idx]
            z_collab = collab[idx]

            z_content = encoder(x)
            use_hard = getattr(config, "hard_negatives", False)
            hard_w = getattr(config, "hard_negative_weight", 0.5)
            loss = info_nce_loss(
                z_content, z_collab, encoder.tau,
                hard_negatives=use_hard,
                hard_weight=hard_w,
                content_features=x if use_hard else None,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        avg_loss = float(np.mean(batch_losses))
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            tau_val = float(encoder.tau.detach().cpu())
            print(f"  Epoch {epoch + 1:3d}/{config.epochs}  |  Loss: {avg_loss:.4f}  |  τ={tau_val:.4f}")

    print("Training complete.\n")
    encoder.eval()
    return encoder, history
