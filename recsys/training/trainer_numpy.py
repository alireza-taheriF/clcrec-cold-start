"""Training loop for the from-scratch NumPy content encoder.

Behavior is identical to the original ``src/train.py``: minibatch InfoNCE over
warm items, hand-written Adam updates, average loss logged every 10 epochs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from recsys.models.numpy_encoder import ContentEncoder, infonce_loss


def train_numpy(data, config) -> Tuple[ContentEncoder, List[float]]:
    """Train a :class:`ContentEncoder` on warm items and return it with loss history."""
    np.random.seed(config.seed)

    content_features = data.content_features
    item_collab_emb  = data.item_collab_emb
    content_dim      = content_features.shape[1]

    encoder     = ContentEncoder(content_dim, config.emb_dim,
                                 config.hidden_dim, config.lr)
    train_items = np.array(data.warm_items)
    history: List[float] = []

    print(f"Training CLCRec (numpy) | epochs={config.epochs} | "
          f"batch={config.batch_size} | tau={config.tau}")

    for epoch in range(config.epochs):
        np.random.shuffle(train_items)
        batch_losses = []

        for start in range(0, len(train_items), config.batch_size):
            batch = train_items[start: start + config.batch_size]
            if len(batch) < 8:            # skip degenerate tail batches
                continue

            x          = content_features[batch]
            z_collab   = item_collab_emb[batch]
            z_content, cache = encoder.forward(x)

            loss, grad = infonce_loss(z_content, z_collab, config.tau)
            batch_losses.append(loss)

            dW1, db1, dW2, db2 = encoder.backward(grad, cache, config.l2_reg)
            encoder.adam_step(dW1, db1, dW2, db2)

        avg_loss = float(np.mean(batch_losses))
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{config.epochs}  |  Loss: {avg_loss:.4f}")

    print("Training complete.\n")
    return encoder, history
