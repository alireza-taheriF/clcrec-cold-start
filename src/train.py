from __future__ import annotations

import numpy as np
from src.model import ContentEncoder, infonce_loss


def _epoch_loss(encoder, content_features, item_collab_emb, items,
                batch_size, tau, l2_reg, update: bool):
    batch_losses = []
    for start in range(0, len(items), batch_size):
        batch = items[start: start + batch_size]
        if len(batch) < 8:
            continue
        x = content_features[batch]
        z_collab = item_collab_emb[batch]
        z_content, cache = encoder.forward(x)
        loss, grad = infonce_loss(z_content, z_collab, tau)
        batch_losses.append(loss)
        if update:
            dW1, db1, dW2, db2 = encoder.backward(grad, cache, l2_reg)
            encoder.adam_step(dW1, db1, dW2, db2)
    return float(np.mean(batch_losses)) if batch_losses else float("nan")


def train(content_features: np.ndarray,
          item_collab_emb: np.ndarray,
          warm_items: list,
          emb_dim: int = 32,
          hidden_dim: int = 64,
          epochs: int = 80,
          batch_size: int = 256,
          lr: float = 0.005,
          tau: float = 0.1,
          l2_reg: float = 1e-4,
          seed: int = 42,
          patience: int = 20,
          val_warm_frac: float = 0.1):
    np.random.seed(seed)
    content_dim = content_features.shape[1]

    encoder = ContentEncoder(content_dim, emb_dim, hidden_dim, lr)
    items = np.array(warm_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(items)

    n_val = int(len(items) * val_warm_frac) if val_warm_frac > 0 else 0
    if n_val >= 8 and len(items) - n_val >= 8:
        val_items = items[:n_val]
        train_items = items[n_val:]
    else:
        val_items = np.array([], dtype=int)
        train_items = items

    history = []
    best_state = None
    best_val = float("inf")
    wait = 0

    print(f"Training CLCRec | epochs={epochs} | batch={batch_size} | τ={tau}")

    for epoch in range(epochs):
        rng.shuffle(train_items)
        train_loss = _epoch_loss(
            encoder, content_features, item_collab_emb, train_items,
            batch_size, tau, l2_reg, update=True,
        )
        if len(val_items) >= 8:
            val_loss = _epoch_loss(
                encoder, content_features, item_collab_emb, val_items,
                batch_size, tau, l2_reg, update=False,
            )
        else:
            val_loss = train_loss

        history.append(train_loss)
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            wait = 0
            best_state = (encoder.W1.copy(), encoder.b1.copy(),
                          encoder.W2.copy(), encoder.b2.copy())
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {train_loss:.4f}  |  Val: {val_loss:.4f}")

        if patience and wait >= patience:
            print(f"  Early stop at epoch {epoch+1} (best val {best_val:.4f})")
            break

    if best_state is not None:
        encoder.W1, encoder.b1, encoder.W2, encoder.b2 = best_state

    print("Training complete.\n")
    return encoder, history
