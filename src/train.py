import numpy as np
from tqdm import tqdm
from src.model import ContentEncoder, infonce_loss


def train(content_features: np.ndarray,
          item_collab_emb: np.ndarray,
          warm_items: list,
          emb_dim:    int   = 32,
          hidden_dim: int   = 64,
          epochs:     int   = 80,
          batch_size: int   = 256,
          lr:         float = 0.005,
          tau:        float = 0.1,
          l2_reg:     float = 1e-4,
          seed:       int   = 42):
    np.random.seed(seed)
    content_dim = content_features.shape[1]

    encoder     = ContentEncoder(content_dim, emb_dim, hidden_dim, lr)
    train_items = np.array(warm_items)
    history     = []

    print(f"Training CLCRec | epochs={epochs} | batch={batch_size} | τ={tau}")

    for epoch in range(epochs):
        np.random.shuffle(train_items)
        batch_losses = []

        for start in range(0, len(train_items), batch_size):
            batch = train_items[start: start + batch_size]
            if len(batch) < 8:
                continue

            x          = content_features[batch]
            z_collab   = item_collab_emb[batch]
            z_content, cache = encoder.forward(x)

            loss, grad = infonce_loss(z_content, z_collab, tau)
            batch_losses.append(loss)

            dW1, db1, dW2, db2 = encoder.backward(grad, cache, l2_reg)

            encoder.adam_step(dW1, db1, dW2, db2)

        avg_loss = float(np.mean(batch_losses))
        history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  Loss: {avg_loss:.4f}")

    print("Training complete.\n")
    return encoder, history
