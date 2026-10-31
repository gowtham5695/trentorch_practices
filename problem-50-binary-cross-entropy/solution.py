import numpy as np

def solve(logits, labels):
    """Compute mean binary cross-entropy from real-valued logits and binary labels, using a stable logits-based formula."""
    logits = np.array(logits, dtype=float)
    labels = np.array(labels, dtype=float)

    loss = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))

    return np.mean(loss)
