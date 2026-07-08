import numpy as np


def compute_similarity(embed_a, embed_b) -> float:
    a = np.asarray(embed_a, dtype=float)
    b = np.asarray(embed_b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(similarity, 0.0, 1.0))
