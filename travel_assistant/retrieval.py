"""Optional local dense embeddings and hybrid ranking."""
import json
from urllib.request import Request, urlopen

import numpy as np

MODEL = "all-minilm"


def embed(passages):
    batches = []
    for start in range(0, len(passages), 128):
        request = Request(
            "http://127.0.0.1:11434/api/embed",
            data=json.dumps({"model": MODEL, "input": passages[start:start + 128]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=90) as response:
            values = np.asarray(json.load(response)["embeddings"], dtype=np.float32)
        if values.ndim != 2 or len(values) != len(passages[start:start + 128]) or not np.isfinite(values).all():
            raise ValueError("The local embedding service returned invalid vectors.")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        batches.append(values / np.maximum(norms, 1e-12))
    return np.concatenate(batches)


def hybrid_scores(lexical, semantic):
    """Fuse independent top-100 rankings, preferring agreement between them."""
    scores = np.zeros_like(lexical)
    for values, minimum in [(lexical, 0), (semantic, .25)]:
        order = np.argsort(-values, kind="stable")[:100]
        for rank, index in enumerate(order, 1):
            if values[index] > minimum:
                scores[index] += 1 / (60 + rank)
    return scores
