"""Evaluation identity validation without loading model/GPU dependencies."""
import json
from pathlib import Path

def sample_labels(metadata_path, count):
    if count < 2:
        raise ValueError("Need at least two image pairs for a cross-pair baseline")
    if metadata_path is None or not Path(metadata_path).exists():
        return [(f"pair_{i}", "unverified") for i in range(count)]
    data=json.loads(Path(metadata_path).read_text())
    samples=data.get('samples', [])
    if len(samples)!=count:
        raise ValueError("Detected pair count does not match evaluation metadata")
    labels=[]
    for s in samples:
        if not all(k in s for k in ['trial','frame_idx','gesture']):
            raise ValueError("Evaluation sample identity is incomplete")
        labels.append((f"{s['trial']} f={s['frame_idx']}",str(s['gesture'])))
    if len(set(labels))!=len(labels):
        raise ValueError("Duplicate evaluation samples")
    return labels


def finite_json(value):
    """Represent undefined/infinite metrics as null, never nonstandard JSON."""
    import math
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
