"""Frame sets for box annotation: JIGSAWS video frames paired with the kinematics row recorded at the same time."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..detection.labeling import _sanitize, assign_split, sample_frame_indices
from .projection import KIN_BLOCK


def jigsaws_frames(root: str | Path, trials: list[str], out_dir: str | Path, *, capture: str = 'capture1',
                   frames_per_trial: int = 20, margin: int = 150, val_ratio: float = 0.2, seed: int = 42) -> list[dict]:
    """Sample frames uniformly from each Suturing trial (skipping the first and last `margin` frames) and pair each
    with its kinematics row. Video and kinematics are both 30 Hz; frame i matches kinematics row i."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in trials:
        video = root / 'video' / f'Suturing_{trial}_{capture}.avi'
        kin_path = root / 'kinematics' / 'AllGestures' / f'Suturing_{trial}.txt'
        if not video.exists() or not kin_path.exists():
            raise FileNotFoundError(f'missing video or kinematics for {trial}: {video}, {kin_path}')
        kin = np.loadtxt(kin_path)
        cap = cv2.VideoCapture(str(video))
        usable = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), len(kin))
        split = assign_split(trial, val_ratio, seed)
        for idx in sample_frame_indices(usable, frames_per_trial, min_frame=margin, max_frame=usable - 1 - margin):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fid = f'suturing_{_sanitize(trial)}_{capture}_f{idx:06d}'
            path = out_dir / f'{fid}.jpg'
            cv2.imwrite(str(path), frame)
            rows.append({'id': fid, 'grp': f'{trial}:{capture}', 'trial': trial, 'capture': capture,
                         'frame_index': idx, 'image_path': str(path), 'kin': kin[idx].tolist(), 'split': split})
        cap.release()
    return rows


def demo_frames(out_dir: str | Path, n: int = 24, seed: int = 3) -> list[dict]:
    """Synthetic frames for tests and the demo: two 'instruments' drawn where a fixed synthetic camera projects
    random tool tips, so the projection model has a real (known) mapping to learn. No JIGSAWS data needed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cam = {'left_tool': np.array([[5200.0, 0, 0], [0, 5200.0, 0]]), 'right_tool': np.array([[5200.0, 0, 0], [0, 5200.0, 0]])}
    offset = {'left_tool': np.array([210.0, 220.0]), 'right_tool': np.array([430.0, 250.0])}
    rows = []
    for i in range(n):
        kin = np.zeros(76)
        img = np.full((480, 640, 3), (214, 222, 228), np.uint8)
        cv2.circle(img, (320, 250), 190, (236, 238, 240), -1)
        for tool, color in (('left_tool', (60, 60, 64)), ('right_tool', (48, 52, 60))):
            tip = rng.uniform(-0.02, 0.02, 3)
            start = KIN_BLOCK[tool]
            kin[start:start + 3] = tip
            kin[start + 3:start + 12] = np.eye(3).ravel()
            cx, cy = cam[tool] @ tip + offset[tool]
            shaft_end = (0, int(cy) + 40) if tool == 'left_tool' else (639, int(cy) + 40)
            cv2.line(img, shaft_end, (int(cx), int(cy)), color, 22)
            cv2.rectangle(img, (int(cx) - 26, int(cy) - 18), (int(cx) + 26, int(cy) + 18), (170, 176, 184), -1)
        fid = f'demo_f{i:03d}'
        path = out_dir / f'{fid}.jpg'
        cv2.imwrite(str(path), img)
        rows.append({'id': fid, 'grp': 'demo:capture1', 'trial': 'demo', 'capture': 'capture1', 'frame_index': i,
                     'image_path': str(path), 'kin': kin.tolist(), 'split': 'train' if i % 5 else 'val'})
    return rows


def demo_truth(row: dict) -> list[dict]:
    """The box each synthetic instrument's jaw occupies (for tests)."""
    kin = np.array(row['kin'])
    out = []
    for tool, off in (('left_tool', (210.0, 220.0)), ('right_tool', (430.0, 250.0))):
        start = KIN_BLOCK[tool]
        cx, cy = 5200.0 * kin[start] + off[0], 5200.0 * kin[start + 1] + off[1]
        out.append({'tool': tool, 'x1': cx - 26, 'y1': cy - 18, 'x2': cx + 26, 'y2': cy + 18})
    return out
