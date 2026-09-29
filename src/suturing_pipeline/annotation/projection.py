"""Kinematics-projected bounding-box pre-labels.

JIGSAWS ships robot kinematics for every video frame but no camera calibration. The instrument tips are
known in the robot's frame, so a small projection learned from the annotator's own boxes can place a
box on each instrument in frames nobody has labeled yet, the same way driving datasets project LiDAR
points into camera images to pre-label objects.

Model: for each tool, the box centre is a linear function of the tool-tip position and the tool's
shaft axis (third column of its rotation matrix), with a per-tool offset. Box size is the median
annotated size for that tool. With few labels the model falls back to position only; candidates are
compared by leave-one-frame-out error, so the pre-labeler reports how far off it expects to be.

Tool mapping (observed in the capture1/capture2 Suturing videos): the arm the kinematics file calls
"slave left" (columns 39-57) appears on the right of the image, and "slave right" (58-76) on the left.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOOLS = ('left_tool', 'right_tool')                     # as seen in the image
KIN_BLOCK = {'left_tool': 57, 'right_tool': 38}         # 0-based start of the slave block in the 76 columns
MODELS = {'position': 4, 'position_axis': 7}            # model -> minimum labeled frames per tool


@dataclass
class Box:
    tool: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def size(self) -> tuple[float, float]:
        return self.x2 - self.x1, self.y2 - self.y1

    def as_dict(self) -> dict:
        return {'tool': self.tool, 'x1': round(self.x1, 1), 'y1': round(self.y1, 1),
                'x2': round(self.x2, 1), 'y2': round(self.y2, 1)}


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def tool_features(kin_row, tool: str, model: str) -> np.ndarray:
    k = np.asarray(kin_row, dtype=float)
    start = KIN_BLOCK[tool]
    tip = k[start:start + 3]
    if model == 'position':
        return np.r_[tip, 1.0]
    axis = k[start + 3:start + 12].reshape(3, 3)[:, 2]
    return np.r_[tip, axis, 1.0]


@dataclass
class ToolModel:
    model: str
    weights: np.ndarray            # (features, 2)
    size: tuple[float, float]
    loo_error_px: float | None
    frames: int


@dataclass
class KinematicsBoxModel:
    """Fits per-tool projections from labeled frames and proposes boxes for unlabeled ones."""
    image_size: tuple[int, int] = (640, 480)
    tools: dict[str, ToolModel] = field(default_factory=dict)

    @staticmethod
    def _fit(samples, tool: str, model: str) -> np.ndarray:
        A = np.array([tool_features(k, tool, model) for k, _ in samples])
        Y = np.array([box.center for _, box in samples])
        W, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return W

    @classmethod
    def loo_error(cls, samples, tool: str, model: str) -> float | None:
        """Leave-one-frame-out centre error in pixels (median)."""
        if len(samples) < MODELS[model] + 1:
            return None
        errs = []
        for i in range(len(samples)):
            train = samples[:i] + samples[i + 1:]
            W = cls._fit(train, tool, model)
            pred = tool_features(samples[i][0], tool, model) @ W
            errs.append(float(np.hypot(*(pred - np.array(samples[i][1].center)))))
        return float(np.median(errs))

    def fit(self, labeled: list[tuple[list[float], list[Box]]]) -> dict:
        """labeled: (kinematics row, boxes the annotator saved) per frame. Returns a summary per tool."""
        summary = {}
        self.tools = {}
        for tool in TOOLS:
            samples = [(k, b) for k, boxes in labeled for b in boxes if b.tool == tool]
            best = None
            for model, min_frames in MODELS.items():
                if len(samples) < min_frames:
                    continue
                err = self.loo_error(samples, tool, model)
                if best is None or (err is not None and (best[1] is None or err < best[1])):
                    best = (model, err)
            if best is None:
                summary[tool] = {'frames': len(samples), 'model': None, 'loo_error_px': None}
                continue
            model, err = best
            sizes = np.array([b.size for _, b in samples])
            self.tools[tool] = ToolModel(model, self._fit(samples, tool, model),
                                         (float(np.median(sizes[:, 0])), float(np.median(sizes[:, 1]))), err, len(samples))
            summary[tool] = {'frames': len(samples), 'model': model,
                             'loo_error_px': None if err is None else round(err, 1)}
        return summary

    def predict(self, kin_row) -> list[Box]:
        w_img, h_img = self.image_size
        out = []
        for tool, tm in self.tools.items():
            cx, cy = tool_features(kin_row, tool, tm.model) @ tm.weights
            w, h = tm.size
            x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2, y2 = min(float(w_img), cx + w / 2), min(float(h_img), cy + h / 2)
            if x2 - x1 > 4 and y2 - y1 > 4:                 # skip tools projected off-screen
                out.append(Box(tool, x1, y1, x2, y2))
        return out
