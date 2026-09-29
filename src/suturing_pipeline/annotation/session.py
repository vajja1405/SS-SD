"""Box-annotation session: frame queue, kinematics pre-labels, saved boxes, quality metrics, YOLO export.

Frames are grouped by trial and camera, because the camera can move between JIGSAWS sessions; each group
gets its own projection model, refit after every saved frame. Every pre-label shown to the annotator is
stored with the box they finally saved, so acceptance and time saved are measured, not assumed.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

from .projection import TOOLS, Box, KinematicsBoxModel, iou

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (id TEXT PRIMARY KEY, grp TEXT NOT NULL, trial TEXT, capture TEXT,
                                   frame_index INTEGER, image_path TEXT NOT NULL, kin TEXT NOT NULL,
                                   split TEXT NOT NULL DEFAULT 'train', pos INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS shown (frame_id TEXT PRIMARY KEY, prelabels TEXT NOT NULL, model TEXT, shown_at REAL);
CREATE TABLE IF NOT EXISTS annotations (frame_id TEXT PRIMARY KEY, boxes TEXT NOT NULL, prelabels TEXT NOT NULL,
                                        seconds REAL, annotator TEXT, saved_at REAL NOT NULL,
                                        version INTEGER NOT NULL DEFAULT 1);
"""
ACCEPT_IOU = 0.5          # pre-label counts as usable
UNCHANGED_IOU = 0.95      # pre-label saved essentially as shown


class ValidationError(ValueError):
    pass


class ConflictError(ValueError):
    """Someone saved this frame after the caller loaded it (HTTP 409); `current` is what is stored now."""
    def __init__(self, message: str, current: dict):
        super().__init__(message)
        self.current = current


def _boxes(raw) -> list[Box]:
    out = []
    for b in raw or []:
        if b.get('tool') not in TOOLS:
            raise ValidationError(f"tool must be one of {TOOLS}")
        x1, y1, x2, y2 = (float(b[k]) for k in ('x1', 'y1', 'x2', 'y2'))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValidationError('boxes must be at least 2 px wide and tall')
        out.append(Box(b['tool'], x1, y1, x2, y2))
    if len({b.tool for b in out}) != len(out):
        raise ValidationError('one box per tool')
    return out


class AnnotationSession:
    def __init__(self, db_path: str = ':memory:', image_size: tuple[int, int] = (640, 480)):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute('PRAGMA table_info(annotations)')}
        if 'version' not in cols:                         # databases created before versioned saves
            self.db.execute('ALTER TABLE annotations ADD COLUMN version INTEGER NOT NULL DEFAULT 1')
        self.lock = threading.RLock()
        self.image_size = image_size
        self.models: dict[str, KinematicsBoxModel] = {}
        self.fit_summary: dict[str, dict] = {}
        for (grp,) in self.db.execute('SELECT DISTINCT grp FROM frames'):
            self._refit(grp)

    # ── frames ────────────────────────────────────────────────────────────
    def add_frames(self, rows: list[dict]) -> int:
        """rows: id, grp, trial, capture, frame_index, image_path, kin (76 floats), split."""
        with self.lock, self.db:
            pos = self.db.execute('SELECT COALESCE(MAX(pos), -1) + 1 FROM frames').fetchone()[0]
            added = 0
            for r in rows:
                if len(r['kin']) != 76:
                    raise ValidationError(f"frame {r['id']}: expected 76 kinematics values")
                cur = self.db.execute('INSERT OR IGNORE INTO frames VALUES (?,?,?,?,?,?,?,?,?)',
                                      (r['id'], r['grp'], r.get('trial'), r.get('capture'), r.get('frame_index'),
                                       str(r['image_path']), json.dumps([float(v) for v in r['kin']]),
                                       r.get('split', 'train'), pos))
                if cur.rowcount:
                    pos += 1
                    added += 1
        return added

    def _frame(self, frame_id: str):
        row = self.db.execute('SELECT id, grp, trial, capture, frame_index, image_path, kin, split FROM frames '
                              'WHERE id = ?', (frame_id,)).fetchone()
        if row is None:
            raise ValidationError(f'unknown frame {frame_id}')
        keys = ('id', 'grp', 'trial', 'capture', 'frame_index', 'image_path', 'kin', 'split')
        out = dict(zip(keys, row))
        out['kin'] = json.loads(out['kin'])
        return out

    def image_path(self, frame_id: str) -> Path:
        return Path(self._frame(frame_id)['image_path'])

    # ── model ─────────────────────────────────────────────────────────────
    def _labeled(self, grp: str):
        rows = self.db.execute('SELECT f.kin, a.boxes FROM annotations a JOIN frames f ON f.id = a.frame_id '
                               'WHERE f.grp = ? ORDER BY a.saved_at', (grp,)).fetchall()
        return [(json.loads(k), _boxes(json.loads(b))) for k, b in rows]

    def _refit(self, grp: str) -> None:
        model = KinematicsBoxModel(self.image_size)
        self.fit_summary[grp] = model.fit(self._labeled(grp))
        self.models[grp] = model

    # ── queue ─────────────────────────────────────────────────────────────
    def next_frame_id(self) -> str | None:
        row = self.db.execute('SELECT id FROM frames WHERE id NOT IN (SELECT frame_id FROM annotations) '
                              'ORDER BY pos LIMIT 1').fetchone()
        return row[0] if row else None

    def get(self, frame_id: str) -> dict:
        with self.lock:
            f = self._frame(frame_id)
            if f['grp'] not in self.models:
                self._refit(f['grp'])
            saved = self.db.execute('SELECT boxes, version FROM annotations WHERE frame_id = ?', (frame_id,)).fetchone()
            pre = [b.as_dict() for b in self.models[f['grp']].predict(f['kin'])]
            if saved is None:           # remember exactly what the annotator was shown
                with self.db:
                    self.db.execute('INSERT OR REPLACE INTO shown VALUES (?,?,?,?)',
                                    (frame_id, json.dumps(pre), json.dumps(self.fit_summary[f['grp']]), time.time()))
            done, total = self.progress()
            return {'id': f['id'], 'trial': f['trial'], 'capture': f['capture'], 'frame_index': f['frame_index'],
                    'width': self.image_size[0], 'height': self.image_size[1], 'prelabels': pre,
                    'model': self.fit_summary[f['grp']], 'boxes': json.loads(saved[0]) if saved else None,
                    'version': saved[1] if saved else 0,
                    'progress': {'done': done, 'total': total}}

    def save(self, frame_id: str, boxes, seconds: float | None = None, annotator: str = '',
             base_version: int | None = None, force: bool = False) -> dict:
        """Save boxes for a frame. `base_version` is the version the caller loaded (0 = never saved). If someone saved
        since, this is a conflict unless `force`. Re-sending the boxes already stored (a retry) changes nothing."""
        with self.lock:
            f = self._frame(frame_id)
            final = _boxes(boxes)
            payload = json.dumps([b.as_dict() for b in final])
            row = self.db.execute('SELECT boxes, version FROM annotations WHERE frame_id = ?', (frame_id,)).fetchone()
            current = row[1] if row else 0
            if row and row[0] == payload:
                return {'saved': True, 'duplicate': True, 'version': current, 'next_id': self.next_frame_id(),
                        'model': self.fit_summary.get(f['grp'])}
            if base_version is not None and base_version != current and not force:
                raise ConflictError('this frame was saved after you opened it',
                                    {'version': current, 'boxes': json.loads(row[0]) if row else None})
            shown = self.db.execute('SELECT prelabels FROM shown WHERE frame_id = ?', (frame_id,)).fetchone()
            with self.db:
                if row:                                   # keep the first save's pre-label comparison and timing
                    self.db.execute('UPDATE annotations SET boxes = ?, annotator = ?, saved_at = ?, version = ? '
                                    'WHERE frame_id = ?', (payload, annotator, time.time(), current + 1, frame_id))
                else:
                    self.db.execute('INSERT INTO annotations (frame_id, boxes, prelabels, seconds, annotator, saved_at, '
                                    'version) VALUES (?,?,?,?,?,?,1)',
                                    (frame_id, payload, shown[0] if shown else '[]', seconds, annotator, time.time()))
            self._refit(f['grp'])
            return {'saved': True, 'version': current + 1, 'next_id': self.next_frame_id(),
                    'model': self.fit_summary[f['grp']]}

    def progress(self) -> tuple[int, int]:
        done = self.db.execute('SELECT COUNT(*) FROM annotations').fetchone()[0]
        total = self.db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
        return done, total

    # ── metrics ───────────────────────────────────────────────────────────
    def stats(self) -> dict:
        rows = self.db.execute('SELECT a.boxes, a.prelabels, a.seconds, f.grp FROM annotations a '
                               'JOIN frames f ON f.id = a.frame_id ORDER BY a.saved_at').fetchall()
        pairs, missed, extra = [], 0, 0
        secs = {'with_prelabels': [], 'without_prelabels': []}
        for boxes, pre, seconds, _ in rows:
            final = {b.tool: b for b in _boxes(json.loads(boxes))}
            shown = {b['tool']: Box(b['tool'], b['x1'], b['y1'], b['x2'], b['y2']) for b in json.loads(pre)}
            for tool, box in final.items():
                if tool in shown:
                    p = shown[tool]
                    pairs.append((iou(p, box), ((p.center[0] - box.center[0]) ** 2 + (p.center[1] - box.center[1]) ** 2) ** 0.5))
                elif shown:
                    missed += 1
            extra += sum(1 for t in shown if t not in final)
            if seconds is not None:
                secs['with_prelabels' if shown else 'without_prelabels'].append(seconds)
        ious = [p[0] for p in pairs]
        done, total = self.progress()
        return {
            'frames': {'annotated': done, 'total': total},
            'prelabels': {
                'compared': len(pairs),
                'accepted_iou_0.5': sum(i >= ACCEPT_IOU for i in ious),
                'saved_unchanged': sum(i >= UNCHANGED_IOU for i in ious),
                'acceptance_rate': round(sum(i >= ACCEPT_IOU for i in ious) / len(ious), 3) if ious else None,
                'median_iou': round(median(ious), 3) if ious else None,
                'median_center_error_px': round(median(p[1] for p in pairs), 1) if pairs else None,
                'tools_missed': missed, 'extra_boxes_deleted': extra},
            'seconds_per_frame': {k: {'frames': len(v), 'median': round(median(v), 1) if v else None}
                                  for k, v in secs.items()},
            'models': self.fit_summary,
        }

    # ── export ────────────────────────────────────────────────────────────
    def export_yolo(self, out_root: str | Path) -> dict:
        """images/<split>/<id>.jpg + labels/<split>/<id>.txt (class cx cy w h, normalized), as train_yolo.py expects."""
        import shutil

        out_root = Path(out_root)
        w_img, h_img = self.image_size
        counts = defaultdict(int)
        rows = self.db.execute('SELECT f.id, f.image_path, f.split, a.boxes FROM annotations a '
                               'JOIN frames f ON f.id = a.frame_id').fetchall()
        for fid, image_path, split, boxes in rows:
            (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
            src = Path(image_path)
            shutil.copyfile(src, out_root / 'images' / split / f'{fid}{src.suffix}')
            lines = []
            for b in _boxes(json.loads(boxes)):
                cx, cy = (b.x1 + b.x2) / 2 / w_img, (b.y1 + b.y2) / 2 / h_img
                lines.append(f'{TOOLS.index(b.tool)} {cx:.6f} {cy:.6f} {(b.x2 - b.x1) / w_img:.6f} {(b.y2 - b.y1) / h_img:.6f}')
            (out_root / 'labels' / split / f'{fid}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
            counts[split] += 1
        return {'frames': dict(counts), 'classes': list(TOOLS), 'root': str(out_root)}
