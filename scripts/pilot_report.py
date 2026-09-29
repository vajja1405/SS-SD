"""Summarise a box-annotation pilot (A/B mode) from its database.

    python scripts/pilot_report.py --db outputs/annotation/pilot.db --out docs/box-annotation-pilot-<date>.json

Reports, per condition (pre-labels shown, pre-labels hidden, learning frames before any model):
frames, boxes saved, median seconds per frame and per saved box, and pre-label quality against the saved boxes.
The hidden condition is the unbiased accuracy estimate: those boxes were drawn without seeing the model's guess.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from suturing_pipeline.annotation.session import AnnotationSession


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--out', default=f"docs/box-annotation-pilot-{time.strftime('%Y-%m-%d')}.json")
    args = ap.parse_args()
    session = AnnotationSession(args.db)
    stats = session.stats()
    rows = session.db.execute('SELECT f.grp, a.boxes, a.prelabels, a.seconds, a.hidden, a.annotator FROM annotations a '
                              'JOIN frames f ON f.id = a.frame_id ORDER BY a.saved_at').fetchall()

    def cond(pre, hidden):
        return 'learning' if not json.loads(pre) else ('hidden' if hidden else 'shown')

    per = {}
    for grp, boxes, pre, secs, hidden, _ in rows:
        c = cond(pre, hidden)
        d = per.setdefault(c, {'frames': 0, 'boxes': 0, 'seconds': [], 'seconds_per_box': []})
        n = len(json.loads(boxes))
        d['frames'] += 1
        d['boxes'] += n
        if secs is not None:
            d['seconds'].append(secs)
            if n:
                d['seconds_per_box'].append(secs / n)
    timing = {c: {'frames': d['frames'], 'boxes_saved': d['boxes'],
                  'boxes_per_frame': round(d['boxes'] / d['frames'], 2),
                  'median_seconds_per_frame': round(median(d['seconds']), 1) if d['seconds'] else None,
                  'median_seconds_per_box': round(median(d['seconds_per_box']), 1) if d['seconds_per_box'] else None}
              for c, d in per.items()}
    report = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'frames': stats['frames'], 'trials': sorted({r[0] for r in rows}), 'annotators': len({r[5] for r in rows}) or 1,
        'prelabels_shown': stats['prelabels'], 'prelabels_hidden_unbiased': stats['hidden_prelabels'],
        'timing': timing, 'models_at_end': stats['models'],
        'notes': ['Hidden pre-labels are compared with boxes drawn without seeing them: the unbiased accuracy.',
                  'Shown pre-labels can anchor the annotator; acceptance above the hidden accuracy suggests anchoring.',
                  'Learning frames come first in each trial, so their times also include practice.',
                  'One annotator and 40 frames: medians and counts, not significance tests.'],
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
