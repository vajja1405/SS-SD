"""Draw instrument boxes on JIGSAWS frames with kinematics-projected pre-labels.

    python scripts/annotate_boxes.py --jigsaws ~/Downloads/Suturing --trials B001,C001 --db outputs/annotation/boxes.db
    python scripts/annotate_boxes.py --demo                        # synthetic frames, no dataset needed

Open http://127.0.0.1:8780 (build the UI first: cd annotation_ui && npm install && npm run build).
Saved boxes export to YOLO format for scripts/train_yolo.py via the UI's Export button or POST /api/export.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from suturing_pipeline.annotation.frames import demo_frames, demo_truth, jigsaws_frames
from suturing_pipeline.annotation.server import create_app
from suturing_pipeline.annotation.session import AnnotationSession


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--jigsaws', help='JIGSAWS Suturing folder (video/, kinematics/)')
    ap.add_argument('--trials', default='B001,C001', help='comma-separated trial ids, e.g. B001,C001')
    ap.add_argument('--capture', default='capture1', choices=['capture1', 'capture2'])
    ap.add_argument('--frames-per-trial', type=int, default=20)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--db', default=':memory:')
    ap.add_argument('--frames-dir', default='outputs/annotation/frames')
    ap.add_argument('--export-dir', default='outputs/annotation/yolo')
    ap.add_argument('--port', type=int, default=8780)
    ap.add_argument('--ab', action='store_true',
                    help='alternate showing and hiding pre-labels once the model is ready (pilot measurement)')
    args = ap.parse_args()

    session = AnnotationSession(args.db, ab_test=args.ab)
    truth = None
    if args.demo:
        rows = demo_frames(Path(tempfile.mkdtemp(prefix='sssd_demo_')))
        truth = {r['id']: demo_truth(r) for r in rows}
        added = session.add_frames(rows)
    elif args.jigsaws:
        rows = jigsaws_frames(Path(args.jigsaws).expanduser(), [t.strip() for t in args.trials.split(',') if t.strip()],
                              args.frames_dir, capture=args.capture, frames_per_trial=args.frames_per_trial)
        added = session.add_frames(rows)
    else:
        ap.error('pass --jigsaws or --demo')
    done, total = session.progress()
    print(f'{added} new frames; {done}/{total} annotated. http://127.0.0.1:{args.port}')
    uvicorn.run(create_app(session, args.export_dir, demo_truth=truth), host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
