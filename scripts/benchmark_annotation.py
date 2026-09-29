"""Latency of the box-annotation session on a large queue (synthetic frames, SQLite file).

    python scripts/benchmark_annotation.py --frames 2000 --groups 5 --annotate 400

Loads synthetic frames split across several trial groups, then saves exact boxes frame by frame (so every save
refits that group's projection, including the leave-one-out model check) and times loading a frame (with its
pre-labels) and saving it. Writes docs/box-annotation-benchmark-<date>.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from suturing_pipeline.annotation.frames import demo_frames, demo_truth
from suturing_pipeline.annotation.session import AnnotationSession


def pct(values):
    a = np.array(values) * 1000
    return {'calls': len(values), 'median_ms': round(float(np.median(a)), 2), 'p95_ms': round(float(np.percentile(a, 95)), 2),
            'max_ms': round(float(a.max()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=2000)
    ap.add_argument('--groups', type=int, default=5)
    ap.add_argument('--annotate', type=int, default=400)
    ap.add_argument('--out', default=f"docs/box-annotation-benchmark-{time.strftime('%Y-%m-%d')}.json")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix='sssd_bench_'))
    rows = demo_frames(work / 'frames', n=args.frames)
    for i, r in enumerate(rows):                        # spread frames over trial groups, interleaved in the queue
        r['grp'] = f'trial{i % args.groups}:capture1'
    t0 = time.perf_counter()
    session = AnnotationSession(str(work / 'bench.db'))
    session.add_frames(rows)
    load_s = time.perf_counter() - t0
    truth = {r['id']: demo_truth(r) for r in rows}

    get_t, save_t = [], []
    for _ in range(args.annotate):
        fid = session.next_frame_id()
        t = time.perf_counter()
        frame = session.get(fid)
        get_t.append(time.perf_counter() - t)
        t = time.perf_counter()
        session.save(fid, truth[fid], seconds=3.0, base_version=frame['version'])
        save_t.append(time.perf_counter() - t)
    t = time.perf_counter()
    stats = session.stats()
    stats_s = time.perf_counter() - t

    per_group = args.annotate // args.groups
    report = {
        'generated': time.strftime('%Y-%m-%d %H:%M'),
        'machine': f'{platform.machine()} {platform.system()} {platform.release()}, Python {platform.python_version()}',
        'queue': {'frames': args.frames, 'trial_groups': args.groups, 'annotated': args.annotate,
                  'labeled_frames_per_group_at_end': per_group},
        'storage': 'SQLite file', 'load_frames_s': round(load_s, 2),
        'get_frame_with_prelabels': pct(get_t),
        'save_and_refit': pct(save_t),
        'save_and_refit_last_50': pct(save_t[-50:]),
        'stats_ms': round(stats_s * 1000, 1),
        'prelabel_acceptance_on_synthetic_frames': stats['prelabels']['acceptance_rate'],
        'note': 'Synthetic frames with a known linear camera: latency only; acceptance here says nothing about real frames.',
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
