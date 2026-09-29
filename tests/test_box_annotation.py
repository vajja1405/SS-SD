"""Box annotation: kinematics projection, session metrics, YOLO export, frame extraction and the HTTP API."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from suturing_pipeline.annotation.frames import demo_frames, demo_truth, jigsaws_frames
from suturing_pipeline.annotation.projection import KIN_BLOCK, Box, KinematicsBoxModel, iou
from suturing_pipeline.annotation.server import create_app
from suturing_pipeline.annotation.session import AnnotationSession, ValidationError


def _kin(rng, axis_tilt=0.0):
    kin = np.zeros(76)
    for tool in KIN_BLOCK:
        s = KIN_BLOCK[tool]
        kin[s:s + 3] = rng.uniform(-0.02, 0.02, 3)
        a = rng.uniform(-1, 1, 3) * axis_tilt
        R = np.eye(3)
        R[:, 2] = [a[0], a[1], 1.0]
        kin[s + 3:s + 12] = R.ravel()
    return kin


def _true_boxes(kin, axis_gain=0.0):
    out = []
    for tool, off in (('left_tool', (200.0, 220.0)), ('right_tool', (430.0, 250.0))):
        s = KIN_BLOCK[tool]
        axis = kin[s + 3:s + 12].reshape(3, 3)[:, 2]
        cx = 5000 * kin[s] + axis_gain * axis[0] + off[0]
        cy = 5000 * kin[s + 1] + axis_gain * axis[1] + off[1]
        out.append(Box(tool, cx - 25, cy - 18, cx + 25, cy + 18))
    return out


def test_projection_waits_for_labels_then_recovers_a_linear_camera():
    rng = np.random.default_rng(0)
    frames = [_kin(rng) for _ in range(12)]
    model = KinematicsBoxModel()
    assert model.fit([(k, _true_boxes(k)) for k in frames[:3]])['left_tool']['model'] is None
    assert model.predict(frames[5]) == []
    summary = model.fit([(k, _true_boxes(k)) for k in frames[:8]])
    assert summary['left_tool']['loo_error_px'] < 0.5
    pred = {b.tool: b for b in model.predict(frames[10])}
    for truth in _true_boxes(frames[10]):
        assert iou(pred[truth.tool], truth) > 0.99


def test_tool_axis_model_is_chosen_when_orientation_moves_the_jaw():
    rng = np.random.default_rng(1)
    frames = [_kin(rng, axis_tilt=0.5) for _ in range(14)]
    summary = KinematicsBoxModel().fit([(k, _true_boxes(k, axis_gain=60)) for k in frames])
    assert summary['right_tool']['model'] == 'position_axis'
    assert summary['right_tool']['loo_error_px'] < 1.0


def test_off_screen_projections_are_dropped():
    rng = np.random.default_rng(2)
    frames = [_kin(rng) for _ in range(6)]
    model = KinematicsBoxModel()
    model.fit([(k, _true_boxes(k)) for k in frames])
    far = frames[0].copy()
    far[KIN_BLOCK['left_tool']] = 1.0            # 5,000 px to the right
    assert [b.tool for b in model.predict(far)] == ['right_tool']


@pytest.fixture
def demo(tmp_path):
    rows = demo_frames(tmp_path / 'frames', n=12)
    session = AnnotationSession()
    session.add_frames(rows)
    return session, rows


def test_session_records_shown_prelabels_and_measures_acceptance(demo):
    session, rows = demo
    for i, r in enumerate(rows[:8]):
        f = session.get(r['id'])
        assert bool(f['prelabels']) == (i >= 4)
        truth = demo_truth(r)
        if i == 7:                                # last frame: annotator moves one pre-label well off, deletes the other
            truth = [dict(truth[0], x1=truth[0]['x1'] + 45, x2=truth[0]['x2'] + 45)]
        session.save(r['id'], truth, seconds=10.0 if i < 4 else 3.0)
    s = session.stats()
    assert s['frames'] == {'annotated': 8, 'total': 12}
    p = s['prelabels']
    assert (p['compared'], p['accepted_iou_0.5'], p['extra_boxes_deleted']) == (7, 6, 1)
    assert s['seconds_per_frame']['with_prelabels'] == {'frames': 4, 'median': 3.0}
    assert s['seconds_per_frame']['without_prelabels'] == {'frames': 4, 'median': 10.0}
    assert session.next_frame_id() == rows[8]['id']


def test_session_rejects_bad_boxes(demo):
    session, rows = demo
    fid = rows[0]['id']
    for bad in ([{'tool': 'scalpel', 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}],
                [{'tool': 'left_tool', 'x1': 5, 'y1': 5, 'x2': 6, 'y2': 30}],
                [{'tool': 'left_tool', 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}] * 2):
        with pytest.raises(ValidationError):
            session.save(fid, bad)
    with pytest.raises(ValidationError):
        session.get('nope')


def test_yolo_export_matches_the_training_layout(demo, tmp_path):
    session, rows = demo
    r = rows[0]
    session.save(r['id'], [{'tool': 'right_tool', 'x1': 320, 'y1': 240, 'x2': 384, 'y2': 288}])
    out = session.export_yolo(tmp_path / 'yolo')
    split = r['split']
    assert out['frames'] == {split: 1} and out['classes'] == ['left_tool', 'right_tool']
    assert (tmp_path / 'yolo' / 'images' / split / f"{r['id']}.jpg").exists()
    line = (tmp_path / 'yolo' / 'labels' / split / f"{r['id']}.txt").read_text().split()
    assert line == ['1', '0.550000', '0.550000', '0.100000', '0.100000']


def test_jigsaws_frames_pairs_each_frame_with_its_kinematics_row(tmp_path):
    root = tmp_path / 'Suturing'
    (root / 'video').mkdir(parents=True)
    (root / 'kinematics' / 'AllGestures').mkdir(parents=True)
    writer = cv2.VideoWriter(str(root / 'video' / 'Suturing_B001_capture1.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
    for i in range(60):
        writer.write(np.full((48, 64, 3), i * 4, np.uint8))
    writer.release()
    kin = np.arange(58 * 76, dtype=float).reshape(58, 76)       # shorter than the video, as in JIGSAWS
    np.savetxt(root / 'kinematics' / 'AllGestures' / 'Suturing_B001.txt', kin)
    rows = jigsaws_frames(root, ['B001'], tmp_path / 'out', frames_per_trial=5, margin=5)
    assert [r['frame_index'] for r in rows] == [5, 17, 28, 40, 52]
    assert all(r['kin'] == kin[r['frame_index']].tolist() for r in rows)
    assert rows[0]['grp'] == 'B001:capture1' and all((tmp_path / 'out' / f"{r['id']}.jpg").exists() for r in rows)


def test_http_api_round_trip(demo, tmp_path):
    session, rows = demo
    client = TestClient(create_app(session, tmp_path / 'yolo', demo_truth={r['id']: demo_truth(r) for r in rows}))
    nxt = client.get('/api/next').json()
    assert nxt == {'id': rows[0]['id'], 'done': 0, 'total': 12}
    frame = client.get(f"/api/frames/{nxt['id']}").json()
    assert frame['prelabels'] == [] and frame['progress'] == {'done': 0, 'total': 12}
    assert client.get(f"/api/frames/{nxt['id']}/image").headers['content-type'] == 'image/jpeg'
    bad = client.put(f"/api/frames/{nxt['id']}/annotation", json={'boxes': [{'tool': 'x', 'x1': 0, 'y1': 0, 'x2': 9, 'y2': 9}]})
    assert bad.status_code == 422
    truth = client.get(f"/api/demo/truth/{nxt['id']}").json()
    saved = client.put(f"/api/frames/{nxt['id']}/annotation", json={'boxes': truth, 'seconds': 4.2}).json()
    assert saved['saved'] and saved['next_id'] == rows[1]['id']
    assert client.get('/api/stats').json()['frames']['annotated'] == 1
    assert client.post('/api/export').json()['frames'] == {rows[0]['split']: 1}
    assert client.get('/api/frames/unknown').status_code == 422
