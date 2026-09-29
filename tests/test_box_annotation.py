"""Box annotation: kinematics projection, session metrics, YOLO export, frame extraction and the HTTP API."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from suturing_pipeline.annotation.frames import demo_frames, demo_truth, jigsaws_frames
from suturing_pipeline.annotation.projection import KIN_BLOCK, Box, KinematicsBoxModel, iou
from suturing_pipeline.annotation.server import create_app
from suturing_pipeline.annotation.session import AnnotationSession, ConflictError, ValidationError


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


def test_closed_form_leave_one_out_matches_refitting_every_frame():
    rng = np.random.default_rng(5)
    for tilt, gain in ((0.0, 0.0), (0.5, 60.0)):
        frames = [_kin(rng, axis_tilt=tilt) for _ in range(15)]
        noisy = []
        for k in frames:
            boxes = _true_boxes(k, axis_gain=gain)
            dx, dy = rng.normal(0, 6, 2)
            noisy.append((k, [Box(b.tool, b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy) for b in boxes]))
        for model in ('position', 'position_axis'):
            samples = [(k, b) for k, boxes in noisy for b in boxes if b.tool == 'left_tool']
            fast = KinematicsBoxModel.loo_error(samples, 'left_tool', model)
            slow = KinematicsBoxModel.loo_error_bruteforce(samples, 'left_tool', model)
            assert fast == pytest.approx(slow, rel=1e-6, abs=1e-6)


def test_saves_are_versioned_so_a_stale_tab_cannot_overwrite(demo):
    session, rows = demo
    fid = rows[0]['id']
    a = [{'tool': 'left_tool', 'x1': 10, 'y1': 10, 'x2': 60, 'y2': 50}]
    b = [{'tool': 'left_tool', 'x1': 20, 'y1': 20, 'x2': 70, 'y2': 60}]
    assert session.get(fid)['version'] == 0
    assert session.save(fid, a, base_version=0)['version'] == 1
    assert session.save(fid, a, base_version=0)['duplicate'] is True        # retry of the same save
    with pytest.raises(ConflictError) as err:
        session.save(fid, b, base_version=0)                               # second tab opened before the save
    assert err.value.current['version'] == 1 and err.value.current['boxes'][0]['x1'] == 10
    assert session.save(fid, b, base_version=1)['version'] == 2
    assert session.save(fid, a, base_version=1, force=True)['version'] == 3
    assert session.get(fid)['boxes'][0]['x1'] == 10
    assert session.stats()['frames']['annotated'] == 1


def test_old_databases_gain_the_version_column(tmp_path):
    import sqlite3
    db = tmp_path / 'old.db'
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE annotations (frame_id TEXT PRIMARY KEY, boxes TEXT NOT NULL, prelabels TEXT NOT NULL,
                         seconds REAL, annotator TEXT, saved_at REAL NOT NULL);""")
    con.close()
    AnnotationSession(str(db))
    cols = {r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(annotations)')}
    assert 'version' in cols


def test_http_conflict_returns_the_stored_boxes(demo, tmp_path):
    session, rows = demo
    client = TestClient(create_app(session, tmp_path / 'yolo'))
    fid = rows[0]['id']
    box = {'tool': 'right_tool', 'x1': 300, 'y1': 200, 'x2': 360, 'y2': 250}
    assert client.put(f'/api/frames/{fid}/annotation', json={'boxes': [box], 'base_version': 0}).json()['version'] == 1
    moved = dict(box, x1=310, x2=370)
    clash = client.put(f'/api/frames/{fid}/annotation', json={'boxes': [moved], 'base_version': 0})
    assert clash.status_code == 409 and clash.json()['detail']['current']['boxes'][0]['x1'] == 300
    forced = client.put(f'/api/frames/{fid}/annotation', json={'boxes': [moved], 'base_version': 0, 'force': True})
    assert forced.json()['version'] == 2


def test_concurrent_saves_and_reads_share_one_connection_safely(demo):
    import threading

    session, rows = demo
    errors = []

    def writer(chunk):
        try:
            for r in chunk:
                f = session.get(r['id'])
                session.save(r['id'], demo_truth(r), seconds=1.0, base_version=f['version'])
        except Exception as e:                      # noqa: BLE001
            errors.append(repr(e))

    def reader():
        try:
            for _ in range(40):
                session.stats(); session.progress(); session.next_frame_id()
        except Exception as e:                      # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(rows[i::3],)) for i in range(3)] + \
              [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert session.stats()['frames']['annotated'] == len(rows)


def test_ab_mode_hides_every_other_prelabel_and_measures_it_unbiased(tmp_path):
    rows = demo_frames(tmp_path / 'frames', n=14)
    session = AnnotationSession(ab_test=True)
    session.add_frames(rows)
    conditions = []
    for r in rows:
        f = session.get(r['id'])
        conditions.append(f['condition'])
        if f['condition'] == 'hidden':
            assert f['prelabels'] == []
            assert session.get(r['id'])['condition'] == 'hidden'          # a reload keeps the condition
        session.save(r['id'], demo_truth(r), seconds=5.0 if f['condition'] == 'hidden' else 2.0,
                     base_version=f['version'])
    assert conditions[:4] == ['learning'] * 4
    assert conditions[4:] == ['shown', 'hidden'] * 5
    s = session.stats()
    assert s['ab_test'] is True
    assert s['prelabels']['compared'] == 10 and s['hidden_prelabels']['compared'] == 10     # 5 frames x 2 tools each
    assert s['hidden_prelabels']['median_iou'] > 0.99                                      # exact synthetic camera
    assert s['seconds_per_frame']['hidden_prelabels'] == {'frames': 5, 'median': 5.0}
    assert s['seconds_per_frame']['with_prelabels'] == {'frames': 5, 'median': 2.0}
    assert s['seconds_per_frame']['learning_no_model_yet']['frames'] == 4
