"""HTTP API and UI host for box annotation (see scripts/annotate_boxes.py)."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session import AnnotationSession, ValidationError

UI_DIST = Path(__file__).resolve().parents[3] / 'annotation_ui' / 'dist'


class BoxIn(BaseModel):
    tool: str
    x1: float
    y1: float
    x2: float
    y2: float


class AnnotationIn(BaseModel):
    boxes: list[BoxIn]
    seconds: float | None = None
    annotator: str = ''


def create_app(session: AnnotationSession, export_dir: str | Path | None = None, demo_truth: dict | None = None) -> FastAPI:
    app = FastAPI(title='SS-SD box annotation')

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get('/api/health')
    def health():
        return {'ok': True}

    @app.get('/api/next')
    def next_frame():
        done, total = session.progress()
        return {'id': session.next_frame_id(), 'done': done, 'total': total}

    @app.get('/api/frames/{frame_id}')
    def frame(frame_id: str):
        return guard(session.get, frame_id)

    @app.get('/api/frames/{frame_id}/image')
    def image(frame_id: str):
        path = guard(session.image_path, frame_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail='image missing on disk')
        return FileResponse(path, media_type='image/jpeg')

    @app.put('/api/frames/{frame_id}/annotation')
    def save(frame_id: str, body: AnnotationIn):
        return guard(session.save, frame_id, [b.model_dump() for b in body.boxes], body.seconds, body.annotator)

    @app.get('/api/stats')
    def stats():
        return session.stats()

    @app.post('/api/export')
    def export():
        if export_dir is None:
            raise HTTPException(status_code=400, detail='server started without --export-dir')
        return session.export_yolo(export_dir)

    if demo_truth is not None:     # synthetic demo only: where each drawn instrument really is (for end-to-end tests)
        @app.get('/api/demo/truth/{frame_id}')
        def truth(frame_id: str):
            if frame_id not in demo_truth:
                raise HTTPException(status_code=404, detail='unknown frame')
            return demo_truth[frame_id]

    if UI_DIST.exists():
        app.mount('/assets', StaticFiles(directory=UI_DIST / 'assets'), name='assets')

        @app.get('/')
        def index():
            return FileResponse(UI_DIST / 'index.html')

    return app
