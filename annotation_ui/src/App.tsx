import { useCallback, useEffect, useRef, useState } from 'react';
import {
  type Box, type Handle, type Tool, TOOLS, clamp, describeModel, hitTest, isUsable, move, resize, toolLabel, upsert,
} from './geometry';

type ModelSummary = Record<string, { frames: number; model: string | null; loo_error_px: number | null }>;

interface Frame {
  id: string;
  trial: string;
  capture: string;
  frame_index: number;
  width: number;
  height: number;
  prelabels: Box[];
  model: ModelSummary;
  boxes: Box[] | null;
  progress: { done: number; total: number };
}

interface Stats {
  frames: { annotated: number; total: number };
  prelabels: { compared: number; acceptance_rate: number | null; median_iou: number | null; median_center_error_px: number | null;
    saved_unchanged: number; tools_missed: number; extra_boxes_deleted: number };
  seconds_per_frame: Record<'with_prelabels' | 'without_prelabels', { frames: number; median: number | null }>;
}

const COLORS: Record<Tool, string> = { left_tool: '#1f9d8a', right_tool: '#d9822b' };

type Drag =
  | { kind: 'draw'; tool: Tool; x0: number; y0: number }
  | { kind: 'move'; index: number; x0: number; y0: number; start: Box }
  | { kind: 'resize'; index: number; handle: Handle };

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function App() {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [finished, setFinished] = useState(false);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [fromPrelabel, setFromPrelabel] = useState<Set<Tool>>(new Set());
  const [active, setActive] = useState<Tool>('left_tool');
  const [selected, setSelected] = useState<number | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const canvas = useRef<HTMLCanvasElement>(null);
  const image = useRef<HTMLImageElement | null>(null);
  const drag = useRef<Drag | null>(null);
  const shownAt = useRef(performance.now());

  const loadStats = useCallback(async () => setStats(await json<Stats>(await fetch('/api/stats'))), []);

  const load = useCallback(async (id?: string | null) => {
    setError('');
    try {
      const target = id ?? (await json<{ id: string | null }>(await fetch('/api/next'))).id;
      if (!target) {
        setFinished(true);
        setFrame(null);
        await loadStats();
        return;
      }
      const f = await json<Frame>(await fetch(`/api/frames/${encodeURIComponent(target)}`));
      const img = new Image();
      img.src = `/api/frames/${encodeURIComponent(target)}/image`;
      await img.decode();
      image.current = img;
      setFrame(f);
      setBoxes(f.boxes ?? f.prelabels);
      setFromPrelabel(new Set(f.boxes ? [] : f.prelabels.map((b) => b.tool)));
      setSelected(null);
      shownAt.current = performance.now();
      await loadStats();
    } catch (e) {
      setError((e as Error).message);
    }
  }, [loadStats]);

  useEffect(() => { void load(); }, [load]);

  // Draw the frame and boxes.
  useEffect(() => {
    const c = canvas.current;
    if (!c || !frame || !image.current) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(image.current, 0, 0, frame.width, frame.height);
    boxes.forEach((b, i) => {
      ctx.strokeStyle = COLORS[b.tool];
      ctx.lineWidth = i === selected ? 3 : 2;
      ctx.setLineDash(fromPrelabel.has(b.tool) ? [6, 4] : []);
      ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS[b.tool];
      for (const [hx, hy] of [[b.x1, b.y1], [b.x2, b.y1], [b.x1, b.y2], [b.x2, b.y2]]) ctx.fillRect(hx - 3, hy - 3, 6, 6);
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText(b.tool === 'left_tool' ? 'L' : 'R', b.x1 + 3, Math.max(12, b.y1 - 4));
    });
  }, [frame, boxes, selected, fromPrelabel]);

  const toImage = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = (frame?.width ?? rect.width) / rect.width;
    return { x: (e.clientX - rect.left) * scale, y: (e.clientY - rect.top) * scale };
  };

  const edited = (tool: Tool) => setFromPrelabel((s) => { const n = new Set(s); n.delete(tool); return n; });

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!frame) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const { x, y } = toImage(e);
    const hit = hitTest(boxes, x, y, 6 * (frame.width / e.currentTarget.getBoundingClientRect().width));
    if (hit.kind === 'handle') {
      drag.current = { kind: 'resize', index: hit.index, handle: hit.handle };
      setSelected(hit.index);
    } else if (hit.kind === 'inside') {
      drag.current = { kind: 'move', index: hit.index, x0: x, y0: y, start: boxes[hit.index] };
      setSelected(hit.index);
    } else {
      drag.current = { kind: 'draw', tool: active, x0: x, y0: y };
      setSelected(null);
    }
  };

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = drag.current;
    if (!d || !frame) return;
    const { x, y } = toImage(e);
    if (d.kind === 'draw') {
      const next = upsert(boxes, clamp({ tool: d.tool, x1: d.x0, y1: d.y0, x2: x, y2: y }, frame.width, frame.height));
      setBoxes(next);
      setSelected(next.length - 1);
      edited(d.tool);
    } else if (d.kind === 'move') {
      setBoxes(boxes.map((b, i) => (i === d.index ? move(d.start, x - d.x0, y - d.y0, frame.width, frame.height) : b)));
      edited(boxes[d.index].tool);
    } else {
      setBoxes(boxes.map((b, i) => (i === d.index ? resize(b, d.handle, x, y, frame.width, frame.height) : b)));
      edited(boxes[d.index].tool);
    }
  };

  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    onMove(e);                 // apply the release point: fast drags may not send a final move event
    drag.current = null;
    setBoxes((bs) => bs.filter((b) => isUsable(b)));
  };

  const save = useCallback(async () => {
    if (!frame) return;
    setError('');
    try {
      const seconds = Math.round((performance.now() - shownAt.current) / 100) / 10;
      const res = await json<{ next_id: string | null }>(await fetch(`/api/frames/${encodeURIComponent(frame.id)}/annotation`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boxes, seconds }),
      }));
      await load(res.next_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [boxes, frame, load]);

  const removeSelected = useCallback(() => {
    if (selected === null) return;
    setBoxes((bs) => bs.filter((_, i) => i !== selected));
    setSelected(null);
  }, [selected]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      if (e.key === '1') setActive('left_tool');
      else if (e.key === '2') setActive('right_tool');
      else if (e.key === 'Enter') { e.preventDefault(); void save(); }
      else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeSelected(); }
      else return;
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [save, removeSelected]);

  const exportYolo = async () => {
    try {
      const out = await json<{ frames: Record<string, number>; root: string }>(await fetch('/api/export', { method: 'POST' }));
      setMessage(`Exported ${Object.values(out.frames).reduce((a, b) => a + b, 0)} frames to ${out.root}`);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const p = stats?.prelabels;
  const s = stats?.seconds_per_frame;
  return (
    <main className="shell">
      <header className="top">
        <h1>Instrument boxes</h1>
        {frame && (
          <p className="meta" data-testid="progress">
            Frame {frame.progress.done + 1} of {frame.progress.total} · trial {frame.trial} · {frame.capture} · #{frame.frame_index}
          </p>
        )}
      </header>
      {error && <p role="alert" className="error">{error}</p>}
      {finished ? (
        <p data-testid="finished">Every frame is annotated.</p>
      ) : frame && (
        <section className="work">
          <canvas
            ref={canvas}
            width={frame.width}
            height={frame.height}
            data-testid="canvas"
            aria-label="Frame; drag to draw a box for the active instrument"
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
          />
          <aside className="side">
            <div className="tools" role="radiogroup" aria-label="Instrument to draw">
              {TOOLS.map((t, i) => (
                <button key={t} role="radio" aria-checked={active === t} onClick={() => setActive(t)} style={{ borderColor: COLORS[t] }}>
                  {toolLabel(t)} <kbd>{i + 1}</kbd>
                </button>
              ))}
            </div>
            <p className="hint" data-testid="model">
              {frame.prelabels.length ? 'Dashed boxes are pre-labels from the robot kinematics. ' : 'No pre-labels yet. '}
              {describeModel(frame.model)}
            </p>
            <ul className="boxes" data-testid="boxes">
              {boxes.map((b, i) => (
                <li key={b.tool} className={i === selected ? 'selected' : ''}>
                  <span style={{ color: COLORS[b.tool] }}>{toolLabel(b.tool)}</span>{' '}
                  <code data-testid={`box-${b.tool}`}>{[b.x1, b.y1, b.x2, b.y2].map((v) => Math.round(v)).join(', ')}</code>
                  {fromPrelabel.has(b.tool) && <small> pre-label</small>}
                </li>
              ))}
            </ul>
            <div className="actions">
              <button onClick={() => void save()}>Save and next <kbd>Enter</kbd></button>
              <button className="secondary" onClick={removeSelected} disabled={selected === null}>Delete box <kbd>Del</kbd></button>
            </div>
          </aside>
        </section>
      )}
      {stats && (
        <section className="stats" aria-label="Pre-label quality">
          <div className="tile"><strong data-testid="acceptance">{p?.acceptance_rate == null ? '—' : `${Math.round(p.acceptance_rate * 100)}%`}</strong>
            <span>pre-labels kept (IoU ≥ 0.5), {p?.compared ?? 0} compared</span></div>
          <div className="tile"><strong>{p?.median_iou ?? '—'}</strong><span>median IoU vs. your box</span></div>
          <div className="tile"><strong>{p?.median_center_error_px ?? '—'}</strong><span>median centre error (px)</span></div>
          <div className="tile"><strong>{s?.with_prelabels.median ?? '—'} / {s?.without_prelabels.median ?? '—'}</strong>
            <span>median seconds per frame, with / without pre-labels</span></div>
          <button className="secondary" onClick={() => void exportYolo()}>Export YOLO labels</button>
          {message && <p data-testid="export-message">{message}</p>}
        </section>
      )}
    </main>
  );
}
