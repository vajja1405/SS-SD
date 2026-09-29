// Box geometry for the annotation canvas (unit-tested in geometry.test.ts). Coordinates are image pixels.

export type Tool = 'left_tool' | 'right_tool';
export const TOOLS: Tool[] = ['left_tool', 'right_tool'];

export interface Box {
  tool: Tool;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export type Handle = 'nw' | 'ne' | 'sw' | 'se';
export type Hit = { kind: 'handle'; index: number; handle: Handle } | { kind: 'inside'; index: number } | { kind: 'none' };

export function normalize(b: Box): Box {
  return { tool: b.tool, x1: Math.min(b.x1, b.x2), y1: Math.min(b.y1, b.y2), x2: Math.max(b.x1, b.x2), y2: Math.max(b.y1, b.y2) };
}

export function clamp(b: Box, width: number, height: number): Box {
  const c = (v: number, hi: number) => Math.min(Math.max(v, 0), hi);
  const n = normalize(b);
  return { tool: n.tool, x1: c(n.x1, width), y1: c(n.y1, height), x2: c(n.x2, width), y2: c(n.y2, height) };
}

const corners = (b: Box): [Handle, number, number][] => [
  ['nw', b.x1, b.y1], ['ne', b.x2, b.y1], ['sw', b.x1, b.y2], ['se', b.x2, b.y2],
];

/** Handles win over box interiors; the last-drawn box wins ties. `tolerance` is in image pixels. */
export function hitTest(boxes: Box[], x: number, y: number, tolerance = 6): Hit {
  for (let i = boxes.length - 1; i >= 0; i--) {
    for (const [handle, hx, hy] of corners(boxes[i])) {
      if (Math.abs(x - hx) <= tolerance && Math.abs(y - hy) <= tolerance) return { kind: 'handle', index: i, handle };
    }
  }
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) return { kind: 'inside', index: i };
  }
  return { kind: 'none' };
}

export function move(b: Box, dx: number, dy: number, width: number, height: number): Box {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  const x1 = Math.min(Math.max(b.x1 + dx, 0), width - w);
  const y1 = Math.min(Math.max(b.y1 + dy, 0), height - h);
  return { tool: b.tool, x1, y1, x2: x1 + w, y2: y1 + h };
}

export function resize(b: Box, handle: Handle, x: number, y: number, width: number, height: number): Box {
  const next = { ...b };
  if (handle === 'nw' || handle === 'sw') next.x1 = x; else next.x2 = x;
  if (handle === 'nw' || handle === 'ne') next.y1 = y; else next.y2 = y;
  return clamp(next, width, height);
}

/** Replace the box for this tool (one box per instrument). */
export function upsert(boxes: Box[], box: Box): Box[] {
  return [...boxes.filter((b) => b.tool !== box.tool), box];
}

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

export function isUsable(b: Box, minSize = 4): boolean {
  return b.x2 - b.x1 >= minSize && b.y2 - b.y1 >= minSize;
}

export function toolLabel(tool: Tool): string {
  return tool === 'left_tool' ? 'Left instrument' : 'Right instrument';
}

export function describeModel(m: Record<string, { frames: number; model: string | null; loo_error_px: number | null }>): string {
  const parts = TOOLS.map((t) => {
    const s = m[t];
    if (!s || !s.model) return `${toolLabel(t)}: learning (${s ? s.frames : 0}/4 frames)`;
    const err = s.loo_error_px === null ? 'error not measured yet' : `about ${Math.round(s.loo_error_px)} px off on held-out frames`;
    return `${toolLabel(t)}: ${err}`;
  });
  return parts.join(' · ');
}
