import { describe, expect, it } from 'vitest';
import { type Box, clamp, describeModel, hitTest, iou, isUsable, move, normalize, resize, upsert } from './geometry';

const box = (x1: number, y1: number, x2: number, y2: number, tool: Box['tool'] = 'left_tool'): Box => ({ tool, x1, y1, x2, y2 });

describe('geometry', () => {
  it('normalizes boxes drawn in any direction and clamps to the image', () => {
    expect(normalize(box(50, 60, 10, 20))).toEqual(box(10, 20, 50, 60));
    expect(clamp(box(-5, 10, 700, 500), 640, 480)).toEqual(box(0, 10, 640, 480));
  });

  it('prefers handles, then the topmost box, then nothing', () => {
    const boxes = [box(10, 10, 100, 100), box(50, 50, 150, 150, 'right_tool')];
    expect(hitTest(boxes, 101, 99)).toEqual({ kind: 'handle', index: 0, handle: 'se' });
    expect(hitTest(boxes, 70, 70)).toEqual({ kind: 'inside', index: 1 });
    expect(hitTest(boxes, 300, 300)).toEqual({ kind: 'none' });
  });

  it('moves inside the image and resizes from any corner', () => {
    expect(move(box(10, 10, 60, 40), -30, 5, 640, 480)).toEqual(box(0, 15, 50, 45));
    expect(move(box(600, 440, 640, 480), 50, 50, 640, 480)).toEqual(box(600, 440, 640, 480));
    expect(resize(box(10, 10, 60, 40), 'nw', 70, 50, 640, 480)).toEqual(box(60, 40, 70, 50));
    expect(resize(box(10, 10, 60, 40), 'se', 90, 80, 640, 480)).toEqual(box(10, 10, 90, 80));
  });

  it('keeps one box per instrument', () => {
    const out = upsert([box(0, 0, 10, 10), box(5, 5, 20, 20, 'right_tool')], box(30, 30, 40, 40));
    expect(out).toEqual([box(5, 5, 20, 20, 'right_tool'), box(30, 30, 40, 40)]);
  });

  it('computes IoU and rejects slivers', () => {
    expect(iou(box(0, 0, 10, 10), box(0, 0, 10, 10))).toBe(1);
    expect(iou(box(0, 0, 10, 10), box(5, 0, 15, 10))).toBeCloseTo(1 / 3);
    expect(isUsable(box(0, 0, 3, 50))).toBe(false);
  });

  it('describes the pre-label model in plain words', () => {
    expect(describeModel({ left_tool: { frames: 2, model: null, loo_error_px: null },
      right_tool: { frames: 6, model: 'position', loo_error_px: 17.4 } }))
      .toBe('Left instrument: learning (2/4 frames) · Right instrument: about 17 px off on held-out frames');
  });
});
