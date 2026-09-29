import { describe, expect, it } from 'vitest';
import { HttpError, clearDraft, loadDraft, retryable, saveDraft, withRetry } from './net';

class MemoryStorage {
  private m = new Map<string, string>();
  getItem(k: string) { return this.m.get(k) ?? null; }
  setItem(k: string, v: string) { this.m.set(k, v); }
  removeItem(k: string) { this.m.delete(k); }
}

describe('drafts', () => {
  it('restore only against the version they were made from', () => {
    const s = new MemoryStorage() as unknown as Storage;
    const d = { boxes: [{ tool: 'left_tool' as const, x1: 1, y1: 2, x2: 30, y2: 40 }], fromPrelabel: [], baseVersion: 0 };
    saveDraft('f1', d, s);
    expect(loadDraft('f1', 0, s)).toEqual(d);
    expect(loadDraft('f1', 1, s)).toBeNull();
    clearDraft('f1', s);
    expect(loadDraft('f1', 0, s)).toBeNull();
    s.setItem('sssd-draft:f2', '{broken');
    expect(loadDraft('f2', 0, s)).toBeNull();
  });
});

describe('retries', () => {
  it('retry network and server errors only', async () => {
    expect(retryable(new TypeError('Failed to fetch'))).toBe(true);
    expect(retryable(new HttpError('down', 502))).toBe(true);
    expect(retryable(new HttpError('conflict', 409))).toBe(false);
    let calls = 0;
    const flaky = async () => { calls++; if (calls < 3) throw new TypeError('reset'); return 'ok'; };
    expect(await withRetry(flaky, [1, 1, 1], async () => {})).toBe('ok');
    expect(calls).toBe(3);
  });
});
