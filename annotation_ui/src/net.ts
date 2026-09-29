// Request retries and local drafts for the annotation UI (unit-tested in net.test.ts).
import type { Box, Tool } from './geometry';

export class HttpError extends Error {
  constructor(message: string, readonly status: number, readonly detail?: unknown) {
    super(message);
  }
}

export function retryable(err: unknown): boolean {
  return !(err instanceof HttpError) || err.status >= 500;
}

export async function withRetry<T>(fn: () => Promise<T>, delays = [250, 750, 2000],
                                   sleep = (ms: number) => new Promise((r) => setTimeout(r, ms)),
                                   onRetry?: (attempt: number) => void): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (!retryable(err) || attempt >= delays.length) throw err;
      onRetry?.(attempt + 1);
      await sleep(delays[attempt]);
    }
  }
}

export async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const d = body.detail;
    throw new HttpError(typeof d === 'string' ? d : d?.message ?? `${res.status} ${res.statusText}`, res.status, d);
  }
  return res.json() as Promise<T>;
}

export interface Draft {
  boxes: Box[];
  fromPrelabel: Tool[];
  baseVersion: number;
}

const draftKey = (frameId: string) => `sssd-draft:${frameId}`;

export function saveDraft(frameId: string, d: Draft, storage: Storage | null = store()): void {
  try { storage?.setItem(draftKey(frameId), JSON.stringify(d)); } catch { /* storage full or blocked */ }
}

/** A draft only applies to the version it was made from; anything else is stale. */
export function loadDraft(frameId: string, version: number, storage: Storage | null = store()): Draft | null {
  try {
    const d = JSON.parse(storage?.getItem(draftKey(frameId)) ?? 'null') as Draft | null;
    return d && d.baseVersion === version && Array.isArray(d.boxes) ? d : null;
  } catch {
    return null;
  }
}

export function clearDraft(frameId: string, storage: Storage | null = store()): void {
  try { storage?.removeItem(draftKey(frameId)); } catch { /* ignore */ }
}

function store(): Storage | null {
  try { return typeof localStorage === 'undefined' ? null : localStorage; } catch { return null; }
}
