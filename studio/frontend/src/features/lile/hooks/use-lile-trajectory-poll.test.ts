// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import { createTrajectoryPoller } from "./use-lile-trajectory-poll";
import type { TrajectoryTail } from "../api/types";

describe("createTrajectoryPoller", () => {
  let now = 0;
  let timers: Array<{ at: number; fn: () => void }> = [];
  const realSetTimeout = globalThis.setTimeout;
  const realClearTimeout = globalThis.clearTimeout;

  beforeEach(() => {
    now = 0;
    timers = [];
    globalThis.setTimeout = ((fn: () => void, ms: number) => {
      const handle = { at: now + ms, fn };
      timers.push(handle);
      return handle as unknown as ReturnType<typeof setTimeout>;
    }) as typeof setTimeout;
    globalThis.clearTimeout = ((h: unknown) => {
      timers = timers.filter((t) => t !== h);
    }) as typeof clearTimeout;
  });

  afterEach(() => {
    globalThis.setTimeout = realSetTimeout;
    globalThis.clearTimeout = realClearTimeout;
  });

  async function advance(ms: number) {
    now += ms;
    const due = timers.filter((t) => t.at <= now);
    timers = timers.filter((t) => t.at > now);
    for (const t of due) t.fn();
    // allow pending fetches in the tick to resolve
    await Promise.resolve();
    await Promise.resolve();
  }

  it("polls with current offset and merges each response", async () => {
    let offset = 0;
    const merged: TrajectoryTail[] = [];
    const fetchTail = mock((o: number) => {
      expect(o).toBe(offset);
      const tail: TrajectoryTail = {
        events: [{ offset, kind: "train_step", loss: 1,
                   batch_id: offset, objective: "sft", batch_size: 1 }],
        next_offset: offset + 1,
        total_size: offset + 1,
      };
      offset += 1;
      return Promise.resolve(tail);
    });

    const poller = createTrajectoryPoller({
      fetchTail,
      getOffset: () => offset,
      mergeTail: (t) => merged.push(t),
      intervalMs: 2000,
    });
    poller.start();
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchTail).toHaveBeenCalledTimes(1);
    expect(merged).toHaveLength(1);

    await advance(2000);
    expect(fetchTail).toHaveBeenCalledTimes(2);
    expect(merged).toHaveLength(2);
    poller.stop();
  });

  it("stops after stop() and swallows fetch errors", async () => {
    let calls = 0;
    const fetchTail = mock(() => {
      calls += 1;
      if (calls === 1) return Promise.reject(new Error("net"));
      return Promise.resolve({ events: [], next_offset: 0, total_size: 0 } as TrajectoryTail);
    });
    const poller = createTrajectoryPoller({
      fetchTail,
      getOffset: () => 0,
      mergeTail: () => {},
      intervalMs: 2000,
    });
    poller.start();
    await Promise.resolve();
    await Promise.resolve();
    expect(calls).toBe(1);

    poller.stop();
    await advance(2000);
    expect(calls).toBe(1); // no further fetches after stop
  });
});
