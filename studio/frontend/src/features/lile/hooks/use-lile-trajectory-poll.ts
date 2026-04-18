// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";
import type { TrajectoryTail } from "../api/types";

type Poller = { start: () => void; stop: () => void };

// Exported for unit testing; the React hook below is a thin wrapper.
// The polling loop is deliberately extracted as pure logic so it can be
// exercised with bun:test fake timers without pulling in a DOM + React
// testing harness (vitest / @testing-library/react / jsdom).
export function createTrajectoryPoller(deps: {
  fetchTail: (sinceOffset: number) => Promise<TrajectoryTail>;
  getOffset: () => number;
  mergeTail: (tail: TrajectoryTail) => void;
  intervalMs: number;
}): Poller {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  async function tick() {
    if (cancelled) return;
    try {
      const tail = await deps.fetchTail(deps.getOffset());
      if (!cancelled) deps.mergeTail(tail);
    } catch {
      // transient; status poll handles recovery signals
    }
    if (!cancelled) timer = setTimeout(tick, deps.intervalMs);
  }

  return {
    start() {
      cancelled = false;
      void tick();
    },
    stop() {
      cancelled = true;
      if (timer !== null) {
        clearTimeout(timer);
        timer = null;
      }
    },
  };
}

export function useLileTrajectoryPoll(opts: { enabled: boolean }) {
  useEffect(() => {
    if (!opts.enabled) return;
    const poller = createTrajectoryPoller({
      fetchTail: (off) => lileClient.getTrajectoryTail(off),
      getOffset: () => useLileCapsuleStore.getState().lastOffset,
      mergeTail: (t) => useLileCapsuleStore.getState().mergeTail(t),
      intervalMs: 2000,
    });
    poller.start();
    return () => poller.stop();
  }, [opts.enabled]);
}
