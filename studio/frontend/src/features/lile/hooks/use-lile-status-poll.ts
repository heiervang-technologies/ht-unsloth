// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";
import type { CapsuleStatus } from "../api/types";

type Poller = { start: () => void; stop: () => void };

// Exported for unit testing; the React hook below is a thin wrapper.
// Backoff: base interval on success, doubles on error up to maxBackoffMs.
export function createStatusPoller(deps: {
  fetchStatus: () => Promise<CapsuleStatus>;
  setStatus: (s: CapsuleStatus) => void;
  baseIntervalMs: number;
  maxBackoffMs: number;
}): Poller {
  let cancelled = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let backoff = deps.baseIntervalMs;

  async function tick() {
    if (cancelled) return;
    try {
      const s = await deps.fetchStatus();
      if (!cancelled) {
        deps.setStatus(s);
        backoff = deps.baseIntervalMs;
      }
    } catch {
      if (!cancelled) {
        deps.setStatus({ running: false });
        backoff = Math.min(backoff * 2, deps.maxBackoffMs);
      }
    }
    if (!cancelled) timer = setTimeout(tick, backoff);
  }

  return {
    start() { cancelled = false; backoff = deps.baseIntervalMs; void tick(); },
    stop() {
      cancelled = true;
      if (timer !== null) { clearTimeout(timer); timer = null; }
    },
  };
}

export function useLileStatusPoll(opts: { enabled: boolean }) {
  useEffect(() => {
    if (!opts.enabled) return;
    const poller = createStatusPoller({
      fetchStatus: () => lileClient.getStatus(),
      setStatus: (s) => useLileCapsuleStore.getState().setStatus(s),
      baseIntervalMs: 2000,
      maxBackoffMs: 5000,
    });
    poller.start();
    return () => poller.stop();
  }, [opts.enabled]);
}
