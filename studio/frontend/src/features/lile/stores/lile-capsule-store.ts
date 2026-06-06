// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type {
  CapsuleStatus, TrajectoryEvent, TrajectoryTail,
} from "../api/types";

type State = {
  status: CapsuleStatus | null;
  trajectory: TrajectoryEvent[];
  lastOffset: number;
  totalSize: number;
  lastCommitToken: number | null;

  setStatus(s: CapsuleStatus | null): void;
  mergeTail(t: TrajectoryTail): void;
  reset(): void;
};

const MAX_ROLLING = 2000;  // cap rolling window

export const useLileCapsuleStore = create<State>((set) => ({
  status: null,
  trajectory: [],
  lastOffset: 0,
  totalSize: 0,
  lastCommitToken: null,

  setStatus: (s) =>
    set(() => ({
      status: s,
      lastCommitToken:
        s && s.running ? s.health.commit_cursor : null,
    })),

  mergeTail: (t) =>
    set((prev) => {
      const seen = new Set(prev.trajectory.map((e) => e.offset));
      const fresh = t.events.filter((e) => !seen.has(e.offset));
      const combined = [...prev.trajectory, ...fresh].slice(-MAX_ROLLING);
      const hasNextOffset = "next_offset" in t;
      const nextOffset = hasNextOffset
        ? (t as { next_offset: number }).next_offset
        : combined.length > 0
          ? Math.max(prev.lastOffset, ...combined.map((e) => e.offset + 1))
          : prev.lastOffset;
      const totalSize =
        "total_size" in t && typeof t.total_size === "number"
          ? t.total_size
          : combined.length;
      return { trajectory: combined, lastOffset: nextOffset, totalSize };
    }),

  reset: () =>
    set({ status: null, trajectory: [], lastOffset: 0,
          totalSize: 0, lastCommitToken: null }),
}));
