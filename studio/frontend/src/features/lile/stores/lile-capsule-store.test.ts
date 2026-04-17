// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, beforeEach } from "bun:test";
import { useLileCapsuleStore } from "./lile-capsule-store";

describe("lile-capsule-store", () => {
  beforeEach(() => useLileCapsuleStore.getState().reset());

  it("mergeTail appends events and advances lastOffset", () => {
    const s = useLileCapsuleStore.getState();
    s.mergeTail({
      events: [{ offset: 0, kind: "train_step", loss: 1.0, batch_id: 0,
                 objective: "sft", batch_size: 1 }],
      next_offset: 1, total_size: 1,
    });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
    expect(useLileCapsuleStore.getState().lastOffset).toBe(1);
  });

  it("mergeTail deduplicates by offset", () => {
    const s = useLileCapsuleStore.getState();
    const ev = { offset: 0, kind: "train_step", loss: 1, batch_id: 0,
                 objective: "sft", batch_size: 1 };
    s.mergeTail({ events: [ev], next_offset: 1, total_size: 1 });
    s.mergeTail({ events: [ev], next_offset: 1, total_size: 1 });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
  });

  it("mergeTail handles old shape (no next_offset)", () => {
    const s = useLileCapsuleStore.getState();
    s.mergeTail({ events: [{ offset: 0, kind: "train_step", loss: 1,
                             batch_id: 0, objective: "sft", batch_size: 1 }] });
    expect(useLileCapsuleStore.getState().trajectory).toHaveLength(1);
    // old shape: lastOffset tracks max offset seen
    expect(useLileCapsuleStore.getState().lastOffset).toBe(1);
  });

  it("setStatus records last commit cursor", () => {
    const s = useLileCapsuleStore.getState();
    s.setStatus({ running: true, externally_managed: false,
                  health: { ok: true, model: "m", queue_depth: 0,
                            commit_cursor: 42, merges: 0 },
                  url: "http://x" });
    expect(useLileCapsuleStore.getState().lastCommitToken).toBe(42);
  });
});
