// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, beforeEach } from "bun:test";
import { render, screen, cleanup } from "@testing-library/react";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";
import { CapsuleStatusStrip } from "./capsule-status-strip";

describe("CapsuleStatusStrip", () => {
  beforeEach(() => {
    cleanup();
    useLileCapsuleStore.getState().reset();
  });

  it("shows offline when store has no status", () => {
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/offline/i)).toBeTruthy();
  });

  it("shows model name and commit when online", () => {
    useLileCapsuleStore.getState().setStatus({
      running: true,
      externally_managed: false,
      health: {
        ok: true,
        model: "qwen3-0.6b",
        queue_depth: 3,
        commit_cursor: 77,
        merges: 2,
      },
      url: "http://127.0.0.1:8765",
    });
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/qwen3-0.6b/)).toBeTruthy();
    expect(screen.getByText(/commit 77/)).toBeTruthy();
    expect(screen.getByText(/queue 3/)).toBeTruthy();
  });
});
