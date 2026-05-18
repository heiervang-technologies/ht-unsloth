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

  it("shows spawned badge with pid when mode=spawned", () => {
    useLileCapsuleStore.getState().setStatus({
      running: true,
      mode: "spawned",
      pid: 4242,
      health: {
        ok: true, model: "qwen3-8b",
        queue_depth: 0, commit_cursor: 0, merges: 0,
      },
      url: "http://127.0.0.1:8768",
    });
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/spawned/)).toBeTruthy();
    expect(screen.getByText(/pid 4242/)).toBeTruthy();
  });

  it("shows external badge when mode=external", () => {
    useLileCapsuleStore.getState().setStatus({
      running: true,
      mode: "external",
      health: {
        ok: true, model: "qwen3-8b",
        queue_depth: 0, commit_cursor: 0, merges: 0,
      },
      url: "http://remote.example:8768",
    });
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/external/)).toBeTruthy();
  });

  it("explains unconfigured offline state", () => {
    useLileCapsuleStore.getState().setStatus({
      running: false,
      mode: "unconfigured",
    });
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/not configured/i)).toBeTruthy();
    expect(screen.getByText(/LILE_DAEMON_URL/)).toBeTruthy();
  });

  it("surfaces unreachable-URL when external probe failed", () => {
    useLileCapsuleStore.getState().setStatus({
      running: false,
      mode: "external-unreachable",
      url: "http://remote.example:8768",
    });
    render(<CapsuleStatusStrip />);
    expect(screen.getByText(/not reachable at http:\/\/remote\.example:8768/)).toBeTruthy();
  });
});
