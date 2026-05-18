// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, spyOn, beforeEach } from "bun:test";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";
import { CapsuleLoadForm } from "./capsule-load-form";

const RUNNING_HEALTH = {
  ok: true,
  model: "qwen3-8b",
  queue_depth: 0,
  commit_cursor: 0,
  merges: 0,
};

describe("CapsuleLoadForm", () => {
  beforeEach(() => {
    cleanup();
    useLileCapsuleStore.getState().reset();
  });

  it("disables Stop button when mode=external (lifecycle not Studio's)", () => {
    useLileCapsuleStore.getState().setStatus({
      running: true,
      mode: "external",
      health: RUNNING_HEALTH,
      url: "http://remote.example:8768",
    });
    render(<CapsuleLoadForm />);
    const stop = screen.getByRole("button", { name: /stop capsule/i });
    expect((stop as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/externally managed/i)).toBeTruthy();
    expect(screen.getByText(/remote\.example:8768/)).toBeTruthy();
  });

  it("enables Stop button when mode=spawned (Studio owns lifecycle)", () => {
    useLileCapsuleStore.getState().setStatus({
      running: true,
      mode: "spawned",
      pid: 4242,
      health: RUNNING_HEALTH,
      url: "http://127.0.0.1:8768",
    });
    render(<CapsuleLoadForm />);
    const stop = screen.getByRole("button", { name: /stop capsule/i });
    expect((stop as HTMLButtonElement).disabled).toBe(false);
  });

  it("calls lileClient.postStop when spawned-mode Stop is clicked", async () => {
    const spy = spyOn(lileClient, "postStop").mockImplementation(() =>
      Promise.resolve({ stopped: true, reason: "subprocess_terminated", pid: 4242 }),
    );
    useLileCapsuleStore.getState().setStatus({
      running: true,
      mode: "spawned",
      pid: 4242,
      health: RUNNING_HEALTH,
      url: "http://127.0.0.1:8768",
    });
    render(<CapsuleLoadForm />);
    fireEvent.click(screen.getByRole("button", { name: /stop capsule/i }));
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("renders the Load form when capsule is offline", () => {
    // No status set — offline state.
    render(<CapsuleLoadForm />);
    expect(screen.getByRole("button", { name: /load capsule/i })).toBeTruthy();
  });

  it("treats running without mode as legacy-running (Stop enabled)", () => {
    // Pre-PR-#55 backends don't send `mode` — make sure we don't accidentally
    // grey out the Stop button for them just because mode is missing.
    useLileCapsuleStore.getState().setStatus({
      running: true,
      externally_managed: false,
      health: RUNNING_HEALTH,
      url: "http://127.0.0.1:8768",
    });
    render(<CapsuleLoadForm />);
    const stop = screen.getByRole("button", { name: /stop capsule/i });
    expect((stop as HTMLButtonElement).disabled).toBe(false);
  });
});
