// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { describe, it, expect, spyOn, beforeEach, afterEach } from "bun:test";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
import { lileClient } from "../api/lile-client";
import { SnapshotsTab } from "./snapshots-tab";

describe("SnapshotsTab", () => {
  beforeEach(() => {
    cleanup();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders rows when the daemon returns {snapshots: string[]}", async () => {
    // Shape that `SnapshotManager.list()` actually returns today:
    // the route wraps `list[str]` in `{"snapshots": [...]}`.
    const spy = spyOn(lileClient, "getSnapshots").mockImplementation(() =>
      Promise.resolve({ snapshots: ["pre_reasoning_restart", "pre_streaming_restart"] }),
    );

    render(<SnapshotsTab />);

    await waitFor(() => {
      expect(screen.getByText("pre_reasoning_restart")).toBeTruthy();
      expect(screen.getByText("pre_streaming_restart")).toBeTruthy();
    });

    expect(screen.queryByText(/no snapshots yet/i)).toBeNull();

    spy.mockRestore();
  });

  it("still accepts {snapshots: SnapshotRow[]} shape for forward-compat", async () => {
    const spy = spyOn(lileClient, "getSnapshots").mockImplementation(() =>
      Promise.resolve({
        snapshots: [
          { name: "v1", created_at: "2026-04-17" },
          { name: "v2" },
        ],
      }),
    );

    render(<SnapshotsTab />);

    await waitFor(() => {
      expect(screen.getByText("v1")).toBeTruthy();
      expect(screen.getByText("v2")).toBeTruthy();
      expect(screen.getByText("2026-04-17")).toBeTruthy();
    });

    spy.mockRestore();
  });

  it("shows empty state when the daemon returns an empty list", async () => {
    const spy = spyOn(lileClient, "getSnapshots").mockImplementation(() =>
      Promise.resolve({ snapshots: [] }),
    );

    render(<SnapshotsTab />);

    await waitFor(() => {
      expect(screen.getByText(/no snapshots yet/i)).toBeTruthy();
    });

    spy.mockRestore();
  });
});

describe("lileClient.getSnapshots URL", () => {
  it("hits /api/lile/v1/state/snapshots (matches daemon route)", async () => {
    const fetchSpy = spyOn(globalThis, "fetch").mockImplementation(() =>
      Promise.resolve(
        new Response(JSON.stringify({ snapshots: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await lileClient.getSnapshots();

    expect(fetchSpy.mock.calls.length).toBeGreaterThanOrEqual(1);
    const url = String(fetchSpy.mock.calls[0][0]);
    expect(url).toBe("/api/lile/v1/state/snapshots");

    fetchSpy.mockRestore();
  });
});
