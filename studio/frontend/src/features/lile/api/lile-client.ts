// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CapsuleStatus, StartRequest, TrajectoryTail } from "./types";

const BASE = "/api/lile";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`lile ${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const lileClient = {
  getStatus(): Promise<CapsuleStatus> {
    return fetch(`${BASE}/capsule/status`).then((r) => json<CapsuleStatus>(r));
  },
  postStart(body: StartRequest): Promise<unknown> {
    return fetch(`${BASE}/capsule/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<unknown>(r));
  },
  postStop(): Promise<unknown> {
    return fetch(`${BASE}/capsule/stop`, { method: "POST" }).then((r) =>
      json<unknown>(r),
    );
  },
  getTrajectoryTail(sinceOffset: number): Promise<TrajectoryTail> {
    const q = sinceOffset > 0 ? `?since_offset=${sinceOffset}` : "";
    return fetch(`${BASE}/v1/state/trajectory/tail${q}`).then((r) =>
      json<TrajectoryTail>(r),
    );
  },
  postTrain(body: unknown): Promise<unknown> {
    return fetch(`${BASE}/v1/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<unknown>(r));
  },
  postFeedback(body: unknown): Promise<unknown> {
    return fetch(`${BASE}/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => json<unknown>(r));
  },
  getSnapshots(): Promise<unknown> {
    return fetch(`${BASE}/v1/state/snapshot/list`).then((r) => json<unknown>(r));
  },
  postSnapshot(name: string): Promise<unknown> {
    return fetch(`${BASE}/v1/state/snapshot/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => json<unknown>(r));
  },
  postMerge(): Promise<unknown> {
    return fetch(`${BASE}/v1/state/merge`, { method: "POST" }).then((r) =>
      json<unknown>(r),
    );
  },
};
