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
    return fetch(`${BASE}/capsule/status`).then(json);
  },
  postStart(body: StartRequest) {
    return fetch(`${BASE}/capsule/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json);
  },
  postStop() {
    return fetch(`${BASE}/capsule/stop`, { method: "POST" }).then(json);
  },
  getTrajectoryTail(sinceOffset: number): Promise<TrajectoryTail> {
    const q = sinceOffset > 0 ? `?since_offset=${sinceOffset}` : "";
    return fetch(`${BASE}/v1/state/trajectory/tail${q}`).then(json);
  },
  postTrain(body: unknown) {
    return fetch(`${BASE}/v1/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json);
  },
  postFeedback(body: unknown) {
    return fetch(`${BASE}/v1/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json);
  },
  getSnapshots() {
    return fetch(`${BASE}/v1/state/snapshot/list`).then(json);
  },
  postSnapshot(name: string) {
    return fetch(`${BASE}/v1/state/snapshot/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then(json);
  },
  postMerge() {
    return fetch(`${BASE}/v1/state/merge`, { method: "POST" }).then(json);
  },
};
