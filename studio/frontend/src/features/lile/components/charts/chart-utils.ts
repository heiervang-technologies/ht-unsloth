// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrajectoryEvent, TrainStepEvent } from "@/features/lile/api/types";

type StoreState = {
  trajectory: TrajectoryEvent[];
};

type Point = { step: number; value: number };

function isTrainStep(e: TrajectoryEvent): e is TrainStepEvent {
  return e.kind === "train_step";
}

export function selectLossSeries(s: StoreState): Point[] {
  return s.trajectory
    .filter(isTrainStep)
    .map((e) => ({ step: e.batch_id, value: e.loss }));
}

export function selectGradNormSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  for (const e of s.trajectory) {
    if (!isTrainStep(e)) continue;
    if (typeof e.grad_norm_total === "number") {
      points.push({ step: e.batch_id, value: e.grad_norm_total });
    }
  }
  return points;
}

export function selectKlSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  for (const e of s.trajectory) {
    if (!isTrainStep(e) || !e.components) continue;
    const kl = e.components["batch.kl.loss"] ?? e.components["kl"] ?? null;
    if (typeof kl === "number" && Number.isFinite(kl)) {
      points.push({ step: e.batch_id, value: kl });
    }
  }
  return points;
}

const EMPTY_POINTS: Point[] = [];

// placeholder: wired in Task 17. Stable reference prevents zustand
// re-subscribing on every trajectory update.
export function selectQueueDepthSeries(_s: StoreState): Point[] {
  return EMPTY_POINTS;
}

export function selectComponentsSeries(
  s: StoreState,
): { key: string; points: Point[] }[] {
  const keyMap = new Map<string, Point[]>();
  for (const e of s.trajectory) {
    if (!isTrainStep(e) || !e.components) continue;
    for (const [key, val] of Object.entries(e.components)) {
      if (typeof val !== "number") continue;
      if (!keyMap.has(key)) keyMap.set(key, []);
      keyMap.get(key)!.push({ step: e.batch_id, value: val });
    }
  }
  return Array.from(keyMap.entries()).map(([key, points]) => ({ key, points }));
}
