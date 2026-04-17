// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useMemo } from "react";
import type { TrajectoryEvent, TrainStepEvent } from "@/features/lile/api/types";
import { useLileCapsuleStore } from "@/features/lile/stores/lile-capsule-store";

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

// Hook wrappers: subscribe to stable `trajectory` reference and memoize
// the derived series. Using raw selectors with useLileCapsuleStore(selector)
// returns a new array every render, which trips useSyncExternalStore's
// snapshot-stability check and triggers "Maximum update depth exceeded".
const useTrajectory = (): TrajectoryEvent[] =>
  useLileCapsuleStore((s) => s.trajectory);

export function useLossSeries(): Point[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectLossSeries({ trajectory }), [trajectory]);
}

export function useGradNormSeries(): Point[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectGradNormSeries({ trajectory }), [trajectory]);
}

export function useKlSeries(): Point[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectKlSeries({ trajectory }), [trajectory]);
}

export function useQueueDepthSeries(): Point[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectQueueDepthSeries({ trajectory }), [trajectory]);
}

export function useComponentsSeries(): { key: string; points: Point[] }[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectComponentsSeries({ trajectory }), [trajectory]);
}
