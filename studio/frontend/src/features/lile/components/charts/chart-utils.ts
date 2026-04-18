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

function stepOf(e: TrainStepEvent, fallbackIndex: number): number {
  return typeof e.commit_token === "number" ? e.commit_token : fallbackIndex;
}

function numericComponent(
  e: TrainStepEvent,
  key: string,
): number | null {
  const v = e.components?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function selectLossSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  let i = 0;
  for (const e of s.trajectory) {
    if (!isTrainStep(e)) continue;
    points.push({ step: stepOf(e, i), value: e.loss });
    i += 1;
  }
  return points;
}

export function selectGradNormSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  let i = 0;
  for (const e of s.trajectory) {
    if (!isTrainStep(e)) continue;
    const gn = numericComponent(e, "grad_norm_total");
    if (gn !== null) points.push({ step: stepOf(e, i), value: gn });
    i += 1;
  }
  return points;
}

export function selectKlSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  let i = 0;
  for (const e of s.trajectory) {
    if (!isTrainStep(e)) continue;
    const kl =
      numericComponent(e, "batch.kl.loss") ?? numericComponent(e, "kl");
    if (kl !== null) points.push({ step: stepOf(e, i), value: kl });
    i += 1;
  }
  return points;
}

const EMPTY_POINTS: Point[] = [];

// placeholder: wired in Task 17. Stable reference prevents zustand
// re-subscribing on every trajectory update.
export function selectQueueDepthSeries(_s: StoreState): Point[] {
  return EMPTY_POINTS;
}

export function selectAdapterNormSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  let i = 0;
  for (const e of s.trajectory) {
    if (!isTrainStep(e)) continue;
    const v = numericComponent(e, "adapter_norm_total");
    if (v !== null) points.push({ step: stepOf(e, i), value: v });
    i += 1;
  }
  return points;
}

export function selectResidualNormSeries(s: StoreState): Point[] {
  const points: Point[] = [];
  let i = 0;
  for (const e of s.trajectory) {
    if (!isTrainStep(e)) continue;
    const v = numericComponent(e, "residual_norm_total");
    if (v !== null) points.push({ step: stepOf(e, i), value: v });
    i += 1;
  }
  return points;
}

export function selectComponentsSeries(
  s: StoreState,
): { key: string; points: Point[] }[] {
  const keyMap = new Map<string, Point[]>();
  let i = 0;
  for (const e of s.trajectory) {
    if (!isTrainStep(e) || !e.components) continue;
    for (const [key, val] of Object.entries(e.components)) {
      if (typeof val !== "number" || !Number.isFinite(val)) continue;
      if (!keyMap.has(key)) keyMap.set(key, []);
      keyMap.get(key)!.push({ step: stepOf(e, i), value: val });
    }
    i += 1;
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

export function useAdapterNormSeries(): Point[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectAdapterNormSeries({ trajectory }), [trajectory]);
}

export function useResidualNormSeries(): Point[] {
  const trajectory = useTrajectory();
  return useMemo(() => selectResidualNormSeries({ trajectory }), [trajectory]);
}
