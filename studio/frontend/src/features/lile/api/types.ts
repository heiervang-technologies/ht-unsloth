// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type HealthReport = {
  ok: boolean;
  model: string;
  queue_depth: number;
  commit_cursor: number;
  merges: number;
};

// Capsule mode (set by studio/backend/routes/lile.py:capsule_status):
//   external      — daemon reachable at LILE_DAEMON_URL; we don't own its lifecycle
//   spawned       — Studio launched the daemon as a subprocess; we own SIGTERM
//   external-unreachable — env points somewhere but /health doesn't respond
//   unconfigured  — no LILE_DAEMON_URL / LILE_HOST set
export type CapsuleMode =
  | "external"
  | "spawned"
  | "external-unreachable"
  | "unconfigured";

export type CapsuleStatus =
  | { running: false; mode?: CapsuleMode; url?: string; pid?: number;
      error?: string }
  | { running: true; mode?: CapsuleMode; externally_managed?: boolean;
      health: HealthReport; url: string; pid?: number };

export type TrainStepEvent = {
  offset: number;
  kind: "train_step";
  batch_id: string;
  objective: string;
  loss: number;
  batch_size: number;
  commit_token?: number;
  components?: Record<string, number | boolean>;
  ts?: number;
};

export type FeedbackEvent = {
  offset: number;
  kind: "feedback";
  response_id: string;
  feedback_kind: "binary" | "critique" | "rewrite";
  value?: unknown;
  ts?: number;
};

export type TrajectoryEvent = TrainStepEvent | FeedbackEvent |
  { offset: number; kind: string; [k: string]: unknown };

export type TrajectoryTail =
  | { events: TrajectoryEvent[]; next_offset: number; total_size: number }
  | { events: TrajectoryEvent[] };  // old shape, back-compat

export type ChatLileBlock = {
  response_id: string;
  commit_cursor: number;
  latency_s: number;
};

export type StartRequest = {
  model?: string;
  max_seq_length?: number;
  lora_rank?: number;
  load_in_4bit?: boolean;
  idle_replay?: boolean;
  frozen_ref?: boolean;
};
