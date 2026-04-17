// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type HealthReport = {
  ok: boolean;
  model: string;
  queue_depth: number;
  commit_cursor: number;
  merges: number;
};

export type CapsuleStatus =
  | { running: false }
  | { running: true; externally_managed: boolean; health: HealthReport;
      url: string };

export type TrainStepEvent = {
  offset: number;
  kind: "train_step";
  batch_id: number;
  objective: string;
  loss: number;
  batch_size: number;
  commit_token?: number;
  grad_norm_total?: number;
  grad_clipped?: boolean;
  components?: Record<string, number>;
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
