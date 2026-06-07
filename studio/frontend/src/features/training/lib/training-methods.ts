// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DatasetFormat, TrainingMethod, TrainingObjective } from "@/types/training";

const BACKEND_TRAINING_TYPE: Record<TrainingMethod, string> = {
  qlora: "LoRA/QLoRA",
  lora: "LoRA/QLoRA",
  full: "Full Finetuning",
};

const TRAINING_METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full",
};

const TRAINING_OBJECTIVE_LABELS: Record<TrainingObjective, string> = {
  sft: "SFT",
  cpt: "CPT",
  "prompt-baking": "Baking",
};

// CPT takes precedence over the weight strategy on the backend's
// `training_type` field — the trainer routes on training_type first, then
// reads is_prompt_baking + load_in_4bit + use_lora to choose the rest.
export function toBackendTrainingType(
  trainingMethod: TrainingMethod,
  objective: TrainingObjective,
): string {
  if (objective === "cpt") return "Continued Pretraining";
  return BACKEND_TRAINING_TYPE[trainingMethod];
}

export function getTrainingMethodLabel(
  trainingMethod: TrainingMethod | string,
): string {
  if (Object.prototype.hasOwnProperty.call(TRAINING_METHOD_LABELS, trainingMethod)) {
    return TRAINING_METHOD_LABELS[trainingMethod as TrainingMethod];
  }
  // Back-compat: legacy persisted runs encoded cpt / prompt-baking on this
  // field. Map them to a label so historical-training-view doesn't break.
  if (trainingMethod === "cpt") return TRAINING_OBJECTIVE_LABELS.cpt;
  if (trainingMethod === "prompt-baking") return TRAINING_OBJECTIVE_LABELS["prompt-baking"];
  return TRAINING_METHOD_LABELS.full;
}

export function getTrainingObjectiveLabel(objective: TrainingObjective): string {
  return TRAINING_OBJECTIVE_LABELS[objective];
}

export function parseBackendTrainingMethod(
  trainingType: unknown,
  loadIn4Bit: unknown,
): TrainingMethod {
  if (trainingType === "Continued Pretraining") {
    // CPT can run with QLoRA / LoRA / Full on the backend; the strategy is
    // recoverable from load_in_4bit + use_lora. Best-effort guess from the
    // 4-bit flag (full+CPT is rare; the prior UI defaulted to LoRA 4-bit).
    return loadIn4Bit ? "qlora" : "lora";
  }
  if (trainingType === "LoRA/QLoRA") {
    return loadIn4Bit ? "qlora" : "lora";
  }
  return "full";
}

export function parseBackendTrainingObjective(
  trainingType: unknown,
  isPromptBaking: unknown,
): TrainingObjective {
  if (trainingType === "Continued Pretraining") return "cpt";
  if (isPromptBaking) return "prompt-baking";
  return "sft";
}

export function isRawTextDatasetFormat(
  datasetFormat: DatasetFormat,
): boolean {
  return datasetFormat === "raw";
}
