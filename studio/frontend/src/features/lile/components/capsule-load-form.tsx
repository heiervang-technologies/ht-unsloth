// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

const MODEL_SUGGESTIONS = [
  "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
  "unsloth/Llama-3.2-1B-unsloth-bnb-4bit",
];

export function CapsuleLoadForm(): ReactElement {
  const status = useLileCapsuleStore((s) => s.status);
  const running = status?.running === true;

  const [model, setModel] = useState("");
  const [maxSeqLength, setMaxSeqLength] = useState(2048);
  const [loraRank, setLoraRank] = useState(16);
  const [loadIn4bit, setLoadIn4bit] = useState(true);
  const [idleReplay, setIdleReplay] = useState(false);
  const [frozenRef, setFrozenRef] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleStart(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const body = {
        ...(model.trim() !== "" ? { model: model.trim() } : {}),
        max_seq_length: maxSeqLength,
        lora_rank: loraRank,
        load_in_4bit: loadIn4bit,
        idle_replay: idleReplay,
        frozen_ref: frozenRef,
      };
      await lileClient.postStart(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleStop() {
    setSubmitting(true);
    setError(null);
    try {
      await lileClient.postStop();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  if (running) {
    return (
      <div className="flex flex-col gap-2">
        <Button
          variant="destructive"
          onClick={handleStop}
          disabled={submitting}
        >
          {submitting ? "Stopping…" : "Stop capsule"}
        </Button>
        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
      </div>
    );
  }

  return (
    <form onSubmit={handleStart} className="flex flex-col gap-4">
      {/* model */}
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="lile-model">Model</Label>
        <Input
          id="lile-model"
          type="text"
          list="lile-model-suggestions"
          placeholder="(use server default)"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={submitting}
        />
        <datalist id="lile-model-suggestions">
          {MODEL_SUGGESTIONS.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      </div>

      {/* max_seq_length */}
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="lile-max-seq-length">Max sequence length</Label>
        <Input
          id="lile-max-seq-length"
          type="number"
          min={1}
          value={maxSeqLength}
          onChange={(e) => setMaxSeqLength(Number(e.target.value))}
          disabled={submitting}
        />
      </div>

      {/* lora_rank */}
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="lile-lora-rank">LoRA rank</Label>
        <Input
          id="lile-lora-rank"
          type="number"
          min={1}
          value={loraRank}
          onChange={(e) => setLoraRank(Number(e.target.value))}
          disabled={submitting}
        />
      </div>

      {/* load_in_4bit */}
      <div className="flex items-center gap-2">
        <Switch
          id="lile-load-in-4bit"
          checked={loadIn4bit}
          onCheckedChange={setLoadIn4bit}
          disabled={submitting}
        />
        <Label htmlFor="lile-load-in-4bit">Load in 4-bit</Label>
      </div>

      {/* idle_replay */}
      <div className="flex items-center gap-2">
        <Switch
          id="lile-idle-replay"
          checked={idleReplay}
          onCheckedChange={setIdleReplay}
          disabled={submitting}
        />
        <Label htmlFor="lile-idle-replay">Idle replay</Label>
      </div>

      {/* frozen_ref */}
      <div className="flex items-center gap-2">
        <Switch
          id="lile-frozen-ref"
          checked={frozenRef}
          onCheckedChange={setFrozenRef}
          disabled={submitting}
        />
        <Label htmlFor="lile-frozen-ref">Frozen reference model</Label>
      </div>

      <div className="flex flex-col gap-2">
        <Button type="submit" disabled={submitting}>
          {submitting ? "Loading…" : "Load capsule"}
        </Button>
        {submitting && (
          <p className="text-sm text-muted-foreground">
            Loading model… this can take 2 min
          </p>
        )}
        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
      </div>
    </form>
  );
}
