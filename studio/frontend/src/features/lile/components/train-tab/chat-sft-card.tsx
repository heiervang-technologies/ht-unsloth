// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormEvent, ReactElement } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { lileClient } from "../../api/lile-client";

type Role = "user" | "assistant";
type Row = { role: Role; content: string };

const defaultRows: Row[] = [
  { role: "user", content: "" },
  { role: "assistant", content: "" },
];

export function ChatSftCard(): ReactElement {
  const [rows, setRows] = useState<Row[]>(defaultRows);
  const [weight, setWeight] = useState(1.0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function updateRow(idx: number, field: keyof Row, value: string) {
    setRows((prev) =>
      prev.map((r, i) => (i === idx ? { ...r, [field]: value } : r)),
    );
  }

  function addTurn() {
    setRows((prev) => {
      const last = prev[prev.length - 1];
      const nextRole: Role = last?.role === "user" ? "assistant" : "user";
      return [...prev, { role: nextRole, content: "" }];
    });
  }

  async function handleTrain(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const userRow = rows.find((r) => r.role === "user" && r.content.trim().length > 0);
      const assistantRow = rows.find((r) => r.role === "assistant" && r.content.trim().length > 0);
      if (!userRow || !assistantRow) {
        setError("Need one non-empty user turn and one non-empty assistant turn.");
        return;
      }
      const objective = weight === 1.0 ? "sft" : "weighted_sft";
      const sample: { prompt: string; response: string; weight?: number } = {
        prompt: userRow.content,
        response: assistantRow.content,
      };
      if (objective === "weighted_sft") sample.weight = weight;
      await lileClient.postTrain({
        objective,
        samples: [sample],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleTrain} className="flex flex-col gap-4">
      <div className="flex flex-col gap-3">
        {rows.map((row, idx) => {
          // Only the first user and first assistant row get the labeled IDs for test/a11y
          const isFirstUser = row.role === "user" && rows.findIndex((r) => r.role === "user") === idx;
          const isFirstAssistant = row.role === "assistant" && rows.findIndex((r) => r.role === "assistant") === idx;
          const textareaId =
            isFirstUser
              ? "sft-user-0"
              : isFirstAssistant
                ? "sft-assistant-0"
                : `sft-${row.role}-${idx}`;

          const turnLabel = `Turn ${idx + 1}`;
          return (
            <div key={idx} className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                {isFirstUser ? (
                  <Label htmlFor={textareaId}>User</Label>
                ) : isFirstAssistant ? (
                  <Label htmlFor={textareaId}>Assistant</Label>
                ) : (
                  <select
                    aria-label={`${turnLabel} role`}
                    value={row.role}
                    onChange={(e) => updateRow(idx, "role", e.target.value)}
                    className="text-sm border rounded px-1 py-0.5"
                    disabled={submitting}
                  >
                    <option value="user">User</option>
                    <option value="assistant">Assistant</option>
                  </select>
                )}
              </div>
              <Textarea
                id={textareaId}
                aria-label={
                  isFirstUser || isFirstAssistant
                    ? undefined
                    : `${turnLabel} ${row.role} content`
                }
                placeholder={row.role === "user" ? "User message…" : "Assistant message…"}
                value={row.content}
                onChange={(e) => updateRow(idx, "content", e.target.value)}
                disabled={submitting}
                rows={3}
              />
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-2">
        <Button type="button" variant="outline" onClick={addTurn} disabled={submitting}>
          Add turn
        </Button>
        <div className="flex-1" />
        <Label htmlFor="sft-weight" className="text-xs text-muted-foreground">
          strength
        </Label>
        <Input
          id="sft-weight"
          type="number"
          min={0.1}
          max={10}
          step="any"
          value={weight}
          onChange={(e) => setWeight(Number.parseFloat(e.target.value) || 1.0)}
          disabled={submitting}
          className="w-20"
          aria-label="Training sample strength (weight)"
        />
        <Button type="submit" disabled={submitting}>
          {submitting ? "Training…" : "Train"}
        </Button>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}
    </form>
  );
}
