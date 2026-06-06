// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormEvent, ReactElement } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { lileClient } from "../../api/lile-client";

export function NtpCard(): ReactElement {
  const [text, setText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleTrain(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await lileClient.postTrain({
        objective: "ntp",
        samples: [{ text }],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleTrain} className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="ntp-text">Text</Label>
        <Textarea
          id="ntp-text"
          placeholder="Enter text for next-token prediction training…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={submitting}
          rows={6}
        />
      </div>

      <Button type="submit" disabled={submitting}>
        {submitting ? "Training…" : "Train"}
      </Button>

      {error && <p className="text-sm text-destructive">{error}</p>}
    </form>
  );
}
