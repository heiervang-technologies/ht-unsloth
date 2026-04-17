// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormEvent, ReactElement } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { lileClient } from "../../api/lile-client";

export function AdvancedJsonCard(): ReactElement {
  const [raw, setRaw] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    setError(null);

    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch (parseErr) {
      setError(
        `JSON parse error: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`,
      );
      return;
    }

    setSubmitting(true);
    try {
      await lileClient.postTrain(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSend} className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="advanced-json">Raw JSON</Label>
        <Textarea
          id="advanced-json"
          placeholder={`{"objective":"sft","samples":[{"messages":[...]}]}`}
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          disabled={submitting}
          rows={8}
          className="font-mono text-xs"
        />
      </div>

      <Button type="submit" disabled={submitting}>
        {submitting ? "Sending…" : "Send"}
      </Button>

      {error && <p className="text-sm text-destructive">{error}</p>}
    </form>
  );
}
