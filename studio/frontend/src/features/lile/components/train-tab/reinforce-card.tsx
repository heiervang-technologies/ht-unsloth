// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { FeedbackEvent } from "../../api/types";
import { lileClient } from "../../api/lile-client";
import { useLileCapsuleStore } from "../../stores/lile-capsule-store";

export function ReinforceCard(): ReactElement {
  const feedbackEvents = useLileCapsuleStore((s) =>
    s.trajectory
      .filter((e): e is FeedbackEvent => e.kind === "feedback")
      .slice(-10),
  );

  const [replayingId, setReplayingId] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  async function handleReplay(event: FeedbackEvent) {
    setReplayingId(event.response_id);
    setErrors((prev) => {
      const next = { ...prev };
      delete next[event.response_id];
      return next;
    });
    try {
      await lileClient.postFeedback({
        response_id: event.response_id,
        feedback_kind: event.feedback_kind,
        value: event.value,
      });
    } catch (err) {
      setErrors((prev) => ({
        ...prev,
        [event.response_id]: err instanceof Error ? err.message : String(err),
      }));
    } finally {
      setReplayingId(null);
    }
  }

  if (feedbackEvents.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No feedback events yet</p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {feedbackEvents.map((event) => (
        <div
          key={event.response_id}
          className="flex items-center justify-between rounded border px-3 py-2 gap-2"
        >
          <div className="flex flex-col gap-0.5 min-w-0">
            <span className="text-xs font-mono truncate">{event.response_id}</span>
            <span className="text-xs text-muted-foreground">{event.feedback_kind}</span>
          </div>
          <div className="flex flex-col items-end gap-1">
            <Button
              size="sm"
              variant="outline"
              onClick={() => handleReplay(event)}
              disabled={replayingId === event.response_id}
            >
              {replayingId === event.response_id ? "Replaying…" : "Replay"}
            </Button>
            {errors[event.response_id] && (
              <p className="text-xs text-destructive">{errors[event.response_id]}</p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
