// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { FeedbackEvent, TrainStepEvent, TrajectoryEvent } from "../api/types";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

function kindVariant(kind: string): "default" | "secondary" | "outline" {
  if (kind === "train_step") return "default";
  if (kind === "feedback") return "secondary";
  return "outline";
}

function eventLabel(event: TrajectoryEvent): string {
  if (event.kind === "train_step") {
    const e = event as TrainStepEvent;
    return `batch_id ${e.batch_id}`;
  }
  return `offset ${event.offset}`;
}

function eventSummary(event: TrajectoryEvent): string {
  if (event.kind === "train_step") {
    const e = event as TrainStepEvent;
    const gradNorm = e.grad_norm_total != null ? e.grad_norm_total.toFixed(3) : "\u2014";
    return `loss=${e.loss.toFixed(4)} grad_norm=${gradNorm}`;
  }
  if (event.kind === "feedback") {
    const e = event as FeedbackEvent;
    return `${e.response_id.slice(0, 8)} ${e.feedback_kind}`;
  }
  return "";
}

type ReplayButtonProps = {
  event: FeedbackEvent;
  replaying: boolean;
  onReplay: (event: FeedbackEvent) => void;
};

function ReplayButton({ event, replaying, onReplay }: ReplayButtonProps): ReactElement {
  return (
    <Button
      type="button"
      size="xs"
      variant="outline"
      onClick={() => onReplay(event)}
      disabled={replaying}
    >
      {replaying ? "Replaying\u2026" : "Replay"}
    </Button>
  );
}

export function TrajectoryTab(): ReactElement {
  const trajectory = useLileCapsuleStore((s) => s.trajectory);
  const [replayingIds, setReplayingIds] = useState<Record<string, boolean>>({});
  const [replayErrors, setReplayErrors] = useState<Record<string, string>>({});

  const events = trajectory.slice(-200).reverse();

  async function handleReplay(event: FeedbackEvent) {
    const key = event.response_id;
    setReplayingIds((prev) => ({ ...prev, [key]: true }));
    setReplayErrors((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    try {
      await lileClient.postFeedback({
        response_id: event.response_id,
        feedback_kind: event.feedback_kind,
        value: event.value,
      });
    } catch (err) {
      setReplayErrors((prev) => ({
        ...prev,
        [key]: err instanceof Error ? err.message : String(err),
      }));
    } finally {
      setReplayingIds((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
    }
  }

  if (events.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No trajectory events yet</p>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      {events.map((event) => {
        const isFeedback = event.kind === "feedback";
        const feedbackEvent = isFeedback ? (event as FeedbackEvent) : null;
        const replayKey = feedbackEvent?.response_id ?? "";
        const summary = eventSummary(event);

        return (
          <details
            key={event.offset}
            className="rounded border px-3 py-2 text-sm"
          >
            <summary className="flex items-center gap-2 cursor-pointer list-none">
              <Badge variant={kindVariant(event.kind)}>{event.kind}</Badge>
              <span className="font-mono text-xs text-muted-foreground">
                {eventLabel(event)}
              </span>
              {summary && (
                <span className="text-xs text-foreground">{summary}</span>
              )}
              {feedbackEvent && (
                <span className="ml-auto flex items-center gap-2">
                  {replayErrors[replayKey] && (
                    <span className="text-xs text-destructive">
                      {replayErrors[replayKey]}
                    </span>
                  )}
                  <ReplayButton
                    event={feedbackEvent}
                    replaying={!!replayingIds[replayKey]}
                    onReplay={handleReplay}
                  />
                </span>
              )}
            </summary>
            <pre className="mt-2 overflow-x-auto text-xs text-muted-foreground whitespace-pre-wrap">
              {JSON.stringify(event, null, 2)}
            </pre>
          </details>
        );
      })}
    </div>
  );
}
