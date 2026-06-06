// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { lileClient } from "../api/lile-client";

export interface FeedbackModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: "critique" | "rewrite";
  responseId: string;
  initialText?: string;
  /** Called after a successful submit (before modal closes). */
  onSuccess?: () => void;
  /** Called when a submit attempt fails. */
  onError?: () => void;
}

export function FeedbackModal({
  open,
  onOpenChange,
  mode,
  responseId,
  initialText,
  onSuccess,
  onError,
}: FeedbackModalProps): ReactElement {
  const [text, setText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pre-fill textarea when opening in rewrite mode
  useEffect(() => {
    if (open) {
      setText(mode === "rewrite" && initialText != null ? initialText : "");
      setError(null);
    }
  }, [open, mode, initialText]);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      await lileClient.postFeedback({
        response_id: responseId,
        feedback_kind: mode,
        value: text,
      });
      onSuccess?.();
      onOpenChange(false);
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      onError?.();
    } finally {
      setSubmitting(false);
    }
  }

  function handleCancel() {
    onOpenChange(false);
  }

  const title = mode === "critique" ? "Critique" : "Rewrite";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>

        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={
            mode === "critique"
              ? "Describe what was wrong with this response…"
              : "Write an improved version of this response…"
          }
          disabled={submitting}
          className="min-h-32"
        />

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={handleCancel}
            disabled={submitting}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            onClick={handleSubmit}
            disabled={submitting || text.trim() === ""}
          >
            {submitting ? "Submitting…" : "Submit"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
