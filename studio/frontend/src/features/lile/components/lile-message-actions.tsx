// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { lileClient } from "../api/lile-client";
import { FeedbackModal } from "./feedback-modal";

type ButtonKey = "up" | "down" | "critique" | "rewrite";
type ButtonStatus = "idle" | "pending" | "ok" | "err";

export interface LileMessageActionsProps {
  responseId: string;
  commitCursor: number;
  latencyS?: number;
  assistantText?: string;
}

export function LileMessageActions({
  responseId,
  commitCursor,
  latencyS,
  assistantText,
}: LileMessageActionsProps): ReactElement {
  const [statuses, setStatuses] = useState<Record<ButtonKey, ButtonStatus>>({
    up: "idle",
    down: "idle",
    critique: "idle",
    rewrite: "idle",
  });

  const [modalOpen, setModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<"critique" | "rewrite">("critique");

  // Per-button reset timers
  const timers = useRef<Partial<Record<ButtonKey, ReturnType<typeof setTimeout>>>>({});

  // Clean up all timers on unmount
  useEffect(() => {
    const t = timers.current;
    return () => {
      for (const handle of Object.values(t)) {
        if (handle != null) clearTimeout(handle);
      }
    };
  }, []);

  const setButtonStatus = useCallback((key: ButtonKey, status: ButtonStatus) => {
    setStatuses((prev) => ({ ...prev, [key]: status }));
  }, []);

  const scheduleReset = useCallback(
    (key: ButtonKey) => {
      if (timers.current[key] != null) {
        clearTimeout(timers.current[key]);
      }
      timers.current[key] = setTimeout(() => {
        setButtonStatus(key, "idle");
        timers.current[key] = undefined;
      }, 1000);
    },
    [setButtonStatus],
  );

  const flashResult = useCallback(
    (key: ButtonKey, succeeded: boolean) => {
      setButtonStatus(key, succeeded ? "ok" : "err");
      scheduleReset(key);
    },
    [setButtonStatus, scheduleReset],
  );

  async function handleBinary(value: boolean) {
    const key: ButtonKey = value ? "up" : "down";
    if (timers.current[key] != null) {
      clearTimeout(timers.current[key]);
      timers.current[key] = undefined;
    }
    setButtonStatus(key, "pending");
    try {
      await lileClient.postFeedback({
        response_id: responseId,
        feedback_kind: "binary",
        value,
      });
      flashResult(key, true);
    } catch {
      flashResult(key, false);
    }
  }

  function handleOpenModal(mode: "critique" | "rewrite") {
    setModalMode(mode);
    setModalOpen(true);
  }

  function renderButtonContent(key: ButtonKey, icon: string): string {
    const s = statuses[key];
    if (s === "ok") return "ok";
    if (s === "err") return "err";
    return icon;
  }

  function buttonVariantFor(key: ButtonKey): "ghost" | "outline" {
    const s = statuses[key];
    return s === "ok" || s === "err" ? "outline" : "ghost";
  }

  function buttonExtraClass(key: ButtonKey): string {
    const s = statuses[key];
    if (s === "ok") return "text-green-600 border-green-600";
    if (s === "err") return "text-red-600 border-red-600";
    return "";
  }

  return (
    <>
      <div className="flex items-center gap-1">
        <Button
          type="button"
          size="icon-sm"
          variant={buttonVariantFor("up")}
          className={buttonExtraClass("up")}
          disabled={statuses.up === "pending"}
          onClick={() => handleBinary(true)}
          aria-label="Thumbs up"
        >
          {renderButtonContent("up", "👍")}
        </Button>

        <Button
          type="button"
          size="icon-sm"
          variant={buttonVariantFor("down")}
          className={buttonExtraClass("down")}
          disabled={statuses.down === "pending"}
          onClick={() => handleBinary(false)}
          aria-label="Thumbs down"
        >
          {renderButtonContent("down", "👎")}
        </Button>

        <Button
          type="button"
          size="icon-sm"
          variant={buttonVariantFor("critique")}
          className={buttonExtraClass("critique")}
          disabled={statuses.critique === "pending"}
          onClick={() => handleOpenModal("critique")}
          aria-label="Critique"
        >
          {renderButtonContent("critique", "💬")}
        </Button>

        <Button
          type="button"
          size="icon-sm"
          variant={buttonVariantFor("rewrite")}
          className={buttonExtraClass("rewrite")}
          disabled={statuses.rewrite === "pending"}
          onClick={() => handleOpenModal("rewrite")}
          aria-label="Rewrite"
        >
          {renderButtonContent("rewrite", "✎")}
        </Button>

        <span className="text-xs text-muted-foreground ml-1 select-none">
          commit={commitCursor}
          {latencyS != null && ` · ${latencyS.toFixed(2)}s`}
        </span>
      </div>

      <FeedbackModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        mode={modalMode}
        responseId={responseId}
        initialText={modalMode === "rewrite" ? assistantText : undefined}
        onSuccess={() => flashResult(modalMode, true)}
        onError={() => flashResult(modalMode, false)}
      />
    </>
  );
}
