// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { authFetch } from "@/features/auth";
import { useTrainingConfigStore } from "@/features/training";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import type { ReactElement, ReactNode } from "react";

// Debounced exact-token count from the backend tokenizer cache.
// Falls back silently to the chars/4 heuristic if the model has no
// tokenizer yet (e.g. selectedModel is null, gated, or just renamed
// out of HF). Cached server-side per process so steady state is ~0ms.
function useExactTokenCount(
  text: string,
  model: string | null,
  hfToken: string,
): { tokens: number | null; loading: boolean } {
  const [tokens, setTokens] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!model || !text) {
      setTokens(null);
      return;
    }
    const handle = setTimeout(() => {
      setLoading(true);
      void authFetch("/api/train/count-tokens", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          model,
          hf_token: hfToken.trim() || null,
        }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (data && typeof data.tokens === "number") setTokens(data.tokens);
          else setTokens(null);
        })
        .catch(() => setTokens(null))
        .finally(() => setLoading(false));
    }, 400);
    return () => clearTimeout(handle);
  }, [text, model, hfToken]);
  return { tokens, loading };
}

function Row({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
        {label}
        {tooltip && (
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                className="text-foreground/70 hover:text-foreground"
              >
                <HugeiconsIcon
                  icon={InformationCircleIcon}
                  className="size-3"
                />
              </button>
            </TooltipTrigger>
            <TooltipContent>{tooltip}</TooltipContent>
          </Tooltip>
        )}
      </span>
      {children}
    </div>
  );
}

export function PromptBakingSection(): ReactElement {
  const store = useTrainingConfigStore();
  const { tokens: exactTokens, loading: tokensLoading } = useExactTokenCount(
    store.bakingSystemPrompt,
    store.selectedModel,
    store.hfToken,
  );

  return (
    <div className="flex flex-col gap-4 pt-1 animate-in fade-in-0 slide-in-from-bottom-1 duration-200">
      {/* System Prompt */}
      <div className="flex flex-col gap-2">
        <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
          System Prompt
          <Tooltip>
            <TooltipTrigger asChild={true}>
              <button
                type="button"
                className="text-foreground/70 hover:text-foreground"
              >
                <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
              </button>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              The system prompt to bake into the model weights via KL divergence.
              The model will learn to behave as if this prompt is always active,
              at zero inference-time cost.
            </TooltipContent>
          </Tooltip>
        </span>
        <textarea
          rows={4}
          placeholder="You are a helpful assistant that..."
          value={store.bakingSystemPrompt}
          onChange={(e) => store.setBakingSystemPrompt(e.target.value)}
          className="w-full resize-none rounded-lg border border-border bg-muted/50 px-3 py-2 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-primary/30 placeholder:text-muted-foreground/50"
        />
        <div className="flex items-center justify-between text-[10px]">
          {!store.bakingSystemPrompt.trim() ? (
            <span className="text-destructive">
              System prompt is required for prompt baking.
            </span>
          ) : (
            <span className="text-muted-foreground/70">
              Per-trajectory cost: {store.bakingTrajectoryLength}
              {" tok × "}
              {store.bakingNumTrajectories}
              {" = "}
              {store.bakingTrajectoryLength * store.bakingNumTrajectories}
              {" tok / step"}
            </span>
          )}
          <span
            className="text-muted-foreground/70 tabular-nums"
            title={
              exactTokens != null
                ? `Exact count from ${store.selectedModel}'s tokenizer.`
                : "Estimate uses chars/4. Pick a model to get the exact tokenizer count."
            }
          >
            {exactTokens != null ? (
              <>
                {exactTokens} tokens
                {tokensLoading ? " …" : ""}
              </>
            ) : (
              <>≈ {Math.ceil(store.bakingSystemPrompt.length / 4)} tokens</>
            )}
            {" · "}
            {store.bakingSystemPrompt.length} chars
          </span>
        </div>
      </div>

      {/* Prefill mode toggle */}
      <Row
        label="Use Prefill Data"
        tooltip="Enable if your dataset already contains completed chat histories. Disables trajectory sampling — the model learns from your data directly."
      >
        <button
          type="button"
          onClick={() => store.setBakingUsePrefill(!store.bakingUsePrefill)}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
            store.bakingUsePrefill ? "bg-primary" : "bg-muted-foreground/30"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              store.bakingUsePrefill ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </Row>

      {/* Trajectory params — only shown in online (non-prefill) mode */}
      {!store.bakingUsePrefill && (
        <>
          <Row
            label="Trajectories"
            tooltip="Number of response trajectories sampled per input to estimate the KL divergence. More trajectories = better gradient estimate but slower."
          >
            <input
              type="number"
              value={store.bakingNumTrajectories}
              onChange={(e) => store.setBakingNumTrajectories(Number(e.target.value))}
              min={1}
              max={32}
              step={1}
              className="w-16 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
            />
          </Row>

          <Row
            label="Trajectory Length"
            tooltip="Max tokens per sampled trajectory. Longer trajectories capture more context but use more VRAM."
          >
            <input
              type="number"
              value={store.bakingTrajectoryLength}
              onChange={(e) => store.setBakingTrajectoryLength(Number(e.target.value))}
              min={16}
              max={2048}
              step={16}
              className="w-20 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
            />
          </Row>

          <Row
            label="Sampling Temperature"
            tooltip="Temperature used when sampling the trajectories. Controls diversity of sampled responses."
          >
            <input
              type="number"
              value={store.bakingSamplingTemperature}
              onChange={(e) =>
                store.setBakingSamplingTemperature(Number(e.target.value))
              }
              min={0.1}
              max={5.0}
              step={0.1}
              className="w-20 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
            />
          </Row>
        </>
      )}

      {/* KL Temperature — applies in both modes */}
      <Row
        label="KL Temperature"
        tooltip="Temperature for computing the prompted (teacher) distribution. Higher = softer targets."
      >
        <input
          type="number"
          value={store.bakingTemperature}
          onChange={(e) => store.setBakingTemperature(Number(e.target.value))}
          min={0.1}
          max={5.0}
          step={0.1}
          className="w-20 text-right font-mono text-xs font-medium bg-muted/50 border border-border rounded-lg px-1.5 py-0.5 focus:outline-none focus:ring-1 focus:ring-primary/30 [&::-webkit-inner-spin-button]:appearance-none"
        />
      </Row>
    </div>
  );
}
