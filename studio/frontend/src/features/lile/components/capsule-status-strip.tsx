// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

export function CapsuleStatusStrip() {
  const status = useLileCapsuleStore((s) => s.status);
  if (!status || !status.running) {
    const mode = status?.mode;
    const label = mode === "unconfigured"
      ? "Lile not configured (set LILE_DAEMON_URL)"
      : mode === "external-unreachable"
      ? `Lile not reachable at ${status?.url ?? "configured URL"}`
      : "Lile daemon not reachable";
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="secondary">offline</Badge>
        <span>{label}</span>
      </div>
    );
  }
  const h = status.health;
  // Prefer the new `mode` field; fall back to legacy `externally_managed`
  // for any client that hasn't refreshed since the spawn-or-connect rework.
  const mode = status.mode
    ?? (status.externally_managed ? "external" : undefined);
  return (
    <div className="flex items-center gap-4 text-sm">
      <Badge variant="default">online</Badge>
      <span className="font-mono">{h.model}</span>
      <span>queue {h.queue_depth}</span>
      <span>commit {h.commit_cursor}</span>
      <span>merges {h.merges}</span>
      {mode === "external" && <Badge variant="outline">external</Badge>}
      {mode === "spawned" && (
        <Badge variant="outline">
          spawned{status.pid ? ` (pid ${status.pid})` : ""}
        </Badge>
      )}
    </div>
  );
}
