// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

export function CapsuleStatusStrip() {
  const status = useLileCapsuleStore((s) => s.status);
  if (!status || !status.running) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="secondary">offline</Badge>
        <span>Lile daemon not reachable</span>
      </div>
    );
  }
  const h = status.health;
  return (
    <div className="flex items-center gap-4 text-sm">
      <Badge variant="default">online</Badge>
      <span className="font-mono">{h.model}</span>
      <span>queue {h.queue_depth}</span>
      <span>commit {h.commit_cursor}</span>
      <span>merges {h.merges}</span>
      {status.externally_managed && <Badge variant="outline">external</Badge>}
    </div>
  );
}
