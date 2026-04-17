// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { lileClient } from "../api/lile-client";
import { useLileCapsuleStore } from "../stores/lile-capsule-store";

type SnapshotRow = { name: string; created_at?: string };

function normalizeSnapshots(result: unknown): SnapshotRow[] {
  if (Array.isArray(result)) return result as SnapshotRow[];
  if (result && typeof result === "object" && "snapshots" in result) {
    const r = result as { snapshots?: unknown };
    return Array.isArray(r.snapshots) ? (r.snapshots as SnapshotRow[]) : [];
  }
  return [];
}

export function SnapshotsTab(): ReactElement {
  const [snapshots, setSnapshots] = useState<SnapshotRow[]>([]);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [merging, setMerging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  async function fetchSnapshots() {
    try {
      const result = await lileClient.getSnapshots();
      if (!mountedRef.current) return;
      setSnapshots(normalizeSnapshots(result));
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    mountedRef.current = true;
    void fetchSnapshots();
    return () => {
      mountedRef.current = false;
    };
  }, []);

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      await lileClient.postSnapshot(name);
      if (!mountedRef.current) return;
      setName("");
      await fetchSnapshots();
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mountedRef.current) setSaving(false);
    }
  }

  async function handleMerge() {
    setMerging(true);
    setError(null);
    try {
      await lileClient.postMerge();
      const status = await lileClient.getStatus();
      useLileCapsuleStore.getState().setStatus(status);
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (mountedRef.current) setMerging(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Snapshot name"
          className="max-w-xs"
        />
        <Button
          type="button"
          size="sm"
          onClick={handleSave}
          disabled={saving || !name.trim()}
        >
          {saving ? "Saving…" : "Save snapshot"}
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={handleMerge}
          disabled={merging}
        >
          {merging ? "Merging…" : "Merge"}
        </Button>
      </div>

      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Created at</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {snapshots.length === 0 ? (
            <TableRow>
              <TableCell colSpan={2} className="text-muted-foreground text-center">
                No snapshots yet
              </TableCell>
            </TableRow>
          ) : (
            snapshots.map((snap) => (
              <TableRow key={snap.name}>
                <TableCell className="font-mono">{snap.name}</TableCell>
                <TableCell>{snap.created_at ?? "\u2014"}</TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
