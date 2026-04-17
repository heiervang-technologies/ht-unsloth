// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { CapsuleStatusStrip } from "@/features/lile/components/capsule-status-strip";
import { CapsuleLoadForm } from "@/features/lile/components/capsule-load-form";
import { LossChartCard } from "@/features/lile/components/charts/loss-chart-card";
import { GradNormChartCard } from "@/features/lile/components/charts/grad-norm-chart-card";
import { KlDivergenceChartCard } from "@/features/lile/components/charts/kl-divergence-chart-card";
import { QueueDepthChartCard } from "@/features/lile/components/charts/queue-depth-chart-card";
import { ComponentsChartCard } from "@/features/lile/components/charts/components-chart-card";
import { ChatSftCard } from "@/features/lile/components/train-tab/chat-sft-card";
import { NtpCard } from "@/features/lile/components/train-tab/ntp-card";
import { ReinforceCard } from "@/features/lile/components/train-tab/reinforce-card";
import { AdvancedJsonCard } from "@/features/lile/components/train-tab/advanced-json-card";
import { TrajectoryTab } from "@/features/lile/components/trajectory-tab";
import { SnapshotsTab } from "@/features/lile/components/snapshots-tab";
import { useLileStatusPoll } from "@/features/lile/hooks/use-lile-status-poll";
import { useLileTrajectoryPoll } from "@/features/lile/hooks/use-lile-trajectory-poll";
import { useLileCapsuleStore } from "@/features/lile/stores/lile-capsule-store";

export function LilePage(): ReactElement {
  useLileStatusPoll({ enabled: true });
  const status = useLileCapsuleStore((s) => s.status);
  useLileTrajectoryPoll({ enabled: status?.running === true });

  return (
    <div className="flex flex-col gap-4 p-4">
      <CapsuleStatusStrip />

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <LossChartCard />
        <GradNormChartCard />
        <KlDivergenceChartCard />
        <QueueDepthChartCard />
      </div>
      <ComponentsChartCard />

      {!status?.running && (
        <Card>
          <CardContent className="pt-6">
            <CapsuleLoadForm />
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="train">
        <TabsList>
          <TabsTrigger value="train">Train</TabsTrigger>
          <TabsTrigger value="trajectory">Trajectory</TabsTrigger>
          <TabsTrigger value="snapshots">Snapshots</TabsTrigger>
        </TabsList>

        <TabsContent value="train" className="mt-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Supervised fine-tuning</CardTitle>
              </CardHeader>
              <CardContent>
                <ChatSftCard />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Next-token-prediction</CardTitle>
              </CardHeader>
              <CardContent>
                <NtpCard />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Reinforce (replay feedback)</CardTitle>
              </CardHeader>
              <CardContent>
                <ReinforceCard />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Advanced JSON</CardTitle>
              </CardHeader>
              <CardContent>
                <AdvancedJsonCard />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="trajectory" className="mt-4">
          <TrajectoryTab />
        </TabsContent>

        <TabsContent value="snapshots" className="mt-4">
          <SnapshotsTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
