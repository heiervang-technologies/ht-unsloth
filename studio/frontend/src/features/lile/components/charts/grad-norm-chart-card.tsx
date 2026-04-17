// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import { useLileCapsuleStore } from "@/features/lile/stores/lile-capsule-store";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { selectGradNormSeries } from "./chart-utils";

const chartConfig = {
  value: { label: "Grad norm", color: "var(--chart-2)" },
} satisfies ChartConfig;

export function GradNormChartCard(): ReactElement {
  const data = useLileCapsuleStore(selectGradNormSeries);

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Grad norm</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="aspect-video w-full">
          <LineChart data={data}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="step" tickLine={false} axisLine={false} tickMargin={8} fontSize={10} />
            <YAxis tickLine={false} axisLine={false} tickMargin={8} fontSize={10} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Line
              type="monotone"
              dataKey="value"
              stroke="var(--color-value)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
