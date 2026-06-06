// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import type { ChartConfig } from "@/components/ui/chart";
import type { ReactElement } from "react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { useAdapterNormSeries, useResidualNormSeries } from "./chart-utils";

const chartConfig = {
  adapter: { label: "LoRA adapter (live)", color: "var(--chart-1)" },
  residual: { label: "Merged residual", color: "var(--chart-3)" },
} satisfies ChartConfig;

export function AdapterNormChartCard(): ReactElement {
  const adapter = useAdapterNormSeries();
  const residual = useResidualNormSeries();

  const byStep = new Map<number, { step: number; adapter?: number; residual?: number }>();
  for (const p of adapter) {
    byStep.set(p.step, { ...(byStep.get(p.step) ?? { step: p.step }), adapter: p.value });
  }
  for (const p of residual) {
    byStep.set(p.step, { ...(byStep.get(p.step) ?? { step: p.step }), residual: p.value });
  }
  const data = Array.from(byStep.values()).sort((a, b) => a.step - b.step);

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Adapter norm</CardTitle>
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
              dataKey="adapter"
              stroke="var(--color-adapter)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="residual"
              stroke="var(--color-residual)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
