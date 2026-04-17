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
import { useComponentsSeries } from "./chart-utils";

const CHART_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

export function ComponentsChartCard(): ReactElement {
  const series = useComponentsSeries();

  // Merge per-key point arrays into flat { step, [key]: value }[] shape.
  // Use a Map keyed on step so a component named "step" cannot clobber it.
  const byStep = new Map<number, Record<string, number>>();
  for (const { key, points } of series) {
    for (const { step, value } of points) {
      let row = byStep.get(step);
      if (!row) {
        row = {};
        byStep.set(step, row);
      }
      row[key] = value;
    }
  }
  const data = Array.from(byStep.entries())
    .sort(([a], [b]) => a - b)
    .map(([step, row]) => ({ step, ...row }));

  const chartConfig: ChartConfig = Object.fromEntries(
    series.map(({ key }, i) => [
      key,
      { label: key, color: CHART_COLORS[i % CHART_COLORS.length] },
    ]),
  );

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Loss components</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="aspect-video w-full">
          <LineChart data={data}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="step" tickLine={false} axisLine={false} tickMargin={8} fontSize={10} />
            <YAxis tickLine={false} axisLine={false} tickMargin={8} fontSize={10} />
            <ChartTooltip content={<ChartTooltipContent />} />
            {series.map(({ key }, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={CHART_COLORS[i % CHART_COLORS.length]}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
