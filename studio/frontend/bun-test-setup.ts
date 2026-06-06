// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Register a happy-dom Window onto globalThis so React + testing-library can
// render into a real DOM. happy-dom v20 no longer ships @happy-dom/global-
// registrator, so we manually mirror the essentials ourselves.

// happy-dom's GlobalWindow mirrors the JS intrinsics (SyntaxError, etc.) onto
// itself as class fields; BrowserWindow does not, which breaks selector
// parsing. Always use GlobalWindow here.
import { GlobalWindow } from "happy-dom";

const win = new GlobalWindow({ url: "http://localhost/" });

// Proxy every enumerable property from the happy-dom Window onto globalThis.
// We do this once at preload time so subsequent imports see the DOM.
const g = globalThis as unknown as Record<string, unknown>;
const src = win as unknown as Record<string, unknown>;

for (const key of Object.getOwnPropertyNames(win)) {
  if (key in g) continue;
  try {
    g[key] = src[key];
  } catch {
    // ignore read-only / symbol-only properties
  }
}

// Explicit aliases that React / testing-library rely on.
g.window = win;
g.document = win.document;
g.navigator = win.navigator;
g.HTMLElement = win.HTMLElement;
g.HTMLAnchorElement = win.HTMLAnchorElement;
g.HTMLSpanElement = win.HTMLSpanElement;
g.HTMLDivElement = win.HTMLDivElement;
g.Element = win.Element;
g.Node = win.Node;
g.Event = win.Event;
g.CustomEvent = win.CustomEvent;
g.getComputedStyle = win.getComputedStyle.bind(win);
g.requestAnimationFrame = win.requestAnimationFrame.bind(win);
g.cancelAnimationFrame = win.cancelAnimationFrame.bind(win);

// React 19 checks for IS_REACT_ACT_ENVIRONMENT; set it so `act` warnings stay quiet.
(g as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
