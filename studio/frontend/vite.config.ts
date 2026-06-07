// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    include: ["@dagrejs/dagre", "@dagrejs/graphlib"],
  },
  server: {
    host: "0.0.0.0",
    // Pinned off 5173 so heierchat's desktop launcher
    // (tauri-plugin-localhost hardcodes :5173) doesn't AddrInUse-panic
    // when our Vite dev server is up. strictPort=true to fail loudly
    // if 5174 itself gets taken, instead of silently drifting upward.
    port: 5174,
    strictPort: true,
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8888",
        changeOrigin: true,
      },
      "/v1": {
        target: "http://127.0.0.1:8888",
        changeOrigin: true,
      },
      "/seed/inspect": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/seed/preview": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/preview": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/validate": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
      "/tools": {
        target: "http://127.0.0.1:8004",
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@dagrejs/dagre": path.resolve(
        __dirname,
        "./node_modules/@dagrejs/dagre/dist/dagre.cjs.js",
      ),
    },
  },
  build: {
    commonjsOptions: {
      include: [/node_modules/, /@dagrejs\/dagre/, /@dagrejs\/graphlib/],
    },
  },
});
