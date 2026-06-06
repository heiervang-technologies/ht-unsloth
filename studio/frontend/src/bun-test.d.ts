// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Minimal ambient types for `bun:test` — we run tests via `bun test`, which
// ships its own type resolution at runtime, but `tsc -b` needs a shim so
// `*.test.ts(x)` files type-check as part of the app build. The @types/bun
// package pulls in Bun's global runtime (conflicts with DOM fetch), so we
// declare only the subset of the test API our suite uses.
declare module "bun:test" {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  type Fn = (...args: any[]) => any;

  export const describe: (name: string, fn: () => void) => void;
  export const it: ((name: string, fn: Fn) => void) & {
    skip: (name: string, fn: Fn) => void;
    only: (name: string, fn: Fn) => void;
  };
  export const test: typeof it;
  export const beforeAll: (fn: Fn) => void;
  export const afterAll: (fn: Fn) => void;
  export const beforeEach: (fn: Fn) => void;
  export const afterEach: (fn: Fn) => void;

  type Matchers = {
    toBe: (v: unknown) => void;
    toEqual: (v: unknown) => void;
    toMatchObject: (v: unknown) => void;
    toBeTruthy: () => void;
    toBeFalsy: () => void;
    toBeNull: () => void;
    toBeUndefined: () => void;
    toBeGreaterThan: (v: number) => void;
    toBeGreaterThanOrEqual: (v: number) => void;
    toBeLessThan: (v: number) => void;
    toBeLessThanOrEqual: (v: number) => void;
    toBeCloseTo: (v: number, precision?: number) => void;
    toContain: (v: unknown) => void;
    toHaveLength: (n: number) => void;
    toHaveBeenCalled: () => void;
    toHaveBeenCalledTimes: (n: number) => void;
    toHaveBeenCalledWith: (...args: unknown[]) => void;
    toThrow: (msg?: string | RegExp) => void;
    not: Matchers;
    resolves: Matchers;
    rejects: Matchers;
  };

  export function expect(actual: unknown): Matchers;

  type MockFn<T extends Fn = Fn> = T & {
    mock: {
      calls: unknown[][];
      results: { type: "return" | "throw"; value: unknown }[];
      instances: unknown[];
    };
    mockImplementation: (impl: T) => MockFn<T>;
    mockImplementationOnce: (impl: T) => MockFn<T>;
    mockReturnValue: (v: unknown) => MockFn<T>;
    mockResolvedValue: (v: unknown) => MockFn<T>;
    mockRejectedValue: (v: unknown) => MockFn<T>;
    mockClear: () => MockFn<T>;
    mockReset: () => MockFn<T>;
    mockRestore: () => void;
  };

  export function mock<T extends Fn>(impl?: T): MockFn<T>;
  export function spyOn<T, K extends keyof T>(
    obj: T,
    method: K,
  ): T[K] extends Fn ? MockFn<T[K]> : MockFn;

  export const jest: {
    fn: typeof mock;
    useFakeTimers: () => void;
    useRealTimers: () => void;
    advanceTimersByTime: (ms: number) => void;
    runAllTimers: () => void;
    clearAllTimers: () => void;
  };
}
