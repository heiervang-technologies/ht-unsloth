// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  refreshSession,
  setMustChangePassword,
} from "@/features/auth";

/** Cache the auth-disabled flag so we only check once. */
let _authDisabled: boolean | null = null;

async function isAuthDisabled(): Promise<boolean> {
  if (_authDisabled !== null) return _authDisabled;
  try {
    const res = await fetch("/api/auth/status");
    if (!res.ok) return false;
    const data = (await res.json()) as {
      initialized: boolean;
      requires_password_change: boolean;
      auth_disabled?: boolean;
    };
    // Auth is disabled when backend says initialized=true,
    // requires_password_change=false, and we have no stored token.
    // The definitive signal is auth_disabled field if present,
    // otherwise infer from the combination.
    _authDisabled = data.auth_disabled === true;
    return _authDisabled;
  } catch {
    return false;
  }
}

async function hasActiveSession(): Promise<boolean> {
  if (await isAuthDisabled()) return true;
  if (hasAuthToken()) return true;
  if (!hasRefreshToken()) return false;
  return refreshSession();
}

interface AuthStatus {
  initialized: boolean;
  requires_password_change: boolean;
}

async function fetchAuthStatus(): Promise<AuthStatus> {
  try {
    const res = await fetch(apiUrl("/api/auth/status"));
    if (!res.ok) return { initialized: true, requires_password_change: mustChangePassword() };
    const status = (await res.json()) as AuthStatus;
    // Server truth wins; keep localStorage in sync both ways.
    if (status.requires_password_change !== mustChangePassword()) {
      setMustChangePassword(status.requires_password_change);
    }
    return status;
  } catch {
    return { initialized: true, requires_password_change: mustChangePassword() };
  }
}

function authRedirect(to: "/login" | "/change-password"): never {
  throw redirect({ to });
}

export async function requireAuth(): Promise<void> {
  if (isTauri) {
    // AppProvider owns backend startup + desktop auth; route guards run before it mounts.
    return;
  }

  if (await hasActiveSession()) {
    const { requires_password_change } = await fetchAuthStatus();
    if (requires_password_change || mustChangePassword()) {
      authRedirect("/change-password");
    }
    return;
  }
  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) {
    authRedirect("/change-password");
  }
  authRedirect(status.initialized ? "/login" : "/change-password");
}

export async function requireGuest(): Promise<void> {
  if (isTauri) {
    throw redirect({ to: "/chat" });
  }
  if (!(await hasActiveSession())) return;
  // Reconcile localStorage before routing.
  await fetchAuthStatus();
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  if (isTauri) {
    throw redirect({ to: "/chat" });
  }

  const status = await fetchAuthStatus();
  if (status.requires_password_change || mustChangePassword()) return;
  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }
  authRedirect(status.initialized ? "/login" : "/change-password");
}
