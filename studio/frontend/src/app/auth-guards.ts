// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { redirect } from "@tanstack/react-router";
import {
  getPostAuthRoute,
  hasAuthToken,
  hasRefreshToken,
  mustChangePassword,
  refreshSession,
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

async function checkAuthInitialized(): Promise<boolean> {
  try {
    const res = await fetch("/api/auth/status");
    if (!res.ok) return true; // fallback to login on error
    const data = (await res.json()) as { initialized: boolean };
    return data.initialized;
  } catch {
    return true; // fallback to login on error
  }
}

async function checkPasswordChangeRequired(): Promise<boolean> {
  if (await isAuthDisabled()) return false;
  try {
    const res = await fetch("/api/auth/status");
    if (!res.ok) return mustChangePassword();
    const data = (await res.json()) as { requires_password_change: boolean };
    return data.requires_password_change || mustChangePassword();
  } catch {
    return mustChangePassword();
  }
}

export async function requireAuth(): Promise<void> {
  if (await hasActiveSession()) {
    if (await checkPasswordChangeRequired()) {
      throw redirect({ to: "/change-password" });
    }
    return;
  }
  const requiresPasswordChange = await checkPasswordChangeRequired();
  if (requiresPasswordChange) throw redirect({ to: "/change-password" });
  const initialized = await checkAuthInitialized();
  throw redirect({ to: initialized ? "/login" : "/change-password" });
}

export async function requireGuest(): Promise<void> {
  if (!(await hasActiveSession())) return;
  throw redirect({ to: getPostAuthRoute() });
}

export async function requirePasswordChangeFlow(): Promise<void> {
  const requiresPasswordChange = await checkPasswordChangeRequired();

  if (requiresPasswordChange) return;

  if (await hasActiveSession()) {
    throw redirect({ to: getPostAuthRoute() });
  }

  const initialized = await checkAuthInitialized();
  throw redirect({ to: initialized ? "/login" : "/change-password" });
}
