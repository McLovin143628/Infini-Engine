/**
 * Namespaced event helpers — the ONLY place `listen`/`emit` from
 * `@tauri-apps/api/event` is called (mirror of the ipc.ts rule for
 * `invoke`).
 *
 * Channels are namespaced `domain://topic` (ROADMAP §2.4): `log://line`,
 * `viewport://rect`, `assets://changed/{id}`, `play://state`,
 * `graph://sync/{id}`. Fixed channels get a typed entry in
 * `EventPayloads`; parameterized channels get a helper that builds the
 * channel string and casts once, here, instead of at every call site.
 */
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type { AssetChanged } from "../bindings/AssetChanged";
import type { ImportEventDto } from "../bindings/ImportEventDto";
import type { LogLine } from "../bindings/LogLine";
import type { ProjectInfoDto } from "../bindings/ProjectInfoDto";
import type { SceneDelta } from "../bindings/SceneDelta";
import type { ViewportKey } from "../bindings/ViewportKey";

export type { UnlistenFn };

/** Payload types per fixed channel. Extend as backends start emitting. */
export interface EventPayloads {
  /** Structured tracing output → Output Log panel (P1.4). */
  "log://line": LogLine;
  /** Global-shortcut chord forwarded from the native viewport (P2.3.4). */
  "viewport://key": ViewportKey;
  /** Incremental world change after any mutation (P3.2). */
  "world://delta": SceneDelta;
  /** Content changed (import/delete/rename/watcher) → re-fetch snapshot (P4.4). */
  "assets://changed": AssetChanged;
  /** Import-job progress (P4.2.4). */
  "assets://import": ImportEventDto;
  /** A project was opened/created → leave the start screen, re-sync (P5.5). */
  "project://changed": ProjectInfoDto;
}

export type EventChannel = keyof EventPayloads;

/** Subscribe to a fixed, typed channel. */
export function listenTo<C extends EventChannel>(
  channel: C,
  handler: (payload: EventPayloads[C]) => void,
): Promise<UnlistenFn> {
  return listen<EventPayloads[C]>(channel, (event) => handler(event.payload));
}

/**
 * Subscribe to a parameterized channel (`assets://changed/{id}`-style).
 * The caller supplies the payload type; keep the cast confined here.
 */
export function listenToDynamic<T>(
  channel: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  return listen<T>(channel, (event) => handler(event.payload));
}

// ── Terminal (P5.3): per-session parameterized channels ────────────────────

/** `pty://output/{id}` payload — base64-encoded shell bytes. */
export interface PtyOutput {
  id: string;
  data: string;
}
/** `pty://exit/{id}` payload. */
export interface PtyExit {
  id: string;
  code: number | null;
}

/** Subscribe to a session's output stream. */
export function onPtyOutput(id: string, handler: (p: PtyOutput) => void): Promise<UnlistenFn> {
  return listenToDynamic<PtyOutput>(`pty://output/${id}`, handler);
}
/** Subscribe to a session's exit. */
export function onPtyExit(id: string, handler: (p: PtyExit) => void): Promise<UnlistenFn> {
  return listenToDynamic<PtyExit>(`pty://exit/${id}`, handler);
}
