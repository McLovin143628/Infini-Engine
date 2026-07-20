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

import type { LogLine } from "../bindings/LogLine";
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
