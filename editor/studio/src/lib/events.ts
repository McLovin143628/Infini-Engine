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
import type { CaptureProgressDto } from "../bindings/CaptureProgressDto";
import type { ImportEventDto } from "../bindings/ImportEventDto";
import type { LogLine } from "../bindings/LogLine";
import type { ProjectInfoDto } from "../bindings/ProjectInfoDto";
import type { SceneDelta } from "../bindings/SceneDelta";
import type { ViewportGizmoDto } from "../bindings/ViewportGizmoDto";
import type { ViewportKey } from "../bindings/ViewportKey";
import type { ViewportToolStatusDto } from "../bindings/ViewportToolStatusDto";

export type { UnlistenFn };

/**
 * Play-In-Editor session state broadcast on `pie://state` (P9.4). Hand-mirrored
 * from `commands::pie::PieStateEvent` (Ring-2-only shape, no ts-rs binding).
 */
export interface PieStateEvent {
  /** A player subprocess is live (playing or paused). */
  running: boolean;
  /** Live but the player's fixed step is frozen. */
  paused: boolean;
  /** `"embedded"`, `"window"`, or `""` when stopped. */
  mode: string;
  /** The player's current fixed-step frame count. */
  frame: number;
  /** A crash / load message when the session died unexpectedly. */
  error: string | null;
}

/**
 * One handler's Blueprint debug observation from a Simulate step (B-P4 tier A′),
 * broadcast on `sim://debug`. Hand-mirrored from `simulate::SimDebugHit`.
 *
 * For hand-built `.inf_act` classes the `hits`/`wires` are keyed by IR `LocalId`
 * (there is no canvas `NodeId` map yet — that activates with graph provenance).
 */
export interface SimDebugEvent {
  classId: string;
  event: string;
  fnName: string;
  /** Breakpoint hits, as IR `LocalId` values. */
  hits: number[];
  /** Latest captured value per wire, `[LocalId, stringified]`. */
  wires: [number, string][];
}

/** Payload types per fixed channel. Extend as backends start emitting. */
export interface EventPayloads {
  /** Structured tracing output → Output Log panel (P1.4). */
  "log://line": LogLine;
  /**
   * Global-shortcut chord forwarded from the native viewport (P2.3.4).
   *
   * The three `viewport://` channels are **shared by every viewport** and their
   * payloads carry a `viewport` id (P23.2a) — one subscription whatever the
   * count, with the listener filtering. See `lib/viewportIds.ts`.
   */
  "viewport://key": ViewportKey;
  /** Transform-gizmo mode echoed from the viewport (W/E/R or IPC) → toolbar (Wave 2). */
  "viewport://gizmo": ViewportGizmoDto;
  /**
   * A viewport tool raised a status message and/or the projected terrain's
   * streamed state changed (P16.4a): the message goes to the status bar, the
   * flag greys out sculpt/paint (a streamed terrain has no editable working set
   * in the document yet).
   */
  "viewport://tool-status": ViewportToolStatusDto;
  /** Incremental world change after any mutation (P3.2). */
  "world://delta": SceneDelta;
  /** Content changed (import/delete/rename/watcher) → re-fetch snapshot (P4.4). */
  "assets://changed": AssetChanged;
  /** Import-job progress (P4.2.4). */
  "assets://import": ImportEventDto;
  /**
   * Capture-wizard stage progress (P25.4).
   *
   * Its own channel rather than a case of `assets://import`: an import job is
   * one unit of work with a source file, and a capture is FIVE stages with
   * different progress models (per photograph, per finish step, and two that
   * honestly report none). Folding it in would have made every existing reader
   * of that channel learn a shape it has no use for.
   */
  "photogrammetry://progress": CaptureProgressDto;
  /** A named content collection changed (create/rename/delete/add/remove) → re-fetch (E-P8). */
  "collections://changed": null;
  /** The project audio mixer was saved → re-fetch the Audio Mixer panel (E-P9). */
  "audio://mixer-changed": null;
  /** A project was opened/created → leave the start screen, re-sync (P5.5). */
  "project://changed": ProjectInfoDto;
  /** A blueprint graph mutated (payload = graph id) → re-sync canvas (P6.2). */
  "graph://sync": string;
  /** A material graph mutated (payload = material id) → re-sync canvas (P7.2). */
  "material://sync": string;
  /** Simulate started (`true`) / stopped (`false`) → sync the play toolbar (P8.4). */
  "sim://state": boolean;
  /** Blueprint debug hits/wire-values from a Simulate step (B-P4 tier A′). */
  "sim://debug": SimDebugEvent[];
  /** Play-In-Editor session state (running/paused/mode/frame/error) (P9.4). */
  "pie://state": PieStateEvent;
  /** A cook/package run started (`true`) / finished (`false`) (P9.2). */
  "package://state": boolean;
  /** A sequence file changed (payload = sequence name) → re-sync the Sequencer (P11.4). */
  "seq://changed": string;
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

// ── Language server (P5.2) ──────────────────────────────────────────────────

/** A minimal LSP diagnostic (the shape rust-analyzer publishes). */
export interface LspDiagnostic {
  range: { start: LspPos; end: LspPos };
  severity?: number; // 1 error, 2 warning, 3 info, 4 hint
  message: string;
  source?: string;
  code?: string | number;
}
export interface LspPos {
  line: number;
  character: number;
}
/** `lsp://diagnostics` payload. */
export interface LspDiagnosticsEvent {
  uri: string;
  diagnostics: LspDiagnostic[];
}

/** Subscribe to published diagnostics (one per file, replacing the prior set). */
export function onLspDiagnostics(handler: (p: LspDiagnosticsEvent) => void): Promise<UnlistenFn> {
  return listenToDynamic<LspDiagnosticsEvent>("lsp://diagnostics", handler);
}
/** Subscribe to server start/stop (payload `{ language }`). */
export function onLspStarted(handler: (p: { language: string }) => void): Promise<UnlistenFn> {
  return listenToDynamic<{ language: string }>("lsp://started", handler);
}
export function onLspStopped(handler: (p: { language: string }) => void): Promise<UnlistenFn> {
  return listenToDynamic<{ language: string }>("lsp://stopped", handler);
}
