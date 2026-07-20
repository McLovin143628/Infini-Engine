/**
 * Typed IPC layer — the ONLY place `invoke` is called.
 *
 * Convention (engine-wide): every backend `#[tauri::command]` gets a typed
 * wrapper here, grouped by domain. Components import these wrappers, never
 * `invoke` directly. Events use namespaced channels (`log://line`,
 * `viewport://rect`, `assets://changed/{id}`, …) via lib/events.ts.
 */
import { invoke } from "@tauri-apps/api/core";

export const app = {
  /** Studio backend version, shown in the status bar. */
  version: (): Promise<string> => invoke<string>("app_version"),
};
