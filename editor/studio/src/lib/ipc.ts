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
  /** Editor backend version, shown in the status bar. */
  version: (): Promise<string> => invoke<string>("app_version"),
};

export const viewport = {
  /** Create (once) the native engine viewport inside this window. */
  attach: (): Promise<void> => invoke("viewport_attach"),
  /**
   * Report the viewport hole's rectangle in PHYSICAL pixels relative to the
   * window client area (callers multiply CSS px by devicePixelRatio).
   */
  setRect: (x: number, y: number, width: number, height: number): Promise<void> =>
    invoke("viewport_set_rect", { x, y, width, height }),
};
