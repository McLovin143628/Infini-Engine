import { useEffect, useRef } from "react";
import { viewport } from "../lib/ipc";
import { toPhysicalRect } from "../lib/viewportRect";
import { PRIMARY_VIEWPORT } from "../lib/viewportIds";
import { registerViewport } from "../lib/viewportOverlay";
import { useSimStore } from "../stores/simStore";
import ViewportToolbar from "./ViewportToolbar";

/**
 * The viewport "hole": an empty div whose rectangle is mirrored to the native
 * wgpu child window (Spike A). The native window sits ABOVE the webview, so
 * nothing rendered here is ever visible while the engine is attached — the
 * fallback text only shows if native attach fails.
 *
 * The viewport toolbar (P8.2c) renders as a strip ABOVE the hole — never over
 * it, since the native child window would occlude any HTML crossing the hole
 * (airspace rule). The measured hole excludes the toolbar automatically.
 *
 * **A registered panel type** (P23.2a): `registerPanels.tsx` declares it, so
 * `panelDefFor("viewport")` resolves and the focus/undo routing can name it
 * like any other panel. The shell still mounts it in the dock's CENTRE cell
 * rather than in a region — the native child window has to track one invariant
 * rectangle (Spike A), and a tab strip that unmounts it would tear the surface
 * down. Registration is what makes the *second* viewport a layout change
 * instead of surgery.
 *
 * `params` carries the viewport id for a future non-singleton instance
 * (`viewport:model`); absent means the scene viewport.
 */
export default function ViewportPanel({ params }: { panelId?: string; params?: string | null } = {}) {
  const holeRef = useRef<HTMLDivElement>(null);
  const id = params || PRIMARY_VIEWPORT;
  // Simulate (P8.4): a live session tints the viewport frame.
  const running = useSimStore((s) => s.running);

  // Tell the airspace refcount this viewport exists (P23.2a): window-wide
  // overlays hide every attached viewport, and one that mounts while the
  // palette is already open must come up hidden rather than punch through it.
  useEffect(() => registerViewport(id), [id]);

  useEffect(() => {
    const el = holeRef.current;
    if (!el) return;

    let raf = 0;
    const report = () => {
      const rect = toPhysicalRect(el.getBoundingClientRect(), window.devicePixelRatio);
      viewport.setRect(rect, id).catch(() => {});
    };
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(report);
    };

    viewport
      .attach(id)
      .then(report)
      .catch((e) => console.error("viewport attach failed:", e));

    const ro = new ResizeObserver(schedule);
    ro.observe(el);
    // Window moves between monitors / DPI changes arrive as resize events.
    window.addEventListener("resize", schedule);

    // Cross-monitor drags can change devicePixelRatio WITHOUT a resize event
    // (same CSS layout, new scale). matchMedia against the current dpr fires
    // exactly when it changes; re-arm for the next value each time.
    let mql: MediaQueryList | null = null;
    const onDprChange = () => {
      schedule();
      armDprListener();
    };
    const armDprListener = () => {
      mql?.removeEventListener("change", onDprChange);
      mql = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      mql.addEventListener("change", onDprChange);
    };
    armDprListener();

    return () => {
      ro.disconnect();
      window.removeEventListener("resize", schedule);
      mql?.removeEventListener("change", onDprChange);
      cancelAnimationFrame(raf);
    };
  }, [id]);

  return (
    <div className="flex flex-1 flex-col gap-1">
      <ViewportToolbar />
      <div
        ref={holeRef}
        data-viewport-hole
        className={`flex flex-1 items-center justify-center rounded border bg-(--ink-bg-0) ${
          running ? "border-(--ink-success)" : "border-(--ink-border)"
        }`}
      >
        <div className="text-center text-(--ink-text-dim)">
          <div className="mb-2 text-3xl">∞</div>
          <div>Native viewport unavailable — see Output Log</div>
        </div>
      </div>
    </div>
  );
}
