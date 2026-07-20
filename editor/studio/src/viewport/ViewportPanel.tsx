import { useEffect, useRef } from "react";
import { viewport } from "../lib/ipc";

/**
 * The viewport "hole": an empty div whose rectangle is mirrored to the native
 * wgpu child window (Spike A). The native window sits ABOVE the webview, so
 * nothing rendered here is ever visible while the engine is attached — the
 * fallback text only shows if native attach fails.
 */
export default function ViewportPanel() {
  const holeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = holeRef.current;
    if (!el) return;

    let raf = 0;
    const report = () => {
      const r = el.getBoundingClientRect();
      const s = window.devicePixelRatio;
      viewport.setRect(r.x * s, r.y * s, r.width * s, r.height * s).catch(() => {});
    };
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(report);
    };

    viewport
      .attach()
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
  }, []);

  return (
    <div
      ref={holeRef}
      data-viewport-hole
      className="flex flex-1 items-center justify-center rounded border border-(--ink-border) bg-(--ink-bg-0)"
    >
      <div className="text-center text-(--ink-text-dim)">
        <div className="mb-2 text-3xl">∞</div>
        <div>Native viewport unavailable — see Output Log</div>
      </div>
    </div>
  );
}
