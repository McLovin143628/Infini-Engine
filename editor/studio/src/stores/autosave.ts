/**
 * Crash-recovery autosave (P3.5.4), re-armable from the editor preferences
 * (Wave E, batch A).
 *
 * This used to be a `setInterval(…, 5000)` inside `App`'s effect body, which
 * meant the period was a literal nobody could change. It lives here now so the
 * setting can be **proved to apply**: the loop re-arms the moment
 * `autosave_interval_s` changes, and a headless test can drive that with fake
 * timers (`autosave.test.ts`) — an arm on the behaviour, not on the round trip.
 *
 * Failure reporting keeps the rate limit it always had: autosave is not
 * command-initiated, so its failures surface in the status bar, but a persistent
 * failure re-toasts only when the message changes or once a minute.
 */
import { scene as sceneIpc } from "../lib/ipc";
import { useSettingsStore } from "./settingsStore";
import { useShellStore } from "./shellStore";

/** Re-toast a repeating failure only every Nth tick (≈1 min at 5 s). */
const REPEAT_TOAST_EVERY = 12;

/** The interval in ms the loop should use for `settings`, guarded. */
export function autosaveIntervalMs(intervalS: number): number {
  // The backend normalizes to [1, 3600] s, but this is the door a value reaches
  // a timer through, and `setInterval(NaN)` fires as fast as the event loop can.
  if (!Number.isFinite(intervalS) || intervalS <= 0) return 5000;
  return Math.round(Math.min(Math.max(intervalS, 1), 3600) * 1000);
}

/**
 * Start the autosave loop and keep it armed at the configured period. Returns a
 * disposer (StrictMode-safe).
 */
export function initAutosave(): () => void {
  let lastErr: string | null = null;
  let failCount = 0;
  let timer: ReturnType<typeof setInterval> | undefined;
  let armedMs = -1;

  const tick = () => {
    void sceneIpc.autosave().then(
      () => {
        lastErr = null;
        failCount = 0;
      },
      (e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        failCount += 1;
        if (msg !== lastErr || failCount % REPEAT_TOAST_EVERY === 0) {
          lastErr = msg;
          useShellStore.getState().pushStatus(`Autosave failed: ${msg}`);
        }
      },
    );
  };

  const arm = (ms: number) => {
    if (ms === armedMs) return;
    armedMs = ms;
    if (timer !== undefined) clearInterval(timer);
    timer = setInterval(tick, ms);
  };

  arm(autosaveIntervalMs(useSettingsStore.getState().settings.autosave_interval_s));
  const unsub = useSettingsStore.subscribe((s) => {
    arm(autosaveIntervalMs(s.settings.autosave_interval_s));
  });

  return () => {
    unsub();
    if (timer !== undefined) clearInterval(timer);
    timer = undefined;
    armedMs = -1;
  };
}
