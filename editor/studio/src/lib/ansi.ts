/**
 * Minimal ANSI escape handling for the Output Log (P1.4.4). Backend
 * tracing lines are structured (no ANSI), but tool output piped through
 * the log later (cargo, cook) carries SGR color codes — strip them for
 * display and search rather than rendering garbage.
 */

// CSI sequences (SGR colors, cursor moves) + OSC titles (BEL/ST-terminated).
// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?/g;

export function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, "");
}
