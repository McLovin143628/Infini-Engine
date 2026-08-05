/**
 * Viewport identity (P23.2a).
 *
 * The backend's `ViewportState` is a keyed map, not a single slot: every
 * `viewport_*` command takes an optional id and every `viewport://…` event
 * carries one. This module holds the one well-known key so the shell, the
 * airspace refcount and the event filters cannot spell it three different ways.
 *
 * Mirrors `commands::viewport::PRIMARY_VIEWPORT` in Ring 2. There is exactly
 * one viewport today; the second native host is P23.2b.
 */

/** The scene viewport — the shell's centre hole, and the PIE embed target. */
export const PRIMARY_VIEWPORT = "primary";

/**
 * True when an event belongs to the viewport we care about.
 *
 * Deliberately **total on the empty string**: Ring 2 stamps the id on the way
 * out, and an unstamped payload (`""`) is a backend bug rather than a message
 * for somebody else — dropping it silently is how a status bar goes quiet with
 * nothing to point at. It is treated as Primary, and the Rust side has a unit
 * test (`the_sink_stamps_the_viewport_id_onto_a_tool_status`) so the case
 * should never arise.
 */
export function isPrimaryViewport(id: string | undefined): boolean {
  return id === undefined || id === "" || id === PRIMARY_VIEWPORT;
}
