/**
 * Zustand store architecture (ROADMAP P1.3.3).
 *
 * Conventions:
 * - One store per domain under `src/stores/`, created with `create<State>()`.
 *   State + actions live together; actions are defined inside the store so
 *   non-React code can drive them via `useXStore.getState()`.
 * - Stores never call `invoke`/`listen` directly — they go through the
 *   typed wrappers in `lib/ipc.ts` / `lib/events.ts`.
 * - Persistent UI state (dock layout, theme) round-trips through backend
 *   commands or localStorage explicitly — no implicit persist middleware,
 *   so what's saved (and when) stays auditable.
 * - Detached panel windows (P1.2.4) mirror store slices over the IPC store
 *   bridge; a store that must sync across windows registers itself with
 *   the bridge rather than importing window plumbing itself.
 */
export { useShellStore } from "./shellStore";
