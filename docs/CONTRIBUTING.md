# Contributing

Ground rules for working on Infinity Engine. The architecture itself is
specified in [ROADMAP.md](ROADMAP.md) — read §2 (architecture) and the phase
you are touching before writing code.

## Commits

Conventional-commit prefixes, imperative subject, ≤ 72 chars:

```
feat(viewport): reparent PIE window into the layout hole
fix(hotreload): keep old state when deserialize fails mid-reload
docs(roadmap): mark P0.2 done
refactor / test / build / ci / chore / perf
```

- Scope = crate or area without the `inf-` prefix (`viewport`, `transpile`,
  `studio`, `ci`).
- A commit compiles and passes tests on its own. Spike/phase milestones get a
  body explaining what was proven, not just what changed.

## Branches & protection

Small batches land directly on `main` while the project is pre-1.0 —
**only with the full local gate green** (below). Anything risky or
multi-day goes on a `feat/…` branch and merges via PR.

Recommended GitHub branch-protection settings for `main` once collaborators
join (repo → Settings → Branches):

- Require status checks: `Rust (windows-latest)`, `Rust (macos-latest)`,
  `Rust (ubuntu-latest)`, `cargo-deny`, `Frontend`, `TS bindings drift`.
- Require branches to be up to date before merging; no force pushes.

## The local gate

Run before every push (CI enforces the same set on 3 OSes):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets   # warnings are errors in CI
cargo nextest run --workspace
cargo test --workspace --doc
cargo deny check
cd editor/studio && npm run typecheck && npm run lint && npm test && npm run build
```

## Non-negotiables (enforced by review + CI)

1. **Three rings**: `crates/` (Ring 0) and `editor/crates/` (Ring 1) never
   name Tauri/webview concepts.
2. **Typed IPC only**: commands in per-domain modules, frontend goes through
   `src/lib/ipc.ts`; payload types derive `serde` + `ts_rs::TS` in
   `inf_editor_core::ipc`, and the committed bindings under
   `editor/studio/src/bindings/` are regenerated with
   `cargo test -p inf-editor-core --test bindings` (CI fails on drift).
3. **f64 world / f32 render** — never introduce f32 world coordinates.
4. **Version pins live once** in `[workspace.dependencies]`; member crates
   use `{ workspace = true }`. New licenses/advisory ignores need a
   rationale comment in `deny.toml`.
5. Hard-won platform rules (wnd_proc reentrancy, message pumping while
   hosting foreign windows, never-unload dylibs, …) live in the memos under
   `docs/memos/` — read them before touching the areas they cover.
