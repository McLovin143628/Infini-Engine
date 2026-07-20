## What & why

<!-- One paragraph: the change and the reason. Link the ROADMAP batch
     (e.g. "P1.2 batch 3") or issue it advances. -->

## How it was verified

<!-- Tests added/updated, and what you ran locally. "CI is green" is the
     floor, not the answer. -->

## Checklist

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, `cargo nextest run --workspace` green
- [ ] `npm run typecheck && npm run lint && npm test` green (if the frontend changed)
- [ ] Three-ring rule respected (`crates/` and `editor/crates/` stay Tauri-free)
- [ ] New IPC payload types live in `inf_editor_core::ipc` with regenerated committed bindings
- [ ] No f32 world coordinates; version pins only in `[workspace.dependencies]`
- [ ] Decision-grade findings recorded in `docs/memos/` (if any)
