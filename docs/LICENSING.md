# Licensing — options for the owner's decision

**Status: OPEN DECISION.** This document lays out the licensing options for Infinity Engine so
the project owner can make an informed, deliberate choice. **No license is picked unilaterally
here.** Until this is resolved, the codebase carries a consistent *placeholder* license field
(`MIT OR Apache-2.0`) in `[workspace.package]` so tooling (cargo-deny, crates metadata) stays
coherent; changing the decision means updating that one field plus adding the corresponding
`LICENSE-*` files at the repo root.

## Why this needs a deliberate choice

Infinity Engine is positioned as a commercial-grade engine. The license determines who can use
it, whether it can be forked into a competitor, whether studios can ship closed-source games on
top of it, and how (or whether) the project can be monetized. This is a business decision, not
an engineering default, so it is surfaced for the owner rather than assumed.

## The realistic options

### Option A — Permissive dual-license: `MIT OR Apache-2.0` (the Rust-ecosystem default)

The overwhelmingly common choice for Rust libraries and for engines that want the widest
adoption (Bevy uses exactly this).

- **Pros:** maximum adoption; zero friction for commercial games; users pick whichever license
  fits their needs; Apache-2.0 adds an explicit patent grant; already compatible with every
  dependency on our cargo-deny allow-list.
- **Cons:** anyone can fork the engine into a closed or competing product; no reciprocal
  obligation to contribute improvements back; monetization must come from services/hosting/
  first-party content, not from the license itself.
- **Mechanics:** add `LICENSE-MIT` and `LICENSE-APACHE` at the repo root; keep the
  `license = "MIT OR Apache-2.0"` field. This is the current placeholder, so choosing it is
  the lowest-effort path.

### Option B — Copyleft: `MPL-2.0` or `GPL-3.0`

- **MPL-2.0** (file-level copyleft): changes to *engine files* must be shared, but games built
  on top (that merely link/use the engine) need not open their own source. A middle ground.
- **GPL-3.0** (strong copyleft): anything that links the engine must also be GPL — effectively
  incompatible with closed-source commercial games, so **not recommended** for an engine meant
  to ship games.
- **Pros (MPL):** improvements to the engine flow back; games stay proprietary.
- **Cons:** more friction than permissive; some studios' legal teams reflexively avoid copyleft;
  a few dependencies may need license review.

### Option C — Source-available / Business Source License (BSL 1.1)

The Unreal-style or Unity-style commercial model: source is visible and usable, but a license
grant (or royalty above a revenue threshold) governs commercial shipping; often converts to a
permissive license after a set time window (BSL "Change Date").

- **Pros:** preserves a direct monetization path (royalties / seat licenses); prevents
  competitors from forking a rival engine; still lets the community read and contribute.
- **Cons:** not OSI-approved "open source"; reduces adoption and community trust; requires
  legal drafting of the Additional Use Grant and revenue terms; incompatible with the
  "MIT OR Apache-2.0" placeholder — every crate's `license` field and the cargo-deny policy
  would need reworking, and some upstream permissive deps' notices must be preserved.

## Dependency license posture (already enforced)

Regardless of the top-level choice, `deny.toml` already pins the allowed licenses for the
dependency tree. Notable decisions made along the way (documented in CLAUDE.md / memos):

- BC7 encoder (`intel_tex_2`) deferred — its ISPC build was a cross-OS liability.
- The rust-analyzer HTTPS auto-installer deferred — `ureq` → `webpki-roots` is
  CDLA-Permissive-2.0, off the allow-list.
- `ring`'s ISC/MIT/OpenSSL license-file needs a `[licenses.clarify]` entry before QUIC/rustls
  networking is enabled (deferred with the P14 net stack).

Any license change must keep the dependency allow-list coherent; run `cargo deny check` after.

## Recommendation to the owner (non-binding)

If the goal is **maximum adoption and ecosystem trust**, choose **Option A (MIT OR Apache-2.0)** —
it is the Rust-native expectation, matches the current placeholder, and requires only dropping in
the two `LICENSE-*` files. If the goal is **direct monetization and fork protection**, evaluate
**Option C (BSL 1.1)** with legal counsel, accepting the adoption trade-off. Copyleft (Option B)
is a rarely-chosen middle path for engines.

## Action items once decided

1. Set the final SPDX expression in `[workspace.package].license` (root `Cargo.toml`).
2. Add the corresponding `LICENSE*` file(s) at the repo root.
3. Re-run `cargo deny check` and reconcile any newly-conflicting dependency licenses.
4. Update the README "License" section and the docs-site Introduction to state the final terms.
</content>
