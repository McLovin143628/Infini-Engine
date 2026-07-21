# Release channels & the updater — design

**Status: DESIGN + partial substance.** This document specifies how Infinity Engine will be
versioned and distributed across release channels, and what an in-app updater needs. The
**in-repo substance shipped in P15.4** is the build-time version stamping (a git short hash
embedded via `build.rs`, surfaced in the About dialog — see below). A *real* signed auto-updater
and a hosted release feed require distribution infrastructure (a signing identity, a release
server / GitHub Releases feed, per-OS notarization) that does not exist in this repo yet; those
are specified here as the ops steps, honestly, rather than faked.

## Versioning scheme

Infinity Engine uses **semantic versioning** (`MAJOR.MINOR.PATCH`) sourced from a single place:
`[workspace.package].version` in the root `Cargo.toml`. Every crate inherits it with
`version = { workspace = true }`, and the Tauri app mirrors it in `tauri.conf.json`.

- **MAJOR** — breaking changes to the `.inf_*` asset schemas that lack a migration, or to the
  public plugin/mod ABI.
- **MINOR** — new features, new asset schema versions *with* migrations (the norm — every schema
  struct carries `schema_version` + a migration fn).
- **PATCH** — bug fixes, no schema or ABI change.

The **git short hash** is embedded at build time (`editor/studio/src-tauri/build.rs` sets
`INF_GIT_HASH`) and shown alongside the version, so any build is traceable to a commit even
between tagged releases. `app_build_info` returns `"Infinity Engine <version> · commit <hash>"`.

## Channels

| Channel | Cadence | Source | Audience |
|---------|---------|--------|----------|
| **stable** | tagged releases (`vX.Y.Z`) | `main` at a release tag | shipping developers |
| **beta** | ~monthly pre-releases (`vX.Y.0-beta.N`) | `main` HEAD, feature-frozen | early adopters |
| **nightly** | per-green-CI build on `main` | every passing `main` commit | contributors / bleeding edge |

A channel is just a metadata tag on a build plus which release feed it subscribes to. The version
string carries the pre-release identifier (`-beta.N`, `-nightly.<date>+<hash>`) so semver ordering
picks the right "newest".

## What an updater needs (spec, not yet built)

An in-app or side-by-side updater requires the following pieces, none of which should be faked:

1. **A signing identity per OS.** Windows Authenticode cert; macOS Developer ID + notarization;
   Linux — GPG-signed AppImage / `.deb`. Without these, an auto-update silently installing a
   binary is a security anti-pattern. This is a business/ops prerequisite.
2. **A release feed.** A static JSON manifest per channel (`stable.json`, `beta.json`,
   `nightly.json`) listing the latest version, per-platform download URLs, SHA-256 digests, and
   the signature. Hostable on GitHub Releases + a small `latest.json`, which is exactly the shape
   Tauri's official updater plugin (`@tauri-apps/plugin-updater`) consumes.
3. **The client.** Tauri v2 ships a first-party updater plugin: it fetches the channel manifest,
   compares versions, verifies the signature against a public key baked into the app, downloads,
   and relaunches. Wiring it is a few hundred lines *once the identity + feed exist*. Until then
   the app ships without update prompts (the honest default) rather than a stub that pretends.
4. **A "what's my version / check for updates" surface.** Partially in place today: the About
   dialog / status bar show version + git hash. "Check for updates" would call the feed and
   diff versions — deferred with the feed.

## Rollout / ops steps (when infrastructure lands)

1. Choose + provision signing identities (Authenticode, Developer ID, GPG).
2. Add a `release` CI workflow: on a `v*` tag, build per-OS bundles (`inf export` / Tauri
   bundler), sign, generate the SHA-256 + signature, and publish a GitHub Release with the
   per-channel `latest.json`.
3. Add `@tauri-apps/plugin-updater` to the app with the public key + channel URL; gate it behind
   a user setting (opt-in for nightly).
4. Extend the CLI/player with the same `--version` banner (the studio pattern in `app.rs` /
   `build.rs`) so headless builds are equally traceable. `inf --version` already prints the
   semver; embedding the git hash there mirrors the studio's `build.rs`.

## What shipped in P15.4 (substance, not spec)

- Single-source version (`[workspace.package].version`), mirrored in `tauri.conf.json`.
- Build-time git-hash embedding via `editor/studio/src-tauri/build.rs` → `INF_GIT_HASH`.
- `app_build_info` command + About-dialog wiring (version + commit).
- This design doc + the channel/versioning scheme for the eventual updater.
