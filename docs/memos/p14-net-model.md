# Memo: networking model — snapshot replication vs. deterministic lockstep (P14.3)

**Status:** decided — **snapshot replication is the default**; deterministic
lockstep is a documented, *viable* alternative kept in reserve (our bit-determinism
makes it real, not aspirational).
**Date:** 2026-07-21
**Scope:** the model the P14.3 networking seed builds toward, and the per-genre
guidance for when to use which. Implementation this phase = the sans-io protocol
core (`inf-net`) + a quinn transport + transform-replication glue in `inf-runtime`.

## The question

Two families of multiplayer models, each a different bargain:

1. **Deterministic lockstep.** Every peer runs the *same* simulation from the same
   seed and exchanges only **inputs**. State is never sent — it is re-derived
   identically on every machine. Bandwidth is O(players × input-size), independent
   of world size (10 units or 10,000 units cost the same). The hard requirement:
   the simulation must be **bit-identical** across machines, forever, or peers
   silently desync.

2. **Snapshot / state replication.** An authoritative server simulates and streams
   **state** (entity transforms, and later velocities/animation/health) to clients,
   which interpolate/extrapolate to hide latency. Clients need not simulate
   identically; a late or lost packet degrades smoothly (interpolate across the
   gap) instead of desyncing. Bandwidth scales with the *replicated* world size,
   mitigated by delta-encoding + interest management.

## What our architecture already gives us

The determinism doctrine (§2.5) was built for the replay harness, but it is exactly
the precondition lockstep needs:

- The fixed-step sim is **bit-deterministic** and *proven* so in CI — the parallel
  ECS schedule produces a byte-identical trace to the serial baseline across pool
  sizes (P9.1), and physics runs `enhanced-determinism` (libm) for cross-platform
  bit-reproducibility (P8.3/P12).
- Structural changes resolve at deterministic sync points; entity identity is a
  stable `Guid`, never a hashed/recycled index.

So deterministic lockstep is **not a research risk for us** — the expensive
precondition is already paid for and continuously verified. That is the whole
reason this memo can name it a real option rather than a someday-maybe.

## Decision: snapshot replication is the default

Reasons the seed builds the snapshot path first:

1. **Latency tolerance and graceful degradation.** Consumer networks lose and
   reorder packets. Snapshot + interpolation hides that; lockstep stalls every peer
   on the slowest link (input-delay or rollback needed to hide it).
2. **Genre coverage.** The engine targets "every genre." Most — shooters, action,
   co-op, MMO-likes, most 3D — expect client-side prediction over an authoritative
   server. Snapshot is the general default; lockstep is a specialization.
3. **It composes with our data model.** A snapshot is a `Guid`-keyed set of
   component states — the same `Guid` the scene/serialization/replay layers already
   key on. `inf-runtime::replication` reads/writes it as pure functions over
   `EcsWorld`, no new identity system.
4. **Anti-cheat.** An authoritative server is the sane default for untrusted
   clients; peer-lockstep trusts every peer's inputs.

### Per-genre guidance

| Genre / shape | Model | Why |
|---|---|---|
| FPS / third-person action | Snapshot + client prediction + server reconciliation | latency hiding is non-negotiable; authoritative server for anti-cheat |
| Co-op / small-party PvE | Snapshot (host-authoritative) | simplest correct default; smooth under jitter |
| MMO / large world | Snapshot + **interest management** (relevance culling) | replicate only what a client can see; bandwidth ∝ visible set, not world |
| RTS (thousands of units) | **Deterministic lockstep** | replicating thousands of unit states is infeasible; inputs are tiny and our sim is bit-deterministic |
| Fighting / rollback | **Deterministic lockstep + rollback** | frame-perfect; rollback re-simulates from a confirmed state — needs exactly our determinism guarantee |
| Turn-based / async | Either; often lockstep-of-commands | trivially deterministic; state is small |

## What the P14.3 seed implements toward this

- **`inf-net` (sans-io protocol core):** channel + reliability framing
  (`ReliableOrdered` / `Unreliable`); a pure reliability `Endpoint` (exactly-once
  in-order over a lossy/reordering/duplicating packet pipe, best-effort dedup for
  unreliable — property-tested under induced loss/reorder/dup); Guid-keyed
  transform **snapshot encoding with delta-against-baseline** (quantization **off**
  in v1 — raw `f64`, correctness first); a numbered-RPC registry (bincode args).
- **quinn transport (`quic` feature):** reliable frames → QUIC streams, unreliable
  → QUIC datagrams; integration-tested over `127.0.0.1` (100 transforms replicated
  client-ward + an RPC round-trip).
- **`inf-runtime::replication`:** `snapshot_world` / `apply_snapshot` — pure fns
  over `EcsWorld`; proven by replicating between two loops with no network.

Replication is taken **at snapshot boundaries** (after a fixed step), so the
transport is IO layered *on top of* the sim and never enters the deterministic
step — §2.5 is preserved either way, which is also what keeps the lockstep option
open (inputs could be exchanged through the same transport and fed to the same
deterministic loop).

## Deliberately deferred (with the shape known)

- **Client prediction + server reconciliation** (the snapshot model's latency
  hiding): the `Endpoint` already carries sequence/ack info a prediction layer
  needs; the reconciliation loop is future work.
- **Interest management / relevance:** the snapshot is a full map today; a per-client
  relevance filter (spatial/interest sets) plugs in at `snapshot_world`.
- **Quantization / bit-packing** of transforms (bandwidth): v1 is raw `f64`.
- **Replicating spawns/despawns and non-transform components** (velocity, anim,
  gameplay): `apply_snapshot` matches existing Guids only today.
- **A lockstep input-exchange harness** proving cross-machine determinism over the
  wire (the CI replay harness already proves it in-process).

## Revisit criteria

Promote lockstep from "documented option" to "shipped path" when a first-party
sample needs it (an RTS or a rollback fighter), or when a genre template's
bandwidth budget makes state replication infeasible. Revisit quantization + interest
management the moment a sample exceeds a per-tick bandwidth budget with raw `f64`
full snapshots.
