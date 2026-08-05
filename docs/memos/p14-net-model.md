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

## Destruction events over the wire (P22.4)

Documentation only — **no net code was written for this section.** P22 built the
destruction it describes; what follows is the contract a P14-style replication
layer will need from it, written down while the people who built it still
remember why each piece is shaped the way it is.

### What the sim actually produces

Destruction state is `BTreeMap<Uuid, FractureState>` on the host (`SimSession::fractures`
/ `RuntimeSim::fractures`) and is **not** in the ECS world — so
`snapshot_world` / `apply_snapshot` do not see it and would not carry it if the
transport shipped tomorrow. There are exactly three observable events, all
produced inside the fixed step:

1. **A chunk detaches.** Either damage spent enough joules on its bonds
   (`PhysicsBridge3D::runtime_destruct`) or the structural solve found it
   unsupported (`step_fractures`). Identity is `(actor Guid, chunk index)`;
   ordering is `detach_order`, a monotone per-actor counter.
2. **A chunk is reclaimed.** The debris budget's despawn — lifetime expiry or the
   level-wide cap. Identity is the same pair; the chunk is `gone` for ever.
3. **`Destroyed` fires.** One edge per actor, the first step on which every chunk
   has come off, carrying the chunk count. It is already an `EventKind` a
   Blueprint can bind, so it is the only one of the three that gameplay can see
   directly today.

Chunk **poses** are not in this list on purpose. They are ordinary rigid-body
state and belong to whatever the transport already does with bodies; a chunk is a
solver-owned body under a synthetic content-derived Guid
(`fracture_chunk_guid`), so a body-replicating snapshot reaches it with no new
concept — it only has to stop assuming a Guid names an entity.

### What is replication-relevant, and what is not

**Relevant:** the detach set and the reclaim set. They change what exists, they
are cheap (two integers per event), and a client that misses one draws a wall
that is not there.

**Not relevant:** the *energy* a blow absorbed, the bond graph, the support solve,
the audit counters. Those are derivations; sending them would be sending the
server's reasoning instead of its conclusions, and a client that re-derived them
would need the `.inf_fracture`, the placement and the contact set to agree
bit-for-bit — which is the lockstep contract, not the snapshot one.

**Explicitly not relevant: the rubble.** P22.4's sub-chunk debris is
render-only dressing, laid by a pure function of `(actor id, chunk index, detach
order, fragment index)` (`inf_render::debris`). Every client that agrees about
the detach set already agrees about the rubble, byte for byte, with nothing sent.

That claim is only worth making because two things were built to support it, and
both are easy to lose: the seed is a **content tuple** rather than a wall clock or
a frame index, and the placement uses **no `sin`/`cos`** — `unit_dir` and
`unit_quat` are rejection samplers over `sqrt` and arithmetic, because `std`
trigonometry's last bits are not guaranteed identical across platforms (the P14
LAW, applied here to bytes that are never serialized precisely so that they never
*need* to be). `the_placement_uses_no_platform_dependent_trigonometry` is what
keeps it true.

### Ordering and idempotency

* **Detach is idempotent.** `FractureState::detach` is a no-op on a chunk that is
  already detached or gone, so a duplicated event is free and a client may apply
  the same message twice.
* **Reclaim is idempotent and terminal.** `gone` never goes back to `false`, so
  reclaim-before-detach (a reordered pair) converges to the same state as the
  other order: the chunk ends gone either way. There is no ABA problem because a
  chunk index is never reused.
* **`Destroyed` is latched** (`destroyed_fired`), so it fires once per actor per
  session however many times the underlying condition is re-evaluated. A client
  that replays it twice must therefore de-duplicate on the actor, not trust the
  message count.
* **Order between actors does not matter**; order *within* an actor does, but only
  because `detach_order` is what the budget sorts on. A late-arriving detach with
  a lower order than one already applied is still correct state; it is only the
  *reclaim priority* that would differ, and that self-corrects on the next sweep.

The one thing that is **not** order-independent is the structural solve: support
propagates through undetached neighbours, so a client that applies detaches in a
different order and re-runs the solve locally can reach a different collapse. The
conclusion is the same one the whole memo reaches — **the server solves, the
client is told what fell.** A client must never run `step_fractures` on
replicated state.

### What a late joiner needs

The minimum full state for one actor is: its Guid, and per chunk the pair
`(detached, gone)` plus `detach_order` for the ones that are detached — i.e. the
`ChunkState` fields `FractureState::bits()` already serializes for the
determinism gates, minus `age_s` (a budget input the server owns) and minus the
pose (which the body snapshot carries anyway). For a 64-chunk actor that is well
under 200 bytes.

The joiner also needs the **placement**, and this is the subtle one: an actor's
`placement` is frozen at the first detach, so a late joiner that derives it from
the entity's current transform will be right for intact actors and wrong for any
actor that was moved before it broke. Send it, or accept that scripted moving
destructibles resynchronise incorrectly.

Nothing here needs a schema bump: scene schema **v20** describes the *authored*
destructible, and none of the above is authored.

### The save-game seam is the same seam

A save game would want exactly the late-joiner payload, for exactly the same
reason, and would hit exactly the same placement caveat. P22.4's ruling is that
**destruction is not persisted** — the engine has no save-game container, and
`.inf_lvl` is the author's document rather than a player's progress
(`simulate_destruction_not_persisted` is what keeps that true). When a save-game
container arrives it should serialize the payload above and nothing more; the
event stream and the snapshot are two encodings of one fact, and adding a third
authority for what is broken is how a repository ends up with two of them
disagreeing.
