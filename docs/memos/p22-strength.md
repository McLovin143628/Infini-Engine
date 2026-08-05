# P22 — material strength, in pascals

*Decision memo, P22.2 (2026-08-04). Companion to `docs/memos/units-doctrine.md`.*

`Destructible::strength` is a **failure stress in pascals (N/m²)**, and
`Destructible::density_kg_m3` is a **material density in kg/m³**. This memo says
where those numbers come from, what values mean which materials, how P22.3 will
consume them, and — the load-bearing half — **why the component's five fields are
all Phase 22 gets**.

---

## 1. Why a stress and not a "durability"

The obvious shape for a destruction system is hit points: give the wall 100,
subtract 30 per rocket, break it at 0. The units doctrine (architecture rule 6,
`docs/memos/units-doctrine.md`) forbids it, and not on aesthetic grounds. A
unitless durability cannot be compared against anything the physics engine
produces. An explosion in this engine delivers an **impulse in N·s**; a collision
reports a **force in N**; a chunk has a **mass in kg** and a shared face has an
**area in m²**. To turn any of those into "how much damage" you need a made-up
conversion constant per weapon and per material, and the moment two of them
disagree the only way to tune the system is to try numbers until it looks right.

A stress has no such problem. Force over area *is* a pressure, so:

```
bond breaks  ⟺  F_across_face / face_area  >  strength
             ⟺  F_across_face             >  strength × face_area
```

Both sides are newtons. That is `Destructible::bond_force_n(area_m2)`, and it
lives on the component rather than in P22.3 for the same reason
`Buoyancy::equilibrium_fraction` lives on its component: it is the *contract*, so
a solve that computes a different threshold is failing this function rather than
its own.

## 2. Why a failure stress and not Young's modulus

The roadmap phrases P22.2's deliverable as "density, Young's-modulus-class
scalars — the DCC spec's derivation, simplified". The simplification made here is
to keep the *class* (an elastic-mechanics quantity in Pa, checkable against a real
materials table) and change the *quantity* (failure stress, not stiffness).

Young's modulus `E` governs how much a material **deflects before it fails**:
`σ = E · ε`, strain against stress. That is exactly the part of the problem a
rigid-body engine does not simulate. Our chunks are rigid; they do not strain,
they do not bend, and they do not store elastic energy. Carrying `E` would mean
carrying a number nothing reads, and then quietly reinterpreting it as a strength
in P22.3 — which is worse than choosing correctly now, because the field docs
would be lying about the units.

The two are related in real materials (a stiff material is usually a strong one)
but not proportionally, and the ratio differs by an order of magnitude across
classes: concrete's `E` is ~30 GPa against a tensile strength of ~3 MPa (ratio
10⁴), steel's is ~200 GPa against ~400 MPa (ratio 500). Picking `E` and dividing
by a constant would misrank the very materials the system exists to distinguish.

**Which strength.** Tensile/shear, not compressive. Structures fail in tension
long before they fail in compression — masonry is roughly ten times weaker in
tension than in compression, which is why arches exist — and a collapsing
building's chunks separate rather than crush. Quoting the compressive figure would
make every building roughly ten times too hard to break.

## 3. The scale

| Class | `strength` (Pa) | `density_kg_m3` | Notes |
|---|---|---|---|
| Plaster / drywall | `1e6` | 800 | Punch-through material |
| Unreinforced masonry, brick | `2e6` | 1900 | Weak in tension; the archetypal collapsing wall |
| Concrete (plain) | `3e6` | 2400 | Tensile figure; compressive is ~10× |
| Glass | `5e6` | 2500 | Brittle: fails near this with no warning |
| Wood, across the grain | `5e6` | 500–750 | Pine 500, oak 750 |
| **Default** | **`5e6`** | **2400** | Masonry/concrete class — see below |
| Reinforced concrete | `1e7` | 2500 | Rebar is what buys the order of magnitude |
| Wood, along the grain | `4e7` | 500–750 | Anisotropy we do not model; pick per use |
| Granite | `1e7` | 2700 | |
| Steel | `4e8` | 7850 | Effectively unbreakable by hand weapons |

These are engineering-handbook figures rounded to one significant digit. They are
*starting points an author overrides*, not a claim that the simulation is
predictive — a single scalar cannot express anisotropy (see wood, twice in the
table), fatigue, or the fact that real masonry fails at its mortar joints rather
than through its bricks.

**The defaults are `5e6` Pa and `2400` kg/m³** — the masonry/concrete class. The
same reasoning as `Buoyancy`'s seasoned-wood default: the component should do
something sensible the instant it is added. A wall at 5 MPa is broken by an
explosion and not by a footstep, and rubble at 2400 kg/m³ reads as rubble rather
than as polystyrene. `Collider3D::density` is deliberately *not* consulted: it
defaults to `1.0`, which is rapier's mass placeholder and not a material density
— the P20.2 finding that buoyancy was designed around, and the reason this
component carries its own honest field.

## 4. How P22.3 will consume them

Nothing here is built yet; this is the interface the two numbers exist to serve,
recorded so P22.2's field set can be judged against it.

1. **Chunk mass.** The cook writes each chunk's `volume_m3` into the
   `.inf_fracture` (from the polytope, exactly). Mass is
   `density_kg_m3 × volume_m3` — `Destructible::chunk_mass_kg`. The chunk's
   collider is a `ColliderShape3D::ConvexHull`, whose volume rapier computes from
   the same solid, so the mass the solver integrates and the mass the gameplay
   layer quotes are one number. (A `Trimesh` could not do this; it is why the
   hull variant exists at all — see `inf_physics::d3::ColliderShape3D::ConvexHull`.)
2. **Bond strength.** Two chunks are bonded iff they are adjacent in the
   `.inf_fracture`'s precomputed graph. The bond holds up to
   `strength × shared_face_area` newtons. The cook already emits the adjacency;
   the shared face area is recoverable from either chunk's geometry.
3. **The structural solve, and how a chunk becomes an anchor.** This is the one
   open design question v20's field set depends on, so it is settled here rather
   than left for P22.3 to discover:

   * **Support is decided by runtime contact, not by an authored flag.** At the
     moment the intact mesh is swapped for chunk bodies, a chunk is an *anchor*
     if it is in contact with static world geometry — the ground, a terrain
     tile, another static body. rapier already reports exactly this
     (`drain_contact_events`, plus the static/dynamic pairing), so the anchor
     set is a query against the world the solve is already stepping.
   * **`AlwaysLoaded` is repurposed as the explicit override.** An author who
     needs a specific piece pinned regardless of contact — the base course of a
     tower, a bridge abutment — marks its entity `AlwaysLoaded`, which already
     means "this exists for the whole run" and now additionally means "this is
     structurally anchored". Reusing it is deliberate: an entity that is *not*
     always loaded cannot be a reliable anchor anyway, because it can stream out
     from under the structure it is holding up. A separate `StructuralAnchor`
     component would be able to express the incoherent combination.
   * Support then propagates along bonds; a chunk with no path to an anchor
     falls. Removing a chunk drives *progressive* collapse, at fixed step,
     deterministically — the roadmap's P22.3 deliverable 2.

   **This is a commitment, not a sketch.** `Destructible`'s five fields are
   frozen and the scene schema is frozen at v20 for the rest of the phase, so
   P22.3 implements the anchoring design above. If it were instead to decide it
   wanted a `StructuralAnchor` slot on the entity record, that would cost a v21
   in both codec mirrors — which is precisely the outcome deciding this now
   exists to prevent.
4. **Damage.** An explosion's impulse is spread over the chunk faces it reaches,
   compared against each bond's threshold. No damage numbers, no per-weapon
   constants: a newton is a newton.

## 5. Why five fields suffice for the whole phase

Scene schema **v20 is Phase 22's only bump**. bincode is positional, so growing
`Destructible` later costs another bump in *both* codec mirrors (the law paid for
at v12, v13, v15 and v16). The fields are therefore frozen as shipped, and the
case has to be made now.

The five partition cleanly by consumer:

* `fracture_seed`, `chunk_count` — **everything the cook needs.** Chunk geometry
  is otherwise a pure function of the mesh, so these two are the entire authored
  surface of the fracture.
* `strength` — **everything the structural solve needs**, per §4.
* `density_kg_m3` — **everything the chunk bodies need**; volume comes from the
  cook.
* `runtime_destruct` — the gate.

And the things that look like missing fields, with where they actually live:

| Looks missing | Why it is not a field |
|---|---|
| A `.inf_fracture` reference | Derived from the entity's own `MeshRef` (`derived_fracture_id`). A reference would be a second authority for the same fact, and a dangling one to advise about. |
| Damage thresholds / hit points | Forbidden by §1. Damage is an impulse compared against `strength × area`. |
| Debris lifetime, budget caps | **Per-tier, not per-asset** — the roadmap's own P22.4 wording. A ceiling authored on one wall cannot bound a scene. |
| Interior material override | Lives in the `.inf_fracture`'s own `slots` list, which versions itself. |
| Fracture hierarchy depth | v1 is two levels by design; a depth-2 hierarchy would be described *inside* the `.inf_fracture`, which has its own schema ladder. |
| Net relevance / replication flags | The P14 net layer's concern; P22.4 documents the events, it does not author them here. |
| A structural-anchor marker | **`AlwaysLoaded`, repurposed** — see §4.3. Anchoring is decided by runtime contact, with that component as the explicit override. |
| How many chunks actually came out | `FractureAsset::requested_chunks` vs `chunks.len()` — a property of the *derived asset*, which has its own version ladder, not of the component. |
| Per-chunk overrides | A chunk is cook output. Authoring per chunk would mean authoring against geometry that changes when the seed does. |

The general rule that falls out: **anything describing the CHUNKS belongs in the
`.inf_fracture`** (a container with its own version ladder, so it can grow for
free), and **anything describing the WHOLE SCENE belongs in settings or a tier**.
The component holds only what describes *this actor's material*, which is why
five fields is the right number rather than a lucky one.
