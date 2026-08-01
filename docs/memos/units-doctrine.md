# Units doctrine — 1 world unit = 1 metre

*P16.2, Next-Gen Wave (ROADMAP §12). The rule has been true in the code since Phase 2 and
written down nowhere. This memo is the written form; `CLAUDE.md` carries the one-line rule and
points here. It is a survey plus a standard, not a change — no code moved to land it.*

## The doctrine

**1 world unit = 1 metre. SI everywhere.**

| Quantity | Unit | Notes |
|---|---|---|
| Length, position, extent | metre (m) | world positions are `f64`/`DVec3` (architecture rule 3) |
| Time | second (s) | fixed timesteps are seconds, never frames |
| Mass | kilogram (kg) | |
| Velocity | metre per second (m/s) | |
| Acceleration | m/s² | gravity is `-9.81`, literally |
| Scatter density | instances per square metre (1/m²) | |
| Angle | radian internally; **degree only at the Details/UI boundary** | the existing Transform convention |
| Ratio / weight / fraction | dimensionless `[0, 1]` | say so in the doc comment |

**No global unit-scale factor exists, and none may be introduced.** There is no
`WORLD_SCALE`, no centimetre mode, no per-project unit setting. Unreal's 1 uu = 1 cm and the
"which scale is this asset in?" tax that comes with it are exactly what we are declining. An
importer converts source units *once*, at import, into metres; nothing downstream re-scales.

The payoff is that physical constants can be written as physical constants. Gravity is
`-9.81`, not `-981` or `-9.81 * SCALE`. A 1.8 m character is `1.8`. A brush radius of `8.0` is
eight metres. Terrain sampled at `meters_per_sample = 8.0` covers tens of kilometres and the
arithmetic says so.

## Evidence it already holds

The survey behind this memo found the convention consistent across every subsystem that has a
physical parameter — the doctrine is descriptive of the existing code, which is why it costs
nothing to adopt:

- **Gravity is real-world gravity.** `crates/inf-scene/src/lib.rs` defaults `gravity_3d` to
  `Vec3d::new(0.0, -9.81, 0.0)`, mirrored by
  `editor/crates/inf-editor-core/src/scene/serialize.rs` ("real-world down for the 3D dynamic
  solver") and by the 2D `gravity_2d: Vec2d::new(0.0, -9.81)` in
  `editor/crates/inf-editor-core/src/samples.rs`. Both rapier facades document the same
  default (`crates/inf-physics/src/d2/ecs.rs`, `d3/ecs.rs`). A value of `-9.81` is only
  correct under m/s².
- **Terrain is sampled in metres.** `crates/inf-terrain/src/data.rs` names the field
  `meters_per_sample` and pins `DEFAULT_METERS_PER_SAMPLE: f64 = 1.0` — one sample per metre by
  default. `crates/inf-terrain/src/brush.rs` documents `radius` as "Brush radius in world
  metres" and resolves strokes through `data.meters_per_sample()`.
- **Scatter density is per square metre.** `crates/inf-pcg/src/scatter.rs` documents
  `ScatterParams::base_density` as "Instances per m² at density 1.0", and the per-cell budget
  is literally `base_density · cell_size²` — an area in m² times a density in 1/m².
- **Editor speeds are m/s.** `editor/crates/inf-viewport/src/camera.rs` documents
  `EditorCamera::fly_speed` as "Metres per second while flying", defaulting to `8.0` within
  `FLY_SPEED_MIN = 0.2` … `FLY_SPEED_MAX = 250.0` — a walk-to-jet range that is only sensible
  read as m/s.
- **The floating origin snaps on a metre grid.** `crates/inf-math/src/lib.rs` defines
  `ORIGIN_SNAP: f64 = 10.0` and documents it as "Origins snap to multiples of this (metres)" —
  chosen as a multiple of the editor grid, which is itself metric.
- **The f64 rationale is stated in metres.** `crates/inf-physics/src/lib.rs` argues for
  `rapier2d-f64`/`rapier3d-f64` on the grounds that "at ~4 km from the origin an f32 coordinate
  already jitters past a millimetre". That precision budget — kilometres out, millimetres of
  error — presumes 1 unit = 1 m.
- **The 3D gizmo snap already carries units.** `SnapSettings` in the same `camera.rs`
  documents `translate` as "Translate increment, world metres", `rotate_deg` as degrees, and
  `scale` as a ratio. This is the shape every physical parameter should have.

## Rules for new code

1. **Document the unit in the doc comment of every physical parameter.** Not "the radius" —
   "Brush radius in world metres". Not "duration" — "Recovery time in seconds". A reviewer must
   never have to infer the unit from a call site.
2. **Name constants with a unit suffix where the unit is not obvious from the type or the
   surrounding doc.** `rotate_deg`, `meters_per_sample`, `ORIGIN_SNAP` (metres, documented) are
   the models; a bare `spacing: f32` in a shader or a bare `grid_size` is not.
3. **Rotation is euler degrees at the Details/UI boundary only.** The existing convention
   stands: `Transform` presents euler degrees so authors type `90`, and math that needs
   radians converts internally at the boundary. Never store degrees in a math kernel; never
   surface radians in a Details row.
4. **Convert once, at the edge.** Importers (heightmaps in feet, glTF in centimetres,
   photogrammetry in arbitrary scale) normalise to metres at import time and write metres into
   the asset. No consumer re-scales.
5. **Dimensionless is a unit too.** Weights, falloffs, blend factors and probabilities are
   documented as dimensionless with their valid range (`[0, 1]`, `[-1, 1]`) so they are not
   mistaken for a length.
6. **No unit-scale factor, ever.** If a subsystem seems to want one, that is a signal the
   values are wrong, not the doctrine.

## Follow-up (non-blocking)

A handful of constants are correct in value but anonymous in name — they carry no unit suffix
and no doc comment. None is a bug; each is a readability debt to be paid **as those files are
next touched**, not in a dedicated sweep:

- `Snap2DSettings::grid_size` (`editor/crates/inf-viewport/src/camera.rs`) is documented as
  "world units" rather than metres, and the pixel-snap sibling mixes a pixel-derived increment
  into the same accessor. Rename/redocument to metres when the 2D snap path is next edited.
- The UE-style dual grid bakes its **1 m and 10 m** spacings as bare literals in
  `crates/inf-render/src/shaders/grid.wgsl`, with only a comment naming the units. Promote them
  to named constants (`GRID_MINOR_M`, `GRID_MAJOR_M`) when the grid pass is next touched — and
  keep them consistent with `ORIGIN_SNAP`, which was chosen as a multiple of them.

Neither blocks Phase 16. The doctrine above is the gate for *new* parameters; these two are
grandfathered until their files come up.
