//! **The grammar → mesh bake** (P23.6): a P19 scattered building collapsed into
//! ONE `inf_mesh::MeshAsset`, through the P23.3 half-edge kernel.
//!
//! # Why this exists, in one sentence from the P22 ledger
//!
//! `samples/phase22-playground`'s block is a hand-authored box, and its own
//! source says why:
//!
//! > A `Destructible` names no asset of its own: the cook fractures the **one
//! > mesh** its actor's `MeshRef.asset` points at. The P19 grammar produces a
//! > *population*: scattered solids placed as instances, with no merged mesh
//! > anywhere in the pipeline. Wiring it would therefore have meant building a
//! > grammar→single-mesh bake first, which is a modelling feature and belongs to
//! > Phase 23.
//!
//! This is that feature. A baked building is a **mesh**, so it can be fractured,
//! virtualized, opened in the Model Editor and referenced by a `MeshRef` like any
//! imported prop — none of which a population can be.
//!
//! # What a part IS, and the finding that shapes the whole module
//!
//! [`inf_ecs::components::ScatteredSolid`] is **always an oriented box**:
//! `center`, `half_extents`, and a *yaw-only* `rotation`. There is no shape enum,
//! no primitive, no part tag and — the part that matters — **no material
//! identity**. That lives on the parallel `instances` list, as
//! `PcgInstance::kind_index`, the index into the grammar's `ModuleDef` palette.
//!
//! The two lists are index-aligned **only** for buildings. `assemble.rs` pushes
//! an instance and a collider together at every one of its five sites, so
//! `BuildingOutput`'s lists correspond element for element; the generic
//! `expand_span` path pushes a collider only when the module declares one, so
//! its lists do not. Hence the two doors here:
//!
//! * [`parts_from_building`] — the one with material slots, because the
//!   alignment holds and `kind_index` names a `ModuleDef`.
//! * [`parts_from_solids`] — the ECS-mirror population
//!   ([`PcgVolume::structures`](inf_ecs::components::PcgVolume)), which has lost
//!   the kind by the time it reaches a `SceneDoc`. Everything lands on one slot,
//!   and saying so is better than inventing a mapping.
//!
//! [`parts_from_output`] refuses an unaligned pair as a **typed value** rather
//! than guessing, because a silently mis-slotted building is exactly the shape of
//! defect this codebase keeps paying for.
//!
//! # v1 scope, stated rather than discovered
//!
//! * **Boxes and prisms only.** That is not a simplification of the grammar, it
//!   is the whole of what a grammar places (see above). A `ModuleDef.mesh` GUID —
//!   the "modular mesh per module" the P19.4 gap names — is *not* read here: the
//!   bake would then depend on the asset database and stop being a pure function.
//! * **Merged, not welded, and not booleaned.** Each part is its own closed
//!   shell. Two abutting parts leave their shared faces inside the solid and
//!   their corners coincident, which is counted and surfaced
//!   ([`BakeReport::coincident_vertices`]) rather than repaired — the P16
//!   advisory doctrine, and the same ruling `ExportReport` already made. See
//!   [`BakeReport::reopenable`] for the consequence, which is real and is
//!   measured rather than assumed.
//! * **No interior detail beyond what the grammar placed.** Furniture is placed
//!   by the assembler and is therefore baked; nothing is added, removed or
//!   simplified.
//! * **The mesh stands on its own local `y = 0`**, centred on its footprint,
//!   exactly like `samples/phase22-playground`'s block — so the entity that
//!   references it is placed at [`BakedMesh::origin_world`] and the geometry
//!   lands where the population was.
//!
//! # Determinism
//!
//! The bake is a pure function of its input: `BTreeMap`-free integer loops, `f64`
//! throughout, no transcendental (the rotation is applied by
//! [`glam::DQuat::mul_vec3`], which is `+ - *` on the quaternion's own
//! components), and no `meshopt` — [`inf_dcc::ExportOptions::optimize`] stays
//! off, per the P23.3 LAW. Two runs produce byte-identical assets, and the
//! assembler upstream is itself pool-size-independent
//! (`inf_pcg::building::assemble_in` maps floors through
//! `inf_core::parallel_map`).

use glam::{DQuat, DVec3};
use inf_dcc::{CornerData, ExportOptions, ExportReport, Mesh, Op};
use inf_ecs::components::ScatteredSolid;
use inf_mesh::MeshAsset;
use inf_pcg::grammar::{Grammar, GrammarOutput};
use inf_pcg::{BuildingOutput, BuildingParams};

/// Metres of surface per UV tile.
///
/// The bake authors planar UVs rather than unwrapping, because a building is
/// hundreds of axis-aligned boxes whose flattening is known in closed form and an
/// LSCM solve over it would be slower and no better. A **constant metres-per-tile
/// keeps the texel density uniform**, which per-face `0..1` UVs would not: a 10 m
/// slab and a 0.2 m mullion would get the same UV range and fifty times the
/// density apart.
pub const UV_METRES_PER_TILE: f64 = 2.0;

/// The name given to every part when the population carries no module identity
/// (the [`parts_from_solids`] door).
pub const UNTYPED_SLOT: &str = "Structure";

/// One placed part: an oriented box and the material slot it draws with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakePart {
    /// World-space centre.
    pub center: DVec3,
    /// Half-extents in the box's own (rotated) frame, metres.
    pub half_extents: DVec3,
    /// Yaw, as the population carries it.
    pub rotation: DQuat,
    /// Index into [`BakeInput::slot_names`].
    pub slot: u32,
}

impl BakePart {
    /// Whether this part can become geometry at all: finite, and positive on
    /// every axis. A zero-thickness box is not a refusal — it is a part the
    /// grammar declined to place, and `Ctx::boxed` already drops those upstream.
    pub fn is_solid(&self) -> bool {
        self.center.is_finite()
            && self.half_extents.is_finite()
            && self.rotation.is_finite()
            && self.half_extents.x > 0.0
            && self.half_extents.y > 0.0
            && self.half_extents.z > 0.0
    }
}

/// A whole population, ready to bake.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BakeInput {
    pub parts: Vec<BakePart>,
    /// Material slot names in slot order — the grammar's `ModuleDef` palette, so
    /// a slot index in the baked asset means the same thing it meant in the DSL.
    pub slot_names: Vec<String>,
}

/// What the bake did, and the two hazards it counts rather than repairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BakeReport {
    /// Parts that became geometry.
    pub parts: usize,
    /// Parts dropped as non-solid ([`BakePart::is_solid`]).
    pub skipped: usize,
    /// Distinct material slots the baked asset actually uses.
    pub slots: usize,
    /// The writer's own verdict.
    pub export: ExportReport,
    /// **Kernel vertices sharing a written position.** Abutting parts guarantee
    /// these; see [`reopenable`](Self::reopenable).
    pub coincident_vertices: usize,
    /// **Whether the baked asset can be re-opened in the Model Editor**, measured
    /// by actually running the reader over it rather than inferred from the
    /// counter above.
    ///
    /// A building whose parts abut exactly is *not* re-openable, and that is the
    /// honest v1 limit: the reader's weld is exact, so the four faces meeting at
    /// a shared slab edge become four faces on one edge, which is non-manifold
    /// and refused. The asset is still perfectly good for what it is for —
    /// rendering, virtualization and **fracture**, none of which need half-edge
    /// topology. Measured, so the day a boolean union lands this flips and the
    /// ledger gets rewritten.
    pub reopenable: bool,
}

/// A baked building.
#[derive(Debug, Clone)]
pub struct BakedMesh {
    /// The asset, in LOCAL space: footprint-centred, standing on `y = 0`.
    pub asset: MeshAsset,
    /// Where to place the entity that references it, so the geometry lands where
    /// the population was.
    pub origin_world: DVec3,
    pub report: BakeReport,
}

/// Why a bake could not produce an asset.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BakeError {
    #[error("the population is empty — there is nothing to bake")]
    Empty,
    #[error(
        "every one of the {parts} parts was dropped as non-solid (zero or \
         non-finite extents), so the bake would write an empty mesh"
    )]
    AllSkipped { parts: usize },
    #[error(
        "the population's {instances} instances and {colliders} colliders are not \
         index-aligned, so a part's material slot cannot be known. Only the \
         BUILDING assembler emits them in lockstep; the generic grammar path \
         places a collider only where a module declares one."
    )]
    Unaligned { instances: usize, colliders: usize },
    #[error("the kernel refused part {part}: {message}")]
    Op { part: usize, message: String },
}

// ── the doors in ────────────────────────────────────────────────────────────

/// A building's parts, with **material slots per archetype part**.
///
/// The slot is the module's palette index, and the names are the palette itself —
/// so `Slab`, `Step`, `Glazing`, `Desk` come out of the bake as the submesh names
/// they had in the DSL, and a material assigned to slot *k* means the same thing
/// on both sides of the bake.
pub fn parts_from_building(
    out: &BuildingOutput,
    grammar: &Grammar,
) -> Result<BakeInput, BakeError> {
    parts_from_pairs(&out.instances, &out.colliders, grammar)
}

/// The generic grammar door. Refuses an unaligned pair rather than guessing.
pub fn parts_from_output(out: &GrammarOutput, grammar: &Grammar) -> Result<BakeInput, BakeError> {
    parts_from_pairs(&out.instances, &out.colliders, grammar)
}

fn parts_from_pairs(
    instances: &[inf_pcg::PcgInstance],
    colliders: &[inf_pcg::PcgCollider],
    grammar: &Grammar,
) -> Result<BakeInput, BakeError> {
    if instances.len() != colliders.len() {
        return Err(BakeError::Unaligned {
            instances: instances.len(),
            colliders: colliders.len(),
        });
    }
    if colliders.is_empty() {
        return Err(BakeError::Empty);
    }
    let slot_names: Vec<String> = grammar.modules().iter().map(|m| m.name.clone()).collect();
    let parts = instances
        .iter()
        .zip(colliders)
        .map(|(inst, col)| BakePart {
            center: col.center,
            half_extents: col.half_extents,
            rotation: col.rotation,
            // A kind outside the palette cannot happen (the assembler resolves it
            // through `module_index`), but a slot the name table does not cover
            // would write a dangling index into the asset, so it is clamped to
            // the untyped tail rather than trusted.
            slot: if (inst.kind_index as usize) < slot_names.len() {
                inst.kind_index
            } else {
                slot_names.len() as u32
            },
        })
        .collect();
    let mut slot_names = slot_names;
    slot_names.push(UNTYPED_SLOT.to_string());
    Ok(BakeInput { parts, slot_names })
}

/// The ECS-mirror door: a `PcgVolume`'s `structures`, which carry **no module
/// identity** — the kind lives on the parallel `evaluated` list and the two are
/// not aligned there either. Everything lands on one slot.
pub fn parts_from_solids(solids: &[ScatteredSolid]) -> BakeInput {
    BakeInput {
        parts: solids
            .iter()
            .map(|s| BakePart {
                center: s.center,
                half_extents: s.half_extents,
                rotation: s.rotation,
                slot: 0,
            })
            .collect(),
        slot_names: vec![UNTYPED_SLOT.to_string()],
    }
}

// ── the bake ────────────────────────────────────────────────────────────────

/// Plan, assemble and bake one building in a single call — the shape a caller
/// actually wants.
pub fn bake_building(
    params: &BuildingParams,
    seed: u64,
    furnish: bool,
) -> Result<BakedMesh, BakeError> {
    let out = inf_pcg::building::build(params, seed, furnish);
    let arch = inf_pcg::archetype(params.archetype);
    let grammar = arch
        .grammar()
        .expect("a shipped palette parses (palettes::tests::every_palette_parses)");
    let input = parts_from_building(&out, &grammar)?;
    bake_grammar_to_mesh(&input)
}

/// **The bake.** Instances → one merged half-edge mesh → one `MeshAsset`.
pub fn bake_grammar_to_mesh(input: &BakeInput) -> Result<BakedMesh, BakeError> {
    if input.parts.is_empty() {
        return Err(BakeError::Empty);
    }
    let solid: Vec<(usize, &BakePart)> = input
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_solid())
        .collect();
    if solid.is_empty() {
        return Err(BakeError::AllSkipped {
            parts: input.parts.len(),
        });
    }

    // ── the local origin ───────────────────────────────────────────────────
    //
    // Footprint-centred and standing on its own `y = 0`, so a `Transform` at
    // `origin_world` puts the geometry back where the population was AND the
    // entity's position reads as "where the building stands" rather than "the
    // middle of it". `phase22_tower_mesh`'s convention, met again.
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for (_, part) in &solid {
        for corner in corners(part) {
            lo = lo.min(corner);
            hi = hi.max(corner);
        }
    }
    let origin_world = DVec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);

    // ── the merged mesh ────────────────────────────────────────────────────
    let mut mesh = Mesh::new();
    for (index, part) in &solid {
        add_box(&mut mesh, part, origin_world).map_err(|message| BakeError::Op {
            part: *index,
            message,
        })?;
    }

    // ── out through the writer ─────────────────────────────────────────────
    //
    // `ExportOptions::default()` — `optimize` OFF. The P23.3 LAW is about the op
    // journal, but the reason (meshopt is not cross-platform) applies to any
    // artifact two machines are expected to agree about, and a committed sample's
    // baked building is exactly that.
    let (mut asset, export) = inf_dcc::to_mesh_asset(&mesh, &ExportOptions::default());

    // **The slot table is patched here, and that is a documented v1 seam.** The
    // kernel's `material_slots` is populated only by `from_mesh_asset` — there is
    // no `Op` that writes the slot *table*, deliberately (a mutation outside the
    // journal is a journal with a hole in it, P23.4's ruling on `merge_into`). A
    // mesh built from ops therefore has an empty table, `to_mesh_asset` names its
    // submeshes `slot_{i}`, and the palette is attached afterwards. The bake is
    // not a journalled session, so nothing is lost by doing it here — but a mesh
    // that IS one cannot do this, which is why `merge_into` still drops slots.
    asset.material_slots = input.slot_names.clone();
    for sm in &mut asset.submeshes {
        if let Some(name) = sm
            .material_slot
            .and_then(|s| input.slot_names.get(s as usize))
        {
            sm.name = name.clone();
        }
    }

    let mut slots: Vec<Option<u32>> = asset.submeshes.iter().map(|s| s.material_slot).collect();
    slots.sort_unstable();
    slots.dedup();
    // MEASURED, not inferred: run the reader the Model Editor would run.
    let reopenable = inf_dcc::from_mesh_asset(&asset).is_ok();
    let report = BakeReport {
        parts: solid.len(),
        skipped: input.parts.len() - solid.len(),
        slots: slots.len(),
        coincident_vertices: export.coincident_vertices,
        export,
        reopenable,
    };
    Ok(BakedMesh {
        asset,
        origin_world,
        report,
    })
}

// ── the box ─────────────────────────────────────────────────────────────────

/// The eight world-space corners, numbered exactly as `inf_dcc::build::cube`
/// numbers them — see [`CUBE_FACES`] for why that matters.
fn corners(part: &BakePart) -> [DVec3; 8] {
    let h = part.half_extents;
    let local = [
        DVec3::new(-h.x, -h.y, -h.z), // 0
        DVec3::new(h.x, -h.y, -h.z),  // 1
        DVec3::new(h.x, -h.y, h.z),   // 2
        DVec3::new(-h.x, -h.y, h.z),  // 3
        DVec3::new(-h.x, h.y, -h.z),  // 4
        DVec3::new(h.x, h.y, -h.z),   // 5
        DVec3::new(h.x, h.y, h.z),    // 6
        DVec3::new(-h.x, h.y, h.z),   // 7
    ];
    local.map(|p| part.center + part.rotation.mul_vec3(p))
}

/// The six quad loops, **character for character `inf_dcc::build::cube`'s**, with
/// its vertex numbering.
///
/// Copied rather than derived because the winding convention is the one thing
/// here that cannot be checked by reading: the kernel's cube shipped uniformly
/// **inside-out** through 63 green tests until a signed-volume assertion existed
/// (the P23.3 LAW), and a bake with its own hand-derived winding would be a second
/// chance to make that mistake. `bake_boxes_are_wound_outward` measures the signed
/// volume here too.
const CUBE_FACES: [([usize; 4], usize, bool); 6] = [
    ([0, 1, 2, 3], 1, false), // −Y
    ([7, 6, 5, 4], 1, true),  // +Y
    ([4, 5, 1, 0], 2, false), // −Z
    ([6, 7, 3, 2], 2, true),  // +Z
    ([7, 4, 0, 3], 0, false), // −X
    ([5, 6, 2, 1], 0, true),  // +X
];

/// Append one part as a closed six-quad shell.
fn add_box(mesh: &mut Mesh, part: &BakePart, origin: DVec3) -> Result<(), String> {
    let world = corners(part);
    let local: Vec<DVec3> = world.iter().map(|&p| p - origin).collect();
    let mut ids = Vec::with_capacity(8);
    for p in &local {
        let out = inf_dcc::ops::apply(
            mesh,
            &Op::AddVertex {
                position: p.to_array(),
            },
        )
        .map_err(|e| e.to_string())?;
        ids.push(out.verts[0]);
    }
    for (loop_indices, axis, positive) in CUBE_FACES {
        // The face's outward normal, authored per corner: a box is flat-shaded
        // and the writer's smooth-fan derivation would round its corners off
        // otherwise. `NormalPolicy::PreserveAuthored` (the default) writes these
        // through, so the baked asset shades like the boxes it is made of.
        let mut axis_v = DVec3::ZERO;
        axis_v[axis] = if positive { 1.0 } else { -1.0 };
        let normal = part.rotation.mul_vec3(axis_v);
        // The two in-plane axes, in metres, so texel density is uniform across
        // the whole building rather than per-face.
        let (ua, va) = match axis {
            0 => (2usize, 1usize),
            1 => (0, 2),
            _ => (0, 1),
        };
        let verts: Vec<_> = loop_indices.iter().map(|&i| ids[i]).collect();
        let corners: Vec<CornerData> = loop_indices
            .iter()
            .map(|&i| CornerData {
                uv: [
                    local[i][ua] / UV_METRES_PER_TILE,
                    local[i][va] / UV_METRES_PER_TILE,
                ],
                normal: Some(normal.to_array()),
            })
            .collect();
        inf_dcc::ops::apply(
            mesh,
            &Op::AddFace {
                verts,
                corners,
                slot: Some(part.slot),
            },
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;
    use inf_pcg::{ArchetypeId, Rect2};

    fn one_box(slot: u32) -> BakeInput {
        BakeInput {
            parts: vec![BakePart {
                center: DVec3::new(0.0, 1.0, 0.0),
                half_extents: DVec3::new(2.0, 1.0, 3.0),
                rotation: DQuat::IDENTITY,
                slot,
            }],
            slot_names: vec!["Slab".into(), "Wall".into()],
        }
    }

    /// The signed volume of the written triangles, by the divergence theorem —
    /// the P23.3 gate, applied to this module's own winding.
    fn signed_volume(asset: &MeshAsset) -> f64 {
        let mut v = 0.0;
        for sm in &asset.submeshes {
            for tri in sm.indices.chunks_exact(3) {
                let p: Vec<DVec3> = tri
                    .iter()
                    .map(|&i| {
                        let q = sm.vertices[i as usize].position;
                        DVec3::new(q[0] as f64, q[1] as f64, q[2] as f64)
                    })
                    .collect();
                v += p[0].dot(p[1].cross(p[2])) / 6.0;
            }
        }
        v
    }

    #[test]
    fn bake_boxes_are_wound_outward() {
        // **The gate that has to exist before any count means anything.** Every
        // other assertion in this module is winding-agnostic, and the kernel's own
        // cube shipped inside-out through 63 green tests for exactly that reason.
        let baked = bake_grammar_to_mesh(&one_box(0)).unwrap();
        let want = 4.0 * 2.0 * 6.0; // 4 m × 2 m × 6 m
        assert!(
            (signed_volume(&baked.asset) - want).abs() < 1e-6,
            "signed volume {} — a negative or halved figure is an inside-out or \
             torn shell",
            signed_volume(&baked.asset)
        );
        assert_eq!(baked.report.parts, 1);
        assert_eq!(baked.report.export.triangles, 12);
    }

    #[test]
    fn the_local_origin_stands_the_mesh_on_its_own_floor() {
        let mut input = one_box(0);
        input.parts[0].center = DVec3::new(50.0, 7.0, -30.0);
        let baked = bake_grammar_to_mesh(&input).unwrap();
        // Local: centred on XZ, bottom at y = 0.
        assert!(
            (baked.asset.bounds.min[1]).abs() < 1e-5,
            "sits on its floor"
        );
        assert!((baked.asset.bounds.max[1] - 2.0).abs() < 1e-5);
        assert!((baked.asset.bounds.min[0] + 2.0).abs() < 1e-5);
        assert!((baked.asset.bounds.max[0] - 2.0).abs() < 1e-5);
        // …and the origin puts it back exactly where it was.
        assert_eq!(baked.origin_world, DVec3::new(50.0, 6.0, -30.0));
    }

    #[test]
    fn a_part_keeps_its_material_slot_and_its_name() {
        let baked = bake_grammar_to_mesh(&one_box(1)).unwrap();
        assert_eq!(baked.asset.material_slots, vec!["Slab", "Wall"]);
        assert_eq!(baked.asset.submeshes.len(), 1);
        assert_eq!(baked.asset.submeshes[0].material_slot, Some(1));
        assert_eq!(
            baked.asset.submeshes[0].name, "Wall",
            "the submesh took the palette's name, not `slot_1`"
        );
    }

    #[test]
    fn parts_are_grouped_into_one_submesh_per_slot() {
        let mut input = one_box(0);
        let mut second = input.parts[0];
        second.center.x += 10.0;
        second.slot = 1;
        input.parts.push(second);
        let mut third = input.parts[0];
        third.center.x += 20.0;
        input.parts.push(third);
        let baked = bake_grammar_to_mesh(&input).unwrap();
        assert_eq!(baked.report.parts, 3);
        assert_eq!(baked.report.slots, 2);
        assert_eq!(baked.asset.submeshes.len(), 2, "one submesh per slot");
        assert_eq!(baked.report.export.triangles, 36);
    }

    #[test]
    fn a_non_solid_part_is_skipped_and_counted() {
        let mut input = one_box(0);
        let mut flat = input.parts[0];
        flat.half_extents.y = 0.0;
        input.parts.push(flat);
        let mut bad = input.parts[0];
        bad.center.x = f64::NAN;
        input.parts.push(bad);
        let baked = bake_grammar_to_mesh(&input).unwrap();
        assert_eq!(baked.report.parts, 1);
        assert_eq!(baked.report.skipped, 2);
    }

    #[test]
    fn an_empty_population_refuses_as_a_value() {
        assert_eq!(
            bake_grammar_to_mesh(&BakeInput::default()).unwrap_err(),
            BakeError::Empty
        );
        let all_flat = BakeInput {
            parts: vec![BakePart {
                center: DVec3::ZERO,
                half_extents: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
                slot: 0,
            }],
            slot_names: vec![UNTYPED_SLOT.into()],
        };
        assert_eq!(
            bake_grammar_to_mesh(&all_flat).unwrap_err(),
            BakeError::AllSkipped { parts: 1 }
        );
    }

    #[test]
    fn an_unaligned_population_refuses_rather_than_guessing() {
        // The generic grammar path: a module with no `collider` places an
        // instance and no solid, so the lists drift. Slotting by index there
        // would assign a wall's material to a window, silently.
        let grammar = Grammar::parse("module Panel = size 2\nWall -> Panel*\n").unwrap();
        let out = GrammarOutput {
            instances: vec![inf_pcg::PcgInstance {
                pos: DVec3::ZERO,
                rotation: DQuat::IDENTITY,
                scale: 1.0,
                kind_index: 0,
            }],
            colliders: vec![],
        };
        assert_eq!(
            parts_from_output(&out, &grammar),
            Err(BakeError::Unaligned {
                instances: 1,
                colliders: 0
            })
        );
    }

    #[test]
    fn the_ecs_mirror_door_says_it_has_one_slot() {
        let solids = vec![ScatteredSolid {
            center: DVec3::new(1.0, 2.0, 3.0),
            half_extents: DVec3::splat(0.5),
            rotation: DQuat::IDENTITY,
        }];
        let input = parts_from_solids(&solids);
        assert_eq!(input.slot_names, vec![UNTYPED_SLOT]);
        assert_eq!(input.parts[0].slot, 0);
        let baked = bake_grammar_to_mesh(&input).unwrap();
        assert_eq!(baked.asset.submeshes.len(), 1);
        assert_eq!(baked.asset.submeshes[0].name, UNTYPED_SLOT);
    }

    #[test]
    fn a_yawed_part_bakes_rotated_geometry() {
        let mut input = one_box(0);
        input.parts[0].rotation = DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2);
        let baked = bake_grammar_to_mesh(&input).unwrap();
        // A quarter turn about Y swaps the X and Z extents.
        assert!(
            (baked.asset.bounds.max[0] - 3.0).abs() < 1e-4,
            "{:?}",
            baked.asset.bounds
        );
        assert!((baked.asset.bounds.max[2] - 2.0).abs() < 1e-4);
        // …and it is still a solid of the same volume.
        assert!((signed_volume(&baked.asset) - 48.0).abs() < 1e-4);
    }

    #[test]
    fn the_bake_is_a_pure_function_of_its_input() {
        let params = BuildingParams::new(
            ArchetypeId::House,
            Rect2::from_center(DVec2::new(0.0, 0.0), DVec2::new(12.0, 9.0)),
            0.0,
            7,
        );
        let a = bake_building(&params, 7, true).unwrap();
        let b = bake_building(&params, 7, true).unwrap();
        assert_eq!(
            inf_asset::encode(&a.asset).unwrap(),
            inf_asset::encode(&b.asset).unwrap(),
            "two bakes of one building are not byte-identical"
        );
        assert_eq!(a.origin_world, b.origin_world);
        assert!(
            !a.report.export.optimized,
            "meshopt must never run on a bake — its output differs between \
             x86_64-msvc and aarch64-apple-darwin (the P18 law)"
        );
    }

    #[test]
    fn a_baked_house_is_a_building_and_not_a_box() {
        let params = BuildingParams::new(
            ArchetypeId::House,
            Rect2::from_center(DVec2::new(0.0, 0.0), DVec2::new(12.0, 9.0)),
            0.0,
            7,
        );
        let baked = bake_building(&params, 7, true).unwrap();
        assert!(
            baked.report.parts > 50,
            "a furnished house baked into {} parts",
            baked.report.parts
        );
        assert!(
            baked.report.slots > 3,
            "the archetype's parts collapsed into {} material slots",
            baked.report.slots
        );
        // It stands on its own floor and is roughly the lot it was planned on.
        assert!(baked.asset.bounds.min[1].abs() < 1e-4);
        assert!(baked.asset.bounds.max[1] > 2.0, "it has a storey");
        // **The honest limit, measured.** Abutting parts share corners, so the
        // reader's exact weld cannot reopen it. Asserted so the day this changes
        // the ledger has to change with it.
        assert!(
            baked.report.coincident_vertices > 0,
            "a real building's parts abut; zero coincident vertices means the \
             counter is not measuring what it claims"
        );
        assert!(
            !baked.report.reopenable,
            "a baked building became re-openable in the Model Editor — the v1 \
             ledger says it is not, and that is now wrong"
        );
    }
}
