//! The property battery: random op sequences against every promise the kernel
//! makes.
//!
//! The generator does not produce `Op`s directly — it produces *choices*, which
//! are resolved against the mesh as it stands (pick the `i`-th live half-edge,
//! and so on). Two consequences, both wanted:
//!
//! * The sequences are **reachable**: a random `HalfId` would be dead almost
//!   always and every property would degenerate into "refusals refuse".
//! * Refusals still happen constantly — a fallback id of `u32::MAX` is generated
//!   whenever a pick has nothing to pick from — so the *inertness* property gets
//!   hammered too.
//!
//! Each property is written so a specific defect makes it fail. The three most
//! load-bearing, and the mutation each was verified against, are named in the
//! batch report: link surgery that forgets a `prev` (caught by
//! `validity_holds_after_every_op`), an op that mutates without journalling
//! (caught by `replay_is_a_pure_function_of_the_ops`), and a seam reconstruction
//! that averages corner attributes (caught by `export_is_a_fixed_point`).

use proptest::prelude::*;

use inf_dcc::{
    cube, cylinder, from_mesh_asset, op_preserves_ids, plane, to_mesh_asset, torus, validate,
    CornerData, ExportOptions, FaceId, HalfId, ImportError, KnifePoint, MergeTarget, Mesh,
    MeshSession, MirrorAxis, Op, SculptFalloff, SculptMode, SelectMode, SelectionSet, VertId,
};

/// A generated op, before it is resolved against a mesh.
#[derive(Debug, Clone, Copy)]
struct Choice {
    kind: u8,
    a: u16,
    b: u16,
    p: u8,
}

fn choice() -> impl Strategy<Value = Choice> {
    (0u8..25, any::<u16>(), any::<u16>(), any::<u8>()).prop_map(|(kind, a, b, p)| Choice {
        kind,
        a,
        b,
        p,
    })
}

fn base_mesh() -> impl Strategy<Value = Mesh> {
    prop_oneof![
        Just(plane(2.0)),
        Just(cube(1.0)),
        Just(cylinder(0.5, 2.0, 6)),
        Just(torus(1.0, 0.3, 6, 4)),
    ]
}

fn pick<T: Copy>(items: &[T], i: u16, fallback: T) -> T {
    if items.is_empty() {
        fallback
    } else {
        items[i as usize % items.len()]
    }
}

/// Resolve a choice against the current mesh. Deliberately allowed to produce
/// ops that will refuse.
fn make_op(mesh: &Mesh, c: Choice) -> Op {
    let verts: Vec<VertId> = mesh.vert_ids().collect();
    let halfs: Vec<HalfId> = mesh.half_ids().collect();
    let faces: Vec<FaceId> = mesh.face_ids().collect();
    let corners: Vec<HalfId> = halfs
        .iter()
        .copied()
        .filter(|&h| mesh.is_boundary(h) == Some(false))
        .collect();
    let dead_v = VertId(u32::MAX);
    let dead_h = HalfId(u32::MAX);
    let dead_f = FaceId(u32::MAX);
    let scale = |x: u16| (x as f64 / 65_535.0) * 2.0 - 1.0;

    match c.kind {
        0 => Op::AddVertex {
            position: [scale(c.a), scale(c.b), c.p as f64 / 255.0],
        },
        1 => Op::RemoveVertex {
            vert: pick(&verts, c.a, dead_v),
        },
        2 => Op::AddFace {
            verts: vec![
                pick(&verts, c.a, dead_v),
                pick(&verts, c.b, dead_v),
                pick(&verts, c.a.wrapping_add(c.b), dead_v),
            ],
            corners: vec![CornerData::default(); 3],
            slot: None,
        },
        3 => Op::RemoveFace {
            face: pick(&faces, c.a, dead_f),
        },
        4 => Op::SplitEdge {
            half: pick(&halfs, c.a, dead_h),
            t: 0.1 + 0.8 * (c.p as f64 / 255.0),
        },
        5 => Op::CollapseEdge {
            half: pick(&halfs, c.a, dead_h),
        },
        6 => Op::SplitFace {
            from: pick(&corners, c.a, dead_h),
            to: pick(&corners, c.b, dead_h),
        },
        7 => Op::WeldVerts {
            keep: pick(&verts, c.a, dead_v),
            merge: pick(&verts, c.b, dead_v),
        },
        8 => Op::TranslateVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            delta: [scale(c.a) * 0.1, scale(c.b) * 0.1, 0.0],
        },
        9 => Op::SetCornerUv {
            half: pick(&corners, c.a, dead_h),
            uv: [scale(c.a), scale(c.b)],
        },
        10 => Op::SetCornerNormal {
            half: pick(&corners, c.a, dead_h),
            normal: if c.p.is_multiple_of(2) {
                None
            } else {
                Some([0.0, 1.0, 0.0])
            },
        },
        11 => Op::SetEdgeSharp {
            half: pick(&halfs, c.a, dead_h),
            sharp: c.p.is_multiple_of(2),
        },
        12 => Op::SetFaceSlot {
            face: pick(&faces, c.a, dead_f),
            slot: None,
        },

        // ── the P23.4 modelling set ────────────────────────────────────────
        //
        // Same rule as above: resolve against the mesh as it stands, so the ops
        // are REACHABLE, and let the fallback ids keep the inertness property
        // fed. Sizes are small and signed so an extrude can go inward, which is
        // where a winding mistake shows up.
        13 => Op::ExtrudeFaces {
            faces: two_faces(&faces, c),
            distance: scale(c.a) * 0.5,
        },
        14 => Op::ExtrudeEdges {
            edges: vec![pick(&halfs, c.a, dead_h)],
            delta: [scale(c.a) * 0.2, scale(c.b) * 0.2, 0.1],
        },
        15 => Op::InsetFaces {
            faces: two_faces(&faces, c),
            amount: 0.05 + 0.2 * (c.p as f64 / 255.0),
            individual: c.p.is_multiple_of(2),
        },
        16 => Op::BevelEdges {
            edges: vec![pick(&halfs, c.a, dead_h)],
            amount: 0.01 + 0.1 * (c.p as f64 / 255.0),
        },
        17 => Op::LoopCut {
            half: pick(&halfs, c.a, dead_h),
            cuts: 1 + (c.p % 3) as u32,
        },
        18 => Op::Knife {
            path: vec![
                KnifePoint::Vertex(pick(&verts, c.a, dead_v)),
                KnifePoint::Vertex(pick(&verts, c.b, dead_v)),
            ],
        },
        19 => Op::MergeVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            target: if c.p.is_multiple_of(2) {
                MergeTarget::Center
            } else {
                MergeTarget::Last
            },
        },
        20 => Op::SubdivideFaces {
            faces: two_faces(&faces, c),
        },
        21 => Op::Mirror {
            axis: match c.p % 3 {
                0 => MirrorAxis::X,
                1 => MirrorAxis::Y,
                _ => MirrorAxis::Z,
            },
            // A plane through a vertex the mesh actually has, so the exact-zero
            // seam weld is genuinely exercised rather than always missing.
            coord: 0.0,
        },

        // ── the P23.5 sculpt / gizmo set ───────────────────────────────────
        //
        // A stroke is generated with a REAL path (several dabs, resampled by the
        // product's own `stroke_dabs`) rather than a single point, because the
        // whole claim of the op is that a multi-dab gesture replays byte for
        // byte — a one-dab generator would test the easy half only.
        22 => {
            let seed = pick(&verts, c.a, dead_v);
            let start = mesh.position(seed).unwrap_or(glam::DVec3::ZERO);
            // **The radius is a fraction of the model, not an absolute.** The
            // first version used 0.4 m, and a script that happened to apply a few
            // shrinking `ScaleVerts` left every dab covering the WHOLE mesh — a
            // Dijkstra plus a normal fan over every vertex, per dab, which took
            // the reachability battery from 0.7 s to 95 s. A brush sized to the
            // model is also the more honest generator: it exercises the same
            // fraction of the surface whatever the script did to the scale.
            let extent = model_extent(mesh);
            let radius = (extent * 0.12).max(1e-9);
            let path: Vec<glam::DVec3> = (0..3)
                .map(|i| {
                    start
                        + glam::DVec3::new(
                            scale(c.b) * radius * i as f64,
                            radius * 0.1 * i as f64,
                            0.0,
                        )
                })
                .collect();
            Op::Sculpt {
                mode: match c.p % 4 {
                    0 => SculptMode::Draw,
                    1 => SculptMode::Smooth,
                    2 => SculptMode::Flatten,
                    _ => SculptMode::Grab,
                },
                dabs: inf_dcc::stroke_dabs(&path, radius)
                    .into_iter()
                    .map(|d| d.to_array())
                    .collect(),
                radius,
                strength: scale(c.a) * radius * 0.3,
                falloff: match c.p % 3 {
                    0 => SculptFalloff::Smooth,
                    1 => SculptFalloff::Linear,
                    _ => SculptFalloff::Sharp,
                },
            }
        }
        23 => Op::RotateVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            pivot: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            radians: scale(c.a) * 0.5,
        },
        _ => Op::ScaleVerts {
            verts: vec![pick(&verts, c.a, dead_v), pick(&verts, c.b, dead_v)],
            pivot: [0.0, 0.0, 0.0],
            factor: [
                1.0 + scale(c.a) * 0.2,
                1.0 + scale(c.b) * 0.2,
                1.0 + scale(c.a) * 0.1,
            ],
        },
    }
}

/// The longest side of the mesh's bounding box, or `0` when it has no vertices.
/// Used to size a generated brush against the model rather than against nothing.
fn model_extent(mesh: &Mesh) -> f64 {
    let (mut lo, mut hi) = (glam::DVec3::splat(f64::MAX), glam::DVec3::splat(f64::MIN));
    let mut any = false;
    for v in mesh.vert_ids() {
        if let Some(p) = mesh.position(v) {
            if p.is_finite() {
                lo = lo.min(p);
                hi = hi.max(p);
                any = true;
            }
        }
    }
    if any {
        (hi - lo).max_element().max(0.0)
    } else {
        0.0
    }
}

/// One or two faces — the region form of an op has to be reached with a set that
/// is sometimes bigger than one, or the border-detection rule is never tested.
fn two_faces(faces: &[FaceId], c: Choice) -> Vec<FaceId> {
    if faces.is_empty() {
        return vec![FaceId(u32::MAX)];
    }
    let first = faces[c.a as usize % faces.len()];
    if c.p.is_multiple_of(3) {
        vec![first]
    } else {
        vec![first, faces[c.b as usize % faces.len()]]
    }
}

/// Drive a session through a choice list, returning how many ops applied and how
/// many refused.
fn drive(session: &mut MeshSession, script: &[Choice]) -> (usize, usize) {
    let (mut applied, mut refused) = (0, 0);
    for &c in script {
        let op = make_op(session.mesh(), c);
        let before = session.mesh().encoded();
        match session.apply(op.clone()) {
            Ok(_) => {
                applied += 1;
                assert_eq!(validate(session.mesh()), Ok(()), "invalid after {op:?}");
            }
            Err(_) => {
                refused += 1;
                assert_eq!(
                    session.mesh().encoded(),
                    before,
                    "a refused {op:?} must leave the mesh byte-identical"
                );
            }
        }
    }
    (applied, refused)
}

/// A deterministic 200-op script over every base mesh, asserting that the
/// generator actually **reaches** both outcomes.
///
/// Without this the whole file could be passing because every generated op
/// refuses — six properties about "the mesh after an op" holding vacuously over
/// a mesh no op ever touched. Vacuous checks hide real intrusions (the P19 law),
/// and a generator is exactly the kind of thing that degrades into one silently
/// when an id-picking rule changes.
#[test]
fn the_generator_reaches_both_applied_and_refused_ops() {
    let bases = [
        ("plane", plane(2.0)),
        ("cube", cube(1.0)),
        ("cylinder", cylinder(0.5, 2.0, 6)),
        ("torus", torus(1.0, 0.3, 6, 4)),
    ];
    for (name, base) in bases {
        // A fixed LCG, so this test says the same thing on every run and machine.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let script: Vec<Choice> = (0..200)
            .map(|_| Choice {
                kind: (next() % 25) as u8,
                a: next() as u16,
                b: next() as u16,
                p: next() as u8,
            })
            .collect();
        let mut session = MeshSession::new(base);
        let (applied, refused) = drive(&mut session, &script);
        assert!(
            applied >= 40,
            "{name}: only {applied}/200 ops applied — the generator has gone vacuous"
        );
        assert!(
            refused >= 20,
            "{name}: only {refused}/200 ops refused — the inertness property is untested"
        );
        assert_eq!(validate(session.mesh()), Ok(()));
    }
}

/// Every one of the twelve modelling / sculpt ops must actually APPLY somewhere
/// in the battery, not merely be generated and refused.
///
/// The P19 vacuity law, aimed at the exact way this file could rot: `make_op`
/// resolves ids against the live mesh, so a change to a picking rule (or an op
/// whose preconditions are tighter than the generator can satisfy) turns a
/// property into a very fast test of nothing. `the_generator_reaches_both_...`
/// counts applications in bulk and would still pass with all nine dead.
#[test]
fn every_modelling_op_applies_at_least_once_somewhere_in_the_battery() {
    let bases = [
        ("plane", plane(2.0)),
        ("cube", cube(1.0)),
        ("cylinder", cylinder(0.5, 2.0, 6)),
        ("torus", torus(1.0, 0.3, 6, 4)),
    ];
    let mut applied = [0usize; 25];
    for (_, base) in bases {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let seed_mesh = base.clone();
        let mut session = MeshSession::new(base);
        for _ in 0..600 {
            let c = Choice {
                kind: (next() % 25) as u8,
                a: next() as u16,
                b: next() as u16,
                p: next() as u8,
            };
            let op = make_op(session.mesh(), c);
            if session.apply(op).is_ok() {
                applied[c.kind as usize] += 1;
            }
            // **A bounded battery.** `Mirror` doubles the mesh, and once the
            // transform ops have pushed geometry off the mirror plane the seam
            // weld stops collapsing anything — so a script that reaches it a few
            // dozen times grows exponentially. The P23.5 generator found exactly
            // that: 6.6 million vertices on the plane and 93 seconds of CI, in a
            // battery whose point is *reachability*, not size. Restarting from
            // the base past the cap keeps every kind reachable (the counts are
            // cumulative across restarts) and keeps the runtime a constant.
            if session.mesh().vert_count() > 4_000 {
                session = MeshSession::new(seed_mesh.clone());
            }
        }
        assert_eq!(validate(session.mesh()), Ok(()));
    }
    for kind in 13..25 {
        assert!(
            applied[kind] > 0,
            "op kind {kind} never applied — the generator cannot reach it, so \
             every property below is vacuous for it. Counts: {applied:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 4_000, ..ProptestConfig::default() })]

    /// A selection is only ever read at the generation it was stamped for.
    ///
    /// The contract the whole selection model rests on, hammered against random
    /// edits: after any op, EITHER the stamp still matches (and every id is still
    /// live) OR the consumer is told to drop. There is no third state in which a
    /// set silently survives a renumbering — and the ops that
    /// `op_preserves_ids` lets through `carry` really do leave every kept id
    /// naming the same polygon.
    #[test]
    fn a_selection_never_outlives_the_generation_it_was_stamped_for(
        base in base_mesh(),
        script in prop::collection::vec(choice(), 1..24),
    ) {
        let mut session = MeshSession::new(base);
        let mut sel = SelectionSet::new(session.generation());
        for &c in &script {
            // Select EVERY face, then edit.
            //
            // Selecting only the first was a gate that did not fire: a structural
            // op rebuilds two or three faces out of dozens, so a one-face
            // selection usually missed them and the property passed with the
            // id-preservation table lying about `SplitEdge` (measured). The set
            // has to contain what the op will touch, whatever it turns out to
            // touch.
            for f in session.mesh().face_ids() {
                sel.set_face(f, true);
            }
            let before_faces: Vec<(FaceId, Vec<VertId>)> = sel
                .faces()
                .iter()
                .map(|&f| (f, session.mesh().face_verts(f).unwrap_or_default()))
                .collect();
            let op = make_op(session.mesh(), c);
            let preserves = op_preserves_ids(&op);
            let Ok(outcome) = session.apply(op) else { continue };
            if preserves {
                sel.carry(session.generation(), session.mesh());
                for (f, loop_verts) in before_faces {
                    if sel.contains_face(f) {
                        prop_assert_eq!(
                            session.mesh().face_verts(f).unwrap_or_default(),
                            loop_verts,
                            "carry kept a face id that changed meaning"
                        );
                    }
                }
            } else {
                sel.adopt(session.generation(), &outcome, session.mesh());
            }
            prop_assert_eq!(sel.generation(), session.generation());
            for &f in sel.faces() {
                prop_assert!(session.mesh().has_face(f), "a dead face is selected");
            }
            for &v in sel.verts() {
                prop_assert!(session.mesh().has_vert(v), "a dead vertex is selected");
            }
            for &h in sel.edges() {
                prop_assert!(session.mesh().has_half(h), "a dead edge is selected");
            }
            prop_assert!(!sel.sync(session.generation()), "already in sync");
        }
    }

    /// Soft-select weights are bounded, seed-anchored and order-independent.
    #[test]
    fn soft_select_weights_are_bounded_and_deterministic(
        base in base_mesh(),
        radius in 0.05f64..3.0,
    ) {
        let seeds: Vec<VertId> = base.vert_ids().step_by(3).collect();
        let mut sel = SelectionSet::new(7);
        for v in &seeds { sel.set_vert(*v, true); }
        let a = sel.soft_weights(&base, SelectMode::Vert, radius, inf_terrain::Falloff::Smooth);
        let b = sel.soft_weights(&base, SelectMode::Vert, radius, inf_terrain::Falloff::Smooth);
        prop_assert_eq!(&a, &b);
        for (&v, &w) in &a {
            prop_assert!((0.0..=1.0).contains(&w), "weight {} at {}", w, v);
            prop_assert!(base.has_vert(v));
        }
        for v in &seeds {
            prop_assert_eq!(a.get(v), Some(&1.0), "a seed is full weight");
        }
    }

    /// The kernel's headline promise: whatever the op, the mesh is still a mesh.
    /// Mutation-verified — dropping the `prev` fix-up in `add_face_raw` makes
    /// this fail with `NextPrevMismatch`.
    #[test]
    fn validity_holds_after_every_op(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        prop_assert_eq!(validate(&base), Ok(()));
        let mut session = MeshSession::new(base);
        let (applied, refused) = drive(&mut session, &script);
        prop_assert!(applied + refused == script.len());
    }

    /// `replay(base, ops)` is the mesh, byte for byte. This is what makes undo
    /// "truncate and replay" sound, and it is what an op that mutates without
    /// journalling breaks.
    #[test]
    fn replay_is_a_pure_function_of_the_ops(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        let replayed = MeshSession::replay(session.base(), &session.ops()[..session.cursor()])
            .expect("journalled ops replay");
        prop_assert_eq!(replayed.encoded(), session.mesh().encoded());
    }

    /// Two runs of the same script agree byte for byte. In one process this
    /// catches order-dependence (an iteration over a hash container, an
    /// allocation that depends on anything but the op sequence); the
    /// cross-machine half of the claim is structural and is pinned by
    /// `tests/determinism_law.rs`.
    #[test]
    fn two_runs_of_a_script_agree(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut a = MeshSession::new(base.clone());
        let mut b = MeshSession::new(base);
        drive(&mut a, &script);
        drive(&mut b, &script);
        prop_assert_eq!(a.mesh().encoded(), b.mesh().encoded());
        prop_assert_eq!(a.ops(), b.ops());
    }

    /// Undo to the base and redo to the head, both ending on the exact bytes
    /// they started from — over checkpoint boundaries and evictions.
    #[test]
    fn undo_and_redo_are_inverses(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base.clone());
        drive(&mut session, &script);
        let head = session.mesh().encoded();
        let steps = session.cursor();
        while session.undo() {
            prop_assert_eq!(validate(session.mesh()), Ok(()));
        }
        prop_assert_eq!(session.mesh().encoded(), base.encoded());
        for _ in 0..steps {
            prop_assert!(session.redo());
        }
        prop_assert_eq!(session.mesh().encoded(), head);
    }

    /// The asset round trip: an exported mesh read back and written again is
    /// byte-identical, and the mesh that came back is valid.
    ///
    /// Exactly two refusals are permitted, and each one has to *prove* it was
    /// entitled: `NoGeometry` only when the mesh has no faces, and
    /// `NonManifoldEdge` only when the writer's own report says why — coincident
    /// distinct vertices (which the reader's exact weld fuses) or a triangulation
    /// diagonal that had to repeat an existing edge. Anything else means the
    /// writer emitted a soup its own reader calls illegal, with nothing to blame.
    #[test]
    fn export_is_a_fixed_point(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        let opts = ExportOptions::default();
        let (a1, report) = to_mesh_asset(session.mesh(), &opts);
        match from_mesh_asset(&a1) {
            Ok(read) => {
                prop_assert_eq!(validate(&read.mesh), Ok(()));
                let (a2, _) = to_mesh_asset(&read.mesh, &opts);
                let e1 = inf_asset::encode(&a1).expect("encodable");
                let e2 = inf_asset::encode(&a2).expect("encodable");
                if e1 == e2 {
                    // The second read is the same mesh as the first, up to
                    // labelling — the canonical-form claim, on real edits.
                    let read2 = from_mesh_asset(&a2).expect("a2 reads back");
                    prop_assert_eq!(read.mesh.canonical(), read2.mesh.canonical());
                } else {
                    // **The third face of the coincidence hazard, found by
                    // P23.4's ops** (P23.3 documented the first two: the read is
                    // refused, or a diagonal repeats an edge). Two kernel
                    // vertices that round to the same `f32` are not a refusal at
                    // all — the reader's exact weld fuses them, the triangles
                    // that used both become degenerate and are *skipped and
                    // counted*, and the mesh comes back legal and smaller.
                    //
                    // The extrude/inset/bevel set makes this ordinary, because
                    // they place new vertices a parameter away from existing
                    // ones and a small enough parameter is nothing in `f32`. It
                    // stays a documented advisory rather than a fix for the
                    // reasons already recorded (nudging geometry falsifies the
                    // model; refusing the export makes a legal intermediate
                    // unsaveable) — but the writer must still have SAID so.
                    prop_assert!(
                        report.coincident_vertices > 0 || report.reused_diagonals > 0,
                        "the round trip moved and the writer's report has nothing \
                         to blame: {:?}",
                        report
                    );
                    prop_assert!(
                        read.report.degenerate_triangles_skipped > 0
                            || read.report.welded_positions < report.vertices,
                        "…and the reader did not actually fuse anything: {:?}",
                        read.report
                    );
                }
            }
            Err(ImportError::NoGeometry) => {
                // Either there was nothing to write, or **every** triangle
                // written collapsed in `f32` and the reader skipped the lot —
                // the coincidence hazard again, in its third symptom. The
                // entitlement is the same one: the writer must have said so.
                prop_assert!(
                    session.mesh().face_count() == 0
                        || report.coincident_vertices > 0
                        || report.reused_diagonals > 0,
                    "the reader found no geometry in an asset written from {}                      face(s), with nothing in the report to blame: {:?}",
                    session.mesh().face_count(),
                    report
                );
            }
            Err(ImportError::NonManifoldEdge { .. }) => {
                prop_assert!(
                    report.coincident_vertices > 0 || report.reused_diagonals > 0,
                    "the reader refused an asset with neither coincident vertices \
                     nor a reused diagonal to blame"
                );
            }
            Err(other) => prop_assert!(false, "the writer produced an unreadable asset: {other}"),
        }
    }

    /// Every written vertex is finite, indices are in range, and the bounds
    /// contain the geometry — the properties a consumer that never heard of this
    /// crate (the renderer, the cook, `fracture_mesh`) relies on.
    #[test]
    fn every_written_asset_is_well_formed(base in base_mesh(), script in prop::collection::vec(choice(), 1..40)) {
        let mut session = MeshSession::new(base);
        drive(&mut session, &script);
        let (asset, report) = to_mesh_asset(session.mesh(), &ExportOptions::default());
        prop_assert_eq!(asset.schema_version, 2);
        prop_assert_eq!(report.submeshes, asset.submeshes.len());
        for sm in &asset.submeshes {
            prop_assert_eq!(sm.indices.len() % 3, 0);
            prop_assert!(sm.skin.is_empty());
            for &i in &sm.indices {
                prop_assert!((i as usize) < sm.vertices.len());
            }
            for v in &sm.vertices {
                for k in 0..3 {
                    prop_assert!(v.position[k].is_finite());
                    prop_assert!(v.normal[k].is_finite());
                    prop_assert!(v.tangent[k].is_finite());
                    prop_assert!(v.position[k] >= asset.bounds.min[k]);
                    prop_assert!(v.position[k] <= asset.bounds.max[k]);
                }
                prop_assert!(v.tangent[3] == 1.0 || v.tangent[3] == -1.0);
                let n2: f32 = v.normal.iter().map(|c| c * c).sum();
                prop_assert!((n2 - 1.0).abs() < 1e-4, "normal not unit: {:?}", v.normal);
            }
        }
    }
}
