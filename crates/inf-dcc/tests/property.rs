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
    cube, cylinder, from_mesh_asset, plane, to_mesh_asset, torus, validate, CornerData,
    ExportOptions, FaceId, HalfId, ImportError, Mesh, MeshSession, Op, VertId,
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
    (0u8..13, any::<u16>(), any::<u16>(), any::<u8>()).prop_map(|(kind, a, b, p)| Choice {
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
        _ => Op::SetFaceSlot {
            face: pick(&faces, c.a, dead_f),
            slot: None,
        },
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
                kind: (next() % 13) as u8,
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

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, max_shrink_iters: 4_000, ..ProptestConfig::default() })]

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
                prop_assert_eq!(
                    inf_asset::encode(&a1).expect("encodable"),
                    inf_asset::encode(&a2).expect("encodable"),
                );
                // And the second read is the same mesh as the first, up to
                // labelling — the canonical-form claim, exercised on real edits.
                let read2 = from_mesh_asset(&a2).expect("a2 reads back");
                prop_assert_eq!(read.mesh.canonical(), read2.mesh.canonical());
            }
            Err(ImportError::NoGeometry) => {
                prop_assert_eq!(session.mesh().face_count(), 0);
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
