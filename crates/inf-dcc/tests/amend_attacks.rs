//! **Adversarial arms against `MeshSession::amend`** (Wave-D audit).
//!
//! The amendment is the wave's signature claim and the easiest thing in the
//! crate to get subtly wrong, because every failure mode is silent: an op that
//! lands on the wrong polygon produces a *valid* mesh. The module's own tests
//! state the feature; these state the attacks, and each one is here because it
//! kills a mutation nothing else killed:
//!
//! * **atk1** — a `Free` op amended so that a DOWNSTREAM op renumbers. Gate 1
//!   waves a `Free` op through unconditionally, so only the topology comparison
//!   can see this; it is also the arm that dies when the comparison is moved
//!   from per-step to once-at-the-end.
//! * **atk2** — a refusal deep in the tail, with structural ops on both sides:
//!   atomicity, and the generation stamp not moving.
//! * **atk3 / atk3b** — the redo tail. Ops beyond the cursor replay, the cursor
//!   does not travel, and the "nothing downstream to protect" skip is keyed on
//!   `ops.len()` rather than on `cursor` (3b is what says so).
//! * **atk4…atk9** — double amendment, non-finite parameters, `usize::MAX`,
//!   a restored session, an amendment whose own op refuses, and a full
//!   undo/redo walk afterwards (which is what catches a stale checkpoint).
//!
//! Every one asserts the same two-outcome law the property battery states in
//! the general case: **refuse inertly, or be a pure function of the ops.**

use inf_dcc::{cube, plane, validate, AmendError, MeshSession, MirrorAxis, Op, OpError};

fn vert_at(m: &inf_dcc::Mesh, x: f64, z: f64) -> inf_dcc::VertId {
    m.vert_ids()
        .find(|&v| {
            let p = m.position(v).unwrap();
            (p.x - x).abs() < 1e-9 && (p.z - z).abs() < 1e-9
        })
        .expect("a vertex there")
}

/// ATTACK 1 — amend a **Free** op so that a DOWNSTREAM op renumbers.
/// Gate 1 waves a Free op through unconditionally; only gate 2 can see this.
#[test]
fn atk1_a_free_amendment_that_renumbers_downstream_is_caught() {
    let mut s = MeshSession::new(plane(2.0));
    let a = vert_at(s.mesh(), 1.0, 1.0);
    let b = vert_at(s.mesh(), 1.0, -1.0);
    let v = vec![a, b];
    // op0: a no-op translate of the EDGE that sits on the future mirror plane.
    s.apply(Op::TranslateVerts {
        verts: v.clone(),
        delta: [0.0, 0.0, 0.0],
    })
    .expect("translate");
    // op1: a mirror at x = 1 — welds the two verts on the plane.
    s.apply(Op::Mirror {
        axis: MirrorAxis::X,
        coord: 1.0,
    })
    .expect("mirror");
    let welded_verts = s.mesh().vert_count();
    // op2: names an id the mirror minted.
    let f = s.mesh().face_ids().next().expect("a face");
    s.apply(Op::ExtrudeFaces {
        faces: vec![f],
        distance: 0.3,
    })
    .expect("extrude");

    let before = s.mesh().encoded();
    let before_ops = s.ops().to_vec();
    let gen_before = s.generation();

    // Move that vertex OFF the mirror plane. Gate 1: Free, so nothing to check.
    let err = s
        .amend(
            0,
            Op::TranslateVerts {
                verts: v.clone(),
                delta: [0.5, 0.0, 0.0],
            },
        )
        .expect_err("the mirror now welds nothing");
    assert!(
        matches!(err, AmendError::IdAllocationChanged { index: 1 }),
        "expected the mirror step to be named, got {err:?}"
    );
    assert_eq!(s.mesh().encoded(), before, "a refusal must be inert");
    assert_eq!(s.ops(), &before_ops[..]);
    assert_eq!(
        s.generation(),
        gen_before,
        "nothing moved, so no id cache went stale"
    );
    // and the fixture really did exercise the hazard
    let mut apart = MeshSession::new(plane(2.0));
    apart
        .apply(Op::TranslateVerts {
            verts: v,
            delta: [0.5, 0.0, 0.0],
        })
        .unwrap();
    apart
        .apply(Op::Mirror {
            axis: MirrorAxis::X,
            coord: 1.0,
        })
        .unwrap();
    assert_ne!(
        apart.mesh().vert_count(),
        welded_verts,
        "the fixture does not exercise the hazard"
    );
}

/// ATTACK 2 — an amendment that makes a **deep** tail op refuse, with
/// structural ops on both sides of it. Atomicity: nothing may be half-written.
#[test]
fn atk2_a_mid_replay_refusal_leaves_the_session_whole() {
    let mut s = MeshSession::new(cube(2.0));
    s.apply(Op::AddMaterialSlots {
        names: vec!["A".into(), "B".into(), "C".into()],
    })
    .expect("slots");
    let f = s.mesh().face_ids().next().expect("a face");
    let out = s
        .apply(Op::ExtrudeFaces {
            faces: vec![f],
            distance: 0.5,
        })
        .expect("extrude");
    let cap = out.faces[0];
    let out = s
        .apply(Op::InsetFaces {
            faces: vec![cap],
            amount: 0.1,
            individual: false,
        })
        .expect("inset");
    let inner = out.faces[0];
    s.apply(Op::SubdivideFaces { faces: vec![inner] })
        .expect("subdivide");
    let target = s.mesh().face_ids().next().expect("a face");
    s.apply(Op::SetFaceSlot {
        face: target,
        slot: Some(2),
    })
    .expect("assign the third slot");
    // …and more structure after the op that will refuse.
    let f2 = s.mesh().face_ids().nth(2).expect("a face");
    s.apply(Op::ExtrudeFaces {
        faces: vec![f2],
        distance: 0.2,
    })
    .expect("extrude again");

    let before = s.mesh().encoded();
    let before_ops = s.ops().to_vec();
    let before_cursor = s.cursor();
    let gen_before = s.generation();
    let err = s
        .amend(
            0,
            Op::AddMaterialSlots {
                names: vec!["A".into()],
            },
        )
        .expect_err("slot 2 stops existing");
    assert!(
        matches!(err, AmendError::TailRefused { index: 4, .. }),
        "{err:?}"
    );
    assert_eq!(s.mesh().encoded(), before);
    assert_eq!(s.ops(), &before_ops[..]);
    assert_eq!(s.cursor(), before_cursor);
    assert_eq!(s.generation(), gen_before);
    assert_eq!(validate(s.mesh()), Ok(()));
    // …and the session still round-trips.
    let back = MeshSession::restore(s.save()).expect("restores");
    assert_eq!(back.mesh().encoded(), before);
}

/// ATTACK 3 — the **redo tail**: undone ops beyond the cursor must replay, and
/// the cursor must not travel.
#[test]
fn atk3_ops_beyond_the_cursor_replay_and_the_cursor_stays() {
    let build = |distance: f64| {
        let mut s = MeshSession::new(cube(2.0));
        let f = s.mesh().face_ids().next().expect("a face");
        let out = s
            .apply(Op::ExtrudeFaces {
                faces: vec![f],
                distance,
            })
            .expect("extrude");
        let cap = out.faces[0];
        s.apply(Op::InsetFaces {
            faces: vec![cap],
            amount: 0.1,
            individual: false,
        })
        .expect("inset");
        let v = s.mesh().vert_ids().nth(3).expect("a vertex");
        s.apply(Op::TranslateVerts {
            verts: vec![v],
            delta: [0.0, 0.05, 0.0],
        })
        .expect("nudge");
        s.apply(Op::SubdivideFaces {
            faces: s.mesh().face_ids().take(1).collect(),
        })
        .expect("subdivide");
        s
    };
    let mut s = build(0.4);
    let authored = build(0.6);
    assert!(s.undo());
    assert!(s.undo());
    assert_eq!(s.cursor(), 2);
    assert_eq!(s.ops().len(), 4);

    let faces = match &s.ops()[0] {
        Op::ExtrudeFaces { faces, .. } => faces.clone(),
        other => panic!("{other:?}"),
    };
    s.amend(
        0,
        Op::ExtrudeFaces {
            faces,
            distance: 0.6,
        },
    )
    .expect("amending under a redo tail");

    assert_eq!(s.cursor(), 2, "the cursor travelled");
    assert_eq!(s.ops().len(), 4, "the redo tail was truncated");
    let want = MeshSession::replay(s.base(), &s.ops()[..2]).expect("replays");
    assert_eq!(s.mesh().encoded(), want.encoded());
    // …and the tail still redoes, onto the re-derived geometry.
    assert!(s.redo());
    assert!(s.redo());
    assert_eq!(s.cursor(), 4);
    assert_eq!(
        s.mesh().encoded(),
        authored.mesh().encoded(),
        "the redone tail is not the session authored that way"
    );
    assert_eq!(s.ops(), authored.ops());
}

/// ATTACK 3b — amending the **last applied** op while a redo tail exists must
/// still run the topology gate (the skip is keyed on `ops.len()`, not `cursor`).
#[test]
fn atk3b_the_last_applied_op_is_not_the_last_op() {
    let mut s = MeshSession::new(plane(2.0));
    s.apply(Op::Mirror {
        axis: MirrorAxis::X,
        coord: 1.0,
    })
    .expect("mirror welds");
    let f = s.mesh().face_ids().next().expect("a face");
    s.apply(Op::ExtrudeFaces {
        faces: vec![f],
        distance: 0.3,
    })
    .expect("extrude");
    assert!(s.undo());
    assert_eq!((s.cursor(), s.ops().len()), (1, 2));

    let before = s.mesh().encoded();
    let err = s
        .amend(
            0,
            Op::Mirror {
                axis: MirrorAxis::X,
                coord: 5.0,
            },
        )
        .expect_err("the UNDONE extrude would land on different geometry");
    assert!(
        matches!(err, AmendError::IdAllocationChanged { .. }),
        "{err:?}"
    );
    assert_eq!(s.mesh().encoded(), before);
}

/// ATTACK 4 — amend the same index twice.
#[test]
fn atk4_a_double_amendment_lands_on_the_second_parameter() {
    let build = |d: f64| {
        let mut s = MeshSession::new(cube(2.0));
        let f = s.mesh().face_ids().next().expect("a face");
        let out = s
            .apply(Op::ExtrudeFaces {
                faces: vec![f],
                distance: d,
            })
            .expect("extrude");
        s.apply(Op::InsetFaces {
            faces: vec![out.faces[0]],
            amount: 0.1,
            individual: false,
        })
        .expect("inset");
        s
    };
    let mut s = build(0.4);
    let faces = match &s.ops()[0] {
        Op::ExtrudeFaces { faces, .. } => faces.clone(),
        other => panic!("{other:?}"),
    };
    s.amend(
        0,
        Op::ExtrudeFaces {
            faces: faces.clone(),
            distance: 0.6,
        },
    )
    .expect("first");
    s.amend(
        0,
        Op::ExtrudeFaces {
            faces,
            distance: 0.8,
        },
    )
    .expect("second");
    assert_eq!(s.mesh().encoded(), build(0.8).mesh().encoded());
    assert_eq!(s.ops(), build(0.8).ops());
    assert_eq!(validate(s.mesh()), Ok(()));
}

/// ATTACK 5 — a NON-FINITE parameter must be refused, inertly, as a value.
#[test]
fn atk5_a_non_finite_amendment_is_refused_as_a_value() {
    let mut s = MeshSession::new(cube(2.0));
    let f = s.mesh().face_ids().next().expect("a face");
    s.apply(Op::ExtrudeFaces {
        faces: vec![f],
        distance: 0.4,
    })
    .expect("extrude");
    s.apply(Op::SubdivideFaces {
        faces: s.mesh().face_ids().take(1).collect(),
    })
    .expect("subdivide");
    let before = s.mesh().encoded();
    for bad in [f64::NAN, f64::INFINITY, -f64::INFINITY] {
        let err = s
            .amend(
                0,
                Op::ExtrudeFaces {
                    faces: vec![f],
                    distance: bad,
                },
            )
            .expect_err("a non-finite distance must refuse");
        assert!(
            matches!(err, AmendError::Refused(_)),
            "{bad} gave {err:?} rather than a refusal"
        );
        assert_eq!(s.mesh().encoded(), before);
    }
}

/// ATTACK 6 — out-of-range indices, including the extremes, refuse as values.
#[test]
fn atk6_out_of_range_indices_refuse_without_panicking() {
    let mut s = MeshSession::new(cube(2.0));
    let before = s.mesh().encoded();
    // an empty journal
    for i in [0usize, 1, usize::MAX] {
        let err = s
            .amend(i, Op::AddVertex { position: [0.0; 3] })
            .expect_err("nothing is applied");
        assert!(matches!(err, AmendError::OutOfRange { .. }), "{err:?}");
    }
    s.apply(Op::AddVertex {
        position: [1.0, 2.0, 3.0],
    })
    .expect("add");
    for i in [1usize, 2, usize::MAX] {
        let err = s
            .amend(
                i,
                Op::AddVertex {
                    position: [4.0, 5.0, 6.0],
                },
            )
            .expect_err("past the cursor");
        assert!(matches!(err, AmendError::OutOfRange { .. }), "{err:?}");
    }
    assert_eq!(s.ops().len(), 1);
    assert_ne!(s.mesh().encoded(), before);
}

/// ATTACK 7 — amending a **restored** session behaves like amending a live one.
#[test]
fn atk7_a_restored_session_amends_the_same_way() {
    let build = |d: f64| {
        let mut s = MeshSession::new(cube(2.0));
        let f = s.mesh().face_ids().next().expect("a face");
        let out = s
            .apply(Op::ExtrudeFaces {
                faces: vec![f],
                distance: d,
            })
            .expect("extrude");
        s.apply(Op::InsetFaces {
            faces: vec![out.faces[0]],
            amount: 0.1,
            individual: false,
        })
        .expect("inset");
        s.apply(Op::SubdivideFaces {
            faces: s.mesh().face_ids().take(2).collect(),
        })
        .expect("subdivide");
        s
    };
    let s = build(0.4);
    let mut back = MeshSession::restore(s.save()).expect("restores");
    let faces = match &back.ops()[0] {
        Op::ExtrudeFaces { faces, .. } => faces.clone(),
        other => panic!("{other:?}"),
    };
    back.amend(
        0,
        Op::ExtrudeFaces {
            faces,
            distance: 0.6,
        },
    )
    .expect("amends");
    assert_eq!(back.mesh().encoded(), build(0.6).mesh().encoded());
}

/// ATTACK 8 — an amendment whose own op refuses names it and stays inert.
#[test]
fn atk8_the_amended_op_refusing_is_a_named_value() {
    let mut s = MeshSession::new(cube(2.0));
    let f = s.mesh().face_ids().next().expect("a face");
    s.apply(Op::ExtrudeFaces {
        faces: vec![f],
        distance: 0.4,
    })
    .expect("extrude");
    s.apply(Op::SubdivideFaces {
        faces: s.mesh().face_ids().take(1).collect(),
    })
    .expect("subdivide");
    let before = s.mesh().encoded();
    let err = s
        .amend(
            0,
            Op::ExtrudeFaces {
                faces: vec![],
                distance: 0.4,
            },
        )
        .expect_err("an empty region is not a re-parameterization");
    // The operand set moved, so gate 1 refuses before anything is replayed.
    assert!(matches!(err, AmendError::StructureChanged), "{err:?}");
    assert_eq!(s.mesh().encoded(), before);

    // …and a genuinely refusing parameter (a bevel wider than its edge).
    let mut s = MeshSession::new(cube(2.0));
    let h = s
        .mesh()
        .half_ids()
        .find(|&h| s.mesh().is_boundary(h) == Some(false))
        .expect("an interior half");
    s.apply(Op::BevelEdges {
        edges: vec![h],
        amount: 0.05,
        segments: 1,
    })
    .expect("bevel");
    s.apply(Op::SubdivideFaces {
        faces: s.mesh().face_ids().take(1).collect(),
    })
    .expect("subdivide");
    let before = s.mesh().encoded();
    let r = s.amend(
        0,
        Op::BevelEdges {
            edges: vec![h],
            amount: 100.0,
            segments: 1,
        },
    );
    match r {
        Err(AmendError::Refused(OpError::NonFinite { .. })) => {}
        Err(AmendError::Refused(_)) | Err(AmendError::IdAllocationChanged { .. }) => {}
        Err(AmendError::TailRefused { .. }) => {}
        other => panic!("a 100 m bevel on a 2 m cube gave {other:?}"),
    }
    assert_eq!(s.mesh().encoded(), before, "inert either way");
}

/// ATTACK 9 — after an amendment, a full undo/redo walk still lands on the
/// amended history at every step (the checkpoints really were dropped).
#[test]
fn atk9_a_full_history_walk_after_an_amendment_is_consistent() {
    let mut s = MeshSession::new(cube(2.0));
    for i in 0..40 {
        let v = s.mesh().vert_ids().nth(i % 8).expect("a vertex");
        s.apply(Op::TranslateVerts {
            verts: vec![v],
            delta: [0.001 * i as f64, 0.0, 0.0],
        })
        .expect("nudge");
    }
    let f = s.mesh().face_ids().next().expect("a face");
    s.apply(Op::ExtrudeFaces {
        faces: vec![f],
        distance: 0.4,
    })
    .expect("extrude");
    for i in 0..40 {
        let v = s.mesh().vert_ids().nth(i % 8).expect("a vertex");
        s.apply(Op::TranslateVerts {
            verts: vec![v],
            delta: [0.0, 0.001 * i as f64, 0.0],
        })
        .expect("nudge");
    }
    s.amend(
        40,
        Op::ExtrudeFaces {
            faces: vec![f],
            distance: 0.9,
        },
    )
    .expect("amends");
    let mut states = Vec::new();
    states.push(s.mesh().encoded());
    while s.undo() {
        assert_eq!(validate(s.mesh()), Ok(()), "at cursor {}", s.cursor());
        states.push(s.mesh().encoded());
    }
    states.reverse();
    let mut i = 1;
    while s.redo() {
        assert_eq!(
            s.mesh().encoded(),
            states[i],
            "redo diverged at cursor {}",
            s.cursor()
        );
        i += 1;
    }
}
