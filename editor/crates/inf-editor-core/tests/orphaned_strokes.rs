//! **A stroke stranded by a tool switch settles into the undo history** (P21.3,
//! the P21.2 audit's N2 ledger item).
//!
//! # The bug this is the gate on
//!
//! Every brush in the viewport mutates the document *live*, one dab per frame,
//! and turns the accumulated dabs into an `EditCommand` only at mouse-up. The
//! mouse-up handler lives inside the pump's **tool-gated** branch, so a tool
//! switch arriving between two frames of a drag — and the toolbar and the
//! `tool.*` shortcuts both push one down a command channel, mid-gesture or not —
//! left the stroke open forever. Its edits stayed in the world, saved like any
//! other edit, and **Ctrl+Z could not reach them**: the un-undoable committed
//! edit `a4e5844` ruled worse than any partial one.
//!
//! P21.2 closed it for the carve brush (`EngineHost::settle_orphaned_carve`).
//! The three terrain brushes — `DragStroke::Height`, `DragStroke::Splat` and
//! `DragStroke::Biome` — had the identical hole, and P21.3 closes it with
//! `EngineHost::settle_orphaned_sculpt`.
//!
//! # Why the test is here and not in `inf-viewport`
//!
//! `inf_viewport::host` is `#[cfg(any(windows, target_os = "macos"))]` **and**
//! an `EngineHost` needs a GPU, so nothing in this repository can construct one
//! in CI. The fix therefore splits into two checkable halves, and both are
//! checked:
//!
//! * **this file** — the settlement itself. `settle_orphaned_sculpt` does
//!   exactly one thing a test can be written about: it reaches `finish_sculpt`,
//!   which commits whichever of the three `DragStroke` kinds is in flight. So
//!   what has to hold is that *a mid-drag commit of each kind records one undo
//!   entry that Ctrl+Z fully reverts and Ctrl+Y fully replays* — for all three,
//!   including the two that had no such test before.
//! * **`viewport_pump_mirror.rs`** — the settler's own existence and its wiring.
//!   `every_cross_frame_gesture_has_a_settler_and_the_pump_calls_it` fails if any
//!   of the four `settle_orphaned_*` functions is deleted (the P21.3 audit's
//!   point: this file alone passed with `settle_orphaned_sculpt` removed, because
//!   it gates the *recorders* the settler calls and not the settler itself), and
//!   `the_orphan_settlers_run_outside_the_tool_gated_branches` pins that they are
//!   called *before* the branch that would otherwise have finished the stroke —
//!   which is the exact way the fix regresses.
//!
//! Neither half alone is the gate. Together they say: the settler exists, the
//! pump calls it, it is called where it can help, and what it calls does not
//! lose the edit.
//!
//! The **transaction** half below needs no such split: `settle_open_transaction`
//! is a `SceneDoc` door, so its failure — one unmatched `begin_transaction`
//! killing Ctrl+Z for the whole session — is reproduced and fixed here directly.

use glam::DVec2;
use inf_ecs::components::Terrain;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_terrain::{BiomeStroke, BrushOp, BrushParams, SplatStroke, Stroke};
use uuid::Uuid;

/// The terrain's full saved image — heights, weights, data maps and biome ids.
///
/// Bytes and not a field-by-field comparison, for the reason the sculpt tests
/// already state: "undo restored it" has to mean *the file it would write is the
/// file it would have written*.
fn image(doc: &SceneDoc, guid: Uuid) -> Vec<u8> {
    let (data, _) = doc.terrain_data_and_origin(guid).unwrap();
    let mut out = Vec::new();
    for (coord, tile) in data.tiles() {
        out.extend_from_slice(&coord.0.to_le_bytes());
        out.extend_from_slice(&coord.1.to_le_bytes());
        out.extend(inf_terrain::asset::encode_tile(tile).unwrap());
    }
    out
}

/// A document with the starter terrain, plus its centre and a sensible brush
/// radius.
fn fixture() -> (SceneDoc, Uuid, DVec2, f64) {
    let mut doc = SceneDoc::new();
    let guid = doc.edit_create(SpawnKind::Terrain, "Ground", None);
    let (min, max) = doc.terrain_bounds(guid).expect("the starter terrain");
    let centre = (min + max) * 0.5;
    let radius = ((max.x - min.x).min(max.y - min.y) * 0.25).max(1.0);
    (doc, guid, centre, radius)
}

/// The three in-flight stroke kinds, each laid down as a **multi-dab drag** and
/// then committed the way a settlement commits it — mid-gesture, from outside
/// the branch that would normally have finished it.
///
/// The dab loop is the point: a per-dab undo record would pass a single-dab
/// test and fail here, because the first Ctrl+Z would put back only the last
/// dab and the image would not match.
#[allow(clippy::type_complexity)]
fn kinds() -> Vec<(
    &'static str,
    Box<dyn Fn(&mut SceneDoc, Uuid, DVec2, f64) -> bool>,
)> {
    vec![
        (
            "height sculpt",
            Box::new(|doc: &mut SceneDoc, g: Uuid, c: DVec2, r: f64| {
                let mut stroke = Stroke::begin();
                for k in 0..4 {
                    let at = c + DVec2::new(k as f64 * r * 0.3, 0.0);
                    doc.sculpt_apply_dab(
                        g,
                        &mut stroke,
                        BrushOp::Raise,
                        BrushParams::new(at, r, 3.0),
                    );
                }
                doc.edit_commit_sculpt(g, stroke)
            }),
        ),
        (
            "splat paint",
            Box::new(|doc: &mut SceneDoc, g: Uuid, c: DVec2, r: f64| {
                let mut stroke = SplatStroke::begin(2);
                for k in 0..4 {
                    let at = c + DVec2::new(k as f64 * r * 0.3, 0.0);
                    doc.paint_apply_dab(g, &mut stroke, BrushParams::new(at, r, 1.0));
                }
                doc.edit_commit_paint(g, stroke)
            }),
        ),
        (
            "biome paint",
            Box::new(|doc: &mut SceneDoc, g: Uuid, c: DVec2, r: f64| {
                let mut stroke = BiomeStroke::begin(2);
                for k in 0..4 {
                    let at = c + DVec2::new(k as f64 * r * 0.3, 0.0);
                    doc.biome_apply_dab(g, &mut stroke, BrushParams::new(at, r, 1.0));
                }
                doc.edit_commit_biome(g, stroke)
            }),
        ),
    ]
}

/// **THE GATE.** For each of the three `DragStroke` kinds: a multi-dab drag
/// settled mid-gesture is **one** undo entry, Ctrl+Z restores the terrain
/// byte-for-byte, and Ctrl+Y puts the whole stroke back.
#[test]
fn a_settled_stroke_of_every_brush_kind_is_one_reversible_undo_step() {
    for (label, commit) in kinds() {
        let (mut doc, guid, centre, radius) = fixture();
        let before = image(&doc, guid);
        let undos = doc.undo_len();

        assert!(
            commit(&mut doc, guid, centre, radius),
            "{label}: the settlement recorded nothing, so the edit is UNREACHABLE by Ctrl+Z"
        );
        let after = image(&doc, guid);
        assert_ne!(after, before, "{label}: the fixture stroke changed nothing");
        assert_eq!(
            doc.undo_len(),
            undos + 1,
            "{label}: a settled stroke must be ONE step, not one per dab"
        );

        assert!(
            doc.undo(),
            "{label}: Ctrl+Z could not reach the settled edit"
        );
        assert_eq!(
            image(&doc, guid),
            before,
            "{label}: one undo put back less than the stroke took"
        );
        assert_eq!(doc.undo_len(), undos, "{label}");

        assert!(doc.redo(), "{label}");
        assert_eq!(
            image(&doc, guid),
            after,
            "{label}: redo did not restore the whole stroke"
        );
    }
}

/// **Nothing is lost, and nothing is invented.** A settlement of a stroke that
/// laid no dab at all (a click that missed the terrain, a tool switch on the
/// very first frame) records nothing — an empty undo entry would make Ctrl+Z
/// appear to do nothing, which is indistinguishable from the bug.
#[test]
fn settling_an_empty_stroke_records_nothing() {
    let (mut doc, guid, _c, _r) = fixture();
    let undos = doc.undo_len();
    let before = image(&doc, guid);

    assert!(!doc.edit_commit_sculpt(guid, Stroke::begin()));
    assert!(!doc.edit_commit_paint(guid, SplatStroke::begin(1)));
    assert!(!doc.edit_commit_biome(guid, BiomeStroke::begin(1)));

    assert_eq!(doc.undo_len(), undos, "an empty stroke recorded a step");
    assert_eq!(image(&doc, guid), before);
}

/// The **splat** and **biome** kinds really touch different layers of the tile,
/// so the byte comparison above is not accidentally testing the height buffer
/// three times.
///
/// Without this the gate would be vacuous in the way that matters: three
/// identical height strokes wearing three different stroke types.
#[test]
fn the_three_kinds_move_three_different_layers() {
    let (mut doc, guid, centre, radius) = fixture();
    let e = doc.entity_of(guid).unwrap();

    let heights_before: Vec<f32> = doc
        .world()
        .world()
        .get::<Terrain>(e)
        .unwrap()
        .data
        .tiles()
        .flat_map(|(_, t)| t.heights().to_vec())
        .collect();

    // Splat: weights move, heights do not.
    let mut stroke = SplatStroke::begin(2);
    doc.paint_apply_dab(guid, &mut stroke, BrushParams::new(centre, radius, 1.0));
    assert!(doc.edit_commit_paint(guid, stroke));
    {
        let data = &doc.world().world().get::<Terrain>(e).unwrap().data;
        let heights: Vec<f32> = data
            .tiles()
            .flat_map(|(_, t)| t.heights().to_vec())
            .collect();
        assert_eq!(heights, heights_before, "a splat paint moved the heights");
        assert!(
            data.tiles().any(|(_, t)| !t.weights().is_empty()),
            "a splat paint materialized no weight buffer"
        );
        assert!(data.biomes_are_default(), "a splat paint wrote biome ids");
    }

    // Biome: ids move, heights do not.
    let mut stroke = BiomeStroke::begin(3);
    doc.biome_apply_dab(guid, &mut stroke, BrushParams::new(centre, radius, 1.0));
    assert!(doc.edit_commit_biome(guid, stroke));
    {
        let data = &doc.world().world().get::<Terrain>(e).unwrap().data;
        let heights: Vec<f32> = data
            .tiles()
            .flat_map(|(_, t)| t.heights().to_vec())
            .collect();
        assert_eq!(heights, heights_before, "a biome paint moved the heights");
        assert!(!data.biomes_are_default(), "a biome paint wrote no ids");
    }

    // Height: the heights move.
    let mut stroke = Stroke::begin();
    doc.sculpt_apply_dab(
        guid,
        &mut stroke,
        BrushOp::Raise,
        BrushParams::new(centre, radius, 3.0),
    );
    assert!(doc.edit_commit_sculpt(guid, stroke));
    {
        let heights: Vec<f32> = doc
            .world()
            .world()
            .get::<Terrain>(e)
            .unwrap()
            .data
            .tiles()
            .flat_map(|(_, t)| t.heights().to_vec())
            .collect();
        assert_ne!(heights, heights_before, "a raise stroke moved no height");
    }
}

// ── the stranded transaction (P21.3 audit ruling) ───────────────────────────

/// **ONE UNMATCHED `begin_transaction` KILLS Ctrl+Z FOR THE SESSION.**
///
/// This is the failure, reproduced: open a transaction, never close it, and from
/// then on every later begin/commit pair bounces the nesting depth `1 → 2 → 1`
/// without ever closing the stranded one. Every edit is folded into it,
/// `undo_len()` stops growing, and undo silently does nothing — while the edits
/// land in the world, the document goes dirty and the save works.
///
/// Reachable from the viewport today: the pump opens `"Move"` when a gizmo drag
/// starts and commits it on release, **both inside the tool-gated select
/// branch**, so *hold a translate handle → Ctrl+Shift+P → `tool.sculpt` →
/// release* strands one.
#[test]
fn a_stranded_transaction_swallows_every_later_edit_until_it_is_settled() {
    let mut doc = SceneDoc::new();
    let a = doc.edit_create(SpawnKind::Empty, "A", None);
    let baseline = doc.undo_len();

    // The leak: a gesture opens a transaction and is interrupted before it can
    // close it.
    doc.begin_transaction("Move");
    doc.edit_rename(a, "moved");
    assert!(doc.has_open_transaction());

    // Every later edit — each of which opens and closes its OWN transaction —
    // now disappears into the stranded one.
    for i in 0..4 {
        doc.begin_transaction("later");
        doc.edit_rename(a, &format!("later-{i}"));
        doc.commit_transaction();
    }
    assert_eq!(
        doc.undo_len(),
        baseline,
        "the history grew — the fixture is not reproducing the leak"
    );
    assert!(
        doc.has_open_transaction(),
        "four matched commits closed the stranded transaction on their own"
    );

    // THE FIX: settling closes it, and everything it swallowed becomes one
    // reachable undo step.
    assert!(doc.settle_open_transaction(), "nothing to settle?");
    assert!(!doc.has_open_transaction());
    assert_eq!(doc.undo_len(), baseline + 1, "the settled entry is missing");
    assert!(doc.undo(), "Ctrl+Z could not reach the settled edit");
    assert_eq!(
        doc.world().name_of(doc.entity_of(a).unwrap()),
        Some("A"),
        "undo did not put the name back"
    );

    // …and the history works normally again afterwards.
    doc.edit_rename(a, "after");
    assert_eq!(doc.undo_len(), baseline + 1);
    assert!(doc.undo());
}

/// Settling when nothing is open is a no-op, and settling an **empty**
/// transaction records nothing — an undo entry that reverts nothing would make
/// Ctrl+Z appear broken in exactly the way this fix exists to prevent.
#[test]
fn settling_records_nothing_when_there_is_nothing_to_settle() {
    let mut doc = SceneDoc::new();
    let undos = doc.undo_len();
    assert!(!doc.settle_open_transaction(), "nothing was open");

    doc.begin_transaction("empty");
    assert!(doc.has_open_transaction());
    assert!(
        !doc.settle_open_transaction(),
        "an empty transaction recorded"
    );
    assert!(!doc.has_open_transaction(), "…but it was still closed");
    assert_eq!(doc.undo_len(), undos);
}

/// A **nested** stranded transaction settles too. `commit` only closes at depth
/// 1, so a gesture that opened two and closed one leaves `depth = 1` with the
/// transaction still open — the same dead end, one level down.
#[test]
fn settling_closes_a_transaction_at_any_nesting_depth() {
    let mut doc = SceneDoc::new();
    let a = doc.edit_create(SpawnKind::Empty, "A", None);
    let undos = doc.undo_len();

    doc.begin_transaction("outer");
    doc.begin_transaction("inner");
    doc.edit_rename(a, "x");
    doc.commit_transaction(); // inner — unwinds to depth 1, closes nothing
    assert!(doc.has_open_transaction());
    assert_eq!(doc.undo_len(), undos);

    assert!(doc.settle_open_transaction());
    assert_eq!(doc.undo_len(), undos + 1);
    assert!(doc.undo());
    assert_eq!(doc.world().name_of(doc.entity_of(a).unwrap()), Some("A"));
}

/// **A replaced document is a different document**, and another thread can tell.
///
/// `scene_open` / `scene_new` do `*doc = …` under the scene lock; the viewport
/// thread holds gestures pointing into the document that was there a moment ago.
/// Settling one of those would commit the **old** level's edit into the new
/// document, where a single Ctrl+Z then applies it — so the host abandons them
/// instead, and this id is how it notices.
///
/// A monotone counter and not a content hash: two identical levels opened in
/// sequence are still two different documents to anything holding a mid-gesture
/// reference into one of them.
#[test]
fn every_document_instance_has_its_own_id() {
    let a = SceneDoc::new();
    let b = SceneDoc::new();
    assert_ne!(a.doc_id(), b.doc_id(), "two documents share an identity");
    assert_eq!(a.doc_id(), a.doc_id(), "an id must be stable");

    // Editing does not change it — only replacement does.
    let mut c = SceneDoc::new();
    let before = c.doc_id();
    let g = c.edit_create(SpawnKind::Empty, "x", None);
    c.edit_rename(g, "y");
    assert_eq!(c.doc_id(), before);

    // …and the swap `scene_open` performs really produces a new one.
    c = SceneDoc::new();
    assert_ne!(c.doc_id(), before, "a replaced document kept its identity");
}
