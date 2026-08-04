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
//! * **`viewport_pump_mirror.rs`** — the wiring. `the_win32_pump_still_drives_…`
//!   pins that the pump calls both settlers, and
//!   `the_orphan_settlers_run_outside_the_tool_gated_branches` pins that they are
//!   called *before* the branch that would otherwise have finished the stroke —
//!   which is the entire content of the fix and the exact way it would regress.
//!
//! Neither half alone is the gate. Together they say: the pump calls it, it is
//! called where it can help, and what it calls does not lose the edit.

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
