//! **The undo stack is bounded by bytes, and the report says how many**
//! (Hardening D).
//!
//! Two halves of one finding:
//!
//! * the history was bounded by [`HISTORY_LIMIT`] — a **count** — which is a
//!   bound on entries and not on memory. 256 sculpt strokes over a large terrain
//!   patch honour it perfectly while the editor holds hundreds of megabytes;
//! * `MemoryReport` charged every entry a flat 512 bytes, so the largest thing
//!   the editor holds was reported as the smallest. Every delta type in the tree
//!   already had a `memory_bytes()` and not one of them had a caller.
//!
//! The arms below are shaped against the *contrast*, because a byte estimate
//! that is merely non-zero proves nothing: a stroke's record must cost
//! **orders of magnitude** more than a rename's, and the flat charge is exactly
//! what could not tell them apart.

use glam::DVec2;
use inf_editor_core::diagnostics::MemoryReport;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_terrain::{BrushOp, BrushParams, Stroke};

/// A document with the starter terrain, its centre and a sensible brush radius.
fn fixture() -> (SceneDoc, uuid::Uuid, DVec2, f64) {
    let mut doc = SceneDoc::new();
    let guid = doc.edit_create(SpawnKind::Terrain, "Ground", None);
    let (min, max) = doc.terrain_bounds(guid).expect("the starter terrain");
    let centre = (min + max) * 0.5;
    let radius = ((max.x - min.x).min(max.y - min.y) * 0.25).max(1.0);
    (doc, guid, centre, radius)
}

/// Lay one multi-dab sculpt stroke and commit it as one undo entry.
fn sculpt(doc: &mut SceneDoc, guid: uuid::Uuid, centre: DVec2, radius: f64) {
    let mut stroke = Stroke::begin();
    for k in 0..4 {
        let at = centre + DVec2::new(k as f64 * radius * 0.3, 0.0);
        doc.sculpt_apply_dab(
            guid,
            &mut stroke,
            BrushOp::Raise,
            BrushParams::new(at, radius, 3.0),
        );
    }
    assert!(
        doc.edit_commit_sculpt(guid, stroke),
        "the fixture stroke must actually record an undo entry"
    );
}

#[test]
fn a_sculpt_stroke_costs_orders_of_magnitude_more_than_a_rename() {
    let (mut doc, guid, centre, radius) = fixture();
    let empty = doc.undo_bytes();

    doc.edit_rename(guid, "Renamed");
    let after_rename = doc.undo_bytes();
    assert!(
        after_rename > empty,
        "a recorded edit must cost something ({empty} -> {after_rename})"
    );

    sculpt(&mut doc, guid, centre, radius);
    let after_sculpt = doc.undo_bytes();
    let rename_cost = after_rename - empty;
    let sculpt_cost = after_sculpt - after_rename;
    assert!(
        sculpt_cost > rename_cost * 100,
        "a multi-dab height stroke is charged {sculpt_cost} bytes against a rename's \
         {rename_cost} — the whole point of the fix is that a flat per-entry charge \
         cannot tell those two apart"
    );
    assert_eq!(
        doc.undo_len(),
        3,
        "and all three are on the stack (the terrain's own creation is the first)"
    );
}

#[test]
fn the_memory_report_charges_the_undo_stack_what_it_actually_holds() {
    let (mut doc, guid, centre, radius) = fixture();
    sculpt(&mut doc, guid, centre, radius);

    let report = MemoryReport::for_scene(&doc);
    assert_eq!(report.undo_depth, 2, "the create and the stroke");
    assert_eq!(
        report.undo_bytes,
        doc.undo_bytes() as u64,
        "the report quotes the document's own figure, not a per-entry guess"
    );
    // The old flat charge was 512 bytes per entry. One stroke is far past it,
    // which is the measurement that says the report stopped lying.
    assert!(
        report.undo_bytes > 512 * 16,
        "one sculpt stroke is charged {} bytes — the flat 512-per-entry estimate \
         it replaced would have said 512",
        report.undo_bytes
    );
    assert!(
        report.total_estimate_bytes >= report.scene_bytes + report.undo_bytes,
        "and the total carries it"
    );
}

#[test]
fn an_undone_stroke_is_still_charged_because_redo_still_holds_it() {
    let (mut doc, guid, centre, radius) = fixture();
    sculpt(&mut doc, guid, centre, radius);
    let held = doc.undo_bytes();

    assert!(doc.undo(), "the stroke undoes");
    assert_eq!(doc.undo_len(), 1, "the terrain's creation is still there");
    assert_eq!(doc.redo_len(), 1);
    assert_eq!(
        doc.undo_bytes(),
        held,
        "moving an entry from one stack to the other frees nothing — the record is \
         the same record, and a budget that only counted the undo half would report \
         a drop that did not happen"
    );
}
