//! Long-session soak seed (P15.2, deliverable 8).
//!
//! A deterministic 10k-cycle random-ish edit/undo/redo/save workout over a
//! [`SceneDoc`], asserting that invariants hold every step and that memory does
//! **not** grow unboundedly — the undo stack stays bounded by the history limit
//! and the live entity count / scene size stay within a fixed ceiling.
//!
//! `#[ignore]`d so it never runs in the normal `cargo nextest` job (it is a
//! minutes-scale stress run); run it manually / nightly with:
//!
//! ```text
//! cargo test -p inf-editor-core --test soak -- --ignored --nocapture
//! ```

use inf_editor_core::diagnostics::MemoryReport;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::{serialize, SceneDoc};

/// Tiny deterministic xorshift RNG (no `rand` dep — the soak must be repeatable).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

fn transform_path(doc: &SceneDoc) -> &'static str {
    doc.world()
        .registry()
        .editable()
        .iter()
        .find(|c| c.display == "Transform")
        .expect("Transform is registered")
        .type_path
}

#[test]
#[ignore = "soak: minutes-scale; run with --ignored"]
fn soak_10k_edit_undo_save_cycles() {
    const CYCLES: usize = 10_000;
    // The undo history limit (`scene::undo::HISTORY_LIMIT`); the undo depth must
    // never exceed this — the core "memory doesn't grow unboundedly" guarantee.
    const UNDO_LIMIT: u64 = 256;
    // A generous live-entity ceiling: creates are gated on it, and deletes keep
    // the world from ballooning, so entity count (and scene bytes) stay bounded.
    const ENTITY_CAP: u64 = 300;

    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut doc = SceneDoc::with_demo();
    let tp = transform_path(&doc);

    let mut peak_bytes = 0u64;
    let mut peak_entities = 0u64;
    let mut saves = 0u64;

    for i in 0..CYCLES {
        let order = doc.order().to_vec();
        let n = order.len() as u64;
        let pick = |rng: &mut Rng| -> Option<uuid::Uuid> {
            if order.is_empty() {
                None
            } else {
                Some(order[rng.below(order.len() as u64) as usize])
            }
        };

        match rng.below(8) {
            // Create (gated on the entity ceiling).
            0..=2 if n < ENTITY_CAP => {
                doc.edit_create(SpawnKind::Cube, &format!("Cube-{i}"), None);
            }
            // Rename.
            3 => {
                if let Some(g) = pick(&mut rng) {
                    doc.edit_rename(g, &format!("Renamed-{i}"));
                }
            }
            // Move (set translation).
            4 => {
                if let Some(g) = pick(&mut rng) {
                    let v = [
                        (rng.below(100) as f64) * 0.1,
                        (rng.below(100) as f64) * 0.1,
                        (rng.below(100) as f64) * 0.1,
                    ];
                    doc.write_prop(g, tp, "translation", &inf_ecs::PropValue::Vec3(v));
                }
            }
            // Delete.
            5 => {
                if let Some(g) = pick(&mut rng) {
                    doc.edit_delete(&[g]);
                }
            }
            // Undo.
            6 => {
                doc.undo();
            }
            // Redo (or a create fallback so the world keeps churning).
            _ => {
                if !doc.redo() && n < ENTITY_CAP {
                    doc.edit_create(SpawnKind::Empty, &format!("Empty-{i}"), None);
                }
            }
        }

        // Invariant: the undo stack never exceeds the history limit.
        let report = MemoryReport::for_scene(&doc);
        assert!(
            report.undo_depth <= UNDO_LIMIT,
            "cycle {i}: undo depth {} exceeded the limit {UNDO_LIMIT}",
            report.undo_depth
        );
        assert!(
            report.entities <= ENTITY_CAP + 4,
            "cycle {i}: entity count {} blew past the ceiling",
            report.entities
        );
        peak_bytes = peak_bytes.max(report.total_estimate_bytes);
        peak_entities = peak_entities.max(report.entities);

        // Periodically round-trip through save/load (exercises serialization and
        // proves a reloaded doc stays consistent), continuing on the loaded doc.
        if i % 1000 == 999 {
            let bytes = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode");
            let file = serialize::decode(&bytes).expect("decode");
            let mut reloaded = SceneDoc::new();
            serialize::apply_to_doc(&mut reloaded, &file);
            assert_eq!(
                reloaded.order().len(),
                doc.order().len(),
                "cycle {i}: reload changed the entity count"
            );
            doc = reloaded;
            saves += 1;
        }
    }

    // Final memory invariant: bounded by construction (undo capped, entities
    // capped), so the peak stays within a sane ceiling — a leak would blow this.
    let final_report = MemoryReport::for_scene(&doc);
    assert!(final_report.undo_depth <= UNDO_LIMIT);
    // ~ENTITY_CAP entities of a small scene: comfortably under 32 MiB estimate.
    assert!(
        peak_bytes < 32 * 1024 * 1024,
        "peak memory estimate {peak_bytes} bytes is implausibly large — possible leak"
    );

    eprintln!(
        "soak complete: {CYCLES} cycles, {saves} save/reload round-trips, \
         peak {peak_entities} entities, peak ~{} KiB, final undo depth {}",
        peak_bytes / 1024,
        final_report.undo_depth
    );
}
