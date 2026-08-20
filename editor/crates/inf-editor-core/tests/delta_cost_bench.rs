//! **The measurement behind lens 3's P23** — what a `world://delta` actually
//! costs on a big level.
//!
//! Not a gate: `#[ignore]`d, because a timing assertion on a shared CI runner is
//! a flake generator (the `dig_stall_bench` precedent, verbatim). Run it with:
//!
//! ```sh
//! cargo test -p inf-editor-core --test delta_cost_bench --release -- --ignored --nocapture
//! ```
//!
//! The finding it tests: `emit_world_delta` calls `SceneDoc::snapshot()`, which
//! builds a `SceneNode` — with `guid.to_string()`, `name.to_string()`, … — for
//! **every** entity, then `diff` builds two `HashMap`s over both snapshots. That
//! fires on every gizmo-drag mouse-move and every sculpt dab, and the proposed
//! fix is to emit only the guids the `EditCommand` touched.
//!
//! Whether that is worth a Ring-1 dirty-set is a question about *milliseconds*,
//! so this prints them: the whole `snapshot` + `diff` round trip a
//! one-entity-moved delta pays, at three level sizes.

use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::delta::diff;
use inf_editor_core::scene::SceneDoc;
use std::time::Instant;

/// A document with `n` sibling entities under the root, each named and
/// transform-bearing — the shape a real level's outliner has.
fn doc_with(n: usize) -> SceneDoc {
    let mut doc = SceneDoc::new();
    for i in 0..n {
        doc.edit_create(SpawnKind::Cube, &format!("Prop {i}"), None);
    }
    doc
}

#[test]
#[ignore = "a measurement, not a gate — see the module docs"]
fn what_one_world_delta_costs() {
    println!("entities | snapshot |     diff | old total |    drag | select-only | nod | speedup");
    for n in [1_000usize, 5_000, 15_000, 50_000, 100_000] {
        let mut doc = doc_with(n);

        // Warm: the first snapshot allocates its Vec capacity.
        let prev = doc.snapshot();

        // Move exactly one entity — the gizmo-drag frame.
        let guid = prev.nodes[n / 2].guid.clone();
        let id: uuid::Uuid = guid.parse().expect("a snapshot guid parses");
        doc.select(&[id], false);

        let mut snap_ns = 0u128;
        let mut diff_ns = 0u128;
        const REPS: u32 = 20;
        for _ in 0..REPS {
            let t = Instant::now();
            let next = doc.snapshot();
            snap_ns += t.elapsed().as_nanos();

            let t = Instant::now();
            let _ = diff(&prev, &next);
            diff_ns += t.elapsed().as_nanos();
        }
        let snap_ms = snap_ns as f64 / REPS as f64 / 1e6;
        let diff_ms = diff_ns as f64 / REPS as f64 / 1e6;

        // IB-13: the same frame through the scoped projection. Two frames are
        // timed, because they cost different things and only one of them is this
        // item's:
        //
        // * a **drag** frame — a transform write — pays `EcsWorld::propagate`,
        //   which is a full DFS over the world with a bundle insert per entity.
        //   That is transform work, not projection work, and it is O(world) by
        //   construction until propagation itself becomes incremental.
        // * a **select** frame does not dirty the world, so propagation is
        //   skipped and what is left IS the projection.
        let mut drag_ns = 0u128;
        let mut sel_ns = 0u128;
        let mut nodes = 0usize;
        let _ = doc.snapshot(); // re-seed the projection baseline
        for i in 0..REPS {
            doc.edit_set_transform(
                id,
                inf_ecs::components::Transform {
                    translation: inf_ecs::Vec3d::new(f64::from(i), 0.0, 0.0),
                    ..inf_ecs::components::Transform::IDENTITY
                },
            );
            let t = Instant::now();
            let d = doc.project_delta();
            drag_ns += t.elapsed().as_nanos();
            nodes = d.added.len() + d.updated.len() + d.removed.len();

            doc.select(&[id], false);
            let t = Instant::now();
            let _ = doc.project_delta();
            sel_ns += t.elapsed().as_nanos();
        }
        let drag_ms = drag_ns as f64 / REPS as f64 / 1e6;
        let sel_ms = sel_ns as f64 / REPS as f64 / 1e6;
        let old = snap_ms + diff_ms;
        println!(
            "{n:>8} | {snap_ms:>6.3}ms | {diff_ms:>6.3}ms | {old:>7.3}ms | {drag_ms:>7.4}ms \
             | {sel_ms:>8.4}ms | {nodes:>3} | {:>5.0}x",
            old / drag_ms.max(1e-9)
        );
    }
}

/// **The gate the measurement earned.**
///
/// The repair is a hoist: the parent→children index is built once per snapshot
/// instead of rescanned per node. Nothing observable changes — the snapshot is
/// byte-identical — so no value arm can see a regression, and a timing
/// assertion on a shared CI runner is a flake generator. Per the campaign's
/// standing rule, the enforcement is a **source pin** on the shape.
///
/// This is not run with `--ignored`: it is the part that protects the 1 014×.
#[test]
fn node_of_does_not_rescan_the_order_list() {
    let src = include_str!("../src/scene/doc.rs");
    let start = src
        .find("fn node_of(")
        .expect("SceneDoc::node_of still exists");
    // Brace-balance from the signature so the scope really is that function.
    let mut depth = 0i32;
    let mut opened = false;
    let mut end = start;
    for (i, ch) in src[start..].char_indices() {
        match ch {
            '{' => {
                depth += 1;
                opened = true;
            }
            '}' => depth -= 1,
            _ => {}
        }
        if opened && depth == 0 {
            end = start + i + 1;
            break;
        }
    }
    let body = &src[start..end];
    assert!(end > start, "node_of never closes");

    // **Whitespace-stripped, because the defect's own formatting defeats a
    // literal match.** The first version of this pin looked for `self.order` and
    // SURVIVED its mutation: rustfmt writes the rescan as `self` and `.order` on
    // two lines, which is exactly how the original read. A byte pin cannot see a
    // semantic change (the P23 law) — and it cannot see a line break either.
    let squeezed: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        !squeezed.contains("self.order"),
        "`node_of` walks `self.order` again. That is the O(n^2) this measured at \
         3 277 ms per world://delta on a 15 000-entity level — and a delta is \
         emitted on every gizmo-drag mouse-move. The index belongs in `snapshot`, \
         built once.\n\nbody:\n{body}"
    );
    assert!(
        body.contains("children_of"),
        "`node_of` no longer takes the hoisted parent index; if the children came \
         back from somewhere else, this pin needs rewriting rather than deleting"
    );

    // …and the hoist is really a hoist: the index is BUILT in exactly one place
    // in the whole file, and both projections reach it through that one place.
    //
    // (This clause moved with IB-13, per this test's own instruction that a
    // relocated index needs the pin rewritten rather than deleted. It was
    // `children_of.entry` appearing once *within `snapshot`*; the builder is now
    // `hierarchy_index`, shared by `snapshot` and `project_delta`, which is a
    // stronger property than the one it replaces — the two projections cannot
    // build the hierarchy differently, and a snapshot disagreeing with the delta
    // that follows it about who owns a child is a tree the Outliner cannot draw.)
    assert_eq!(
        src.matches("children_of.entry").count(),
        1,
        "the parent index must be built in exactly one place"
    );
    assert!(
        src.contains("fn hierarchy_index("),
        "the shared index builder is gone; if it moved, rewrite this pin"
    );
    assert_eq!(
        src.matches("self.hierarchy_index()").count(),
        2,
        "`snapshot` and `project_delta`'s full arm are the two callers of the \
         one index builder — {} call it",
        src.matches("self.hierarchy_index()").count()
    );
    for who in [
        "pub fn snapshot(&mut self)",
        "pub fn project_delta(&mut self)",
    ] {
        let at = src
            .find(who)
            .unwrap_or_else(|| panic!("{who} still exists"));
        let scope = &src[at..(at + 4000).min(src.len())];
        assert!(
            scope.contains("self.hierarchy_index()"),
            "`{who}` no longer goes through the shared index"
        );
    }
}
