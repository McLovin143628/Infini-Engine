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
    println!("entities |  snapshot |      diff |     total |  nodes in delta");
    for n in [1_000usize, 5_000, 15_000] {
        let mut doc = doc_with(n);

        // Warm: the first snapshot allocates its Vec capacity.
        let prev = doc.snapshot();

        // Move exactly one entity — the gizmo-drag frame.
        let guid = prev.nodes[n / 2].guid.clone();
        let id: uuid::Uuid = guid.parse().expect("a snapshot guid parses");
        doc.select(&[id], false);

        let mut snap_ns = 0u128;
        let mut diff_ns = 0u128;
        let mut updated = 0usize;
        const REPS: u32 = 20;
        for _ in 0..REPS {
            let t = Instant::now();
            let next = doc.snapshot();
            snap_ns += t.elapsed().as_nanos();

            let t = Instant::now();
            let d = diff(&prev, &next);
            diff_ns += t.elapsed().as_nanos();
            updated = d.added.len() + d.updated.len() + d.removed.len();
        }
        let snap_ms = snap_ns as f64 / REPS as f64 / 1e6;
        let diff_ms = diff_ns as f64 / REPS as f64 / 1e6;
        println!(
            "{n:>8} | {snap_ms:>7.3}ms | {diff_ms:>7.3}ms | {:>7.3}ms | {updated:>4} of {n}",
            snap_ms + diff_ms
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

    // …and the hoist is really in `snapshot`, exactly once.
    let snap = src
        .find("pub fn snapshot(&mut self)")
        .expect("SceneDoc::snapshot still exists");
    let scope = &src[snap..(snap + 2500).min(src.len())];
    assert_eq!(
        scope.matches("children_of.entry").count(),
        1,
        "the parent index must be built once, in `snapshot`"
    );
}
