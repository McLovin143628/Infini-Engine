//! **Every writer of `scene.terrains` outside the projection must tell the seam**
//! (round-2 finding B9).
//!
//! # The finding
//!
//! Hardening Wave E made `inf_render::apply_seam` skippable inside
//! `rebuild_scene`: when every volume and every terrain was carried forward
//! unchanged — nothing dropped, nothing added, nothing rebuilt — last frame's
//! per-vertex seam terms are still the right ones, and the per-vertex walk
//! (which samples every terrain per vertex) is skipped whole.
//!
//! That argument is sound in the **player**, whose only writer of
//! `scene.terrains` is `project_scene` itself. This host has two more:
//!
//! * `sync_streamed_terrain` — the camera advanced the streamer's cut, so new
//!   pages are resident and old ones are not. It re-projects each streamed
//!   slot **in place**, because `sync_from_doc` is version-gated and the camera
//!   does not bump the document version (nor should it).
//! * `after_terrain_edit` — a sculpt dab. Same shape, same in-place write, once
//!   per dab of every stroke.
//!
//! Neither called `apply_seam`, and — this is the part that makes it invisible
//! — neither could be caught by the skip condition, because the skip compares
//! the projection against **itself**: by the time `rebuild_scene` next runs,
//! `prev_terrains` holds the list these writers already updated, every
//! `tile_signature` matches, everything is carried, nothing is left over, and
//! the seam is skipped. A cave mouth beside a streamed terrain would keep terms
//! computed against pre-dab, pre-page heights for the rest of the session,
//! while the shipped build recomputes them — a new **editor-vs-shipping
//! divergence in the seam the mirrored-pair discipline exists to protect**.
//!
//! # Why this is a source gate
//!
//! `EngineHost` owns a real GPU context and a `SurfaceTarget`; there is no way
//! to construct one in a test, page a terrain in and read a voxel vertex's seam
//! term back. And the failure mode is silence — the mouth shades slightly
//! wrong, at a boundary, for a session. So the enforcement is the one Wave C
//! stated: *where a fix's correctness is "the code still does this", only a
//! source pin is enforcement.*
//!
//! What is pinned is deliberately **the whole rule, not the two fixes**: the
//! gate finds every assignment into `self.scene.terrains` outside
//! `rebuild_scene` and requires the enclosing function to set `seam_dirty`. A
//! third writer added later fails this file by name rather than shipping the
//! same defect a third time.

/// The host's source text. Line endings are normalized before any search — the
/// P22 CRLF law, met by every gate in this repo that reads a `.rs`.
fn host_src() -> String {
    include_str!("../src/host.rs").replace("\r\n", "\n")
}

/// Every `fn NAME(` in the file, paired with its body up to the closing brace
/// at method indentation.
fn methods(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(rel) = src[at..].find("fn ") {
        let start = at + rel;
        at = start + 3;
        let Some(paren) = src[start..].find('(') else {
            continue;
        };
        let name = src[start + 3..start + paren].trim().to_string();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let rest = &src[start..];
        let end = match rest.find("\n    }\n") {
            Some(e) => e,
            None => rest.len(),
        };
        out.push((name, rest[..end].to_string()));
    }
    out
}

/// The one writer that is allowed not to arm the flag, because it is the
/// projection itself — it consumes the flag rather than setting it.
const THE_PROJECTION: &str = "rebuild_scene";

#[test]
fn every_out_of_band_terrain_writer_arms_the_seam() {
    let src = host_src();
    let mut writers: Vec<String> = Vec::new();
    let mut unarmed: Vec<String> = Vec::new();

    for (name, body) in methods(&src) {
        if name == THE_PROJECTION {
            continue;
        }
        // An in-place write into the projected terrain list. Both known writers
        // reach it through `self.scene.terrains.get_mut(..)`; a direct
        // `self.scene.terrains[i] = ..` or a `push` would be the same defect.
        let writes = body.contains("self.scene.terrains.get_mut(")
            || body.contains("self.scene.terrains.push(")
            || body.contains("self.scene.terrains[");
        if !writes {
            continue;
        }
        // `std::mem::take` alone is the projection's own borrow of the list and
        // is not a write of new content; every real writer below also assigns.
        if !body.contains(" = projected") && !body.contains("*dst = ") {
            continue;
        }
        writers.push(name.clone());
        if !body.contains("self.seam_dirty = true") {
            unarmed.push(name);
        }
    }

    assert!(
        writers.len() >= 2,
        "this gate found {} out-of-band terrain writer(s); it is calibrated \
         against the two the finding names (`sync_streamed_terrain` and \
         `after_terrain_edit`), so a smaller number means the search no longer \
         matches the code and the gate is vacuous",
        writers.len()
    );
    assert!(
        unarmed.is_empty(),
        "these functions re-project `scene.terrains` in place without setting \
         `seam_dirty`: {unarmed:?}. The next `rebuild_scene` compares the list \
         they already updated against itself, carries everything, and skips \
         `apply_seam` — so a voxel volume beside that terrain keeps seam terms \
         computed against the old heights while the shipped player recomputes \
         them. Set `self.seam_dirty = true` (and `self.synced_version = None`, \
         since the camera does not bump the document version)."
    );

    // And each of them must also force the projection to run at all: the flag
    // is only read inside `rebuild_scene`, which `sync_from_doc` skips when the
    // document version has not moved — and a camera step does not move it.
    for name in &writers {
        let body = methods(&src)
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b)
            .expect("just found it");
        assert!(
            body.contains("self.synced_version = None"),
            "`{name}` arms `seam_dirty` but never invalidates `synced_version`, \
             so the projection that would consume the flag may not run for as \
             long as the document is unedited"
        );
    }
}

#[test]
fn the_projection_consumes_the_flag_and_clears_it() {
    let src = host_src();
    let body = methods(&src)
        .into_iter()
        .find(|(n, _)| n == THE_PROJECTION)
        .map(|(_, b)| b)
        .expect("`rebuild_scene` occurs nowhere — was it renamed?");

    assert!(
        body.contains("if self.seam_dirty"),
        "`rebuild_scene` no longer reads `seam_dirty`, so the two out-of-band \
         writers set a flag nothing acts on — which is the finding with an \
         extra field"
    );
    assert!(
        body.contains("inf_render::apply_seam("),
        "`rebuild_scene` no longer calls `apply_seam` at all"
    );
    assert!(
        body.contains("self.seam_dirty = false"),
        "`rebuild_scene` never clears `seam_dirty`, so once armed the seam is \
         recomputed on every projection for ever — which retires the Wave E \
         skip rather than correcting it"
    );
    // The clear must be inside the branch that actually recomputed. A clear
    // above the `if` would disarm the flag on a projection that skipped.
    let at_if = body.find("if self.seam_dirty").expect("checked above");
    let at_clear = body.find("self.seam_dirty = false").expect("checked above");
    assert!(
        at_clear > at_if,
        "`seam_dirty` is cleared before the branch that would act on it"
    );
}
