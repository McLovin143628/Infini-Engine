// vsm_clear.wgsl — **the per-page clear** (P27.3).
//
// P27.2 cleared the whole atlas with `LoadOp::Clear` and re-rasterized every
// resident page in the same pass, so every slot that mattered was rewritten in
// the pass that wiped it. P27.3 keeps pages across frames, so the pass loads
// (`LoadOp::Load`) and the clear becomes a **rectangle** — one per dirty page,
// leaving every cached page's texels exactly as the frame that drew them left
// them.
//
// ── why this is a draw and not a `clear_texture`
//
// `LoadOp::Clear` clears an attachment, not a scissor rectangle, and
// `CommandEncoder::clear_texture` clears whole subresources — an atlas slot is a
// *rectangle of one texture* and there is no view of a rectangle (the same
// inversion `vsm_raster`'s module docs describe for the raster itself). A
// depth-only triangle with `depth_compare: Always` writes the clear value
// wherever it is rasterized, which makes "which texels" a question the scissor
// answers.
//
// ── and this is where the scissor stops being defence in depth
//
// The P27.2 ledger recorded that deleting `set_scissor_rect` failed nothing:
// clipping happens against the clip volume *before* the viewport transform, so a
// caster triangle cannot land outside its own viewport rect whatever the scissor
// says. That argument does not reach the clear, because the clear is drawn with
// the viewport left at the **whole atlas** — deliberately, since a clear needs a
// rectangle and not a projection. Without the scissor this triangle covers the
// entire atlas and wipes every cached page. The mutation flips from survivor to
// killed, and `a_cached_page_is_not_cleared_by_its_neighbours_clear` is what
// kills it.
//
// One triangle covering NDC `[-1, 3]²` — the standard oversized fullscreen
// triangle, so three vertices cover the square with no diagonal seam.

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    // `inf_render::vsm::VSM_DEPTH_CLEAR` — reverse-Z, so 0 is "far / nothing".
    // A literal, because a uniform for one constant would be a bind group this
    // pipeline otherwise does not have; `the_page_clear_writes_the_depth_clear_value`
    // reads it back out of this file.
    return vec4<f32>(x, y, 0.0, 1.0);
}
