//! **The layered-terrain gate** (Wave-T audit): the texture document's §3 B —
//! *"For terrain or large props, store blend weights in a virtual texture mask
//! and sample shared, tileable PBR materials."*
//!
//! # Why this file exists
//!
//! Wave T shipped `terrain_layers()` in `terrain.wgsl` and **nothing anywhere in
//! the workspace named it**. A repo-wide search for the function turned up the
//! shader and the memo and no test at all, so:
//!
//! * the wave's own review found the layer uv derived from `world_local` — which
//!   is rebased every time the floating origin snaps — and fixed it to the
//!   absolute world XZ, and *no arm measured either the defect or the fix*;
//! * the 1/256 weight gate, the coverage sum and the renormalisation that stops
//!   a half-textured terrain going dark were all prose.
//!
//! # What this file can and cannot assert, said plainly
//!
//! `terrain_layers` calls `dpdx`/`dpdy`, which are fragment-stage builtins, so it
//! **cannot be executed from a compute probe** the way `vt_surface` is in
//! `vt_sampling.rs`, and executing it from a fragment probe means standing up
//! terrain's four bind groups including the whole lit environment. So this takes
//! the other shape the house already uses for exactly this situation
//! (`vt_sampling::the_wgsl_implements_the_twin_above`): a **CPU twin that
//! carries the measurement**, plus a **source gate scoped to the function** that
//! pins the twin and the shader to the same arithmetic.
//!
//! The honest bound, and it is recorded in
//! `docs/memos/wave-t-textures-disposition.md` rather than only here: the
//! textured branch of the terrain fragment shader has **not been executed on a
//! GPU**. What *is* executed is the untextured default, by every committed golden — which
//! is what makes "the goldens did not move" a measurement of the no-op and not of
//! the feature.

/// The shipped shader, read the way the gates read it. `.wgsl` is `text eol=lf`
/// in `.gitattributes`, so the substrings below mean the same thing on every
/// checkout — the P22 CRLF law.
const TERRAIN_WGSL: &str = include_str!("../src/shaders/terrain.wgsl");

/// The body of one WGSL function, by **brace matching from its signature** —
/// a SCOPE, not a whole-file grep (the P23 law: a byte pin that reads the file
/// cannot see where a spelling moved to).
fn wgsl_fn(src: &str, signature: &str) -> String {
    let start = src
        .find(signature)
        .unwrap_or_else(|| panic!("`{signature}` is not in terrain.wgsl any more"));
    let open = src[start..]
        .find('{')
        .expect("a function signature with no body");
    // `&src.as_bytes()[..]` rather than `src[..].as_bytes()` — the same bytes,
    // and the spelling clippy's `sliced_string_as_bytes` asks for because the
    // latter re-validates a UTF-8 boundary it has already proved.
    let bytes = &src.as_bytes()[start + open..];
    let mut depth = 0i32;
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return src[start..start + open + i + 1].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("`{signature}` never closes");
}

/// Code only: comments are prose and a gate that reads them is pinning a
/// paragraph.
fn code_of(body: &str) -> String {
    body.lines()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// **The layer uv is anchored to the WORLD, and the shader spells it that way**
/// (Wave-T audit).
///
/// # The defect, as a number
///
/// `in.world_local` is the fragment's position *relative to the floating
/// origin*, and the origin is re-anchored whenever the camera drifts past
/// `inf_math::REBASE_DISTANCE` — landing on a multiple of `inf_math::ORIGIN_SNAP`
/// (10 m). A uv derived from it is therefore not a property of the ground: it
/// steps by `Δorigin / tex_scale` tiles at every rebase, and the ground texture
/// slides out from under the terrain while the player walks in a straight line.
///
/// The twin below measures it rather than asserting it: at the *smallest*
/// possible non-zero rebase — one snap quantum, 10 m — a render-local uv moves
/// **2.5 tiles** at a 4 m tiling, and a real rebase is at least
/// `REBASE_DISTANCE`, i.e. two orders of magnitude worse. The world-anchored
/// derivation moves by exactly zero, which is the whole claim.
#[test]
fn a_layer_uv_is_world_anchored_across_an_origin_rebase() {
    // ── the twin ────────────────────────────────────────────────────────────
    // Line for line with `terrain_layers`, and the gate below pins that they
    // stay together.
    let world_anchored = |world_local_xz: [f32; 2], axis: [f32; 2], tex_scale: f32| -> [f32; 2] {
        let inv = 1.0 / tex_scale.max(0.001);
        [
            (world_local_xz[0] - axis[0]) * inv,
            (world_local_xz[1] - axis[1]) * inv,
        ]
    };
    // …and the spelling it replaced, kept so the defect has a number attached.
    let render_local = |world_local_xz: [f32; 2], _axis: [f32; 2], tex_scale: f32| -> [f32; 2] {
        let inv = 1.0 / tex_scale.max(0.001);
        [world_local_xz[0] * inv, world_local_xz[1] * inv]
    };

    // One fixed patch of ground, seen from two origins. `grid_axis_viewport.xy`
    // is `(-origin.x, -origin.z)` — asserted in `camera.rs`'s own unit test and
    // relied on by `cloud.wgsl` for the same purpose — so `local - axis` is the
    // absolute world position back.
    const TEX_SCALE: f32 = 4.0;
    let ground_world_xz = [2_600.0f32, -175.0];
    let uv_from = |origin_xz: [f64; 2], f: &dyn Fn([f32; 2], [f32; 2], f32) -> [f32; 2]| {
        let local = [
            ground_world_xz[0] - origin_xz[0] as f32,
            ground_world_xz[1] - origin_xz[1] as f32,
        ];
        let axis = [-origin_xz[0] as f32, -origin_xz[1] as f32];
        f(local, axis, TEX_SCALE)
    };

    // The two origins a rebase moves between. Both are multiples of
    // `ORIGIN_SNAP`, which is what `inf_math::snap` guarantees.
    let before = [2_000.0f64, 0.0];
    let after = [2_000.0 + inf_math::ORIGIN_SNAP, 0.0];
    assert_eq!(
        inf_math::ORIGIN_SNAP,
        10.0,
        "the measurement below is quoted in snap quanta"
    );

    let a = uv_from(before, &world_anchored);
    let b = uv_from(after, &world_anchored);
    assert_eq!(
        a, b,
        "the layer uv moved when the floating origin did: {a:?} → {b:?}. A \
         shared ground material is a property of the ground, not of where the \
         renderer happens to have parked its origin"
    );
    // …and it is the RIGHT uv: absolute world XZ over the tiling.
    assert_eq!(
        a,
        [
            ground_world_xz[0] / TEX_SCALE,
            ground_world_xz[1] / TEX_SCALE
        ],
        "the uv is anchored, but not to the world"
    );

    // THE MEASUREMENT, and the anti-vacuity: the derivation this replaced really
    // does slide, by exactly the number the shader's comment claims.
    let ra = uv_from(before, &render_local);
    let rb = uv_from(after, &render_local);
    let slide = (ra[0] - rb[0]).abs();
    assert!(
        (slide - 2.5).abs() < 1e-4,
        "the render-local uv slid {slide} tiles across one snap quantum, not the \
         2.5 the shader's comment states — the fixture no longer measures the \
         defect it was built for"
    );
    // And a real rebase is not one quantum: it happens when the camera drifts
    // past REBASE_DISTANCE, so the shipped slide would have been this big.
    let far = [2_000.0 + inf_math::REBASE_DISTANCE, 0.0];
    let slide_real = (ra[0] - uv_from(far, &render_local)[0]).abs();
    assert!(
        slide_real > 100.0,
        "a rebase moves the origin by at least REBASE_DISTANCE, so the slide is \
         hundreds of tiles, not {slide_real}"
    );

    // ── the source gate: the shader is the twin ─────────────────────────────
    let body = wgsl_fn(TERRAIN_WGSL, "fn terrain_layers(");
    let code = code_of(&body);
    assert!(
        code.contains("let uv = (world.xz - view.grid_axis_viewport.xy) * inv;"),
        "`terrain_layers` no longer derives its uv from the ABSOLUTE world \
         position. The twin above measures what that costs: 2.5 tiles of visible \
         jump per snap quantum, hundreds per real rebase.\n--- the function ---\n{code}"
    );
    // The ban, by scope rather than by file: the render-local spelling must not
    // come back beside it.
    assert!(
        !code.contains("let uv = world.xz * inv"),
        "the render-local uv is back in `terrain_layers`"
    );
    // **The two vertical projections are anchored on ALL THREE axes** (TER2a).
    // `grid_axis_viewport.xy` is `-origin.xz` and `mode_axis.y` is `-origin.y`,
    // so a `wx` built from only the first two leaves `y` render-local — and `y`
    // is the axis a cliff's material runs along, which is the whole point of the
    // triplanar projections. The slide would be the same `10 / tex_scale` tiles
    // per snap the arm above measures for XZ, on the axis where it shows.
    assert!(
        code.contains("view.mode_axis.y,"),
        "`terrain_layers` builds its triplanar world position without \
         `mode_axis.y`, so the ZY and XY projections slide vertically on \
         every origin rebase\n--- the function ---\n{code}"
    );
    assert!(
        code.contains("let uv_x = wx.zy * inv;") && code.contains("let uv_z = wx.xy * inv;"),
        "the two vertical projections are not derived from the anchored position"
    );
    // …and their derivatives come from ONE position derivative rather than six
    // of their own: measured at +0.252 ms of terrain pass against +0.126 ms for
    // the fetches they were taken for.
    assert!(
        code.contains("let dwx = dpdx(world);") && code.contains("let dwy = dpdy(world);"),
        "the triplanar derivatives are taken per projection again"
    );
    assert!(
        !code.contains("dpdx(uv_x)") && !code.contains("dpdx(uv_z)"),
        "a per-projection derivative is back in `terrain_layers`"
    );

    // …and the call site passes the render-local position the subtraction above
    // expects, rather than something already absolute (which would double it).
    let fs = code_of(&wgsl_fn(TERRAIN_WGSL, "fn fs("));
    assert!(
        fs.contains("terrain_layers(in.world_local, n, w, tex_scale)"),
        "the fragment shader calls `terrain_layers` with something other than \
         `in.world_local`; the subtraction inside it undoes exactly that rebase"
    );
}

/// **The blend is weight-gated and renormalised inside its covered fraction**
/// (Wave-T audit) — the two rules that are the difference between the layered
/// path working and half-working, neither of which had an arm.
#[test]
fn the_layer_blend_gates_on_the_masks_own_quantum_and_does_not_darken() {
    // ── the twin ────────────────────────────────────────────────────────────
    // `terrain_layers`' accumulation, and the call site's `mix`, over four
    // layers of which only the ones flagged `textured` sample a material.
    let blend =
        |weights: [f32; 4], textured: [bool; 4], layer_rgb: [f32; 4], flat: f32| -> (f32, usize) {
            let mut acc = 0.0f32;
            let mut coverage = 0.0f32;
            let mut fetches = 0usize;
            for k in 0..4 {
                let wk = weights[k];
                if wk < 0.0039 {
                    continue;
                }
                if !textured[k] {
                    continue;
                }
                fetches += 1;
                acc += layer_rgb[k] * wk;
                coverage += wk;
            }
            if coverage > 0.0 {
                acc /= coverage;
            }
            // The call site: `mix(albedo, layered.albedo, layered.coverage)`.
            (flat * (1.0 - coverage) + acc * coverage, fetches)
        };

    // (1) **A half-textured terrain does not darken.** Two of four layers carry a
    // material; the other two keep their flat colour. Without the renormalise the
    // textured half would arrive already scaled by its coverage and then be mixed
    // in at that coverage again — the square, i.e. dark.
    let (half, fetches) = blend(
        [0.5, 0.5, 0.0, 0.0],
        [true, false, false, false],
        [0.8; 4],
        0.8,
    );
    assert!(
        (half - 0.8).abs() < 1e-6,
        "a terrain whose textured layer is the same colour as its flat one came \
         out at {half} instead of 0.8 — the blend is scaling by coverage twice"
    );
    assert_eq!(fetches, 1, "only the textured layer may cost a fetch");

    // (2) **The gate is the mask's own quantum.** Splat weights are bytes
    // renormalised to sum to 255, so 1/256 is the smallest weight that exists;
    // below it a layer cannot change a texel of an 8-bit output and must cost no
    // fetch at all.
    let (_, none) = blend([1.0 / 512.0, 0.0, 0.0, 0.0], [true; 4], [1.0; 4], 0.3);
    assert_eq!(none, 0, "a sub-quantum weight still fetched a material");
    let (_, one) = blend([2.0 / 256.0, 0.0, 0.0, 0.0], [true; 4], [1.0; 4], 0.3);
    assert_eq!(one, 1, "a weight above the quantum must be sampled");
    // ANTI-VACUITY: four textured layers at full weight really do cost four.
    let (_, four) = blend([0.25; 4], [true; 4], [1.0; 4], 0.3);
    assert_eq!(four, 4);

    // (3) A wholly untextured fragment is the flat path, untouched — the
    // property every committed terrain golden rests on.
    let (flat, zero) = blend([0.25; 4], [false; 4], [1.0; 4], 0.42);
    assert_eq!((flat, zero), (0.42, 0));

    // ── the source gate: the shader is the twin ─────────────────────────────
    let code = code_of(&wgsl_fn(TERRAIN_WGSL, "fn terrain_layers("));
    assert!(
        code.contains("if (wk < 0.0039) {"),
        "the per-layer weight gate is gone; every fragment now pays four \
         material fetches:\n{code}"
    );
    assert!(
        code.contains("out.albedo = out.albedo / out.coverage;")
            && code.contains("out.roughness = out.roughness / out.coverage;"),
        "the covered-fraction renormalisation is gone — this is how a \
         half-textured terrain goes dark:\n{code}"
    );
    let fs = code_of(&wgsl_fn(TERRAIN_WGSL, "fn fs("));
    assert!(
        fs.contains("mix(albedo, layered.albedo, layered.coverage)"),
        "the call site no longer mixes on the covered fraction, so the \
         renormalisation above has nothing carrying its other half"
    );
    // The whole branch is behind `layered.any`, which is what keeps an
    // untextured terrain instruction-for-instruction what it was.
    assert!(
        fs.contains("if (layered.any) {"),
        "the layered result is applied unconditionally; an untextured terrain is \
         no longer the flat-colour path the goldens pin"
    );
}

/// The scope extractor is not a substring search — asserted, because every claim
/// above rests on it reading the right function.
#[test]
fn the_scope_extractor_reads_a_scope() {
    let body = wgsl_fn(TERRAIN_WGSL, "fn terrain_layers(");
    assert!(body.starts_with("fn terrain_layers("));
    assert!(body.ends_with('}'));
    assert_eq!(
        body.matches('{').count(),
        body.matches('}').count(),
        "the extracted body is unbalanced"
    );
    // It stops at its own closing brace rather than running to the end of file.
    assert!(
        !body.contains("fn vs("),
        "the extractor ran past `terrain_layers` into the next function"
    );
    assert!(
        TERRAIN_WGSL.len() > body.len() * 3,
        "the extractor returned most of the file"
    );
}
