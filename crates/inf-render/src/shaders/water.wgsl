// Water surfaces (P20.1) — one pipeline, three bodies.
//
// The vertex stage maps a unit grid onto whichever footprint the body has
// (a camera-following ocean patch, a lake rectangle, or a spline river ribbon)
// and then displaces it by the Gerstner sum the CPU derived. The fragment stage
// shades it: Fresnel + sky reflection, screen-space refraction, Beer-Lambert
// absorption through the water column, terrain-depth shore blending, and foam
// from wave crests, shallows and flow.
//
// ## Why time is not here
//
// Each wave's phase arrives ALREADY REDUCED: `wrap(φ − ωt + k·(d·origin_xz))`,
// computed in f64 on the CPU. So this shader never sees the level clock and
// never sees a world-space coordinate — it evaluates the sum at render-local
// positions and still gets the world phase. See `crate::water` for why.
//
// ## Depth, two mechanisms — the P17.3 arrangement
//
// The depth attachment is bound READ-ONLY (`depth_ops: None`) and the same view
// is additionally bound as a `texture_depth_multisampled_2d`. The hardware test
// (Greater, reverse-Z, writes off) rejects water behind opaque geometry per MSAA
// sample; the sampled copy gives the fragment the distance to whatever is behind
// it, which is what the absorption, the shore fade and the refraction guard are
// all functions of. Reading it is the ONLY way to know how deep the water is:
// the surface itself carries no thickness.

// Body kinds — these must match `crate::water::WaterKindGpu::code()`.
const WATER_OCEAN: u32 = 0u;
const WATER_LAKE: u32 = 1u;
const WATER_RIVER: u32 = 2u;

// Water's index of refraction is 1.333, so the normal-incidence Fresnel
// reflectance is ((1.333−1)/(1.333+1))² ≈ 0.02. Not a taste constant.
const WATER_F0: f32 = 0.02;

struct Water {
    // x = kind, y = wave count, z = grid vertices per side, w = frame count.
    dims: vec4<u32>,
    // x = frame offset into `water_frames`, y = refraction enabled (0/1),
    // z = SSR enabled (0/1, wave VIS1a), w reserved.
    flags: vec4<u32>,
    // Wave VIS1a, the SSR block: x = march distance (m), y = relative thickness,
    // z = step budget, w = intensity.
    ssr: vec4<f32>,
    // xy = render-local centre (ocean: the snapped camera position; lake: the
    // region centre), zw = half extent in metres.
    center_extent: vec4<f32>,
    // x = render-local still-water Y, y = flow speed (m/s), z = shore fade (m),
    // w = opacity.
    level_flow: vec4<f32>,
    // rgb = shallow colour, a = roughness.
    shallow: vec4<f32>,
    // rgb = deep colour, a = refraction offset (m).
    deep: vec4<f32>,
    // rgb = Beer-Lambert extinction (m⁻¹), a = reference depth (m) used where
    // the depth buffer has nothing behind the surface.
    absorption: vec4<f32>,
    // rgb = foam colour, a = crest threshold.
    foam: vec4<f32>,
    // x = foam shore depth (m), y = foam flow speed (m/s), zw reserved.
    foam2: vec4<f32>,
    // Authored sky gradient (zenith / horizon / ground) — the reflection source
    // when the physical atmosphere is off.
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
    // 8 Gerstner components, two vec4 each:
    //   [2i]   = (dir.x, dir.y, amplitude, wavenumber)
    //   [2i+1] = (steepness Q, reduced phase, 0, 0)
    waves: array<vec4<f32>, 16>,
};

struct WaterFrame {
    // xyz = render-local centre, w = full width (m).
    center_width: vec4<f32>,
    // xyz = unit tangent (flow), w = depth to the bed (m).
    tangent_depth: vec4<f32>,
    // xyz = unit horizontal across-vector, w = arc length (m).
    right_s: vec4<f32>,
    // x = P19.1 flow-map foam gain (>= 1; 1 = no flow map), yzw reserved.
    flow_extra: vec4<f32>,
};

@group(1) @binding(4) var<uniform> water: Water;
@group(1) @binding(5) var<storage, read> water_frames: array<WaterFrame>;
@group(1) @binding(6) var water_scene_depth: texture_depth_multisampled_2d;
@group(1) @binding(7) var water_scene_color: texture_2d<f32>;
@group(1) @binding(8) var water_scene_smp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Render-local displaced surface position.
    @location(0) world: vec3<f32>,
    // The wave-space parameter this vertex was evaluated at (world XZ for an
    // ocean/lake, (arc length, lateral) for a river) — the fragment re-evaluates
    // the normal here, so it must be the PARAMETER point, not the displaced one.
    @location(1) param: vec2<f32>,
    // x = depth to the bed (m; rivers), y = |lateral| / (width/2) for rivers and
    // 0 elsewhere, z = the river frame's flow-map foam gain (1 elsewhere, and on
    // a river over terrain with no P19.1 bake).
    @location(2) profile: vec3<f32>,
    // The river frame's tangent/across basis, so the fragment can rotate a
    // river-local normal into the world. Identity (1,0,0)/(0,0,1) elsewhere.
    @location(3) along: vec3<f32>,
    @location(4) across: vec3<f32>,
};

// `sign(q) · q²` on [-1,1]: a grid that is dense near the camera and coarse at
// the far ring. A uniform 8 km grid at 64 quads has 125 m cells, which is three
// wavelengths of authored swell per cell — the waves would simply not exist.
// This puts the innermost cell at ~8 m and the outermost at ~250 m.
fn water_grade(q: f32) -> f32 {
    return sign(q) * q * q;
}

// The Gerstner displacement at parameter `p`: xz = horizontal crowding toward
// the crests, y = vertical.
fn water_displace(p: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(0.0, 0.0, 0.0);
    let n = water.dims.y;
    for (var i = 0u; i < n; i = i + 1u) {
        let a = water.waves[2u * i];
        let b = water.waves[2u * i + 1u];
        let dir = a.xy;
        let amp = a.z;
        let k = a.w;
        let theta = k * dot(dir, p) + b.y;
        let s = sin(theta);
        let c = cos(theta);
        let qa = b.x * amp;
        d.x = d.x + qa * dir.x * c;
        d.z = d.z + qa * dir.y * c;
        d.y = d.y + amp * s;
    }
    return d;
}

// The analytic Gerstner normal (GPU Gems 1, ch. 1) in the wave's own frame:
// x = along the first basis axis, z = along the second, y = up.
fn water_normal(p: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(0.0, 1.0, 0.0);
    let cnt = water.dims.y;
    for (var i = 0u; i < cnt; i = i + 1u) {
        let a = water.waves[2u * i];
        let b = water.waves[2u * i + 1u];
        let theta = a.w * dot(a.xy, p) + b.y;
        let wa = a.w * a.z;
        n.x = n.x - a.xy.x * wa * cos(theta);
        n.z = n.z - a.xy.y * wa * cos(theta);
        n.y = n.y - b.x * wa * sin(theta);
    }
    return normalize(n);
}

// The surface-folding measure the Gerstner model already contains: at 1 the
// trochoid cusps and a real wave would be breaking. This is where foam belongs,
// rather than at a hand-picked height.
fn water_crest(p: vec2<f32>) -> f32 {
    var f = 0.0;
    let cnt = water.dims.y;
    for (var i = 0u; i < cnt; i = i + 1u) {
        let a = water.waves[2u * i];
        let b = water.waves[2u * i + 1u];
        let theta = a.w * dot(a.xy, p) + b.y;
        f = f + b.x * a.z * a.w * sin(theta);
    }
    return clamp(f, 0.0, 1.0);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let n = water.dims.z;
    let ix = vi % n;
    let iz = vi / n;
    let u = f32(ix) / f32(n - 1u);
    let v = f32(iz) / f32(n - 1u);

    var base: vec3<f32>;
    var param: vec2<f32>;
    var depth_m = water.absorption.w;
    var bank = 0.0;
    // The flow-map foam gain (P20.4). 1 everywhere but a river, and 1 on a river
    // whose terrain was never eroded — so this term is the exact identity for
    // every scene authored before it existed.
    var flow_gain = 1.0;
    var along = vec3<f32>(1.0, 0.0, 0.0);
    var across = vec3<f32>(0.0, 0.0, 1.0);

    let kind = water.dims.x;
    if (kind == WATER_RIVER) {
        // Map u along the ribbon, v across it, interpolating the CPU-built
        // centreline frames. Even arc-length spacing is what makes `u` a
        // *distance* rather than a spline parameter.
        let cnt = max(water.dims.w, 2u);
        let f = u * f32(cnt - 1u);
        let i0 = min(u32(floor(f)), cnt - 1u);
        let i1 = min(i0 + 1u, cnt - 1u);
        let t = f - floor(f);
        let off = water.flags.x;
        let a = water_frames[off + i0];
        let b = water_frames[off + i1];
        let center = mix(a.center_width.xyz, b.center_width.xyz, t);
        let width = mix(a.center_width.w, b.center_width.w, t);
        let right = normalize(mix(a.right_s.xyz, b.right_s.xyz, t));
        let tangent = normalize(mix(a.tangent_depth.xyz, b.tangent_depth.xyz, t));
        let s = mix(a.right_s.w, b.right_s.w, t);
        let lateral = (v - 0.5) * width;
        base = center + right * lateral;
        param = vec2<f32>(s, lateral);
        depth_m = mix(a.tangent_depth.w, b.tangent_depth.w, t);
        bank = select(0.0, abs(lateral) / (0.5 * width), width > 0.0);
        // Interpolated exactly like the width and the depth: the gain is a
        // per-frame property of the ribbon, not a per-body constant.
        flow_gain = max(mix(a.flow_extra.x, b.flow_extra.x, t), 1.0);
        // The river's wave frame: +param.x runs downstream, +param.y across.
        along = vec3<f32>(tangent.x, 0.0, tangent.z);
        let along_len = length(along);
        along = select(vec3<f32>(1.0, 0.0, 0.0), along / max(along_len, 1e-6), along_len > 1e-6);
        across = vec3<f32>(-along.z, 0.0, along.x);
    } else {
        // Ocean: a graded, camera-snapped patch. Lake: a uniform rectangle.
        var qx = u * 2.0 - 1.0;
        var qz = v * 2.0 - 1.0;
        if (kind == WATER_OCEAN) {
            qx = water_grade(qx);
            qz = water_grade(qz);
        }
        base = vec3<f32>(
            water.center_extent.x + qx * water.center_extent.z,
            water.level_flow.x,
            water.center_extent.y + qz * water.center_extent.w,
        );
        // The wave parameter is the RENDER-LOCAL xz: the floating origin is
        // already folded into each wave's reduced phase.
        param = base.xz;
    }

    let d = water_displace(param);
    // A river's crowding is expressed in its own frame; an ocean's is already
    // world-aligned.
    let disp = select(
        vec3<f32>(d.x, d.y, d.z),
        along * d.x + vec3<f32>(0.0, d.y, 0.0) + across * d.z,
        kind == WATER_RIVER,
    );
    let world = base + disp;

    out.world = world;
    out.param = param;
    out.profile = vec3<f32>(depth_m, bank, flow_gain);
    out.along = along;
    out.across = across;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    return out;
}

// The sky half of the reflection. The physical sky-view LUT when the atmosphere
// is on, the authored three-colour gradient otherwise.
fn water_sky_reflection(dir: vec3<f32>) -> vec3<f32> {
    if (atmos.params.x >= 0.5) {
        return atmos_sample_skyview(atmos.planet.z, normalize(dir));
    }
    let t = dir.y;
    if (t >= 0.0) {
        return mix(water.horizon.rgb, water.zenith.rgb, pow(clamp(t, 0.0, 1.0), 0.55));
    }
    return mix(water.horizon.rgb, water.ground.rgb, clamp(-t * 3.0, 0.0, 1.0));
}

// **The boat finally reflects itself** (wave VIS1a) — the P20.3 routed item.
//
// P20.1's ruling was: *"Not SSR, deliberately. A wave-perturbed normal at the
// grazing angles that dominate a water surface reflects toward the horizon, which
// is exactly where a screen-space march has nothing to hit: the ray leaves the
// frame within a few steps and every pixel takes the miss path."* Every sentence
// of that is still true, and it is **an argument about the grazing angles, not
// about the surface**. What it justified — asking the sky directly and never
// marching — was right while there was no way to tell the two regimes apart.
//
// There is now, and it is the term the shader already computes: `cos_v`, the
// cosine between the surface normal and the eye. A grazing pixel has `cos_v`
// near zero and is exactly the pixel the ruling describes; a pixel looking down
// into the water — the one under the hull, the one against the jetty, the one
// that has a cliff standing in it — has `cos_v` well away from zero and is the
// pixel the ruling never had a way to serve. `water_grazing_weight` is that
// split, and it is a **fade**, not a threshold, so there is no line across the
// surface where the behaviour changes.
//
// The two other halves of the original argument are answered by construction:
//
// * **the miss path is still the sky.** A march that leaves the frame, finds
//   nothing, or lands off the previous colour returns zero confidence and the
//   sky term is what remains. Nothing goes black.
// * **there is no extra resolve.** Water already runs a private
//   `color_msaa → scene_hdr` resolve for refraction (`passes/water.rs`), so the
//   opaque scene colour of THIS frame is already bound at `water_scene_color`
//   and the depth of this frame at `water_scene_depth`. Unlike the opaque SSR in
//   `env_lighting.wgsl`, this one has no frame of latency at all — which is the
//   whole reason the water pass is where colour-sourced reflection lands first.
//   `flags.z` is therefore gated on refraction as well: a tier that does not pay
//   for the resolve has nothing to reflect.
fn water_grazing_weight(cos_v: f32) -> f32 {
    // Nothing below ~3° above the surface (cos_v < 0.05); full by ~14° (0.25).
    //
    // **The band is low on purpose, and the numbers are the argument.** A boat is
    // reflected at ordinary viewing angles — an eye eight metres up looking twenty
    // metres out reads cos_v ≈ 0.3, and that is the shot the ruling was denying.
    // What P20.1 describes is the *extreme*: an eye a few centimetres above the
    // waterline, where cos_v is hundredths and the reflected ray runs at the
    // horizon. Two things then hold it off, and the belt is the cheaper of them:
    // this fade returns 0 before a single tap is spent, and even without it the
    // march would leave the frame and return no confidence. Setting the band high
    // enough to cover "grazing" loosely would have cost the shot the wave exists
    // to get.
    return smoothstep(0.05, 0.25, cos_v);
}

// March the reflection ray against this frame's opaque depth and return
// `vec4(colour, confidence)`. Confidence 0 is a miss.
//
// `in_pos` is the fragment's `@builtin(position)`, so the march starts at the
// water surface as the depth buffer sees it.
fn water_ssr(world: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let size = view.grid_axis_viewport.zw;
    let max_dist = water.ssr.x;
    let thickness = water.ssr.y;
    let steps = i32(max(water.ssr.z, 1.0));
    let inv = 1.0 / f32(steps);
    var prev_t = 0.0;
    for (var s = 1; s <= steps; s = s + 1) {
        let u = f32(s) * inv;
        // The same quadratic march `env_lighting.wgsl` uses: dense where a
        // contact reflection lives, coarse where a miss is cheap.
        let t = max_dist * u * u;
        let p = world + dir * t;
        let clip = view.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            return vec4<f32>(0.0);
        }
        let ndc = clip.xyz / clip.w;
        if (any(abs(ndc.xy) > vec2<f32>(1.0)) || ndc.z <= 0.0) {
            return vec4<f32>(0.0);
        }
        let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
        let texel = vec2<i32>(clamp(uv * size, vec2<f32>(0.0), size - vec2<f32>(1.0)));
        let scene_z = textureLoad(water_scene_depth, texel, 0);
        if (scene_z <= 0.0) {
            prev_t = t;
            continue; // sky behind this texel — keep going
        }
        let penetration = 1.0 - ndc.z / scene_z;
        if (penetration > 0.0 && penetration < thickness) {
            var lo = prev_t;
            var hi = t;
            for (var b = 0; b < 3; b = b + 1) {
                let mid = 0.5 * (lo + hi);
                let q = world + dir * mid;
                let qc = view.view_proj * vec4<f32>(q, 1.0);
                let qn = qc.xyz / max(qc.w, 1e-6);
                let quv = qn.xy * vec2<f32>(0.5, -0.5) + 0.5;
                let qt = vec2<i32>(clamp(quv * size, vec2<f32>(0.0), size - vec2<f32>(1.0)));
                let qz = textureLoad(water_scene_depth, qt, 0);
                if (qz > 0.0 && qn.z < qz) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let hc = view.view_proj * vec4<f32>(world + dir * hi, 1.0);
            let hn = hc.xyz / max(hc.w, 1e-6);
            let huv = clamp(hn.xy * vec2<f32>(0.5, -0.5) + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
            // Dissolve over the outer 8 % of the frame, and over the last quarter
            // of the ray, for `env_lighting.wgsl`'s reasons.
            let e = min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y));
            let edge = smoothstep(0.0, 0.08, e);
            let far = 1.0 - smoothstep(0.75, 1.0, hi / max(max_dist, 1e-4));
            let c = textureSampleLevel(water_scene_color, water_scene_smp, huv, 0.0).rgb;
            return vec4<f32>(c, edge * far);
        }
        prev_t = t;
    }
    return vec4<f32>(0.0);
}

// The reflection a water fragment sees: the sky, with whatever the scene puts in
// front of it blended over at non-grazing angles.
fn water_reflection(world: vec3<f32>, dir: vec3<f32>, cos_v: f32) -> vec3<f32> {
    let sky = water_sky_reflection(dir);
    // `flags.z` is 0 on every scene with SSR off, so this branch is untaken and
    // every committed water golden runs exactly the arithmetic it always did.
    if (water.flags.z == 1u) {
        let w = water_grazing_weight(cos_v) * water.ssr.w;
        if (w > 0.0) {
            // Start a little above the surface: the march is against the OPAQUE
            // depth, and a ray that begins exactly on the water plane over a
            // shallow bottom reads its own floor as a hit at step one.
            let hit = water_ssr(world + dir * 0.05, dir);
            if (hit.w > 0.0) {
                return mix(sky, hit.rgb, clamp(w * hit.w, 0.0, 1.0));
            }
        }
    }
    return sky;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let eye = view.eye.xyz;
    let to_eye = eye - in.world;
    let dist = length(to_eye);
    let vdir = to_eye / max(dist, 1e-6);

    // Surface normal: evaluated in the wave frame, then rotated onto the body's
    // basis (identity for an ocean/lake).
    let ln = water_normal(in.param);
    var n = normalize(in.along * ln.x + vec3<f32>(0.0, ln.y, 0.0) + in.across * ln.z);
    if (dot(n, vdir) < 0.0) {
        // A back-facing normal at a grazing crest would flip the Fresnel term
        // and make the wave black. Fold it rather than discard.
        n = normalize(n - 2.0 * dot(n, vdir) * vdir);
    }

    // ── the water column behind this fragment ────────────────────────────
    let texel = vec2<i32>(in.pos.xy);
    let scene_d = textureLoad(water_scene_depth, texel, 0);
    var column = water.absorption.w;
    var has_floor = false;
    if (scene_d > 0.0) {
        // Unproject the opaque hit and measure along the view ray. Both points
        // are render-local, so this is a plain metric distance.
        let ndc = vec2<f32>(
            (in.pos.x / view.grid_axis_viewport.z) * 2.0 - 1.0,
            1.0 - (in.pos.y / view.grid_axis_viewport.w) * 2.0,
        );
        let hit = unproject(ndc, scene_d);
        column = max(length(hit - in.world), 0.0);
        has_floor = true;
    }

    // Shore blend: the surface fades out where the ground comes up to meet it.
    // A screen-space depth difference, so it works against terrain, a jetty and
    // a boat hull with one expression. The CPU twin is `inf_water::shore_blend`
    // and both use the same smoothstep.
    let fade = max(water.level_flow.z, 1e-4);
    let shore = select(1.0, smoothstep(0.0, fade, column), has_floor);

    // ── absorption (Beer-Lambert, the P17 fog math) ──────────────────────
    let transmit = exp(-water.absorption.rgb * column);
    let body = mix(water.deep.rgb, water.shallow.rgb, transmit);

    // ── refraction ───────────────────────────────────────────────────────
    var behind = body;
    if (water.flags.y == 1u) {
        let size = view.grid_axis_viewport.zw;
        let base_uv = in.pos.xy / size;
        // Offset by the surface normal's horizontal component, scaled by the
        // authored displacement and damped by distance so a far wave does not
        // smear the whole background.
        let scale = water.deep.w / max(dist * 0.05, 1.0);
        let uv = clamp(base_uv + n.xz * scale, vec2<f32>(0.0), vec2<f32>(1.0));
        // The classic artefact guard: if the offset lands on geometry NEARER
        // than the water, we would be refracting something in front of the
        // surface. Fall back to the unperturbed sample.
        let od = textureLoad(water_scene_depth, vec2<i32>(uv * size), 0);
        let ok = (od > 0.0) && (od <= in.pos.z);
        let pick = select(base_uv, uv, ok);
        behind = textureSampleLevel(water_scene_color, water_scene_smp, pick, 0.0).rgb * transmit
            + water.deep.rgb * (1.0 - transmit);
    }

    // ── reflection + Fresnel ─────────────────────────────────────────────
    let cos_v = clamp(dot(n, vdir), 0.0, 1.0);
    let fresnel = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - cos_v, 5.0);
    let refl_dir = reflect(-vdir, n);
    let sky = water_reflection(in.world, refl_dir, cos_v);

    // Sun specular: a GGX-ish lobe about the reflected direction. Water is very
    // smooth, so this is a small, bright highlight rather than a broad sheen.
    let rough = clamp(water.shallow.w, 0.02, 1.0);
    let a2 = rough * rough * rough * rough;
    let h = normalize(vdir + view.sun_dir.xyz);
    let ndh = max(dot(n, h), 0.0);
    let dgg = a2 / max(3.14159265 * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0), 1e-6);
    let sun_spec = dgg * max(dot(n, view.sun_dir.xyz), 0.0);

    var color = mix(behind, sky, fresnel) + sky * sun_spec * 0.15;

    // ── foam ─────────────────────────────────────────────────────────────
    // Three sources, combined by max rather than added: foam is a coverage
    // fraction, and two overlapping causes do not make more than full white.
    let crest = water_crest(in.param);
    let crest_foam = smoothstep(water.foam.w, min(water.foam.w + 0.2, 1.0), crest);
    let shore_band = select(
        0.0,
        1.0 - smoothstep(0.0, max(water.foam2.x, 1e-4), column),
        has_floor && water.foam2.x > 0.0,
    );
    // P20.4: the P19.1 flow map modulates the SPEED the authored threshold sees,
    // so a river running down a channel the erosion pass carved foams harder than
    // the same river across an unmapped plain. `in.profile.z` is 1 wherever there
    // is no map, which makes this expression bit-identical to the P20.1 one for
    // every scene that has not been eroded.
    let flow_foam = select(
        0.0,
        smoothstep(0.35, 1.0, abs(water.level_flow.y) * in.profile.z / max(water.foam2.y, 1e-4)),
        water.foam2.y > 0.0,
    );
    // A river also foams against its banks, where the flow drags on the bed.
    let bank_foam = select(0.0, smoothstep(0.75, 1.0, in.profile.y) * flow_foam, flow_foam > 0.0);
    let foam = clamp(max(max(crest_foam, shore_band), max(flow_foam * 0.5, bank_foam)), 0.0, 1.0);
    color = mix(color, water.foam.rgb, foam);

    // ── aerial perspective ───────────────────────────────────────────────
    color = atmos_apply(color, in.world);

    // Foam is opaque; the shore fade and the authored opacity govern the rest.
    let alpha = clamp(max(shore * water.level_flow.w, foam * shore), 0.0, 1.0);
    return vec4<f32>(color, alpha);
}
