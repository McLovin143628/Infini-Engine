// Cloud noise generators (P17.3) — the GPU mirror of `crate::clouds`.
//
// Composed into every cloud consumer by `passes::cloud_noise_source()`, which
// takes no tokens because NOTHING HERE BINDS ANYTHING. That is deliberate: the
// bake writes the volumes as *storage* textures while the march reads them as
// *sampled* textures, so a single file that declared them could not serve both.
// The bindings + the density function live in `cloud_field.wgsl`.
//
// UNITS — SI metres and m^-1 throughout, matching `crate::clouds`'s module docs.
// (The atmosphere's kilometres appear only where a cloud radiance meets the
// aerial-perspective term, in `cloud.wgsl`.)
//
// DETERMINISM — every lattice value comes from `cloud_hash`, a pure-integer
// avalanche. There is no trigonometry anywhere in this file: the house law is
// that f32 trig is not bit-portable, and a cloud field that shifted between a
// golden bless and a CI re-render would be a permanently flaky gate. What float
// arithmetic there is (the fades and lerps) is written in the same order as the
// Rust, so the two agree to within the FMA-contraction envelope the parity gate
// documents (`clouds::CPU_GPU_TEXEL_TOLERANCE`).

const CLOUD_PI: f32 = 3.14159265359;

// Mirrors `clouds::SHAPE_TILE_M` / `DETAIL_TILE_M` / `WEATHER_TILE_M`.
const CLOUD_SHAPE_TILE_M: f32 = 8192.0;
const CLOUD_DETAIL_TILE_M: f32 = 256.0;
// Mirrors `clouds::DETAIL_CURL_M` / `DETAIL_COARSE_SCALE` /
// `DETAIL_COARSE_WEIGHT`.
const CLOUD_DETAIL_CURL_M: f32 = 60.0;
const CLOUD_DETAIL_COARSE_SCALE: f32 = 4.0;
const CLOUD_DETAIL_COARSE_WEIGHT: f32 = 0.35;
const CLOUD_WEATHER_TILE_M: f32 = 40960.0;
// Mirrors `clouds::SHAPE_PERLIN_PERIOD` / `SHAPE_WORLEY_CELLS` /
// `DETAIL_WORLEY_CELLS` / `WEATHER_PERIOD`.
const CLOUD_SHAPE_PERLIN_PERIOD: i32 = 4;
const CLOUD_SHAPE_WORLEY_CELLS: i32 = 4;
const CLOUD_DETAIL_WORLEY_CELLS: i32 = 2;
const CLOUD_WEATHER_PERIOD: i32 = 8;
// Mirrors `clouds::WEATHER_CONVECTION_OCTAVE` / `CLOUD_BASE_LIFT` /
// `CLOUD_TOP_WEAK`.
const CLOUD_WEATHER_CONVECTION: i32 = 4;
const CLOUD_BASE_LIFT: f32 = 0.06;
const CLOUD_TOP_WEAK: f32 = 0.62;
// Mirrors `clouds::WEATHER_CONTRAST` / `COVERAGE_BIAS_SLOPE` / `COVERAGE_BIAS_OFFSET`.
const CLOUD_WEATHER_CONTRAST: f32 = 3.0;
const CLOUD_COVERAGE_SLOPE: f32 = 2.4;
const CLOUD_COVERAGE_OFFSET: f32 = 1.4;
// Mirrors `clouds::BACK_LOBE_G` / `FORWARD_LOBE_WEIGHT`.
const CLOUD_BACK_LOBE_G: f32 = -0.45;
const CLOUD_FORWARD_LOBE_WEIGHT: f32 = 0.6;
// Mirrors `clouds::MAX_SHADOW_MARCH_M`.
const CLOUD_MAX_SHADOW_MARCH_M: f32 = 20000.0;

// CPU mirror: `clouds::cloud_hash`.
fn cloud_hash(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    var h = x * 0x8da6b343u + y * 0xd8163841u + z * 0xcb1ab31fu + seed * 0x16521623u;
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return h;
}

// CPU mirror: `clouds::hash_unit`. 24 bits, so every result is exactly
// representable in f32 and both sides agree bit for bit.
fn cloud_hash_unit(h: u32) -> f32 {
    return f32(h & 0x00ffffffu) / 16777216.0;
}

// Euclidean remainder. WGSL's `%` truncates toward zero (like Rust's), so a
// negative lattice index would land outside the wrap without this.
fn cloud_wrap(a: i32, p: i32) -> u32 {
    let m = a % p;
    return u32(select(m + p, m, m >= 0));
}

// CPU mirror: `clouds::fade`. Same Horner order.
fn cloud_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn cloud_lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Perlin's improved-noise gradient: 12 cube-edge directions, so the dot product
// is a pair of adds and nothing can round differently across adapters.
// CPU mirror: `clouds::grad3`.
fn cloud_grad3(h: u32, x: f32, y: f32, z: f32) -> f32 {
    let hh = h & 15u;
    var u = y;
    if (hh < 8u) { u = x; }
    var v = z;
    if (hh < 4u) {
        v = y;
    } else if (hh == 12u || hh == 14u) {
        v = x;
    }
    var a = u;
    if ((hh & 1u) != 0u) { a = -u; }
    var b = v;
    if ((hh & 2u) != 0u) { b = -v; }
    return a + b;
}

// Tileable 3D Perlin gradient noise, roughly [-1, 1].
// CPU mirror: `clouds::perlin3_tiled`.
fn cloud_perlin3(p: vec3<f32>, period: i32, seed: u32) -> f32 {
    let per = max(period, 1);
    let fl = floor(p);
    let t = p - fl;
    let i = vec3<i32>(fl);
    let u = vec3<f32>(cloud_fade(t.x), cloud_fade(t.y), cloud_fade(t.z));

    var c: array<f32, 8>;
    for (var k = 0; k < 8; k = k + 1) {
        let d = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let h = cloud_hash(
            cloud_wrap(i.x + d.x, per),
            cloud_wrap(i.y + d.y, per),
            cloud_wrap(i.z + d.z, per),
            seed,
        );
        c[k] = cloud_grad3(h, t.x - f32(d.x), t.y - f32(d.y), t.z - f32(d.z));
    }
    let x00 = cloud_lerp(c[0], c[1], u.x);
    let x10 = cloud_lerp(c[2], c[3], u.x);
    let x01 = cloud_lerp(c[4], c[5], u.x);
    let x11 = cloud_lerp(c[6], c[7], u.x);
    let y0 = cloud_lerp(x00, x10, u.y);
    let y1 = cloud_lerp(x01, x11, u.y);
    return cloud_lerp(y0, y1, u.z);
}

// Tileable 3D Worley (cellular) noise in [0, 1], INVERTED so 1 is a cell centre —
// which is what gives a cumulus its cauliflower silhouette.
// CPU mirror: `clouds::worley3_tiled`.
fn cloud_worley3(p: vec3<f32>, cells: i32, seed: u32) -> f32 {
    let n = max(cells, 1);
    let g = p * f32(n);
    let base = floor(g);
    let f = g - base;
    let bi = vec3<i32>(base);

    var best = 1.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let h = cloud_hash(
                    cloud_wrap(bi.x + dx, n),
                    cloud_wrap(bi.y + dy, n),
                    cloud_wrap(bi.z + dz, n),
                    seed ^ 0x9e3779b9u,
                );
                let fp = vec3<f32>(
                    f32(dx) + cloud_hash_unit(h),
                    f32(dy) + cloud_hash_unit(cloud_hash(h, 1u, 0u, 0u)),
                    f32(dz) + cloud_hash_unit(cloud_hash(h, 2u, 0u, 0u)),
                );
                let d = fp - f;
                let sq = d.x * d.x + d.y * d.y + d.z * d.z;
                best = min(best, sq);
            }
        }
    }
    return 1.0 - min(sqrt(best), 1.0);
}

// Three Worley octaves at the Guerrilla weights. CPU mirror: `clouds::worley_fbm`.
fn cloud_worley_fbm(p: vec3<f32>, cells: i32, seed: u32) -> f32 {
    return cloud_worley3(p, cells, seed) * 0.625
        + cloud_worley3(p, cells * 2, seed + 1u) * 0.25
        + cloud_worley3(p, cells * 4, seed + 2u) * 0.125;
}

// [lo, hi] -> [0, 1], clamped. CPU mirror: `clouds::remap`.
fn cloud_remap(v: f32, lo: f32, hi: f32) -> f32 {
    return clamp((v - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
}

// The RGBA of a SHAPE volume texel at normalized centre `p`.
// CPU mirror: `clouds::shape_texel` (before quantization).
fn cloud_shape_value(p: vec3<f32>, seed: u32) -> vec4<f32> {
    var perlin = 0.0;
    var amp = 1.0;
    var per = CLOUD_SHAPE_PERLIN_PERIOD;
    var norm = 0.0;
    for (var o = 0u; o < 3u; o = o + 1u) {
        perlin = perlin + amp * cloud_perlin3(p * f32(per), per, seed + o * 101u);
        norm = norm + amp;
        amp = amp * 0.5;
        per = per * 2;
    }
    let pn = clamp(perlin / norm * 0.5 + 0.5, 0.0, 1.0);

    let w0 = cloud_worley_fbm(p, CLOUD_SHAPE_WORLEY_CELLS, seed + 11u);
    // The Perlin-Worley remap (Schneider/Guerrilla): dissolve the Perlin field by
    // the inverted Worley fBm, keeping Perlin's connected topology and Worley's
    // rounded billows.
    let pw = cloud_remap(pn, w0 - 1.0, 1.0);

    // Single octaves, not fBm: three fBms would each reach 4x their base
    // frequency, putting the alpha channel at 128 cells — far past what even a
    // 128^3 volume can store, so what got baked would be aliasing rather than
    // detail. CPU mirror: `clouds::shape_value`.
    return vec4<f32>(
        pw,
        cloud_worley3(p, CLOUD_SHAPE_WORLEY_CELLS * 2, seed + 23u),
        cloud_worley3(p, CLOUD_SHAPE_WORLEY_CELLS * 4, seed + 37u),
        cloud_worley3(p, CLOUD_SHAPE_WORLEY_CELLS * 8, seed + 53u),
    );
}

// The RGBA of a DETAIL (erosion) volume texel. Alpha is pinned to 1 so a captured
// volume is visible in a debugger rather than fully transparent.
// CPU mirror: `clouds::detail_texel`.
fn cloud_detail_value(p: vec3<f32>, seed: u32) -> vec4<f32> {
    return vec4<f32>(
        cloud_worley3(p, CLOUD_DETAIL_WORLEY_CELLS, seed + 71u),
        cloud_worley3(p, CLOUD_DETAIL_WORLEY_CELLS * 2, seed + 83u),
        cloud_worley3(p, CLOUD_DETAIL_WORLEY_CELLS * 4, seed + 97u),
        1.0,
    );
}

// ── the weather (2D coverage / type) field ─────────────────────────────────

fn cloud_value2(x: f32, z: f32, period: i32, seed: u32) -> f32 {
    let per = max(period, 1);
    let fx = floor(x);
    let fz = floor(z);
    let tx = x - fx;
    let tz = z - fz;
    let ix = i32(fx);
    let iz = i32(fz);
    var v: array<f32, 4>;
    for (var k = 0; k < 4; k = k + 1) {
        let dx = k & 1;
        let dz = (k >> 1) & 1;
        v[k] = cloud_hash_unit(cloud_hash(cloud_wrap(ix + dx, per), 0u, cloud_wrap(iz + dz, per), seed));
    }
    let u = cloud_fade(tx);
    let w = cloud_fade(tz);
    return cloud_lerp(cloud_lerp(v[0], v[1], u), cloud_lerp(v[2], v[3], u), w);
}

// The weather at world (x, z) metres: (coverage, type, convection), all [0, 1].
// Analytic rather than baked — see the rationale on `clouds::weather`.
// CPU mirror: `clouds::weather`.
fn cloud_weather(x_m: f32, z_m: f32, seed: u32, coverage: f32, cloud_type: f32) -> vec3<f32> {
    let s = f32(CLOUD_WEATHER_PERIOD) / CLOUD_WEATHER_TILE_M;
    let u = x_m * s;
    let v = z_m * s;
    let c = cloud_value2(u, v, CLOUD_WEATHER_PERIOD, seed + 211u) * 0.65
        + cloud_value2(u * 3.0, v * 3.0, CLOUD_WEATHER_PERIOD * 3, seed + 223u) * 0.35;
    let t = cloud_value2(u * 2.0, v * 2.0, CLOUD_WEATHER_PERIOD * 2, seed + 233u);
    // The convection octave (SKY2) — per-cloud rather than per-region, which is
    // what lets neighbouring cells build to different heights.
    let n = f32(CLOUD_WEATHER_CONVECTION);
    let k = cloud_value2(
        u * n,
        v * n,
        CLOUD_WEATHER_PERIOD * CLOUD_WEATHER_CONVECTION,
        seed + 241u,
    );

    // Widen the raw field before biasing it: two octaves of interpolated hash pile
    // up around 0.5, and a narrow field means the authored slider crosses the
    // density threshold in a tenth of its travel. CPU mirror: `clouds::weather`.
    let cw = clamp((c - 0.5) * CLOUD_WEATHER_CONTRAST + 0.5, 0.0, 1.0);

    let cov0 = clamp(coverage, 0.0, 1.0);
    // Bias, not multiply: 0 is genuinely cloudless and 1 genuinely solid. The
    // slope/offset are calibrated against realised sky cover — see
    // `clouds::weather`.
    let cov = clamp(cw + (cov0 * CLOUD_COVERAGE_SLOPE - CLOUD_COVERAGE_OFFSET), 0.0, 1.0);
    let ty = clamp(clamp(cloud_type, 0.0, 1.0) + (t - 0.5) * 0.5, 0.0, 1.0);
    return vec3<f32>(cov, ty, k);
}

// Vertical density profile at relative height h within the slab, for a cloud of
// type t (0 = stratus sheet, 1 = cumulus tower) and local convective strength k.
//
// v2 (SKY2). The v1 form held a cumulus at full strength from 0.22 to 0.60 and
// only tapered over the last two fifths — and because `grad` multiplies the
// shape BEFORE the coverage dissolve, a flat grad means the same points survive
// at every height in that band, which is a slab rather than a cloud. Now the
// taper is continuous to a per-cell ceiling, so the column narrows into a tower,
// and neighbouring cells reach different heights.
// CPU mirror: `clouds::height_gradient`.
fn cloud_height_gradient(h_in: f32, t_in: f32, k_in: f32) -> f32 {
    let h = clamp(h_in, 0.0, 1.0);
    let t = clamp(t_in, 0.0, 1.0);
    let k = clamp(k_in, 0.0, 1.0);
    // This cloud's own floor. Only a cumulus lifts; a sheet keeps the slab's.
    let floor = k * CLOUD_BASE_LIFT * t;
    let hl = clamp((h - floor) / max(1.0 - floor, 1e-3), 0.0, 1.0);
    let stratus = smoothstep(0.0, 0.08, hl) * (1.0 - smoothstep(0.20, 0.40, hl));
    // At full convection and full type this is the v1 CURVE — same 0.02 -> 0.22
    // onset, same 0.6 -> 1.0 taper — read at `hl` rather than at `h`, because the
    // base lift is maximal in the same corner. `the_strong_cell_is_v1_through_the_base_lift`
    // pins that relation. The variation is scaled by `t` so a sheet-like deck
    // keeps a system-wide ceiling.
    let top = 1.0 + (CLOUD_TOP_WEAK + (1.0 - CLOUD_TOP_WEAK) * k - 1.0) * t;
    let cumulus = smoothstep(0.02, 0.22, hl) * (1.0 - smoothstep(top * 0.6, top, hl));
    return stratus + (cumulus - stratus) * t;
}

fn cloud_hg(g: f32, cos_t: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * CLOUD_PI * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5));
}

// Two-lobe Henyey-Greenstein. One lobe makes a cloud look like plastic: real
// droplets scatter strongly forward AND appreciably back, which is where the
// silver lining comes from. CPU mirror: `clouds::phase`.
fn cloud_phase(cos_t: f32, g_in: f32) -> f32 {
    let g = clamp(g_in, 0.0, 0.95);
    return cloud_hg(g, cos_t) * CLOUD_FORWARD_LOBE_WEIGHT
        + cloud_hg(CLOUD_BACK_LOBE_G * min(g / 0.8, 1.0), cos_t) * (1.0 - CLOUD_FORWARD_LOBE_WEIGHT);
}
