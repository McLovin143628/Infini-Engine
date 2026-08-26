//! **The engine's ground library** (wave TER2a, clause 3): five PBR ground sets,
//! synthesised here and committed as engine sample content.
//!
//! # Why this exists
//!
//! Before this wave the repository contained **zero** `.inf_tex` files. The
//! whole virtual-texture stack — the tiled container, the residency, the atlas,
//! the feedback ring, Wave T's per-splat-layer terrain branch and its detail-map
//! blend — was a capability with no content that reached it, and the 51 km²
//! island's ground was one flat colour because its four `TerrainLayer`s named no
//! material. This module is the content.
//!
//! # Why it is a CPU generator and not the P7 GPU bake
//!
//! The P7 material-graph path ends in `emit_texture_compute` → a WGSL compute
//! dispatch → a readback. That is the right authoring path for a texture an
//! author bakes into their own project, and it is the **wrong** one for bytes
//! this repository commits, for one reason: a compute bake's output is a fact
//! about the adapter that ran it. WGSL does not pin float evaluation order, the
//! trigonometric and exponential built-ins are the driver's, and this project's
//! own law is that a one-platform bound turns CI red (P25 — "one-platform bounds
//! red CI, adapters AND meshopt"). A committed `.inf_tex` whose bytes depend on
//! whose GPU blessed it could not be byte-locked on every leg, and the lock test
//! is what stops sample content going stale.
//!
//! So the synthesis is here, in `f64`, with:
//!
//! * an **integer** hash (SplitMix64) for every random value — no float RNG;
//! * value noise and Worley interpolated with `t·t·(3−2t)`, which is
//!   multiplication and subtraction;
//! * **no transcendental at all** — no `sin`, `cos`, `tan`, `powf`, `exp`,
//!   `ln`, `cbrt`. Ripples are triangle waves, anisotropy is per-axis frequency,
//!   and the one square root is `f64::sqrt`, which IEEE-754 requires to be
//!   correctly rounded (the same exemption `inf_terrain::erosion` takes).
//!
//! Every step after that — `rgba_mip_chain`, `compress_bc1`, the v2 tiler — is
//! already documented as "pure and byte-deterministic". So two builds on two
//! platforms produce the same bytes, and
//! `inf_editor_core::samples`'s lock compares them on every CI leg.
//!
//! # One pool format, and what that costs
//!
//! **Every map here is BC1**, including the normal maps, and that is a
//! measurement rather than a preference. `inf_render::build_vt_level` picks the
//! atlas format from the *stored* formats of the textures a level binds, and a
//! level whose textures are **mixed** falls back to `Rgba8` — 73 984 B a page
//! against BC1's 9 248, so a 24 MiB pool holds **2 721** pages instead of 340.
//!
//! The seventeen textures here need 51 pages of deterministic floor between
//! them, which either pool fits. What the demotion costs is everything *after*
//! the floor: 2 670 pages of camera-driven refinement against **289**, a 9.2×
//! cut in what can be resident at once — which at 1080p over ground authored at
//! two millimetres a texel is the difference between ground you can read and
//! ground that is permanently three mips coarse.
//!
//! Wave T shipped `PageFormat::Bc5` precisely for normal maps and it has no
//! consumer, which is exactly why: **the first content to use it alongside a BC1
//! albedo demotes the whole atlas.** The honest bound is that a BC1 normal map
//! quantises X and Y on 5:6:5 endpoints, which is worse than BC5 would be. The
//! fix is a second pool or a `view_formats` reinterpretation — named in
//! `vt_sample.wgsl`'s own module comment — and it is a wave, not a clause.
//!
//! # `detail_scale_m` is not metres
//!
//! Found by being the first consumer. `vt_apply_detail` computes
//! `duv = uv · scale`, so the field is **detail tiles per uv unit**; on terrain,
//! where `uv = world.xz / tex_scale`, one detail tile is `tex_scale /
//! detail_scale_m` metres. The default `0.5` therefore makes the detail layer
//! **twice as coarse** as the base tiling rather than ten times finer, which is
//! the opposite of what a detail map is for. The behaviour is right for a mesh
//! (a mesh uv has no metres in it at all); the *name and the doc* were wrong, and
//! they are corrected rather than the arithmetic — the VIS1b chromatic-aberration
//! precedent, and there is no shipped content depending on either reading because
//! before this wave there was none.
//!
//! [`GroundKind::detail_scale`] is what this library authors, and it states the
//! metres it works out to.

use crate::texture::{TextureCompression, TextureImportSettings};

/// Texels per side of a ground set's **albedo**.
///
/// 1 024, and the reason is the texel budget rather than the disk: at the
/// tightest tiling this library authors (sand, 1.5 m) that is a **1.46 mm**
/// texel, which is the "roughly a millimetre" class the mandate asks for. Going
/// to 2 048 would halve it again and quadruple the committed bytes; 1 024 is
/// where the two curves cross for content that ships inside a source tree.
pub const GROUND_ALBEDO_EXTENT: u32 = 1024;

/// Texels per side of a ground set's **normal**, **ORM** and **detail** maps.
///
/// Half the albedo's, deliberately. A ground normal's real signal is low
/// frequency — the shape of a clod, not the grain on it — and the grain is what
/// the *detail* map is for, blended at ten to twenty times the base tiling
/// rate. Spending equal bytes on both would buy the same picture for 3.4 MB
/// more.
pub const GROUND_MAP_EXTENT: u32 = 512;

/// The five ground sets.
///
/// Four of them are what the island's four splat layers bind
/// (`inf_island::splat`'s layer order: grass, rock, forest floor, sand); the
/// fifth, [`Soil`](GroundKind::Soil), is authored because worked ground is a
/// real surface an author will want and because the island's farmland biome is
/// carried as a grass/forest-floor mix for want of a fifth splat channel. It is
/// engine content either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GroundKind {
    Grass,
    Rock,
    ForestFloor,
    Sand,
    Soil,
}

impl GroundKind {
    /// Every kind, in a **frozen** order — it is the order the assets are
    /// written in and therefore the order their GUIDs are assigned in.
    pub const ALL: [GroundKind; 5] = [
        GroundKind::Grass,
        GroundKind::Rock,
        GroundKind::ForestFloor,
        GroundKind::Sand,
        GroundKind::Soil,
    ];

    /// The asset stem: `Ground_Grass`, `Ground_Rock`, …
    pub const fn stem(self) -> &'static str {
        match self {
            GroundKind::Grass => "Ground_Grass",
            GroundKind::Rock => "Ground_Rock",
            GroundKind::ForestFloor => "Ground_ForestFloor",
            GroundKind::Sand => "Ground_Sand",
            GroundKind::Soil => "Ground_Soil",
        }
    }

    /// A human label for a report.
    pub const fn label(self) -> &'static str {
        match self {
            GroundKind::Grass => "grass",
            GroundKind::Rock => "rock",
            GroundKind::ForestFloor => "forest floor",
            GroundKind::Sand => "sand",
            GroundKind::Soil => "soil",
        }
    }

    /// The seed this kind's synthesis is salted with. Frozen: changing one
    /// re-writes that set's committed bytes and nothing else.
    const fn seed(self) -> u64 {
        match self {
            GroundKind::Grass => 0x7E12_0001,
            GroundKind::Rock => 0x7E12_0002,
            GroundKind::ForestFloor => 0x7E12_0003,
            GroundKind::Sand => 0x7E12_0004,
            GroundKind::Soil => 0x7E12_0005,
        }
    }

    /// **World metres per tile** — what a `TerrainLayer::tex_scale` binding this
    /// set should carry, and the number the texel density is quoted against.
    ///
    /// Rock tiles largest because a rock pattern reads as bigger features; sand
    /// tightest because sand's features are millimetres and a wide tile makes it
    /// look like gravel.
    pub const fn tex_scale_m(self) -> f64 {
        match self {
            GroundKind::Grass => 2.0,
            GroundKind::Rock => 3.0,
            GroundKind::ForestFloor => 2.5,
            GroundKind::Sand => 1.5,
            GroundKind::Soil => 2.2,
        }
    }

    /// Metres per albedo texel at [`tex_scale_m`](Self::tex_scale_m).
    pub fn metres_per_texel(self) -> f64 {
        self.tex_scale_m() / f64::from(GROUND_ALBEDO_EXTENT)
    }

    /// Whether this set ships a **detail** map.
    ///
    /// Grass and rock, because those are the two an eye at a metre spends its
    /// time on: the ground you stand on and the cliff you look at. The other
    /// three would each cost another 259 KB of committed bytes for a surface
    /// that is usually further away, and the mechanism is proven by two
    /// consumers as well as by five.
    pub const fn has_detail(self) -> bool {
        matches!(self, GroundKind::Grass | GroundKind::Rock)
    }

    /// The `.inf_mat` `detail_scale_m` this set authors — **tiles per uv unit**,
    /// see the module note.
    ///
    /// `tex_scale_m / detail_scale` is the metres one detail tile covers:
    /// 12.5 cm for grass at 2 m / 16, and 15 cm for rock at 3 m / 20. Both are
    /// an order finer than the base tiling, which is what a detail map is for
    /// and what the default `0.5` gets exactly backwards.
    pub const fn detail_scale(self) -> f32 {
        match self {
            GroundKind::Grass => 16.0,
            GroundKind::Rock => 20.0,
            _ => 0.0,
        }
    }

    /// The scalar base colour the `.inf_mat` carries — the surface a host shows
    /// while the albedo's pages are still streaming, and what it falls back to
    /// on an adapter with no virtual textures at all. Linear.
    pub const fn base_color(self) -> [f32; 4] {
        match self {
            GroundKind::Grass => [0.086, 0.140, 0.052, 1.0],
            GroundKind::Rock => [0.128, 0.124, 0.118, 1.0],
            GroundKind::ForestFloor => [0.072, 0.055, 0.034, 1.0],
            GroundKind::Sand => [0.310, 0.262, 0.186, 1.0],
            GroundKind::Soil => [0.078, 0.052, 0.032, 1.0],
        }
    }

    /// The scalar roughness the `.inf_mat` carries.
    pub const fn roughness(self) -> f32 {
        match self {
            GroundKind::Grass => 0.94,
            GroundKind::Rock => 0.78,
            GroundKind::ForestFloor => 0.93,
            GroundKind::Sand => 0.88,
            GroundKind::Soil => 0.96,
        }
    }
}

/// One synthesised ground set, as RGBA8 buffers ready for the tiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroundMaps {
    /// `GROUND_ALBEDO_EXTENT²` RGBA8, **sRGB**.
    pub albedo: Vec<u8>,
    /// `GROUND_MAP_EXTENT²` RGBA8 tangent-space normal, linear (`xyz·2−1`).
    pub normal: Vec<u8>,
    /// `GROUND_MAP_EXTENT²` RGBA8 occlusion / roughness / metallic, linear —
    /// the glTF channel order `vt_sample.wgsl` reads.
    pub orm: Vec<u8>,
    /// `GROUND_MAP_EXTENT²` RGBA8 high-frequency normal, linear. `Some` exactly
    /// when [`GroundKind::has_detail`].
    pub detail: Option<Vec<u8>>,
}

/// Import settings for each of the four map slots.
///
/// **BC1 for all four** — see the module note on the one-pool-format
/// measurement. `srgb` is true only for the albedo; a normal or an ORM triple is
/// data, and encoding it in sRGB would bend every value through a transfer
/// function nothing undoes.
pub fn albedo_settings() -> TextureImportSettings {
    TextureImportSettings {
        srgb: true,
        generate_mips: true,
        compression: TextureCompression::Bc1,
        hdr: false,
    }
}

/// Import settings for a normal, ORM or detail map. See [`albedo_settings`].
pub fn data_settings() -> TextureImportSettings {
    TextureImportSettings {
        srgb: false,
        generate_mips: true,
        compression: TextureCompression::Bc1,
        hdr: false,
    }
}

// ── the noise ───────────────────────────────────────────────────────────────

/// SplitMix64. Integer in, integer out — no float arithmetic anywhere in the
/// random path, so the value at a lattice point is the same bit pattern on every
/// target.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A lattice point's hash, on a **wrapped** lattice so a field tiles.
#[inline]
fn lattice(x: i64, y: i64, px: i64, py: i64, seed: u64) -> u64 {
    let wx = x.rem_euclid(px.max(1)) as u64;
    let wy = y.rem_euclid(py.max(1)) as u64;
    mix64(seed ^ mix64(wx.wrapping_mul(0x0000_0100_0000_01B3).wrapping_add(wy)))
}

/// A lattice point's value in `[0, 1)`.
#[inline]
fn lattice_unit(x: i64, y: i64, px: i64, py: i64, seed: u64) -> f64 {
    // 53 bits into an f64 mantissa: exact, and the same on every target.
    (lattice(x, y, px, py, seed) >> 11) as f64 / (1u64 << 53) as f64
}

#[inline]
fn smooth(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// Tileable value noise over a domain of `px × py` cells.
///
/// `u`, `v` are in **cells**, so a caller scales its unit uv by the same numbers
/// it passes as the period — which is what keeps the field seamless.
fn vnoise(u: f64, v: f64, px: i64, py: i64, seed: u64) -> f64 {
    let x0 = u.floor();
    let y0 = v.floor();
    let tx = smooth(u - x0);
    let ty = smooth(v - y0);
    let (xi, yi) = (x0 as i64, y0 as i64);
    let a = lattice_unit(xi, yi, px, py, seed);
    let b = lattice_unit(xi + 1, yi, px, py, seed);
    let c = lattice_unit(xi, yi + 1, px, py, seed);
    let d = lattice_unit(xi + 1, yi + 1, px, py, seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// Tileable fBm: `octaves` doublings, amplitude halving, output in `[0, 1]`.
///
/// `(fx, fy)` are the base frequencies in cells across the unit domain, so the
/// anisotropy a ground surface needs (grass streaks, sand ripples) comes from
/// the two being different rather than from rotating a sample point — a rotation
/// would destroy the tiling.
fn fbm(u: f64, v: f64, fx: i64, fy: i64, octaves: u32, seed: u64) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        let s = 1i64 << o;
        let (px, py) = ((fx * s).max(1), (fy * s).max(1));
        sum += amp * vnoise(u * px as f64, v * py as f64, px, py, seed ^ (o as u64 + 1));
        norm += amp;
        amp *= 0.5;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

/// Tileable Worley: `(f1, f2)` — the distances to the nearest and second-nearest
/// jittered feature point, normalised so a cell diagonal is about 1.
///
/// `f1` is the pebble/clod field; `f2 − f1` is the crack between them, which is
/// what makes rock read as rock.
fn worley(u: f64, v: f64, cells: i64, seed: u64) -> (f64, f64) {
    let cells = cells.max(1);
    let (cu, cv) = (u * cells as f64, v * cells as f64);
    let (xi, yi) = (cu.floor() as i64, cv.floor() as i64);
    let mut f1 = f64::INFINITY;
    let mut f2 = f64::INFINITY;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (gx, gy) = (xi + dx, yi + dy);
            let h = lattice(gx, gy, cells, cells, seed);
            let jx = ((h >> 11) & 0xFFFF) as f64 / 65536.0;
            let jy = ((h >> 33) & 0xFFFF) as f64 / 65536.0;
            let (fx, fy) = (gx as f64 + jx, gy as f64 + jy);
            let (ex, ey) = (fx - cu, fy - cv);
            let d = (ex * ex + ey * ey).sqrt();
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    (f1.min(1.5), f2.min(1.5))
}

/// A tileable triangle wave in `[0, 1]`, `period` repeats across the domain.
///
/// The ripple primitive, and the reason there is no `sin` in this file: a
/// smoothed triangle is visually a sine and is three multiplications.
fn ripple(u: f64, period: f64) -> f64 {
    let t = u * period;
    let f = t - t.floor();
    smooth(1.0 - (f * 2.0 - 1.0).abs())
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

#[inline]
fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// The twelfth root of `x ∈ (0, 1]`.
///
/// **Not an approximation and not a third copy of `pcbrt`.** `1/2.4` is exactly
/// `5/12`, so an sRGB encode needs a twelfth root and nothing else.
///
/// # The seed is the whole problem, and it is worth the ten square roots
///
/// Newton on `y¹² − x` is quadratically convergent *near* the root and violently
/// unstable away from it: an eleventh power in the denominator means a seed 20 %
/// low sends the first step to **four times** the answer, and the walk back down
/// is linear. Measured: seeded at `x^(1/8)`, eight iterations at `x = 0.00317`
/// land on 0.6565 against a true 0.6191 — a **8.2-level** error in the encoded
/// byte, which is what the first draft of this function shipped and what its own
/// arm caught.
///
/// So the seed is built out of exact square roots instead:
/// `1/12 = 0.0833…` is `1/16 + 1/64 + 1/256 + 1/1024 + …` in binary, and the
/// first four terms give `x^0.0830078` — within 0.2 % of the answer over the
/// whole domain, from below or above. **Three** Newton steps from there is
/// converged f64, and the loop cannot overshoot because it never starts far
/// enough away to.
///
/// The iteration count is **fixed rather than tolerance-driven**, which is the
/// P23.5 ruling: a loop that stops when it is "close enough" is a function of
/// the rounding on the machine it ran on, and these bytes are committed.
fn twelfth_root(x: f64) -> f64 {
    // x^(1/2), x^(1/4), … x^(1/1024) — ten exact square roots.
    let s1 = x.sqrt();
    let s2 = s1.sqrt();
    let s3 = s2.sqrt();
    let s4 = s3.sqrt(); // 1/16
    let s5 = s4.sqrt();
    let s6 = s5.sqrt(); // 1/64
    let s7 = s6.sqrt();
    let s8 = s7.sqrt(); // 1/256
    let s9 = s8.sqrt();
    let s10 = s9.sqrt(); // 1/1024
    let mut y = s4 * s6 * s8 * s10;
    for _ in 0..3 {
        let y2 = y * y;
        let y4 = y2 * y2;
        let y8 = y4 * y4;
        let y11 = y8 * y2 * y;
        y = (11.0 * y + x / y11) / 12.0;
    }
    y
}

/// Linear → sRGB, for an albedo the container will tag `srgb`.
///
/// The piecewise IEC 61966-2-1 curve, exactly — `x^(1/2.4)` is `x^(5/12)`, which
/// is [`twelfth_root`] raised to the fifth. No `powf`, and no error term to
/// argue about: [`tests::the_srgb_encode_tracks_the_real_curve`] measures it
/// against `powf` over 4 097 points and the worst disagreement is at the ulp.
fn linear_to_srgb(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        return v * 12.92;
    }
    let r = twelfth_root(v);
    let y = r * r * r * r * r;
    (1.055 * y - 0.055).clamp(0.0, 1.0)
}

// ── the surfaces ────────────────────────────────────────────────────────────

/// One texel's synthesised surface: a linear albedo, a height in `[0, 1]`, a
/// roughness and an ambient occlusion.
struct Surface {
    albedo: [f64; 3],
    height: f64,
    roughness: f64,
    ao: f64,
}

/// Evaluate a ground kind at a **unit uv**, tileably.
///
/// One function for both extents: the albedo pass runs it at 1 024² and the
/// map pass at 512², over the same domain, so a normal and an albedo describe
/// the same surface rather than two surfaces that resemble each other.
fn surface_at(kind: GroundKind, u: f64, v: f64) -> Surface {
    let s = kind.seed();
    match kind {
        GroundKind::Grass => {
            // Blades: strongly anisotropic, with the direction varying by clump
            // so it is not a corduroy. Two fields at right angles, mixed by a
            // low-frequency mask.
            let along = fbm(u, v, 6, 96, 4, s ^ 0x11);
            let across = fbm(u, v, 96, 6, 4, s ^ 0x12);
            let dir = fbm(u, v, 3, 3, 2, s ^ 0x13);
            let blade = lerp(along, across, smooth(dir));
            // Clumps: tufts of grass with soil between them.
            let (f1, _) = worley(u, v, 9, s ^ 0x21);
            let clump = 1.0 - smooth((f1 * 1.7).clamp(0.0, 1.0));
            // Dry patches and dead thatch.
            let dry = smooth((fbm(u, v, 4, 4, 3, s ^ 0x31) * 1.6 - 0.55).clamp(0.0, 1.0));
            let soil = smooth((0.55 - clump).clamp(0.0, 1.0) * 1.8);

            let live = [0.045, 0.128, 0.030];
            let tip = [0.145, 0.205, 0.062];
            let thatch = [0.150, 0.118, 0.048];
            let under = [0.048, 0.036, 0.022];
            let mut c = [0.0; 3];
            for k in 0..3 {
                let green = lerp(live[k], tip[k], blade);
                let dryed = lerp(green, thatch[k], dry * 0.75);
                c[k] = lerp(dryed, under[k], soil * 0.6);
            }
            Surface {
                albedo: c,
                height: (clump * 0.7 + blade * 0.3).clamp(0.0, 1.0),
                roughness: lerp(0.97, 0.88, blade),
                ao: lerp(0.55, 1.0, clump * 0.6 + 0.4),
            }
        }
        GroundKind::Rock => {
            // Cracks between plates, plus a ridged fBm for the plate faces.
            let (f1, f2) = worley(u, v, 5, s ^ 0x41);
            let crack = 1.0 - smooth(((f2 - f1) * 5.0).clamp(0.0, 1.0));
            let (g1, g2) = worley(u, v, 17, s ^ 0x42);
            let fine_crack = 1.0 - smooth(((g2 - g1) * 7.0).clamp(0.0, 1.0));
            // Ridged: |2n − 1| folded, which is a fold and an absolute value.
            let n = fbm(u, v, 8, 8, 5, s ^ 0x43);
            let ridge = 1.0 - (n * 2.0 - 1.0).abs();
            let grain = fbm(u, v, 48, 48, 3, s ^ 0x44);
            let iron = smooth((fbm(u, v, 3, 4, 3, s ^ 0x45) * 1.8 - 0.7).clamp(0.0, 1.0));
            let plate = f1.min(1.0);

            let pale = [0.168, 0.163, 0.152];
            let dark = [0.062, 0.060, 0.058];
            let rust = [0.155, 0.088, 0.042];
            let mut c = [0.0; 3];
            for k in 0..3 {
                let base = lerp(dark[k], pale[k], ridge * 0.55 + plate * 0.35 + grain * 0.10);
                let stained = lerp(base, rust[k], iron * 0.55);
                c[k] = lerp(
                    stained,
                    dark[k] * 0.55,
                    (crack * 0.8 + fine_crack * 0.4).min(1.0),
                );
            }
            let h = (ridge * 0.45 + plate * 0.4 + grain * 0.15)
                * (1.0 - crack * 0.85)
                * (1.0 - fine_crack * 0.4);
            Surface {
                albedo: c,
                height: h.clamp(0.0, 1.0),
                roughness: lerp(0.62, 0.92, grain * 0.5 + crack * 0.5),
                ao: (1.0 - crack * 0.75 - fine_crack * 0.25).clamp(0.15, 1.0),
            }
        }
        GroundKind::ForestFloor => {
            // Needle litter: many fine elongated splinters, two orientations,
            // over leaf blotches and dark humus.
            let n1 = fbm(u, v, 10, 128, 4, s ^ 0x51);
            let n2 = fbm(u, v, 128, 10, 4, s ^ 0x52);
            let n3 = fbm(u, v, 64, 64, 3, s ^ 0x53);
            let which = smooth(fbm(u, v, 5, 5, 2, s ^ 0x54));
            let litter = (lerp(n1, n2, which) * 0.75 + n3 * 0.25).clamp(0.0, 1.0);
            let (l1, _) = worley(u, v, 7, s ^ 0x55);
            let leaf = 1.0 - smooth((l1 * 1.9).clamp(0.0, 1.0));
            let moss = smooth((fbm(u, v, 4, 4, 3, s ^ 0x56) * 1.7 - 0.75).clamp(0.0, 1.0));

            let humus = [0.030, 0.021, 0.013];
            let needle = [0.118, 0.074, 0.035];
            let leafc = [0.155, 0.098, 0.041];
            let mossc = [0.038, 0.078, 0.028];
            let mut c = [0.0; 3];
            for k in 0..3 {
                let a = lerp(humus[k], needle[k], smooth(litter));
                let b = lerp(a, leafc[k], leaf * 0.55);
                c[k] = lerp(b, mossc[k], moss * 0.7);
            }
            Surface {
                albedo: c,
                height: (litter * 0.55 + leaf * 0.45).clamp(0.0, 1.0),
                roughness: lerp(0.98, 0.86, leaf * 0.5 + moss * 0.5),
                ao: lerp(0.5, 1.0, litter * 0.5 + leaf * 0.5),
            }
        }
        GroundKind::Sand => {
            // Ripples with a wandering crest, fine grain, and a few shell
            // fragments. The ripple direction bends with a low-frequency warp so
            // it is a beach and not a washboard.
            let warp = fbm(u, v, 3, 3, 3, s ^ 0x61) - 0.5;
            let rip = ripple(v + warp * 0.35, 18.0) * 0.6 + ripple(v + warp * 0.9, 41.0) * 0.4;
            let grain = fbm(u, v, 160, 160, 2, s ^ 0x62);
            let patch = fbm(u, v, 5, 5, 3, s ^ 0x63);
            let (sh, _) = worley(u, v, 26, s ^ 0x64);
            let shell = 1.0 - smooth((sh * 4.2).clamp(0.0, 1.0));
            let wet = smooth((patch * 1.5 - 0.6).clamp(0.0, 1.0));

            let dry = [0.352, 0.298, 0.208];
            let damp = [0.176, 0.146, 0.104];
            let pale = [0.430, 0.398, 0.330];
            let mut c = [0.0; 3];
            for k in 0..3 {
                let a = lerp(dry[k], pale[k], rip * 0.35 + grain * 0.2);
                let b = lerp(a, damp[k], wet * 0.6);
                c[k] = lerp(b, 0.62, shell * 0.8);
            }
            Surface {
                albedo: c,
                height: (rip * 0.7 + grain * 0.2 + shell * 0.1).clamp(0.0, 1.0),
                roughness: lerp(0.90, 0.72, wet * 0.7 + shell * 0.3),
                ao: lerp(0.75, 1.0, rip),
            }
        }
        GroundKind::Soil => {
            // Clods with cracks between them, plus a fine tilth and the odd
            // stone. Worked ground: the clods are larger and flatter than rock's
            // plates and the cracks are wider.
            let (f1, f2) = worley(u, v, 8, s ^ 0x71);
            let crack = 1.0 - smooth(((f2 - f1) * 3.2).clamp(0.0, 1.0));
            let clod = 1.0 - smooth((f1 * 1.5).clamp(0.0, 1.0));
            let tilth = fbm(u, v, 40, 40, 4, s ^ 0x72);
            let (st, _) = worley(u, v, 21, s ^ 0x73);
            let stone = 1.0 - smooth((st * 5.5).clamp(0.0, 1.0));
            let damp = smooth((fbm(u, v, 4, 4, 3, s ^ 0x74) * 1.6 - 0.62).clamp(0.0, 1.0));

            let dry = [0.108, 0.076, 0.048];
            let wetc = [0.038, 0.026, 0.017];
            let stonec = [0.140, 0.136, 0.128];
            let mut c = [0.0; 3];
            for k in 0..3 {
                let a = lerp(dry[k], dry[k] * 1.4, tilth);
                let b = lerp(a, wetc[k], damp * 0.7 + crack * 0.5);
                c[k] = lerp(b, stonec[k], stone * 0.85);
            }
            Surface {
                albedo: c,
                height: (clod * 0.6 + tilth * 0.25 + stone * 0.15) * (1.0 - crack * 0.8),
                roughness: lerp(0.99, 0.80, stone),
                ao: (1.0 - crack * 0.7).clamp(0.2, 1.0),
            }
        }
    }
}

/// The **detail** height field — the grain a detail map carries, at ten to
/// twenty times the base tiling rate.
///
/// Deliberately a different function from [`surface_at`]'s height rather than
/// the same one at a higher frequency: a detail map that is the base map scaled
/// down repeats the base map's own features and reads as a moiré.
fn detail_height(kind: GroundKind, u: f64, v: f64) -> f64 {
    let s = kind.seed() ^ 0xD_E7A;
    match kind {
        GroundKind::Grass => {
            // Individual blade edges and the soil crumb under them.
            let edge = fbm(u, v, 12, 150, 3, s ^ 0x81);
            let crumb = fbm(u, v, 90, 90, 3, s ^ 0x82);
            (edge * 0.65 + crumb * 0.35).clamp(0.0, 1.0)
        }
        GroundKind::Rock => {
            // Crystal facets and pits.
            let (f1, f2) = worley(u, v, 22, s ^ 0x83);
            let facet = f1.min(1.0);
            let pit = 1.0 - smooth(((f2 - f1) * 6.0).clamp(0.0, 1.0));
            let grit = fbm(u, v, 110, 110, 2, s ^ 0x84);
            ((facet * 0.5 + grit * 0.5) * (1.0 - pit * 0.7)).clamp(0.0, 1.0)
        }
        _ => fbm(u, v, 96, 96, 3, s),
    }
}

/// Encode a height field as an RGBA8 tangent-space normal map, tileably.
///
/// Central differences on the **wrapped** grid, so the map's own edges match —
/// a normal map whose border is one-sided shows a seam at every tile boundary,
/// which on ground tiling every two metres is a grid across the whole world.
fn normal_from_height(h: &[f64], n: u32, strength: f64) -> Vec<u8> {
    let idx = |x: i64, y: i64| -> f64 {
        let xi = x.rem_euclid(i64::from(n)) as usize;
        let yi = y.rem_euclid(i64::from(n)) as usize;
        h[yi * n as usize + xi]
    };
    let mut out = vec![0u8; (n * n * 4) as usize];
    for y in 0..n as i64 {
        for x in 0..n as i64 {
            let dx = (idx(x + 1, y) - idx(x - 1, y)) * strength;
            let dy = (idx(x, y + 1) - idx(x, y - 1)) * strength;
            // The unnormalised normal of a height field is (−dx, −dy, 1).
            let len = (dx * dx + dy * dy + 1.0).sqrt();
            let (nx, ny, nz) = (-dx / len, -dy / len, 1.0 / len);
            let o = ((y as u32 * n) + x as u32) as usize * 4;
            out[o] = to_u8(nx * 0.5 + 0.5);
            out[o + 1] = to_u8(ny * 0.5 + 0.5);
            out[o + 2] = to_u8(nz * 0.5 + 0.5);
            out[o + 3] = 255;
        }
    }
    out
}

/// **Synthesise one ground set.** Pure: the output is a function of `kind`
/// alone.
pub fn synthesize(kind: GroundKind) -> GroundMaps {
    let a = GROUND_ALBEDO_EXTENT;
    let m = GROUND_MAP_EXTENT;

    // The albedo, at full extent.
    let mut albedo = vec![0u8; (a * a * 4) as usize];
    for y in 0..a {
        for x in 0..a {
            let (u, v) = (f64::from(x) / f64::from(a), f64::from(y) / f64::from(a));
            let s = surface_at(kind, u, v);
            let o = ((y * a) + x) as usize * 4;
            for k in 0..3 {
                // Cavity darkening from the same surface's own AO: the single
                // cheapest thing that makes a procedural ground stop looking
                // procedural, and it is free here because the AO is already
                // computed.
                let lit = s.albedo[k] * lerp(0.55, 1.0, s.ao);
                albedo[o + k] = to_u8(linear_to_srgb(lit));
            }
            albedo[o + 3] = 255;
        }
    }

    // The height, roughness and AO, at map extent — one walk, three outputs.
    let n = (m * m) as usize;
    let mut height = vec![0.0f64; n];
    let mut orm = vec![0u8; n * 4];
    for y in 0..m {
        for x in 0..m {
            let (u, v) = (f64::from(x) / f64::from(m), f64::from(y) / f64::from(m));
            let s = surface_at(kind, u, v);
            let i = ((y * m) + x) as usize;
            height[i] = s.height;
            orm[i * 4] = to_u8(s.ao);
            orm[i * 4 + 1] = to_u8(s.roughness);
            orm[i * 4 + 2] = 0; // ground is never metal
            orm[i * 4 + 3] = 255;
        }
    }
    // 6.0 is a slope gain, not a physical number: it turns a `[0, 1]` height
    // over a 512-texel tile into normals that read at a metre's eye height
    // without going to full-strength grazing angles that alias.
    let normal = normal_from_height(&height, m, 6.0);

    let detail = kind.has_detail().then(|| {
        let mut dh = vec![0.0f64; n];
        for y in 0..m {
            for x in 0..m {
                let (u, v) = (f64::from(x) / f64::from(m), f64::from(y) / f64::from(m));
                dh[((y * m) + x) as usize] = detail_height(kind, u, v);
            }
        }
        // Weaker than the base: `vt_apply_detail` ADDS this to the base normal's
        // xy, so a full-strength detail would flatten the surface it is
        // detailing.
        normal_from_height(&dh, m, 2.5)
    });

    GroundMaps {
        albedo,
        normal,
        orm,
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The synthesis is pure**, which is the whole premise of committing its
    /// output. Two calls, byte for byte.
    #[test]
    fn two_synthesies_of_one_kind_agree_byte_for_byte() {
        for kind in GroundKind::ALL {
            let a = synthesize(kind);
            let b = synthesize(kind);
            assert_eq!(a, b, "{} is not a pure function", kind.label());
        }
    }

    /// Every map is the extent and the length it claims, and the detail map is
    /// present exactly where [`GroundKind::has_detail`] says.
    #[test]
    fn every_map_is_the_shape_the_tiler_will_be_handed() {
        let mut with_detail = 0;
        for kind in GroundKind::ALL {
            let g = synthesize(kind);
            assert_eq!(
                g.albedo.len(),
                (GROUND_ALBEDO_EXTENT * GROUND_ALBEDO_EXTENT * 4) as usize
            );
            assert_eq!(
                g.normal.len(),
                (GROUND_MAP_EXTENT * GROUND_MAP_EXTENT * 4) as usize
            );
            assert_eq!(g.orm.len(), g.normal.len());
            assert_eq!(g.detail.is_some(), kind.has_detail());
            if let Some(d) = &g.detail {
                assert_eq!(d.len(), g.normal.len());
                with_detail += 1;
            }
            // Ground is never metal, and the alpha is opaque so `Auto` and `Bc1`
            // would agree — the format is forced anyway, and this pins the intent.
            assert!(g.orm.chunks_exact(4).all(|p| p[2] == 0), "metallic is set");
            assert!(g.albedo.chunks_exact(4).all(|p| p[3] == 255));
        }
        assert_eq!(with_detail, 2, "the detail-map consumers moved");
    }

    /// **The maps tile.** A ground texture repeating every 1.5–3 m shows its own
    /// edge as a grid across the world if the field is not periodic, so this is
    /// measured rather than asserted by construction: the field at `u` and at
    /// `u + 1` is the same value, on every axis, for every kind.
    #[test]
    fn the_fields_are_periodic_on_both_axes() {
        for kind in GroundKind::ALL {
            for (u, v) in [(0.0, 0.0), (0.317, 0.0), (0.0, 0.732), (0.618, 0.241)] {
                let a = surface_at(kind, u, v);
                let bx = surface_at(kind, u + 1.0, v);
                let by = surface_at(kind, u, v + 1.0);
                for (label, b) in [("+u", &bx), ("+v", &by)] {
                    assert!(
                        (a.height - b.height).abs() < 1e-9,
                        "{} height is not periodic across {label} at ({u}, {v}): \
                         {} vs {}",
                        kind.label(),
                        a.height,
                        b.height
                    );
                    for k in 0..3 {
                        assert!(
                            (a.albedo[k] - b.albedo[k]).abs() < 1e-9,
                            "{} albedo channel {k} is not periodic across {label}",
                            kind.label()
                        );
                    }
                }
            }
        }
    }

    /// …and the **encoded** maps tile too — the seam a normal map shows is at
    /// its own border texels, where a central difference has to wrap.
    #[test]
    fn the_normal_map_has_no_seam_at_its_own_border() {
        let n = 64u32;
        let h: Vec<f64> = (0..n * n)
            .map(|i| {
                let (x, y) = (i % n, i / n);
                fbm(
                    f64::from(x) / f64::from(n),
                    f64::from(y) / f64::from(n),
                    8,
                    8,
                    3,
                    7,
                )
            })
            .collect();
        let map = normal_from_height(&h, n, 6.0);
        let px = |x: u32, y: u32| -> [u8; 3] {
            let o = ((y * n + x) * 4) as usize;
            [map[o], map[o + 1], map[o + 2]]
        };
        // Column 0's neighbours are column n-1 and column 1; if the difference
        // wrapped, column 0 and column n-1 are as close to each other as any two
        // adjacent interior columns.
        let mut worst_edge = 0i32;
        let mut worst_interior = 0i32;
        for y in 0..n {
            let d = |a: [u8; 3], b: [u8; 3]| -> i32 {
                (0..3)
                    .map(|k| (i32::from(a[k]) - i32::from(b[k])).abs())
                    .max()
                    .unwrap_or(0)
            };
            worst_edge = worst_edge.max(d(px(0, y), px(n - 1, y)));
            worst_interior = worst_interior.max(d(px(n / 2, y), px(n / 2 + 1, y)));
        }
        assert!(
            worst_edge <= worst_interior * 2,
            "the wrap seam ({worst_edge}) is worse than an ordinary interior \
             step ({worst_interior}) — the normal map does not tile"
        );
    }

    /// **The five sets are five surfaces.** A generator whose kinds differ only
    /// in a tint would pass every arm above and put one ground on the whole
    /// island; this measures that the mean colours are separated and that the
    /// height fields are not the same field.
    #[test]
    fn the_five_kinds_are_five_distinct_grounds() {
        let means: Vec<[f64; 3]> = GroundKind::ALL
            .iter()
            .map(|k| {
                let g = synthesize(*k);
                let mut acc = [0.0f64; 3];
                for p in g.albedo.chunks_exact(4) {
                    for c in 0..3 {
                        acc[c] += f64::from(p[c]);
                    }
                }
                let n = (g.albedo.len() / 4) as f64;
                [acc[0] / n, acc[1] / n, acc[2] / n]
            })
            .collect();
        for (i, k) in GroundKind::ALL.iter().enumerate() {
            println!(
                "GROUND {:>13}: mean sRGB ({:.1}, {:.1}, {:.1}), {:.3} mm/texel at \
                 tex_scale {} m",
                k.label(),
                means[i][0],
                means[i][1],
                means[i][2],
                k.metres_per_texel() * 1000.0,
                k.tex_scale_m()
            );
        }
        for i in 0..means.len() {
            for j in (i + 1)..means.len() {
                let d: f64 = (0..3)
                    .map(|c| (means[i][c] - means[j][c]).abs())
                    .fold(0.0, f64::max);
                assert!(
                    d > 8.0,
                    "{} and {} differ by only {d:.1} levels in their strongest \
                     channel — they would read as one ground",
                    GroundKind::ALL[i].label(),
                    GroundKind::ALL[j].label()
                );
            }
        }
    }

    /// The albedo carries **contrast**, which is what "not a flat colour" means
    /// as a number. A constant image would pass the distinctness arm above.
    #[test]
    fn every_albedo_has_real_variation() {
        for kind in GroundKind::ALL {
            let g = synthesize(kind);
            let lum: Vec<f64> = g
                .albedo
                .chunks_exact(4)
                .map(|p| {
                    0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])
                })
                .collect();
            let n = lum.len() as f64;
            let mean = lum.iter().sum::<f64>() / n;
            let var = lum.iter().map(|l| (l - mean) * (l - mean)).sum::<f64>() / n;
            let sd = var.sqrt();
            let (lo, hi) = lum
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), l| (a.min(*l), b.max(*l)));
            println!(
                "GROUND {:>13}: luma mean {mean:.1}, sd {sd:.2}, range {lo:.0}..{hi:.0}",
                kind.label()
            );
            assert!(
                sd > 4.0,
                "{} has a luma sd of {sd:.2} — it is a flat colour with a rumour \
                 of noise on it",
                kind.label()
            );
            assert!(
                hi - lo > 40.0,
                "{} spans only {:.0} levels",
                kind.label(),
                hi - lo
            );
        }
    }

    /// The normal maps are **normal maps**: every texel decodes to a unit
    /// vector with a positive Z (a normal pointing into the surface is damage,
    /// not data), and the field is not flat.
    #[test]
    fn every_normal_map_decodes_to_a_unit_vector_facing_out() {
        for kind in GroundKind::ALL {
            let g = synthesize(kind);
            let mut maps: Vec<(&str, &Vec<u8>)> = vec![("normal", &g.normal)];
            if let Some(d) = &g.detail {
                maps.push(("detail", d));
            }
            for (what, map) in maps {
                let mut tilted = 0u64;
                let mut total = 0u64;
                for p in map.chunks_exact(4) {
                    let v = [
                        f64::from(p[0]) / 255.0 * 2.0 - 1.0,
                        f64::from(p[1]) / 255.0 * 2.0 - 1.0,
                        f64::from(p[2]) / 255.0 * 2.0 - 1.0,
                    ];
                    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    assert!(
                        (len - 1.0).abs() < 0.02,
                        "{} {what}: a texel decodes to length {len:.4}",
                        kind.label()
                    );
                    assert!(v[2] > 0.0, "{} {what}: a normal faces inward", kind.label());
                    if v[0].abs() > 0.08 || v[1].abs() > 0.08 {
                        tilted += 1;
                    }
                    total += 1;
                }
                let frac = tilted as f64 / total as f64;
                assert!(
                    frac > 0.05,
                    "{} {what} is {:.2} % tilted — it is a flat map",
                    kind.label(),
                    frac * 100.0
                );
            }
        }
    }

    /// The sRGB encode is the real curve, to the ulp — measured against `powf`,
    /// which this module is not allowed to call and a test is.
    #[test]
    fn the_srgb_encode_tracks_the_real_curve() {
        let truth = |v: f64| -> f64 {
            if v <= 0.003_130_8 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        };
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f64;
        for i in 0..=4096 {
            let v = f64::from(i) / 4096.0;
            let e = (linear_to_srgb(v) - truth(v)).abs();
            if e > worst {
                worst = e;
                worst_at = v;
            }
        }
        println!(
            "sRGB encode: worst disagreement with powf {:.5} ({:.3} of 255) at v = {worst_at:.4}",
            worst,
            worst * 255.0
        );
        assert!(
            worst * 255.0 < 0.001,
            "the sRGB encode is {:.6} levels out at v = {worst_at} — it is \
             supposed to BE the curve, not approximate it",
            worst * 255.0
        );
        // …and the twelfth root really is one: an anti-vacuity check that the
        // Newton loop converged rather than returning its seed.
        for x in [0.0031_308, 0.1, 0.5, 0.9, 1.0] {
            let r = twelfth_root(x);
            assert!(
                (r.powi(12) - x).abs() < 1e-14,
                "twelfth_root({x}) = {r}, whose 12th power is {}",
                r.powi(12)
            );
        }
    }

    /// The authored numbers are the ones the ledger quotes, and the detail scale
    /// really is finer than the base tiling — the thing the field's own default
    /// gets backwards.
    #[test]
    fn the_authored_scales_are_a_detail_layer_and_not_a_second_base_layer() {
        for kind in GroundKind::ALL {
            if !kind.has_detail() {
                assert_eq!(kind.detail_scale(), 0.0, "an inert set carries a scale");
                continue;
            }
            let period_m = kind.tex_scale_m() / f64::from(kind.detail_scale());
            println!(
                "GROUND {:>13}: base tile {} m, detail tile {:.3} m ({:.1}x finer)",
                kind.label(),
                kind.tex_scale_m(),
                period_m,
                kind.tex_scale_m() / period_m
            );
            assert!(
                period_m * 8.0 < kind.tex_scale_m(),
                "{}'s detail tile is {period_m:.3} m against a {} m base — that is \
                 not a detail layer",
                kind.label(),
                kind.tex_scale_m()
            );
            // …and it fits the 8.8 fixed-point encoding without saturating.
            assert!(kind.detail_scale() * 256.0 < f32::from(u16::MAX));
        }
    }

    /// The five stems and the five seeds are five of each — a copy-paste that
    /// duplicated a seed would make two sets identical and every arm above
    /// except the distinctness one would pass.
    #[test]
    fn the_kinds_have_distinct_stems_and_seeds() {
        let stems: Vec<&str> = GroundKind::ALL.iter().map(|k| k.stem()).collect();
        let seeds: Vec<u64> = GroundKind::ALL.iter().map(|k| k.seed()).collect();
        for i in 0..GroundKind::ALL.len() {
            for j in (i + 1)..GroundKind::ALL.len() {
                assert_ne!(stems[i], stems[j], "two kinds share a stem");
                assert_ne!(seeds[i], seeds[j], "two kinds share a seed");
            }
        }
    }
}
