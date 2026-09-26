//! **THE CAR SHELLS** (wave VEH3f.2b) -- five closed car bodies built in the
//! P23 DCC kernel, committed as samples, licence-free because they are ours.
//!
//! # What a shell IS, and what it replaced
//!
//! Wave VEH3f's "hero bodies" were three or four lofted PANELS hung on the box
//! family: the audit measured their side outline 2.4-8.7 % off the all-box car.
//! A shell is one closed body mesh per car -- wheel arches cut through it, a
//! glasshouse whose window openings are apertures, a bonnet and a boot recessed
//! into it -- and the parts VEH3c hinges, sheds and shatters (doors, bonnet,
//! boot, bumpers, every pane) cut FROM that body as separate closed meshes, so
//! the door that swings on the VEH3c revolute is the piece of skin that was in
//! the aperture. A steering wheel on the VEH3d hub, a dashboard, seats, lamps,
//! a grille and mirrors ride the same parts table.
//!
//! # How it reaches a car -- the ART door, unchanged
//!
//! A shell is a row of `inf_ecs::vehicle_art`'s table (`shell = true`): its
//! parts are `[[art.parts]]` rows in the SAME TOML shape an imported Fab car's
//! are, its meshes live at the SAME computed identities (`art_body_guid`,
//! `art_part_guid`, `art_wheel_guid`), and a roster row wears one by saying
//! `art = "shell_sedan"` -- so a row swaps imported art for a shell, or back,
//! with no code. The one difference is scale: an imported body is metres at
//! one car's size, a shell is hull FRACTIONS (every `BodyPart`'s own
//! convention), so the rig hangs it with the chassis half-extents as its scale
//! and any row of its family fits it.
//!
//! # How a body is modelled
//!
//! A loft along `+Z` in the kernel's own ops (`Op::AddVertex`, `Op::AddFace`,
//! `Op::SetEdgeSharp`): each station is a 34-point ring whose points are named
//! SLOTS on the car's cross-section (floor, arch, rocker, door skin, belt,
//! glasshouse, rail, roof). A feature is a slot's alternative position over a
//! z-range -- the wheel arch notch, the door aperture, the glass and bonnet
//! recesses -- entered and left across a double station a few millimetres
//! apart, so every notch has real end walls and the tube stays one closed
//! manifold. The ends are fans to a pole. Parts are small lofts of the same
//! slot functions (a door is the skin between its sill and belt slots, a pane
//! the skin between the belt and the rail), which is why they fit their
//! apertures.
//!
//! # The arches fit the ROWS that wear them
//!
//! The arch is not a guess: [`envelope`] reads every catalogue row that names
//! the shell (the island's row and the roster's), puts each row's tyre at its
//! SETTLED height (its drop plus its static sag), and sizes the arch ellipse
//! to contain every tyre's upper half with [`ARCH_CLEAR_M`] to spare -- so a
//! row that names a shell its wheels do not fit changes the committed bytes,
//! and `committed_sample_matches_generators` says so on all three platforms.
//!
//! # Determinism
//!
//! `+ - * /`, `sqrt` and `pcos64`/`psin64` only; the kernel's own ops; the bytes
//! are compared against the committed files on every CI platform.

use std::path::{Path, PathBuf};

use inf_dcc::ops::apply;
use inf_dcc::{to_mesh_asset, CornerData, ExportOptions, Mesh, NormalPolicy, Op, VertId};
use inf_ecs::vehicle::VehicleDef;
use inf_math::{pcos64, psin64};
use uuid::Uuid;

use crate::vehicle_bodies::HeroMesh;

/// The committed folder, under `samples/`.
pub const VEHICLE_SHELLS_FOLDER: &str = "vehicle-shells";

/// How far apart a feature's two stations are, hull fractions of `z`.
const EPS: f64 = 0.003;

/// **The clearance an arch keeps round a settled tyre**, metres.
pub const ARCH_CLEAR_M: f64 = 0.045;

/// **The drawn tyre's width, as a fraction of the primitive's** (the rig
/// scales a tyre to `2 r TYRE_WIDTH_FRAC` wide; the shell's tyre fills
/// [`TYRE_FILL`] of that, a road tyre's 0.25 m on a 0.34 m wheel).
pub const TYRE_FILL: f64 = 0.60;

/// Faces meeting at more than this dihedral angle get a sharp edge, degrees.
const SHARP_DEG: f64 = 40.0;

/// **The five shells.**
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Shell {
    /// A four-door saloon.
    Sedan,
    /// A two-door coupe (the roster's coupes and the island's sports car).
    Coupe,
    /// A five-door SUV.
    Suv,
    /// A two-door pickup with an open bed.
    Pickup,
    /// The saloon with a push bar -- the police cruiser (its light bar is the
    /// livery's own part, drawn with this shell's mesh).
    Cruiser,
}

impl Shell {
    /// Every shell, in file order.
    pub const ALL: [Shell; 5] = [
        Shell::Sedan,
        Shell::Coupe,
        Shell::Suv,
        Shell::Pickup,
        Shell::Cruiser,
    ];

    /// The art-table key a row's `art = "..."` spells.
    pub fn key(self) -> &'static str {
        match self {
            Shell::Sedan => "shell_sedan",
            Shell::Coupe => "shell_coupe",
            Shell::Suv => "shell_suv",
            Shell::Pickup => "shell_pickup",
            Shell::Cruiser => "shell_cruiser",
        }
    }

    /// The island catalogue row that wears it.
    pub fn island_row(self) -> &'static str {
        match self {
            Shell::Sedan => "sedan",
            Shell::Coupe => "sports",
            Shell::Suv => "suv",
            Shell::Pickup => "truck",
            Shell::Cruiser => "cruiser",
        }
    }

    /// The art key.
    pub fn art(self) -> inf_ecs::roster::ArtKey {
        inf_ecs::roster::ArtKey::from_name(self.key())
            .unwrap_or_else(|| panic!("`{}` is a row of vehicle_art.toml", self.key()))
    }

    /// The shell whose BODY this one wears: the cruiser is the saloon.
    pub fn body_of(self) -> Shell {
        match self {
            Shell::Cruiser => Shell::Sedan,
            s => s,
        }
    }

    /// Every shell that wears this one's body (the saloon and the cruiser
    /// share one, so their arches fit both families of rows).
    fn body_twins(self) -> &'static [Shell] {
        match self.body_of() {
            Shell::Sedan => &[Shell::Sedan, Shell::Cruiser],
            Shell::Coupe => &[Shell::Coupe],
            Shell::Suv => &[Shell::Suv],
            Shell::Pickup => &[Shell::Pickup],
            Shell::Cruiser => unreachable!(),
        }
    }
}

// ── the rows, and the arch they need ─────────────────────────────────────────

/// **Every catalogue row that wears `shell`**, `(id, def)`: the island's own
/// catalogue first, then the roster, each in id order.
pub fn shell_rows(shell: Shell) -> Vec<(String, VehicleDef)> {
    let key = shell.key();
    let mut out = Vec::new();
    for (id, d) in crate::vehicle::island_vehicles().0 {
        if d.art.map(|k| k.name()) == Some(key) {
            out.push((id, d));
        }
    }
    for (id, d) in inf_ecs::roster::roster().0.iter() {
        if d.art.map(|k| k.name()) == Some(key) {
            out.push((id.clone(), *d));
        }
    }
    out
}

/// One wheel arch, hull fractions: an ellipse centred on `(z, y)` with radii
/// `(rz, ry)`, open below its centre to the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arch {
    /// Centre along the car.
    pub z: f64,
    /// Centre height.
    pub y: f64,
    /// Half-length.
    pub rz: f64,
    /// Rise above the centre.
    pub ry: f64,
}

impl Arch {
    /// The opening's top at `z`, or `None` outside the arch.
    pub fn open_at(&self, z: f64) -> Option<f64> {
        let t = (z - self.z) / self.rz;
        if t.abs() >= 1.0 {
            return None;
        }
        Some(self.y + self.ry * (1.0 - t * t).max(0.0).sqrt())
    }
}

/// **Where a row's settled tyre is**, hull fractions: `(z_front, z_rear, y,
/// radius_z, radius_y, inner_x)` -- the tyre's circle with [`ARCH_CLEAR_M`]
/// added, and the inside face of the drawn tyre less the same clearance.
fn tyre_fracs(def: &VehicleDef) -> (f64, f64, f64, f64, f64, f64) {
    let h = def.half_extents;
    let (hw, hh, hl) = (h.x.abs(), h.y.abs(), h.z.abs());
    let r = def.wheel_radius_m;
    let sag = (def.static_travel_frac().clamp(0.0, 1.0)) * def.class.travel_m.max(0.0);
    let y = (def.wheel_drop_m + sag) / hh;
    let o = def.wheel_offset_z_m;
    let zf = (o + def.half_wheelbase_m) / hl;
    let zr = (o - def.half_wheelbase_m) / hl;
    let rr = r + ARCH_CLEAR_M;
    let tyre_half_w = r * inf_ecs::vehicle::TYRE_WIDTH_FRAC * TYRE_FILL;
    let x_in = (def.half_track_m - tyre_half_w - ARCH_CLEAR_M) / hw;
    (zf, zr, y, rr / hl, rr / hh, x_in)
}

/// **The two arches and the wheel-well's inner wall** a shell's body cuts, fitted
/// to every row that wears it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Envelope {
    /// The front arch.
    pub front: Arch,
    /// The rear arch.
    pub rear: Arch,
    /// The wheel well's inner wall, a fraction of the half-width.
    pub x_in: f64,
    /// How many rows it was fitted to.
    pub rows: usize,
}

/// The smallest rise an arch centred on `(zc, yc)` with half-length `rz` needs
/// to contain every circle's upper half, or `None` when a circle pokes past
/// its ends.
fn rise_for(circles: &[(f64, f64, f64, f64)], zc: f64, yc: f64, rz: f64) -> Option<f64> {
    let mut ry: f64 = 0.0;
    for c in circles {
        if (c.0 - zc).abs() + c.2 >= rz {
            return None;
        }
        for k in 0..=64 {
            let th = std::f64::consts::PI * k as f64 / 64.0;
            let (z, y) = (c.0 + c.2 * pcos64(th), c.1 + c.3 * psin64(th));
            if y <= yc {
                continue;
            }
            let t = (z - zc) / rz;
            let room = (1.0 - t * t).max(1e-9).sqrt();
            ry = ry.max((y - yc) / room);
        }
    }
    Some(ry)
}

/// **Fit one arch to circles** `(z, y, rz, ry)` (hull fractions; `hh`/`hl` the
/// design row's half-height and half-length, metres): the ellipse that
/// contains every circle's upper half with the least metres of daylight --
/// its crown over the highest tyre, plus a third of its extra length -- found
/// by a grid over its centre height and its half-length. Arithmetic only.
fn fit_arch(circles: &[(f64, f64, f64, f64)], hh: f64, hl: f64) -> Arch {
    let (mut zlo, mut zhi) = (f64::MAX, f64::MIN);
    let (mut ylo, mut yhi, mut top) = (f64::MAX, f64::MIN, f64::MIN);
    for c in circles {
        zlo = zlo.min(c.0 - c.2);
        zhi = zhi.max(c.0 + c.2);
        ylo = ylo.min(c.1);
        yhi = yhi.max(c.1);
        top = top.max(c.1 + c.3);
    }
    let zc = 0.5 * (zlo + zhi);
    let extent = 0.5 * (zhi - zlo);
    let mut best: Option<(f64, Arch)> = None;
    for i in 0..=24 {
        let yc = ylo - 0.15 + (yhi - ylo + 0.15) * i as f64 / 24.0;
        for j in 0..=40 {
            let rz = extent * (1.02 + 0.6 * j as f64 / 40.0) + 0.002;
            let Some(ry) = rise_for(circles, zc, yc, rz) else {
                continue;
            };
            let ry = ry * 1.01 + 0.003;
            let cost = (yc + ry - top) * hh + 0.33 * (rz - extent) * hl;
            if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                best = Some((
                    cost,
                    Arch {
                        z: zc,
                        y: yc,
                        rz,
                        ry,
                    },
                ));
            }
        }
    }
    best.map(|(_, a)| a).expect("an arch contains its tyres")
}

/// **The arch envelope of `shell`'s body** over every row that wears it (and
/// its twin's rows: the saloon and the cruiser share a body).
pub fn envelope(shell: Shell) -> Envelope {
    let mut front = Vec::new();
    let mut rear = Vec::new();
    let mut x_in: f64 = 0.8;
    let mut rows = 0usize;
    for s in shell.body_twins() {
        for (_, def) in shell_rows(*s) {
            let (zf, zr, y, rz, ry, xi) = tyre_fracs(&def);
            front.push((zf, y, rz, ry));
            rear.push((zr, y, rz, ry));
            x_in = x_in.min(xi);
            rows += 1;
        }
    }
    if rows == 0 {
        // No row names it yet: the island row's own numbers, so the generator
        // still builds a car (a table edit in progress, never a committed state
        // -- `every_shell_is_worn` holds the rows to it).
        let def = crate::vehicle::island_vehicles()
            .get(shell.body_of().island_row())
            .copied()
            .expect("the island row");
        let (zf, zr, y, rz, ry, xi) = tyre_fracs(&def);
        front.push((zf, y, rz, ry));
        rear.push((zr, y, rz, ry));
        x_in = x_in.min(xi);
    }
    let dh = design_half(shell.body_of());
    Envelope {
        front: fit_arch(&front, dh[1], dh[2]),
        rear: fit_arch(&rear, dh[1], dh[2]),
        x_in: x_in.max(0.3),
        rows,
    }
}

// ── the profile ──────────────────────────────────────────────────────────────

/// A smooth 0..1 ramp over `[a, b]` (either order) -- `3t^2 - 2t^3`.
fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// A curve through `(z, v)` control points (ascending `z`): Catmull-Rom in the
/// point index, solved for `z` by bisection -- arithmetic only.
#[derive(Clone, Debug)]
struct Curve(Vec<(f64, f64)>);

impl Curve {
    fn at(&self, z: f64) -> f64 {
        let p = &self.0;
        if z <= p[0].0 {
            return p[0].1;
        }
        if z >= p[p.len() - 1].0 {
            return p[p.len() - 1].1;
        }
        let i = p.iter().rposition(|q| q.0 <= z).unwrap_or(0).min(p.len() - 2);
        let get = |k: isize| {
            let k = k.clamp(0, p.len() as isize - 1) as usize;
            p[k]
        };
        let (p0, p1, p2, p3) = (
            get(i as isize - 1),
            get(i as isize),
            get(i as isize + 1),
            get(i as isize + 2),
        );
        let cr = |a: f64, b: f64, c: f64, d: f64, t: f64| {
            let t2 = t * t;
            let t3 = t2 * t;
            0.5 * ((2.0 * b)
                + (-a + c) * t
                + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
                + (-a + 3.0 * b - 3.0 * c + d) * t3)
        };
        // z(t) is monotone for control points this well spread; bisect it.
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            let zm = cr(p0.0, p1.0, p2.0, p3.0, mid);
            if zm < z {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        cr(p0.1, p1.1, p2.1, p3.1, 0.5 * (lo + hi))
    }
}

/// **One shell's proportions**, hull fractions (`x` of the half-width, `y` of
/// the half-height, `z` of the half-length; `+Z` is the nose).
#[derive(Clone, Debug)]
struct Profile {
    /// The upper silhouette: nose face, bonnet, windscreen, roof, backlight,
    /// deck, tail.
    top: Curve,
    /// The underside.
    bot: Curve,
    /// Where the windscreen meets the bonnet.
    cowl: f64,
    /// The roof's leading edge.
    roof_front: f64,
    /// The roof's trailing edge.
    roof_rear: f64,
    /// Where the backlight meets the deck (the boot's forward edge).
    deck_front: f64,
    /// The beltline.
    belt: f64,
    /// The rocker's top -- a door's sill.
    sill: f64,
    /// The cabin's centre wall, half-width.
    spine: f64,
    /// The roof's edge half-width (the tumblehome's top).
    greenhouse_edge: f64,
    /// The bonnet's and the deck's edge half-width.
    fender_edge: f64,
    /// The body's half-width at the belt, mid-car.
    body_half: f64,
    /// How much the plan pulls in at the nose and the tail.
    nose_round: f64,
    tail_round: f64,
    /// Four doors or two.
    four_doors: bool,
    /// The pickup's bed: `(rear z, front z, floor y)`.
    bed: Option<(f64, f64, f64)>,
    /// Whether the rear deck is a separate boot lid.
    boot: bool,
    /// Whether the backlight is a recessed pane.
    backlight: bool,
}

fn profile(shell: Shell) -> Profile {
    match shell.body_of() {
        Shell::Sedan | Shell::Cruiser => Profile {
            top: Curve(vec![
                (-1.0, -0.14),
                (-0.985, 0.08),
                (-0.95, 0.21),
                (-0.80, 0.28),
                (-0.66, 0.31),
                (-0.54, 0.58),
                (-0.42, 0.87),
                (-0.32, 0.95),
                (-0.18, 0.99),
                (-0.04, 0.975),
                (0.08, 0.90),
                (0.22, 0.49),
                (0.37, 0.11),
                (0.62, 0.04),
                (0.86, -0.04),
                (0.955, -0.15),
                (0.99, -0.31),
                (1.0, -0.44),
            ]),
            bot: Curve(vec![
                (-1.0, -0.60),
                (-0.93, -0.74),
                (-0.82, -0.94),
                (-0.62, -1.0),
                (0.62, -1.0),
                (0.82, -0.93),
                (0.93, -0.76),
                (1.0, -0.62),
            ]),
            cowl: 0.37,
            roof_front: 0.08,
            roof_rear: -0.42,
            deck_front: -0.66,
            belt: 0.22,
            sill: -0.78,
            spine: 0.10,
            greenhouse_edge: 0.76,
            fender_edge: 0.92,
            body_half: 0.985,
            nose_round: 0.16,
            tail_round: 0.12,
            four_doors: true,
            bed: None,
            boot: true,
            backlight: true,
        },
        Shell::Coupe => Profile {
            top: Curve(vec![
                (-1.0, -0.05),
                (-0.985, 0.16),
                (-0.95, 0.25),
                (-0.84, 0.30),
                (-0.72, 0.36),
                (-0.50, 0.64),
                (-0.28, 0.92),
                (-0.12, 0.99),
                (0.02, 0.97),
                (0.14, 0.72),
                (0.30, 0.16),
                (0.55, 0.06),
                (0.82, -0.02),
                (0.95, -0.12),
                (0.99, -0.26),
                (1.0, -0.36),
            ]),
            bot: Curve(vec![
                (-1.0, -0.72),
                (-0.9, -0.86),
                (-0.76, -0.99),
                (-0.6, -1.0),
                (0.6, -1.0),
                (0.8, -0.95),
                (0.93, -0.82),
                (1.0, -0.70),
            ]),
            cowl: 0.30,
            roof_front: 0.02,
            roof_rear: -0.28,
            deck_front: -0.72,
            belt: 0.18,
            sill: -0.78,
            spine: 0.10,
            greenhouse_edge: 0.70,
            fender_edge: 0.90,
            body_half: 0.985,
            nose_round: 0.18,
            tail_round: 0.12,
            four_doors: false,
            bed: None,
            boot: true,
            backlight: true,
        },
        Shell::Suv => Profile {
            top: Curve(vec![
                (-1.0, 0.00),
                (-0.99, 0.30),
                (-0.97, 0.66),
                (-0.93, 0.92),
                (-0.86, 0.98),
                (-0.40, 1.0),
                (0.10, 0.99),
                (0.20, 0.94),
                (0.32, 0.55),
                (0.43, 0.14),
                (0.66, 0.08),
                (0.88, 0.02),
                (0.96, -0.08),
                (0.99, -0.22),
                (1.0, -0.34),
            ]),
            bot: Curve(vec![
                (-1.0, -0.72),
                (-0.9, -0.86),
                (-0.78, -0.98),
                (-0.6, -1.0),
                (0.6, -1.0),
                (0.8, -0.95),
                (0.92, -0.84),
                (1.0, -0.72),
            ]),
            cowl: 0.43,
            roof_front: 0.20,
            roof_rear: -0.90,
            deck_front: -0.985,
            belt: 0.14,
            sill: -0.76,
            spine: 0.10,
            greenhouse_edge: 0.80,
            fender_edge: 0.92,
            body_half: 0.985,
            nose_round: 0.14,
            tail_round: 0.06,
            four_doors: true,
            bed: None,
            boot: false,
            backlight: true,
        },
        Shell::Pickup => Profile {
            top: Curve(vec![
                (-1.0, 0.04),
                (-0.985, 0.10),
                (-0.60, 0.11),
                (-0.10, 0.12),
                (-0.06, 0.40),
                (-0.03, 0.90),
                (0.00, 0.98),
                (0.20, 1.0),
                (0.34, 0.97),
                (0.43, 0.60),
                (0.52, 0.16),
                (0.72, 0.11),
                (0.90, 0.05),
                (0.965, -0.06),
                (0.99, -0.20),
                (1.0, -0.32),
            ]),
            bot: Curve(vec![
                (-1.0, -0.70),
                (-0.92, -0.80),
                (-0.80, -0.98),
                (-0.6, -1.0),
                (0.6, -1.0),
                (0.8, -0.95),
                (0.92, -0.84),
                (1.0, -0.72),
            ]),
            cowl: 0.52,
            roof_front: 0.34,
            roof_rear: -0.01,
            deck_front: -0.03,
            belt: 0.12,
            sill: -0.76,
            spine: 0.10,
            greenhouse_edge: 0.82,
            fender_edge: 0.94,
            body_half: 0.985,
            nose_round: 0.12,
            tail_round: 0.04,
            four_doors: false,
            bed: Some((-0.94, -0.09, -0.50)),
            boot: false,
            backlight: false,
        },
    }
}

// ── the slot functions ───────────────────────────────────────────────────────

/// The body's plan scale at `z`.
fn plan(p: &Profile, z: f64) -> f64 {
    let a = smooth(0.80, 1.0, z);
    let b = smooth(0.82, 1.0, -z);
    1.0 - p.nose_round * a * a - p.tail_round * b * b
}

/// The greenhouse factor at `z`: 0 over the bonnet and the deck, 1 on the roof.
fn greenhouse(p: &Profile, z: f64) -> f64 {
    if z >= p.cowl {
        0.0
    } else if z >= p.roof_front {
        smooth(p.cowl, p.roof_front, z)
    } else if z >= p.roof_rear {
        1.0
    } else if z >= p.deck_front {
        1.0 - smooth(p.roof_rear, p.deck_front, z)
    } else {
        0.0
    }
}

/// Every level of the cross-section at `z`, in order bottom to top, made
/// strictly increasing so no ring folds.
#[derive(Clone, Copy, Debug)]
struct Levels {
    yb: f64,
    sill: f64,
    belt: f64,
    shoulder: f64,
    top: f64,
    crown: f64,
    x_body: f64,
    x_rock: f64,
    x_edge: f64,
    edge_w: f64,
}

fn levels(p: &Profile, z: f64) -> Levels {
    let g = greenhouse(p, z);
    let crown = lerp(0.035, 0.10, g);
    let yb = p.bot.at(z);
    let top_raw = p.top.at(z);
    let sill = p.sill.max(yb + 0.06);
    let belt = p.belt.min(top_raw - crown - 0.05).max(sill + 0.06);
    let shoulder = (top_raw - crown).max(belt + 0.04);
    let top = top_raw.max(shoulder + 0.012);
    let s = plan(p, z);
    let x_body = p.body_half * s;
    Levels {
        yb,
        sill,
        belt,
        shoulder,
        top,
        crown: top - shoulder,
        x_body,
        x_rock: x_body * 0.955,
        x_edge: lerp(p.fender_edge, p.greenhouse_edge, g) * s,
        edge_w: lerp(0.05, 0.10, g),
    }
}

/// The roof (or bonnet, or deck) height at half-width `x`.
fn roof_y(l: &Levels, x: f64) -> f64 {
    let t = (x / l.x_edge).clamp(-1.0, 1.0);
    l.shoulder + l.crown * (1.0 - t * t)
}

/// What a station's top recess is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Recess {
    /// A pane or a lid flush in the skin: lowered by `depth`, `edge` in from
    /// the top edge.
    Skin { edge: f64, depth: f64 },
    /// The pickup's bed: down to the floor, `edge` walls.
    Bed { edge: f64, floor: f64 },
}

/// How deep a glass recess is (hull fractions of `y`).
const GLASS_DEPTH: f64 = 0.035;
/// How deep a bonnet or a boot lid sits.
const LID_DEPTH: f64 = 0.035;
/// A shut line's gap, hull fractions.
const SHUT_GAP: f64 = 0.006;

/// **Which features are live at a station.**
#[derive(Clone, Copy, Debug)]
struct Features {
    arch: Option<f64>,
    pocket: bool,
    recess: Option<Recess>,
}

/// The door apertures, `(rear z, front z)`, front door first.
fn door_ranges(p: &Profile, env: &Envelope) -> Vec<(f64, f64)> {
    let front_arch_rear = env.front.z - env.front.rz;
    let rear_arch_front = env.rear.z + env.rear.rz;
    let front = p.cowl.min(front_arch_rear - 0.03);
    let rear = rear_arch_front + 0.03;
    if p.four_doors {
        vec![(-0.02, front), (rear.max(-0.46), -0.06)]
    } else {
        let back = match p.bed {
            Some((_, bf, _)) => (bf + 0.08).max(rear),
            None => rear.max(-0.36),
        };
        vec![(back, front)]
    }
}

fn features(p: &Profile, env: &Envelope, z: f64) -> Features {
    let arch = env.front.open_at(z).or_else(|| env.rear.open_at(z));
    let pocket = arch.is_none() && door_ranges(p, env).iter().any(|(a, b)| z > *a && z < *b);
    let recess = if let Some((br, bf, floor)) = p.bed {
        (z > br && z < bf).then_some(Recess::Bed { edge: 0.07, floor })
    } else {
        None
    }
    .or_else(|| {
        (z > p.roof_front && z < p.cowl).then_some(Recess::Skin {
            edge: 0.11,
            depth: GLASS_DEPTH,
        })
    })
    .or_else(|| {
        (p.backlight && z > p.deck_front && z < p.roof_rear).then_some(Recess::Skin {
            edge: 0.11,
            depth: GLASS_DEPTH,
        })
    })
    .or_else(|| {
        (z > p.cowl + 2.0 * SHUT_GAP && z < 0.955).then_some(Recess::Skin {
            edge: 0.05,
            depth: LID_DEPTH,
        })
    })
    .or_else(|| {
        (p.boot && z > -0.965 && z < p.deck_front - 2.0 * SHUT_GAP).then_some(Recess::Skin {
            edge: 0.05,
            depth: LID_DEPTH,
        })
    });
    Features {
        arch,
        pocket,
        recess,
    }
}

/// The slots of the ring's right half, bottom to top (the bottom centre and
/// the top centre are the two ring points on the mirror plane).
const HALF: usize = 16;
const F1: usize = 0;
const A0: usize = 1;
const B: usize = 2;
const C: usize = 3;
const D: usize = 4;
const E: usize = 5;
const S1: usize = 6;
const S2: usize = 7;
const W: usize = 8;
const G1: usize = 9;
const G2: usize = 10;
const R: usize = 11;
const K1: usize = 12;
const K1R: usize = 13;
const K2: usize = 14;
const K3: usize = 15;

/// **The ring at `z`**: `(bottom centre, right half, top centre)` in `(x, y)`.
fn ring(p: &Profile, env: &Envelope, z: f64) -> ([f64; 2], [[f64; 2]; HALF], [f64; 2]) {
    let l = levels(p, z);
    let f = features(p, env, z);
    let x_in = env.x_in.min(l.x_rock * 0.9);
    let mut s = [[0.0f64; 2]; HALF];
    s[F1] = [0.5 * x_in, l.yb];
    s[A0] = [x_in, l.yb];
    s[B] = [0.5 * (x_in + l.x_rock), l.yb];
    s[C] = [l.x_rock, l.yb];
    s[D] = [l.x_rock, lerp(l.yb, l.sill, 0.55)];
    s[E] = [l.x_body * 0.985, l.sill];
    s[S1] = [l.x_body, lerp(l.sill, l.belt, 0.35)];
    s[S2] = [l.x_body, lerp(l.sill, l.belt, 0.75)];
    s[W] = [l.x_body * 0.995, l.belt];
    let wx = l.x_body * 0.99;
    s[G1] = [lerp(wx, l.x_edge, 0.45), lerp(l.belt, l.shoulder, 0.45)];
    s[G2] = [lerp(wx, l.x_edge, 0.85), lerp(l.belt, l.shoulder, 0.85)];
    s[R] = [l.x_edge, l.shoulder];
    let k1 = l.x_edge - l.edge_w;
    s[K1] = [k1, roof_y(&l, k1)];
    s[K1R] = [0.8 * k1, roof_y(&l, 0.8 * k1)];
    s[K2] = [0.5 * k1, roof_y(&l, 0.5 * k1)];
    s[K3] = [0.2 * k1, roof_y(&l, 0.2 * k1)];
    let mut top = [0.0, l.top];
    let bottom = [0.0, l.yb];

    if let Some(yo) = f.arch {
        let lip = l.x_body * 1.01;
        let wy = l.belt.max(yo + 0.04);
        s[B] = [x_in, yo];
        s[C] = [lip, yo];
        s[D] = [lip, lerp(yo, wy, 0.2)];
        s[E] = [l.x_body * 1.005, lerp(yo, wy, 0.4)];
        s[S1] = [l.x_body, lerp(yo, wy, 0.6)];
        s[S2] = [l.x_body, lerp(yo, wy, 0.8)];
        s[W] = [l.x_body * 0.995, wy];
        let ry = l.shoulder.max(wy + 0.03);
        s[G1] = [lerp(wx, l.x_edge, 0.45), lerp(wy, ry, 0.45)];
        s[G2] = [lerp(wx, l.x_edge, 0.85), lerp(wy, ry, 0.85)];
        s[R] = [l.x_edge, ry];
    }
    if f.pocket {
        let ceiling = (l.shoulder - 0.06).max(l.belt + 0.05);
        s[S1] = [p.spine, l.sill];
        s[S2] = [p.spine, lerp(l.sill, ceiling, 0.5)];
        s[W] = [p.spine, ceiling];
        s[G1] = [lerp(p.spine, l.x_edge, 0.55), ceiling];
        s[G2] = [l.x_edge - 0.03, ceiling];
    }
    match f.recess {
        Some(Recess::Skin { edge, depth }) => {
            let k = l.x_edge - edge;
            s[K1] = [k, roof_y(&l, k)];
            s[K1R] = [k, roof_y(&l, k) - depth];
            s[K2] = [0.5 * k, roof_y(&l, 0.5 * k) - depth];
            s[K3] = [0.2 * k, roof_y(&l, 0.2 * k) - depth];
            top = [0.0, l.top - depth];
        }
        Some(Recess::Bed { edge, floor }) => {
            let k = l.x_edge - edge;
            s[K1] = [k, roof_y(&l, k)];
            s[K1R] = [k, floor];
            s[K2] = [0.5 * k, floor];
            s[K3] = [0.2 * k, floor];
            top = [0.0, floor];
        }
        None => {}
    }
    (bottom, s, top)
}

/// Flatten a ring to its 34 points, counter-clockwise seen from the nose.
fn ring_points(bottom: [f64; 2], s: &[[f64; 2]; HALF], top: [f64; 2]) -> Vec<[f64; 2]> {
    let mut v = Vec::with_capacity(2 * HALF + 2);
    v.push(bottom);
    // Up the RIGHT (+X) flank...
    v.extend(s.iter().copied());
    v.push(top);
    // ...and down the left.
    v.extend(s.iter().rev().map(|q| [-q[0], q[1]]));
    v
}

/// **The body's stations**: a base sampling, the arch arcs, and every
/// feature's double station.
fn stations(p: &Profile, env: &Envelope) -> Vec<f64> {
    let mut z: Vec<f64> = (0..=56).map(|i| -1.0 + 2.0 * i as f64 / 56.0).collect();
    for a in [env.front, env.rear] {
        for k in 0..=10 {
            let t = -1.0 + 2.0 * k as f64 / 10.0;
            let zz = a.z + a.rz * t * 0.985;
            z.push(zz);
        }
        z.push(a.z - a.rz - EPS);
        z.push(a.z - a.rz + EPS);
        z.push(a.z + a.rz - EPS);
        z.push(a.z + a.rz + EPS);
    }
    let mut edges = vec![p.roof_front, p.roof_rear, 0.955, p.cowl + 2.0 * SHUT_GAP];
    edges.push(p.cowl);
    if p.backlight {
        edges.push(p.deck_front);
    }
    if p.boot {
        edges.push(-0.965);
        edges.push(p.deck_front - 2.0 * SHUT_GAP);
    }
    if let Some((br, bf, _)) = p.bed {
        edges.push(br);
        edges.push(bf);
    }
    for (a, b) in door_ranges(p, env) {
        edges.push(a);
        edges.push(b);
    }
    for e in edges {
        z.push(e - EPS);
        z.push(e + EPS);
    }
    z.retain(|v| (-1.0..=1.0).contains(v));
    z.sort_by(|a, b| a.total_cmp(b));
    z.dedup_by(|a, b| (*a - *b).abs() < 0.5 * EPS);
    z
}

// ── the mesh builder ─────────────────────────────────────────────────────────

/// A mesh assembled through the kernel's own ops.
struct Builder {
    mesh: Mesh,
}

impl Builder {
    fn new() -> Self {
        Self { mesh: Mesh::new() }
    }

    fn vert(&mut self, p: [f64; 3]) -> VertId {
        apply(&mut self.mesh, &Op::AddVertex { position: p })
            .expect("a finite vertex")
            .verts[0]
    }

    fn face(&mut self, vs: &[VertId]) {
        let corners = vs
            .iter()
            .map(|v| {
                let q = self.mesh.position(*v).expect("a live vertex");
                CornerData {
                    uv: [0.5 + 0.5 * q.z, 0.5 + 0.5 * q.y],
                    normal: None,
                }
            })
            .collect();
        apply(
            &mut self.mesh,
            &Op::AddFace {
                verts: vs.to_vec(),
                corners,
                slot: None,
            },
        )
        .unwrap_or_else(|e| panic!("a shell face: {e:?}"));
    }

    /// **A closed tube**: `rings` (each the same length, each a closed loop)
    /// lofted in order and capped at both ends with a fan to a pole. Winding
    /// is chosen so the tube's signed volume is positive (normals out).
    fn tube(&mut self, rings: &[Vec<[f64; 3]>]) {
        let n = rings[0].len();
        let flip = tube_volume(rings) < 0.0;
        let ids: Vec<Vec<VertId>> = rings
            .iter()
            .map(|r| r.iter().map(|q| self.vert(*q)).collect())
            .collect();
        let pole = |r: &Vec<[f64; 3]>, out: f64| {
            let mut c = [0.0; 3];
            for q in r {
                for k in 0..3 {
                    c[k] += q[k] / r.len() as f64;
                }
            }
            c[2] += out;
            c
        };
        let dz = rings[rings.len() - 1][0][2] - rings[0][0][2];
        let bump = 0.004 * dz.signum();
        let p0 = self.vert(pole(&rings[0], -bump));
        let p1 = self.vert(pole(&rings[rings.len() - 1], bump));
        let quad = |b: &mut Builder, a: [VertId; 4]| {
            if flip {
                b.face(&[a[3], a[2], a[1], a[0]]);
            } else {
                b.face(&a);
            }
        };
        for s in 0..rings.len() - 1 {
            for i in 0..n {
                let j = (i + 1) % n;
                quad(self, [ids[s][i], ids[s][j], ids[s + 1][j], ids[s + 1][i]]);
            }
        }
        let last = rings.len() - 1;
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b) = if flip {
                ([ids[0][i], ids[0][j], p0], [ids[last][j], ids[last][i], p1])
            } else {
                ([ids[0][j], ids[0][i], p0], [ids[last][i], ids[last][j], p1])
            };
            self.face(&a);
            self.face(&b);
        }
    }

    /// Mark every edge whose faces meet at more than [`SHARP_DEG`] sharp.
    fn auto_sharp(&mut self) {
        let cos_lim = pcos64(SHARP_DEG.to_radians());
        let mut normals = std::collections::BTreeMap::new();
        for f in self.mesh.face_ids().collect::<Vec<_>>() {
            let vs = self.mesh.face_verts(f).expect("a live face");
            let ps: Vec<_> = vs
                .iter()
                .map(|v| self.mesh.position(*v).expect("live"))
                .collect();
            let mut nrm = glam::DVec3::ZERO;
            for i in 0..ps.len() {
                let (a, b) = (ps[i], ps[(i + 1) % ps.len()]);
                nrm.x += (a.y - b.y) * (a.z + b.z);
                nrm.y += (a.z - b.z) * (a.x + b.x);
                nrm.z += (a.x - b.x) * (a.y + b.y);
            }
            normals.insert(f, nrm.normalize_or_zero());
        }
        let halfs: Vec<_> = self.mesh.half_ids().collect();
        let mut ops = Vec::new();
        for h in halfs {
            let Some(t) = self.mesh.twin(h) else { continue };
            if t < h {
                continue;
            }
            let (Some(Some(fa)), Some(Some(fb))) = (self.mesh.face_of(h), self.mesh.face_of(t))
            else {
                continue;
            };
            let (na, nb) = (normals[&fa], normals[&fb]);
            if na == glam::DVec3::ZERO || nb == glam::DVec3::ZERO {
                continue;
            }
            if na.dot(nb) < cos_lim {
                ops.push(h);
            }
        }
        for h in ops {
            apply(
                &mut self.mesh,
                &Op::SetEdgeSharp {
                    half: h,
                    sharp: true,
                },
            )
            .expect("a live edge");
        }
    }

    fn finish(mut self) -> Mesh {
        self.auto_sharp();
        self.mesh
    }
}

/// The signed volume a tube's quads and caps enclose, wound as `tube` winds
/// them unflipped.
fn tube_volume(rings: &[Vec<[f64; 3]>]) -> f64 {
    let n = rings[0].len();
    let tri = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
        (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            / 6.0
    };
    let cen = |r: &Vec<[f64; 3]>| {
        let mut c = [0.0; 3];
        for q in r {
            for k in 0..3 {
                c[k] += q[k] / r.len() as f64;
            }
        }
        c
    };
    let mut v = 0.0;
    for s in 0..rings.len() - 1 {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b, c, d) = (rings[s][i], rings[s][j], rings[s + 1][j], rings[s + 1][i]);
            v += tri(a, b, c) + tri(a, c, d);
        }
    }
    let (c0, c1) = (cen(&rings[0]), cen(&rings[rings.len() - 1]));
    let last = rings.len() - 1;
    for i in 0..n {
        let j = (i + 1) % n;
        v += tri(rings[0][j], rings[0][i], c0);
        v += tri(rings[last][i], rings[last][j], c1);
    }
    v
}

// ── the body ─────────────────────────────────────────────────────────────────

/// **The shell's closed body**, hull fractions.
pub fn body_mesh(shell: Shell) -> Mesh {
    let p = profile(shell);
    let env = envelope(shell);
    let rings: Vec<Vec<[f64; 3]>> = stations(&p, &env)
        .into_iter()
        .map(|z| {
            let (b, s, t) = ring(&p, &env, z);
            ring_points(b, &s, t)
                .into_iter()
                .map(|q| [q[0], q[1], z])
                .collect()
        })
        .collect();
    let mut b = Builder::new();
    b.tube(&rings);
    b.finish()
}

// ── the parts ────────────────────────────────────────────────────────────────

/// One part of a shell: its name (the naming rule decides its kind), its box
/// in hull fractions, and its mesh in hull fractions.
pub struct ShellPart {
    /// The part's name.
    pub name: String,
    /// Box centre, hull fractions.
    pub centre: [f64; 3],
    /// Box half-extents, hull fractions.
    pub half: [f64; 3],
    /// The mesh, hull fractions (not yet in its unit box).
    pub mesh: Mesh,
}

/// Every vertex of a mesh, `[x, y, z]`.
fn verts_of(m: &Mesh) -> Vec<[f64; 3]> {
    m.vert_ids()
        .map(|v| {
            let q = m.position(v).expect("live");
            [q.x, q.y, q.z]
        })
        .collect()
}

fn part(name: &str, mesh: Mesh) -> ShellPart {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in verts_of(&mesh) {
        for k in 0..3 {
            lo[k] = lo[k].min(q[k]);
            hi[k] = hi[k].max(q[k]);
        }
    }
    // Rounded to 1e-4 so the committed table carries the exact numbers the
    // generator uses (four decimals, the art table's own precision).
    let r4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
    let centre = [
        r4(0.5 * (lo[0] + hi[0])),
        r4(0.5 * (lo[1] + hi[1])),
        r4(0.5 * (lo[2] + hi[2])),
    ];
    let half = [
        r4(0.5 * (hi[0] - lo[0])).max(0.0005),
        r4(0.5 * (hi[1] - lo[1])).max(0.0005),
        r4(0.5 * (hi[2] - lo[2])).max(0.0005),
    ];
    ShellPart {
        name: name.to_string(),
        centre,
        half,
        mesh,
    }
}

/// The body's own stations inside `(a, b)` plus the two ends.
fn span(p: &Profile, env: &Envelope, a: f64, b: f64) -> Vec<f64> {
    let mut z = vec![a];
    z.extend(stations(p, env).into_iter().filter(|v| *v > a + EPS && *v < b - EPS));
    z.push(b);
    z
}

/// **A door**: the body's skin between its sill and its belt over the door's
/// z-range, a door's thickness deep, on the `side` flank.
fn door(p: &Profile, env: &Envelope, name: &str, side: f64, a: f64, b: f64) -> ShellPart {
    let (a, b) = (a + SHUT_GAP, b - SHUT_GAP);
    let t = 0.06;
    let rings: Vec<Vec<[f64; 3]>> = span(p, env, a, b)
        .into_iter()
        .map(|z| {
            let l = levels(p, z);
            let outer = [
                [l.x_body * 0.985 - 0.004, l.sill + SHUT_GAP],
                [l.x_body - 0.004, lerp(l.sill, l.belt, 0.35)],
                [l.x_body - 0.004, lerp(l.sill, l.belt, 0.75)],
                [l.x_body * 0.995 - 0.004, l.belt - SHUT_GAP],
            ];
            let mut r: Vec<[f64; 3]> = outer.iter().map(|q| [side * q[0], q[1], z]).collect();
            r.extend(outer.iter().rev().map(|q| [side * (q[0] - t), q[1], z]));
            r
        })
        .collect();
    let mut bld = Builder::new();
    bld.tube(&rings);
    part(name, bld.finish())
}

/// **A side pane**: the glasshouse skin between the belt and the aperture's
/// ceiling over a door's z-range, set a little in from the door skin.
fn side_glass(p: &Profile, env: &Envelope, name: &str, side: f64, a: f64, b: f64) -> ShellPart {
    let (a, b) = (a + SHUT_GAP, b - SHUT_GAP);
    let rings: Vec<Vec<[f64; 3]>> = span(p, env, a, b)
        .into_iter()
        .map(|z| {
            let l = levels(p, z);
            let ceiling = (l.shoulder - 0.06).max(l.belt + 0.05) - 0.004;
            let wx = l.x_body * 0.99;
            let slope = |y: f64| {
                let t = ((y - l.belt) / (l.shoulder - l.belt)).clamp(0.0, 1.0);
                lerp(wx, l.x_edge, t) - 0.02
            };
            let ys = [l.belt + 0.004, lerp(l.belt, ceiling, 0.5), ceiling];
            let outer: Vec<[f64; 2]> = ys.iter().map(|y| [slope(*y), *y]).collect();
            let mut r: Vec<[f64; 3]> = outer.iter().map(|q| [side * q[0], q[1], z]).collect();
            r.extend(
                outer
                    .iter()
                    .rev()
                    .map(|q| [side * (q[0] - 0.025), q[1], z]),
            );
            r
        })
        .collect();
    let mut bld = Builder::new();
    bld.tube(&rings);
    part(name, bld.finish())
}

/// **A slab in the top recess** over `(a, b)`: a pane (windscreen, backlight)
/// or a lid (bonnet, boot), its outer face where the skin was.
fn top_slab(p: &Profile, env: &Envelope, name: &str, a: f64, b: f64, edge: f64, depth: f64) -> ShellPart {
    let rings: Vec<Vec<[f64; 3]>> = span(p, env, a, b)
        .into_iter()
        .map(|z| {
            let l = levels(p, z);
            let k = l.x_edge - edge - SHUT_GAP;
            let xs = [k, 0.6 * k, 0.25 * k, 0.0, -0.25 * k, -0.6 * k, -k];
            let mut r: Vec<[f64; 3]> = xs
                .iter()
                .map(|x| [*x, roof_y(&l, *x) - 0.004, z])
                .collect();
            r.extend(
                xs.iter()
                    .rev()
                    .map(|x| [*x, roof_y(&l, *x) - depth + 0.004, z]),
            );
            r
        })
        .collect();
    let mut bld = Builder::new();
    bld.tube(&rings);
    part(name, bld.finish())
}

/// The body's frontmost `z` at lateral `x` (`nose` true) or its rearmost.
fn end_z(p: &Profile, x: f64, nose: bool) -> f64 {
    let (mut lo, mut hi) = (0.6f64, 1.0f64);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let z = if nose { mid } else { -mid };
        if p.body_half * plan(p, z) >= x.abs() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let z = 0.5 * (lo + hi);
    if nose {
        z
    } else {
        -z
    }
}

/// **A bumper**: a beam across the nose (or the tail) wrapped to the plan,
/// standing proud of the body.
fn bumper(p: &Profile, name: &str, nose: bool) -> ShellPart {
    let s = if nose { 1.0 } else { -1.0 };
    let zb = s * 0.93;
    let xb = p.body_half * plan(p, zb) * 1.005;
    let yb = p.bot.at(s * 0.97) - 0.02;
    let y_hi = if nose { -0.40 } else { -0.32 };
    let rings: Vec<Vec<[f64; 3]>> = (0..=24)
        .map(|i| {
            let x = -xb + 2.0 * xb * i as f64 / 24.0;
            let ze = end_z(p, x * 0.985, nose);
            let (zo, zi) = (ze + s * 0.012, ze - s * 0.05);
            let r = 0.025;
            let pts = [
                [zi, yb],
                [zo - s * r, yb],
                [zo, yb + r],
                [zo, y_hi - r],
                [zo - s * r, y_hi],
                [zi, y_hi],
            ];
            pts.iter().map(|q| [x, q[1], q[0]]).collect()
        })
        .collect();
    let mut bld = Builder::new();
    bld.tube(&rings);
    part(name, bld.finish())
}

/// A closed rounded box (a 12-point superellipse section lofted over 6
/// stations) spanning `lo..hi`, hull fractions -- lamps, mirrors, seats.
fn rounded_box(bld: &mut Builder, lo: [f64; 3], hi: [f64; 3]) {
    let c = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let h = [
        0.5 * (hi[0] - lo[0]),
        0.5 * (hi[1] - lo[1]),
        0.5 * (hi[2] - lo[2]),
    ];
    let rings: Vec<Vec<[f64; 3]>> = (0..=5)
        .map(|k| {
            let t = -1.0 + 2.0 * k as f64 / 5.0;
            let taper = 1.0 - 0.12 * t * t * t * t;
            (0..12)
                .map(|i| {
                    let th = std::f64::consts::TAU * i as f64 / 12.0;
                    let (co, si) = (pcos64(th), psin64(th));
                    let sq = |v: f64| v.signum() * v.abs().sqrt();
                    [
                        c[0] + h[0] * taper * sq(co),
                        c[1] + h[1] * taper * sq(si),
                        c[2] + h[2] * t,
                    ]
                })
                .collect()
        })
        .collect();
    bld.tube(&rings);
}

/// Two mirror-image rounded boxes (`x` mirrored) as one part.
fn pair(name: &str, lo: [f64; 3], hi: [f64; 3]) -> ShellPart {
    let mut bld = Builder::new();
    rounded_box(&mut bld, lo, hi);
    rounded_box(&mut bld, [-hi[0], lo[1], lo[2]], [-lo[0], hi[1], hi[2]]);
    part(name, bld.finish())
}

fn single(name: &str, boxes: &[([f64; 3], [f64; 3])]) -> ShellPart {
    let mut bld = Builder::new();
    for (lo, hi) in boxes {
        rounded_box(&mut bld, *lo, *hi);
    }
    part(name, bld.finish())
}

/// **The steering wheel**, in its own column frame (the rim in `XY`, the
/// column along `+Z`) at unit size: a rim, three spokes and a boss.
fn steering_wheel() -> Mesh {
    let mut bld = Builder::new();
    // The rim: a torus of revolution in XY (major 0.45, minor 0.05).
    let (nu, nv) = (28usize, 8usize);
    let mut ids = Vec::new();
    for i in 0..nu {
        let a = std::f64::consts::TAU * i as f64 / nu as f64;
        let mut row = Vec::new();
        for j in 0..nv {
            let b = std::f64::consts::TAU * j as f64 / nv as f64;
            let rr = 0.45 + 0.05 * pcos64(b);
            row.push(bld.vert([rr * pcos64(a), rr * psin64(a), 0.05 * psin64(b)]));
        }
        ids.push(row);
    }
    for i in 0..nu {
        let i2 = (i + 1) % nu;
        for j in 0..nv {
            let j2 = (j + 1) % nv;
            bld.face(&[ids[i][j], ids[i2][j], ids[i2][j2], ids[i][j2]]);
        }
    }
    // The spokes (at 90, 210, 330 degrees) and the boss.
    for deg in [90.0f64, 210.0, 330.0] {
        let a = deg.to_radians();
        let (co, si) = (pcos64(a), psin64(a));
        let rings: Vec<Vec<[f64; 3]>> = [0.10f64, 0.42]
            .iter()
            .map(|r| {
                let (cx, cy) = (r * co, r * si);
                let (px, py) = (-si, co);
                [(-0.05, -0.02), (0.05, -0.02), (0.05, 0.02), (-0.05, 0.02)]
                    .iter()
                    .map(|(u, w)| [cx + px * u, cy + py * u, *w])
                    .collect()
            })
            .collect();
        bld.tube(&rings);
    }
    let boss: Vec<Vec<[f64; 3]>> = [-0.08f64, 0.10]
        .iter()
        .map(|z| {
            (0..12)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 12.0;
                    [0.13 * pcos64(a), 0.13 * psin64(a), *z]
                })
                .collect()
        })
        .collect();
    bld.tube(&boss);
    bld.finish()
}

/// **A tyre** in the primitive cylinder's frame (axis `+Y`, radius 0.5,
/// height 1): a lathe of a tyre's section, [`TYRE_FILL`] of the height wide.
pub fn tyre_mesh() -> Mesh {
    let w = 0.5 * TYRE_FILL;
    // (radius, y) around the section, outside first.
    let prof = [
        (0.30, -0.80 * w),
        (0.40, -0.98 * w),
        (0.47, -0.92 * w),
        (0.50, -0.60 * w),
        (0.50, 0.60 * w),
        (0.47, 0.92 * w),
        (0.40, 0.98 * w),
        (0.30, 0.80 * w),
    ];
    lathe(&prof, 32)
}

/// **A rim** in the same frame: a barrel, a recessed face and five spokes,
/// symmetric about the tyre's mid-plane so it reads from either flank.
pub fn rim_mesh() -> Mesh {
    let w = 0.5 * TYRE_FILL;
    let prof = [
        (0.10, -0.12 * w),
        (0.31, -0.20 * w),
        (0.31, -0.78 * w),
        (0.33, -0.80 * w),
        (0.33, 0.80 * w),
        (0.31, 0.78 * w),
        (0.31, 0.20 * w),
        (0.10, 0.12 * w),
    ];
    lathe(&prof, 32)
}

/// A closed surface of revolution about `Y` of a closed section `prof`
/// (`(radius, y)` pairs, a loop).
fn lathe(prof: &[(f64, f64)], seg: usize) -> Mesh {
    let rings: Vec<Vec<[f64; 3]>> = (0..seg)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / seg as f64;
            prof.iter()
                .map(|(r, y)| [r * pcos64(a), *y, r * psin64(a)])
                .collect()
        })
        .collect();
    let mut bld = Builder::new();
    let n = prof.len();
    let ids: Vec<Vec<VertId>> = rings
        .iter()
        .map(|r| r.iter().map(|q| bld.vert(*q)).collect())
        .collect();
    for i in 0..seg {
        let i2 = (i + 1) % seg;
        for j in 0..n {
            let j2 = (j + 1) % n;
            bld.face(&[ids[i][j], ids[i][j2], ids[i2][j2], ids[i2][j]]);
        }
    }
    let mut m = bld.finish();
    if signed_volume(&m) < 0.0 {
        m = flipped(&m);
    }
    m
}

/// A mesh's signed volume (positive = normals out).
pub fn signed_volume(m: &Mesh) -> f64 {
    let mut v = 0.0;
    for f in m.face_ids() {
        let vs = m.face_verts(f).expect("live");
        let ps: Vec<_> = vs.iter().map(|q| m.position(*q).expect("live")).collect();
        for k in 1..ps.len() - 1 {
            v += ps[0].dot(ps[k].cross(ps[k + 1])) / 6.0;
        }
    }
    v
}

fn flipped(m: &Mesh) -> Mesh {
    let mut bld = Builder::new();
    let mut map = std::collections::BTreeMap::new();
    for v in m.vert_ids() {
        let q = m.position(v).expect("live");
        map.insert(v, bld.vert([q.x, q.y, q.z]));
    }
    for f in m.face_ids() {
        let vs: Vec<VertId> = m
            .face_verts(f)
            .expect("live")
            .iter()
            .rev()
            .map(|v| map[v])
            .collect();
        bld.face(&vs);
    }
    bld.finish()
}

/// The typical row's half-extents, metres -- the island row's -- for parts
/// sized in metres (a seat, a mirror, a wheel) and placed in fractions.
fn design_half(shell: Shell) -> [f64; 3] {
    let d = crate::vehicle::island_vehicles()
        .get(shell.island_row())
        .copied()
        .expect("the island row");
    [d.half_extents.x, d.half_extents.y, d.half_extents.z]
}

/// The shell's seats: `(driver z, cushion top y)`, hull fractions.
fn seat_plan(shell: Shell, doors: &[(f64, f64)]) -> (f64, f64) {
    let (a, b) = doors[0];
    let z = lerp(a, b, if doors.len() > 1 { 0.34 } else { 0.30 });
    let _ = shell;
    (z, inf_ecs::boarding::SEAT_CUSHION_FRAC_Y)
}

/// **Every part of a shell**, hull fractions: doors, panes, lids, bumpers,
/// lamps, grille, mirrors, seats, the dashboard and the steering wheel (the
/// cruiser adds its push bar).
pub fn shell_parts(shell: Shell) -> Vec<ShellPart> {
    let p = profile(shell);
    let env = envelope(shell);
    let dh = design_half(shell);
    let m = |metres: f64, axis: usize| metres / dh[axis];
    let doors = door_ranges(&p, &env);
    let mut out = Vec::new();
    // The doors and their panes: left is -X, right +X (the driver's).
    let names: &[(&str, &str)] = if doors.len() > 1 {
        &[("fl", "rl"), ("fr", "rr")]
    } else {
        &[("l", ""), ("r", "")]
    };
    for (si, side) in [-1.0f64, 1.0].into_iter().enumerate() {
        for (di, (a, b)) in doors.iter().enumerate() {
            let tag = if di == 0 { names[si].0 } else { names[si].1 };
            out.push(door(&p, &env, &format!("door_{tag}"), side, *a, *b));
            out.push(side_glass(
                &p,
                &env,
                &format!("glass_side_{tag}"),
                side,
                *a,
                *b,
            ));
        }
    }
    out.push(top_slab(
        &p,
        &env,
        "glass_windscreen",
        p.roof_front + EPS,
        p.cowl - EPS,
        0.11,
        GLASS_DEPTH,
    ));
    if p.backlight {
        out.push(top_slab(
            &p,
            &env,
            "glass_rear",
            p.deck_front + EPS,
            p.roof_rear - EPS,
            0.11,
            GLASS_DEPTH,
        ));
    }
    out.push(top_slab(
        &p,
        &env,
        "bonnet",
        p.cowl + 2.0 * SHUT_GAP + EPS,
        0.955 - EPS,
        0.05,
        LID_DEPTH,
    ));
    if p.boot {
        out.push(top_slab(
            &p,
            &env,
            "boot",
            -0.965 + EPS,
            p.deck_front - 2.0 * SHUT_GAP - EPS,
            0.05,
            LID_DEPTH,
        ));
    }
    out.push(bumper(&p, "bumper_front", true));
    out.push(bumper(&p, "bumper_rear", false));
    // The lamps: a pod at each nose corner and each tail corner, sat on the
    // body's own end surface.
    let zf = end_z(&p, 0.72, true);
    out.push(pair(
        "lamp_front",
        [0.50, -0.27, zf - 0.035],
        [0.86, -0.15, zf + 0.004],
    ));
    let zr = end_z(&p, 0.72, false);
    out.push(pair(
        "lamp_rear",
        [0.52, -0.20, zr - 0.012],
        [0.90, if p.bed.is_some() { 0.02 } else { 0.06 }, zr + 0.03],
    ));
    let zg = end_z(&p, 0.0, true);
    out.push(single(
        "grille",
        &[([-0.44, -0.37, zg - 0.03], [0.44, -0.20, zg + 0.004])],
    ));
    // The mirrors: a head on a stalk at the front door's leading corner.
    let (_, df) = doors[0];
    let lm = levels(&p, df);
    let mx = lm.x_body;
    out.push(pair(
        "mirror",
        [mx - 0.02, lm.belt + 0.02, df - m(0.05, 2)],
        [mx + m(0.20, 0), lm.belt + m(0.13, 1), df - 0.004],
    ));
    // The seats: two fronts and a bench (a cushion box whose TOP is where
    // the boarding sockets seat a pelvis, with its backrest above it).
    let (sz, cushion) = seat_plan(shell, &doors);
    let sx = 0.45;
    let (hx, hz) = (m(0.24, 0), m(0.25, 2));
    // The part's BOX is the cushion alone -- its top face is what
    // `inf_ecs::boarding::sockets_of` reads -- and the backrest rises out of it.
    let seat = |name: &str, lo: [f64; 3], hi: [f64; 3], back_h: f64| {
        let mut sp = single(
            name,
            &[
                (lo, hi),
                (
                    [lo[0], hi[1] - m(0.08, 1), lo[2] - m(0.12, 2)],
                    [hi[0], hi[1] + back_h, lo[2]],
                ),
            ],
        );
        let r4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
        sp.centre = [
            r4(0.5 * (lo[0] + hi[0])),
            r4(0.5 * (lo[1] + hi[1])),
            r4(0.5 * (lo[2] + hi[2])),
        ];
        sp.half = [
            r4(0.5 * (hi[0] - lo[0])),
            r4(0.5 * (hi[1] - lo[1])),
            r4(0.5 * (hi[2] - lo[2])),
        ];
        sp
    };
    for (name, x) in [("seat_fr", sx), ("seat_fl", -sx)] {
        out.push(seat(
            name,
            [x - hx, cushion - m(0.10, 1), sz - hz],
            [x + hx, cushion, sz + hz],
            m(0.62, 1),
        ));
    }
    if doors.len() > 1 {
        let rz = lerp(doors[1].0, doors[1].1, 0.45);
        out.push(seat(
            "seat_rear",
            [-0.72, cushion - m(0.10, 1) + 0.02, rz - hz],
            [0.72, cushion + 0.02, rz + hz],
            m(0.56, 1),
        ));
    }
    // The dashboard: across the cabin at the front aperture's wall, with a
    // binnacle over the driver's side.
    let dz = doors[0].1;
    out.push(single(
        "dash",
        &[
            ([-0.90, -0.16, dz - m(0.16, 2)], [0.90, lm.belt - 0.02, dz + 0.01]),
            (
                [sx - 0.18, lm.belt - 0.04, dz - m(0.10, 2)],
                [sx + 0.18, lm.belt + m(0.04, 1), dz],
            ),
        ],
    ));
    // The steering wheel: on the driver's side, a hand ahead of the seat.
    let hub_z = sz + m(inf_ecs::boarding::MAX_RIM_M + 0.12, 2);
    let hub_y = cushion + m(0.36, 1);
    let rim = 0.18;
    let mut hub = part("hub", steering_wheel());
    hub.centre = [sx, hub_y, hub_z];
    let r4 = |v: f64| (v * 10_000.0).round() / 10_000.0;
    hub.half = [r4(m(rim, 0)), r4(m(rim, 1)), r4(m(0.25, 2))];
    out.push(hub);
    if shell == Shell::Cruiser {
        let zp = end_z(&p, 0.0, true);
        out.push(single(
            "bumper_push",
            &[
                ([-0.42, -0.86, zp + 0.03], [-0.34, -0.24, zp + 0.06]),
                ([0.34, -0.86, zp + 0.03], [0.42, -0.24, zp + 0.06]),
                ([-0.46, -0.66, zp + 0.04], [0.46, -0.58, zp + 0.08]),
                ([-0.46, -0.36, zp + 0.04], [0.46, -0.28, zp + 0.08]),
            ],
        ));
    }
    out
}

/// **The cruiser's light bar**, drawn at the LIVERY's own part box
/// (`crate::vehicle::SEDAN_BAR`: the livery adds the part, the shell only
/// supplies its mesh), in that box's unit frame.
fn light_bar_mesh() -> Mesh {
    let mut bld = Builder::new();
    rounded_box(&mut bld, [-0.5, -0.5, -0.5], [0.5, 0.05, 0.5]);
    rounded_box(&mut bld, [-0.46, 0.0, -0.42], [-0.04, 0.5, 0.42]);
    rounded_box(&mut bld, [0.04, 0.0, -0.42], [0.46, 0.5, 0.42]);
    bld.finish()
}

// ── the files ────────────────────────────────────────────────────────────────

/// Map a mesh from hull fractions into its part's unit box.
fn into_unit(m: &Mesh, centre: [f64; 3], half: [f64; 3]) -> Mesh {
    let mut bld = Builder::new();
    let mut map = std::collections::BTreeMap::new();
    for v in m.vert_ids() {
        let q = m.position(v).expect("live");
        let u = [
            (q.x - centre[0]) / (2.0 * half[0]),
            (q.y - centre[1]) / (2.0 * half[1]),
            (q.z - centre[2]) / (2.0 * half[2]),
        ];
        map.insert(v, bld.vert(u));
    }
    for f in m.face_ids() {
        let vs: Vec<VertId> = m
            .face_verts(f)
            .expect("live")
            .iter()
            .map(|v| map[v])
            .collect();
        bld.face(&vs);
    }
    // The sharp edges carry across by angle, which a per-axis scale can change;
    // re-derive them in the frame the mesh is drawn from.
    bld.finish()
}

fn export(m: &Mesh) -> inf_mesh::MeshAsset {
    to_mesh_asset(
        m,
        &ExportOptions {
            normals: NormalPolicy::Recompute,
            optimize: false,
        },
    )
    .0
}

/// **Every shell mesh**, in file order: per shell its body (at
/// `art_body_guid`), its tyre (`art_wheel_guid(k, 0)`: all four wheels draw
/// it), its rim (`art_part_guid(k, "rim")`), then every part at
/// `art_part_guid` -- and the cruiser's light bar at the livery part's name.
pub fn shell_meshes() -> Vec<HeroMesh> {
    let mut out = Vec::new();
    for shell in Shell::ALL {
        let k = shell.art();
        let name = shell.key();
        let body = body_mesh(shell);
        out.push(HeroMesh {
            file: format!("{name}_body.inf_mesh"),
            guid: inf_ecs::roster::art_body_guid(k),
            asset: export(&into_unit(&body, [0.0; 3], [1.0; 3])),
        });
        out.push(HeroMesh {
            file: format!("{name}_wheel0.inf_mesh"),
            guid: inf_ecs::roster::art_wheel_guid(k, 0),
            asset: export(&tyre_mesh()),
        });
        out.push(HeroMesh {
            file: format!("{name}_rim.inf_mesh"),
            guid: inf_ecs::roster::art_part_guid(k, inf_ecs::vehicle::SHELL_RIM_PART),
            asset: export(&rim_mesh()),
        });
        for sp in shell_parts(shell) {
            let unit = if sp.name == "hub" {
                // Already in its column frame at unit size.
                sp.mesh
            } else {
                into_unit(&sp.mesh, sp.centre, sp.half)
            };
            out.push(HeroMesh {
                file: format!("{name}_{}.inf_mesh", sp.name),
                guid: inf_ecs::roster::art_part_guid(k, &sp.name),
                asset: export(&unit),
            });
        }
        if shell == Shell::Cruiser {
            out.push(HeroMesh {
                file: format!("{name}_{}.inf_mesh", inf_ecs::dispatch::LIGHT_BAR_PART),
                guid: inf_ecs::roster::art_part_guid(k, inf_ecs::dispatch::LIGHT_BAR_PART),
                asset: export(&light_bar_mesh()),
            });
        }
    }
    out
}

/// Every file the library writes, payloads and sidecars, sorted.
pub fn shell_files() -> Vec<String> {
    let mut v: Vec<String> = shell_meshes()
        .into_iter()
        .flat_map(|m| [m.file.clone(), format!("{}.toml", m.file)])
        .collect();
    v.sort();
    v
}

/// The committed folder.
pub fn vehicle_shells_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../samples")
        .join(VEHICLE_SHELLS_FOLDER)
}

/// **Write the library** -- every shell mesh and its sidecar.
pub fn write_vehicle_shells(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for m in shell_meshes() {
        let bytes = inf_asset::encode(&m.asset).map_err(|e| format!("encode {}: {e}", m.file))?;
        let path = dir.join(&m.file);
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(m.guid),
            inf_asset::AssetKind::Mesh,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .map_err(|e| format!("write the sidecar for {}: {e}", m.file))?;
    }
    Ok(())
}

/// **The art-table rows the shells ARE** -- the `[[art]]` blocks
/// `vehicle_art.toml` must carry, generated: every part's name and box.
pub fn shell_art_toml() -> String {
    let mut s = String::new();
    for shell in Shell::ALL {
        s.push_str(&format!(
            "[[art]]\nkey = \"{}\"\nsource = \"inf-dcc\"\npack = \"ours\"\nfacing = \"+X\"\nshell = true\n",
            shell.key()
        ));
        for sp in shell_parts(shell) {
            s.push_str(&format!(
                "[[art.parts]]\nname = \"{}\"\ncentre = [{:.4}, {:.4}, {:.4}]\nhalf = [{:.4}, {:.4}, {:.4}]\n",
                sp.name,
                sp.centre[0],
                sp.centre[1],
                sp.centre[2],
                sp.half[0],
                sp.half[1],
                sp.half[2]
            ));
        }
        s.push('\n');
    }
    s
}

/// A shell body's Guid (the art table's rule), for a caller that holds none.
pub fn shell_body_guid(shell: Shell) -> Uuid {
    inf_ecs::roster::art_body_guid(shell.art())
}

/// **What a gate measures a shell by** (wave VEH3f.2b) -- closedness, the
/// arch rays and the outline against the box family, over TRIANGLES in metres,
/// so the unit tests below run them on the generator's output and
/// `veh3f2b_gate` on the committed bytes.
pub mod measure {
    /// One triangle, three positions.
    pub type Tri = [[f64; 3]; 3];

    /// Welded-edge closedness of an exported asset: every edge (keyed by its
    /// two positions' bits) on exactly two triangles.
    pub fn closed(asset: &inf_mesh::MeshAsset) -> (usize, usize) {
        let mut edges: std::collections::BTreeMap<([u32; 3], [u32; 3]), usize> =
            Default::default();
        for sm in &asset.submeshes {
            let key = |i: u32| sm.vertices[i as usize].position.map(f32::to_bits);
            for t in sm.indices.chunks(3) {
                for k in 0..3 {
                    let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
                    *edges.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
        }
        let bad = edges.values().filter(|n| **n != 2).count();
        (edges.len(), bad)
    }

    /// An exported asset's triangles, each vertex mapped by `f`.
    pub fn tris_of_asset(a: &inf_mesh::MeshAsset, f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<Tri> {
        let mut out = Vec::new();
        for sm in &a.submeshes {
            for t in sm.indices.chunks(3) {
                let v = |i: u32| {
                    let p = sm.vertices[i as usize].position;
                    f([p[0] as f64, p[1] as f64, p[2] as f64])
                };
                out.push([v(t[0]), v(t[1]), v(t[2])]);
            }
        }
        out
    }

    /// Moller-Trumbore: does the ray `o + t d` (`t > 0`) cross `tri`?
    pub fn ray_hits(o: [f64; 3], d: [f64; 3], tri: &Tri) -> bool {
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (e1, e2) = (sub(tri[1], tri[0]), sub(tri[2], tri[0]));
        let pv = cross(d, e2);
        let det = dot(e1, pv);
        if det.abs() < 1e-14 {
            return false;
        }
        let tv = sub(o, tri[0]);
        let u = dot(tv, pv) / det;
        if !(0.0..=1.0).contains(&u) {
            return false;
        }
        let qv = cross(tv, e1);
        let v = dot(d, qv) / det;
        if v < 0.0 || u + v > 1.0 {
            return false;
        }
        dot(e2, qv) / det > 1e-9
    }

    /// Rasterise triangles projected on two axes, 1 cm cells, over `lo..lo+n`.
    fn raster(tris: &[Tri], ax: (usize, usize), lo: [f64; 2], n: [usize; 2]) -> Vec<bool> {
        let cell = 0.01;
        let mut g = vec![false; n[0] * n[1]];
        for t in tris {
            let p: Vec<[f64; 2]> = t.iter().map(|v| [v[ax.0], v[ax.1]]).collect();
            let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
            if area.abs() < 1e-12 {
                continue;
            }
            let (mut a0, mut a1, mut b0, mut b1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for q in &p {
                a0 = a0.min(q[0]);
                a1 = a1.max(q[0]);
                b0 = b0.min(q[1]);
                b1 = b1.max(q[1]);
            }
            let i0 = ((a0 - lo[0]) / cell).floor().max(0.0) as usize;
            let i1 = (((a1 - lo[0]) / cell).ceil().max(0.0) as usize).min(n[0]);
            let j0 = ((b0 - lo[1]) / cell).floor().max(0.0) as usize;
            let j1 = (((b1 - lo[1]) / cell).ceil().max(0.0) as usize).min(n[1]);
            for i in i0..i1 {
                for j in j0..j1 {
                    let (x, y) = (
                        lo[0] + (i as f64 + 0.5) * cell,
                        lo[1] + (j as f64 + 0.5) * cell,
                    );
                    let e = |a: [f64; 2], b: [f64; 2]| {
                        (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
                    };
                    let (e0, e1, e2) = (e(p[0], p[1]), e(p[1], p[2]), e(p[2], p[0]));
                    if (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
                    {
                        g[i * n[1] + j] = true;
                    }
                }
            }
        }
        g
    }

    /// The side view WITH DEPTH: per 1 cm `(z, y)` cell, the largest `x` any
    /// triangle reaches there (the surface a viewer on the `+X` flank sees
    /// first), or `None`.
    fn side_depth(tris: &[Tri], lo: [f64; 2], n: [usize; 2]) -> Vec<Option<f64>> {
        let cell = 0.01;
        let mut g: Vec<Option<f64>> = vec![None; n[0] * n[1]];
        for t in tris {
            let p: Vec<[f64; 3]> = t.iter().map(|v| [v[2], v[1], v[0]]).collect();
            let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
            if area.abs() < 1e-12 {
                continue;
            }
            let (mut a0, mut a1, mut b0, mut b1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for q in &p {
                a0 = a0.min(q[0]);
                a1 = a1.max(q[0]);
                b0 = b0.min(q[1]);
                b1 = b1.max(q[1]);
            }
            let i0 = ((a0 - lo[0]) / cell).floor().max(0.0) as usize;
            let i1 = (((a1 - lo[0]) / cell).ceil().max(0.0) as usize).min(n[0]);
            let j0 = ((b0 - lo[1]) / cell).floor().max(0.0) as usize;
            let j1 = (((b1 - lo[1]) / cell).ceil().max(0.0) as usize).min(n[1]);
            for i in i0..i1 {
                for j in j0..j1 {
                    let (x, y) = (
                        lo[0] + (i as f64 + 0.5) * cell,
                        lo[1] + (j as f64 + 0.5) * cell,
                    );
                    let w0 = ((p[1][0] - x) * (p[2][1] - y) - (p[2][0] - x) * (p[1][1] - y)) / area;
                    let w1 = ((p[2][0] - x) * (p[0][1] - y) - (p[0][0] - x) * (p[2][1] - y)) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9 {
                        continue;
                    }
                    let d = w0 * p[0][2] + w1 * p[1][2] + w2 * p[2][2];
                    let c = &mut g[i * n[1] + j];
                    if c.is_none_or(|e| d > e) {
                        *c = Some(d);
                    }
                }
            }
        }
        g
    }

    /// **The all-box car**: the family's part boxes (seats aside) at the row's
    /// half-extents `h`, metres.
    pub fn box_family(def: &inf_ecs::vehicle::VehicleDef) -> Vec<Tri> {
        let h = [def.half_extents.x, def.half_extents.y, def.half_extents.z];
        let cube = inf_dcc::to_mesh_asset(
            &inf_dcc::cube(1.0),
            &inf_dcc::ExportOptions {
                normals: inf_dcc::NormalPolicy::Recompute,
                optimize: false,
            },
        )
        .0;
        let mut boxes = Vec::new();
        for p in def.body.parts() {
            if p.kind == inf_ecs::vehicle::BodyPartKind::Seat {
                continue;
            }
            let c = [p.centre.x * h[0], p.centre.y * h[1], p.centre.z * h[2]];
            let hh = [p.half.x * h[0], p.half.y * h[1], p.half.z * h[2]];
            boxes.extend(tris_of_asset(&cube, |q| {
                [
                    c[0] + q[0] * 2.0 * hh[0],
                    c[1] + q[1] * 2.0 * hh[1],
                    c[2] + q[2] * 2.0 * hh[2],
                ]
            }));
        }
        boxes
    }

    /// **The side and top outlines of `car` against `boxes`**, `(side, top)`
    /// as symmetric difference over union on a 1 cm raster over the row's
    /// half-extents `h` -- the measurement VEH3f's panels scored 2.4-8.7 % on.
    pub fn outline_delta(car: &[Tri], boxes: &[Tri], h: [f64; 3]) -> (f64, f64) {
        let lo = [-h[0] * 1.3, -h[1] * 1.3, -h[2] * 1.3];
        let n = [
            (h[0] * 2.6 / 0.01) as usize,
            (h[1] * 2.6 / 0.01) as usize,
            (h[2] * 2.6 / 0.01) as usize,
        ];
        let diff = |ax: (usize, usize)| {
            let (l, nn) = ([lo[ax.0], lo[ax.1]], [n[ax.0], n[ax.1]]);
            let (a, b) = (raster(boxes, ax, l, nn), raster(car, ax, l, nn));
            let x = a.iter().zip(&b).filter(|(p, q)| p != q).count() as f64;
            let u = a.iter().zip(&b).filter(|(p, q)| **p || **q).count() as f64;
            x / u.max(1.0)
        };
        (diff((2, 1)), diff((0, 2)))
    }

    /// **The FLANK outline** of `car` against `boxes`: the side view's cells
    /// whose first-seen surface is the flank skin (`x >= 0.75 hw`), symmetric
    /// difference over union. A cut arch shows here -- the first surface seen
    /// through it is the wheel well's inner wall -- where the plain side
    /// outline is filled by that same wall.
    pub fn flank_delta(car: &[Tri], boxes: &[Tri], h: [f64; 3]) -> f64 {
        let lo = [-h[2] * 1.3, -h[1] * 1.3];
        let n = [(h[2] * 2.6 / 0.01) as usize, (h[1] * 2.6 / 0.01) as usize];
        let skin = 0.75 * h[0];
        let flank = |t: &[Tri]| -> Vec<bool> {
            side_depth(t, lo, n)
                .into_iter()
                .map(|d| d.is_some_and(|x| x >= skin))
                .collect()
        };
        let (a, b) = (flank(boxes), flank(car));
        let x = a.iter().zip(&b).filter(|(p, q)| p != q).count() as f64;
        let u = a.iter().zip(&b).filter(|(p, q)| **p || **q).count() as f64;
        x / u.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::measure::*;

    fn tris_of(m: &Mesh, f: impl Fn([f64; 3]) -> [f64; 3]) -> Vec<Tri> {
        tris_of_asset(&export(m), f)
    }

    /// The generator's car at the island row's size, metres (the hub aside
    /// when `hub` is false).
    fn car_tris(shell: Shell, hub: bool) -> (Vec<Tri>, Vec<Tri>, [f64; 3]) {
        let def = crate::vehicle::island_vehicles()
            .get(shell.island_row())
            .copied()
            .expect("the row");
        let h = [def.half_extents.x, def.half_extents.y, def.half_extents.z];
        let m = |p: [f64; 3]| [p[0] * h[0], p[1] * h[1], p[2] * h[2]];
        let mut car = tris_of(&body_mesh(shell), m);
        for sp in shell_parts(shell) {
            if sp.name == "hub" && !hub {
                continue;
            }
            car.extend(tris_of(&sp.mesh, m));
        }
        (car, box_family(&def), h)
    }

    fn outline_delta_of(shell: Shell) -> (f64, f64) {
        let (car, boxes, h) = car_tris(shell, false);
        outline_delta(&car, &boxes, h)
    }

    fn flank_delta_of(shell: Shell) -> f64 {
        let (car, boxes, h) = car_tris(shell, true);
        flank_delta(&car, &boxes, h)
    }

    /// **Every shell is closed** -- the body and every part, each welded edge
    /// of the exported asset on exactly two triangles. Mutation -> red: a
    /// tube's end cap skipped (`Builder::tube` without its fans).
    #[test]
    fn every_shell_body_and_part_is_closed() {
        let mut n = 0usize;
        for shell in Shell::ALL {
            let (_, bad) = closed(&export(&body_mesh(shell)));
            assert_eq!(bad, 0, "{shell:?}'s body has {bad} open edge(s)");
            for sp in shell_parts(shell) {
                let (_, bad) = closed(&export(&sp.mesh));
                assert_eq!(bad, 0, "{shell:?} {}: {bad} open edge(s)", sp.name);
                n += 1;
            }
            assert!(signed_volume(&body_mesh(shell)) > 0.0, "{shell:?} inside out");
        }
        assert!(n >= 100, "only {n} parts swept");
    }

    /// **The arches are CUT for every row that wears a shell** -- from each
    /// settled wheel centre a ray outward along the axle exits through air
    /// (no body triangle crossed), on both flanks, front and rear. The
    /// VEH3f panels fail this at every wheel (their lower slab spans the
    /// wheel). Mutation -> red: the arch feature disabled in `features`.
    #[test]
    fn a_ray_from_every_wheel_centre_exits_through_air() {
        let mut rays = 0usize;
        for shell in Shell::ALL {
            let body = tris_of(&body_mesh(shell), |p| p);
            let rows = shell_rows(shell);
            assert!(!rows.is_empty(), "{shell:?} is worn by no row");
            for (id, def) in rows {
                let (zf, zr, y, _, _, _) = tyre_fracs(&def);
                let x = def.half_track_m / def.half_extents.x;
                for z in [zf, zr] {
                    for side in [-1.0f64, 1.0] {
                        let hits = body
                            .iter()
                            .filter(|t| ray_hits([side * x, y, z], [side, 0.0, 0.0], t))
                            .count();
                        assert_eq!(
                            hits, 0,
                            "{shell:?} on `{id}`: the wheel at z {z:.3} x {:.3} is under {hits} \
                             body triangle(s)",
                            side * x
                        );
                        rays += 1;
                    }
                }
            }
        }
        assert!(rays >= 200, "only {rays} wheels swept");
    }

    /// **A shell's side outline is a car's, not the box family's** -- >= 25 %
    /// symmetric difference over union on the side view (the VEH3f panels:
    /// 3.6-8.7 %). Mutation -> red: the body lofted with every feature off
    /// and the profile's curves flattened to the hull box.
    #[test]
    fn a_shells_side_outline_is_not_the_box_family() {
        for shell in Shell::ALL {
            let (side, top) = outline_delta_of(shell);
            let flank = flank_delta_of(shell);
            println!(
                "SHELL {shell:?}: outline vs the box family side {:.1} % top {:.1} %; flank {:.1} %",
                100.0 * side,
                100.0 * top,
                100.0 * flank
            );
            assert!(
                side >= 0.12,
                "{shell:?}: the side outline is only {:.1} % off the box car",
                100.0 * side
            );
            assert!(
                flank >= 0.25,
                "{shell:?}: the flank outline is only {:.1} % off the box car",
                100.0 * flank
            );
        }
    }

    /// **The committed art table carries exactly the parts the generator cuts**
    /// -- the TOML a row names is the geometry the files hold.
    #[test]
    fn the_committed_art_table_carries_the_shells_parts() {
        let text = inf_ecs::vehicle_art::VEHICLE_ART_TOML;
        let at = text
            .find("[[art]]\nkey = \"shell_sedan\"")
            .expect("the shell rows are in vehicle_art.toml");
        assert_eq!(
            text[at..].replace("\r\n", "\n").trim_end(),
            shell_art_toml().trim_end(),
            "vehicle_art.toml's shell rows drifted from the generator -- run \
             `cargo test -p inf-editor-core write_the_shells -- --ignored` and \
             paste its TOML"
        );
    }

    /// Write the library alone (the full `INF_BLESS_SAMPLES` run rewrites it
    /// with everything else): `cargo test write_the_shells -- --ignored`.
    #[test]
    #[ignore]
    fn write_the_shells() {
        write_vehicle_shells(&vehicle_shells_dir()).expect("write the shells");
        for m in shell_meshes() {
            let n = std::fs::metadata(vehicle_shells_dir().join(&m.file))
                .map(|x| x.len())
                .unwrap_or(0);
            println!("{:40} {n:>8} B", m.file);
        }
        print!("{}", shell_art_toml());
    }

    /// MEASURE FIRST (brief item 3): every row that draws a seat -- the drawn
    /// driver cushion's top face vs the fraction rule's height, metres.
    #[test]
    #[ignore]
    fn probe_the_drawn_cushions() {
        let mut rows: Vec<(String, VehicleDef)> = inf_ecs::roster::roster()
            .0
            .iter()
            .map(|(k, d)| (k.clone(), *d))
            .collect();
        rows.extend(crate::vehicle::island_vehicles().0);
        for (id, def) in rows {
            let parts: &[inf_ecs::vehicle::BodyPart] = match def.art {
                Some(k) if !k.parts().is_empty() => k.parts(),
                _ => def.body.parts(),
            };
            let driver = parts
                .iter()
                .filter(|p| p.kind == inf_ecs::vehicle::BodyPartKind::Seat)
                .filter(|p| p.centre.x >= -1e-9)
                .max_by(|a, b| a.centre.x.total_cmp(&b.centre.x));
            let Some(d) = driver else { continue };
            let hy = def.half_extents.y;
            let top = (d.centre.y + d.half.y) * hy;
            let rule = inf_ecs::boarding::SEAT_CUSHION_FRAC_Y * hy;
            println!(
                "CUSHION {id:30} {:?} top {top:+.3} rule {rule:+.3} delta {:+.3} m",
                def.body,
                top - rule
            );
        }
    }

    #[test]
    #[ignore]
    fn probe_the_shells() {
        for shell in Shell::ALL {
            let env = envelope(shell);
            println!(
                "{shell:?}: front z {:.3} y {:.3} rz {:.3} ry {:.3} top {:.3}; x_in {:.3}; rows {}",
                env.front.z,
                env.front.y,
                env.front.rz,
                env.front.ry,
                env.front.y + env.front.ry,
                env.x_in,
                env.rows
            );
            for (id, def) in shell_rows(shell) {
                let (zf, _, y, rz, ry, xi) = tyre_fracs(&def);
                if id == shell.island_row() {
                    let h = def.half_extents;
                    println!(
                        "    ROW {id}: h [{:.4}, {:.4}, {:.4}] r {:.4} track {:.4} wb {:.4} wheel_y {:.4}",
                        h.x,
                        h.y,
                        h.z,
                        def.wheel_radius_m,
                        def.half_track_m,
                        def.half_wheelbase_m,
                        y * h.y
                    );
                }
                println!(
                    "    {id:28} zf {zf:.3} y {y:.3} rz {rz:.3} ry {ry:.3} top {:.3} x_in {xi:.3}",
                    y + ry
                );
            }
            let body = export(&body_mesh(shell));
            let (e, bad) = closed(&body);
            let v: usize = body.submeshes.iter().map(|s| s.vertices.len()).sum();
            let t: usize = body.submeshes.iter().map(|s| s.indices.len() / 3).sum();
            println!("  body: {v} verts, {t} tris, {e} edges, {bad} open");
            for sp in shell_parts(shell) {
                let a = export(&sp.mesh);
                let (e, bad) = closed(&a);
                println!(
                    "  {:18} c={:?} h={:?} edges {e} open {bad}",
                    sp.name, sp.centre, sp.half
                );
            }
        }
    }
}
