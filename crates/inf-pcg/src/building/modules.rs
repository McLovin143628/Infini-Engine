//! **A building module's own geometry** (island wave I8b) — the day the seven
//! palettes stopped drawing cubes.
//!
//! # What was actually wrong, and it is not what the ledger said
//!
//! Wave I8a recorded the settlements as *"365 545 wall-sized opaque boxes"*.
//! They were not wall-sized. [`crate::scatter::PcgInstance::scale`] — and
//! its ECS mirror — is **one uniform `f64`**, and every module the assembler
//! places carries `scale: 1.0`, so a 0.12 × 3.5 × 1.5 m curtain-wall panel, a
//! 10 × 0.2 × 10 m floor slab and a 0.9 m desk all drew as the **same one-metre
//! cube**. The colliders were right the whole time; a building has never once
//! been *drawn* at the dimensions it is *built* at.
//!
//! That is why this module ships two things and not one:
//!
//! * an [`extent`](crate::scatter::PcgInstance::extent) on the instance — the
//!   half-extents of the box the module actually occupies, so the drawn thing is
//!   the same size as the solid thing;
//! * a **mesh** per palette module, so the box the module occupies is drawn as a
//!   framed panel, a glazed opening, a fascia'd slab or a legged desk rather than
//!   as a rectangular prism.
//!
//! # Unit space, and why the meshes are proportional
//!
//! Every mesh here is authored in the **unit box `[-0.5, 0.5]³`** and is scaled
//! onto its instance's extent at projection. Two consequences, both deliberate:
//!
//! * a module's mesh is a function of its *shape family* and nothing else, so
//!   there are nineteen meshes rather than one per palette entry;
//! * every feature is **proportional** — a frame rail is a fraction of the
//!   panel, never a fixed 40 mm — because the same mesh is stretched onto a
//!   0.3 m mullion and a 12 m slab. A fixed-size chamfer would be invisible on
//!   one and a metre deep on the other.
//!
//! # Boxes, composed
//!
//! Every shape is a small union of axis-aligned boxes with flat normals: two to
//! eight of them, 24 to 96 triangles. No half-edge kernel (`inf-dcc` is a
//! **dev** dependency of the shipped player and this code is Ring 0 on its draw
//! path), no trigonometry (the P14 law — these vertices are a pure function of
//! constants and reach a content hash), and no boolean. The relief is in the
//! **silhouette**: a window has a frame standing proud of a recessed pane, a
//! desk has legs with air between them, a slab has a fascia. That is what
//! separates "real geometry, modestly" from a re-textured box, and the meshlet
//! path is not asked to carry any of it.
//!
//! **One family is not a box** (wave VEN1a): [`ModuleShape::Pole`] and
//! [`ModuleShape::Stool`] are eight-sided prisms, because a chrome dance pole's
//! entire contribution to a near-black room is one bright vertical specular
//! streak and a square post does not make one. The octagon is a table of eight
//! **literal** constants — see [`ModuleMesh::push_prism_y`] for why a `cos` call
//! is not available to geometry that reaches a content hash.
//!
//! # The GUIDs are content-derived
//!
//! A palette module names no asset — there is no `.inf_mesh` file to point at,
//! and inventing one per module would put twelve files in `samples/` to express
//! what a function already answers. [`module_mesh_guid`] mints one from the
//! shape family's own name under a private salt, which is the synthetic-guid
//! rule this repository already uses for door leaves, PCG doorways, structure
//! colliders and fracture chunks. Both hosts register the same table under the
//! same ids (`ScatterMeshes` is keyed on `u128`), so the editor's viewport and
//! the shipped player resolve one GUID to one geometry by construction.

use uuid::Uuid;

use super::palettes::archetypes;

/// Salt for [`module_mesh_guid`] — its own constant, so a module mesh can never
/// alias a door leaf, a doorway, a structure collider or an imported asset.
const MODULE_MESH_SALT: u128 = 0x6008_0200_4255_494c_444d_4f44_554c_4553;

/// One module mesh, in unit space (`[-0.5, 0.5]³`), ready for
/// `inf_render::ScatterGeometry::from_streams`.
///
/// Positions and normals are `f32` because that is what both consumers take;
/// the values themselves are exact binary fractions written as literals, so
/// there is no rounding to be platform-dependent about.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModuleMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl ModuleMesh {
    /// Triangles in the mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Append one axis-aligned box with flat per-face normals.
    ///
    /// `center` and `half` are in unit space. A degenerate half-extent is
    /// **refused** rather than emitted: a zero-thickness face is two coincident
    /// triangles, which is the non-manifold state P23 pays for elsewhere.
    fn push_box(&mut self, center: [f32; 3], half: [f32; 3]) {
        if !(half[0] > 0.0 && half[1] > 0.0 && half[2] > 0.0) {
            return;
        }
        // (normal, u axis, v axis) per face, in the order -X +X -Y +Y -Z +Z.
        //
        // **The pair is chosen so that `ê_u × ê_v == n` on every face**, which
        // is what lets one winding serve all six. The first draft used the same
        // pair for a face and its opposite and flipped the winding on the sign;
        // that is right for four faces and wrong for ±Z, because `ê_x × ê_y` is
        // `+ê_z` while `ê_z × ê_y` and `ê_x × ê_z` are both negative. The
        // symmetry a reader expects is not there, so it is written out.
        const FACES: [([f32; 3], usize, usize); 6] = [
            ([-1.0, 0.0, 0.0], 2, 1),
            ([1.0, 0.0, 0.0], 1, 2),
            ([0.0, -1.0, 0.0], 0, 2),
            ([0.0, 1.0, 0.0], 2, 0),
            ([0.0, 0.0, -1.0], 1, 0),
            ([0.0, 0.0, 1.0], 0, 1),
        ];
        for (n, ua, va) in FACES {
            let axis = if n[0] != 0.0 {
                0
            } else if n[1] != 0.0 {
                1
            } else {
                2
            };
            let sign = n[axis];
            let base = self.positions.len() as u32;
            for (du, dv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let mut p = center;
                p[axis] += sign * half[axis];
                p[ua] += du * half[ua];
                p[va] += dv * half[va];
                self.positions.push(p);
                self.normals.push(n);
            }
            // One winding, because the axis pair above already carries the
            // orientation.
            self.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    /// Append an **eight-sided prism about the Y axis** (wave VEN1a) — the one
    /// non-box primitive in this module, and the reason a chrome dance pole
    /// reads as a cylinder rather than as a square post.
    ///
    /// # No trigonometry, and that is the P14 law and not fastidiousness
    ///
    /// The octagon's directions are the eight `[±1, 0]`, `[0, ±1]` and
    /// `[±√2/2, ±√2/2]` written as **literals**. These vertices reach a
    /// `ScatterGeometry` content key and a committed level's bytes, and
    /// `f32::cos` is not bit-portable across platforms — the law this repository
    /// paid for at P14 and met again at P21.3 and P22. A table of eight
    /// constants is a pure function of nothing at all.
    ///
    /// # Eight sides, not sixteen
    ///
    /// A pole is a 60–80 mm shaft in a room lit only by practicals: its whole
    /// visual contribution is one bright vertical specular streak, and eight
    /// facets carry that at 28 triangles where sixteen would cost 60 and every
    /// family in this table has to stay under 128.
    ///
    /// `radius` is in unit space; `y0`/`y1` are the prism's ends.
    fn push_prism_y(&mut self, radius: f32, y0: f32, y1: f32) {
        /// √2/2, the octagon's diagonal component.
        const D: f32 = 0.707_106_77;
        const RING: [[f32; 2]; 8] = [
            [1.0, 0.0],
            [D, D],
            [0.0, 1.0],
            [-D, D],
            [-1.0, 0.0],
            [-D, -D],
            [0.0, -1.0],
            [D, -D],
        ];
        if !(radius > 0.0 && y1 > y0) {
            return;
        }
        let p = |k: usize, y: f32| [RING[k][0] * radius, y, RING[k][1] * radius];
        // ── the eight side quads ──
        //
        // `A(k, y0) B(k, y1) C(k+1, y1) D(k+1, y0)` — the winding whose geometric
        // normal is `(dz, 0, -dx)` for `d = p[k+1] - p[k]`, which is the outward
        // radial direction of the face's own midpoint. Worked out rather than
        // guessed, because the box table above records what guessing the
        // symmetry cost the first time.
        for k in 0..8 {
            let j = (k + 1) % 8;
            let n = {
                let (x, z) = (RING[k][0] + RING[j][0], RING[k][1] + RING[j][1]);
                // The midpoint direction, normalized by its own length — which
                // for two adjacent octagon vertices is a constant, but is
                // written as the division so a different `RING` stays correct.
                let l = (x * x + z * z).sqrt();
                [x / l, 0.0, z / l]
            };
            let base = self.positions.len() as u32;
            for v in [p(k, y0), p(k, y1), p(j, y1), p(j, y0)] {
                self.positions.push(v);
                self.normals.push(n);
            }
            self.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        // ── the two caps, as fans from vertex 0 ──
        //
        // The ring runs counter-clockwise in `(x, z)`, so the fan
        // `(0, k, k+1)` has a geometric normal of −Y: the bottom cap takes it
        // as written and the top cap takes the reverse.
        for (y, up) in [(y0, false), (y1, true)] {
            let n = [0.0, if up { 1.0 } else { -1.0 }, 0.0];
            let base = self.positions.len() as u32;
            for k in 0..8 {
                self.positions.push(p(k, y));
                self.normals.push(n);
            }
            for k in 1..7u32 {
                if up {
                    self.indices.extend([base, base + k + 1, base + k]);
                } else {
                    self.indices.extend([base, base + k, base + k + 1]);
                }
            }
        }
    }
}

/// The nineteen shape families a palette module draws as.
///
/// A family and not a per-module mesh, because a palette entry's *dimensions*
/// live on the instance now: an office mullion and a house quoin are the same
/// shape at two sizes, and giving each its own copy of the same vertices would
/// put two identical uploads in the GPU cache under two ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModuleShape {
    /// A solid wall panel: a leaf with a raised border on both faces.
    Panel,
    /// A **glazed** panel: a rectangular frame standing proud of a thin,
    /// recessed pane. The one family
    /// [`ModuleShape::is_glazing`] answers `true` for.
    Glazing,
    /// A vertical member: a shaft with a base and a cap.
    Column,
    /// A horizontal deck: a plate with a fascia lip around its rim.
    Deck,
    /// One stair tread: a slab with a nosing.
    Tread,
    /// A head or a sill course: a band with a proud drip.
    Course,
    /// A table-shaped piece: a top on four legs.
    Legged,
    /// A cabinet-shaped piece: a carcass on a recessed plinth, with a rail.
    Carcass,
    /// An upholstered piece: a base, a back and two arms.
    Soft,
    /// A planter: a tapering pot under a canopy.
    Planter,
    /// A crate: a body with corner battens.
    Crate,
    /// A roller shutter: a leaf with horizontal ribs.
    Shutter,
    /// **A raised in-room platform** (wave VEN1a): a plank deck on a skirt — a
    /// stage, a catwalk, a riser. Distinct from [`Deck`](ModuleShape::Deck),
    /// which is a *storey's* floor and has a fascia lip rather than a solid
    /// skirt a bench can be pushed against.
    Stage,
    /// **A vertical round shaft** (wave VEN1a): an eight-sided prism with a
    /// floor plate and a ceiling plate. The dance pole, and the only family in
    /// this table that is not a union of axis-aligned boxes.
    Pole,
    /// **A counter run** (wave VEN1a): a carcass under a top that **overhangs
    /// the front**, with a foot rail below it. A bar, not a shop counter —
    /// [`Carcass`](ModuleShape::Carcass) is the shop counter and has neither.
    Bar,
    /// **A stool** (wave VEN1a): a round seat on a round pedestal over a base
    /// plate.
    Stool,
    /// **A screen** (wave VEN1a): a bezel around a panel standing **proud** of
    /// it, so the lit surface is in front of its frame rather than recessed
    /// behind it — the one thing that tells a television from a window at a
    /// glance, and the reason this is not [`Glazing`](ModuleShape::Glazing).
    Screen,
    /// **A sign plate** (wave VEN1a): a face on standoffs over a backer, so the
    /// lit plate floats off the wall it is bolted to and its glow has somewhere
    /// to spill.
    Sign,
    /// **A string-light run** (wave VEN1a): two cable strands with bulbs hung
    /// off them. Authored as one module rather than as one bulb, because a
    /// chain is a *run* — the assembler places it along a wall the way it places
    /// a counter, and a per-bulb module would be forty instances where one will
    /// do.
    Festoon,
}

impl ModuleShape {
    /// Every family, in the canonical order [`module_meshes`] emits.
    ///
    /// **Append-only.** The order is not a wire contract — every id is derived
    /// from [`name`](ModuleShape::name) and never from a position — but a table
    /// both hosts register is easier to read against a diff when it grows at the
    /// end, and the seven venue families (wave VEN1a) are appended for that
    /// reason rather than filed beside their nearest relatives.
    pub const ALL: [ModuleShape; 19] = [
        ModuleShape::Panel,
        ModuleShape::Glazing,
        ModuleShape::Column,
        ModuleShape::Deck,
        ModuleShape::Tread,
        ModuleShape::Course,
        ModuleShape::Legged,
        ModuleShape::Carcass,
        ModuleShape::Soft,
        ModuleShape::Planter,
        ModuleShape::Crate,
        ModuleShape::Shutter,
        ModuleShape::Stage,
        ModuleShape::Pole,
        ModuleShape::Bar,
        ModuleShape::Stool,
        ModuleShape::Screen,
        ModuleShape::Sign,
        ModuleShape::Festoon,
    ];

    /// The stable name the GUID is derived from. **Never change one of these
    /// without meaning to change the id** — it is the content key, and both
    /// hosts mint their table from it.
    pub fn name(self) -> &'static str {
        match self {
            ModuleShape::Panel => "panel",
            ModuleShape::Glazing => "glazing",
            ModuleShape::Column => "column",
            ModuleShape::Deck => "deck",
            ModuleShape::Tread => "tread",
            ModuleShape::Course => "course",
            ModuleShape::Legged => "legged",
            ModuleShape::Carcass => "carcass",
            ModuleShape::Soft => "soft",
            ModuleShape::Planter => "planter",
            ModuleShape::Crate => "crate",
            ModuleShape::Shutter => "shutter",
            ModuleShape::Stage => "stage",
            ModuleShape::Pole => "pole",
            ModuleShape::Bar => "bar",
            ModuleShape::Stool => "stool",
            ModuleShape::Screen => "screen",
            ModuleShape::Sign => "sign",
            ModuleShape::Festoon => "festoon",
        }
    }

    /// Whether this family is a window — the one thing that glows at night
    /// (island wave I8b clause 3).
    pub fn is_glazing(self) -> bool {
        self == ModuleShape::Glazing
    }

    /// This family's mesh, in unit space.
    pub fn mesh(self) -> ModuleMesh {
        let mut m = ModuleMesh::default();
        match self {
            // A leaf 60 % of the module's depth, with a border standing proud on
            // both faces — so a wall run reads as a course of panels rather than
            // as one flat sheet, from either side.
            ModuleShape::Panel => {
                m.push_box([0.0, 0.0, 0.0], [0.3, 0.5, 0.5]);
                for (cy, cz, hy, hz) in [
                    (0.45, 0.0, 0.05, 0.5),
                    (-0.45, 0.0, 0.05, 0.5),
                    (0.0, 0.45, 0.4, 0.05),
                    (0.0, -0.45, 0.4, 0.05),
                ] {
                    m.push_box([0.0, cy, cz], [0.5, hy, hz]);
                }
            }
            // Four frame members around a pane at 30 % depth. The pane is
            // separate geometry so a night window has something to emit from
            // that is not the frame.
            ModuleShape::Glazing => {
                for (cy, cz, hy, hz) in [
                    (0.44, 0.0, 0.06, 0.5),
                    (-0.44, 0.0, 0.06, 0.5),
                    (0.0, 0.44, 0.38, 0.06),
                    (0.0, -0.44, 0.38, 0.06),
                ] {
                    m.push_box([0.0, cy, cz], [0.5, hy, hz]);
                }
                m.push_box([0.0, 0.0, 0.0], [0.15, 0.38, 0.38]);
            }
            // A shaft with a base and a cap: the silhouette a pilaster, a quoin
            // and a mullion share.
            ModuleShape::Column => {
                m.push_box([0.0, 0.0, 0.0], [0.4, 0.5, 0.4]);
                m.push_box([0.0, -0.44, 0.0], [0.5, 0.06, 0.5]);
                m.push_box([0.0, 0.44, 0.0], [0.5, 0.06, 0.5]);
            }
            // A plate with a fascia lip on all four rims — what makes a floor
            // edge visible from the storey below.
            ModuleShape::Deck => {
                m.push_box([0.0, 0.1, 0.0], [0.5, 0.4, 0.5]);
                for (cx, cz, hx, hz) in [
                    (0.46, 0.0, 0.04, 0.5),
                    (-0.46, 0.0, 0.04, 0.5),
                    (0.0, 0.46, 0.46, 0.04),
                    (0.0, -0.46, 0.46, 0.04),
                ] {
                    m.push_box([cx, -0.25, cz], [hx, 0.25, hz]);
                }
            }
            // A tread with a nosing that overhangs the riser below it.
            ModuleShape::Tread => {
                m.push_box([0.0, -0.1, 0.0], [0.5, 0.4, 0.42]);
                m.push_box([0.0, 0.4, 0.0], [0.5, 0.1, 0.5]);
            }
            // A band with a drip course under its outer face.
            ModuleShape::Course => {
                m.push_box([0.0, 0.05, 0.0], [0.5, 0.45, 0.5]);
                m.push_box([0.0, -0.42, 0.0], [0.5, 0.08, 0.42]);
            }
            // A top on four legs — the one family with real air in it.
            ModuleShape::Legged => {
                m.push_box([0.0, 0.4, 0.0], [0.5, 0.1, 0.5]);
                for (sx, sz) in [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    m.push_box([sx * 0.42, -0.15, sz * 0.42], [0.08, 0.35, 0.08]);
                }
            }
            // A carcass on a recessed plinth, split by a rail: a cabinet, a
            // locker, a shelf stack, a counter.
            ModuleShape::Carcass => {
                m.push_box([0.0, 0.05, 0.0], [0.5, 0.45, 0.5]);
                m.push_box([0.0, -0.45, 0.0], [0.42, 0.05, 0.42]);
                m.push_box([0.0, 0.05, 0.46], [0.48, 0.03, 0.04]);
            }
            // A base, a back and two arms.
            ModuleShape::Soft => {
                m.push_box([0.0, -0.15, 0.0], [0.5, 0.35, 0.5]);
                m.push_box([0.0, 0.25, -0.38], [0.5, 0.25, 0.12]);
                m.push_box([0.44, 0.15, 0.06], [0.06, 0.15, 0.44]);
                m.push_box([-0.44, 0.15, 0.06], [0.06, 0.15, 0.44]);
            }
            // A pot under a canopy — two stacked boxes and a stem.
            ModuleShape::Planter => {
                m.push_box([0.0, -0.35, 0.0], [0.38, 0.15, 0.38]);
                m.push_box([0.0, 0.0, 0.0], [0.08, 0.25, 0.08]);
                m.push_box([0.0, 0.3, 0.0], [0.5, 0.2, 0.5]);
            }
            // A body with battens on its four vertical corners.
            ModuleShape::Crate => {
                m.push_box([0.0, 0.0, 0.0], [0.46, 0.5, 0.46]);
                for (sx, sz) in [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                    m.push_box([sx * 0.44, 0.0, sz * 0.44], [0.06, 0.48, 0.06]);
                }
            }
            // A leaf with five horizontal ribs.
            ModuleShape::Shutter => {
                m.push_box([0.0, 0.0, 0.0], [0.3, 0.5, 0.5]);
                for k in 0..5 {
                    let y = -0.4 + 0.2 * k as f32;
                    m.push_box([0.0, y, 0.0], [0.5, 0.06, 0.48]);
                }
            }
            // A plank deck over a solid skirt: the top face is what a dancer
            // stands on and the skirt is what a bench at the edge is pushed
            // against, so the two read as one riser and not as a floating slab.
            ModuleShape::Stage => {
                m.push_box([0.0, 0.35, 0.0], [0.5, 0.15, 0.5]);
                for (cx, cz, hx, hz) in [
                    (0.44, 0.0, 0.06, 0.44),
                    (-0.44, 0.0, 0.06, 0.44),
                    (0.0, 0.44, 0.5, 0.06),
                    (0.0, -0.44, 0.5, 0.06),
                ] {
                    m.push_box([cx, -0.15, cz], [hx, 0.35, hz]);
                }
            }
            // An octagonal shaft floor to ceiling, with a plate at each end so
            // it lands on something instead of ending in mid-air.
            ModuleShape::Pole => {
                m.push_prism_y(0.5, -0.42, 0.42);
                m.push_box([0.0, -0.46, 0.0], [0.42, 0.04, 0.42]);
                m.push_box([0.0, 0.46, 0.0], [0.42, 0.04, 0.42]);
            }
            // A carcass under a top that overhangs the FRONT (`+Z`), with a foot
            // rail under the overhang. The overhang is the whole silhouette of a
            // bar: it is what a stool tucks under and what an elbow rests on.
            ModuleShape::Bar => {
                m.push_box([0.0, -0.06, -0.15], [0.5, 0.44, 0.35]);
                m.push_box([0.0, 0.43, 0.0], [0.5, 0.07, 0.5]);
                m.push_box([0.0, -0.3, 0.36], [0.5, 0.05, 0.06]);
                m.push_box([0.0, -0.46, -0.2], [0.44, 0.04, 0.28]);
            }
            // A round seat on a round pedestal over a base plate.
            ModuleShape::Stool => {
                m.push_prism_y(0.5, 0.32, 0.5);
                m.push_prism_y(0.14, -0.4, 0.32);
                m.push_box([0.0, -0.46, 0.0], [0.34, 0.04, 0.34]);
            }
            // A bezel around a panel standing PROUD of it. The panel is separate
            // geometry, like `Glazing`'s pane, so a lit screen has a surface of
            // its own to emit from that is not its frame — and it is in FRONT of
            // the frame, which is what tells a television from a window.
            ModuleShape::Screen => {
                m.push_box([0.0, 0.0, -0.34], [0.5, 0.5, 0.16]);
                for (cx, cy, hx, hy) in [
                    (0.45, 0.0, 0.05, 0.5),
                    (-0.45, 0.0, 0.05, 0.5),
                    (0.0, 0.45, 0.4, 0.05),
                    (0.0, -0.45, 0.4, 0.05),
                ] {
                    m.push_box([cx, cy, 0.1], [hx, hy, 0.28]);
                }
                m.push_box([0.0, 0.0, 0.34], [0.42, 0.42, 0.16]);
            }
            // A face on two standoffs over a backer. The gap is the point: a
            // sign bolted flat to a wall has nowhere for its glow to spill, and
            // in the reference every neon plate throws a halo onto the boards
            // behind it.
            ModuleShape::Sign => {
                m.push_box([0.0, 0.0, -0.44], [0.5, 0.5, 0.06]);
                for cx in [0.3f32, -0.3] {
                    m.push_box([cx, 0.0, -0.1], [0.06, 0.24, 0.3]);
                }
                m.push_box([0.0, 0.0, 0.36], [0.5, 0.5, 0.14]);
            }
            // Two cable strands with three bulbs each, hung at opposite depths
            // so the run reads as a swag rather than as a painted line.
            ModuleShape::Festoon => {
                for cz in [0.42f32, -0.42] {
                    m.push_box([0.0, 0.42, cz], [0.5, 0.08, 0.08]);
                    for k in 0..3 {
                        let x = -0.32 + 0.32 * k as f32;
                        m.push_box([x, -0.1, cz], [0.11, 0.4, 0.08]);
                    }
                }
            }
        }
        m
    }
}

/// The GUID a module of `shape` draws under.
///
/// A pure function of the family's [`name`](ModuleShape::name) under
/// `MODULE_MESH_SALT`, so the id is the same in every process, on every
/// platform and in both hosts, and is derived from *what the mesh is* rather
/// than from where it happens to sit in a table.
pub fn module_mesh_guid(shape: ModuleShape) -> Uuid {
    let mut x = MODULE_MESH_SALT;
    for b in shape.name().as_bytes() {
        x = x.rotate_left(11) ^ (*b as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// Which shape family a palette module belongs to, **by name**.
///
/// An exhaustive match over every module the seven palettes declare, with no
/// wildcard arm: a palette that adds a module and forgets to classify it gets a
/// `None` here and fails
/// `tests::every_palette_module_has_a_shape`
/// rather than silently drawing a rectangular prism again.
pub fn shape_of(module: &str) -> Option<ModuleShape> {
    Some(match module {
        // Glazed openings — the ones that light up. `Pane` is the leaf the
        // assembler hangs in a window void; the other two are glazed *wall*
        // modules, and a curtain wall is as much a window as a casement is.
        "Pane" | "Glazing" | "Shopfront" => ModuleShape::Glazing,
        // Solid wall leaves.
        "Spandrel" | "Wall" | "Balcony" | "Partition" | "Cladding" | "Brick" | "Ashlar"
        | "Panelled" | "Solid" => ModuleShape::Panel,
        // Vertical members.
        "Mullion" | "Pier" | "Column" | "Quoin" | "Pilaster" | "Stall" => ModuleShape::Column,
        // Horizontal decks.
        "Slab" | "Roof" => ModuleShape::Deck,
        "Step" => ModuleShape::Tread,
        "Lintel" | "Parapet" => ModuleShape::Course,
        // Furniture.
        "Desk" | "Table" | "Bench" => ModuleShape::Legged,
        "Cabinet" | "Locker" | "Wardrobe" | "Units" | "Shelf" | "Rack" | "Counter" | "Basin" => {
            ModuleShape::Carcass
        }
        "Sofa" | "Bed" => ModuleShape::Soft,
        "Plant" => ModuleShape::Planter,
        "Crate" => ModuleShape::Crate,
        "RollDoor" => ModuleShape::Shutter,
        // ── the venue vocabulary (wave VEN1a) ──
        //
        // A `Catwalk` is a `Stage` that happens to be long and thin, which is a
        // fact about the *extent* the assembler writes and not about the shape:
        // one family, two names, exactly as `Mullion` and `Quoin` share
        // `Column`.
        "Stage" | "Catwalk" => ModuleShape::Stage,
        "Pole" => ModuleShape::Pole,
        "BarRun" => ModuleShape::Bar,
        "Stool" => ModuleShape::Stool,
        "Screen" => ModuleShape::Screen,
        "Neon" => ModuleShape::Sign,
        "Festoon" => ModuleShape::Festoon,
        _ => return None,
    })
}

/// The GUID a named palette module draws under, or `None` for a module with no
/// classified shape.
pub fn module_guid_for(module: &str) -> Option<Uuid> {
    shape_of(module).map(module_mesh_guid)
}

/// **The whole table**, in [`ModuleShape::ALL`] order — what a host registers so
/// the GUIDs the assembler writes resolve to geometry.
///
/// One entry per *family*, not per palette entry, so twelve uploads serve every
/// module of every archetype.
pub fn module_meshes() -> Vec<(Uuid, ModuleMesh)> {
    ModuleShape::ALL
        .into_iter()
        .map(|s| (module_mesh_guid(s), s.mesh()))
        .collect()
}

/// How brightly a glazed module emits **at full night**, as a multiplier on its
/// own colour.
///
/// One number for the whole engine rather than a per-archetype knob: a lit
/// window is a lit window, and seven values would be seven chances for one
/// district to be brighter than another for no authored reason. The *hour* is
/// applied by the projector, not here — see
/// [`PcgInstance::glow`](crate::scatter::PcgInstance::glow).
pub const GLAZING_GLOW: f32 = 1.6;

/// Every module name any shipped palette declares, sorted and deduplicated.
///
/// Exists for the arms below and for a caller that wants to know the vocabulary
/// without parsing seven grammars.
pub fn declared_modules() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for arch in archetypes() {
        let Ok(g) = arch.grammar() else { continue };
        for m in g.modules() {
            out.push(m.name.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **THE ANTI-VACUITY ARM.** Every module every palette declares maps to a
    /// shape, so "buildings stopped being cubes" is a statement about all of
    /// them and not about the ones somebody remembered.
    #[test]
    fn every_palette_module_has_a_shape() {
        let names = declared_modules();
        assert!(names.len() > 30, "only {} modules declared", names.len());
        for n in &names {
            assert!(shape_of(n).is_some(), "module {n} has no shape family");
        }
    }

    /// Twelve families, twelve distinct GUIDs, and none of them nil.
    #[test]
    fn the_family_guids_are_distinct() {
        let mut ids: Vec<Uuid> = ModuleShape::ALL.into_iter().map(module_mesh_guid).collect();
        assert!(ids.iter().all(|i| !i.is_nil()));
        ids.sort();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "two families mint one id");
    }

    /// The GUID is a function of the NAME, so a family that keeps its name keeps
    /// its id whatever else moves — and two names never agree.
    #[test]
    fn the_guid_follows_the_name() {
        assert_eq!(
            module_mesh_guid(ModuleShape::Panel),
            module_mesh_guid(ModuleShape::Panel)
        );
        assert_ne!(
            module_mesh_guid(ModuleShape::Panel),
            module_mesh_guid(ModuleShape::Glazing)
        );
    }

    /// Every family produces a real, bounded, closed-ish mesh: inside the unit
    /// box, more than a cube's twelve triangles, and under the scatter ceiling.
    #[test]
    fn every_family_is_real_geometry_inside_the_unit_box() {
        for s in ModuleShape::ALL {
            let m = s.mesh();
            assert!(
                m.triangle_count() > 12,
                "{}: {} triangles — that is a box",
                s.name(),
                m.triangle_count()
            );
            assert!(
                m.triangle_count() <= 128,
                "{}: {} triangles is not modest",
                s.name(),
                m.triangle_count()
            );
            assert_eq!(m.positions.len(), m.normals.len());
            assert!(m.indices.iter().all(|i| (*i as usize) < m.positions.len()));
            for p in &m.positions {
                for c in p {
                    assert!(
                        (-0.5..=0.5).contains(c),
                        "{}: a vertex at {c} left the unit box",
                        s.name()
                    );
                }
            }
            // …and it FILLS the unit box on every axis, so scaling it onto an
            // extent produces something the size of the collider rather than a
            // shrunken proxy inside it.
            for axis in 0..3 {
                let lo = m.positions.iter().map(|p| p[axis]).fold(f32::MAX, f32::min);
                let hi = m.positions.iter().map(|p| p[axis]).fold(f32::MIN, f32::max);
                assert!(
                    lo <= -0.499 && hi >= 0.499,
                    "{}: axis {axis} spans [{lo}, {hi}] — the mesh does not fill its box",
                    s.name()
                );
            }
        }
    }

    /// **A face must point out of its own box.** The winding loop serves six
    /// faces from one table, so this is the arm that says the negative-side flip
    /// is right: every triangle's geometric normal agrees with the vertex normal
    /// it was authored with.
    #[test]
    fn every_face_points_the_way_its_normal_says() {
        for s in ModuleShape::ALL {
            let m = s.mesh();
            for t in m.indices.chunks_exact(3) {
                let (a, b, c) = (
                    m.positions[t[0] as usize],
                    m.positions[t[1] as usize],
                    m.positions[t[2] as usize],
                );
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cross = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let n = m.normals[t[0] as usize];
                let dot = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
                assert!(dot > 0.0, "{}: an inside-out face (dot {dot})", s.name());
            }
        }
    }

    /// Exactly one family glows, and the two glazed palette modules find it.
    #[test]
    fn glazing_is_one_family_and_two_palette_modules() {
        let glowing: Vec<ModuleShape> = ModuleShape::ALL
            .into_iter()
            .filter(|s| s.is_glazing())
            .collect();
        assert_eq!(glowing, vec![ModuleShape::Glazing]);
        assert_eq!(shape_of("Glazing"), Some(ModuleShape::Glazing));
        assert_eq!(shape_of("Shopfront"), Some(ModuleShape::Glazing));
        assert_eq!(shape_of("Brick"), Some(ModuleShape::Panel));
    }

    /// **The pole is round, and the assertion says what "round" means** (wave
    /// VEN1a).
    ///
    /// Every other family is a union of axis-aligned boxes, so a
    /// `push_prism_y` that silently produced one would pass every other arm in
    /// this module: the span check, the winding check and the triangle bounds
    /// are all satisfied by a cube. What separates a prism from a box is that
    /// its side normals point in **more than four** directions, and that its
    /// silhouette has vertices strictly inside the box's corners.
    #[test]
    fn a_prism_is_not_a_box() {
        for s in [ModuleShape::Pole, ModuleShape::Stool] {
            let m = s.mesh();
            // Side normals: horizontal, and the four DIAGONAL ones a union of
            // axis-aligned boxes can never produce. (Counting distinct
            // horizontals would not do it — the pole's own end plates are boxes
            // and contribute the four axis directions themselves.)
            let mut diagonals: Vec<(i32, i32)> = m
                .normals
                .iter()
                .filter(|n| n[1].abs() < 0.01 && n[0].abs() > 0.01 && n[2].abs() > 0.01)
                .map(|n| ((n[0] * 1000.0) as i32, (n[2] * 1000.0) as i32))
                .collect();
            diagonals.sort_unstable();
            diagonals.dedup();
            assert_eq!(
                diagonals.len(),
                8,
                "{}: {} distinct diagonal side normals — an axis-aligned box has none,                  and an octagon's eight faces all sit BETWEEN the axes",
                s.name(),
                diagonals.len()
            );
            // A corner of the unit box is 0.707 out on both axes; the octagon's
            // diagonal vertex is 0.354. So no vertex may sit near a corner.
            assert!(
                !m.positions
                    .iter()
                    .any(|p| p[0].abs() > 0.45 && p[2].abs() > 0.45),
                "{}: a vertex sits in a corner of the unit box — that is a box",
                s.name()
            );
            // …and it really does reach the box's faces, or "round" would be
            // satisfied by a small cylinder rattling inside it.
            assert!(m.positions.iter().any(|p| p[0] >= 0.499));
            assert!(m.positions.iter().any(|p| p[2] <= -0.499));
        }
    }

    /// The table is what a host registers: nineteen entries, distinct ids, real
    /// meshes.
    #[test]
    fn the_table_is_the_families() {
        let t = module_meshes();
        assert_eq!(t.len(), ModuleShape::ALL.len());
        for (id, m) in &t {
            assert!(!id.is_nil());
            assert!(m.triangle_count() > 12);
        }
    }
}
