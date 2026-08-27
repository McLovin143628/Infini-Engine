//! **A building module's own geometry** (island wave I8b) — the day the seven
//! palettes stopped drawing cubes.
//!
//! # What was actually wrong, and it is not what the ledger said
//!
//! Wave I8a recorded the settlements as *"365 545 wall-sized opaque boxes"*.
//! They were not wall-sized. [`crate::scatter::ScatteredInstance::scale`] — and
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
//!   there are 21 meshes rather than one per palette entry;
//! * every feature is **proportional** — a frame rail is a fraction of the
//!   panel, never a fixed 40 mm — because the same mesh is stretched onto a
//!   0.3 m mullion and a 12 m slab. A fixed-size chamfer would be invisible on
//!   one and a metre deep on the other.
//!
//! # Boxes, composed
//!
//! Every shape is a small union of axis-aligned boxes with flat normals: 3 to 6
//! of them, 36 to 72 triangles. No half-edge kernel (`inf-dcc` is a **dev**
//! dependency of the shipped player and this code is Ring 0 on its draw path),
//! no trigonometry (the P14 law — these vertices are a pure function of
//! constants and reach a content hash), and no boolean. The relief is in the
//! **silhouette**: a window has a frame standing proud of a recessed pane, a
//! desk has legs with air between them, a slab has a fascia. That is what
//! separates "real geometry, modestly" from a re-textured box, and the meshlet
//! path is not asked to carry any of it.
//!
//! # The GUIDs are content-derived
//!
//! A palette module names no asset — there is no `.inf_mesh` file to point at,
//! and inventing one per module would put 21 files in `samples/` to express
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
}

/// The shape families the 21 module meshes come in.
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
    /// recessed pane. The one family [`is_glazing`] answers `true` for.
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
}

impl ModuleShape {
    /// Every family, in the canonical order [`module_meshes`] emits.
    pub const ALL: [ModuleShape; 12] = [
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
        }
        m
    }
}

/// The GUID a module of `shape` draws under.
///
/// A pure function of the family's [`name`](ModuleShape::name) under
/// [`MODULE_MESH_SALT`], so the id is the same in every process, on every
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
/// [`every_palette_module_has_a_shape`](tests::every_palette_module_has_a_shape)
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

    /// The table is what a host registers: twelve entries, distinct ids, real
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
