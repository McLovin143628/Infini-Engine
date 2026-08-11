//! **Cloth & hair authoring, Ring 1** (P24.4): the two doors that turn an open
//! Model Editor mesh into a `.inf_cloth` or a `.inf_hair`.
//!
//! P24.4's first two batches built the payloads, the solvers and both hosts'
//! fixed steps — and left `ClothAsset::from_garment` and `HairAsset::grow` with
//! **no caller outside a test**. The `.inf_hair` upgrade remedy already said
//! "Model Editor ▸ Hair ▸ Grow Guides", naming a control that did not exist. This
//! module is what makes those two sentences true.
//!
//! # Why Ring 1 and not the command module
//!
//! The [`crate::dcc`] law, verbatim: a `#[tauri::command]` cannot be driven from
//! a test on any CI leg, so a rule written there is a rule no gate can see. The
//! Ring-2 commands are four lines each — parse ids, call one of these, write the
//! asset — and everything that decides *what a garment is* lives here, where
//! `cargo test -p inf-editor-core` reaches it.
//!
//! # What is authored, and what is derived
//!
//! Deliberately split, because the split is what keeps the payload frozen at v1:
//!
//! * **Authored** — the material knobs, the body radius, the strand length and
//!   segment count, and the groom (clump strength, curl radius, curl turns).
//!   These are *inputs*, held in the panel.
//! * **Derived** — the constraint sets, the roots, the clump numbering, the
//!   collision capsules and the grown strand points. These are what the asset
//!   carries.
//!
//! So the recipe is **not** persisted, and re-grooming means re-entering the
//! numbers. That is the P23 `Op::Unwrap` doctrine applied here (journal the
//! result, so a later build replays a fact rather than its own opinion of the
//! recipe), and it is the reason neither `.inf_cloth` nor `.inf_hair` needs a
//! schema move to gain grooming. The bound is ledgered rather than hidden.
//!
//! # Units
//!
//! Metres, everywhere, and kg⁻¹ for the inverse masses the pin list writes.
//! Compliance is m/N — see [`inf_anim::ClothMaterial`] for why it is compliance
//! and not a 0..1 "stiffness".

use std::collections::BTreeMap;

use inf_anim::{
    ClothAsset, ClothError, ClothMaterial, HairAsset, HairGroom, HairMaterial, HairRoot, Skeleton,
};
use inf_dcc::{Mesh, SelectionSet, VertId};

/// What an authoring door refused, and why — values, never panics, because every
/// one of these is reachable from a mesh somebody modelled.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GroomError {
    /// The mesh has no faces, or no vertices — nothing to simulate.
    #[error(
        "this mesh has {verts} vertices and {faces} faces; a garment needs at least one triangle"
    )]
    EmptyMesh { verts: usize, faces: usize },
    /// No faces are selected, so there is no scalp to grow from.
    #[error(
        "select the scalp faces first — hair grows from a face selection, and \
         nothing is selected"
    )]
    NoScalp,
    /// A face in the selection has fewer than three corners (it was deleted, or
    /// the selection is from an older generation).
    #[error("face {0} is not a polygon any more — re-select the scalp")]
    DegenerateFace(u32),
    /// The payload builder refused. Carries its own message, which already names
    /// the offending quantity.
    #[error("{0}")]
    Payload(String),
}

impl From<ClothError> for GroomError {
    fn from(e: ClothError) -> Self {
        GroomError::Payload(e.to_string())
    }
}

/// The garment knobs the Model Editor's Cloth section exposes.
///
/// Every field is a physical parameter with a unit, and the defaults are
/// [`ClothMaterial::default`]'s mid-weight draping fabric plus a body radius that
/// suits a human-scale rig.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GarmentSpec {
    /// The garment's material and solver budget.
    pub material: ClothMaterial,
    /// The uniform radius of the body's collision capsules, metres. `0` derives
    /// no capsules at all, which is a garment that hangs through its wearer —
    /// legal, and what an unskinned test sheet wants.
    pub body_radius_m: f32,
}

impl Default for GarmentSpec {
    fn default() -> Self {
        Self {
            material: ClothMaterial::default(),
            // 8 cm around a bone is a torso-ish human limb. Uniform — see
            // `inf_anim::body_capsules` for the bound that carries.
            body_radius_m: 0.08,
        }
    }
}

/// What [`garment_from_session`] did, so the panel can say it rather than the
/// author having to open the asset to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GarmentReport {
    /// Particles (= mesh vertices).
    pub particles: usize,
    /// Triangles the garment draws.
    pub triangles: usize,
    /// Stretch constraints (unique mesh edges).
    pub stretch: usize,
    /// Bend constraints (interior edges).
    pub bend: usize,
    /// Particles pinned by the selection.
    pub pinned: usize,
    /// Collision capsules derived from the bound skeleton.
    pub capsules: usize,
}

/// **The garment door**: an open mesh, a vertex selection and a spec become a
/// `.inf_cloth`.
///
/// * Positions and triangles come from the **half-edge** mesh, not from
///   [`crate::dcc::tessellate`]. That is load-bearing and not a shortcut: the
///   export writer splits corners on sharp edges and UV seams, so a tessellated
///   garment would arrive with its seam vertices *unwelded* and the cloth would
///   come apart along every seam it was authored with. What a solver needs is the
///   welded topology, which is exactly what the kernel holds.
/// * The **selected vertices are the pins** (`inv_mass = 0`) — a collar on a
///   shoulder, a waistband on a belt. That is the whole pinning UI: select, then
///   press the button.
/// * `skeleton` is the character the garment is worn by. When present its bones
///   become the collision capsules through [`inf_anim::body_capsules`]; when
///   absent the garment collides against nothing and the report says `0`.
///
/// N-gons are fan-triangulated about their first corner, which matches what the
/// mesh writer does for a convex face and is what the constraint builder needs;
/// the resulting triangles are the ones the garment *draws*, so the picture and
/// the simulation cannot disagree.
pub fn garment_from_session(
    mesh: &Mesh,
    selection: &SelectionSet,
    source_mesh: [u8; 16],
    spec: GarmentSpec,
    skeleton: Option<&Skeleton>,
) -> Result<(ClothAsset, GarmentReport), GroomError> {
    let (positions, index_of) = welded_positions(mesh);
    let indices = triangle_list(mesh, &index_of)?;
    if positions.len() < 3 || indices.is_empty() {
        return Err(GroomError::EmptyMesh {
            verts: positions.len(),
            faces: mesh.face_count(),
        });
    }
    // The pins, in ascending particle order — a `BTreeSet` walked in order, so
    // two sessions that selected the same vertices in a different sequence build
    // the same asset.
    let pinned: Vec<u32> = selection
        .verts()
        .iter()
        .filter_map(|v| index_of.get(v).copied())
        .collect();
    let capsules = match skeleton {
        Some(sk) => inf_anim::body_capsules(sk, spec.body_radius_m),
        None => Vec::new(),
    };
    let asset =
        ClothAsset::from_garment(source_mesh, &positions, &indices, &pinned, spec.material)?
            .with_capsules(capsules);
    let report = GarmentReport {
        particles: asset.particle_count(),
        triangles: asset.triangle_count(),
        stretch: asset.distance.len(),
        bend: asset.bending.len(),
        pinned: pinned.len(),
        capsules: asset.collision.len(),
    };
    Ok((asset, report))
}

/// The hairstyle knobs the Model Editor's Hair section exposes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroomSpec {
    /// Strand length, metres.
    pub length_m: f32,
    /// Segments per strand. More segments bend more smoothly and cost more.
    pub segments: u16,
    /// The strand material and ribbon width.
    pub material: HairMaterial,
    /// The groom shape — clump strength, curl radius, curl turns.
    pub groom: HairGroom,
    /// **Clump cell size, metres.** Roots whose positions fall in the same cell
    /// of a grid this size share a clump number, so clumping pulls *neighbours*
    /// together rather than an arbitrary set of roots scattered over the scalp.
    /// `0` puts every root in its own clump, which is "no clumping" spelled as
    /// data.
    pub clump_spacing_m: f32,
    /// The joint a root rides when the scalp mesh carries no skin weights.
    pub fallback_joint: u16,
    /// The uniform body-capsule radius, metres — the head and shoulders a strand
    /// must not pass through. Same knob and same bound as [`GarmentSpec`].
    pub body_radius_m: f32,
}

impl Default for GroomSpec {
    fn default() -> Self {
        Self {
            length_m: 0.25,
            segments: 6,
            material: HairMaterial::default(),
            groom: HairGroom::default(),
            // 2 cm cells: roughly a finger's width, which is what a clump of hair
            // is.
            clump_spacing_m: 0.02,
            fallback_joint: 0,
            body_radius_m: 0.08,
        }
    }
}

/// What [`groom_from_session`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroomReport {
    /// Guide strands grown (one per selected face).
    pub strands: usize,
    /// Particles across every strand.
    pub particles: usize,
    /// Distinct clumps the roots fell into.
    pub clumps: usize,
    /// Roots whose joint came from the mesh's own skin weights (the rest took
    /// [`GroomSpec::fallback_joint`]).
    pub skinned_roots: usize,
    /// Collision capsules derived from the bound skeleton.
    pub capsules: usize,
}

/// **The hair door**: the selected faces are the scalp, and one guide grows out
/// of each of them.
///
/// A face rather than a vertex is the sampling unit because a face has a
/// **normal**, and a strand's growth direction is the scalp's outward normal —
/// which is the difference between hair growing out of a head and hair growing
/// along it. The root sits at the face centroid.
///
/// Each root's joint is the **dominant influence** of the face's corners (the
/// joint with the largest summed weight), so a strand grown on the head rides the
/// head joint with nothing to pick by hand. An unskinned scalp takes
/// [`GroomSpec::fallback_joint`] and the report counts how many roots that was.
///
/// Clump numbers come from a grid over the root positions
/// ([`GroomSpec::clump_spacing_m`]), assigned in ascending cell order through a
/// `BTreeMap` — so the numbering is a pure function of the geometry rather than
/// of face order or of a hash seed.
///
/// The roots come out in **face-id order** and the whole thing is deterministic:
/// the same mesh and the same selection grow the same hairstyle on every machine.
pub fn groom_from_session(
    mesh: &Mesh,
    selection: &SelectionSet,
    scalp: [u8; 16],
    spec: GroomSpec,
    skeleton: Option<&Skeleton>,
) -> Result<(HairAsset, GroomReport), GroomError> {
    if selection.faces().is_empty() {
        return Err(GroomError::NoScalp);
    }
    let mut placed: Vec<(glam::DVec3, glam::DVec3, u16, bool)> = Vec::new();
    for &f in selection.faces() {
        let Some(verts) = mesh.face_verts(f) else {
            continue;
        };
        if verts.len() < 3 {
            return Err(GroomError::DegenerateFace(f.index() as u32));
        }
        let mut centroid = glam::DVec3::ZERO;
        for v in &verts {
            centroid += mesh.position(*v).unwrap_or(glam::DVec3::ZERO);
        }
        centroid /= verts.len() as f64;
        let normal = inf_dcc::face_normal(mesh, f).unwrap_or(glam::DVec3::Y);
        let (joint, from_skin) = dominant_joint(mesh, &verts, spec.fallback_joint);
        placed.push((centroid, normal, joint, from_skin));
    }
    if placed.is_empty() {
        return Err(GroomError::NoScalp);
    }

    // Clump numbering: ascending cell order, so the ids are stable under any
    // face ordering the selection happens to have.
    let cell = |p: glam::DVec3| -> (i64, i64, i64) {
        if spec.clump_spacing_m <= 0.0 || !spec.clump_spacing_m.is_finite() {
            // Every root its own cell — "no clumping", as data.
            return (i64::MIN, i64::MIN, i64::MIN);
        }
        let s = spec.clump_spacing_m as f64;
        (
            (p.x / s).floor() as i64,
            (p.y / s).floor() as i64,
            (p.z / s).floor() as i64,
        )
    };
    let one_per_root = spec.clump_spacing_m <= 0.0 || !spec.clump_spacing_m.is_finite();
    let mut cells: BTreeMap<(i64, i64, i64), u16> = BTreeMap::new();
    if !one_per_root {
        for (p, _, _, _) in &placed {
            cells.entry(cell(*p)).or_insert(0);
        }
        for (n, slot) in cells.values_mut().enumerate() {
            *slot = (n % u16::MAX as usize) as u16;
        }
    }

    let mut skinned_roots = 0usize;
    let roots: Vec<HairRoot> = placed
        .iter()
        .enumerate()
        .map(|(i, (p, n, joint, from_skin))| {
            if *from_skin {
                skinned_roots += 1;
            }
            HairRoot {
                joint: *joint,
                offset: [p.x as f32, p.y as f32, p.z as f32],
                direction: [n.x as f32, n.y as f32, n.z as f32],
                clump: if one_per_root {
                    (i % u16::MAX as usize) as u16
                } else {
                    cells.get(&cell(*p)).copied().unwrap_or(0)
                },
            }
        })
        .collect();
    let capsules = match skeleton {
        Some(sk) => inf_anim::body_capsules(sk, spec.body_radius_m),
        None => Vec::new(),
    };
    let asset = HairAsset::grow(
        scalp,
        &roots,
        spec.length_m,
        spec.segments,
        spec.material,
        spec.groom,
    )?
    .with_capsules(capsules);
    let clumps = roots
        .iter()
        .map(|r| r.clump)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let report = GroomReport {
        strands: asset.strand_count(),
        particles: asset.particle_count(),
        clumps,
        skinned_roots,
        capsules: asset.collision.len(),
    };
    Ok((asset, report))
}

/// The mesh's vertices as a dense position list, plus `VertId → particle index`.
///
/// Dense because a `VertId` is a slot in a kernel arena with holes in it after a
/// delete, and a garment's constraint indices must be contiguous. Ascending
/// `VertId` order, so the mapping is a pure function of the mesh.
fn welded_positions(mesh: &Mesh) -> (Vec<[f32; 3]>, BTreeMap<VertId, u32>) {
    let mut positions = Vec::with_capacity(mesh.vert_count());
    let mut index_of = BTreeMap::new();
    for v in mesh.vert_ids() {
        let p = mesh.position(v).unwrap_or(glam::DVec3::ZERO);
        index_of.insert(v, positions.len() as u32);
        positions.push([p.x as f32, p.y as f32, p.z as f32]);
    }
    (positions, index_of)
}

/// Every face fan-triangulated about its first corner, in face-id order.
fn triangle_list(mesh: &Mesh, index_of: &BTreeMap<VertId, u32>) -> Result<Vec<u32>, GroomError> {
    let mut indices = Vec::new();
    for f in mesh.face_ids() {
        let Some(verts) = mesh.face_verts(f) else {
            continue;
        };
        if verts.len() < 3 {
            continue;
        }
        let anchor = index_of.get(&verts[0]).copied().unwrap_or(0);
        for w in verts[1..].windows(2) {
            let (Some(&b), Some(&c)) = (index_of.get(&w[0]), index_of.get(&w[1])) else {
                continue;
            };
            indices.extend_from_slice(&[anchor, b, c]);
        }
    }
    Ok(indices)
}

/// The joint with the largest summed influence across `verts`, and whether it
/// came from real weights.
///
/// Ties break toward the **lowest joint index**, so the answer does not depend on
/// which corner the walk started at.
fn dominant_joint(mesh: &Mesh, verts: &[VertId], fallback: u16) -> (u16, bool) {
    if !mesh.is_skinned() {
        return (fallback, false);
    }
    let mut totals: BTreeMap<u16, f32> = BTreeMap::new();
    for v in verts {
        let Some(w) = mesh.vert_weights(*v) else {
            continue;
        };
        for (j, weight) in w.joints.iter().zip(w.weights.iter()) {
            if *weight > 0.0 {
                *totals.entry(*j).or_insert(0.0) += *weight;
            }
        }
    }
    let mut best: Option<(u16, f32)> = None;
    for (j, total) in totals {
        match best {
            Some((_, b)) if total <= b => {}
            _ => best = Some((j, total)),
        }
    }
    match best {
        Some((j, _)) => (j, true),
        None => (fallback, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_dcc::{Mesh, SelectMode, SelectionSet};

    const SOURCE: [u8; 16] = [0x24; 16];

    /// A 1 m cube — six quads, eight welded corners.
    fn cube() -> Mesh {
        inf_dcc::cube(1.0)
    }

    /// A rig whose bones are a chain, so `body_capsules` has something to derive
    /// from and a garment has something to collide with.
    fn rig() -> Skeleton {
        let mut joints = Vec::new();
        let mut global = glam::Mat4::IDENTITY;
        for i in 0..3 {
            let local = inf_anim::JointTransform::from_trs(
                if i == 0 {
                    glam::Vec3::ZERO
                } else {
                    glam::Vec3::Y * 0.5
                },
                glam::Quat::IDENTITY,
                glam::Vec3::ONE,
            );
            global *= local.to_mat4();
            joints.push(inf_anim::Joint {
                name: format!("j{i}"),
                parent: if i == 0 { None } else { Some(i as u16 - 1) },
                inverse_bind: global.inverse().to_cols_array(),
                local_bind: local,
            });
        }
        Skeleton::new(joints).unwrap()
    }

    fn empty_selection(_mesh: &Mesh) -> SelectionSet {
        SelectionSet::new(0)
    }

    /// **The garment door builds a simulatable garment out of the open mesh.**
    ///
    /// The headline: before this batch `ClothAsset::from_garment` had no caller
    /// outside a test, so "authored in the Model Editor" was a sentence about a
    /// button that did not exist.
    #[test]
    fn a_cube_becomes_a_garment_with_real_constraints() {
        let mesh = cube();
        let sel = empty_selection(&mesh);
        let (asset, report) =
            garment_from_session(&mesh, &sel, SOURCE, GarmentSpec::default(), Some(&rig()))
                .expect("a cube is a garment");
        assert_eq!(asset.validate(), Ok(()));
        // Eight WELDED corners — not 24 split ones. This is the assertion that
        // says the positions came off the half-edge mesh and not off the export
        // writer, which splits corners and would hand the solver a cube whose
        // faces are not joined to each other.
        assert_eq!(report.particles, 8, "the corners must be welded");
        assert_eq!(report.triangles, 12, "six quads, fan-triangulated");
        assert!(report.stretch >= 12, "a cube has at least its 12 edges");
        assert!(
            report.bend > 0,
            "an interior edge must produce a bend spring"
        );
        assert_eq!(report.pinned, 0, "nothing was selected");
        // Two bones -> two capsules.
        assert_eq!(report.capsules, 2);
        // Deterministic: the same mesh and the same spec build the same bytes.
        let (again, _) =
            garment_from_session(&mesh, &sel, SOURCE, GarmentSpec::default(), Some(&rig()))
                .unwrap();
        assert_eq!(
            inf_asset::encode(&asset).unwrap(),
            inf_asset::encode(&again).unwrap()
        );
    }

    /// **The selection is the pin list.** Selecting four corners pins exactly
    /// those particles and leaves the rest free — with a control that the same
    /// mesh with nothing selected pins none, so "pinned" is a statement about the
    /// selection rather than about the mesh.
    #[test]
    fn the_selected_vertices_are_the_pins() {
        let mesh = cube();
        let mut sel = empty_selection(&mesh);
        let picked: Vec<_> = mesh.vert_ids().take(4).collect();
        for v in &picked {
            sel.set_vert(*v, true);
        }
        assert_eq!(sel.len(SelectMode::Vert), 4);
        let (asset, report) =
            garment_from_session(&mesh, &sel, SOURCE, GarmentSpec::default(), None).unwrap();
        assert_eq!(report.pinned, 4);
        let pinned = asset.inv_mass.iter().filter(|m| **m == 0.0).count();
        assert_eq!(pinned, 4, "a pin is an inverse mass of zero");
        assert_eq!(asset.inv_mass.len() - pinned, 4, "the rest stay free");
        // The control.
        let (_, none) = garment_from_session(
            &mesh,
            &empty_selection(&mesh),
            SOURCE,
            GarmentSpec::default(),
            None,
        )
        .unwrap();
        assert_eq!(none.pinned, 0);
        // …and no skeleton is no capsules, rather than a panic or joint 0.
        assert_eq!(none.capsules, 0);
    }

    /// A body radius of zero derives no capsules — a value, not a refusal.
    #[test]
    fn a_zero_body_radius_derives_no_capsules() {
        let mesh = cube();
        let sel = empty_selection(&mesh);
        let spec = GarmentSpec {
            body_radius_m: 0.0,
            ..GarmentSpec::default()
        };
        let (_, report) = garment_from_session(&mesh, &sel, SOURCE, spec, Some(&rig())).unwrap();
        assert_eq!(report.capsules, 0);
    }

    /// **The hair door grows one guide per selected face**, rooted at the
    /// centroid and pointing along the face normal.
    #[test]
    fn the_selected_faces_are_the_scalp() {
        let mesh = cube();
        let mut sel = empty_selection(&mesh);
        let faces: Vec<_> = mesh.face_ids().take(2).collect();
        for f in &faces {
            sel.set_face(*f, true);
        }
        let (asset, report) =
            groom_from_session(&mesh, &sel, SOURCE, GroomSpec::default(), Some(&rig()))
                .expect("two faces grow two guides");
        assert_eq!(asset.validate(), Ok(()));
        assert_eq!(report.strands, 2);
        assert_eq!(report.particles, 2 * 7, "6 segments is 7 particles");
        assert_eq!(report.capsules, 2);
        // The roots sit ON the faces they grew from, and the strands leave along
        // the outward normal — a strand growing INTO the head is the failure this
        // catches (a centroid plus an inward normal would put the tip inside).
        for (s, f) in asset.strands.iter().zip(&faces) {
            let n = inf_dcc::face_normal(&mesh, *f).unwrap();
            let root = glam::Vec3::from_array(s.points[0]);
            let tip = glam::Vec3::from_array(s.points[s.len() - 1]);
            let along = (tip - root).normalize();
            let outward = glam::Vec3::new(n.x as f32, n.y as f32, n.z as f32).normalize();
            assert!(
                along.dot(outward) > 0.9,
                "strand runs {along:?}, face normal is {outward:?}"
            );
        }
        // Deterministic.
        let (again, _) =
            groom_from_session(&mesh, &sel, SOURCE, GroomSpec::default(), Some(&rig())).unwrap();
        assert_eq!(asset.strands, again.strands);
    }

    /// **No selection is a refusal by name**, not a hairstyle with no strands
    /// and not a whole-mesh scalp.
    #[test]
    fn growing_hair_with_nothing_selected_is_refused() {
        let mesh = cube();
        let sel = empty_selection(&mesh);
        assert_eq!(
            groom_from_session(&mesh, &sel, SOURCE, GroomSpec::default(), None).unwrap_err(),
            GroomError::NoScalp
        );
    }

    /// **Clump numbering is spatial, and it is a function of the geometry.**
    ///
    /// A sweep rather than two magic numbers, because the grid is anchored at the
    /// world ORIGIN and not at the model: the cube straddles zero, so even a cell
    /// far wider than the whole mesh still splits it along the axis planes. That
    /// is a real property of a world-anchored grid (two parts of one character
    /// clump consistently wherever the character stands), so what is asserted is
    /// what actually holds — bigger cells never produce MORE clumps, the tightest
    /// spacing isolates every root, and a wide cell merges something.
    #[test]
    fn clump_numbers_come_from_the_grid_not_from_face_order() {
        let mesh = cube();
        let mut sel = empty_selection(&mesh);
        for f in mesh.face_ids() {
            sel.set_face(f, true);
        }
        let with = |spacing: f32| {
            let spec = GroomSpec {
                clump_spacing_m: spacing,
                ..GroomSpec::default()
            };
            groom_from_session(&mesh, &sel, SOURCE, spec, None)
                .unwrap()
                .1
        };
        let counts: Vec<usize> = [0.02f32, 0.4, 1.2, 4.0]
            .iter()
            .map(|s| with(*s).clumps)
            .collect();
        assert_eq!(counts[0], 6, "2 cm cells separate every face");
        assert!(
            counts.windows(2).all(|w| w[1] <= w[0]),
            "a wider cell must never produce MORE clumps: {counts:?}"
        );
        assert!(
            *counts.last().unwrap() < 6,
            "a cell wider than the model must merge something: {counts:?}"
        );
        // Zero spacing is "no clumping", spelled as data rather than as a flag.
        assert_eq!(with(0.0).clumps, 6);
    }

    /// **An unskinned scalp takes the fallback joint and SAYS SO.**
    ///
    /// The fallback exists and is counted, so "every root took joint 0" and
    /// "every root read a weight" are different numbers rather than the same
    /// silence.
    #[test]
    fn an_unskinned_scalp_takes_the_fallback_joint_and_says_so() {
        let mesh = cube();
        let mut sel = empty_selection(&mesh);
        sel.set_face(mesh.face_ids().next().unwrap(), true);
        let spec = GroomSpec {
            fallback_joint: 2,
            ..GroomSpec::default()
        };
        let (asset, report) = groom_from_session(&mesh, &sel, SOURCE, spec, None).unwrap();
        assert_eq!(report.skinned_roots, 0, "a cube carries no skin weights");
        assert_eq!(asset.strands[0].root_joint, 2, "the fallback was used");
    }

    /// A degenerate spec is refused by the payload builder, and its message
    /// reaches the caller rather than being flattened into "failed".
    #[test]
    fn a_degenerate_groom_is_refused_with_the_payloads_own_words() {
        let mesh = cube();
        let mut sel = empty_selection(&mesh);
        sel.set_face(mesh.face_ids().next().unwrap(), true);
        let spec = GroomSpec {
            segments: 0,
            ..GroomSpec::default()
        };
        let err = groom_from_session(&mesh, &sel, SOURCE, spec, None).unwrap_err();
        assert!(
            matches!(&err, GroomError::Payload(m) if !m.is_empty()),
            "expected the payload's own refusal, got {err:?}"
        );
    }
}
