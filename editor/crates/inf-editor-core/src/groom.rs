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
    /// A file the headless door was handed could not be read as what it claimed
    /// to be, or a rule it was given selects nothing (wave CHAR1b.2).
    #[error("{what} could not be used: {why}")]
    Unreadable {
        /// Which input.
        what: &'static str,
        /// The decoder's or the rule's own words.
        why: String,
    },
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

/// **The headless garment door** (wave CHAR1b.2, the CHAR1a audit's item 87) —
/// author a `.inf_cloth` from files, with no open Model-Editor session.
///
/// # Why this exists
///
/// [`garment_from_session`] is the *authoring* door and it takes a live
/// half-edge [`Mesh`] and a [`SelectionSet`] — which is to say it takes an open
/// Model Editor. That made a garment something only a person with the editor
/// running could produce: no CLI, no CI arm, no script, and no way for a wave to
/// put a cape on a character and photograph it without a human clicking. The
/// audit recorded that as *"`inf character garment` so a cloth can be authored
/// without an open Model-Editor session"*.
///
/// This is that door, and it is deliberately the SAME function underneath:
/// nothing here decides what a garment is. It reads the two files, turns the
/// mesh into the kernel's own representation ([`inf_dcc::build::from_mesh_asset`],
/// the Model Editor's own reader), makes the pin selection out of a rule instead
/// of out of a click, and calls [`garment_from_session`].
///
/// # The pin rule, and why it is a rule rather than a click
///
/// `pin_top` is the fraction of the garment's own height, measured from its
/// highest vertex down, whose vertices are pinned. `0.05` on a cape is its
/// collar; `0.0` pins nothing (a free sheet). A rule and not a click because the
/// interesting property of a headless door is that it is REPRODUCIBLE: the same
/// two files and the same fraction give the same `.inf_cloth`, byte for byte,
/// which is what lets a gate assert on one.
///
/// A garment whose rule selects **no** vertices is refused rather than written:
/// an unpinned cape falls off the character on the first step, and that is a
/// mistake worth a message rather than a file.
pub fn garment_from_files(
    mesh_bytes: &[u8],
    skeleton_bytes: Option<&[u8]>,
    source_mesh: [u8; 16],
    pin_top: f64,
    spec: GarmentSpec,
) -> Result<(ClothAsset, GarmentReport), GroomError> {
    let asset: inf_mesh::MeshAsset =
        inf_asset::decode(mesh_bytes).map_err(|e| GroomError::Unreadable {
            what: "the garment mesh",
            why: e.to_string(),
        })?;
    let import = inf_dcc::build::from_mesh_asset(&asset).map_err(|e| GroomError::Unreadable {
        what: "the garment mesh",
        why: e.to_string(),
    })?;
    let mesh = import.mesh;
    let skeleton: Option<inf_anim::SkeletonAsset> = match skeleton_bytes {
        Some(b) => Some(inf_asset::decode(b).map_err(|e| GroomError::Unreadable {
            what: "the wearer's skeleton",
            why: e.to_string(),
        })?),
        None => None,
    };
    // ── the pins, by the rule ────────────────────────────────────────────────
    let mut selection = SelectionSet::new(0);
    if pin_top > 0.0 {
        let hi = mesh
            .vert_ids()
            .filter_map(|v| mesh.position(v))
            .fold(f64::NEG_INFINITY, |a, p| a.max(p.y));
        let lo = mesh
            .vert_ids()
            .filter_map(|v| mesh.position(v))
            .fold(f64::INFINITY, |a, p| a.min(p.y));
        let span = hi - lo;
        if !span.is_finite() || span <= 0.0 {
            return Err(GroomError::Unreadable {
                what: "the garment mesh",
                why: "it has no height at all, so a top-fraction pin rule cannot name a collar"
                    .to_string(),
            });
        }
        let cut = hi - span * pin_top.clamp(0.0, 1.0);
        for v in mesh.vert_ids() {
            if mesh.position(v).is_some_and(|p| p.y >= cut) {
                selection.set_vert(v, true);
            }
        }
        if selection.verts().is_empty() {
            return Err(GroomError::Unreadable {
                what: "the pin rule",
                why: format!(
                    "the top {:.1} % of a {span:.3} m garment holds no vertex, so nothing would \
                     hold it up",
                    pin_top * 100.0
                ),
            });
        }
    }
    garment_from_session(
        &mesh,
        &selection,
        source_mesh,
        spec,
        skeleton.as_ref().map(|s| &s.skeleton),
    )
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

// ── WEARABLE SHELLS (wave OUTFIT1) ──────────────────────────────────────────

/// **Which part of a body a shell wearable covers**, named by the joints that
/// deform it rather than by a height band.
///
/// A height band was the first design and it is wrong on a bind pose: a
/// mannequin's hands hang at hip height, so "everything between 5 % and 55 % of
/// the body" is a pair of trousers **and a pair of gloves**. The joint that owns
/// a vertex is what the author actually means — a sleeve stops at the elbow
/// because it stops at `lowerarm`, on any pose, on any proportions, and on both
/// the generated rig and the MetaHuman body rig, which share UE's bone names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellPart {
    /// A short-sleeved tee: the spine, the collar, the shoulders and the upper
    /// arms.
    Shirt,
    /// Trousers: the pelvis, the thighs and the calves — never the feet.
    Trousers,
}

impl ShellPart {
    /// Whether a vertex whose dominant joint is `joint` belongs to this part.
    ///
    /// Prefix matching on UE's bone names, which is what both rigs in this tree
    /// use: `spine_01…spine_05`, `clavicle_l/r`, `upperarm_l/r` and their twists,
    /// `thigh_l/r`, `calf_l/r`, `head`. A name this does not recognise belongs to
    /// no part, which is the safe answer — an unrecognised rig produces an empty
    /// garment and the door refuses with a count rather than dressing a
    /// character in its own eyeballs.
    fn owns(self, joint: &str) -> bool {
        let j = joint.to_ascii_lowercase();
        match self {
            // `neck_01` and not `neck_02`: a collar sits on the base of the neck.
            ShellPart::Shirt => {
                j.starts_with("spine")
                    || j.starts_with("clavicle")
                    || j.starts_with("upperarm")
                    || j == "neck_01"
            }
            // `foot` and `ball` are deliberately absent: these are trousers, and
            // this engine has no shoe.
            ShellPart::Trousers => j == "pelvis" || j.starts_with("thigh") || j.starts_with("calf"),
        }
    }

    /// How far the shell stands off the skin, metres.
    ///
    /// Trousers stand off further than the shirt so a tucked hem does not
    /// z-fight the waistband.
    fn offset_m(self) -> f32 {
        match self {
            ShellPart::Shirt => 0.010,
            ShellPart::Trousers => 0.014,
        }
    }
}

/// What [`wearable_shell`] produced, so a caller can say it rather than open the
/// asset to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellReport {
    /// Vertices kept.
    pub vertices: usize,
    /// Triangles kept.
    pub triangles: usize,
    /// Vertices the part owns (the selection), before triangles are filtered.
    pub selected: usize,
}

/// **A fitted garment, shrink-wrapped off the body that wears it** (wave
/// OUTFIT1) — the door the committed default characters get dressed through.
///
/// # Why a shell and not a modelled garment
///
/// The public repository's default character has stood in the pipeline's default
/// underwear since it existed, and the two ways out of that are to commit a
/// modelled tee (which is either licensed content this repository may not carry,
/// or a week of somebody's modelling) or to derive one. A shell is derived: take
/// the body's own surface over the joints a tee covers, push it out along its
/// normals by a centimetre, and keep the skin stream unchanged.
///
/// What that buys, and it is not a consolation prize:
///
/// * it **deforms perfectly**, because it is skinned by exactly the weights the
///   body under it is skinned by — no transfer, no proximity solve, no seam;
/// * it is **licence-free and ours**, being a function of a mesh this repository
///   already commits;
/// * it is **small** — a subset of one body's triangles;
/// * it is **deterministic**, so the committed bytes are reproducible and the
///   folder can be byte-locked like every other sample.
///
/// The honest bound: a shell has the silhouette of the body it came off, so this
/// is a *fitted* tee and *fitted* trousers. A garment with its own drape is a
/// `.inf_cloth` (see [`garment_from_files`]) or an imported mesh — both of which
/// ride the same wearable door.
///
/// # Refusals
///
/// An empty result is a refusal with the two counts, not an empty asset: a
/// garment with no triangles draws nothing, and a caller that wrote one would
/// have committed a file that looks like content and is not.
pub fn wearable_shell(
    body: &inf_mesh::MeshAsset,
    skeleton: &Skeleton,
    part: ShellPart,
    slot_name: &str,
) -> Result<(inf_mesh::MeshAsset, ShellReport), GroomError> {
    let offset = part.offset_m();
    let mut verts: Vec<inf_mesh::MeshVertex> = Vec::new();
    let mut skin: Vec<inf_mesh::VertexSkin> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut selected_total = 0usize;

    for sm in &body.submeshes {
        if sm.skin.len() != sm.vertices.len() {
            // A rigid submesh cannot be dressed: there is no joint to ask.
            continue;
        }
        let keep: Vec<bool> = sm
            .vertices
            .iter()
            .zip(sm.skin.iter())
            .map(|(_, k)| dominant_joint_name(skeleton, k).map(|n| part.owns(n)) == Some(true))
            .collect();
        selected_total += keep.iter().filter(|k| **k).count();
        // Remap lazily: only the vertices a KEPT triangle actually references
        // reach the output, so a stray owned vertex with no owned neighbours
        // costs nothing.
        let mut remap: Vec<u32> = vec![u32::MAX; sm.vertices.len()];
        for tri in sm.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if a >= keep.len() || b >= keep.len() || c >= keep.len() {
                continue;
            }
            if !(keep[a] && keep[b] && keep[c]) {
                continue;
            }
            for &i in tri {
                let i = i as usize;
                if remap[i] == u32::MAX {
                    let mut v = sm.vertices[i];
                    let n = v.normal;
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > 1e-6 {
                        for axis in 0..3 {
                            v.position[axis] += n[axis] / len * offset;
                        }
                    }
                    remap[i] = verts.len() as u32;
                    verts.push(v);
                    skin.push(sm.skin[i]);
                }
                indices.push(remap[i]);
            }
        }
    }

    if indices.len() < 3 {
        return Err(GroomError::EmptyMesh {
            verts: selected_total,
            faces: indices.len() / 3,
        });
    }
    let report = ShellReport {
        vertices: verts.len(),
        triangles: indices.len() / 3,
        selected: selected_total,
    };
    let sub = inf_mesh::SubMesh {
        name: slot_name.to_string(),
        vertices: verts,
        indices,
        material_slot: Some(0),
        skin,
    };
    Ok((
        inf_mesh::MeshAsset::new(vec![sub], vec![slot_name.to_string()]),
        report,
    ))
}

/// How far down the head a hair cap reaches, as a fraction of the head's own
/// vertical span measured from its crown.
///
/// 0.45 leaves the face: the generated body's head spans 0.224 m and its brow
/// sits a little under halfway up, so a cap that stops 45 % of the way down
/// covers the skull and the crown of the ears and nothing else.
pub const HAIR_CAP_DROP: f32 = 0.45;

/// How far a hair cap stands off the skull it is fitted to, metres.
pub const HAIR_CAP_OFFSET_M: f32 = 0.008;

/// The cap's latitude rings — the committed default, and small on purpose: 6 ×
/// 20 is 240 triangles, which is 4 % of the committed body's 5 718 and reads as
/// a smooth skull at portrait distance.
pub const HAIR_CAP_RINGS: usize = 6;

/// The cap's longitude segments — see [`HAIR_CAP_RINGS`].
pub const HAIR_CAP_SEGMENTS: usize = 20;

/// **A hair cap, generated over the head the rig actually has** (wave OUTFIT1).
///
/// # Why this is generated and the clothes are shrink-wrapped
///
/// [`wearable_shell`] takes the body's own surface, and that works for a shirt
/// because a torso has surface to spare. It does not work for hair: measured on
/// the committed body, the head owns **68** vertices in total and the crown band
/// of it closes exactly **one** triangle. A shell there is not a haircut, it is a
/// triangle. So the cap is authored geometry — an ellipsoidal dome fitted to the
/// measured head, at whatever tessellation the caller asks for — and it is bound
/// **rigidly to the head joint**, weight 1, which is what a real hair card set is
/// bound to as well.
///
/// That rigid bind is the whole reason it can ride the wearables door: the mesh
/// names the wearer's own skeleton, so `inf_ecs::wearable` reads it as a
/// wearable, it shares the wearer's palette, and the head joint's animated
/// transform carries it. No socket, no attachment pass, no second pose.
///
/// # The honest bound
///
/// This is a CAP, not carded hair. Cards are quads with an alpha mask cut out of
/// them, and this repository commits no hair alpha texture — a card without one
/// is a solid quad, which is worse than a dome. Carded hair reaches the engine
/// through the bridge (the MetaHuman grooms ship `<Groom>_CardsMesh_*` meshes and
/// their masked materials), and it rides this same wearables door when it does.
///
/// Refuses, with a count, when the rig has no `head` joint or when no vertex is
/// deformed by it — a cap fitted to nothing would be a dome at the world origin.
pub fn hair_cap(
    body: &inf_mesh::MeshAsset,
    skeleton: &Skeleton,
    rings: usize,
    segments: usize,
) -> Result<(inf_mesh::MeshAsset, ShellReport), GroomError> {
    let Some(head) = skeleton.index_of("head") else {
        return Err(GroomError::EmptyMesh { verts: 0, faces: 0 });
    };
    // The head's own box, measured off the vertices it deforms. Not the mesh
    // bounds and not the joint's position: a joint is a point and a skull is a
    // volume, and the cap has to sit on the volume.
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    let mut owned = 0usize;
    for sm in &body.submeshes {
        if sm.skin.len() != sm.vertices.len() {
            continue;
        }
        for (v, k) in sm.vertices.iter().zip(sm.skin.iter()) {
            if dominant_joint_name(skeleton, k) != Some("head") {
                continue;
            }
            owned += 1;
            for axis in 0..3 {
                lo[axis] = lo[axis].min(v.position[axis]);
                hi[axis] = hi[axis].max(v.position[axis]);
            }
        }
    }
    let rings = rings.max(2);
    let segments = segments.max(3);
    if owned == 0 {
        return Err(GroomError::EmptyMesh { verts: 0, faces: 0 });
    }
    // The crown sits `HAIR_CAP_OFFSET_M` ABOVE the highest vertex the head
    // deforms, in all three axes and not only sideways: a dome whose top was the
    // skull's top exactly let the skull through it by however much the offset
    // pushed the sides out — measured at 7 mm on the committed body, which is a
    // scalp poking through its own hair.
    let centre = [
        (lo[0] + hi[0]) * 0.5,
        hi[1] + HAIR_CAP_OFFSET_M,
        (lo[2] + hi[2]) * 0.5,
    ];
    let radius = [
        (hi[0] - lo[0]) * 0.5 + HAIR_CAP_OFFSET_M,
        (hi[1] - lo[1]) * 0.5 + HAIR_CAP_OFFSET_M,
        (hi[2] - lo[2]) * 0.5 + HAIR_CAP_OFFSET_M,
    ];
    // The dome is an ellipsoid centred half a head down from the crown, so its
    // top touches the crown and its equator is the widest part of the skull.
    let cy = centre[1] - radius[1];
    // …and it stops `HAIR_CAP_DROP` of the head's height below the crown.
    let span = hi[1] - lo[1];
    let cut_y = hi[1] - span * HAIR_CAP_DROP;
    let cos_max = (((cut_y - cy) / radius[1]) as f64).clamp(-1.0, 1.0);
    // `inf_math::pacos64` and not `f64::acos`: this angle decides the geometry of
    // a mesh this repository COMMITS, and a std transcendental is not
    // bit-portable (the P14 law). The same reason the ring and segment loops
    // below take `psin64`/`pcos64`.
    let theta_max = inf_math::pacos64(cos_max);

    let mut verts: Vec<inf_mesh::MeshVertex> = Vec::new();
    let mut skin: Vec<inf_mesh::VertexSkin> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let bind = inf_mesh::VertexSkin {
        joints: [head, 0, 0, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    // `inf_math::psin`/`pcos` and not `f32::sin`: this geometry is COMMITTED, and
    // the P14 law is that f32 std trig is not bit-portable, so a cap generated on
    // one machine and blessed on another would differ in its last bits.
    for r in 0..=rings {
        let theta = theta_max * (r as f64) / (rings as f64);
        let (st, ct) = (inf_math::psin64(theta), inf_math::pcos64(theta));
        for sgm in 0..segments {
            let phi = std::f64::consts::TAU * (sgm as f64) / (segments as f64);
            let (sp, cp) = (inf_math::psin64(phi), inf_math::pcos64(phi));
            let n = [(st * cp) as f32, ct as f32, (st * sp) as f32];
            let position = [
                centre[0] + radius[0] * n[0],
                cy + radius[1] * n[1],
                centre[2] + radius[2] * n[2],
            ];
            verts.push(inf_mesh::MeshVertex {
                position,
                normal: n,
                uv: [
                    (sgm as f32) / (segments as f32),
                    (r as f32) / (rings as f32),
                ],
                ..Default::default()
            });
            skin.push(bind);
        }
    }
    for r in 0..rings {
        for sgm in 0..segments {
            let next = (sgm + 1) % segments;
            let a = (r * segments + sgm) as u32;
            let b = (r * segments + next) as u32;
            let c = ((r + 1) * segments + sgm) as u32;
            let d = ((r + 1) * segments + next) as u32;
            // Wound so the outside faces out: the ring below is further from the
            // crown, so `a, c, b` and `b, c, d` are counter-clockwise seen from
            // outside the dome.
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let report = ShellReport {
        vertices: verts.len(),
        triangles: indices.len() / 3,
        selected: owned,
    };
    let sub = inf_mesh::SubMesh {
        name: "hair_cap".to_string(),
        vertices: verts,
        indices,
        material_slot: Some(0),
        skin,
    };
    Ok((
        inf_mesh::MeshAsset::new(vec![sub], vec!["hair_cap".to_string()]),
        report,
    ))
}

/// The name of the joint a skinned vertex is most influenced by, or `None` when
/// the rig does not have it.
///
/// The heaviest of the four influences, ties broken by the LOWER joint index so
/// the answer is a function of the data rather than of iteration order — the
/// same rule the cloth pin solve uses one door up.
fn dominant_joint_name<'s>(skeleton: &'s Skeleton, skin: &inf_mesh::VertexSkin) -> Option<&'s str> {
    let mut best: Option<(u16, f32)> = None;
    for (j, w) in skin.joints.iter().zip(skin.weights.iter()) {
        match best {
            Some((_, bw)) if *w <= bw => {}
            _ => best = Some((*j, *w)),
        }
    }
    let (joint, weight) = best?;
    if weight <= 0.0 {
        return None;
    }
    skeleton.joint(joint as usize).map(|j| j.name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_dcc::{Mesh, SelectMode, SelectionSet};

    const SOURCE: [u8; 16] = [0x24; 16];

    /// **The committed default body's clothes, measured on the committed
    /// body** (wave OUTFIT1). Not a synthetic fixture: the shell is only worth
    /// anything if it covers what a person would call a shirt on the mesh this
    /// repository actually ships, and the two numbers that matter are what it
    /// KEEPS and what it must never keep.
    ///
    /// The hands are the arm this exists for. A height band over a bind pose
    /// puts a mannequin's hands at hip height and dresses them in trousers; the
    /// joint rule cannot, and the assertion is that not one hand-owned vertex
    /// survives into either garment.
    #[test]
    fn the_committed_bodys_shell_garments_cover_a_body_and_never_its_hands() {
        let dir = crate::samples::starter_character_dir();
        let body: inf_mesh::MeshAsset =
            inf_asset::decode(&std::fs::read(dir.join("Starter_Body.inf_mesh")).expect("body"))
                .expect("decode body");
        let rig: inf_anim::SkeletonAsset =
            inf_asset::decode(&std::fs::read(dir.join("Starter.inf_skel")).expect("rig"))
                .expect("decode rig");
        let skeleton = &rig.skeleton;

        let body_tris: usize = body.submeshes.iter().map(|s| s.triangle_count()).sum();
        for (part, slot) in [
            (ShellPart::Shirt, "outfit_shirt"),
            (ShellPart::Trousers, "outfit_trousers"),
        ] {
            let (mesh, report) =
                wearable_shell(&body, skeleton, part, slot).expect("the shell is empty");
            println!(
                "{part:?}: {} verts, {} tris of the body's {body_tris} ({} selected)",
                report.vertices, report.triangles, report.selected
            );
            assert!(
                report.triangles > 0 && report.triangles < body_tris,
                "{part:?} kept {} of {body_tris} triangles",
                report.triangles
            );
            // Every kept vertex is still skinned, index-aligned, and normalized:
            // a garment that lost its skin stream draws in bind pose for ever.
            let sub = &mesh.submeshes[0];
            assert_eq!(sub.skin.len(), sub.vertices.len(), "{part:?} lost its skin");
            // THE HANDS. Nothing a hand deforms may be in a shirt or a pair of
            // trousers, whatever the pose put the hands next to.
            let hand_owned = sub
                .skin
                .iter()
                .filter(|k| {
                    dominant_joint_name(skeleton, k)
                        .map(|n| n.starts_with("hand") || n.starts_with("index"))
                        == Some(true)
                })
                .count();
            assert_eq!(hand_owned, 0, "{part:?} is dressing the character's hands");
            // …and it stands OFF the skin rather than z-fighting it.
            assert!(
                mesh.bounds.max[0] - mesh.bounds.min[0] > 0.0,
                "{part:?} is degenerate"
            );
        }

        // THE HAIR CAP is generated, not shrink-wrapped, and the reason is a
        // measurement: the committed body's head owns 68 vertices in total and a
        // crown band of it closes ONE triangle. The cap is a dome fitted to the
        // measured skull, rigidly bound to the head joint — which is what makes
        // it a wearable at all — and it stops above the shirt's collar.
        let (cap, cap_report) = hair_cap(&body, skeleton, 6, 20).expect("the hair cap is empty");
        println!(
            "HairCap: {} verts, {} tris over {} head vertices",
            cap_report.vertices, cap_report.triangles, cap_report.selected
        );
        let head = skeleton.index_of("head").expect("the rig has a head");
        assert!(
            cap.submeshes[0]
                .skin
                .iter()
                .all(|k| k.joints[0] == head && (k.weights[0] - 1.0).abs() < 1e-6),
            "the hair cap is not rigidly bound to the head joint"
        );
        // THE ANATOMICAL CLAIM, and not a restatement of the constant: the cap
        // is entirely above the HEAD JOINT, which is the base of the skull. A
        // drop that grew until it covered the face would put vertices below it.
        let globals = inf_anim::global_transforms(skeleton, &inf_anim::Pose::rest(skeleton));
        let head_y = globals[head as usize].w_axis.y;
        println!(
            "  head joint at {head_y:.3} m; cap {:.3}..{:.3} m; body crown {:.3} m",
            cap.bounds.min[1], cap.bounds.max[1], body.bounds.max[1]
        );
        assert!(
            cap.bounds.min[1] > head_y,
            "the hair cap reaches {:.3} m, below the head joint at {head_y:.3} m — it is on the neck",
            cap.bounds.min[1]
        );
        // …and it sits ON the skull rather than floating: its crown is within a
        // centimetre and a half of the body's own highest point.
        assert!(
            (cap.bounds.max[1] - body.bounds.max[1]).abs() < 0.015,
            "the cap's crown is {:.3} m and the body's is {:.3} m",
            cap.bounds.max[1],
            body.bounds.max[1]
        );
    }

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
