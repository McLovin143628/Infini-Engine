//! **THE SKELETAL VEHICLE SPLIT** (wave VEH3f.2a, the bridge's gap B3) -- one
//! skinned Unreal car in, a chassis, four wheels, its doors, its panes, its
//! steering wheel and a seat out, each a RIGID mesh the engine draws on the part
//! the VEH3a-d rig already moves.
//!
//! # What the packs give, measured
//!
//! Every car pack on this machine ships its vehicles as ONE skinned mesh whose
//! wheels (and, on the calibration pack, whose two front doors) are BONES: every
//! vertex of a wheel is weighted 1.0 to that wheel's bone, every vertex of a door
//! to the door's hinge bone, and the rest to the root (read off the glTF the
//! exporter wrote: 7 joints on `SK_sedane_LOD0`, `Left_Door` at the hinge,
//! `(0.85, 0.786, -0.85)` in the glTF frame). So a triangle's PART is its dominant
//! joint, and a part's pivot is that joint's rest position -- which is exactly
//! where the rig must turn it.
//!
//! # The frame
//!
//! Unreal's glTF exporter writes UE `(x, y, z)` cm as glTF `(x, z, y)` / 100
//! (measured: the sedan's front-left wheel bone, UE `(141.49, -68.48, 36.31)`,
//! arrives at `(1.4149, 0.3631, -0.6848)`). A car pack faces UE `+X`, so it
//! arrives facing glTF `+X`; this engine's forward is `+Z`. [`to_engine`] turns
//! `+X` onto `+Z` about the vertical -- `(x, y, z) -> (-z, y, x)`, a proper
//! rotation, so no winding flips -- and the construction pack (`+Y` in UE, glTF
//! `+Z`) keeps the identity VEH3f measured. The facing is the MANIFEST's field,
//! never an assumption.
//!
//! # What each part is drawn in
//!
//! * the **body**: metres, centred on the collider its geometry implies (the
//!   VEH3f rule), drawn by the rig's `art_body` part at the identity;
//! * a **wheel**: metres, centred on its bone, axle on `X` -- the wheel entity's
//!   own spin and steer turn it about the bone;
//! * a **door**, a **pane**, a **seat**: in the part's own UNIT BOX, so the
//!   part's scale restores metres exactly (the DCC hero panels' rule) and the
//!   bodywork moves the mesh exactly as it moves the box. A door's box is chosen
//!   with its FRONT edge and lateral CENTRE on the hinge bone, which is where
//!   `inf_ecs::vehicle::Hinge::of` puts every door's axis;
//! * the **steering wheel**: in its COLUMN frame (rim in local `XY`, column on
//!   local `+Z`, toward the driver), unit-boxed, so `hub_rim_euler` seats it on
//!   the PACK's measured column rake (the art table's `hub_rake_deg`) and rolls
//!   it with the rack; the grips follow the same plane.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_ecs::vehicle_art::Facing;
use inf_mesh::{Aabb, MeshAsset, MeshVertex, SubMesh};

/// What a material slot becomes in the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotRole {
    /// The car's paint: no material of its own -- the section wears the
    /// entity's (the roster row's paint).
    Paint,
    /// Window glass: a dark engine glass surface (a translucent pack glass
    /// drawn opaque would be a white sheet).
    Glass,
    /// Not drawn: a translucent lens or reflector cover the opaque meshlet path
    /// cannot draw honestly. Stated per slot in the vehicle TOML.
    Hidden,
    /// The pack's own material.
    Plain,
}

/// A pack-frame point into the engine frame. See the module note.
pub fn to_engine(f: Facing, p: DVec3) -> DVec3 {
    match f {
        Facing::PlusX => DVec3::new(-p.z, p.y, p.x),
        Facing::PlusY => p,
    }
}

fn v3(p: [f32; 3]) -> DVec3 {
    DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

fn f3(p: DVec3) -> [f32; 3] {
    [p.x as f32, p.y as f32, p.z as f32]
}

/// How close two vertices are to be one, metres.
const WELD_M: f64 = 1e-3;

/// The smallest glass piece that is a PANE (a VEH3c glass part), m^2 -- a quarter
/// light is ~0.1, a side mirror's glass ~0.02.
pub const PANE_MIN_AREA_M2: f64 = 0.08;

/// The most panes one body carries.
pub const MAX_PANES: usize = 8;

/// Where a body with no Blueprint seat puts its driver's cushion, behind the
/// drawn rim's centre, metres -- the three DrivableCars bodies' own Blueprint
/// seats measure 0.296, 0.328 and 0.313 m behind their rims. (The first cut
/// took 0.53 m, the Blueprint's `Driver` to its `steering_wheel` COMPONENT --
/// the column's root, not the rim -- and a driver seated there could not reach
/// the wheel: 83 mm short, measured by the gate's boarding arm.)
pub const SEAT_BEHIND_HUB_M: f64 = 0.31;

/// A door part's half-thickness, metres -- the families' own doors are 0.045 of
/// a saloon's half-width, about this.
pub const DOOR_HALF_THICK_M: f64 = 0.05;

/// A drawn cushion's half-extents, metres (the families' own seat size).
pub const SEAT_HALF_M: [f64; 3] = [0.24, 0.06, 0.25];

/// The steering wheel's input when the pack draws it as its OWN mesh (the
/// calibration pack) rather than skinned to a bone.
#[derive(Clone, Debug)]
pub struct SteeringMesh {
    /// The mesh, in its own glTF frame.
    pub mesh: MeshAsset,
    /// Its placement in the vehicle's glTF frame: rotation, then translation.
    pub rotation: glam::DQuat,
    pub translation: DVec3,
}

/// Everything the split reads.
#[derive(Clone, Debug)]
pub struct SkelVehicleIn<'a> {
    pub art: &'a str,
    pub facing: Facing,
    /// The skinned LOD-0 mesh, glTF frame, metres.
    pub mesh: &'a MeshAsset,
    /// The skin's joints: name and WORLD rest position, glTF frame.
    pub joints: &'a [(String, DVec3)],
    /// A role per material slot (glTF material index).
    pub roles: &'a [SlotRole],
    /// A separate steering-wheel mesh, when the pack draws one.
    pub steering: Option<SteeringMesh>,
    /// The Blueprint's driver seat, glTF frame, when it has one.
    pub seat: Option<DVec3>,
    /// The Blueprint's in-car camera, glTF frame.
    pub camera: Option<DVec3>,
    /// The pack has no door meshes: put INVISIBLE door proxies on the flanks
    /// beside the seat, so boarding has a door to open (stated as a proxy).
    pub door_proxy: bool,
}

/// One art part's mesh and box.
#[derive(Clone, Debug)]
pub struct ArtPartOut {
    /// The part's name -- the naming rule decides its kind.
    pub name: String,
    /// Its box, metres, relative to the collider centre.
    pub centre: DVec3,
    pub half: DVec3,
    /// Its mesh in its unit box (or, for the hub, its column frame unit box).
    /// `None` for a part the art has no geometry for (a seat).
    pub mesh: Option<MeshAsset>,
    /// Triangles.
    pub triangles: usize,
}

/// What one car split into.
#[derive(Clone, Debug)]
pub struct SkelVehicleSplit {
    /// The collider's half-extents, metres.
    pub half_extents: DVec3,
    /// The body, metres, collider-centred.
    pub body: MeshAsset,
    /// FL, FR, RL, RR (`VehicleDef::wheel_mounts`' order), hub-centred.
    pub wheels: Vec<MeshAsset>,
    /// Each wheel's hub (its bone), relative to the collider centre.
    pub hubs: Vec<DVec3>,
    /// The wheels' bone names, in the same order.
    pub wheel_bones: Vec<String>,
    /// The mean tyre radius measured off the geometry, metres.
    pub wheel_radius: f64,
    /// Doors, panes, the hub, the seats.
    pub parts: Vec<ArtPartOut>,
    /// The art's ground, relative to the collider centre (negative).
    pub ground_below: f64,
    /// The in-car camera, relative to the collider centre.
    pub camera: Option<DVec3>,
    /// The drawn steering wheel's column rake the pack authored, degrees, in
    /// `inf_ecs::vehicle::hub_rim_euler`'s convention: the rest pitch that
    /// reproduces the pack's own wheel exactly, and the grips' plane with it.
    pub pack_rake_deg: Option<f64>,
    /// Triangles in, and dropped as hidden.
    pub triangles: usize,
    pub hidden_triangles: usize,
}

impl SkelVehicleSplit {
    /// **The catalogue keys this body measures at** -- the row's geometry.
    pub fn geometry(&self) -> BTreeMap<&'static str, f64> {
        let mut g = BTreeMap::new();
        g.insert("half_width_m", self.half_extents.x);
        g.insert("half_height_m", self.half_extents.y);
        g.insert("half_length_m", self.half_extents.z);
        g.insert("ground_m", self.ground_below);
        if self.hubs.len() == 4 {
            let h = &self.hubs;
            g.insert("wheel_radius_m", self.wheel_radius);
            g.insert(
                "half_track_m",
                h.iter().map(|p| p.x.abs()).sum::<f64>() / 4.0,
            );
            let front = 0.5 * (h[0].z + h[1].z);
            let rear = 0.5 * (h[2].z + h[3].z);
            g.insert("half_wheelbase_m", 0.5 * (front - rear));
            g.insert("wheel_offset_z_m", 0.5 * (front + rear));
            g.insert("wheel_drop_m", h.iter().map(|p| p.y).sum::<f64>() / 4.0);
        }
        g
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Group {
    Chassis,
    Wheel(usize),
    Door(usize),
    Steering,
}

fn group_of(name: &str, joint: usize) -> Group {
    let n = name.to_ascii_lowercase();
    if n.contains("steer") {
        Group::Steering
    } else if n.contains("wheel") {
        Group::Wheel(joint)
    } else if n.contains("door") {
        Group::Door(joint)
    } else {
        Group::Chassis
    }
}

/// A vertex into the engine frame (position, normal, tangent; uv kept).
fn vert_engine(f: Facing, v: &MeshVertex, origin: DVec3) -> MeshVertex {
    let p = to_engine(f, v3(v.position)) - origin;
    let n = to_engine(f, v3(v.normal));
    let t = to_engine(
        f,
        DVec3::new(
            v.tangent[0] as f64,
            v.tangent[1] as f64,
            v.tangent[2] as f64,
        ),
    );
    MeshVertex {
        position: f3(p),
        normal: f3(n),
        uv: v.uv,
        tangent: [t.x as f32, t.y as f32, t.z as f32, v.tangent[3]],
    }
}

/// Rebuild a mesh from the triangles `pick` takes, each vertex through `map`.
/// Submesh order, slot index and vertex order are the source's, so two runs over
/// one input give identical bytes.
fn rebuild(
    src: &MeshAsset,
    pick: &dyn Fn(usize, usize) -> bool,
    map: &dyn Fn(&MeshVertex) -> MeshVertex,
) -> MeshAsset {
    let mut subs: Vec<SubMesh> = Vec::new();
    let mut bounds = Aabb::empty();
    for (si, s) in src.submeshes.iter().enumerate() {
        let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
        let mut verts: Vec<MeshVertex> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        for (ti, t) in s.indices.chunks_exact(3).enumerate() {
            if !pick(si, ti) {
                continue;
            }
            for i in t {
                let n = *remap.entry(*i).or_insert_with(|| {
                    verts.push(map(&s.vertices[*i as usize]));
                    (verts.len() - 1) as u32
                });
                idx.push(n);
            }
        }
        if idx.is_empty() {
            continue;
        }
        for v in &verts {
            bounds.grow(v.position);
        }
        subs.push(SubMesh {
            name: s.name.clone(),
            vertices: verts,
            indices: idx,
            material_slot: s.material_slot,
            skin: Vec::new(),
        });
    }
    MeshAsset {
        submeshes: subs,
        bounds,
        ..src.clone()
    }
}

/// Union-find over welded vertex keys.
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// **The symmetric 3x3 eigenvector with the SMALLEST eigenvalue** -- a thin
/// disc's normal. Cyclic Jacobi, a fixed 32 sweeps: deterministic for one input.
fn min_axis(c: [[f64; 3]; 3]) -> DVec3 {
    let mut a = c;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-18 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let cs = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * cs;
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            #[allow(clippy::needless_range_loop)]
            // a Jacobi rotation reads a[r][p] and writes a[p][r]
            for r in 0..3 {
                if r != p && r != q {
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = cs * arp - sn * arq;
                    a[p][r] = a[r][p];
                    a[r][q] = sn * arp + cs * arq;
                    a[q][r] = a[r][q];
                }
            }
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = cs * vp - sn * vq;
                row[q] = sn * vp + cs * vq;
            }
        }
    }
    let k = (0..3)
        .min_by(|i, j| a[*i][*i].total_cmp(&a[*j][*j]))
        .unwrap_or(0);
    DVec3::new(v[0][k], v[1][k], v[2][k]).normalize_or_zero()
}

/// **Split one skinned car.** See the module note.
pub fn split_skel_vehicle(i: &SkelVehicleIn) -> Result<SkelVehicleSplit, String> {
    let f = i.facing;
    let src = i.mesh;
    let role = |slot: Option<u32>| -> SlotRole {
        slot.and_then(|s| i.roles.get(s as usize).copied())
            .unwrap_or(SlotRole::Plain)
    };
    // ── every triangle's group (its dominant joint) ──
    let mut groups: Vec<Vec<Group>> = Vec::with_capacity(src.submeshes.len());
    let mut hidden = 0usize;
    for s in &src.submeshes {
        let mut g = Vec::with_capacity(s.indices.len() / 3);
        for t in s.indices.chunks_exact(3) {
            let mut w: BTreeMap<u16, f32> = BTreeMap::new();
            for k in t {
                if let Some(sk) = s.skin.get(*k as usize) {
                    for (j, wt) in sk.joints.iter().zip(sk.weights.iter()) {
                        if *wt > 0.0 {
                            *w.entry(*j).or_insert(0.0) += *wt;
                        }
                    }
                }
            }
            let joint = w
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
                .map(|(j, _)| *j as usize)
                .unwrap_or(0);
            let name = i.joints.get(joint).map(|j| j.0.as_str()).unwrap_or("");
            g.push(group_of(name, joint));
        }
        if role(s.material_slot) == SlotRole::Hidden {
            hidden += g.len();
        }
        groups.push(g);
    }
    let triangles: usize = src.submeshes.iter().map(|s| s.indices.len() / 3).sum();
    let live = |si: usize, _ti: usize| role(src.submeshes[si].material_slot) != SlotRole::Hidden;
    let in_group = |si: usize, ti: usize, g: Group| live(si, ti) && groups[si][ti] == g;
    // ── the engine-frame bounds of each group ──
    let mut body_lo = DVec3::splat(f64::MAX);
    let mut body_hi = DVec3::splat(f64::MIN);
    let mut ground = f64::MAX;
    for (si, s) in src.submeshes.iter().enumerate() {
        for (ti, t) in s.indices.chunks_exact(3).enumerate() {
            if !live(si, ti) {
                continue;
            }
            let g = groups[si][ti];
            for k in t {
                let p = to_engine(f, v3(s.vertices[*k as usize].position));
                match g {
                    Group::Chassis | Group::Door(_) => {
                        body_lo = body_lo.min(p);
                        body_hi = body_hi.max(p);
                    }
                    Group::Wheel(_) => ground = ground.min(p.y),
                    Group::Steering => {}
                }
            }
        }
    }
    if body_lo.x >= body_hi.x {
        return Err(format!("{}: no body geometry", i.art));
    }
    if ground == f64::MAX {
        ground = body_lo.y;
    }
    let clearance = (body_lo.y - ground).max(0.15);
    let collider_lo = ground + clearance;
    let centre = DVec3::new(
        0.5 * (body_lo.x + body_hi.x),
        0.5 * (collider_lo + body_hi.y),
        0.5 * (body_lo.z + body_hi.z),
    );
    let half = DVec3::new(
        0.5 * (body_hi.x - body_lo.x),
        0.5 * (body_hi.y - collider_lo),
        0.5 * (body_hi.z - body_lo.z),
    );
    let joint_at = |j: usize| -> DVec3 {
        i.joints
            .get(j)
            .map(|(_, p)| to_engine(f, *p) - centre)
            .unwrap_or(DVec3::ZERO)
    };
    // ── the wheels ──
    let mut wheel_joints: Vec<usize> = groups
        .iter()
        .flatten()
        .filter_map(|g| match g {
            Group::Wheel(j) => Some(*j),
            _ => None,
        })
        .collect();
    wheel_joints.sort();
    wheel_joints.dedup();
    // FL, FR, RL, RR: front is +Z; index 0 is the -X side (`wheel_mounts`).
    let mut ordered: Vec<(usize, DVec3)> =
        wheel_joints.iter().map(|j| (*j, joint_at(*j))).collect();
    let corner = |p: DVec3| -> u8 {
        let front = p.z > 0.0;
        match (front, p.x < 0.0) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        }
    };
    ordered.sort_by(|a, b| corner(a.1).cmp(&corner(b.1)).then(a.0.cmp(&b.0)));
    let four = ordered.len() == 4 && (0..4).all(|k| corner(ordered[k].1) == k as u8);
    if !four {
        return Err(format!(
            "{}: {} wheel bones, not one on each corner -- not a four-wheeled car",
            i.art,
            ordered.len()
        ));
    }
    let mut wheels = Vec::new();
    let mut hubs = Vec::new();
    let mut wheel_bones = Vec::new();
    let mut radii = Vec::new();
    for (j, hub) in &ordered {
        let origin = centre + *hub;
        let w = rebuild(src, &|si, ti| in_group(si, ti, Group::Wheel(*j)), &|v| {
            vert_engine(f, v, origin)
        });
        let mut r = 0.0f64;
        for s in &w.submeshes {
            for v in &s.vertices {
                let p = v3(v.position);
                r = r.max((p.y * p.y + p.z * p.z).sqrt());
            }
        }
        radii.push(r);
        wheels.push(w);
        hubs.push(*hub);
        wheel_bones.push(i.joints.get(*j).map(|x| x.0.clone()).unwrap_or_default());
    }
    let wheel_radius = radii.iter().sum::<f64>() / radii.len() as f64;
    // ── the glass panes, out of the body's own glass ──
    let mut pane_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut panes: Vec<ArtPartOut> = Vec::new();
    {
        let mut keys: BTreeMap<(i64, i64, i64), usize> = BTreeMap::new();
        let mut parent: Vec<usize> = Vec::new();
        let mut tri_node: Vec<((usize, usize), usize, f64)> = Vec::new();
        for (si, s) in src.submeshes.iter().enumerate() {
            if role(s.material_slot) != SlotRole::Glass {
                continue;
            }
            for (ti, t) in s.indices.chunks_exact(3).enumerate() {
                if groups[si][ti] != Group::Chassis {
                    continue;
                }
                let ps: Vec<DVec3> = t
                    .iter()
                    .map(|k| to_engine(f, v3(s.vertices[*k as usize].position)))
                    .collect();
                let area = 0.5 * (ps[1] - ps[0]).cross(ps[2] - ps[0]).length();
                let mut nodes = [0usize; 3];
                for (k, p) in ps.iter().enumerate() {
                    let key = (
                        (p.x / WELD_M).round() as i64,
                        (p.y / WELD_M).round() as i64,
                        (p.z / WELD_M).round() as i64,
                    );
                    let n = *keys.entry(key).or_insert_with(|| {
                        parent.push(parent.len());
                        parent.len() - 1
                    });
                    nodes[k] = n;
                }
                for k in 1..3 {
                    let (a, b) = (find(&mut parent, nodes[0]), find(&mut parent, nodes[k]));
                    if a != b {
                        parent[a.max(b)] = a.min(b);
                    }
                }
                tri_node.push(((si, ti), nodes[0], area));
            }
        }
        let mut pieces: BTreeMap<usize, (Vec<(usize, usize)>, f64)> = BTreeMap::new();
        for (tri, n, area) in tri_node {
            let root = find(&mut parent, n);
            let e = pieces.entry(root).or_default();
            e.0.push(tri);
            e.1 += area;
        }
        // (triangles, area, centre, normal) per connected glass piece.
        type Piece = (Vec<(usize, usize)>, f64, DVec3, DVec3);
        let mut big: Vec<Piece> = Vec::new();
        for (tris, area) in pieces.into_values() {
            if area < PANE_MIN_AREA_M2 {
                continue;
            }
            let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
            for (si, ti) in &tris {
                let s = &src.submeshes[*si];
                for k in 0..3 {
                    let p = to_engine(f, v3(s.vertices[s.indices[ti * 3 + k] as usize].position))
                        - centre;
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            big.push((tris, area, lo, hi));
        }
        // The largest first, then a fixed spatial order among the kept.
        big.sort_by(|a, b| b.1.total_cmp(&a.1));
        big.truncate(MAX_PANES);
        big.sort_by(|a, b| {
            let (ca, cb) = (0.5 * (a.2 + a.3), 0.5 * (b.2 + b.3));
            cb.z.total_cmp(&ca.z).then(ca.x.total_cmp(&cb.x))
        });
        for (k, (tris, _, lo, hi)) in big.into_iter().enumerate() {
            for t in &tris {
                pane_of.insert(*t, k);
            }
            let c = 0.5 * (lo + hi);
            let h = (0.5 * (hi - lo)).max(DVec3::splat(0.005));
            let m = rebuild(src, &|si, ti| pane_of.get(&(si, ti)) == Some(&k), &|v| {
                unit_box(vert_engine(f, v, centre), c, h)
            });
            let n = tris.len();
            panes.push(ArtPartOut {
                name: format!("glass_{k}"),
                centre: c,
                half: h,
                mesh: Some(m),
                triangles: n,
            });
        }
    }
    // ── the body: every live chassis triangle not in a pane ──
    let body = rebuild(
        src,
        &|si, ti| in_group(si, ti, Group::Chassis) && !pane_of.contains_key(&(si, ti)),
        &|v| vert_engine(f, v, centre),
    );
    // ── the doors ──
    let mut parts: Vec<ArtPartOut> = Vec::new();
    let mut door_joints: Vec<usize> = groups
        .iter()
        .flatten()
        .filter_map(|g| match g {
            Group::Door(j) => Some(*j),
            _ => None,
        })
        .collect();
    door_joints.sort();
    door_joints.dedup();
    for j in door_joints {
        let pivot = joint_at(j);
        let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
        let mut n = 0usize;
        for (si, s) in src.submeshes.iter().enumerate() {
            for (ti, t) in s.indices.chunks_exact(3).enumerate() {
                if !in_group(si, ti, Group::Door(j)) {
                    continue;
                }
                n += 1;
                for k in t {
                    let p = to_engine(f, v3(s.vertices[*k as usize].position)) - centre;
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        if n == 0 {
            continue;
        }
        // The box: lateral centre ON the pivot, front edge ON the pivot, rear
        // edge at the door's own rear, height its own -- and THIN, a door
        // skin's [`DOOR_HALF_THICK_M`]: the box is the door's collider once it
        // swings, and the pack's door (skin, trim and armrest) is a quarter of a
        // metre deep. The mesh keeps every vertex it has; unit-box coordinates
        // past +-0.5 restore to metres under the part's scale all the same.
        let hx = DOOR_HALF_THICK_M;
        let rear = lo.z.min(pivot.z - 0.05);
        let c = DVec3::new(pivot.x, 0.5 * (lo.y + hi.y), 0.5 * (rear + pivot.z));
        let h = DVec3::new(hx, 0.5 * (hi.y - lo.y), 0.5 * (pivot.z - rear));
        let m = rebuild(src, &|si, ti| in_group(si, ti, Group::Door(j)), &|v| {
            unit_box(vert_engine(f, v, centre), c, h)
        });
        parts.push(ArtPartOut {
            name: if pivot.x > 0.0 { "door_l" } else { "door_r" }.to_string(),
            centre: c,
            half: h,
            mesh: Some(m),
            triangles: n,
        });
    }
    parts.extend(panes);
    // ── the steering wheel ──
    let mut pack_rake_deg = None;
    let steer_mesh: Option<MeshAsset> = match &i.steering {
        Some(sm) => Some(rebuild(&sm.mesh, &|_, _| true, &|v| {
            let p = sm.rotation * v3(v.position) + sm.translation;
            let n = sm.rotation * v3(v.normal);
            let t = sm.rotation
                * DVec3::new(
                    v.tangent[0] as f64,
                    v.tangent[1] as f64,
                    v.tangent[2] as f64,
                );
            vert_engine(
                f,
                &MeshVertex {
                    position: f3(p),
                    normal: f3(n),
                    uv: v.uv,
                    tangent: [t.x as f32, t.y as f32, t.z as f32, v.tangent[3]],
                },
                centre,
            )
        })),
        None => {
            let any = groups.iter().flatten().any(|g| *g == Group::Steering);
            any.then(|| {
                rebuild(src, &|si, ti| in_group(si, ti, Group::Steering), &|v| {
                    vert_engine(f, v, centre)
                })
            })
        }
    };
    let mut hub_at: Option<DVec3> = None;
    if let Some(m) = steer_mesh {
        let pts: Vec<DVec3> = m
            .submeshes
            .iter()
            .flat_map(|s| s.vertices.iter().map(|v| v3(v.position)))
            .collect();
        if pts.len() >= 3 {
            let mean = pts.iter().fold(DVec3::ZERO, |a, p| a + *p) / pts.len() as f64;
            let mut c = [[0.0f64; 3]; 3];
            for p in &pts {
                let d = *p - mean;
                let d = [d.x, d.y, d.z];
                for (r, row) in c.iter_mut().enumerate() {
                    for (k, cell) in row.iter_mut().enumerate() {
                        *cell += d[r] * d[k];
                    }
                }
            }
            // The column, pointing at the driver (behind the hub: -Z).
            let mut z = min_axis(c);
            if z.z > 0.0 {
                z = -z;
            }
            // Local X: the chassis -X in the rim's plane (hub_rim_euler's rest
            // maps local X to chassis -X).
            let x = (DVec3::NEG_X - z * DVec3::NEG_X.dot(z)).normalize_or_zero();
            let y = z.cross(x);
            // Rake: the column's tilt from horizontal is the rim's from vertical.
            // asin(x) = pi/2 - acos(x), through the portable acos: this number
            // is COMMITTED content (the art table's `hub_rake_deg`).
            pack_rake_deg = Some(
                (std::f64::consts::FRAC_PI_2 - inf_math::pacos64((-z.y).clamp(-1.0, 1.0)))
                    .to_degrees(),
            );
            let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
            for p in &pts {
                let d = *p - mean;
                let l = DVec3::new(d.dot(x), d.dot(y), d.dot(z));
                lo = lo.min(l);
                hi = hi.max(l);
            }
            // The hub: the rim's centre in its plane, and ON the rim's plane
            // along the column -- the mean column height of the outermost
            // vertices, because a pack's wheel mesh carries a column stub whose
            // length would otherwise drag the "centre" a hand's width off the
            // rim the grips are laid on.
            let mid = 0.5 * (lo + hi);
            let rim_r = pts
                .iter()
                .map(|p| {
                    let d = *p - mean;
                    let (a, b) = (d.dot(x) - mid.x, d.dot(y) - mid.y);
                    (a * a + b * b).sqrt()
                })
                .fold(0.0f64, f64::max);
            let (mut zs, mut zn) = (0.0f64, 0usize);
            for p in &pts {
                let d = *p - mean;
                let (a, b) = (d.dot(x) - mid.x, d.dot(y) - mid.y);
                if (a * a + b * b).sqrt() > 0.85 * rim_r {
                    zs += d.dot(z);
                    zn += 1;
                }
            }
            let rim_z = if zn > 0 { zs / zn as f64 } else { mid.z };
            let hub = mean + x * mid.x + y * mid.y + z * rim_z;
            let mut r = 0.0f64;
            let mut t = 0.0f64;
            for p in &pts {
                let d = *p - hub;
                let l = DVec3::new(d.dot(x), d.dot(y), d.dot(z));
                r = r.max((l.x * l.x + l.y * l.y).sqrt());
                t = t.max(l.z.abs());
            }
            let h = DVec3::new(r.max(0.01), r.max(0.01), t.max(0.005));
            let n = m.triangle_count();
            let col = rebuild(&m, &|_, _| true, &|v| {
                let d = v3(v.position) - hub;
                let nn = v3(v.normal);
                let tt = DVec3::new(
                    v.tangent[0] as f64,
                    v.tangent[1] as f64,
                    v.tangent[2] as f64,
                );
                MeshVertex {
                    position: f3(DVec3::new(
                        d.dot(x) / (2.0 * h.x),
                        d.dot(y) / (2.0 * h.y),
                        d.dot(z) / (2.0 * h.z),
                    )),
                    normal: f3(DVec3::new(nn.dot(x), nn.dot(y), nn.dot(z))),
                    uv: v.uv,
                    tangent: [
                        tt.dot(x) as f32,
                        tt.dot(y) as f32,
                        tt.dot(z) as f32,
                        v.tangent[3],
                    ],
                }
            });
            hub_at = Some(hub);
            parts.push(ArtPartOut {
                name: "hub".to_string(),
                centre: hub,
                half: h,
                mesh: Some(col),
                triangles: n,
            });
        }
    }
    // ── the seats: the Blueprint's driver, or behind the drawn hub, or (a pack
    //    with neither) where every family seats its driver -- the hull
    //    fractions `inf_ecs::boarding` has always used ──
    //
    // A seat from the hull fractions is DRAWN as nothing: it would restate
    // the rule `sockets_of` applies to a hull with no cushion, and a drawn
    // cushion switches off that rule's head-room clamp (the imported sports
    // car, 0.491 m high, needs it). Its plan still places the door proxies.
    let measured_seat = i
        .seat
        .map(|p| to_engine(f, p) - centre)
        .or_else(|| hub_at.map(|h| DVec3::new(h.x, 0.0, h.z - SEAT_BEHIND_HUB_M)));
    let seat_plan = measured_seat.or_else(|| {
        Some(DVec3::new(
            inf_ecs::boarding::SEAT_LATERAL_FRAC_X * half.x,
            0.0,
            inf_ecs::boarding::SEAT_FRONT_FRAC_Z * half.z,
        ))
    });
    if let Some(s) = measured_seat {
        // The cushion's TOP on the H-point `sockets_of` seats the driver at
        // (`veh3f_gate::the_drawn_seat_is_the_seat_the_body_sits_on`), so the
        // part's centre is half a cushion below it.
        let y = inf_ecs::boarding::SEAT_CUSHION_FRAC_Y * half.y - SEAT_HALF_M[1];
        let h = DVec3::new(SEAT_HALF_M[0], SEAT_HALF_M[1], SEAT_HALF_M[2]);
        let driver_x = if s.x.abs() < 0.05 { 0.0 } else { s.x };
        parts.push(ArtPartOut {
            name: "seat_r".to_string(),
            centre: DVec3::new(driver_x.abs(), y, s.z),
            half: h,
            mesh: None,
            triangles: 0,
        });
        if driver_x.abs() > 0.05 {
            parts.push(ArtPartOut {
                name: "seat_l".to_string(),
                centre: DVec3::new(-driver_x.abs(), y, s.z),
                half: h,
                mesh: None,
                triangles: 0,
            });
        }
    }
    // ── the door PROXIES, for a pack with no door meshes ──
    //
    // Beside the driver's seat on each flank: the front edge 0.55 m ahead of
    // the cushion (the A-pillar of every car the packs ship), a metre long, from
    // the sill to the waist, just inside the skin. Nothing is drawn on them in a
    // project that has the art (their mesh is a hidden sliver): they are what the
    // boarding opens, and the report says so.
    let has_door = parts.iter().any(|p| p.name.starts_with("door"));
    if i.door_proxy && !has_door {
        if let Some(s) = seat_plan {
            for side in [1.0f64, -1.0] {
                let front = s.z + 0.55;
                let rear = front - 1.0;
                let c = DVec3::new(side * (half.x - 0.04), -0.15 * half.y, 0.5 * (front + rear));
                let h = DVec3::new(0.04, 0.45 * half.y, 0.5 * (front - rear));
                parts.push(ArtPartOut {
                    name: if side > 0.0 { "door_l" } else { "door_r" }.to_string(),
                    centre: c,
                    half: h,
                    mesh: None,
                    triangles: 0,
                });
            }
        }
    }
    Ok(SkelVehicleSplit {
        half_extents: half,
        body,
        wheels,
        hubs,
        wheel_bones,
        wheel_radius,
        parts,
        ground_below: ground - centre.y,
        camera: i.camera.map(|c| to_engine(f, c) - centre),
        pack_rake_deg,
        triangles,
        hidden_triangles: hidden,
    })
}

/// A collider-frame vertex into a part's unit box `(c, h)`.
fn unit_box(v: MeshVertex, c: DVec3, h: DVec3) -> MeshVertex {
    let p = v3(v.position);
    MeshVertex {
        position: f3(DVec3::new(
            (p.x - c.x) / (2.0 * h.x),
            (p.y - c.y) / (2.0 * h.y),
            (p.z - c.z) / (2.0 * h.z),
        )),
        ..v
    }
}

/// **The body TOML** -- the measured numbers in the committed art table's own
/// spelling, so a transcription is a paste; plus the roster geometry, the pack
/// numbers the engine does NOT take (with why), and the licence.
#[allow(clippy::too_many_arguments)]
pub fn skel_body_toml(
    split: &SkelVehicleSplit,
    art: &str,
    source: &str,
    pack: &str,
    facing: &str,
    licence: &str,
    fab_url: &str,
    pack_lods: &[(u32, u32)],
    roles: &[(String, SlotRole)],
    bp_wheel_radius_m: Option<f64>,
    door_proxy: bool,
) -> String {
    let h = split.half_extents;
    let mut s = String::new();
    s.push_str(&format!(
        "# {art} -- split by the VEH3f.2a skeletal bridge from {source} ({pack}).\n"
    ));
    s.push_str(&format!("# licence: {licence}\n# listing: {fab_url}\n"));
    s.push_str(&format!(
        "# {} triangles in ({} hidden), {} wheels, {} parts; pack LODs {:?} (the engine draws a meshlet DAG per section)\n",
        split.triangles,
        split.hidden_triangles,
        split.wheels.len(),
        split.parts.len(),
        pack_lods
    ));
    if let Some(r) = bp_wheel_radius_m {
        s.push_str(&format!(
            "# wheel radius: geometry {:.4} m, the pack's Chaos wheel {:.4} m\n",
            split.wheel_radius, r
        ));
    }
    if let Some(r) = split.pack_rake_deg {
        s.push_str(&format!(
            "# steering column: the pack's rake {r:.2} deg (the families' constant is {:.1} deg)\n",
            inf_ecs::boarding::WHEEL_RAKE_DEG
        ));
    }
    s.push_str("# slots:");
    for (name, role) in roles {
        s.push_str(&format!(" {name}={role:?}"));
    }
    s.push_str("\n\n[geometry]\n");
    for (k, v) in split.geometry() {
        s.push_str(&format!("{k} = {v:.4}\n"));
    }
    s.push_str("\n# ── the art table row (crates/inf-ecs/src/vehicle_art.toml) ──\n[[art]]\n");
    s.push_str(&format!(
        "key = \"{art}\"\nsource = \"{source}\"\npack = \"{pack}\"\nfacing = \"{facing}\"\n"
    ));
    s.push_str(&format!(
        "half_extents_m = [{:.4}, {:.4}, {:.4}]\n",
        h.x, h.y, h.z
    ));
    s.push_str(&format!("pack_lods = {}\n", pack_lods.len().max(1)));
    if door_proxy {
        s.push_str("door_proxy = true\n");
    }
    if let Some(c) = split.camera {
        s.push_str(&format!(
            "camera_m = [{:.4}, {:.4}, {:.4}]\n",
            c.x, c.y, c.z
        ));
    }
    if let Some(r) = split.pack_rake_deg {
        s.push_str(&format!("hub_rake_deg = {r:.3}\n"));
    }
    for p in &split.parts {
        s.push_str(&format!(
            "[[art.parts]]\nname = \"{}\"\ncentre = [{:.4}, {:.4}, {:.4}]\nhalf = [{:.4}, {:.4}, {:.4}]\n",
            p.name,
            p.centre.x / h.x,
            p.centre.y / h.y,
            p.centre.z / h.z,
            p.half.x / h.x,
            p.half.y / h.y,
            p.half.z / h.z,
        ));
    }
    s
}

/// **A synthetic skinned car** (wave VEH3f.2a) -- ours, generated, in the
/// exporter's glTF frame (facing `+X`, up `+Y`, the car's right on `+Z`) --
/// for the gate and the unit tests: a body box on the root, a window pane on
/// the body in the GLASS slot, four wheel "tyres" (octagonal prisms, axle on
/// `Z`) each weighted to its wheel bone, two doors weighted to their hinge
/// bones, and a raked steering disc weighted to `SteeringWheel`. Slot 0 is
/// paint, slot 1 glass, slot 2 plain.
pub mod fixture {
    use glam::DVec3;
    use inf_mesh::{MeshAsset, MeshVertex, SubMesh, VertexSkin};

    use super::SlotRole;

    /// The joints: `(name, world rest position)`, glTF frame.
    pub fn joints() -> Vec<(String, DVec3)> {
        vec![
            ("Root".into(), DVec3::ZERO),
            ("Front_Left_Wheel".into(), DVec3::new(1.40, 0.34, -0.75)),
            ("Front_Right_Wheel".into(), DVec3::new(1.40, 0.34, 0.75)),
            ("Rear_Left_Wheel".into(), DVec3::new(-1.45, 0.34, -0.75)),
            ("Rear_Right_Wheel".into(), DVec3::new(-1.45, 0.34, 0.75)),
            ("Left_Door".into(), DVec3::new(0.85, 0.80, -0.86)),
            ("Right_Door".into(), DVec3::new(0.85, 0.80, 0.86)),
            ("SteeringWheel".into(), DVec3::new(0.60, 0.95, -0.40)),
        ]
    }

    /// The roles of the three slots.
    pub fn roles() -> Vec<SlotRole> {
        vec![SlotRole::Paint, SlotRole::Glass, SlotRole::Plain]
    }

    /// The tyre radius the fixture's wheels are built at, metres.
    pub const TYRE_R: f64 = 0.34;

    /// The steering disc's rake, degrees, in `hub_rim_euler`'s convention.
    pub const RAKE_DEG: f64 = -24.0;

    fn quad_box(lo: DVec3, hi: DVec3, joint: u16, out: &mut Vec<(DVec3, u16)>) {
        let c = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
        let v = [
            c(lo.x, lo.y, lo.z),
            c(hi.x, lo.y, lo.z),
            c(hi.x, hi.y, lo.z),
            c(lo.x, hi.y, lo.z),
            c(lo.x, lo.y, hi.z),
            c(hi.x, lo.y, hi.z),
            c(hi.x, hi.y, hi.z),
            c(lo.x, hi.y, hi.z),
        ];
        let faces = [
            [0, 2, 1, 0, 3, 2],
            [4, 5, 6, 4, 6, 7],
            [0, 1, 5, 0, 5, 4],
            [3, 7, 6, 3, 6, 2],
            [0, 4, 7, 0, 7, 3],
            [1, 2, 6, 1, 6, 5],
        ];
        for f in faces {
            for i in f {
                out.push((v[i], joint));
            }
        }
    }

    /// An octagonal prism about an axis through `c` along `axis` (unit), of
    /// radius `r` and half-width `w`: a tyre, or (flat) a steering disc.
    fn prism(c: DVec3, axis: DVec3, r: f64, w: f64, joint: u16, out: &mut Vec<(DVec3, u16)>) {
        // Two in-plane axes, exact for the axes this fixture uses.
        let a = if axis.y.abs() < 0.9 {
            DVec3::Y
        } else {
            DVec3::X
        };
        let u = axis.cross(a).normalize();
        let v = axis.cross(u).normalize();
        // An octagon at the eight exact directions (no trig: the committed
        // fixture's bytes must be the same on every platform).
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let dirs = [
            (1.0, 0.0),
            (s, s),
            (0.0, 1.0),
            (-s, s),
            (-1.0, 0.0),
            (-s, -s),
            (0.0, -1.0),
            (s, -s),
        ];
        let ring = |side: f64| -> Vec<DVec3> {
            dirs.iter()
                .map(|(x, y)| c + u * (x * r) + v * (y * r) + axis * (side * w))
                .collect()
        };
        let (p, q) = (ring(-1.0), ring(1.0));
        for k in 0..8 {
            let n = (k + 1) % 8;
            for t in [[p[k], p[n], q[n]], [p[k], q[n], q[k]]] {
                for x in t {
                    out.push((x, joint));
                }
            }
            for t in [[c - axis * w, p[n], p[k]], [c + axis * w, q[k], q[n]]] {
                for x in t {
                    out.push((x, joint));
                }
            }
        }
    }

    fn submesh(tris: Vec<(DVec3, u16)>, slot: u32) -> SubMesh {
        let vertices: Vec<MeshVertex> = tris
            .iter()
            .map(|(p, _)| MeshVertex {
                position: [p.x as f32, p.y as f32, p.z as f32],
                ..Default::default()
            })
            .collect();
        let skin: Vec<VertexSkin> = tris
            .iter()
            .map(|(_, j)| VertexSkin {
                joints: [*j, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            })
            .collect();
        SubMesh {
            name: format!("slot{slot}"),
            indices: (0..vertices.len() as u32).collect(),
            vertices,
            material_slot: Some(slot),
            skin,
        }
    }

    /// The car.
    pub fn car() -> MeshAsset {
        let j = joints();
        let mut paint = Vec::new();
        let mut glass = Vec::new();
        let mut plain = Vec::new();
        // The body, and a windscreen-sized pane on it (0.9 m^2, a PANE).
        quad_box(
            DVec3::new(-2.3, 0.25, -0.85),
            DVec3::new(2.3, 1.35, 0.85),
            0,
            &mut paint,
        );
        quad_box(
            DVec3::new(0.55, 1.10, -0.65),
            DVec3::new(0.60, 1.45, 0.65),
            0,
            &mut glass,
        );
        // The wheels, axle on glTF Z (the car's lateral), on their bones.
        for k in 1..=4u16 {
            prism(j[k as usize].1, DVec3::Z, TYRE_R, 0.11, k, &mut plain);
        }
        // The doors: from the hinge bone (their FRONT edge) back 1.0 m, 5 cm
        // thick, just outside the body.
        for (k, side) in [(5u16, -1.0f64), (6u16, 1.0f64)] {
            let p = j[k as usize].1;
            quad_box(
                DVec3::new(p.x - 1.0, 0.35, p.z - side * 0.05),
                DVec3::new(p.x, 1.10, p.z),
                k,
                &mut paint,
            );
        }
        // The steering disc: rim plane raked so its face turns up at the
        // driver (behind it, glTF -X), by RAKE_DEG.
        let rake = RAKE_DEG.to_radians();
        let axis = DVec3::new(-inf_math::pcos64(rake), -inf_math::psin64(rake), 0.0).normalize();
        prism(j[7].1, axis, 0.18, 0.02, 7, &mut plain);
        MeshAsset::new(
            vec![submesh(paint, 0), submesh(glass, 1), submesh(plain, 2)],
            vec!["paint".into(), "glass".into(), "plain".into()],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split() -> SkelVehicleSplit {
        let mesh = fixture::car();
        let joints = fixture::joints();
        let roles = fixture::roles();
        split_skel_vehicle(&SkelVehicleIn {
            art: "fixture",
            facing: Facing::PlusX,
            mesh: &mesh,
            joints: &joints,
            roles: &roles,
            steering: None,
            seat: None,
            camera: None,
            door_proxy: false,
        })
        .expect("the fixture splits")
    }

    /// **Every part lands on its bone** (wave VEH3f.2a): the wheels' hubs are
    /// their bones' rest positions and their meshes are centred on them; the
    /// front wheels (glTF +X) come out at the engine's front (+Z); each door's
    /// box has its front edge and lateral centre on its hinge bone; the pane is
    /// out of the body; the steering wheel carries the fixture's rake.
    ///
    /// **Mutations**: `to_engine` the identity for +X (the car imported facing
    /// sideways) -> red on the front wheels; the door box's front edge at the
    /// door's own bounds -> red on the pivot.
    #[test]
    fn the_split_puts_every_part_on_its_bone() {
        let s = split();
        let joints = fixture::joints();
        // The collider centre, recovered from one hub. Wheel 0 is the -X front:
        // the car's RIGHT front (glTF +Z is the car's right, and the engine's
        // right is -X), so its bone is `Front_Right_Wheel`.
        let centre = to_engine(Facing::PlusX, joints[2].1) - s.hubs[0];
        let want = |j: usize| to_engine(Facing::PlusX, joints[j].1) - centre;
        assert_eq!(s.wheels.len(), 4);
        for (k, bone) in [(0usize, 2usize), (1, 1), (2, 4), (3, 3)] {
            let d = (s.hubs[k] - want(bone)).length();
            assert!(d < 1e-3, "wheel {k} is {d} m off its bone");
            let b = s.wheels[k].bounds;
            let mid = DVec3::new(
                0.5 * (b.min[0] + b.max[0]) as f64,
                0.5 * (b.min[1] + b.max[1]) as f64,
                0.5 * (b.min[2] + b.max[2]) as f64,
            );
            assert!(mid.length() < 1e-3, "wheel {k}'s mesh centre is {mid:?}");
        }
        assert!(
            s.hubs[0].z > 0.0 && s.hubs[1].z > 0.0,
            "the front wheels are not at +Z"
        );
        assert!(
            s.hubs[0].x < 0.0 && s.hubs[1].x > 0.0,
            "wheel 0 is not the -X front"
        );
        assert!((s.wheel_radius - fixture::TYRE_R).abs() < 1e-3);
        for (name, bone) in [("door_l", 5usize), ("door_r", 6usize)] {
            let d = s
                .parts
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("no {name}"));
            let pivot = want(bone);
            assert!((d.centre.x - pivot.x).abs() < 1e-3, "{name} centre x");
            assert!(
                (d.centre.z + d.half.z - pivot.z).abs() < 1e-3,
                "{name}'s front edge is {} m off its hinge bone",
                d.centre.z + d.half.z - pivot.z
            );
        }
        assert!(s.parts.iter().any(|p| p.name.starts_with("glass")));
        let rake = s.pack_rake_deg.expect("a hub");
        assert!(
            (rake - fixture::RAKE_DEG).abs() < 0.5,
            "the hub's rake is {rake} deg"
        );
    }

    /// **Two splits of one input are the same bytes** (wave VEH3f.2a).
    #[test]
    fn a_split_is_deterministic() {
        let (a, b) = (split(), split());
        let enc = |m: &MeshAsset| inf_asset::encode(m).expect("encodes");
        assert_eq!(enc(&a.body), enc(&b.body));
        for (x, y) in a.wheels.iter().zip(&b.wheels) {
            assert_eq!(enc(x), enc(y));
        }
        for (x, y) in a.parts.iter().zip(&b.parts) {
            assert_eq!(x.name, y.name);
            if let (Some(p), Some(q)) = (&x.mesh, &y.mesh) {
                assert_eq!(enc(p), enc(q));
            }
        }
    }
}
