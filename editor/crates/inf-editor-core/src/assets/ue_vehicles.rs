//! **THE VEHICLE SPLIT** (wave VEH3f) -- the UE bridge's answer to a fused
//! vehicle: one static mesh in, a chassis mesh, four wheel meshes and a
//! per-body TOML out.
//!
//! # What the construction pack gives, measured
//!
//! Every machine in `ConstructionVehiclesPack1` is ONE static mesh with its
//! wheels, tracks, glass and cab fused in (`veh3f-vehicle-asset-scout.md`, and
//! the export's own census: 7-12 material slots each, no sockets, no parts).
//! The only thing that tells a wheel from a bonnet is its MATERIAL SLOT
//! (`MI_TruckWheel`, `MI_WheelLoader_Tire`, `MI_Forklift_Wheels`,
//! `MI_MobileCrane_Wheels`, `*_Tracks`), and the only thing that tells one
//! wheel from the next is that they do not touch.
//!
//! # The split
//!
//! 1. The glTF's frame IS this engine's. Measured on the export: Unreal's
//!    `(X, Y, Z)` crosses the glTF exporter as `(x, z, y)` (the exporter's own
//!    handedness flip, winding kept), and every machine in this pack faces
//!    Unreal `+Y` -- its long axis runs `-389..535` cm on the tractor, the
//!    excavator's boom and the forklift's forks reach `+Y` -- so it arrives
//!    facing glTF `+Z`, which is this engine's forward. No transform, stated
//!    rather than assumed.
//! 2. Every triangle of a wheel-slot submesh is sorted into CONNECTED PIECES
//!    by a union-find over its vertices welded at a millimetre (a UV seam
//!    splits a vertex; it does not split a wheel).
//! 3. The four CORNER pieces -- front and rear, near and off side -- become
//!    the rig's four wheels, each hub-centred with its axle on `X`, so the
//!    wheel entity's own spin and steer turn it. Every other piece (a dump
//!    truck's twin rears, a third axle, a track) stays in the chassis mesh and
//!    does not spin, which is the four-wheel rig stated rather than hidden.
//! 4. The chassis mesh is re-centred on the collider the geometry implies:
//!    `x`/`z` on the machine's own box, `y` from its underside to its roof.
//! 5. The measured geometry -- half-extents, wheel radius, track, wheelbase,
//!    wheel drop and offset -- goes into `<key>.vehicle.toml` beside the
//!    meshes, in catalogue keys, so a roster row can be checked against the
//!    machine it draws.
//!
//! The meshes are written at the GUIDs the committed rows name
//! ([`inf_ecs::roster::art_body_guid`] / [`art_wheel_guid`](inf_ecs::roster::art_wheel_guid)),
//! into the LOCAL project only -- the committed fallback at those GUIDs is what
//! a checkout without the art draws.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_ecs::roster::ArtKey;
use inf_mesh::{Aabb, MeshAsset, MeshVertex, SubMesh};

/// A material slot is a wheel's (or a track's) when its name holds one of these.
pub const WHEEL_SLOT_WORDS: &[&str] = &["wheel", "tire", "tyre", "track"];

/// The smallest piece that is a wheel, metres of radius -- below it a piece is
/// a hub cap or a bolt ring that happened to carry a tyre material.
pub const MIN_WHEEL_RADIUS_M: f64 = 0.15;

/// How close two vertices are to be one, metres.
const WELD_M: f64 = 1e-3;

/// **One connected piece of wheel geometry**, engine frame.
#[derive(Clone, Debug)]
pub struct WheelPiece {
    /// Its bounds' centre.
    pub centre: DVec3,
    /// Half its height/length, whichever is larger -- a tyre's radius.
    pub radius: f64,
    /// Its lateral extent.
    pub width: f64,
    /// `(submesh, triangle)` pairs.
    pub tris: Vec<(usize, usize)>,
}

/// **What one machine split into.**
#[derive(Clone, Debug)]
pub struct VehicleSplit {
    /// Which machine.
    pub key: ArtKey,
    /// The chassis mesh, centred on the collider.
    pub body: MeshAsset,
    /// The four rig wheels (FL, FR, RL, RR), hub-centred. Empty for a tracked
    /// machine, whose tracks stay in the body.
    pub wheels: Vec<MeshAsset>,
    /// The collider the geometry implies: half-extents.
    pub half_extents: DVec3,
    /// Where each rig wheel's hub is, relative to the collider centre
    /// (FL, FR, RL, RR).
    pub hubs: Vec<DVec3>,
    /// The mean corner radius.
    pub wheel_radius: f64,
    /// How many wheel pieces were found and how many stayed in the body.
    pub pieces: usize,
    /// Triangles in the source.
    pub triangles: usize,
    /// Where the art's ground is, relative to the collider centre (negative):
    /// its lowest vertex. A row that draws it rests there when its settled
    /// contact is `ground_m` below its chassis origin.
    pub ground_below: f64,
}

impl VehicleSplit {
    /// **The catalogue keys this machine measures at**, as a TOML table -- the
    /// geometry a roster row that draws it must author. `wheel_drop_m` and
    /// `wheel_offset_z_m` are the hubs' mean height and fore-aft centre.
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
                (h.iter().map(|p| p.x.abs()).sum::<f64>()) / 4.0,
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

/// The glTF frame into the engine's -- the identity for this pack (module note).
fn to_engine(p: [f32; 3]) -> DVec3 {
    DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)
}

fn is_wheel_slot(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    WHEEL_SLOT_WORDS.iter().any(|w| n.contains(w))
}

/// Union-find over welded vertex keys.
fn find(parent: &mut Vec<usize>, mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// **Split one machine.** `asset` is the imported LOD-0 `.inf_mesh` (glTF
/// frame, metres). `manifest_slots` is the exporter's `material_slots` for the
/// mesh, in Unreal's slot order: the glTF the exporter writes carries its
/// materials UNNAMED (measured on every machine in the pack), so the manifest
/// is the only place a slot's `MI_TruckWheel` is spelled. The asset's own slot
/// name is the fallback for a glTF that does name them.
pub fn split_vehicle(
    asset: &MeshAsset,
    key: ArtKey,
    manifest_slots: &[Option<String>],
) -> Result<VehicleSplit, String> {
    let slot_name = |s: &SubMesh| -> String {
        s.material_slot
            .and_then(|i| {
                manifest_slots
                    .get(i as usize)
                    .cloned()
                    .flatten()
                    .or_else(|| asset.material_slots.get(i as usize).cloned())
            })
            .unwrap_or_default()
    };
    // ── every vertex into the engine frame, and the whole machine's box ──
    let mut lo = DVec3::splat(f64::MAX);
    let mut hi = DVec3::splat(f64::MIN);
    let mut body_lo_y = f64::MAX;
    for s in &asset.submeshes {
        let wheel = is_wheel_slot(&slot_name(s));
        for v in &s.vertices {
            let p = to_engine(v.position);
            lo = lo.min(p);
            hi = hi.max(p);
            // A tracked machine's tracks ARE its underside: its box spans them.
            if !wheel || key.tracked() {
                body_lo_y = body_lo_y.min(p.y);
            }
        }
    }
    if !(lo.x < hi.x) {
        return Err(format!("{}: the mesh has no geometry", key.name()));
    }
    // ── the wheel pieces ──
    let mut keys: BTreeMap<(i64, i64, i64), usize> = BTreeMap::new();
    let mut parent: Vec<usize> = Vec::new();
    let weld = |p: DVec3| {
        (
            (p.x / WELD_M).round() as i64,
            (p.y / WELD_M).round() as i64,
            (p.z / WELD_M).round() as i64,
        )
    };
    let mut tri_node: Vec<((usize, usize), usize)> = Vec::new();
    for (si, s) in asset.submeshes.iter().enumerate() {
        if !is_wheel_slot(&slot_name(s)) {
            continue;
        }
        for (ti, t) in s.indices.chunks_exact(3).enumerate() {
            let mut nodes = [0usize; 3];
            for (k, idx) in t.iter().enumerate() {
                let p = to_engine(s.vertices[*idx as usize].position);
                let key = weld(p);
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
            tri_node.push(((si, ti), nodes[0]));
        }
    }
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (tri, n) in tri_node {
        let root = find(&mut parent, n);
        groups.entry(root).or_default().push(tri);
    }
    let mut pieces: Vec<WheelPiece> = Vec::new();
    for tris in groups.into_values() {
        let (mut a, mut b) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
        for (si, ti) in &tris {
            let s = &asset.submeshes[*si];
            for k in 0..3 {
                let p = to_engine(s.vertices[s.indices[ti * 3 + k] as usize].position);
                a = a.min(p);
                b = b.max(p);
            }
        }
        let e = b - a;
        pieces.push(WheelPiece {
            centre: 0.5 * (a + b),
            radius: 0.5 * e.y.max(e.z),
            width: e.x,
            tris,
        });
    }
    let n_pieces = pieces.len();
    // ── the four corners (never a tracked machine's) ──
    let ground = lo.y;
    let clearance = (body_lo_y - ground).max(0.15);
    let collider_lo = ground + clearance;
    let centre = DVec3::new(
        0.5 * (lo.x + hi.x),
        0.5 * (collider_lo + hi.y),
        0.5 * (lo.z + hi.z),
    );
    let half = DVec3::new(
        0.5 * (hi.x - lo.x),
        0.5 * (hi.y - collider_lo),
        0.5 * (hi.z - lo.z),
    );
    let mut corners: Vec<usize> = Vec::new();
    if !key.tracked() {
        let cand: Vec<usize> = (0..pieces.len())
            // A tyre stands on the ground: its lowest point is the machine's (a spare
            // on a mixer's chute and a bumper guard carry the tyre material too).
            .filter(|i| {
                pieces[*i].radius >= MIN_WHEEL_RADIUS_M
                    && pieces[*i].centre.y - pieces[*i].radius <= ground + 0.05
            })
            .collect();
        let pick = |left: bool, front: bool| -> Option<usize> {
            cand.iter()
                .copied()
                .filter(|i| (pieces[*i].centre.x < centre.x) == left)
                .max_by(|a, b| {
                    let (za, zb) = (pieces[*a].centre.z, pieces[*b].centre.z);
                    let o = za.total_cmp(&zb);
                    if front {
                        o
                    } else {
                        o.reverse()
                    }
                })
        };
        for (left, front) in [(true, true), (false, true), (true, false), (false, false)] {
            // The extreme piece on that corner names the AXLE; the TYRE is the
            // largest piece on the same side within a radius of it (the
            // extreme piece is as often a hub cap as a tyre -- measured: 64-282
            // wheel pieces a machine, a tyre, a rim, and the bolts).
            if let Some(e) = pick(left, front) {
                let near = |i: &usize| {
                    let d = pieces[*i].centre - pieces[e].centre;
                    (pieces[*i].centre.x < centre.x) == left
                        && (d.y * d.y + d.z * d.z).sqrt() < pieces[e].radius.max(0.3)
                        && d.x.abs() < 0.3
                };
                let tyre = cand
                    .iter()
                    .copied()
                    .filter(near)
                    // The largest to 2 cm, then the OUTERMOST: a twin rear's two
                    // tyres measure 0.576 and 0.579 m, and the rig wheel is the outer.
                    .max_by(|a, b| {
                        let r = |i: &usize| (pieces[*i].radius * 50.0).round() as i64;
                        r(a).cmp(&r(b)).then(
                            (pieces[*a].centre.x - centre.x)
                                .abs()
                                .total_cmp(&(pieces[*b].centre.x - centre.x).abs()),
                        )
                    })
                    .unwrap_or(e);
                if !corners.contains(&tyre) {
                    corners.push(tyre);
                }
            }
        }
        if corners.len() != 4 {
            return Err(format!(
                "{}: {} wheel pieces and {} corners -- not a four-cornered machine",
                key.name(),
                n_pieces,
                corners.len()
            ));
        }
    }
    // A rig WHEEL is its tyre and every piece inside it -- the rim, the hub
    // cap, the bolts: a piece whose centre lies inside the tyre's radius, on
    // its axle, within its width. A twin rear's inner tyre is a tyre-width
    // inboard and stays in the body with the third axle.
    let mut in_corner: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (c, t) in corners.iter().enumerate() {
        let tyre = &pieces[*t];
        for p in &pieces {
            let d = p.centre - tyre.centre;
            let inside = (d.y * d.y + d.z * d.z).sqrt() + p.radius <= tyre.radius * 1.02
                && d.x.abs() <= 0.5 * tyre.width + 0.02;
            if std::ptr::eq(p, tyre) || inside {
                for tri in &p.tris {
                    in_corner.entry(*tri).or_insert(c);
                }
            }
        }
    }
    // ── write the body: every triangle not in a corner, re-centred ──
    let rebuild = |pick: &dyn Fn(usize, usize) -> bool, origin: DVec3| -> MeshAsset {
        let mut subs: Vec<SubMesh> = Vec::new();
        let mut bounds = Aabb::empty();
        for (si, s) in asset.submeshes.iter().enumerate() {
            let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
            let mut verts: Vec<MeshVertex> = Vec::new();
            let mut idx: Vec<u32> = Vec::new();
            for (ti, t) in s.indices.chunks_exact(3).enumerate() {
                if !pick(si, ti) {
                    continue;
                }
                for i in t {
                    let n = *remap.entry(*i).or_insert_with(|| {
                        let v = s.vertices[*i as usize];
                        let p = to_engine(v.position) - origin;
                        let nrm = to_engine(v.normal);
                        let tan = to_engine([v.tangent[0], v.tangent[1], v.tangent[2]]);
                        verts.push(MeshVertex {
                            position: [p.x as f32, p.y as f32, p.z as f32],
                            normal: [nrm.x as f32, nrm.y as f32, nrm.z as f32],
                            uv: v.uv,
                            tangent: [tan.x as f32, tan.y as f32, tan.z as f32, v.tangent[3]],
                        });
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
            ..asset.clone()
        }
    };
    let body = rebuild(&|si, ti| !in_corner.contains_key(&(si, ti)), centre);
    let mut wheels = Vec::new();
    let mut hubs = Vec::new();
    for (c, i) in corners.iter().enumerate() {
        let hub = pieces[*i].centre;
        wheels.push(rebuild(&|si, ti| in_corner.get(&(si, ti)) == Some(&c), hub));
        hubs.push(hub - centre);
    }
    let wheel_radius = if corners.is_empty() {
        0.0
    } else {
        corners.iter().map(|i| pieces[*i].radius).sum::<f64>() / corners.len() as f64
    };
    Ok(VehicleSplit {
        key,
        body,
        wheels,
        half_extents: half,
        hubs,
        wheel_radius,
        pieces: n_pieces,
        triangles: asset.submeshes.iter().map(|s| s.indices.len() / 3).sum(),
        ground_below: ground - centre.y,
    })
}

/// **A wheel that is not there** -- a millimetre triangle at the hub (wave
/// VEH3f). A TRACKED machine's rig still has four wheel sensors, and its rows
/// still name four wheel GUIDs; with the tracks left in the body, what those
/// GUIDs should draw in the local project is nothing, and one degenerate-free
/// sliver inside the track frame is the cheapest honest nothing a mesh can be
/// (the committed fallback's tyres would otherwise show through the art).
pub fn hidden_wheel(template: &MeshAsset) -> MeshAsset {
    let v = |x: f32, y: f32| MeshVertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    let vertices = vec![v(0.0, 0.0), v(1e-3, 0.0), v(0.0, 1e-3)];
    MeshAsset {
        submeshes: vec![SubMesh {
            name: "hidden".to_string(),
            vertices,
            indices: vec![0, 1, 2],
            material_slot: None,
            skin: Vec::new(),
        }],
        bounds: Aabb::from_points([[0.0, 0.0, 0.0], [1e-3, 1e-3, 0.0]]),
        material_slots: Vec::new(),
        material_slot_assets: Vec::new(),
        ..template.clone()
    }
}

/// **The per-body TOML** -- the machine's measured geometry in catalogue keys,
/// the parts and sockets this art does NOT carry (stated, not implied), and
/// the licence.
pub fn body_toml(split: &VehicleSplit, source: &str, licence: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# {} -- split by the VEH3f UE bridge from {source}.\n",
        split.key.name()
    ));
    s.push_str(&format!("# licence: {licence}\n"));
    s.push_str(&format!(
        "# {} triangles in, {} wheel pieces found, {} became rig wheels.\n",
        split.triangles,
        split.pieces,
        split.wheels.len()
    ));
    s.push_str("[geometry]\n");
    for (k, v) in split.geometry() {
        s.push_str(&format!("{k} = {v:.4}\n"));
    }
    s.push_str("\n[parts]\n# The art's doors, bonnet and glass are FUSED into one mesh: no VEH3c part\n# proxy is drawn over it. The row keeps its family's seat for boarding.\nproxies = []\n");
    s.push_str("\n[sockets]\n# None on disk (the export found 0 StaticMeshSockets); boarding derives the\n# seat from the family's seat part.\nfrom_art = []\n");
    s
}
