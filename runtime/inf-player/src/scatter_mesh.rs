//! The authored meshes scattered content draws (wave TER2b) — the shipped
//! player's half of `inf_render::ScatterMeshes`.
//!
//! # What this closes
//!
//! `inf_pcg::PcgKind::mesh` has named a `.inf_mesh` since P10.5 and **nothing
//! drew it**. The GUID was read by the cook's dependency closure — so the bytes
//! reached the pack — and by nothing on the draw path: `push_scatter` built a
//! `PrimMesh::Cube` for every instance whatever its kind said. Wave TER2a
//! authored three ground-cover props for the island, committed them, cooked them,
//! and shipped 16 771 tinted cubes.
//!
//! Both ends of that gap are now closed: `PcgInstance::mesh` carries the GUID
//! through evaluation (`kind_index` could not — it is rule-local and populations
//! are concatenated), and this module turns the GUID into the two flat arrays the
//! scatter raster pulls from.
//!
//! # Why it enumerates `.inf_pcg` rather than `.inf_mesh`
//!
//! A pack's meshes are mostly not scatter kinds — a character's body is a
//! `.inf_mesh` too — and uploading every one of them to a storage buffer at boot
//! would cost megabytes to draw nothing. So the walk is *demand-shaped*: read the
//! scatter documents, take the GUIDs their kinds name, and load exactly those.
//! It runs once, at level load, because a projection runs every frame in the
//! shipped player and must not open a file.
//!
//! # The vgeom path is deliberately not used
//!
//! A `.inf_vmesh` is a meshlet DAG with no plain index list in it, and the cook
//! does not derive one for a small prop in any case (`[vgeom] min_triangles`).
//! Ground cover is a handful of triangles at a handful of metres; it wants a
//! vertex buffer, not a virtualized one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_render::{ScatterGeometry, ScatterMeshes};
use uuid::Uuid;

/// The triangle ceiling both hosts refuse past — the renderer's own, re-exported
/// so a reader of this module can see which number it enforces.
pub use inf_render::MAX_SCATTER_MESH_TRIANGLES;

/// Every mesh GUID a `.inf_pcg` document's scatter kinds name, deduplicated and
/// in **sorted** order.
///
/// Sorted rather than in document order because the result feeds a load loop
/// whose warnings and (on a duplicate GUID) whose winner would otherwise depend
/// on which document the pack index happened to list first.
fn kind_meshes(payload: &inf_pcg::PcgAssetPayload) -> Vec<Uuid> {
    let mut out: Vec<Uuid> = payload
        .document
        .layers
        .iter()
        .flat_map(|l| &l.rules)
        .flat_map(|r| &r.kinds)
        .filter_map(|k| k.mesh)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// One mesh's payload bytes as scatter geometry, or `None` with a warning.
fn geometry_from_bytes(id: Uuid, bytes: &[u8]) -> Option<Arc<ScatterGeometry>> {
    let mesh: inf_mesh::MeshAsset = match inf_asset::decode(bytes) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("inf-player: scatter mesh {id} does not decode: {e}");
            return None;
        }
    };
    let (positions, normals, _uvs, _tangents, indices) = mesh.vgeom_streams();
    let geom = ScatterGeometry::from_streams(&positions, &normals, &indices);
    if geom.is_empty() {
        tracing::warn!("inf-player: scatter mesh {id} has no drawable triangles");
        return None;
    }
    if geom.triangle_count() > MAX_SCATTER_MESH_TRIANGLES {
        tracing::warn!(
            "inf-player: scatter mesh {id} is {} triangles, past the {} the scatter \
             path draws per instance; it falls back to the placeholder primitive",
            geom.triangle_count(),
            MAX_SCATTER_MESH_TRIANGLES
        );
        return None;
    }
    Some(Arc::new(geom))
}

/// Load every scatter-kind mesh a **cooked pack** names.
///
/// Never fails: a document that does not decode, a mesh that is absent and a mesh
/// past the triangle ceiling are each skipped with a warning, and the instances
/// that named them draw the placeholder primitive. One bad asset must not take a
/// level down — the `VmeshRegistry::from_pack` rule.
pub fn from_pack(reader: &PackReader) -> ScatterMeshes {
    let mut docs: Vec<AssetId> = reader
        .index()
        .filter(|e| e.kind == AssetKind::Pcg)
        .map(|e| e.guid)
        .collect();
    docs.sort_by_key(|a| a.0);
    let mut wanted: Vec<Uuid> = Vec::new();
    for guid in docs {
        let Ok(bytes) = reader.read(guid) else {
            continue;
        };
        match inf_asset::decode::<inf_pcg::PcgAssetPayload>(&bytes) {
            Ok(p) => wanted.extend(kind_meshes(&p)),
            Err(e) => tracing::warn!("inf-player: .inf_pcg {guid} does not decode: {e}"),
        }
    }
    wanted.sort();
    wanted.dedup();
    let mut out: ScatterMeshes = HashMap::new();
    for id in wanted {
        let Ok(bytes) = reader.read(AssetId(id)) else {
            tracing::warn!("inf-player: a scatter kind names mesh {id} and the pack has none");
            continue;
        };
        if let Some(g) = geometry_from_bytes(id, &bytes) {
            out.insert(id.as_u128(), g);
        }
    }
    out
}

/// The loose-directory twin of [`from_pack`] — the `.inf_lvl` dev path, where
/// assets are files beside the level rather than entries in a pack.
///
/// Non-recursive and path-sorted, exactly like `VmeshRegistry::from_dir`.
pub fn from_dir(dir: &Path) -> ScatterMeshes {
    let mut docs: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_pcg"))
            .collect(),
        Err(_) => return HashMap::new(),
    };
    docs.sort();
    let mut wanted: Vec<Uuid> = Vec::new();
    for p in docs {
        let Ok(bytes) = std::fs::read(&p) else {
            continue;
        };
        match inf_asset::decode::<inf_pcg::PcgAssetPayload>(&bytes) {
            Ok(d) => wanted.extend(kind_meshes(&d)),
            Err(e) => tracing::warn!("inf-player: bad .inf_pcg {}: {e}", p.display()),
        }
    }
    wanted.sort();
    wanted.dedup();
    // A loose directory is keyed by sidecar GUID, so the mesh files are found by
    // walking them once rather than by guessing a file name from a GUID.
    let mut by_guid: HashMap<Uuid, PathBuf> = HashMap::new();
    let mut meshes: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_mesh"))
            .collect(),
        Err(_) => Vec::new(),
    };
    meshes.sort();
    for p in meshes {
        if let Ok(side) = inf_asset::AssetSidecar::load(&p) {
            by_guid.insert(side.guid.uuid(), p);
        }
    }
    let mut out: ScatterMeshes = HashMap::new();
    for id in wanted {
        let Some(path) = by_guid.get(&id) else {
            tracing::warn!(
                "inf-player: a scatter kind names mesh {id} and {} has none",
                dir.display()
            );
            continue;
        };
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        if let Some(g) = geometry_from_bytes(id, &bytes) {
            out.insert(id.as_u128(), g);
        }
    }
    out
}

/// The **PIE payload** twin of [`from_pack`] (wave FIX2): the scatter kinds a
/// payload's `.inf_pcg` documents name, loaded from the `.inf_mesh` bytes the
/// same payload carries.
///
/// **Demand-shaped, exactly like [`from_pack`]**, and that is the whole reason
/// this function reads two vectors instead of one. `ScenePayload::meshes` is a
/// general bag of `.inf_mesh` bytes — a character's body is in there too — and
/// uploading every entry to a storage buffer would cost megabytes to draw
/// nothing (the module's own opening rule). So the graphs decide: decode each
/// carried `.inf_pcg`, take the GUIDs its kinds name, and take exactly those
/// meshes.
///
/// Deterministic: the wanted set is sorted and deduplicated before anything is
/// loaded, so a payload whose graphs arrive in a different order produces the
/// same table with the same duplicate winner.
///
/// A graph that does not decode, and a kind whose mesh the payload does not
/// carry, are each skipped with a warning and their instances draw the scatter
/// path's own placeholder primitive — the `from_pack` rule, and not the
/// `MeshRef.asset` one: a scattered instance has no authored transform of its
/// own to make a wrong claim about.
pub fn from_payload(pcgs: &[(Uuid, Vec<u8>)], meshes: &[(Uuid, Vec<u8>)]) -> ScatterMeshes {
    let mut wanted: Vec<Uuid> = Vec::new();
    for (guid, bytes) in pcgs {
        match inf_asset::decode::<inf_pcg::PcgAssetPayload>(bytes) {
            Ok(p) => wanted.extend(kind_meshes(&p)),
            Err(e) => tracing::warn!("inf-player: .inf_pcg {guid} does not decode: {e}"),
        }
    }
    wanted.sort();
    wanted.dedup();
    let by_guid: HashMap<Uuid, &[u8]> = meshes.iter().map(|(g, b)| (*g, b.as_slice())).collect();
    let mut out: ScatterMeshes = HashMap::new();
    for id in wanted {
        let Some(bytes) = by_guid.get(&id) else {
            tracing::warn!(
                "inf-player: a scatter kind names mesh {id} and the PIE payload carries none"
            );
            continue;
        };
        if let Some(g) = geometry_from_bytes(id, bytes) {
            out.insert(id.as_u128(), g);
        }
    }
    out
}

/// **Register the building module meshes** (island wave I8b) — the twelve shape
/// families every palette module draws.
///
/// They name no `.inf_mesh` file and never will: the geometry is a function of
/// the module's own name, minted under a private salt by
/// `inf_pcg::building::modules`. So neither loader above can find them by
/// scanning a pack or a directory, and both hosts add them to the table they
/// just built instead — from one Ring-0 source, through one Ring-0 flattener.
///
/// **Existing entries win.** A project that really does ship an `.inf_mesh`
/// under one of these ids has authored it deliberately, and an engine default
/// must not overwrite authored content.
///
/// MIRROR: identical in `inf_viewport::host`, pinned by `inf-editor-core`'s
/// `tests/projector_mirror.rs`.
pub fn add_building_modules(table: &mut inf_render::ScatterMeshes) {
    // MIRROR-BEGIN building_module_table
    for (id, m) in inf_pcg::building::modules::module_meshes() {
        let key = id.as_u128();
        if table.contains_key(&key) {
            continue;
        }
        let g = inf_render::ScatterGeometry::from_streams(&m.positions, &m.normals, &m.indices);
        if g.is_empty() || g.triangle_count() > inf_render::MAX_SCATTER_MESH_TRIANGLES {
            continue;
        }
        table.insert(key, std::sync::Arc::new(g));
    }
    // MIRROR-END building_module_table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(meshes: &[Option<Uuid>]) -> inf_pcg::PcgAssetPayload {
        let rule = inf_pcg::PcgRule {
            name: "r".into(),
            sampler: inf_pcg::SamplerDef::Constant(1.0),
            scatter: inf_pcg::ScatterParams::default(),
            kinds: meshes
                .iter()
                .map(|m| inf_pcg::PcgKind {
                    mesh: *m,
                    weight: 1.0,
                })
                .collect(),
        };
        inf_pcg::PcgAssetPayload::new(inf_pcg::PcgDocument::single_layer("l", vec![rule]))
    }

    #[test]
    fn kind_meshes_is_sorted_deduplicated_and_skips_bare_transforms() {
        let a = Uuid::from_u128(0x22);
        let b = Uuid::from_u128(0x11);
        let got = kind_meshes(&doc(&[Some(a), None, Some(b), Some(a)]));
        assert_eq!(got, vec![b, a], "sorted and deduplicated");
        // …and a document of bare transforms wants nothing at all, which is what
        // keeps a level with no cover from paying for this walk.
        assert!(kind_meshes(&doc(&[None, None])).is_empty());
    }

    #[test]
    fn a_mesh_past_the_triangle_ceiling_is_refused_rather_than_uploaded() {
        // One triangle is fine…
        let one = inf_mesh::MeshAsset::new(
            vec![inf_mesh::SubMesh {
                name: "t".into(),
                vertices: vec![inf_mesh::MeshVertex::default(); 3],
                indices: vec![0, 1, 2],
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["m".into()],
        );
        let id = Uuid::from_u128(1);
        let bytes = inf_asset::encode(&one).expect("encode");
        assert!(geometry_from_bytes(id, &bytes).is_some());

        // …and one past the ceiling is not. Built by repeating the same triangle,
        // so the refusal is about the COUNT and not about the geometry.
        let n = MAX_SCATTER_MESH_TRIANGLES + 1;
        let big = inf_mesh::MeshAsset::new(
            vec![inf_mesh::SubMesh {
                name: "t".into(),
                vertices: vec![inf_mesh::MeshVertex::default(); 3],
                indices: (0..n).flat_map(|_| [0u32, 1, 2]).collect(),
                material_slot: Some(0),
                skin: Vec::new(),
            }],
            vec!["m".into()],
        );
        let bytes = inf_asset::encode(&big).expect("encode");
        assert!(
            geometry_from_bytes(id, &bytes).is_none(),
            "{n} triangles is past the {MAX_SCATTER_MESH_TRIANGLES} ceiling"
        );
    }
}
