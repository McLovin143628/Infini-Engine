//! **In-editor `.inf_vmesh` derivation** (P18.3): `.inf_vmesh` stops being
//! cook-only.
//!
//! Until this batch a meshlet LOD DAG existed only inside a cooked pack, which is
//! why the interactive viewport could not draw a `MeshRef.asset` at all:
//! [`inf_render::RenderScene`] has exactly one door for real (non-primitive)
//! geometry — [`VgeomAsset`](inf_render::VgeomAsset) + its instances — and nothing
//! in the editor ever built one. So the editor derives its own, beside the mesh,
//! from the same [`inf_vgeom::build_vgeom`] the cook runs, into the same v2 paged
//! image [`inf_vgeom::build_vgeom_asset`] writes.
//!
//! # The three rules
//!
//! 1. **The id is computed, never indexed.** The derived asset's GUID is
//!    [`inf_vgeom::derived_vmesh_id`] of the mesh's GUID — the identical bijection
//!    the cook and the shipped player use (one Ring-0 constant since P18.3), so
//!    the viewport resolves `MeshRef.asset` → its vmesh by *computing* the id, with
//!    no side table to keep in sync and no second source of truth about which
//!    vmesh belongs to which mesh.
//!
//! 2. **It is content-hash cached, like every other derived artifact.** The
//!    sidecar's `import` table records the **source mesh's** content hash
//!    ([`SOURCE_HASH_KEY`]); a derivation whose recorded hash still matches the
//!    mesh on disk is skipped outright. This is the `ImportCache` /
//!    `ThumbnailCache` precedent — a re-import rebuilds, a rescan of unchanged
//!    content does not, and "unchanged" is decided by bytes rather than by
//!    timestamps.
//!
//! 3. **Every mesh with a triangle gets one.** The *cook* only virtualizes meshes
//!    past `VgeomCookOptions::min_triangles` (2048 by default) because below it the
//!    classic path is cheaper. The editor cannot make that trade: vgeom is its
//!    **only** real-geometry path, so a mesh below the threshold would keep drawing
//!    the placeholder cube this batch exists to delete. See [`MIN_TRIANGLES`] for
//!    the honest consequence.
//!
//! # Determinism
//!
//! [`inf_vgeom::build_vgeom`] is pool-size-invariant (its per-group simplification
//! runs through `inf_core::parallel_map`'s in-order collect) and
//! [`inf_vgeom::build_vgeom_asset`] is a pure function of the DAG, so two
//! derivations of one mesh produce **byte-identical** payloads — the property the
//! cook already relies on, now relied on twice.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, ContentHash, Result};
use inf_mesh::MeshAsset;

use super::AssetProject;

/// The sidecar `import` key holding the **source mesh's** content hash (lowercase
/// hex). Present only on editor-derived `.inf_vmesh` assets; a cooked pack has no
/// sidecars at all, so there is no ambiguity about which producer wrote a payload.
pub const SOURCE_HASH_KEY: &str = "source_mesh_hash";

/// The sidecar `import` key naming the source mesh's GUID, so a human reading the
/// TOML can see what the artifact came from without inverting the id bijection.
pub const SOURCE_MESH_KEY: &str = "source_mesh";

/// Minimum triangles for the editor to derive a DAG. **One**, deliberately.
///
/// The cook's threshold is 2048 (`VgeomCookOptions::min_triangles`): below it a
/// virtualized mesh costs more than it saves, and the cook can afford that opinion
/// because a below-threshold mesh still ships its `.inf_mesh`. The editor cannot —
/// `RenderScene` has no classic `.inf_mesh` draw path, only `PrimMesh` primitives
/// and vgeom — so any mesh without a derived DAG renders as a placeholder cube.
///
/// **Honest consequence, recorded rather than papered over:** a *shipped* build of
/// a scene containing a sub-2048-triangle imported mesh still draws that
/// placeholder, because the cook declines to derive one. That is a cook default
/// this batch deliberately does not touch (it is outside the file boundary and
/// changes shipped bytes); lowering `min_triangles` to 1 is the one-line follow-up
/// that makes the editor and the player agree for small props.
pub const MIN_TRIANGLES: usize = 1;

/// What [`ensure_vmesh`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmeshDerivation {
    /// A DAG was built and written; the derived asset id is carried back.
    Built(AssetId),
    /// An up-to-date derived asset was already on disk (hash hit) — nothing ran.
    Cached(AssetId),
    /// The mesh has no geometry to virtualize (fewer than [`MIN_TRIANGLES`]
    /// triangles, or no indices at all).
    Skipped,
}

impl VmeshDerivation {
    /// The derived asset id, if one exists after this call.
    pub fn asset(self) -> Option<AssetId> {
        match self {
            VmeshDerivation::Built(id) | VmeshDerivation::Cached(id) => Some(id),
            VmeshDerivation::Skipped => None,
        }
    }

    /// Whether a DAG was actually built (as opposed to served from the cache).
    pub fn rebuilt(self) -> bool {
        matches!(self, VmeshDerivation::Built(_))
    }
}

/// One planned derivation: everything the build and the commit need, resolved
/// from the database **once**, so neither of them has to hold it.
///
/// This split is the whole point of the type. `build_vgeom` on a real character
/// runs for seconds; doing that under the project mutex is the P16.4a stall
/// wearing a different hat — the asset commands, the progress tick that carries
/// the job's own events, and the wizard's Cancel all take that lock. So: plan
/// under a short lock, build holding nothing, re-acquire briefly to register.
#[derive(Debug, Clone)]
pub struct VmeshPlan {
    /// The source `.inf_mesh`.
    pub mesh: AssetId,
    /// Its derived `.inf_vmesh` id (the computed bijection).
    pub derived: AssetId,
    /// Where the source payload lives.
    mesh_path: PathBuf,
    /// The source's content hash — recorded in the derived sidecar, and the thing
    /// the next run compares against.
    mesh_hash: ContentHash,
    /// Where the derived payload goes.
    out_path: PathBuf,
}

/// Whether `mesh_id` needs a derivation, and what one would involve.
///
/// `None` = nothing to do: not a mesh, or a derived asset already on disk whose
/// recorded source hash still matches. Pure database reads — no file IO beyond an
/// `exists()` — so this is the short-lock phase.
///
/// `taken` carries the output paths already claimed by *this* planning pass, so
/// two meshes cannot be planned onto one file (each would see the other's path as
/// free, since neither has been written yet).
pub fn plan_vmesh(
    project: &AssetProject,
    mesh_id: AssetId,
    taken: &mut BTreeSet<PathBuf>,
) -> Option<VmeshPlan> {
    let entry = project.db().get(mesh_id)?;
    if entry.kind() != AssetKind::Mesh {
        return None;
    }
    let mesh_hash = entry.sidecar.content_hash;
    let mesh_path = entry.path.clone();
    let mesh_name = entry.name.clone();
    let derived = inf_vgeom::derived_vmesh_id(mesh_id);

    // The cache check. The recorded hash is the SOURCE mesh's, not the payload's:
    // the question is "would rebuilding produce the same bytes?", and the answer is
    // a property of the input. (The payload's own hash lives in `content_hash`, as
    // for any asset.)
    if let Some(existing) = project.db().get(derived) {
        if existing.path.exists() && recorded_source_hash(&existing.sidecar) == Some(mesh_hash) {
            return None;
        }
    }
    let out_path = derived_path(project, &mesh_path, &mesh_name, derived, taken);
    taken.insert(out_path.clone());
    Some(VmeshPlan {
        mesh: mesh_id,
        derived,
        mesh_path,
        mesh_hash,
        out_path,
    })
}

/// Build a planned derivation's payload. **Holds no lock and touches no database**
/// — it reads one file and runs [`build_payload`], which is where the seconds go.
///
/// `Ok(None)` = the mesh has no geometry to virtualize (fewer than
/// [`MIN_TRIANGLES`] triangles, or no indices), which is a decision that needs the
/// decoded payload and so cannot be made while planning.
pub fn build_vmesh(plan: &VmeshPlan) -> Result<Option<Vec<u8>>> {
    let bytes = std::fs::read(&plan.mesh_path)?;
    let mesh: MeshAsset = inf_asset::decode(&bytes)?;
    if mesh.triangle_count() < MIN_TRIANGLES {
        return Ok(None);
    }
    let (positions, normals, uvs, indices) = mesh.vgeom_streams();
    if indices.len() < 3 {
        return Ok(None);
    }
    Ok(Some(build_payload(&positions, &normals, &uvs, &indices)?))
}

/// Write a built payload and register it. The short second lock phase.
///
/// The payload is a **raw 16-byte-aligned image**, exactly like `.inf_terrain`:
/// routing it through the generic `write_asset` would frame it with a bincode
/// length prefix and knock every section off its alignment, which is why
/// `register_written_asset` exists. Same reasoning, same door — and, since P18.3's
/// audit, the same **atomic write** discipline: temp file + rename, so the content
/// watcher (which fires on the write) can only ever observe the whole old payload
/// or the whole new one. A multi-megabyte `fs::write` is a wide window for a
/// reader to see a truncated image, and `VgeomSource::from_payload` would reject
/// it — turning a race into a mesh that silently stops rendering.
pub fn commit_vmesh(
    project: &mut AssetProject,
    plan: &VmeshPlan,
    payload: &[u8],
) -> Result<AssetId> {
    write_payload_atomically(&plan.out_path, payload)?;
    let import = derived_import_table(plan.mesh, plan.mesh_hash);
    project.register_written_asset(
        plan.out_path.clone(),
        AssetKind::MeshletMesh,
        ContentHash::of(payload),
        Some(format!("derived:{}", plan.mesh)),
        Some(import),
        // Always the computed id: the bijection IS the identity, so a derivation
        // must never mint a fresh GUID (that would orphan the previous artifact and
        // break the viewport's compute-the-id lookup).
        Some(plan.derived),
    )
}

/// Write `payload` to `path` atomically (temp file in the same directory, then
/// rename over the target).
///
/// The same discipline `inf_terrain::write_terrain_asset` and the sidecar writer
/// use, and for the same reason: a rename is atomic on both NTFS and POSIX, so
/// there is no instant at which `path` holds half an image. A failed attempt
/// removes its own temp file rather than leaving litter in the content root.
///
/// A crash between the temp write and the rename leaves `<name>.inf_vmesh.<pid>.tmp`
/// and no change to the asset — the derivation simply re-runs next open (it is
/// content-hash driven, so it will), which is why the stray is inert rather than
/// something the database could adopt: `.tmp` is not a recognized asset extension.
fn write_payload_atomically(path: &Path, payload: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(format!(".{}.tmp", std::process::id()));
        PathBuf::from(s)
    };
    if let Err(e) = std::fs::write(&tmp, payload).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

/// Ensure `mesh_id`'s derived `.inf_vmesh` exists and is current — the
/// **synchronous** door, for a caller that already holds the project and is
/// itself the unit of work (the import orchestrator).
///
/// Cheap and idempotent on the hit path: [`plan_vmesh`] answers from the database
/// alone. The sweep deliberately does **not** use this — see [`VmeshPlan`].
pub fn ensure_vmesh(project: &mut AssetProject, mesh_id: AssetId) -> Result<VmeshDerivation> {
    if project.db().get(mesh_id).is_none() {
        return Err(inf_asset::AssetError::UnknownAsset(mesh_id));
    }
    let mut taken = BTreeSet::new();
    let Some(plan) = plan_vmesh(project, mesh_id, &mut taken) else {
        // Either not a mesh, or already current. Distinguish for the caller.
        let derived = inf_vgeom::derived_vmesh_id(mesh_id);
        return Ok(if project.db().contains(derived) {
            VmeshDerivation::Cached(derived)
        } else {
            VmeshDerivation::Skipped
        });
    };
    let Some(payload) = build_vmesh(&plan)? else {
        return Ok(VmeshDerivation::Skipped);
    };
    commit_vmesh(project, &plan, &payload)?;
    Ok(VmeshDerivation::Built(plan.derived))
}

/// Plan a derivation for **every** `.inf_mesh` in the project (the short-lock
/// phase of the project-open sweep).
///
/// Mesh-id ordered, so the sweep is deterministic; already-current meshes are not
/// planned at all, which is what makes re-running it on an up-to-date project
/// cost a hash comparison each.
pub fn plan_sweep(project: &AssetProject) -> Vec<VmeshPlan> {
    let meshes: BTreeSet<AssetId> = project
        .db()
        .iter()
        .filter(|e| e.kind() == AssetKind::Mesh)
        .map(|e| e.id())
        .collect();
    let mut taken = BTreeSet::new();
    meshes
        .into_iter()
        .filter_map(|id| plan_vmesh(project, id, &mut taken))
        .collect()
}

/// Derive (or refresh) the `.inf_vmesh` for every `.inf_mesh` in the project,
/// **holding the project throughout**.
///
/// The simple door, for tests and for a caller that owns the project outright. The
/// editor's background sweep does not use it — it interleaves
/// [`plan_sweep`] / [`build_vmesh`] / [`commit_vmesh`] so the lock is held only for
/// the database work. Returns the assets actually built, in mesh-id order.
/// Individual failures are logged and skipped: one unreadable mesh must not stop
/// the rest of a project from rendering.
pub fn sweep(project: &mut AssetProject) -> Vec<AssetId> {
    let plans = plan_sweep(project);
    let mut built = Vec::new();
    for plan in plans {
        match build_vmesh(&plan).and_then(|p| {
            p.map(|payload| commit_vmesh(project, &plan, &payload))
                .transpose()
        }) {
            Ok(Some(id)) => built.push(id),
            Ok(None) => {}
            Err(e) => tracing::warn!(
                "inf-editor-core: vmesh derivation failed for {}: {e}",
                plan.mesh
            ),
        }
    }
    built
}

/// Delete the `.inf_vmesh` derived from `mesh_id`, if the editor wrote one.
///
/// Called when the mesh itself is deleted. Leaving the artifact behind would be
/// worse than a leak: the viewport resolves by **computing** the id, so a stale
/// `.inf_vmesh` keeps drawing geometry for a mesh asset that no longer exists —
/// precisely the "no stale render" case P18.3 owes. Returns whether anything was
/// removed.
pub fn remove_derived_vmesh(project: &mut AssetProject, mesh_id: AssetId) -> bool {
    let derived_id = inf_vgeom::derived_vmesh_id(mesh_id);
    let Some(entry) = project.db().get(derived_id) else {
        return false;
    };
    // Only ever remove an artifact this module wrote — an authored `.inf_vmesh`
    // that happens to sit on the derived id is not ours to delete.
    if entry.kind() != AssetKind::MeshletMesh || recorded_source_hash(&entry.sidecar).is_none() {
        return false;
    }
    project.delete(derived_id, true).is_ok()
}

/// Build the v2 paged `.inf_vmesh` image for a mesh's raw streams.
///
/// Split out so the determinism gate can call it twice on one input without an
/// `AssetProject`, and so the *only* place build parameters are chosen is here —
/// [`inf_vgeom::BuildParams::default`], the same values the cook passes, because a
/// DAG built with different parameters in the editor would mean the preview and
/// the shipped build disagree about geometry, not just about detail.
pub fn build_payload(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Result<Vec<u8>> {
    let _span = tracing::info_span!(
        "editor_derive_vmesh",
        verts = positions.len(),
        tris = indices.len() / 3
    )
    .entered();
    let dag = inf_vgeom::build_vgeom(
        positions,
        normals,
        uvs,
        indices,
        inf_vgeom::BuildParams::default(),
    );
    Ok(inf_vgeom::build_vgeom_asset(&dag)
        .map_err(|e| inf_asset::AssetError::Import(e.to_string()))?
        .into_bytes())
}

/// The sidecar `import` table an editor-derived vmesh carries.
fn derived_import_table(mesh_id: AssetId, mesh_hash: ContentHash) -> toml::Table {
    let mut t = toml::Table::new();
    t.insert(
        SOURCE_HASH_KEY.to_string(),
        toml::Value::String(mesh_hash.to_hex()),
    );
    t.insert(
        SOURCE_MESH_KEY.to_string(),
        toml::Value::String(mesh_id.to_string()),
    );
    t
}

/// The source-mesh hash a derived sidecar records, if it is one of ours.
///
/// A malformed value reads as "no recorded hash", i.e. a rebuild — the safe way
/// to be wrong about a cache.
fn recorded_source_hash(sidecar: &inf_asset::AssetSidecar) -> Option<ContentHash> {
    let hex = sidecar.import.as_ref()?.get(SOURCE_HASH_KEY)?.as_str()?;
    u128::from_str_radix(hex, 16).ok().map(ContentHash)
}

/// Where the derived payload lands: **beside its mesh**, under the mesh's own
/// stem, so a content root stays browsable and a human can see which artifact came
/// from which mesh without reading a sidecar.
///
/// Reuses the existing path when the asset is already registered, so a rebuild
/// overwrites in place instead of accumulating `_1`, `_2`, … copies. `taken` holds
/// the paths this planning pass has already claimed: `unique_asset_path` only
/// checks the filesystem, and during planning nothing has been written yet, so
/// without it two meshes could be planned onto one file.
fn derived_path(
    project: &AssetProject,
    mesh_path: &Path,
    mesh_name: &str,
    derived_id: AssetId,
    taken: &BTreeSet<PathBuf>,
) -> PathBuf {
    if let Some(existing) = project.db().get(derived_id) {
        return existing.path.clone();
    }
    let dir = mesh_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| project.root().to_path_buf());
    let mut candidate = project
        .unique_asset_path(&dir, mesh_name, "inf_vmesh")
        .unwrap_or_else(|_| dir.join(format!("{mesh_name}.inf_vmesh")));
    if taken.contains(&candidate) {
        // Fall back to a name that cannot collide: the derived id is unique by
        // construction, so this terminates immediately.
        candidate = dir.join(format!(
            "{mesh_name}_{:08x}.inf_vmesh",
            derived_id.uuid().as_u128() as u32
        ));
    }
    candidate
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_mesh::{MeshVertex, SubMesh};

    /// A grid mesh dense enough to produce several meshlets (and therefore a real
    /// multi-page DAG) but fast to build.
    fn grid_mesh(n: u32) -> MeshAsset {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(MeshVertex {
                    position: [i as f32, ((i * j) % 7) as f32 * 0.1, j as f32],
                    normal: [0.0, 1.0, 0.0],
                    uv: [i as f32 / n as f32, j as f32 / n as f32],
                    ..Default::default()
                });
            }
        }
        let idx = |i: u32, j: u32| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                indices.extend_from_slice(&[idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        MeshAsset::new(
            vec![SubMesh {
                name: "grid".into(),
                vertices,
                indices,
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        )
    }

    fn project_with_mesh(dir: &std::path::Path, mesh: &MeshAsset) -> (AssetProject, AssetId) {
        let mut proj = AssetProject::open(dir).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let id = proj
            .write_asset(&d, "Grid", mesh, None, vec![], None)
            .unwrap();
        (proj, id)
    }

    /// The headline: an imported mesh gains a derived, **loadable** v2 `.inf_vmesh`
    /// under the computed id, with real geometry in it.
    #[test]
    fn deriving_a_mesh_writes_a_loadable_paged_asset_under_the_computed_id() {
        let dir = tempfile::tempdir().unwrap();
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &grid_mesh(12));

        let out = ensure_vmesh(&mut proj, mesh_id).unwrap();
        assert!(out.rebuilt(), "first derivation must build");
        let vmesh_id = out.asset().unwrap();
        assert_eq!(vmesh_id, inf_vgeom::derived_vmesh_id(mesh_id));

        let entry = proj.db().get(vmesh_id).expect("registered in the DB");
        assert_eq!(entry.kind(), AssetKind::MeshletMesh);
        let bytes = std::fs::read(&entry.path).unwrap();
        assert!(
            inf_vgeom::asset::is_v2(&bytes),
            "the v2 paged image, not v1"
        );
        let source = inf_vgeom::VgeomSource::from_payload(bytes).expect("indexable");
        assert!(source.meshlet_count() > 0, "the DAG carries meshlets");
        assert!(!source.pages().is_empty(), "and at least a root page");
    }

    /// Determinism: two builds of one mesh are byte-identical, and the pool the
    /// build ran on cannot change that (`build_vgeom` is pool-size-invariant).
    #[test]
    fn derivation_is_byte_deterministic() {
        let mesh = grid_mesh(10);
        let (p, n, u, i) = mesh.vgeom_streams();
        let a = build_payload(&p, &n, &u, &i).unwrap();
        let b = build_payload(&p, &n, &u, &i).unwrap();
        assert_eq!(a, b, "two derivations of one mesh must agree byte for byte");
        assert!(!a.is_empty());
    }

    /// The cache: an unchanged mesh never rebuilds, and the artifact on disk is
    /// untouched (same bytes, same modification identity from the DB's view).
    #[test]
    fn an_unchanged_mesh_is_a_cache_hit_and_rebuilds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &grid_mesh(9));

        let first = ensure_vmesh(&mut proj, mesh_id).unwrap();
        assert!(first.rebuilt());
        let vmesh_id = first.asset().unwrap();
        let before = std::fs::read(&proj.db().get(vmesh_id).unwrap().path).unwrap();

        for _ in 0..3 {
            let again = ensure_vmesh(&mut proj, mesh_id).unwrap();
            assert_eq!(again, VmeshDerivation::Cached(vmesh_id), "no rebuild");
        }
        let after = std::fs::read(&proj.db().get(vmesh_id).unwrap().path).unwrap();
        assert_eq!(before, after);
    }

    /// **Re-import invalidation**: rewriting the mesh payload under the same GUID
    /// changes its content hash, so the next `ensure` rebuilds and the artifact's
    /// bytes actually change. This is the lifecycle case a stale-cache bug hides in.
    #[test]
    fn rewriting_the_mesh_invalidates_the_derived_asset() {
        let dir = tempfile::tempdir().unwrap();
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &grid_mesh(8));
        let vmesh_id = ensure_vmesh(&mut proj, mesh_id).unwrap().asset().unwrap();
        let before = std::fs::read(&proj.db().get(vmesh_id).unwrap().path).unwrap();
        let path_before = proj.db().get(vmesh_id).unwrap().path.clone();

        // Re-import: same GUID, different geometry.
        proj.rewrite_payload(mesh_id, &grid_mesh(14), vec![])
            .unwrap();
        let out = ensure_vmesh(&mut proj, mesh_id).unwrap();
        assert!(out.rebuilt(), "a changed source must rebuild");
        assert_eq!(
            out.asset(),
            Some(vmesh_id),
            "the id is stable across rebuilds"
        );

        let entry = proj.db().get(vmesh_id).unwrap();
        assert_eq!(entry.path, path_before, "rebuilt in place, no _1 copy");
        let after = std::fs::read(&entry.path).unwrap();
        assert_ne!(before, after, "the derived bytes must follow the source");
        // And the fresh source hash is recorded, so the NEXT call is a hit again.
        assert_eq!(
            ensure_vmesh(&mut proj, mesh_id).unwrap(),
            VmeshDerivation::Cached(vmesh_id)
        );
    }

    /// Deleting the mesh takes its derived artifact with it — otherwise the
    /// viewport, which resolves by computing the id, keeps drawing geometry for an
    /// asset that is gone.
    #[test]
    fn deleting_a_mesh_removes_its_derived_vmesh() {
        let dir = tempfile::tempdir().unwrap();
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &grid_mesh(8));
        let vmesh_id = ensure_vmesh(&mut proj, mesh_id).unwrap().asset().unwrap();
        let vmesh_path = proj.db().get(vmesh_id).unwrap().path.clone();

        assert!(proj.delete(mesh_id, false).unwrap().is_empty(), "deleted");
        assert!(!proj.db().contains(vmesh_id), "derived asset unregistered");
        assert!(!vmesh_path.exists(), "derived payload removed from disk");
        assert!(
            !inf_asset::sidecar_path(&vmesh_path).exists(),
            "and its sidecar with it"
        );
    }

    /// **The derived payload is written atomically** (P18.3 audit).
    ///
    /// It is multiple megabytes, and the content watcher fires on the write — so a
    /// plain `fs::write` gives a reader a wide window to see a **truncated image**,
    /// which `VgeomSource::from_payload` rejects, turning a race into a mesh that
    /// silently stops rendering. Temp + rename removes the window: a rename is
    /// atomic on NTFS and POSIX alike, so the target is only ever the whole old
    /// payload or the whole new one.
    ///
    /// The decisive assertion is that a **failed** write does not create or touch
    /// the target at all — a plain `fs::write` would have created and truncated it
    /// before failing, which is exactly the partial state being ruled out.
    #[test]
    fn a_failed_payload_write_never_truncates_the_target() {
        let dir = tempfile::tempdir().unwrap();
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &grid_mesh(9));
        let vmesh_id = ensure_vmesh(&mut proj, mesh_id).unwrap().asset().unwrap();
        let path = proj.db().get(vmesh_id).unwrap().path.clone();
        let good = std::fs::read(&path).unwrap();
        assert!(inf_vgeom::VgeomSource::from_payload(good.clone()).is_ok());

        // Make the temp write fail by parking a DIRECTORY where it goes.
        let tmp = {
            let mut s = path.as_os_str().to_os_string();
            s.push(format!(".{}.tmp", std::process::id()));
            PathBuf::from(s)
        };
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(write_payload_atomically(&path, b"a shorter payload").is_err());

        // The previous payload is untouched — byte for byte, and still parseable.
        assert_eq!(
            std::fs::read(&path).unwrap(),
            good,
            "the target was modified"
        );
        assert!(
            inf_vgeom::VgeomSource::from_payload(std::fs::read(&path).unwrap()).is_ok(),
            "a reader could observe a payload that no longer parses"
        );
        std::fs::remove_dir(&tmp).unwrap();

        // A successful write replaces it wholly, and leaves no temp litter beside
        // the asset for the watcher (or the DB scan) to trip over.
        write_payload_atomically(&path, &good).unwrap();
        let strays: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(strays.is_empty(), "left temp files: {strays:?}");
    }

    /// A rebuild is observable only as old-or-new, never as an unreadable file:
    /// every payload this module ever leaves at the target parses.
    #[test]
    fn every_committed_payload_parses() {
        let dir = tempfile::tempdir().unwrap();
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &grid_mesh(8));
        let vmesh_id = ensure_vmesh(&mut proj, mesh_id).unwrap().asset().unwrap();
        let path = proj.db().get(vmesh_id).unwrap().path.clone();
        for n in [10u32, 12, 9] {
            proj.rewrite_payload(mesh_id, &grid_mesh(n), vec![])
                .unwrap();
            assert!(ensure_vmesh(&mut proj, mesh_id).unwrap().rebuilt());
            let bytes = std::fs::read(&path).unwrap();
            assert!(
                inf_vgeom::VgeomSource::from_payload(bytes).is_ok(),
                "a rebuild left an unreadable payload at n = {n}"
            );
        }
    }

    /// A degenerate mesh derives nothing rather than writing an empty artifact —
    /// the entity stays on the primitive-placeholder path, which is the correct
    /// rendering for a mesh with no triangles.
    #[test]
    fn a_mesh_with_no_triangles_derives_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let empty = MeshAsset::new(
            vec![SubMesh {
                name: "empty".into(),
                vertices: vec![],
                indices: vec![],
                material_slot: None,
                skin: Vec::new(),
            }],
            vec![],
        );
        let (mut proj, mesh_id) = project_with_mesh(dir.path(), &empty);
        assert_eq!(
            ensure_vmesh(&mut proj, mesh_id).unwrap(),
            VmeshDerivation::Skipped
        );
        assert!(!proj.db().contains(inf_vgeom::derived_vmesh_id(mesh_id)));
    }

    /// The project-wide sweep covers content imported before P18.3 existed, and is
    /// idempotent afterwards.
    #[test]
    fn the_sweep_derives_every_mesh_once() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let ids: Vec<AssetId> = (0..3)
            .map(|k| {
                proj.write_asset(&d, &format!("M{k}"), &grid_mesh(6 + k), None, vec![], None)
                    .unwrap()
            })
            .collect();

        assert_eq!(sweep(&mut proj).len(), 3, "all three derived");
        assert!(sweep(&mut proj).is_empty(), "second sweep is all hits");
        for id in ids {
            assert!(proj.db().contains(inf_vgeom::derived_vmesh_id(id)));
        }
    }
}
