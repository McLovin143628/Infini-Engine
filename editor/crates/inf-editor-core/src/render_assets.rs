//! **The interactive viewport's asset store** (P18.3): the loose-file half of
//! what `inf_player::vmesh::VmeshRegistry` is for a cooked pack.
//!
//! This closes the oldest documented gap in the engine. Since P4 a `MeshRef.asset`
//! — an entity bound to an imported glTF/OBJ — has drawn a **placeholder
//! primitive** in the editor while the shipped player drew its real geometry, for
//! one structural reason: the viewport thread had no asset database and
//! `.inf_vmesh` was cook-only. [`assets::vmesh`](crate::assets::vmesh) fixed the
//! second half; this module is the first.
//!
//! # Why it lives in Ring 1 and not in the viewport host
//!
//! Verbatim the reasoning that put [`crate::terrain_stream`] here:
//! `inf_viewport::host` is `#[cfg(any(windows, target_os = "macos"))]`, so
//! anything written there is invisible to the Linux CI leg. Keeping the store, the
//! resolution rules and the skinning-pose derivation here — platform neutral, GPU
//! free — means the tests below run on all three OSes and the host is left with
//! nothing but call sites. Same shape as [`EditorTerrainStreams`], same reason.
//!
//! [`EditorTerrainStreams`]: crate::terrain_stream::EditorTerrainStreams
//!
//! # Resolution is computed, never indexed
//!
//! `MeshRef.asset` names a `.inf_mesh` GUID. Its DAG's GUID is
//! [`inf_vgeom::derived_vmesh_id`] of it — the same Ring-0 bijection the cook
//! writes with and the player reads with — so a lookup is an XOR plus a hash-map
//! hit, with no side table that could disagree with the filesystem.
//!
//! # The one deliberate difference from the player, and why
//!
//! The player keys [`inf_render::VgeomAsset::id`] by the derived **GUID**. This
//! store keys it by the derived payload's **content hash**
//! ([`content_key`](LoadedVgeom::id)). The reason is that the two hosts have
//! different guarantees about their content:
//!
//! * a cooked pack is **immutable** — an id names one sequence of bytes forever;
//! * a project's content root is **not** — a re-import, an external tool, or a
//!   `rewrite_payload` can hand the same GUID different geometry mid-session.
//!
//! Both renderer paths cache GPU state by `VgeomAsset::id`
//! (`ClassicVgeomNode::geom` never evicts; `VgeomStreamer` holds pool blocks sized
//! from the source it registered), so a content change under a stable id is a
//! *stale render*, not a reload. Making the id content-addressed makes that
//! unrepresentable: changed bytes are a different asset, the old one leaves
//! `wants` and is fully evicted, and the new one pages in from scratch. It also
//! deduplicates — two entities on two mesh assets with identical geometry share
//! one upload. Everything else in the projection is byte-for-byte the player's
//! rule, which is what the mirror gate pins.
//!
//! # Skeletal meshes
//!
//! [`resolve_skinned`](EditorRenderAssets::resolve_skinned) **is a mirror since
//! P18.5**, and this note used to say the opposite. When P18.3 wrote it the shipped
//! player had no `SkeletalMesh` branch at all, so there was nothing to keep in sync
//! and GPU skinning had only ever been exercised by the `golden_skinned_mesh`
//! headless golden. P18.5 gave the player the matching branch
//! (`inf_player::skinned`), which closed that as the PIE-vs-shipping divergence it
//! had become — so this function and [`skinned_mesh_data`] are now pinned **character
//! for character** against their player twins by `tests/projector_mirror.rs`
//! (`the_skinned_pose_rule_is_identical_in_both_stores`,
//! `the_bind_space_rebuild_is_identical_in_both_stores`), with the receiver
//! (`&mut self` here, `&self` there — a difference in who owns the cache, not in the
//! rule) as the single normalized token. Edit either side and the other must follow.
//!
//! The rule itself (P24.1): the pose the **sim** evaluated for this entity when a
//! state machine published one, else the entity's [`AnimPlayer`] play-head sampled
//! through [`inf_anim::sample_clip`], else the rest pose.
//!
//! The **ONE permitted difference** between the two copies is the receiver:
//! `&mut self` here (the viewport owns its store mutably) against `&self` there
//! (that store memoizes behind its own locks because the projector holds it
//! immutably). That is a difference in who owns the cache, not in the rule, and
//! `projector_mirror.rs` normalizes exactly that token — doc block included —
//! before comparing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::Mat4;
use inf_anim::{AnimClip, Skeleton};
use inf_ecs::components::{AnimPlayer, SkeletalMesh};
use inf_ecs::pose::EvaluatedPose;
use inf_mesh::MeshAsset;
use inf_render::{SkinnedMeshData, SkinnedVertex};
use inf_vgeom::VgeomSource;
use uuid::Uuid;

/// Depth cap on the content-root walk — deep enough for any content layout,
/// shallow enough that a symlink loop cannot hang project open. Mirrors
/// `terrain_stream`'s cap for the same reason.
const MAX_CONTENT_DEPTH: u32 = 16;

/// The loose asset extensions this store indexes.
const INDEXED_EXTENSIONS: [&str; 4] = ["inf_vmesh", "inf_mesh", "inf_skel", "inf_anim"];

/// One indexed, opened `.inf_vmesh`.
#[derive(Clone)]
pub struct LoadedVgeom {
    /// The **content-addressed** render key (see the module docs) — the payload's
    /// `ContentHash` as a `u128`. Stable for stable bytes, different for different
    /// bytes, which is exactly the invalidation contract both render nodes' caches
    /// need.
    pub id: u128,
    pub source: Arc<VgeomSource>,
}

/// A skeletal mesh resolved to something the skinned pass can draw.
#[derive(Clone)]
pub struct SkinnedDraw {
    /// Bind-space geometry, shared between every entity using the same
    /// `(mesh, skeleton)` pair.
    pub mesh: Arc<SkinnedMeshData>,
    /// `global · inverse_bind` per joint, for this entity's current pose.
    pub palette: Vec<Mat4>,
    /// The `(mesh, skeleton)` pair, so a caller can deduplicate
    /// [`RenderScene::skinned_meshes`](inf_render::RenderScene::skinned_meshes)
    /// entries across instances.
    pub key: (Uuid, Uuid),
}

/// Every loose render asset the interactive viewport can draw, indexed by GUID
/// over a project's content root.
#[derive(Default)]
pub struct EditorRenderAssets {
    /// Where loose assets are looked up. `None` (the default) disables the whole
    /// store — a primitive-only document is unaffected either way, which is what
    /// keeps a viewport with no project byte-identical to its pre-P18.3 self.
    content_root: Option<PathBuf>,
    /// Asset GUID → loose file, rebuilt when the root changes and, lazily, when a
    /// lookup misses (see [`resolve_path`](Self::resolve_path)).
    index: HashMap<Uuid, PathBuf>,
    /// GUIDs a rescan has already failed to find, so a genuinely dangling
    /// reference costs **one** directory walk rather than one per frame. Cleared
    /// whenever the index is rebuilt.
    rescanned_for: HashSet<Uuid>,
    /// Opened vmesh sources by **mesh** GUID. `None` is a negative entry: this
    /// mesh has no usable DAG, so stop trying every frame.
    vgeom: HashMap<Uuid, Option<LoadedVgeom>>,
    /// Bind-space skinned geometry by `(mesh GUID, skeleton GUID)`.
    skinned: HashMap<(Uuid, Uuid), Option<Arc<SkinnedMeshData>>>,
    /// Decoded skeletons by GUID (shared by every entity bound to one).
    skeletons: HashMap<Uuid, Option<Arc<Skeleton>>>,
    /// Decoded clips by GUID.
    clips: HashMap<Uuid, Option<Arc<AnimClip>>>,
}

impl EditorRenderAssets {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point the store at a project's content root (or `None` to disable it).
    ///
    /// Rescans the loose-asset index and drops **everything** loaded, so a project
    /// switch can never serve the previous project's geometry.
    pub fn set_content_root(&mut self, root: Option<PathBuf>) {
        self.clear();
        self.index = match &root {
            Some(dir) => content_paths_by_guid(dir),
            None => HashMap::new(),
        };
        self.rescanned_for.clear();
        self.content_root = root;
    }

    /// The active content root.
    pub fn content_root(&self) -> Option<&Path> {
        self.content_root.as_deref()
    }

    /// Number of indexed loose assets.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Number of meshes with an opened DAG (negative entries excluded).
    pub fn loaded_vgeom(&self) -> usize {
        self.vgeom.values().filter(|v| v.is_some()).count()
    }

    /// Number of opened bind-space skinned meshes.
    pub fn loaded_skinned(&self) -> usize {
        self.skinned.values().filter(|v| v.is_some()).count()
    }

    /// Rebuild the loose-asset index **and drop every opened payload**.
    ///
    /// Called when the content database changed (an import landed, the watcher saw
    /// an external edit). Unlike terrain's `refresh_index`, which keeps live
    /// streams because a terrain's *tiles* are re-read per page, this drops what it
    /// holds: a `.inf_vmesh` is opened once and then only sliced, so a payload
    /// rewritten under the same GUID would otherwise be served from the old
    /// mapping forever. Re-opening costs a header + page-directory parse per
    /// referenced asset — a few hundred bytes each — which is the price of never
    /// rendering stale geometry.
    pub fn refresh_index(&mut self) {
        let Some(dir) = self.content_root.clone() else {
            return;
        };
        self.index = content_paths_by_guid(&dir);
        self.rescanned_for.clear();
        self.drop_loaded();
    }

    /// Release every opened payload, keeping the index and the root.
    ///
    /// The level-switch door (File ▸ Open / File ▸ New): the store is keyed on
    /// *asset* GUIDs rather than entity GUIDs, so nothing here is invalidated by a
    /// new document — but every byte it holds belongs to the old one's working set
    /// and nothing else would ever free it.
    pub fn clear(&mut self) {
        self.drop_loaded();
    }

    /// Drop everything opened for meshes **not** in `keep` — the per-projection
    /// audit ([`EditorTerrainStreams::retain_only`]'s twin).
    ///
    /// [`EditorTerrainStreams::retain_only`]: crate::terrain_stream::EditorTerrainStreams::retain_only
    ///
    /// `keep` is **every asset GUID the projection referenced** — `MeshRef.asset`,
    /// `SkeletalMesh.mesh`, `SkeletalMesh.skeleton`, `AnimPlayer.clip`. An asset
    /// that left the document — deleted, hidden, unbound, or belonging to a level
    /// that has since been replaced — is a whole `.inf_vmesh` mapping, or decoded
    /// skinned geometry, held for nothing. This is the only place that knows which
    /// assets are live, so it is the only place that can do it. (The P16.4b lesson,
    /// in mesh form.)
    ///
    /// Passing the *whole* referenced set rather than just meshes is what lets
    /// skeletons and clips be collected too: they are keyed independently of the
    /// mesh that reaches them, so a mesh-only `keep` would leak every skeleton a
    /// session ever posed.
    pub fn retain_only(&mut self, keep: impl IntoIterator<Item = Uuid>) {
        let live: HashSet<Uuid> = keep.into_iter().collect();
        self.vgeom.retain(|mesh, _| live.contains(mesh));
        self.skinned.retain(|(mesh, _), _| live.contains(mesh));
        self.skeletons.retain(|id, _| live.contains(id));
        self.clips.retain(|id, _| live.contains(id));
    }

    /// Forget one mesh asset's opened payloads (a targeted re-import invalidation).
    pub fn invalidate(&mut self, mesh_id: Uuid) {
        self.vgeom.remove(&mesh_id);
        self.skinned.retain(|(mesh, _), _| *mesh != mesh_id);
        self.rescanned_for.clear();
    }

    fn drop_loaded(&mut self) {
        self.vgeom.clear();
        self.skinned.clear();
        self.skeletons.clear();
        self.clips.clear();
    }

    // ── resolution ──────────────────────────────────────────────────────────

    /// Resolve a `MeshRef.asset` to its virtualized geometry.
    ///
    /// **MIRROR** of `inf_player::vmesh::VmeshRegistry::resolve`: the derived id is
    /// computed the same way, resolution is deliberately *independent of the render
    /// setting* (the tier decides whether the meshlet path or the classic
    /// discrete-LOD fallback draws the resolved content — the scene content is the
    /// same either way), and `None` means "no DAG ⇒ the primitive placeholder",
    /// which stays correct content for a primitive `MeshRef`.
    ///
    /// Opens the payload on first use and caches both hits and misses.
    pub fn resolve_vgeom(&mut self, mesh_id: Uuid) -> Option<LoadedVgeom> {
        if let Some(hit) = self.vgeom.get(&mesh_id) {
            return hit.clone();
        }
        let loaded = self.open_vgeom(mesh_id);
        self.vgeom.insert(mesh_id, loaded.clone());
        loaded
    }

    fn open_vgeom(&mut self, mesh_id: Uuid) -> Option<LoadedVgeom> {
        let vmesh_id = inf_vgeom::derived_vmesh_id(inf_asset::AssetId(mesh_id)).uuid();
        let path = self.resolve_path(vmesh_id)?;
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("inf-editor-core: .inf_vmesh {}: {e}", path.display());
                return None;
            }
        };
        // The render key is the payload's content hash (see the module docs), read
        // from the bytes actually loaded rather than from a sidecar that could be
        // stale relative to them.
        let id = inf_asset::ContentHash::of(&bytes).0;
        match VgeomSource::from_payload(bytes) {
            Ok(source) => Some(LoadedVgeom {
                id,
                source: Arc::new(source),
            }),
            Err(e) => {
                tracing::warn!("inf-editor-core: bad .inf_vmesh {}: {e}", path.display());
                None
            }
        }
    }

    /// Resolve a [`SkeletalMesh`] (+ its optional [`AnimPlayer`], + the pose the
    /// sim evaluated for this entity) to a drawable skinned instance: bind-space
    /// geometry plus the skinning palette for the pose that is actually in force.
    ///
    /// The pose rule, in order:
    ///  1. no skeleton asset, or a skeleton with no joints ⇒ `None` (the caller
    ///     keeps the placeholder — a skinned mesh with no skeleton is not a
    ///     renderable thing);
    ///  2. a `posed` entry from the sim ([`inf_ecs::pose`]) whose skeleton is the
    ///     one bound here and whose joint count matches ⇒ **that pose**, because
    ///     an `AnimStateMachine` beats an `AnimPlayer` and this is where "the
    ///     machine wins" stops being a comment;
    ///  3. no `AnimPlayer`, or one with no clip, or a clip that will not resolve ⇒
    ///     **rest pose** ([`inf_anim::Pose::rest`]);
    ///  4. otherwise the clip sampled at the play-head, honouring `looping`.
    ///
    /// Rule 2's two guards are not defensive noise. The projector reads the pose
    /// out of a sim-side store keyed by entity and resolves the skeleton out of
    /// its own asset store: if a character were re-bound to a different rig
    /// between the fixed step and the projection, applying the old pose would
    /// deform it by another skeleton's hierarchy — silently, and only on the
    /// frames where it happened. A mismatch falls through to rules 3/4, which is
    /// always a pose the bound skeleton can wear.
    ///
    /// Rule 3 is what makes a freshly dropped character *visible* rather than
    /// invisible-until-you-press-play, and it is deliberately the same fallback
    /// the components document (`AnimPlayer::clip: None → the bind pose`).
    pub fn resolve_skinned(
        &mut self,
        sm: &SkeletalMesh,
        player: Option<&AnimPlayer>,
        posed: Option<&EvaluatedPose>,
    ) -> Option<SkinnedDraw> {
        let mesh_id = sm.mesh?;
        let skeleton_id = sm.skeleton?;
        let skeleton = self.skeleton(skeleton_id)?;
        if skeleton.is_empty() {
            return None;
        }
        let mesh = self.skinned_geometry(mesh_id, skeleton_id)?;
        let from_sim =
            posed.filter(|p| p.skeleton == skeleton_id && p.pose.len() == skeleton.len());
        let pose = match (from_sim, player.and_then(|p| p.clip.map(|c| (p, c)))) {
            (Some(p), _) => p.pose.clone(),
            (None, Some((p, clip_id))) => match self.clip(clip_id) {
                Some(clip) => inf_anim::sample_clip(&skeleton, &clip, p.t as f32, p.looping),
                None => inf_anim::Pose::rest(&skeleton),
            },
            (None, None) => inf_anim::Pose::rest(&skeleton),
        };
        Some(SkinnedDraw {
            mesh,
            palette: inf_anim::skinning_matrices(&skeleton, &pose),
            key: (mesh_id, skeleton_id),
        })
    }

    /// A decoded skeleton, cached (hit or miss) by GUID.
    pub fn skeleton(&mut self, id: Uuid) -> Option<Arc<Skeleton>> {
        if let Some(hit) = self.skeletons.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<inf_anim::SkeletonAsset>(id)
            .map(|a| Arc::new(a.skeleton));
        self.skeletons.insert(id, loaded.clone());
        loaded
    }

    /// A decoded animation clip, cached (hit or miss) by GUID.
    pub fn clip(&mut self, id: Uuid) -> Option<Arc<AnimClip>> {
        if let Some(hit) = self.clips.get(&id) {
            return hit.clone();
        }
        let loaded = self
            .load_payload::<inf_anim::AnimClipAsset>(id)
            .map(|a| Arc::new(a.clip));
        self.clips.insert(id, loaded.clone());
        loaded
    }

    /// Bind-space skinned geometry for a `(mesh, skeleton)` pair, cached.
    fn skinned_geometry(
        &mut self,
        mesh_id: Uuid,
        skeleton_id: Uuid,
    ) -> Option<Arc<SkinnedMeshData>> {
        let key = (mesh_id, skeleton_id);
        if let Some(hit) = self.skinned.get(&key) {
            return hit.clone();
        }
        let built = self
            .load_payload::<MeshAsset>(mesh_id)
            .and_then(|m| skinned_mesh_data(&m))
            .map(Arc::new);
        self.skinned.insert(key, built.clone());
        built
    }

    fn load_payload<T: inf_asset::AssetPayload>(&mut self, id: Uuid) -> Option<T> {
        let path = self.resolve_path(id)?;
        let bytes = std::fs::read(&path).ok()?;
        match inf_asset::decode::<T>(&bytes) {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!("inf-editor-core: decode {}: {e}", path.display());
                None
            }
        }
    }

    /// The loose file for an asset GUID, rescanning **once** on a miss.
    ///
    /// The index is a snapshot taken when the root was set, so an asset written
    /// after it — a fresh import — would otherwise never be found. A GUID that is
    /// genuinely absent is remembered in `rescanned_for`, so a dangling reference
    /// costs one directory walk for the session rather than one per frame.
    fn resolve_path(&mut self, id: Uuid) -> Option<PathBuf> {
        if let Some(p) = self.index.get(&id) {
            return Some(p.clone());
        }
        let root = self.content_root.clone()?;
        if !self.rescanned_for.insert(id) {
            return None;
        }
        self.index = content_paths_by_guid(&root);
        self.index.get(&id).cloned()
    }
}

/// Build bind-space [`SkinnedMeshData`] from a `.inf_mesh` that carries skinning
/// influences, or `None` if none of its submeshes do.
///
/// Submeshes are concatenated into one vertex + index buffer (indices rebased),
/// exactly like the thumbnailer's `combined_geometry` — the skinned pass draws one
/// instance per entity, and per-submesh material slots are the same follow-up
/// they are on the rigid path. An **unskinned** submesh inside a skinned mesh is
/// kept and pinned to joint 0 with weight 1, which is what a rigid part welded to
/// a skeleton's root means; dropping it would silently lose geometry.
///
/// **MIRROR** of the other host's `skinned_mesh_data` — keep the two
/// byte-identical, **this doc block included** (`projector_mirror.rs`). It has to
/// be: the two hosts would otherwise upload *different vertex buffers* for the
/// same asset, which no scene-level comparison can see. Side-neutral wording on
/// purpose — a note that names the *other* file is a note only one copy can carry,
/// and a comment that exists on one side only is the drift this gate is for.
pub fn skinned_mesh_data(mesh: &MeshAsset) -> Option<SkinnedMeshData> {
    if !mesh.submeshes.iter().any(|s| s.is_skinned()) {
        return None;
    }
    let mut vertices: Vec<SkinnedVertex> = Vec::with_capacity(mesh.vertex_count());
    let mut indices: Vec<u32> = Vec::new();
    for sm in &mesh.submeshes {
        let base = vertices.len() as u32;
        for (i, v) in sm.vertices.iter().enumerate() {
            let skin = sm.skin.get(i).copied().unwrap_or_default().normalized();
            vertices.push(SkinnedVertex {
                pos: v.position,
                normal: v.normal,
                joints: skin.joints.map(u32::from),
                weights: skin.weights,
            });
        }
        indices.extend(sm.indices.iter().map(|&i| i + base));
    }
    if vertices.is_empty() || indices.len() < 3 {
        return None;
    }
    Some(SkinnedMeshData { vertices, indices })
}

/// Index every loose render asset under `dir` by its GUID, read from the sibling
/// `inf_asset` `.toml` sidecar.
///
/// Recursive over a whole project **content root** (imports land in subfolders),
/// deterministic (each directory's entries are path-sorted before descending), and
/// the editor's own dot-directories (`.inf` import cache / thumbnails,
/// `.infinity` settings) are not walked — the same rules, and the same reasons, as
/// `terrain_stream::terrain_paths_by_guid`.
pub fn content_paths_by_guid(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut files = Vec::new();
    collect_files(dir, 0, &mut files);
    let mut out = HashMap::new();
    for p in files {
        if let Ok(side) = inf_asset::AssetSidecar::load(&p) {
            out.insert(side.guid.uuid(), p);
        }
    }
    out
}

fn collect_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > MAX_CONTENT_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let hidden = path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect_files(&path, depth + 1, out);
            }
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| INDEXED_EXTENSIONS.contains(&e))
        {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{vmesh, AssetProject};
    use inf_anim::{AnimClip, Joint, JointTrack, JointTransform, QuatTrack};
    use inf_asset::AssetId;
    use inf_mesh::{MeshVertex, SubMesh, VertexSkin};

    /// A grid dense enough for a real multi-meshlet DAG, cheap enough for CI.
    fn grid_mesh(n: u32) -> MeshAsset {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(MeshVertex {
                    position: [i as f32, ((i * j) % 5) as f32 * 0.25, j as f32],
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

    /// A 3-vertex triangle skinned to a 2-joint chain.
    fn skinned_mesh() -> MeshAsset {
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        MeshAsset::new(
            vec![SubMesh {
                name: "tri".into(),
                vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 2.0)],
                indices: vec![0, 1, 2],
                material_slot: None,
                skin: vec![
                    VertexSkin {
                        joints: [0, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    VertexSkin {
                        joints: [0, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    VertexSkin {
                        joints: [1, 0, 0, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                ],
            }],
            vec![],
        )
    }

    fn two_joint_skeleton() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                local_bind: JointTransform::default(),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
            },
            Joint {
                name: "tip".into(),
                parent: Some(0),
                local_bind: JointTransform::from_trs(
                    glam::Vec3::new(0.0, 1.0, 0.0),
                    glam::Quat::IDENTITY,
                    glam::Vec3::ONE,
                ),
                inverse_bind: Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0))
                    .to_cols_array(),
            },
        ])
        .unwrap()
    }

    /// A one-second clip rotating the tip joint 90° about X.
    fn wave_clip() -> AnimClip {
        AnimClip {
            name: "wave".into(),
            duration: 1.0,
            tracks: vec![JointTrack {
                joint: 1,
                translation: None,
                rotation: Some(QuatTrack::new(
                    vec![0.0, 1.0],
                    vec![
                        glam::Quat::IDENTITY.to_array(),
                        glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2).to_array(),
                    ],
                    inf_anim::Interpolation::Linear,
                )),
                scale: None,
            }],
        }
    }

    /// A project with one derived mesh; returns `(dir, root, mesh guid)`.
    fn project_with_derived_mesh(n: u32) -> (tempfile::TempDir, PathBuf, Uuid) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(n), None, vec![], None)
            .unwrap();
        assert!(vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap().rebuilt());
        (dir, root, mesh_id.uuid())
    }

    /// **The headline gate**: an imported mesh, derived and indexed, resolves to
    /// real streamable geometry — which is precisely what the editor viewport could
    /// not do before P18.3.
    #[test]
    fn an_imported_mesh_resolves_to_real_geometry() {
        let (_dir, root, mesh_id) = project_with_derived_mesh(12);
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));

        let loaded = store.resolve_vgeom(mesh_id).expect("real geometry");
        assert!(loaded.source.meshlet_count() > 0, "the DAG has meshlets");
        assert!(!loaded.source.pages().is_empty(), "and a page directory");
        assert!(
            loaded.source.bounds().1 > 0.0,
            "with a real bounding sphere"
        );
        assert_eq!(store.loaded_vgeom(), 1);
    }

    /// Resolution is a pure function of content: two stores over the same root
    /// agree on the id and on everything the renderer reads from the source.
    #[test]
    fn resolution_is_deterministic_across_stores() {
        let (_dir, root, mesh_id) = project_with_derived_mesh(10);
        let mut a = EditorRenderAssets::new();
        let mut b = EditorRenderAssets::new();
        a.set_content_root(Some(root.clone()));
        b.set_content_root(Some(root));
        let (x, y) = (
            a.resolve_vgeom(mesh_id).unwrap(),
            b.resolve_vgeom(mesh_id).unwrap(),
        );
        assert_eq!(x.id, y.id, "the content key is reproducible");
        assert_eq!(x.source.meshlet_count(), y.source.meshlet_count());
        assert_eq!(x.source.pages(), y.source.pages());
    }

    /// An unresolvable mesh (no DAG on disk) is a clean `None` — the caller keeps
    /// its primitive placeholder — and it costs exactly **one** directory walk,
    /// not one per frame.
    #[test]
    fn a_dangling_mesh_reference_misses_cheaply() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(dir.path().to_path_buf()));
        let ghost = Uuid::from_u128(0xDEAD_BEEF);
        for _ in 0..5 {
            assert!(store.resolve_vgeom(ghost).is_none());
        }
        assert_eq!(store.loaded_vgeom(), 0);
    }

    /// With no content root the store is inert — which is what makes a viewport
    /// with no project open behave exactly as it did before this batch.
    #[test]
    fn no_content_root_means_no_resolution() {
        let mut store = EditorRenderAssets::new();
        assert!(store.resolve_vgeom(Uuid::from_u128(1)).is_none());
        assert_eq!(store.index_len(), 0);
    }

    /// **Re-import invalidation.** Rewriting the mesh under the same GUID must
    /// change the render key, because both renderer nodes cache GPU state by it —
    /// a stable key across changed bytes is a permanently stale draw.
    #[test]
    fn a_reimport_changes_the_content_key() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(8), None, vec![], None)
            .unwrap();
        vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();

        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let before = store.resolve_vgeom(mesh_id.uuid()).unwrap();

        // Re-import (same GUID, new geometry) + re-derive.
        proj.rewrite_payload(mesh_id, &grid_mesh(14), vec![])
            .unwrap();
        assert!(vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap().rebuilt());

        // Without the refresh the store still serves the old mapping — which is
        // exactly why Ring 2 pushes `refresh_index` on `assets://changed`.
        assert_eq!(store.resolve_vgeom(mesh_id.uuid()).unwrap().id, before.id);
        store.refresh_index();
        let after = store.resolve_vgeom(mesh_id.uuid()).unwrap();
        assert_ne!(after.id, before.id, "the render key must follow the bytes");
        assert_ne!(after.source.meshlet_count(), before.source.meshlet_count());
    }

    /// Deleting the mesh asset degrades the entity to its placeholder rather than
    /// rendering stale geometry or panicking.
    #[test]
    fn deleting_a_mesh_degrades_to_the_placeholder() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Meshes").unwrap();
        let mesh_id = proj
            .write_asset(&d, "Grid", &grid_mesh(8), None, vec![], None)
            .unwrap();
        vmesh::ensure_vmesh(&mut proj, mesh_id).unwrap();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        assert!(store.resolve_vgeom(mesh_id.uuid()).is_some());

        assert!(proj.delete(mesh_id, false).unwrap().is_empty());
        store.refresh_index();
        assert!(
            store.resolve_vgeom(mesh_id.uuid()).is_none(),
            "a deleted mesh must stop resolving — no stale render"
        );
    }

    /// **Level switch releases streams** (the P16.4b lesson, tested rather than
    /// asserted): the per-projection audit drops what the new document does not
    /// reference, and `clear` drops everything.
    #[test]
    fn a_level_switch_releases_every_opened_payload() {
        let (_dir, root, mesh_id) = project_with_derived_mesh(9);
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        store.resolve_vgeom(mesh_id).unwrap();
        assert_eq!(store.loaded_vgeom(), 1);

        // A projection that no longer references it releases it…
        store.retain_only(std::iter::empty());
        assert_eq!(store.loaded_vgeom(), 0, "retain_only released the mapping");

        // …and a projection that does keeps it.
        store.resolve_vgeom(mesh_id).unwrap();
        store.retain_only([mesh_id]);
        assert_eq!(store.loaded_vgeom(), 1);

        // The document-replaced door drops everything, index intact.
        store.clear();
        assert_eq!(store.loaded_vgeom(), 0);
        assert!(store.index_len() > 0, "the index survives a level switch");
    }

    // ── skeletal ────────────────────────────────────────────────────────────

    fn project_with_character() -> (tempfile::TempDir, PathBuf, Uuid, Uuid, Uuid) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let mut proj = AssetProject::open(&root).unwrap();
        let d = proj.content_dir("Character").unwrap();
        let mesh = proj
            .write_asset(&d, "Body", &skinned_mesh(), None, vec![], None)
            .unwrap();
        let skel = proj
            .write_asset(
                &d,
                "Rig",
                &inf_anim::SkeletonAsset::new(two_joint_skeleton()),
                None,
                vec![],
                None,
            )
            .unwrap();
        let clip = proj
            .write_asset(
                &d,
                "Wave",
                &inf_anim::AnimClipAsset::new(wave_clip(), Some(*skel.uuid().as_bytes())),
                None,
                vec![],
                None,
            )
            .unwrap();
        (dir, root, mesh.uuid(), skel.uuid(), clip.uuid())
    }

    /// **SkeletalMesh rest-pose projection**: a character with no `AnimPlayer`
    /// resolves to real bind-space geometry and an identity-ish palette, so it is
    /// visible the moment it is placed.
    #[test]
    fn a_skeletal_mesh_projects_its_rest_pose() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));

        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let draw = store
            .resolve_skinned(&sm, None, None)
            .expect("rest pose draws");
        assert_eq!(draw.mesh.vertices.len(), 3);
        assert_eq!(draw.mesh.indices, vec![0, 1, 2]);
        assert_eq!(draw.palette.len(), 2, "one matrix per joint");
        // Rest pose: `global · inverse_bind` is the identity for a bind-pose
        // skeleton, which is what makes the drawn mesh its authored shape.
        for m in &draw.palette {
            assert!(
                (*m - Mat4::IDENTITY)
                    .to_cols_array()
                    .iter()
                    .all(|v| v.abs() < 1e-5),
                "rest palette must be identity, got {m:?}"
            );
        }
        // The influences survived the bind-space rebuild.
        assert_eq!(draw.mesh.vertices[2].joints[0], 1);
        assert_eq!(draw.key, (mesh, skel));
    }

    /// With an `AnimPlayer` the palette follows the play-head — the same clip
    /// sampling the runtime does, so the viewport shows the pose the game would.
    #[test]
    fn an_anim_player_drives_the_palette() {
        let (_dir, root, mesh, skel, clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };

        let rest = store.resolve_skinned(&sm, None, None).unwrap().palette;
        let posed = store
            .resolve_skinned(
                &sm,
                Some(&AnimPlayer {
                    clip: Some(clip),
                    // Mid-clip on purpose: `t == duration` on a LOOPING player
                    // wraps back to 0, which would compare the rest pose to
                    // itself and pass for the wrong reason.
                    t: 0.5,
                    ..AnimPlayer::default()
                }),
                None,
            )
            .unwrap()
            .palette;
        assert_ne!(rest[1].to_cols_array(), posed[1].to_cols_array());
        // …and it is deterministic: the same play-head yields the same palette.
        let again = store
            .resolve_skinned(
                &sm,
                Some(&AnimPlayer {
                    clip: Some(clip),
                    // Mid-clip on purpose: `t == duration` on a LOOPING player
                    // wraps back to 0, which would compare the rest pose to
                    // itself and pass for the wrong reason.
                    t: 0.5,
                    ..AnimPlayer::default()
                }),
                None,
            )
            .unwrap()
            .palette;
        assert_eq!(posed[1].to_cols_array(), again[1].to_cols_array());
    }

    /// An `AnimPlayer` whose clip cannot be resolved falls back to the rest pose
    /// rather than to nothing — a missing clip must not make a character vanish.
    #[test]
    fn an_unresolvable_clip_falls_back_to_rest() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let rest = store.resolve_skinned(&sm, None, None).unwrap().palette;
        let ghost = store
            .resolve_skinned(
                &sm,
                Some(&AnimPlayer {
                    clip: Some(Uuid::from_u128(0xC0FFEE)),
                    t: 0.5,
                    ..AnimPlayer::default()
                }),
                None,
            )
            .unwrap()
            .palette;
        assert_eq!(rest[1].to_cols_array(), ghost[1].to_cols_array());
    }

    /// Half-bound skeletal entities keep the placeholder (no skeleton ⇒ nothing to
    /// deform with), and an unskinned mesh is not a skinned draw.
    #[test]
    fn an_unbound_skeletal_mesh_stays_a_placeholder() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));

        assert!(store
            .resolve_skinned(&SkeletalMesh::default(), None, None)
            .is_none());
        assert!(store
            .resolve_skinned(
                &SkeletalMesh {
                    mesh: Some(mesh),
                    skeleton: None
                },
                None,
                None,
            )
            .is_none());
        assert!(store
            .resolve_skinned(
                &SkeletalMesh {
                    mesh: Some(Uuid::from_u128(7)),
                    skeleton: Some(skel)
                },
                None,
                None,
            )
            .is_none());
        // A rigid mesh with no skin stream is not skinned geometry.
        assert!(skinned_mesh_data(&grid_mesh(4)).is_none());
    }

    /// The GUID index skips the editor's own dot-directories, so a `.inf_vmesh`
    /// parked in the import cache can never be served as content.
    #[test]
    fn the_index_skips_dot_directories() {
        let dir = tempfile::tempdir().unwrap();
        let hidden = dir.path().join(".inf").join("import-cache");
        std::fs::create_dir_all(&hidden).unwrap();
        let payload = hidden.join("Stray.inf_vmesh");
        std::fs::write(&payload, b"not a real payload").unwrap();
        let side = inf_asset::AssetSidecar::new(
            AssetId::new(),
            inf_asset::AssetKind::MeshletMesh,
            inf_asset::ContentHash::of(b"not a real payload"),
        );
        side.save(&payload).unwrap();

        assert!(content_paths_by_guid(dir.path()).is_empty());
    }
    /// An [`EvaluatedPose`] on `skeleton` with the tip joint bent 30° about X —
    /// what a state machine in a non-entry state publishes.
    fn bent_pose(skeleton: Uuid) -> EvaluatedPose {
        let sk = two_joint_skeleton();
        let mut pose = inf_anim::Pose::rest(&sk);
        pose.locals[1].rotation = glam::Quat::from_rotation_x(30f32.to_radians()).to_array();
        EvaluatedPose {
            skeleton,
            pose,
            sockets: Vec::new(),
        }
    }

    /// A pose with the WRONG joint count whose acceptance would be **visible**.
    ///
    /// The load-bearing detail is which joint is bent (P24.1 audit M4). The first
    /// cut bent the tip and then truncated the tip away, so the short pose
    /// evaluated byte-identical to rest — `global_transforms` back-fills a missing
    /// joint from its bind transform — and the assertion below passed with or
    /// without the length guard it exists to prove. Bending the **root**, which
    /// survives the truncation, makes an accepted short pose produce a different
    /// palette from rest, so deleting the guard fails the test.
    fn short_pose_a_guardless_store_would_wear(skeleton: Uuid) -> EvaluatedPose {
        let sk = two_joint_skeleton();
        let mut pose = inf_anim::Pose::rest(&sk);
        pose.locals[0].rotation = glam::Quat::from_rotation_x(40f32.to_radians()).to_array();
        pose.locals.truncate(1);
        EvaluatedPose {
            skeleton,
            pose,
            sockets: Vec::new(),
        }
    }

    /// **The P24.1 headline gate, store half**: when the sim published a pose for
    /// this entity, THAT is what the palette is built from — the `AnimPlayer` is
    /// overruled, because an `AnimStateMachine` beats a clip play-head and this is
    /// the only place that fact reaches the renderer.
    #[test]
    fn a_sim_evaluated_pose_beats_the_anim_player() {
        let (_dir, root, mesh, skel, clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let play = AnimPlayer {
            clip: Some(clip),
            t: 0.5,
            ..AnimPlayer::default()
        };

        let rest = store.resolve_skinned(&sm, None, None).unwrap().palette;
        let from_clip = store
            .resolve_skinned(&sm, Some(&play), None)
            .unwrap()
            .palette;
        let posed = bent_pose(skel);
        let from_sim = store
            .resolve_skinned(&sm, Some(&play), Some(&posed))
            .unwrap()
            .palette;

        // All three are distinct: the machine's pose is neither the rest pose nor
        // the clip's. (Comparing only against rest would pass for a store that
        // simply kept sampling the clip.)
        assert_ne!(from_sim[1].to_cols_array(), rest[1].to_cols_array());
        assert_ne!(from_sim[1].to_cols_array(), from_clip[1].to_cols_array());
        // …and with no player at all the sim pose still wins over rest.
        let alone = store
            .resolve_skinned(&sm, None, Some(&posed))
            .unwrap()
            .palette;
        assert_eq!(alone[1].to_cols_array(), from_sim[1].to_cols_array());
    }

    /// Rule 2's guards: a pose evaluated against a **different skeleton**, or one
    /// whose joint count does not match, is refused and the store falls through to
    /// the `AnimPlayer` / rest arms. Applying it would deform a character by
    /// another rig's hierarchy — silently, and only on the frames it happened.
    #[test]
    fn a_pose_from_another_skeleton_is_refused() {
        let (_dir, root, mesh, skel, _clip) = project_with_character();
        let mut store = EditorRenderAssets::new();
        store.set_content_root(Some(root));
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        let rest = store.resolve_skinned(&sm, None, None).unwrap().palette;

        let mut wrong_rig = bent_pose(skel);
        wrong_rig.skeleton = Uuid::from_u128(0xBAD_5CE1);
        assert_eq!(
            store
                .resolve_skinned(&sm, None, Some(&wrong_rig))
                .unwrap()
                .palette[1]
                .to_cols_array(),
            rest[1].to_cols_array(),
            "a pose from another rig must not be worn"
        );

        let wrong_len = short_pose_a_guardless_store_would_wear(skel);
        let drawn = store
            .resolve_skinned(&sm, None, Some(&wrong_len))
            .unwrap()
            .palette;
        // Both joints, because a short pose that IS worn moves both: the bent root
        // is joint 0's palette outright and rides through joint 1's global.
        assert_eq!(
            drawn[0].to_cols_array(),
            rest[0].to_cols_array(),
            "a pose with the wrong joint count must not be worn"
        );
        assert_eq!(
            drawn[1].to_cols_array(),
            rest[1].to_cols_array(),
            "a pose with the wrong joint count must not be worn"
        );
        // ANTI-VACUITY: the refused pose really would have changed the palette —
        // otherwise the two assertions above hold whether the guard exists or not,
        // which is exactly the hole the audit found.
        let worn = inf_anim::skinning_matrices(&two_joint_skeleton(), &wrong_len.pose);
        assert_ne!(
            worn[0].to_cols_array(),
            rest[0].to_cols_array(),
            "the short pose evaluates to the rest palette, so refusing it is \
             indistinguishable from accepting it"
        );
    }
}
