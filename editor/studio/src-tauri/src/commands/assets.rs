//! Asset / Content Drawer commands (Phase 4).
//!
//! The [`AssetProject`] (asset DB + import cache over the project content root)
//! lives here behind an `Arc<Mutex<…>>` shared with the background import
//! worker. Commands mutate it and emit `assets://changed` (a version bump the
//! frontend re-fetches on); a background tick drains import-job progress
//! (`assets://import`) and the file watcher (external edits → rescan → changed).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use inf_asset::{AssetId, AssetKind, AssetWatcher};
use inf_editor_core::assets::{
    biome_set, data, material_instance, queue::tick as asset_tick, snapshot, sprite_sheet,
    AssetProject, ImportProgress, ImportQueue,
};
use inf_editor_core::ipc::{
    AssetChanged, AssetRefDto, AssetSnapshot, BiomeDefDto, BiomeSetDto, DataAssetDto, DeleteResult,
    ImportEventDto, MatOverridesDto, MatValuesDto, MaterialInstanceDto, SpriteSheetDto,
};
use inf_editor_core::thumbnail::{ThumbnailCache, Thumbnailer};
use inf_material::MaterialAsset;
use inf_terrain::{BiomeDef, BiomeSet};
use tauri::{AppHandle, Emitter, Manager, State};

/// The debounce for the content watcher, and the background tick interval.
const WATCH_DEBOUNCE: Duration = Duration::from_millis(250);
const TICK: Duration = Duration::from_millis(120);

/// Lazily-initialized asset subsystem (set up in [`init_assets_on_boot`]).
#[derive(Default)]
pub struct AssetState {
    inner: Mutex<Option<AssetInner>>,
}

struct AssetInner {
    project: Arc<Mutex<AssetProject>>,
    queue: ImportQueue,
    watcher: Option<AssetWatcher>,
    /// The thumbnail renderer + its disk cache, behind their own mutex so a slow
    /// headless-GPU render + PNG/file IO serializes only against *other*
    /// thumbnail renders — never behind the whole asset state (the `inner` lock).
    thumbs: Arc<Mutex<ThumbnailRig>>,
}

/// Thumbnail renderer + disk cache, guarded together (see [`AssetInner::thumbs`]).
struct ThumbnailRig {
    thumb: Thumbnailer,
    cache: ThumbnailCache,
}

impl AssetState {
    /// The display name of an asset (for the drag-to-scene placeholder). Public
    /// so the scene command can name a dropped-asset entity.
    pub fn asset_name(&self, id: AssetId) -> Option<String> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        proj.db().get(id).map(|e| e.name.clone())
    }

    /// The [`AssetKind`] of an asset. Public for the same reason
    /// [`Self::asset_name`] is: the scene command has to know whether a dropped
    /// asset is a **mesh** before it can bind it to the entity it spawns.
    pub fn asset_kind(&self, id: AssetId) -> Option<AssetKind> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        proj.db().get(id).map(|e| e.kind())
    }

    /// Write a baked texture as a new `.inf_tex` asset under `Content/baked`
    /// (P7.3 material bake). Returns the new asset id; the file watcher then
    /// re-syncs the Content Drawer.
    ///
    /// P26.1: the payload is a **v2 tiled image**, which is why this takes a
    /// `TiledTextureImage` and goes through `write_tiled_texture` — the generic
    /// writer would frame the raw image with a bincode length prefix and knock
    /// every tile off its 16-byte boundary. The type makes that a compile error.
    pub fn write_texture_asset(
        &self,
        name: &str,
        tex: &inf_material::TiledTextureImage,
    ) -> Result<AssetId, String> {
        self.with_project(|proj| {
            let dir = proj.content_dir("baked").map_err(|e| e.to_string())?;
            proj.write_tiled_texture(&dir, name, tex, None, None)
                .map_err(|e| e.to_string())
        })
    }

    /// Load a material's resolved PBR parameters (Content-Drawer apply-by-drag,
    /// P7.1). A `MaterialInstance` (P7.4) is resolved against its parent chain.
    /// `None` if the asset is missing or not a material/instance.
    pub fn load_material(&self, id: AssetId) -> Option<MaterialAsset> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        resolve_material(&proj, id, 0)
    }

    /// The `.inf_mat` a scene `Material.asset` binding must name for the asset an
    /// author applied (P26.3b audit) — the instance chain walked to its root.
    ///
    /// Apply-by-drag accepts a `.inf_mati`, and a binding that named one resolved
    /// **nowhere**: the cook derives a `.inf_matd` and closes the texture edge for
    /// `AssetKind::Material` only, and the PIE loader is kind-checked to the same
    /// set, so the surface silently lost its maps on both wires with no advisory
    /// (the asset is in the project, so nothing looked dangling). The reasoning
    /// lives once, in
    /// [`inf_editor_core::assets::material_instance::material_binding_root`].
    pub fn material_binding_id(&self, id: AssetId) -> Option<AssetId> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        inf_editor_core::assets::material_instance::material_binding_root(&proj, id, 0)
    }

    /// `load_blueprint_class` with the REASON kept (Wave E).
    ///
    /// The `Option` twin below exists for Simulate and the PIE payload builder,
    /// where "no blueprint bound" and "the file will not parse" both mean *run
    /// without gameplay logic*. A user who asked to OPEN this blueprint needs
    /// the other answer: which asset, and what is wrong with it.
    pub fn load_blueprint_class_result(
        &self,
        id: AssetId,
    ) -> Result<inf_blueprint::BlueprintClass, String> {
        let guard = self.inner.lock().map_err(|_| "asset state poisoned")?;
        let inner = guard.as_ref().ok_or("assets not initialized")?;
        let proj = inner.project.lock().map_err(|_| "asset project poisoned")?;
        let entry = proj.db().get(id).ok_or_else(|| format!("no asset {id}"))?;
        let kind = entry.kind();
        if !matches!(
            kind,
            inf_asset::AssetKind::Blueprint | inf_asset::AssetKind::Script
        ) {
            return Err(format!("{} is not a blueprint or a script", entry.name));
        }
        let path = entry.path.clone();
        let name = entry.name.clone();
        // SCRIPT1b: an InfiniScript arrives through `inf_script::source` — the
        // ONE file door the cook and the PIE payload builder also use, so the
        // editor and the build cannot disagree about what a script means.
        if kind == inf_asset::AssetKind::Script {
            return inf_script::compile_path(&path, format!("script:{id}"))
                .map(|(class, _warnings)| class)
                .map_err(|d| inf_script::render(&d));
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("read {name}: {e}"))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("{name} will not parse: {e}"))
    }

    /// Load a `.inf_act` blueprint **class** by its asset GUID (P9.5): decodes
    /// the committed JSON payload. `None` if the asset is missing or not a
    /// blueprint. Used by Simulate to resolve a scene's persisted `ActorClass`
    /// bindings to runnable classes.
    /// **A blueprint that is present but unreadable is logged, not silently
    /// dropped** (C4-42).
    ///
    /// The DB entry was already found, so a failure past that point is a real
    /// one — a corrupt, locked or half-written `.inf_act` — and both consumers
    /// (`bound_actors` in Simulate, and the PIE payload builder) read `None` as
    /// **"no blueprint bound"**. The actor then runs with no gameplay logic at
    /// all, and nothing anywhere says why.
    pub fn load_blueprint_class(&self, id: AssetId) -> Option<inf_blueprint::BlueprintClass> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        let kind = entry.kind();
        if !matches!(
            kind,
            inf_asset::AssetKind::Blueprint | inf_asset::AssetKind::Script
        ) {
            return None;
        }
        let path = entry.path.clone();
        // SCRIPT1b: an InfiniScript compiles through the one file door. A script
        // that will not parse is logged with its LINE, which a JSON decode
        // error cannot give — and the actor runs with no gameplay logic, exactly
        // as an unreadable `.inf_act` does.
        if kind == inf_asset::AssetKind::Script {
            return match inf_script::compile_path(&path, format!("script:{id}")) {
                Ok((class, warnings)) => {
                    for w in warnings {
                        tracing::warn!(asset = %id, "{w}");
                    }
                    Some(class)
                }
                Err(diags) => {
                    tracing::error!(
                        asset = %id,
                        path = %path.display(),
                        "InfiniScript will not compile; this actor will run with NO \
                         gameplay logic:
                    {}",
                        inf_script::render(&diags)
                    );
                    None
                }
            };
        }
        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(class) => Some(class),
                Err(e) => {
                    tracing::error!(
                        asset = %id,
                        path = %path.display(),
                        "blueprint class will not parse ({e}); this actor will run with NO \
                         gameplay logic"
                    );
                    None
                }
            },
            Err(e) => {
                tracing::error!(
                    asset = %id,
                    path = %path.display(),
                    "blueprint class could not be read ({e}); this actor will run with NO \
                     gameplay logic"
                );
                None
            }
        }
    }

    /// Load a `.inf_pcg` graph asset's **raw bytes** by its asset GUID (P10.6):
    /// the bytes streamed to the PIE player so it evaluates a scene's
    /// [`PcgVolume`](inf_ecs::components::PcgVolume) scatter exactly like the
    /// shipping pack path. `None` if the asset is missing or not a PCG graph.
    pub fn load_pcg_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        self.load_asset_bytes(id, AssetKind::Pcg)
    }

    /// Load an asset's **raw payload bytes** by GUID, refusing anything that is
    /// not of `kind`.
    ///
    /// The shape [`load_pcg_bytes`](Self::load_pcg_bytes) and friends were all
    /// written by hand; stated once here so a new kind is one line rather than a
    /// fifth copy of the same lock-walk + kind guard.
    pub fn load_asset_bytes(&self, id: AssetId, kind: AssetKind) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != kind {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Load a `.inf_biomes` [`BiomeSet`] asset's **raw bytes** by its asset GUID
    /// (P19.3): the bytes the biome→PCG evaluate command decodes, and the ones
    /// streamed to the PIE player so it runs the same binding as the shipping
    /// pack path. `None` if the asset is missing or is not a biome set.
    pub fn load_biome_set_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        self.load_asset_bytes(id, AssetKind::BiomeSet)
    }

    /// Load an `.inf_tex` asset's **raw bytes** by its asset GUID — the pixels a
    /// PCG `mask.image` node names (P19.3). `None` if the asset is missing or is
    /// not a texture.
    pub fn load_texture_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        self.load_asset_bytes(id, AssetKind::Texture)
    }

    /// Load a P11 animation asset's **raw bytes** by its asset GUID (P11.4): the
    /// `.inf_skel` / `.inf_anim` / `.inf_sm` bytes streamed to the PIE player so it
    /// resolves state machines + root-motion clips exactly like the shipping pack
    /// path. `None` if the asset is missing or not an animation asset.
    pub fn load_anim_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if !matches!(
            entry.kind(),
            inf_asset::AssetKind::Skeleton
                | inf_asset::AssetKind::AnimClip
                | inf_asset::AssetKind::StateMachine
        ) {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Load an `.inf_audio` asset's **raw payload bytes** by its GUID (P12.3), so
    /// the Simulate wiring can decode + seed it into the [`SimSession`]'s audio
    /// engine. `None` if the asset is missing or not an audio asset.
    pub fn load_audio_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != inf_asset::AssetKind::Audio {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Raw payload bytes of a `.inf_cloth` asset (P24.4), for seeding a Simulate
    /// session's garments. Kind-checked exactly like
    /// [`load_audio_bytes`](Self::load_audio_bytes), for the same reason: a
    /// mistyped reference must miss rather than feed arbitrary bytes to the cloth
    /// decoder.
    pub fn load_cloth_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != inf_asset::AssetKind::Cloth {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Raw payload bytes of a `.inf_hair` asset (P24.4). Kind-checked exactly
    /// like [`load_cloth_bytes`](Self::load_cloth_bytes).
    pub fn load_hair_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != inf_asset::AssetKind::Hair {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Raw payload bytes of a `.inf_voxel` asset (P21.2), for seeding a Simulate
    /// session's voxel volumes. `None` when the id is unknown or names another
    /// kind — the kind check is what stops a mistyped reference feeding arbitrary
    /// bytes to the voxel parser. Mirrors [`load_audio_bytes`](Self::load_audio_bytes).
    pub fn load_voxel_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != inf_asset::AssetKind::VoxelVolume {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Raw payload bytes of a `.inf_mesh` asset (P22.3), for **deriving** a
    /// destructible actor's `.inf_fracture`.
    ///
    /// The odd one out among these loaders: every other one hands back the bytes
    /// of the asset a component names, but `Destructible` names no asset at all
    /// (the strength memo's §5 — a reference would be a second authority for the
    /// same fact). The fracture is derived from the actor's own `MeshRef.asset`,
    /// by the same `inf_mesh::fracture_mesh` the cook runs, so what the caller
    /// needs from here is the MESH. Kind-checked exactly like
    /// [`load_voxel_bytes`](Self::load_voxel_bytes), for the same reason.
    pub fn load_mesh_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != inf_asset::AssetKind::Mesh {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// Raw payload bytes of the four kinds `ScenePayload` v8 carries (P26.3b):
    /// `.inf_cloth`, `.inf_hair`, `.inf_mat` and `.inf_tex`.
    ///
    /// One loader rather than four because all four are the same act — read this
    /// asset's committed bytes — and one kind check rather than none because the
    /// alternative is shipping a garment's bytes in the texture slot when a
    /// binding is mistyped. A ref of the wrong kind resolves to `None`, which is
    /// the same outcome as a dangling one: the surface renders off its scalars,
    /// the character wears nothing, and the cook's advisory is where the author
    /// is told.
    pub fn load_binding_bytes(&self, id: AssetId) -> Option<Vec<u8>> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if !matches!(
            entry.kind(),
            inf_asset::AssetKind::Cloth
                | inf_asset::AssetKind::Hair
                | inf_asset::AssetKind::Material
                | inf_asset::AssetKind::Texture
        ) {
            return None;
        }
        std::fs::read(&entry.path).ok()
    }

    /// The on-disk **path** of a `.inf_terrain` asset (wave GTA1), for the PIE
    /// payload's `terrain_paths` route. Kind-checked exactly like
    /// [`load_voxel_bytes`](Self::load_voxel_bytes), for the same reason: a
    /// mistyped reference must miss rather than hand the tile reader a path to
    /// arbitrary bytes.
    ///
    /// It replaces `load_terrain_bytes` outright rather than sitting beside it:
    /// its one caller is the PIE payload builder, which now names the file, and
    /// a byte-reading twin nothing calls is a second way to do the thing that
    /// broke.
    ///
    /// This is what lets the island play: its terrain is 342 742 272 B, and a PIE
    /// frame is capped at 268 435 456, so the bytes route refuses the frame
    /// outright. The player is a child of this process on this machine and opens
    /// the same file through the same door a `--level` boot uses.
    pub fn terrain_path(&self, id: AssetId) -> Option<std::path::PathBuf> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        let entry = proj.db().get(id)?;
        if entry.kind() != inf_asset::AssetKind::Terrain {
            return None;
        }
        Some(entry.path.clone())
    }

    /// Create a new material instance of `parent` (P7.4). Returns the new id.
    pub fn create_material_instance(&self, parent: AssetId, name: &str) -> Result<AssetId, String> {
        self.with_project(|proj| {
            let inst = inf_material::MaterialInstance::new(parent);
            let dir = proj.content_dir("materials").map_err(|e| e.to_string())?;
            proj.write_asset(&dir, name, &inst, None, inst.dependencies(), None)
                .map_err(|e| e.to_string())
        })
    }

    /// Queue a chunked heightmap → `.inf_terrain` import on the shared import
    /// worker (P16.4a). Returns the job id; progress arrives on `assets://import`
    /// like every other import.
    ///
    /// `anchor` is the open level's geo-anchor (Wave G). It is passed in rather
    /// than read here because the asset state has no idea which level is open —
    /// and without it `use_georeference` is a recorded preference with nothing
    /// behind it.
    pub fn submit_terrain_import(
        &self,
        source: PathBuf,
        settings: inf_editor_core::assets::TerrainImportSettings,
        anchor: Option<inf_math::geo::GeoAnchor>,
        name: Option<String>,
    ) -> Result<u64, String> {
        let mut guard = self.inner.lock().map_err(|e| e.to_string())?;
        let inner = guard.as_mut().ok_or("assets not initialized")?;
        Ok(inner.queue.submit_terrain(source, settings, anchor, name))
    }

    /// Ask an in-flight cancellable import to stop. `false` for an unknown or
    /// already-finished job.
    pub fn cancel_import(&self, job: u64) -> Result<bool, String> {
        let guard = self.inner.lock().map_err(|e| e.to_string())?;
        let inner = guard.as_ref().ok_or("assets not initialized")?;
        Ok(inner.queue.cancel(job))
    }

    /// The current content-root directory (where per-project editor metadata —
    /// e.g. the sorting-layer registry — is stored). `None` before boot.
    pub fn content_root(&self) -> Option<PathBuf> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        Some(proj.root().to_path_buf())
    }

    /// Read a texture's sprite-sheet slice model + pixel dimensions (P8.2a).
    pub fn read_sprite_slices(
        &self,
        id: AssetId,
    ) -> Result<(sprite_sheet::SpriteSheetSlices, u32, u32), String> {
        self.with_project(|proj| sprite_sheet::read_slices(proj, id).map_err(|e| e.to_string()))
    }

    /// Persist a texture's sprite-sheet slice model into its sidecar (P8.2a).
    pub fn write_sprite_slices(
        &self,
        id: AssetId,
        slices: &sprite_sheet::SpriteSheetSlices,
    ) -> Result<(), String> {
        self.with_project(|proj| {
            sprite_sheet::write_slices(proj, id, slices).map_err(|e| e.to_string())
        })
    }

    // `pub(super)` so sibling command modules (e.g. the P11.2 state-machine
    // editor's `sm_list_clips`) can enumerate the asset db through the same guard.
    /// The shared project handle, for work that must run **off the async
    /// workers** (round-2, the sm/pcg MED).
    ///
    /// `with_project` borrows through `&State`, which is not `Send`, so a
    /// caller that needs `spawn_blocking` cannot use it. Handing out the `Arc`
    /// lets the closure own what it locks. Same lock, same order — this is a
    /// clone of a pointer, not a second door.
    pub(super) fn project_handle(&self) -> Result<Arc<Mutex<AssetProject>>, String> {
        let guard = self.inner.lock().map_err(|e| e.to_string())?;
        let inner = guard.as_ref().ok_or("assets not initialized")?;
        Ok(inner.project.clone())
    }

    pub(super) fn with_project<R>(
        &self,
        f: impl FnOnce(&mut AssetProject) -> Result<R, String>,
    ) -> Result<R, String> {
        let guard = self.inner.lock().map_err(|e| e.to_string())?;
        let inner = guard.as_ref().ok_or("assets not initialized")?;
        let mut proj = inner.project.lock().map_err(|e| e.to_string())?;
        f(&mut proj)
    }

    /// Rebuild the asset subsystem against a new content root (on project
    /// open/switch). The old import worker + watcher stop when their `AssetInner`
    /// is dropped; the shared background tick picks up the new one. Emits
    /// `assets://changed` so the Content Drawer re-syncs.
    pub fn reroot(&self, app: &AppHandle, content_root: PathBuf) {
        match build_inner(app, &content_root) {
            Some(inner) => {
                // Swap the new inner in and take the OLD one OUT within one short
                // critical section, then drop the old inner *after* releasing the
                // lock: dropping an `AssetInner` joins the import worker
                // (`ImportQueue::drop`), which can block for seconds on an in-flight
                // glTF import. Holding `self.inner` across that join would stall
                // every asset command; taking it out first keeps the lock hold
                // short and never exposes a `None` inner (`replace` is atomic under
                // the lock, so no command observes a torn swap).
                let old = match self.inner.lock() {
                    Ok(mut guard) => guard.replace(inner),
                    Err(_) => None,
                };
                drop(old); // joins the old worker with the lock released
                tracing::info!("asset system re-rooted to {}", content_root.display());
            }
            None => tracing::error!("asset re-root failed for {}", content_root.display()),
        }
        emit_changed(app, self);
    }
}

/// Build a fresh [`AssetInner`] rooted at `content_root`: open the project (seed
/// starter content if empty), spawn the import worker + file watcher, and open
/// the (content-hash-keyed, shared) thumbnail cache. `None` on failure.
fn build_inner(app: &AppHandle, content_root: &std::path::Path) -> Option<AssetInner> {
    let project = match AssetProject::open(content_root) {
        Ok(p) => Arc::new(Mutex::new(p)),
        Err(e) => {
            tracing::error!("asset project open failed: {e}");
            return None;
        }
    };
    seed_starter_content(&project, content_root);

    let mut queue = ImportQueue::spawn(project.clone());
    // P18.3: derive a `.inf_vmesh` for every mesh that lacks a current one, so a
    // project whose content was imported before the editor could build meshlet
    // DAGs shows real geometry rather than placeholder cubes. Queued on the import
    // worker (never the UI thread, never the tick), and a no-op after the first
    // run — every mesh is a content-hash hit.
    queue.submit_vmesh_sweep();
    let watcher = AssetWatcher::watch(content_root, WATCH_DEBOUNCE)
        .map_err(|e| tracing::warn!("asset watcher: {e}"))
        .ok();
    let cache_dir = app.path().app_data_dir().ok()?.join("thumbnails");
    let cache = match ThumbnailCache::open(cache_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("thumbnail cache: {e}");
            return None;
        }
    };
    // Project open/rescan is a non-hot moment: evict thumbnails whose content
    // hash no longer matches any live asset (orphaned by re-imports/edits).
    if let Ok(proj) = project.lock() {
        let removed = cache.sweep(&proj);
        if removed > 0 {
            tracing::info!("thumbnail cache: swept {removed} orphaned preview(s)");
        }
    }
    Some(AssetInner {
        project,
        queue,
        watcher,
        thumbs: Arc::new(Mutex::new(ThumbnailRig {
            thumb: Thumbnailer::default(),
            cache,
        })),
    })
}

/// Boot the asset subsystem at the default content root (`<app_data>/Content`),
/// so the app works before any project is opened. Opening a project re-roots it
/// via [`AssetState::reroot`]. Starts the background progress/rescan tick once.
pub fn init_assets_on_boot(app: &AppHandle) {
    let Ok(base) = app.path().app_data_dir() else {
        tracing::warn!("no app data dir; asset system disabled");
        return;
    };
    let root = base.join("Content");
    if let Some(inner) = build_inner(app, &root) {
        if let Some(state) = app.try_state::<AssetState>() {
            *state.inner.lock().expect("asset state") = Some(inner);
        }
    }
    spawn_tick(app.clone());
    tracing::info!("asset system ready ({})", root.display());
}

/// Seed a few starter material assets **and the starter character** on a fresh
/// (empty) content root so the Content Drawer isn't blank on first run.
///
/// # Why the character is here (wave GTA1)
///
/// `inf new` scaffolds the seventeen files of `samples/starter-character` into
/// every 3D project (`inf_project::template::STARTER_CHARACTER`). The editor's
/// own boot root — `<app_data>/Content`, the document you get before opening any
/// project — had no such scaffolding, so the boot level's character and the
/// `Place Actor ▸ Starter Character` row would both have named guids with no
/// bytes behind them and drawn placeholder cubes.
///
/// The **same** constant, copied verbatim with its sidecars (each names the
/// committed guid, so the scan adopts them with their identity intact) rather
/// than re-derived through `write_asset`, which would mint new ids.
fn seed_starter_content(project: &Arc<Mutex<AssetProject>>, root: &std::path::Path) {
    let Ok(mut proj) = project.lock() else { return };
    // Read BEFORE anything is written, or seeding the character would make the
    // root look non-fresh and the materials would never land.
    let fresh = proj.db().is_empty();
    // The character is seeded whenever it is ABSENT rather than only on a fresh
    // root: an editor that has been run before this wave has a content root with
    // three materials in it and no character, and "only on first run" would leave
    // every such machine with a boot level full of placeholder cubes.
    let has_character = inf_editor_core::samples::starter_character_ids()
        .skeleton
        .is_some_and(|id| proj.db().get(id).is_some());
    if !has_character {
        seed_starter_character(&mut proj, root);
    }
    if !fresh {
        return;
    }
    let Ok(dir) = proj.content_dir("Materials") else {
        return;
    };
    let starters = [
        ("DefaultLit", [0.8, 0.8, 0.8, 1.0], 0.0, 0.5),
        ("Metal", [0.9, 0.9, 0.92, 1.0], 1.0, 0.25),
        ("Rubber", [0.1, 0.1, 0.1, 1.0], 0.0, 0.9),
    ];
    for (name, base_color, metallic, roughness) in starters {
        let mat = MaterialAsset {
            base_color,
            metallic,
            roughness,
            ..Default::default()
        };
        let _ = proj.write_asset(&dir, name, &mat, None, vec![], None);
    }
}

/// Copy the committed starter character into `root/Characters/` and scan it in.
///
/// Verbatim bytes, sidecars included: each `.toml` names the guid
/// `inf_editor_core::samples::starter_character_ids` hands out, so the scan
/// adopts the files with the identity the boot level and the Place Actor row
/// both reference. Writing them through `write_asset` would mint new ids and the
/// references would dangle — which is the whole failure this seeding exists to
/// prevent.
fn seed_starter_character(proj: &mut AssetProject, root: &std::path::Path) {
    for (rel, bytes) in inf_project::template::STARTER_CHARACTER {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                tracing::warn!("could not seed {}: {e}", path.display());
                continue;
            }
        }
        if let Err(e) = std::fs::write(&path, *bytes) {
            tracing::warn!("could not seed {}: {e}", path.display());
        }
    }
    match proj.rescan() {
        Ok(n) => tracing::info!("seeded the starter character ({n} assets in the root)"),
        Err(e) => tracing::warn!("the seeded starter character did not scan: {e}"),
    }
}

/// The background tick: drain import progress → `assets://import`, drain the
/// file watcher → rescan → `assets://changed`, and refresh the viewport's terrain
/// index when a `.inf_terrain` lands.
///
/// The *logic* is [`inf_editor_core::assets::tick`] (Ring 1, unit-tested); this
/// is only the emit side. Two rules make it safe:
///
/// * The tick **never blocks on the project mutex** — `tick` uses `try_lock`, so a
///   multi-minute terrain import cannot stop the progress events it reports.
/// * The tick holds `state.inner` only for the (now lock-free) `tick` call, so
///   `terrain_import_cancel` and every other asset command can still get in while
///   an import runs.
fn spawn_tick(app: AppHandle) {
    // Last version successfully read; reused when a tick loses the try_lock race
    // (a version one tick stale is harmless — the frontend re-fetches either way).
    let mut last_version = 0u64;
    std::thread::Builder::new()
        .name("asset-tick".into())
        .spawn(move || loop {
            std::thread::sleep(TICK);
            let Some(state) = app.try_state::<AssetState>() else {
                continue;
            };
            let outcome = {
                let mut guard = match state.inner.lock() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                let Some(inner) = guard.as_mut() else {
                    continue;
                };
                let AssetInner {
                    project,
                    queue,
                    watcher,
                    ..
                } = inner;
                asset_tick(queue, project, watcher.as_ref())
            };
            for ev in &outcome.events {
                let _ = app.emit("assets://import", import_event(ev));
            }
            // ── SCRIPT1b: HOT RELOAD ────────────────────────
            //
            // In its OWN function, and that is a decision rather than
            // tidiness. `dcc.rs`'s invalidation gate bans every spelling of
            // play state from this loop, because the P23 law is that an asset
            // edit is not in the document and Simulate has no opinion about
            // whether the viewport is told. Hot reload is the one thing in
            // this tick that legitimately asks whether a session is live, so
            // it is kept where a reader can see it is NOT the invalidation —
            // and the gate reads both directions, so neither can grow into
            // the other.
            hot_reload_scripts(&app, &outcome.scripts);
            if let Some(v) = outcome.version {
                last_version = v;
            }
            if outcome.content_changed {
                let _ = app.emit(
                    "assets://changed",
                    AssetChanged {
                        version: last_version,
                    },
                );
            }
            // An asset the viewport resolves by GUID landed — a terrain (P16.4a),
            // or a mesh and its derived meshlet DAG (P18.3). The viewport's
            // loose-asset index predates it, so refresh it in place or the entity
            // the wizard (or a drag-drop) just spawned draws nothing.
            //
            // **Broadcast** (P23.2a): a landed asset is a fact about the
            // content root, not about one window, so every attached viewport
            // re-indexes. A `Primary`-only refresh would leave a second
            // viewport drawing nothing where the wizard just spawned something.
            if outcome.index_stale {
                if let Some(viewport) = app.try_state::<super::ViewportState>() {
                    viewport.refresh_asset_index(super::Target::All);
                }
            }
        })
        .expect("spawn asset tick");
}

/// **Hot-reload every `.infini` the watcher saw** (SCRIPT1b).
///
/// Compile each through `inf_script::source` — the ONE file door, the same
/// one the cook and the PIE payload builder use — and swap the class into a
/// running Simulate, which applies it on its next fixed step.
///
/// **Failure is contained here, before anything is queued.** A script that
/// does not compile produces diagnostics, not a class: the previous good
/// program keeps running and the session never learns there was an edit. So
/// the containment bound is tighter than the one the memo states for a
/// *runtime* failure — a broken edit does not take a handler down, because a
/// broken edit never becomes code.
///
/// The diagnostics go to the **Output Log** through `tracing`, which is the
/// smallest honest editor surface; the Problems panel and a `.infini`
/// language mode are SCRIPT2's.
///
/// Separate from [`spawn_tick`] so the asset-invalidation path stays free of
/// play state (`dcc.rs`'s `the_save_pushes_its_invalidation_unconditionally`),
/// and so this function stays free of the invalidation — the same gate reads
/// both.
fn hot_reload_scripts(app: &AppHandle, scripts: &[(uuid::Uuid, std::path::PathBuf)]) {
    for (asset, path) in scripts {
        match inf_script::compile_path(path, format!("script:{asset}")) {
            Ok((class, warnings)) => {
                for w in &warnings {
                    tracing::warn!(script = %path.display(), "{w}");
                }
                if let Some(state) = app.try_state::<super::SimState>() {
                    if state.reload_class(*asset, class) {
                        tracing::info!(
                            script = %path.display(),
                            "InfiniScript recompiled; the new program takes over \
                             on the next fixed step"
                        );
                    }
                }
            }
            Err(diags) => {
                tracing::error!(
                    script = %path.display(),
                    "InfiniScript did not compile; the PREVIOUS program keeps \
                     running:\n{}",
                    inf_script::render(&diags)
                );
            }
        }
    }
}

fn import_event(ev: &ImportProgress) -> ImportEventDto {
    let base = |id: u64, source: &std::path::Path, phase: &str| ImportEventDto {
        job: id,
        source: source.to_string_lossy().into_owned(),
        phase: phase.into(),
        produced: vec![],
        primary: None,
        cached: false,
        error: None,
        done: None,
        total: None,
        stage: None,
        advisories: vec![],
    };
    match ev {
        ImportProgress::Started { id, source } => base(*id, source, "started"),
        ImportProgress::Progress {
            id,
            source,
            done,
            total,
            stage,
        } => ImportEventDto {
            done: Some(*done),
            total: Some(*total),
            stage: Some(stage.clone()),
            ..base(*id, source, "progress")
        },
        ImportProgress::Finished {
            id,
            source,
            produced,
            primary,
            cached,
            advisories,
        } => ImportEventDto {
            produced: produced.iter().map(|a| a.to_string()).collect(),
            primary: primary.map(|a| a.to_string()),
            cached: *cached,
            advisories: advisories.clone(),
            ..base(*id, source, "finished")
        },
        ImportProgress::Failed { id, source, error } => ImportEventDto {
            error: Some(error.clone()),
            ..base(*id, source, "failed")
        },
    }
}

/// Broadcast `assets://changed` after a write. `pub(super)` so sibling command
/// modules that create assets (P24.1's `skel`) announce them the same way.
pub(super) fn emit_changed(app: &AppHandle, state: &AssetState) {
    let version = state.with_project(|p| Ok(p.version())).unwrap_or_default();
    let _ = app.emit("assets://changed", AssetChanged { version });
}

fn parse_id(s: &str) -> Result<AssetId, String> {
    s.parse::<AssetId>()
        .map_err(|e| format!("bad asset id: {e}"))
}

// ── reads ──────────────────────────────────────────────────────────────────

/// The full content snapshot (Content Drawer load + resync).
#[tauri::command]
pub async fn assets_snapshot(state: State<'_, AssetState>) -> Result<AssetSnapshot, String> {
    state.with_project(|p| Ok(snapshot::build(p)))
}

/// The assets that reference `id` (the "Show References" context action).
#[tauri::command]
pub async fn asset_references(
    id: String,
    state: State<'_, AssetState>,
) -> Result<Vec<AssetRefDto>, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| {
        Ok(p.referenced_by(id)
            .into_iter()
            .map(|r| snapshot::ref_dto(p, r))
            .collect())
    })
}

/// A rendered thumbnail as a PNG data URL, or `null` if the kind has no preview
/// (or a 3D kind with no GPU adapter). Cached on disk by content hash.
#[tauri::command]
pub async fn asset_thumbnail(
    id: String,
    state: State<'_, AssetState>,
) -> Result<Option<String>, String> {
    let id = parse_id(&id)?;
    // Clone the shared handles under a SHORT hold of the whole-asset-state lock,
    // then release it: the (headless-GPU render + PNG encode + file IO) below
    // must serialize only against *other* thumbnail renders (the `thumbs` mutex),
    // never behind the rest of the asset commands. `get_or_render` itself locks
    // `project` only briefly (to key by content hash + resolve what to draw) and
    // releases it before the GPU work.
    let (project, thumbs) = {
        let guard = state.inner.lock().map_err(|e| e.to_string())?;
        let inner = guard.as_ref().ok_or("assets not initialized")?;
        (inner.project.clone(), inner.thumbs.clone())
    };
    // GPU render + PNG encode + file read are blocking — keep them off the async
    // workers. Concurrent requests for the same *uncached* thumbnail serialize on
    // `thumbs` (one renders, the next then hits the disk cache); we deliberately
    // avoid per-key locking (last write wins, byte-identical content).
    tauri::async_runtime::spawn_blocking(move || {
        let mut rig = thumbs.lock().map_err(|e| e.to_string())?;
        let ThumbnailRig { thumb, cache } = &mut *rig;
        let Some(path) = cache.get_or_render(&project, id, thumb) else {
            return Ok(None);
        };
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        Ok(Some(format!("data:image/png;base64,{}", base64(&bytes))))
    })
    .await
    .map_err(|e| format!("asset_thumbnail task failed to run: {e}"))?
}

// ── mutations ───────────────────────────────────────────────────────────────

/// Import external files into a destination folder (default "Imported"). Returns
/// the queued job ids; progress arrives on `assets://import`.
#[tauri::command]
pub async fn asset_import(
    sources: Vec<String>,
    dest_folder: Option<String>,
    state: State<'_, AssetState>,
) -> Result<Vec<u64>, String> {
    let sub = dest_folder.unwrap_or_else(|| "Imported".into());
    let mut guard = state.inner.lock().map_err(|e| e.to_string())?;
    let inner = guard.as_mut().ok_or("assets not initialized")?;
    let dest = {
        let proj = inner.project.lock().map_err(|e| e.to_string())?;
        proj.content_dir(&sub).map_err(|e| e.to_string())?
    };
    let jobs = sources
        .into_iter()
        .map(|s| inner.queue.submit(PathBuf::from(s), dest.clone()))
        .collect();
    Ok(jobs)
}

/// Create a new authored asset: "mat" (Material), "biomeset" (a `.inf_biomes`
/// seeded with the starter biomes, P19.2), or a data kind "struct"/"enum"/"table"
/// (P4.5). Returns the new GUID.
#[tauri::command]
pub async fn asset_create(
    app: AppHandle,
    kind: String,
    folder: Option<String>,
    name: Option<String>,
    state: State<'_, AssetState>,
) -> Result<String, String> {
    let default_folder = match kind.as_str() {
        "mat" => "Materials",
        "biomeset" => biome_set::BIOME_SET_FOLDER,
        _ => "Data",
    };
    let sub = folder.unwrap_or_else(|| default_folder.to_string());
    let name = name.unwrap_or_else(|| match kind.as_str() {
        "mat" => "New Material".into(),
        "struct" => "NewStruct".into(),
        "enum" => "NewEnum".into(),
        "table" => "NewTable".into(),
        "biomeset" => "NewBiomes".into(),
        _ => "New Asset".into(),
    });
    let id = state.with_project(|p| {
        let dir = p.content_dir(&sub).map_err(|e| e.to_string())?;
        match kind.as_str() {
            "mat" => p
                .write_asset(&dir, &name, &MaterialAsset::default(), None, vec![], None)
                .map_err(|e| e.to_string()),
            "struct" | "enum" | "table" => {
                data::create_default(p, &kind, &dir, &name).map_err(|e| e.to_string())
            }
            "biomeset" => biome_set::create_default(p, &dir, &name).map_err(|e| e.to_string()),
            other => Err(format!("cannot create asset kind: {other}")),
        }
    })?;
    emit_changed(&app, &state);
    Ok(id.to_string())
}

/// Create a material instance of `parentId` (P7.4). Returns the new GUID.
#[tauri::command]
pub async fn asset_create_material_instance(
    app: AppHandle,
    parent_id: String,
    name: Option<String>,
    state: State<'_, AssetState>,
) -> Result<String, String> {
    let pid = parent_id.parse::<AssetId>().map_err(|e| e.to_string())?;
    let base = state
        .asset_name(pid)
        .unwrap_or_else(|| "Material".to_string());
    let name = name.unwrap_or_else(|| format!("{base} Instance"));
    let id = state.create_material_instance(pid, &name)?;
    emit_changed(&app, &state);
    Ok(id.to_string())
}

/// Load a material instance for the override editor (E-P2): the parent identity,
/// the parent's resolved PBR baseline (inherited values), and this instance's
/// sparse overrides.
#[tauri::command]
pub async fn asset_get_material_instance(
    id: String,
    state: State<'_, AssetState>,
) -> Result<MaterialInstanceDto, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| {
        let view = material_instance::get_material_instance(p, id).map_err(|e| e.to_string())?;
        Ok(MaterialInstanceDto {
            parent: view.parent.to_string(),
            parent_name: view.parent_name,
            resolved: MatValuesDto::from_material(&view.resolved_parent),
            overrides: MatOverridesDto::from_overrides(&view.overrides),
        })
    })
}

/// Save edited overrides onto a material instance (E-P2). Re-encodes the payload
/// through the standard rewrite path; the content-hash change invalidates the
/// thumbnail via `assets://changed`.
#[tauri::command]
pub async fn asset_save_material_instance(
    app: AppHandle,
    id: String,
    overrides: MatOverridesDto,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let id = parse_id(&id)?;
    let overrides = overrides.to_overrides();
    state.with_project(|p| {
        material_instance::save_material_instance(p, id, overrides).map_err(|e| e.to_string())
    })?;
    emit_changed(&app, &state);
    Ok(())
}

// ── biome sets (P19.2) ──────────────────────────────────────────────────────

/// A biome definition as the inline editor sees it. `pub(super)` so the terrain
/// commands can project the same shape into `TerrainBiomesDto` — one conversion,
/// so the toolbar's swatches and the editor's rows can never disagree.
pub(super) fn biome_def_dto(b: &BiomeDef) -> BiomeDefDto {
    BiomeDefDto {
        id: b.id,
        name: b.name.clone(),
        color: b.color,
        splat_layer: b.splat_layer,
        pcg_graph: b.pcg_graph.map(|g| AssetId(g).to_string()),
        water_hint: b.water_hint,
        structure_hint: b.structure_hint.clone(),
    }
}

/// The inverse. An **unparseable** `pcg_graph` is an error rather than a silent
/// `None`: dropping it would quietly break the biome's P19.3 scatter binding (and
/// the sidecar dependency edge that keeps the graph alive through a
/// delete-with-references check) at the moment the author believed they had set
/// it. Absent (`null`) and empty are both simply "no graph".
fn biome_def_from_dto(d: &BiomeDefDto) -> Result<BiomeDef, String> {
    let pcg_graph = match d.pcg_graph.as_deref() {
        Some(s) if !s.is_empty() => Some(
            s.parse::<AssetId>()
                .map_err(|e| format!("biome {} has an invalid pcg_graph {s:?}: {e}", d.id))?
                .uuid(),
        ),
        _ => None,
    };
    Ok(BiomeDef {
        id: d.id,
        name: d.name.clone(),
        color: d.color,
        splat_layer: d.splat_layer,
        pcg_graph,
        water_hint: d.water_hint,
        structure_hint: d.structure_hint.clone(),
    })
}

/// Load a `.inf_biomes` for the inline editor. Errors when the asset is missing,
/// is not a biome set, or fails to decode (the payload's `migrate` validates, so
/// a hand-edited file surfaces here rather than as an ambiguous lookup later).
#[tauri::command]
pub async fn asset_get_biome_set(
    id: String,
    state: State<'_, AssetState>,
) -> Result<BiomeSetDto, String> {
    let asset_id = parse_id(&id)?;
    state.with_project(|p| {
        let set = biome_set::get(p, asset_id).map_err(|e| e.to_string())?;
        // The DISPLAY name is the asset entry's — that is what the Content Drawer
        // shows, what Rename changes, and what the toolbar's set picker lists. The
        // payload's own `name` is only a fallback for an entry we cannot read.
        let name = p
            .db()
            .get(asset_id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| set.name.clone());
        Ok(BiomeSetDto {
            id: id.clone(),
            name,
            biomes: set.biomes.iter().map(biome_def_dto).collect(),
        })
    })
}

/// Save an edited `.inf_biomes`.
///
/// [`biome_set::save`] **validates** — duplicate ids, the reserved id `0`, a blank
/// name or an out-of-range splat layer are refused and the file is left alone —
/// so the error string this returns is the message the editor surfaces.
#[tauri::command]
pub async fn asset_save_biome_set(
    app: AppHandle,
    id: String,
    set: BiomeSetDto,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let asset_id = parse_id(&id)?;
    let mut biomes = Vec::with_capacity(set.biomes.len());
    for b in &set.biomes {
        biomes.push(biome_def_from_dto(b)?);
    }
    let payload = BiomeSet {
        schema_version: inf_terrain::BIOME_SET_SCHEMA_VERSION,
        name: set.name,
        biomes,
    };
    state.with_project(|p| biome_set::save(p, asset_id, payload).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    // A colour edit changes what the Biomes overlay must tint with, so every
    // terrain bound to this set has a stale palette the instant the save lands.
    super::terrain::push_biome_palettes(&app, &state);
    Ok(())
}

/// Load a data asset (struct/enum/table) for editing. `null` if not a data kind.
#[tauri::command]
pub async fn asset_data(
    id: String,
    state: State<'_, AssetState>,
) -> Result<Option<DataAssetDto>, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| data::to_dto(p, id).map_err(|e| e.to_string()))
}

/// Save an edited data asset.
#[tauri::command]
pub async fn asset_data_save(
    app: AppHandle,
    data_asset: DataAssetDto,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    state.with_project(|p| data::save_dto(p, &data_asset).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(())
}

/// Import a CSV/JSON file into an existing table asset, replacing its contents.
#[tauri::command]
pub async fn asset_table_import(
    app: AppHandle,
    id: String,
    source: String,
    state: State<'_, AssetState>,
) -> Result<Vec<String>, String> {
    let id = parse_id(&id)?;
    // The advisories reach the caller (C4-42): a cell that would not become its
    // column's type is imported as that type's zero, and a `.inf_table` full of
    // zeros is indistinguishable from a table of zeros unless somebody says so.
    let advisories = state.with_project(|p| {
        data::import_table_into(p, id, &PathBuf::from(&source)).map_err(|e| e.to_string())
    })?;
    for note in &advisories {
        tracing::warn!(target: "assets", "table import: {note}");
    }
    emit_changed(&app, &state);
    Ok(advisories)
}

/// The generated Rust source for a struct/enum asset (codegen preview).
#[tauri::command]
pub async fn asset_rust_source(id: String, state: State<'_, AssetState>) -> Result<String, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| data::rust_source(p, id).map_err(|e| e.to_string()))
}

/// Delete an asset. When still referenced (and `force` is false) nothing is
/// deleted and the referrers are returned so the UI can warn.
#[tauri::command]
pub async fn asset_delete(
    app: AppHandle,
    id: String,
    force: bool,
    state: State<'_, AssetState>,
) -> Result<DeleteResult, String> {
    let id = parse_id(&id)?;
    let result = state.with_project(|p| {
        let blockers = p.delete(id, force).map_err(|e| e.to_string())?;
        Ok(if blockers.is_empty() {
            DeleteResult {
                deleted: true,
                blockers: vec![],
            }
        } else {
            DeleteResult {
                deleted: false,
                blockers: blockers
                    .into_iter()
                    .map(|b| snapshot::ref_dto(p, b))
                    .collect(),
            }
        })
    })?;
    if result.deleted {
        emit_changed(&app, &state);
    }
    Ok(result)
}

/// Rename an asset.
#[tauri::command]
pub async fn asset_rename(
    app: AppHandle,
    id: String,
    name: String,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let id = parse_id(&id)?;
    state.with_project(|p| p.rename(id, &name).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(())
}

/// Duplicate an asset (fresh GUID). Returns the new id.
#[tauri::command]
pub async fn asset_duplicate(
    app: AppHandle,
    id: String,
    state: State<'_, AssetState>,
) -> Result<String, String> {
    let id = parse_id(&id)?;
    let new_id = state.with_project(|p| p.duplicate(id).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(new_id.to_string())
}

/// Replace an asset's tags.
#[tauri::command]
pub async fn asset_set_tags(
    app: AppHandle,
    id: String,
    tags: Vec<String>,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let id = parse_id(&id)?;
    state.with_project(|p| p.set_tags(id, tags).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(())
}

// ── sprite-sheet slicing (P8.2a) ─────────────────────────────────────────────

/// Read a texture's sprite-sheet slice model + pixel dimensions (the Sprite
/// Sheet panel loads this to draw the grid overlay and list slices).
#[tauri::command]
pub async fn texture_get_slices(
    id: String,
    state: State<'_, AssetState>,
) -> Result<SpriteSheetDto, String> {
    let asset_id = parse_id(&id)?;
    let (slices, w, h) = state.read_sprite_slices(asset_id)?;
    Ok(slices.to_dto(id, w, h))
}

/// Persist a texture's sprite-sheet slice model into its sidecar (deterministic
/// TOML, merged beside the texture-import settings). Emits `assets://changed`.
#[tauri::command]
pub async fn texture_set_slices(
    app: AppHandle,
    slices: SpriteSheetDto,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let asset_id = parse_id(&slices.texture_id)?;
    let model = sprite_sheet::SpriteSheetSlices::from_dto(&slices);
    state.write_sprite_slices(asset_id, &model)?;
    emit_changed(&app, &state);
    Ok(())
}

/// Resolve a material or material-instance asset to concrete PBR parameters,
/// following the instance→parent chain (depth-guarded against cycles).
fn resolve_material(proj: &AssetProject, id: AssetId, depth: u32) -> Option<MaterialAsset> {
    if depth > 16 {
        return None; // pathological instance chain
    }
    match proj.db().get(id)?.kind() {
        AssetKind::Material => proj.load_payload::<MaterialAsset>(id).ok(),
        AssetKind::MaterialInstance => {
            let inst = proj
                .load_payload::<inf_material::MaterialInstance>(id)
                .ok()?;
            let parent = resolve_material(proj, inst.parent, depth + 1)?;
            Some(inst.resolve(&parent))
        }
        _ => None,
    }
}

/// Standard base64 (no line breaks) for thumbnail data URLs — avoids a dep.
pub(crate) fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
