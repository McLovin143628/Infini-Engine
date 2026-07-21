//! `.inf_lvl` serialization (P3.5).
//!
//! A level is written as two files, per the asset-system rule (ROADMAP §3):
//!   * `<name>.inf_lvl`      — bincode payload (fast, compact scene data);
//!   * `<name>.inf_lvl.toml` — human-readable, git-diffable sidecar metadata
//!     (schema version, GUID, title, entity count, content hash).
//!
//! Determinism is load-bearing: entities serialize in creation order with a
//! fixed component layout, so save → load → save is **byte-identical** (the
//! phase gate). Every record carries concrete, `serde`-derived components — not
//! reflection — so the format is stable and diffable. `schema_version` +
//! [`migrate`] keep old files loadable forever.

use std::path::{Path, PathBuf};

use inf_ecs::components::{
    ActorClass, Camera, CharacterController2D, CharacterController3D, Collider2D, Collider3D,
    Light, Light2D, Material, MeshRef, NineSlice, RigidBody2D, RigidBody3D, Sprite, Text2D,
    Tilemap, Transform, Visibility,
};
use inf_ecs::math::{Vec2d, Vec3d};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::scene::SceneDoc;

/// Current on-disk schema. Bump on any breaking layout change and add a step to
/// [`migrate`].
///
/// * v1 — P3.5: transform + mesh/material/light/camera.
/// * v2 — P8.2b: appended the five 2D components (sprite / tilemap / nine-slice
///   / text / 2D light). Older v1 payloads load with those slots defaulted to
///   `None` (see [`decode`] + [`SceneFileV1`]).
/// * v3 — P9.5: appended the six physics components (`rigid_body_2d` /
///   `collider_2d` / `character_controller_2d` + the 3D trio) and the per-entity
///   blueprint-class binding (`actor`), plus a file-level [`LevelSettings`]
///   record (gravity + sim rate). Older v1/v2 payloads load with the new slots
///   defaulted and default settings (see [`decode`] + [`SceneFileV2`]).
pub const SCHEMA_VERSION: u32 = 3;

/// File-level simulation settings (P9.5 · schema v3). Replaces the player's
/// hard-coded `DEFAULT_GRAVITY`/`DEFAULT_HZ`. The serde defaults **preserve the
/// pre-v3 behaviour exactly**:
///
/// * `gravity_2d` = [`Vec2d::ZERO`] — the 2D **character-self-gravity**
///   convention: the platformer character applies its own gravity in the
///   blueprint (`vy -= GRAVITY*dt`), so a nonzero world gravity would double it.
/// * `gravity_3d` = `(0, -9.81, 0)` — real-world down for the 3D dynamic solver.
/// * `sim_hz` = `60` — the fixed update rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelSettings {
    #[serde(default)]
    pub gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    pub gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    pub sim_hz: f64,
}

fn default_gravity_3d() -> Vec3d {
    Vec3d::new(0.0, -9.81, 0.0)
}
fn default_sim_hz() -> f64 {
    60.0
}

impl Default for LevelSettings {
    fn default() -> Self {
        Self {
            gravity_2d: Vec2d::ZERO,
            gravity_3d: default_gravity_3d(),
            sim_hz: default_sim_hz(),
        }
    }
}

/// One entity's persisted state. All component slots are always present in the
/// binary stream (bincode is not self-describing — `Option` encodes its own
/// tag, but a field may never be conditionally skipped).
///
/// **Layout is append-only across schema versions.** New component slots are
/// added at the end; a payload from schema `v(N-1)` is decoded via its
/// version-specific record ([`EntityRecordV1`]) and lifted with the new slots
/// defaulted — never by reinterpreting the shorter byte stream.
///
/// # ⚠️ NOT YET PERSISTED: `inf_ecs::Terrain` (P10.1)
///
/// A spawned [`inf_ecs::components::Terrain`] (Add ▸ Terrain, [`SpawnKind::Terrain`])
/// is **live-session only** — this v3 `EntityRecord` has **no `terrain` slot**, so a
/// terrain is dropped on save and absent on reload (see the guard test
/// `terrain_is_not_persisted_yet_v4_todo`). This is deliberately *not* fixed in the
/// P10 glue batch: adding the slot is a schema-v3→v4 migration (a frozen
/// `EntityRecordV3` + a `SceneFileV3`→current lift, a v3 forever-load fixture, and
/// `record_of`/`raw_spawn_record`/`apply_to_doc` wiring), which must land as its own
/// versioned change, not smuggled into a glue batch.
///
/// **TODO(P10 · v4):** bump [`SCHEMA_VERSION`] to 4 and append
/// `pub terrain: Option<inf_ecs::components::Terrain>` here (`#[serde(default)]`),
/// mirroring how the v2 2D slots and v3 physics slots were added. `TerrainData`
/// already round-trips through serde deterministically, so the record change is the
/// only missing piece. Undo/redo of a terrain spawn are affected by the same gap
/// (the Create/Delete commands snapshot through `EntityRecord`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecord {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
    // ── v2 (P8.2b) 2D components ──────────────────────────────────────────
    /// A 2D sprite quad.
    #[serde(default)]
    pub sprite: Option<Sprite>,
    /// A chunked 2D tilemap (sparse, multi-chunk content persists in full).
    #[serde(default)]
    pub tilemap: Option<Tilemap>,
    /// A 9-slice bordered panel.
    #[serde(default)]
    pub nine_slice: Option<NineSlice>,
    /// A bitmap-text label.
    #[serde(default)]
    pub text2d: Option<Text2D>,
    /// A 2D radial light.
    #[serde(default)]
    pub light_2d: Option<Light2D>,
    // ── v3 (P9.5) physics components + actor binding ──────────────────────
    /// A 2D rigid body.
    #[serde(default)]
    pub rigid_body_2d: Option<RigidBody2D>,
    /// A 2D collider.
    #[serde(default)]
    pub collider_2d: Option<Collider2D>,
    /// A 2D kinematic character mover tuning block.
    #[serde(default)]
    pub character_controller_2d: Option<CharacterController2D>,
    /// A 3D rigid body.
    #[serde(default)]
    pub rigid_body_3d: Option<RigidBody3D>,
    /// A 3D collider.
    #[serde(default)]
    pub collider_3d: Option<Collider3D>,
    /// A 3D kinematic character mover tuning block.
    #[serde(default)]
    pub character_controller_3d: Option<CharacterController3D>,
    /// The GUID of the `.inf_act` blueprint-class asset bound to this entity
    /// (the [`ActorClass`] link); `None` when the entity runs no blueprint.
    #[serde(default)]
    pub actor: Option<Uuid>,
}

/// A schema-v1 [`EntityRecord`] (pre-P8.2b) — exactly the byte layout written by
/// older editors, used only to decode legacy payloads. Kept frozen forever so
/// the committed v1 fixture (and any level saved before P8.2b) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV1 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
}

impl EntityRecordV1 {
    /// Lift a v1 record to the **v2** shape (2D component slots default to
    /// `None`). First hop of the v1→v2→v3 chain.
    fn into_v2(self) -> EntityRecordV2 {
        EntityRecordV2 {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
        }
    }
}

/// A schema-v1 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV1 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV1>,
}

impl SceneFileV1 {
    /// Lift a v1 file to the **v2** shape (first hop of the v1→v2→v3 chain).
    fn into_v2(self) -> SceneFileV2 {
        SceneFileV2 {
            schema_version: 2,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV1::into_v2)
                .collect(),
        }
    }
}

/// A schema-**v2** [`EntityRecord`] (pre-P9.5) — the exact byte layout written
/// by P8.2b..P9.4 editors, used only to decode legacy payloads. Frozen forever
/// so the committed v2 fixture (and any level saved before P9.5) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV2 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
    #[serde(default)]
    pub sprite: Option<Sprite>,
    #[serde(default)]
    pub tilemap: Option<Tilemap>,
    #[serde(default)]
    pub nine_slice: Option<NineSlice>,
    #[serde(default)]
    pub text2d: Option<Text2D>,
    #[serde(default)]
    pub light_2d: Option<Light2D>,
}

impl EntityRecordV2 {
    /// Lift a v2 record to the current (v3) shape: physics slots + actor default
    /// to `None`.
    fn into_current(self) -> EntityRecord {
        EntityRecord {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: self.sprite,
            tilemap: self.tilemap,
            nine_slice: self.nine_slice,
            text2d: self.text2d,
            light_2d: self.light_2d,
            rigid_body_2d: None,
            collider_2d: None,
            character_controller_2d: None,
            rigid_body_3d: None,
            collider_3d: None,
            character_controller_3d: None,
            actor: None,
        }
    }
}

/// A schema-v2 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV2 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV2>,
}

impl SceneFileV2 {
    /// Lift a v2 file to the current (v3) shape (default [`LevelSettings`]).
    fn into_current(self) -> SceneFile {
        SceneFile {
            schema_version: SCHEMA_VERSION,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV2::into_current)
                .collect(),
            settings: LevelSettings::default(),
        }
    }
}

/// Just the leading `schema_version` field — decoded first (bincode reads fields
/// in order and stops) to pick the right versioned record before decoding the
/// whole payload.
#[derive(Deserialize)]
struct SceneFileHeader {
    schema_version: u32,
}

/// The full level payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    pub schema_version: u32,
    pub title: String,
    /// Entities in creation order (parents precede children).
    pub entities: Vec<EntityRecord>,
    /// File-level simulation settings (schema v3). `#[serde(default)]` keeps the
    /// dual-format (TOML/JSON) round trip working for older, settings-less docs.
    #[serde(default)]
    pub settings: LevelSettings,
}

/// Sidecar metadata (TOML). Deterministic field order → stable git diffs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    pub schema_version: u32,
    pub guid: Uuid,
    pub title: String,
    pub entity_count: u32,
    /// xxh3 of the bincode payload — a cheap integrity + change signal.
    pub content_hash: String,
}

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Serialize a single entity's state (used by [`to_scene_file`] and by undo,
/// which snapshots entities before a destructive edit).
pub fn record_of(doc: &SceneDoc, guid: Uuid) -> Option<EntityRecord> {
    let world = doc.world();
    let w = world.world();
    let e = world.entity_of(guid)?;
    let parent = world.parent_of(e).and_then(|p| world.guid_of(p));
    Some(EntityRecord {
        guid,
        name: world.name_of(e).unwrap_or("").to_string(),
        parent,
        transform: w
            .get::<Transform>(e)
            .copied()
            .unwrap_or(Transform::IDENTITY),
        visible: w.get::<Visibility>(e).map(|v| v.visible).unwrap_or(true),
        mesh: w.get::<MeshRef>(e).copied(),
        material: w.get::<Material>(e).copied(),
        light: w.get::<Light>(e).copied(),
        camera: w.get::<Camera>(e).copied(),
        sprite: w.get::<Sprite>(e).cloned(),
        tilemap: w.get::<Tilemap>(e).cloned(),
        nine_slice: w.get::<NineSlice>(e).cloned(),
        text2d: w.get::<Text2D>(e).cloned(),
        light_2d: w.get::<Light2D>(e).copied(),
        rigid_body_2d: w.get::<RigidBody2D>(e).copied(),
        collider_2d: w.get::<Collider2D>(e).copied(),
        character_controller_2d: w.get::<CharacterController2D>(e).copied(),
        rigid_body_3d: w.get::<RigidBody3D>(e).copied(),
        collider_3d: w.get::<Collider3D>(e).copied(),
        character_controller_3d: w.get::<CharacterController3D>(e).copied(),
        actor: w.get::<ActorClass>(e).map(|a| a.0),
    })
}

/// Project the document into a serializable [`SceneFile`].
pub fn to_scene_file(doc: &SceneDoc) -> SceneFile {
    let entities = doc
        .order()
        .iter()
        .filter_map(|&guid| record_of(doc, guid))
        .collect();
    SceneFile {
        schema_version: SCHEMA_VERSION,
        title: doc.title().to_string(),
        entities,
        settings: doc.settings(),
    }
}

/// Encode a [`SceneFile`] to the deterministic bincode payload.
pub fn encode(file: &SceneFile) -> Result<Vec<u8>, String> {
    bincode::serde::encode_to_vec(file, bincode_config()).map_err(|e| format!("encode: {e}"))
}

/// Decode a bincode payload, running migrations to the current schema. The
/// leading `schema_version` is decoded first to select the versioned record —
/// an older, shorter payload is never reinterpreted as the current (longer)
/// layout.
pub fn decode(bytes: &[u8]) -> Result<SceneFile, String> {
    let (header, _): (SceneFileHeader, usize) =
        bincode::serde::decode_from_slice(bytes, bincode_config())
            .map_err(|e| format!("decode header: {e}"))?;
    match header.schema_version {
        0 | 1 => {
            let (v1, _): (SceneFileV1, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v1: {e}"))?;
            migrate(v1.into_v2().into_current())
        }
        2 => {
            let (v2, _): (SceneFileV2, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v2: {e}"))?;
            migrate(v2.into_current())
        }
        3 => {
            let (file, _): (SceneFile, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode: {e}"))?;
            migrate(file)
        }
        n => Err(format!(
            "scene schema v{n} is newer than this editor (v{SCHEMA_VERSION})"
        )),
    }
}

/// Upgrade an older [`SceneFile`] to [`SCHEMA_VERSION`]. Newer-than-current is a
/// hard error (the editor is older than the file).
pub fn migrate(file: SceneFile) -> Result<SceneFile, String> {
    if file.schema_version > SCHEMA_VERSION {
        return Err(format!(
            "scene schema v{} is newer than this editor (v{SCHEMA_VERSION})",
            file.schema_version
        ));
    }
    // Records are already lifted to the current shape by the versioned decode
    // (v1→v2→v3); nothing more to do here. Future upgrades chain in `decode`.
    Ok(file)
}

/// Rebuild a document from a decoded [`SceneFile`]. Entities are recreated in
/// file order, so parents always exist before their children.
pub fn apply_to_doc(doc: &mut SceneDoc, file: &SceneFile) {
    doc.reset();
    for rec in &file.entities {
        let e = doc.spawn_bare(rec.guid, &rec.name, rec.parent);
        let w = doc.world_mut().world_mut();
        w.entity_mut(e).insert((
            rec.transform,
            Visibility {
                visible: rec.visible,
            },
        ));
        if let Some(m) = rec.mesh {
            w.entity_mut(e).insert(m);
        }
        if let Some(m) = rec.material {
            w.entity_mut(e).insert(m);
        }
        if let Some(l) = rec.light {
            w.entity_mut(e).insert(l);
        }
        if let Some(c) = rec.camera {
            w.entity_mut(e).insert(c);
        }
        if let Some(s) = &rec.sprite {
            w.entity_mut(e).insert(s.clone());
        }
        if let Some(t) = &rec.tilemap {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(n) = &rec.nine_slice {
            w.entity_mut(e).insert(n.clone());
        }
        if let Some(t) = &rec.text2d {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(l) = &rec.light_2d {
            w.entity_mut(e).insert(*l);
        }
        if let Some(c) = &rec.rigid_body_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.collider_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.character_controller_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.rigid_body_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.collider_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.character_controller_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(actor) = rec.actor {
            w.entity_mut(e).insert(ActorClass(actor));
        }
    }
    doc.set_title(&file.title);
    doc.set_settings(file.settings);
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
}

fn hash_hex(bytes: &[u8]) -> String {
    format!("{:016x}", xxh3(bytes))
}

/// Minimal xxh3-64. (inf-editor-core doesn't pull xxhash-rust; the asset DB
/// crate will. A small local implementation keeps the sidecar hash cheap here.)
fn xxh3(bytes: &[u8]) -> u64 {
    // FNV-1a 64 — not xxh3, but a stable content signal for the sidecar. The
    // real content-addressed hashing lands with the asset DB (P4.1); this only
    // needs to change when the payload changes.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Build the sidecar for a document + its encoded payload.
pub fn sidecar(doc: &SceneDoc, guid: Uuid, payload: &[u8]) -> Sidecar {
    Sidecar {
        schema_version: SCHEMA_VERSION,
        guid,
        title: doc.title().to_string(),
        entity_count: doc.order().len() as u32,
        content_hash: hash_hex(payload),
    }
}

/// The sidecar path for a `.inf_lvl` payload path (`foo.inf_lvl` →
/// `foo.inf_lvl.toml`).
pub fn sidecar_path(payload_path: &Path) -> PathBuf {
    let mut s = payload_path.as_os_str().to_os_string();
    s.push(".toml");
    PathBuf::from(s)
}

/// Save `doc` to `path` (payload) + its `.toml` sidecar. Returns the level GUID
/// written (fresh if `guid` is `None`).
pub fn save(doc: &SceneDoc, path: &Path, guid: Option<Uuid>) -> Result<Uuid, String> {
    let file = to_scene_file(doc);
    let payload = encode(&file)?;
    let guid = guid.unwrap_or_else(Uuid::new_v4);
    let side = sidecar(doc, guid, &payload);
    let toml = toml::to_string_pretty(&side).map_err(|e| format!("sidecar toml: {e}"))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, &payload).map_err(|e| format!("write payload: {e}"))?;
    std::fs::write(sidecar_path(path), toml).map_err(|e| format!("write sidecar: {e}"))?;
    Ok(guid)
}

/// Load a `.inf_lvl` payload into a fresh document.
pub fn load(path: &Path) -> Result<SceneDoc, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let file = decode(&bytes)?;
    let mut doc = SceneDoc::new();
    apply_to_doc(&mut doc, &file);
    doc.mark_saved();
    Ok(doc)
}

// ── autosave / crash recovery (P3.5.4) ───────────────────────────────────

/// The crash-recovery payload path inside `dir` (the app data dir).
pub fn recovery_path(dir: &Path) -> PathBuf {
    dir.join("crash-recovery.inf_lvl")
}

/// Write the document to the recovery file (called on a debounced autosave).
pub fn write_recovery(doc: &SceneDoc, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let payload = encode(&to_scene_file(doc))?;
    std::fs::write(recovery_path(dir), payload).map_err(|e| format!("write recovery: {e}"))
}

/// If a recovery file exists, load it and delete it (consumed on startup so a
/// clean exit removes it). Returns `None` when there's nothing to recover.
pub fn take_recovery(dir: &Path) -> Option<SceneDoc> {
    let path = recovery_path(dir);
    if !path.exists() {
        return None;
    }
    let doc = load(&path).ok();
    let _ = std::fs::remove_file(&path);
    doc
}

/// Remove the recovery file (called on a clean save / exit).
pub fn clear_recovery(dir: &Path) {
    let _ = std::fs::remove_file(recovery_path(dir));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;

    fn transform_path(doc: &SceneDoc) -> &'static str {
        doc.world()
            .registry()
            .editable()
            .iter()
            .find(|c| c.display == "Transform")
            .unwrap()
            .type_path
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut doc = SceneDoc::with_demo();
        // Author an extra edit so it's not just the demo.
        let c = doc.create(SpawnKind::Cone, "Cone", None);
        let tp = transform_path(&doc);
        doc.write_prop(
            c,
            tp,
            "translation",
            &inf_ecs::PropValue::Vec3([1.0, 2.0, 3.0]),
        );

        let file1 = to_scene_file(&doc);
        let bytes1 = encode(&file1).unwrap();

        // Load into a new doc and re-encode.
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&doc2)).unwrap();

        assert_eq!(bytes1, bytes2, "save→load→save must be byte-identical");
    }

    #[test]
    fn save_load_through_disk_preserves_scene_and_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Level.inf_lvl");

        let mut doc = SceneDoc::with_demo();
        let n_before = doc.snapshot().nodes.len();
        let guid = save(&doc, &path, None).unwrap();

        // Sidecar exists, is TOML, and names the level GUID + entity count.
        let toml = std::fs::read_to_string(sidecar_path(&path)).unwrap();
        assert!(toml.contains(&guid.to_string()));
        assert!(toml.contains("entity_count"));

        let mut loaded = load(&path).unwrap();
        assert_eq!(loaded.snapshot().nodes.len(), n_before);
        assert!(!loaded.is_dirty(), "a freshly loaded doc is clean");
    }

    /// GUARD (P10.1): a spawned `Terrain` is **not** persisted by the v3 schema —
    /// there is no `terrain` slot on [`EntityRecord`]. This test pins the current
    /// (lossy) behavior so the day someone adds the v4 slot, this test flips and
    /// forces them to update it deliberately. See the `EntityRecord` doc TODO.
    #[test]
    fn terrain_is_not_persisted_yet_v4_todo() {
        use inf_ecs::components::Terrain;

        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Terrain, "Terrain", None);
        // The live entity carries a non-empty starter terrain.
        let e = doc.entity_of(g).unwrap();
        assert!(
            !doc.world()
                .world()
                .get::<Terrain>(e)
                .unwrap()
                .data
                .is_empty(),
            "spawned terrain is live in-session"
        );

        // Save → load round trip.
        let bytes = encode(&to_scene_file(&doc)).unwrap();
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes).unwrap());

        // The entity survives (name/transform persist) …
        let e2 = loaded.entity_of(g).expect("entity itself persists");
        // … but its Terrain component is GONE (no v3 slot). When v4 adds the slot,
        // this assertion flips to `is_some()` and the test name's TODO is done.
        assert!(
            loaded.world().world().get::<Terrain>(e2).is_none(),
            "terrain is dropped on save/load until the v4 slot exists"
        );
    }

    #[test]
    fn recovery_round_trips_then_clears() {
        let dir = tempfile::tempdir().unwrap();
        let doc = SceneDoc::with_demo();
        write_recovery(&doc, dir.path()).unwrap();
        assert!(recovery_path(dir.path()).exists());
        let recovered = take_recovery(dir.path());
        assert!(recovered.is_some());
        assert!(!recovery_path(dir.path()).exists(), "consumed on recovery");
    }

    #[test]
    fn migrate_rejects_newer_schema() {
        let mut file = SceneFile {
            schema_version: SCHEMA_VERSION + 1,
            title: "x".into(),
            entities: vec![],
            settings: LevelSettings::default(),
        };
        assert!(migrate(file.clone()).is_err());
        file.schema_version = SCHEMA_VERSION;
        assert!(migrate(file).is_ok());
    }

    #[test]
    fn decode_rejects_future_schema() {
        // A payload whose leading version is newer than us must fail cleanly,
        // not decode as v2 garbage.
        let file = SceneFile {
            schema_version: SCHEMA_VERSION + 3,
            title: "future".into(),
            entities: vec![],
            settings: LevelSettings::default(),
        };
        let bytes = encode(&file).unwrap();
        assert!(decode(&bytes).is_err());
    }

    // ── v2 (P8.2b) 2D-component persistence ───────────────────────────────

    use inf_ecs::components::{
        Light2D, NineSlice, Sprite, Text2D, Tilemap, Transform as EcsTransform,
    };
    use inf_ecs::math::{Color, Vec2d};

    /// Insert a component onto `guid` (test-only; bypasses undo). A macro sits
    /// in for a generic fn so no `bevy_ecs::Bundle` bound has to be named (this
    /// crate deliberately doesn't depend on bevy directly).
    macro_rules! insert {
        ($doc:expr, $guid:expr, $comp:expr) => {{
            if let Some(e) = $doc.entity_of($guid) {
                $doc.world_mut().world_mut().entity_mut(e).insert($comp);
                $doc.world_mut().mark_dirty();
            }
        }};
    }

    /// Author one entity per 2D component (tilemap carries multi-chunk content)
    /// plus a 3D actor, so a round trip exercises every persisted slot.
    fn authored_2d_scene() -> SceneDoc {
        let mut doc = SceneDoc::new();
        doc.set_title("2D Level");

        // A plain 3D cube (mixed 2D/3D scene).
        doc.create(SpawnKind::Cube, "Cube", None);

        let spr = doc.create(SpawnKind::Empty, "Sprite", None);
        insert!(
            doc,
            spr,
            Sprite {
                texture: Some(uuid::Uuid::from_u128(0xABCD)),
                size: Vec2d::new(2.0, 3.0),
                pivot: Vec2d::new(0.25, 0.75),
                color: Color::new(0.2, 0.4, 0.6, 0.8),
                sorting_layer: -2,
                order: 4,
                flip_x: true,
                ..Default::default()
            }
        );

        let map = doc.create(SpawnKind::Empty, "Tilemap", None);
        let mut tm = Tilemap {
            atlas_cols: 4,
            atlas_rows: 4,
            ..Default::default()
        };
        // Two occupied chunks → the multi-chunk requirement.
        tm.set_tile(1, 1, 5);
        tm.set_tile(2, 3, 9);
        tm.set_tile(100, -50, 8);
        insert!(doc, map, tm);

        let panel = doc.create(SpawnKind::Empty, "Panel", None);
        insert!(
            doc,
            panel,
            NineSlice {
                size: Vec2d::new(6.0, 4.0),
                border_uv: [0.2, 0.3, 0.25, 0.15],
                ..Default::default()
            }
        );

        let label = doc.create(SpawnKind::Empty, "Label", None);
        insert!(
            doc,
            label,
            Text2D {
                text: "Hello\nInfinity".to_string(),
                tracking: 0.1,
                ..Default::default()
            }
        );

        let lamp = doc.create(SpawnKind::Empty, "Light2D", None);
        insert!(
            doc,
            lamp,
            Light2D {
                color: Color::new(1.0, 0.5, 0.2, 1.0),
                intensity: 2.5,
                radius: 8.0,
            }
        );

        doc.world_mut().propagate();
        doc
    }

    #[test]
    fn round_trip_with_2d_components_is_byte_identical() {
        let doc = authored_2d_scene();
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();

        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&doc2)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "save→load→save with all five 2D components must be byte-identical"
        );

        // The reloaded doc keeps every component's data, incl. multi-chunk tiles.
        let file = to_scene_file(&doc2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        let s = by_name("Sprite").sprite.as_ref().unwrap();
        assert_eq!(s.texture, Some(uuid::Uuid::from_u128(0xABCD)));
        assert_eq!(s.sorting_layer, -2);
        assert!(s.flip_x);

        let tm = by_name("Tilemap").tilemap.as_ref().unwrap();
        assert_eq!(tm.get_tile(1, 1), 5);
        assert_eq!(tm.get_tile(2, 3), 9);
        assert_eq!(tm.get_tile(100, -50), 8);
        assert_eq!(
            tm.chunks.len(),
            2,
            "multi-chunk content survives the round trip"
        );

        assert!(by_name("Panel").nine_slice.is_some());
        assert_eq!(
            by_name("Label").text2d.as_ref().unwrap().text,
            "Hello\nInfinity"
        );
        assert_eq!(by_name("Light2D").light_2d.unwrap().radius, 8.0);
        // The 3D cube carries none of the 2D slots.
        assert!(by_name("Cube").sprite.is_none());
        assert!(by_name("Cube").tilemap.is_none());
    }

    #[test]
    fn two_d_scene_survives_disk_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TwoD.inf_lvl");
        let doc = authored_2d_scene();
        save(&doc, &path, None).unwrap();
        let mut loaded = load(&path).unwrap();
        let file = to_scene_file(&loaded);
        assert!(file.entities.iter().any(|r| r.tilemap.is_some()));
        assert!(file.entities.iter().any(|r| r.sprite.is_some()));
        assert_eq!(loaded.snapshot().nodes.len(), 6);
    }

    #[test]
    fn scene_file_is_dual_format_serde_safe() {
        // The asset rule (ROADMAP §4) requires records serialize in the
        // human-readable format too — the chunked tilemap, atlas rects and UUIDs
        // all have to survive TOML/JSON, not just bincode.
        let file = to_scene_file(&authored_2d_scene());

        let toml_s = toml::to_string(&file).expect("scene serializes to TOML");
        let back: SceneFile = toml::from_str(&toml_s).expect("scene deserializes from TOML");
        assert_eq!(back, file, "TOML round trip preserves every 2D component");

        let json = serde_json::to_string(&file).unwrap();
        let back_json: SceneFile = serde_json::from_str(&json).unwrap();
        assert_eq!(back_json, file);
    }

    // ── schema-migration fixture discipline (ROADMAP §3) ──────────────────

    /// The committed pre-P8.2b (schema v1) payload, load-tested forever.
    fn v1_fixture_bytes() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v1.inf_lvl");
        std::fs::read(path).expect("committed v1 fixture is present")
    }

    /// Rebuild the exact schema-v1 `SceneFile` the fixture was generated from,
    /// so its provenance is reproducible from frozen legacy types. Any change to
    /// [`EntityRecordV1`]/[`SceneFileV1`] that alters the v1 layout breaks this.
    fn v1_reference() -> SceneFileV1 {
        let g = uuid::Uuid::from_u128;
        SceneFileV1 {
            schema_version: 1,
            title: "Fixture Level".into(),
            entities: vec![
                EntityRecordV1 {
                    guid: g(0x1001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform {
                        translation: inf_ecs::math::Vec3d::ZERO,
                        rotation: inf_ecs::math::Vec3d::ZERO,
                        scale: inf_ecs::math::Vec3d::new(20.0, 1.0, 20.0),
                    },
                    visible: true,
                    mesh: Some(inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Plane,
                    }),
                    material: Some(inf_ecs::components::Material {
                        base_color: Color::new(0.3, 0.32, 0.35, 1.0),
                        ..Default::default()
                    }),
                    light: None,
                    camera: None,
                },
                EntityRecordV1 {
                    guid: g(0x1002),
                    name: "Hero".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(-2.0, 0.5, 0.0)),
                    visible: true,
                    mesh: Some(inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Cube,
                    }),
                    material: Some(inf_ecs::components::Material::default()),
                    light: None,
                    camera: None,
                },
                EntityRecordV1 {
                    guid: g(0x1003),
                    name: "Sun".into(),
                    parent: None,
                    transform: EcsTransform::IDENTITY,
                    visible: true,
                    mesh: None,
                    material: None,
                    light: Some(inf_ecs::components::Light {
                        kind: inf_ecs::components::LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 1.0,
                    }),
                    camera: None,
                },
                EntityRecordV1 {
                    guid: g(0x1004),
                    name: "Cam".into(),
                    parent: None,
                    transform: EcsTransform::IDENTITY,
                    visible: false,
                    mesh: None,
                    material: None,
                    light: None,
                    camera: Some(inf_ecs::components::Camera::default()),
                },
            ],
        }
    }

    #[test]
    fn v1_fixture_is_reproducible_and_genuinely_v1() {
        let bytes = v1_fixture_bytes();
        // The very first byte is the schema version varint (1).
        assert_eq!(bytes[0], 1, "fixture must be a genuine schema-v1 payload");
        let rebuilt = bincode::serde::encode_to_vec(v1_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed fixture must match our frozen v1 writer"
        );
    }

    #[test]
    fn v1_fixture_loads_forever() {
        let file = decode(&v1_fixture_bytes()).expect("v1 fixture decodes");
        // Migrated up to the current schema.
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "Fixture Level");
        assert_eq!(file.entities.len(), 4);

        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // Legacy 3D data preserved.
        assert!(by_name("Ground").mesh.is_some());
        assert!(by_name("Ground").material.is_some());
        assert!(by_name("Sun").light.is_some());
        assert!(by_name("Cam").camera.is_some());
        assert!(!by_name("Cam").visible);
        // Every 2D + v3 slot defaulted on the old payload.
        for r in &file.entities {
            assert!(r.sprite.is_none());
            assert!(r.tilemap.is_none());
            assert!(r.nine_slice.is_none());
            assert!(r.text2d.is_none());
            assert!(r.light_2d.is_none());
            assert!(r.rigid_body_2d.is_none());
            assert!(r.collider_2d.is_none());
            assert!(r.character_controller_2d.is_none());
            assert!(r.actor.is_none());
        }
        // Legacy files carry no settings → the defaults (2D gravity zero).
        assert_eq!(file.settings, LevelSettings::default());
    }

    #[test]
    fn v1_fixture_loads_into_a_document() {
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &decode(&v1_fixture_bytes()).unwrap());
        assert_eq!(doc.snapshot().nodes.len(), 4);
        assert_eq!(doc.title(), "Fixture Level");
    }

    // ── schema-v3 (P9.5) physics + actor + settings persistence ────────────

    use inf_ecs::components::{
        ActorClass, BodyKind2D, CharacterController2D as CC2D, Collider2D, ColliderShape2DKind,
        RigidBody2D,
    };
    use inf_ecs::math::Vec3d;

    /// Author a scene exercising every v3 slot: a static ground (rb2d+collider),
    /// a player (rb2d + collider + character controller + an actor binding), and
    /// non-default level settings — so a round trip covers physics + actor +
    /// settings.
    fn authored_v3_scene() -> (SceneDoc, uuid::Uuid) {
        let actor_guid = uuid::Uuid::from_u128(0xAC70_0001);
        let mut doc = SceneDoc::new();
        doc.set_title("Physics Level");
        doc.set_settings(LevelSettings {
            gravity_2d: Vec2d::new(0.0, -20.0),
            gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
            sim_hz: 120.0,
        });

        let ground = doc.create(SpawnKind::Empty, "Ground", None);
        insert!(
            doc,
            ground,
            RigidBody2D {
                kind: BodyKind2D::Static,
                ..Default::default()
            }
        );
        insert!(
            doc,
            ground,
            Collider2D {
                shape_kind: ColliderShape2DKind::Box,
                half_extents: Vec2d::new(3.0, 0.5),
                ..Default::default()
            }
        );

        let player = doc.create(SpawnKind::Empty, "Player", None);
        insert!(
            doc,
            player,
            RigidBody2D {
                kind: BodyKind2D::Kinematic,
                fixed_rotation: true,
                ..Default::default()
            }
        );
        insert!(
            doc,
            player,
            Collider2D {
                shape_kind: ColliderShape2DKind::Capsule,
                half_extents: Vec2d::new(0.3, 0.35),
                radius: 0.3,
                ..Default::default()
            }
        );
        insert!(doc, player, CC2D::default());
        insert!(doc, player, ActorClass(actor_guid));

        doc.world_mut().propagate();
        (doc, actor_guid)
    }

    #[test]
    fn round_trip_with_v3_physics_and_actor_is_byte_identical() {
        let (doc, actor_guid) = authored_v3_scene();
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], 3, "authored payload is a genuine schema-v3 file");

        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&doc2)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "save→load→save with physics + actor + settings must be byte-identical"
        );

        // The reloaded doc keeps the physics components, the actor binding, and
        // the non-default settings.
        let file = to_scene_file(&doc2);
        let player = file.entities.iter().find(|r| r.name == "Player").unwrap();
        assert_eq!(player.rigid_body_2d.unwrap().kind, BodyKind2D::Kinematic);
        assert_eq!(
            player.collider_2d.unwrap().shape_kind,
            ColliderShape2DKind::Capsule
        );
        assert!(player.character_controller_2d.is_some());
        assert_eq!(player.actor, Some(actor_guid));
        assert_eq!(doc2.settings().gravity_2d, Vec2d::new(0.0, -20.0));
        assert_eq!(doc2.settings().sim_hz, 120.0);
    }

    #[test]
    fn v3_scene_is_dual_format_serde_safe() {
        // The dual-format rule: physics + actor + settings must survive TOML/JSON.
        let (doc, _) = authored_v3_scene();
        let file = to_scene_file(&doc);
        let toml_s = toml::to_string(&file).expect("v3 scene serializes to TOML");
        let back: SceneFile = toml::from_str(&toml_s).expect("v3 scene deserializes from TOML");
        assert_eq!(
            back, file,
            "TOML round trip preserves physics + actor + settings"
        );
        let json = serde_json::to_string(&file).unwrap();
        assert_eq!(serde_json::from_str::<SceneFile>(&json).unwrap(), file);
    }

    // ── v2 forever-load fixture discipline (mirrors the v1 fixture) ─────────

    /// The committed pre-P9.5 (schema v2) payload, load-tested forever.
    fn v2_fixture_bytes() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v2.inf_lvl");
        std::fs::read(path).expect("committed v2 fixture is present")
    }

    /// Rebuild the exact schema-v2 `SceneFile` the v2 fixture was generated from,
    /// from the frozen [`EntityRecordV2`]/[`SceneFileV2`] types. Any change to the
    /// v2 layout breaks this (the provenance lock).
    fn v2_reference() -> SceneFileV2 {
        let g = uuid::Uuid::from_u128;
        let mut tm = Tilemap {
            atlas_cols: 2,
            atlas_rows: 2,
            ..Default::default()
        };
        tm.set_tile(0, 0, 1);
        tm.set_tile(1, 0, 2);
        SceneFileV2 {
            schema_version: 2,
            title: "V2 Fixture Level".into(),
            entities: vec![
                EntityRecordV2 {
                    guid: g(0x2001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform {
                        translation: inf_ecs::math::Vec3d::ZERO,
                        rotation: inf_ecs::math::Vec3d::ZERO,
                        scale: inf_ecs::math::Vec3d::new(10.0, 1.0, 1.0),
                    },
                    visible: true,
                    mesh: Some(inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Plane,
                    }),
                    material: Some(inf_ecs::components::Material::default()),
                    light: None,
                    camera: None,
                    sprite: None,
                    tilemap: Some(tm),
                    nine_slice: None,
                    text2d: None,
                    light_2d: None,
                },
                EntityRecordV2 {
                    guid: g(0x2002),
                    name: "Sprite".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(1.0, 2.0, 0.0)),
                    visible: true,
                    mesh: None,
                    material: None,
                    light: None,
                    camera: None,
                    sprite: Some(Sprite {
                        size: Vec2d::new(0.8, 1.2),
                        color: Color::new(0.9, 0.4, 0.3, 1.0),
                        ..Default::default()
                    }),
                    tilemap: None,
                    nine_slice: None,
                    text2d: None,
                    light_2d: Some(Light2D {
                        color: Color::WHITE,
                        intensity: 1.0,
                        radius: 5.0,
                    }),
                },
            ],
        }
    }

    #[test]
    fn v2_fixture_is_reproducible_and_genuinely_v2() {
        let bytes = v2_fixture_bytes();
        assert_eq!(bytes[0], 2, "fixture must be a genuine schema-v2 payload");
        let rebuilt = bincode::serde::encode_to_vec(v2_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v2 fixture must match our frozen v2 writer"
        );
    }

    #[test]
    fn v2_fixture_loads_forever_and_lifts_to_v3() {
        let file = decode(&v2_fixture_bytes()).expect("v2 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V2 Fixture Level");
        assert_eq!(file.entities.len(), 2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // v2 data preserved through the v2→v3 lift.
        assert!(by_name("Ground").tilemap.is_some());
        assert!(by_name("Sprite").sprite.is_some());
        assert!(by_name("Sprite").light_2d.is_some());
        // Every v3 slot defaulted, and settings default (2D gravity zero).
        for r in &file.entities {
            assert!(r.rigid_body_2d.is_none());
            assert!(r.collider_2d.is_none());
            assert!(r.rigid_body_3d.is_none());
            assert!(r.actor.is_none());
        }
        assert_eq!(file.settings, LevelSettings::default());
    }
}
