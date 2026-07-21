//! Level loading seams (P9.3 item 1 · wired to real content in P9.5).
//!
//! The player loads a world through two narrow traits so the P9.2 pieces (the
//! pack format + the runtime `.inf_lvl` decode in `inf-scene`) slot in cleanly:
//!
//! * [`LevelSource`] — produces the **raw serialized level bytes**.
//!   [`DevDirLevelSource`] reads an `.inf_lvl` file straight off disk (the
//!   `--level` dev-dir path); [`PackLevelSource`] opens a cooked
//!   `content.inf_pack` (+ `manifest.toml`) and returns the root level's bytes
//!   (the `--pack` / exported-game path).
//! * [`WorldBuilder`] — **decodes those bytes into a populated [`BuiltWorld`]**
//!   (an ECS world + the blueprint actors to tick + gravity/rate).
//!   [`InfSceneWorldBuilder`] is the real, P9.2-backed decoder;
//!   [`StubWorldBuilder`] is kept only for the "reader not wired" error surface.
//!
//! ## What the level format persists (and what it does not)
//!
//! `.inf_lvl` (the frozen `inf_scene` schema v2, mirroring the editor's
//! `EntityRecord`) persists per entity: `guid`, `name`, `parent`, `transform`,
//! `visible`, and the renderable/authoring components `mesh` / `material` /
//! `light` / `camera` / `sprite` / `tilemap` / `nine_slice` / `text2d` /
//! `light_2d`. [`InfSceneWorldBuilder`] instantiates **all** of them, so a cooked
//! level renders exactly as the editor viewport shows it.
//!
//! It does **not** yet persist the 2D **physics** components (`RigidBody2D` /
//! `Collider2D` / `CharacterController2D`) nor a per-entity **blueprint-class
//! binding**. Both are documented follow-ups (a physics-component + level-settings
//! record in `.inf_lvl`, and a class-link component). Until they land:
//!
//! * gravity/rate come from [`DEFAULT_GRAVITY`] / [`DEFAULT_HZ`] (a level-settings
//!   record is the follow-up), matching the platformer/`--demo` convention where
//!   the character applies its own gravity in the blueprint;
//! * actors bind via the P8/P9 **`CharacterController2D` heuristic**
//!   ([`resolve_actors`]) — every entity carrying a `CharacterController2D` gets
//!   the discovered actor class. Because the current level format carries no
//!   `CharacterController2D`, a decoded sample level yields an empty actor list
//!   today; the programmatic [`crate::demo`] world remains the runnable-gameplay
//!   proof, while the level/pack path proves faithful scene instantiation +
//!   deterministic headless run + cooked-==-uncooked determinism.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use glam::DVec2;
use uuid::Uuid;

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_blueprint::BlueprintClass;
use inf_ecs::components::CharacterController2D;
use inf_ecs::{EcsWorld, Guid};
use inf_scene::RuntimeEntity;

/// The cook's default pack file name (kept in sync with
/// `inf_packager::DEFAULT_PACK_NAME`; duplicated here so the shipped player does
/// not depend on the cook pipeline).
pub const PACK_FILE: &str = "content.inf_pack";

/// The cook's manifest file name (kept in sync with `inf_packager::MANIFEST_FILE`).
pub const MANIFEST_FILE: &str = "manifest.toml";

/// Default world gravity for a loaded level (a level-settings record is the
/// follow-up). [`DVec2::ZERO`] matches the platformer/`--demo` convention: the
/// character applies its own gravity in the blueprint, so a nonzero world gravity
/// would double it.
pub const DEFAULT_GRAVITY: DVec2 = DVec2::ZERO;

/// Default fixed update rate (Hz) for a loaded level.
pub const DEFAULT_HZ: f64 = 60.0;

/// A populated world ready to hand to [`RuntimeSim`](crate::runtime_sim::RuntimeSim).
pub struct BuiltWorld {
    /// The ECS world (entities + components).
    pub world: EcsWorld,
    /// The blueprint actors to tick: `(entity Guid, class)`.
    pub actors: Vec<(Uuid, BlueprintClass)>,
    /// World gravity for the 2D physics bridge (the platformer uses `ZERO`).
    pub gravity: DVec2,
    /// Fixed update rate (Hz).
    pub hz: f64,
    /// A human label for logs / the window title.
    pub label: String,
}

/// Produces the raw serialized bytes of a level (an `.inf_lvl` payload). The
/// dev-dir implementation reads a file; the pack-backed one (P9.2) reads a pack
/// entry.
pub trait LevelSource {
    /// The level's raw bytes.
    fn level_bytes(&self) -> Result<Vec<u8>, String>;
    /// A human label for logs / the window title.
    fn label(&self) -> String;
}

/// Decodes raw level bytes into a populated [`BuiltWorld`]. The runtime reader
/// (`inf-scene`, P9.2) provides the real implementation.
pub trait WorldBuilder {
    fn build(&self, level_bytes: &[u8]) -> Result<BuiltWorld, String>;
}

/// v1 [`LevelSource`]: read an `.inf_lvl` file straight off disk.
pub struct DevDirLevelSource {
    path: PathBuf,
}

impl DevDirLevelSource {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl LevelSource for DevDirLevelSource {
    fn level_bytes(&self) -> Result<Vec<u8>, String> {
        std::fs::read(&self.path).map_err(|e| format!("read level {}: {e}", self.path.display()))
    }

    fn label(&self) -> String {
        self.path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("level")
            .to_string()
    }
}

/// The placeholder [`WorldBuilder`] kept only to surface a clear "reader not
/// wired" error (unused on the real paths, which use [`InfSceneWorldBuilder`]).
pub struct StubWorldBuilder;

impl WorldBuilder for StubWorldBuilder {
    fn build(&self, _level_bytes: &[u8]) -> Result<BuiltWorld, String> {
        Err("no world builder wired (use InfSceneWorldBuilder)".to_string())
    }
}

/// The real [`WorldBuilder`]: decode `.inf_lvl` bytes with the Ring-0
/// [`inf_scene`] reader and instantiate a populated [`EcsWorld`], then bind the
/// discovered blueprint actor classes to `CharacterController2D` entities.
///
/// Holds the actor classes discovered beside the level (dev-dir) or in the pack
/// so [`WorldBuilder::build`] — which only sees the level bytes — can attach them.
pub struct InfSceneWorldBuilder {
    actors: Vec<BlueprintClass>,
    gravity: DVec2,
    hz: f64,
}

impl InfSceneWorldBuilder {
    /// Build with explicit gravity/rate.
    pub fn new(actors: Vec<BlueprintClass>, gravity: DVec2, hz: f64) -> Self {
        Self {
            actors,
            gravity,
            hz,
        }
    }

    /// Build with the documented defaults ([`DEFAULT_GRAVITY`] / [`DEFAULT_HZ`]).
    pub fn with_defaults(actors: Vec<BlueprintClass>) -> Self {
        Self::new(actors, DEFAULT_GRAVITY, DEFAULT_HZ)
    }
}

impl WorldBuilder for InfSceneWorldBuilder {
    fn build(&self, level_bytes: &[u8]) -> Result<BuiltWorld, String> {
        let level = inf_scene::decode(level_bytes).map_err(|e| format!("decode level: {e}"))?;
        let title = level.title;
        let mut world = populate_world(level.entities);
        world.propagate();
        let actors = resolve_actors(&world, &self.actors);
        tracing::info!(
            "inf-player: built '{}' — {} actor(s) bound",
            if title.is_empty() { "level" } else { &title },
            actors.len()
        );
        Ok(BuiltWorld {
            world,
            actors,
            gravity: self.gravity,
            hz: self.hz,
            label: if title.is_empty() {
                "level".to_string()
            } else {
                title
            },
        })
    }
}

/// Instantiate an [`EcsWorld`] from a decoded level's entities: spawn each with
/// its stable `Guid` + name, insert every persisted component, then rebuild the
/// hierarchy. Entities arrive parents-first (the `inf_scene` invariant), but the
/// reparent is a deliberate second pass so it is robust to order.
pub fn populate_world(entities: Vec<RuntimeEntity>) -> EcsWorld {
    let mut world = EcsWorld::new();
    let mut by_guid: HashMap<Uuid, inf_ecs::Entity> = HashMap::new();
    let mut pending_parents: Vec<(inf_ecs::Entity, Uuid)> = Vec::new();

    for e in entities {
        let RuntimeEntity {
            guid,
            name,
            parent,
            transform,
            visible,
            mesh,
            material,
            light,
            camera,
            sprite,
            tilemap,
            nine_slice,
            text2d,
            light_2d,
        } = e;

        let entity = world.spawn_with_guid(guid, &name, None);
        by_guid.insert(guid, entity);
        if !visible {
            world.set_visible(entity, false);
        }
        {
            // Overwrite the identity transform and add each present component.
            let mut em = world.world_mut().entity_mut(entity);
            em.insert(transform);
            if let Some(c) = mesh {
                em.insert(c);
            }
            if let Some(c) = material {
                em.insert(c);
            }
            if let Some(c) = light {
                em.insert(c);
            }
            if let Some(c) = camera {
                em.insert(c);
            }
            if let Some(c) = sprite {
                em.insert(c);
            }
            if let Some(c) = tilemap {
                em.insert(c);
            }
            if let Some(c) = nine_slice {
                em.insert(c);
            }
            if let Some(c) = text2d {
                em.insert(c);
            }
            if let Some(c) = light_2d {
                em.insert(c);
            }
        }
        world.mark_dirty();
        if let Some(p) = parent {
            pending_parents.push((entity, p));
        }
    }

    for (child, parent_guid) in pending_parents {
        if let Some(&pe) = by_guid.get(&parent_guid) {
            world.reparent(child, Some(pe));
        }
    }
    world
}

/// Bind actor classes to controllable entities (the P8/P9 heuristic mirrored from
/// the editor's `samples::character_actors`): every entity carrying a
/// `CharacterController2D` — in `Guid` order — is ticked with the first discovered
/// actor class. Empty when there are no classes or no such entities.
///
/// Per-entity blueprint-class binding (so different actors run on different
/// entities) is the documented follow-up; until the binding is persisted in
/// `.inf_lvl`, one class drives every character (exactly what the sample needs).
pub fn resolve_actors(world: &EcsWorld, classes: &[BlueprintClass]) -> Vec<(Uuid, BlueprintClass)> {
    let Some(class) = classes.first() else {
        return Vec::new();
    };
    let w = world.world();
    let mut guids: Vec<Uuid> = w
        .iter_entities()
        .filter(|e| e.contains::<CharacterController2D>())
        .filter_map(|e| e.get::<Guid>().map(|g| g.0))
        .collect();
    guids.sort();
    guids.into_iter().map(|g| (g, class.clone())).collect()
}

/// Read and decode every `.inf_act` blueprint class in `dir` (non-recursive),
/// sorted by path for a deterministic order. Malformed files are logged + skipped.
pub fn load_actor_classes_from_dir(dir: &Path) -> Vec<BlueprintClass> {
    let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_act"))
            .collect(),
        Err(_) => return Vec::new(),
    };
    files.sort();
    let mut out = Vec::new();
    for p in files {
        match std::fs::read(&p) {
            Ok(bytes) => match serde_json::from_slice::<BlueprintClass>(&bytes) {
                Ok(c) => out.push(c),
                Err(e) => tracing::warn!("inf-player: bad .inf_act {}: {e}", p.display()),
            },
            Err(e) => tracing::warn!("inf-player: read {}: {e}", p.display()),
        }
    }
    out
}

/// The minimal slice of the cook `manifest.toml` the player boots from. Unknown
/// fields are ignored (toml) so this stays decoupled from `inf_packager`'s full
/// `CookManifest`.
#[derive(Debug, Default, serde::Deserialize)]
struct BootManifest {
    #[serde(default)]
    project_name: String,
    #[serde(default)]
    packs: Vec<String>,
    #[serde(default)]
    root_level: Option<Uuid>,
}

/// A [`LevelSource`] backed by a cooked `content.inf_pack` (+ optional
/// `manifest.toml`). Opens the pack, resolves the root level GUID, and reads its
/// bytes; `.inf_act` classes are read straight out of the pack too.
pub struct PackLevelSource {
    reader: PackReader,
    root_level: AssetId,
    label: String,
}

impl PackLevelSource {
    /// Open a pack given either the **directory** holding `content.inf_pack` +
    /// `manifest.toml`, or the **pack file** itself (its sibling `manifest.toml`
    /// is used when present).
    pub fn open(path: &Path) -> Result<Self, String> {
        let (default_pack, manifest_path) = if path.is_dir() {
            (path.join(PACK_FILE), path.join(MANIFEST_FILE))
        } else {
            (path.to_path_buf(), path.with_file_name(MANIFEST_FILE))
        };

        let manifest: Option<BootManifest> = std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|t| toml::from_str(&t).ok());

        // A manifest may name a non-default pack file (only meaningful for a dir).
        let pack_path = match (&manifest, path.is_dir()) {
            (Some(m), true) if !m.packs.is_empty() => path.join(&m.packs[0]),
            _ => default_pack,
        };

        let reader = PackReader::open(&pack_path)
            .map_err(|e| format!("open pack {}: {e}", pack_path.display()))?;

        // Root level: the manifest's, else the lowest-GUID level entry in the pack.
        let root_level = manifest
            .as_ref()
            .and_then(|m| m.root_level)
            .map(AssetId)
            .filter(|id| reader.contains(*id))
            .or_else(|| {
                reader
                    .index()
                    .find(|e| e.kind == AssetKind::Level)
                    .map(|e| e.guid)
            })
            .ok_or_else(|| "pack has no level to boot".to_string())?;

        let label = manifest
            .as_ref()
            .map(|m| m.project_name.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "pack".to_string());

        Ok(Self {
            reader,
            root_level,
            label,
        })
    }

    /// Decode every `.inf_act` blueprint class stored in the pack (GUID order).
    pub fn actor_classes(&self) -> Result<Vec<BlueprintClass>, String> {
        let mut out = Vec::new();
        for e in self.reader.index() {
            if e.kind != AssetKind::Blueprint {
                continue;
            }
            let bytes = self
                .reader
                .read(e.guid)
                .map_err(|err| format!("read actor {}: {err}", e.guid))?;
            let class = serde_json::from_slice::<BlueprintClass>(&bytes)
                .map_err(|err| format!("decode actor {}: {err}", e.guid))?;
            out.push(class);
        }
        Ok(out)
    }
}

impl LevelSource for PackLevelSource {
    fn level_bytes(&self) -> Result<Vec<u8>, String> {
        self.reader
            .read(self.root_level)
            .map_err(|e| format!("read level {}: {e}", self.root_level))
    }

    fn label(&self) -> String {
        self.label.clone()
    }
}

/// Load a world by piping a [`LevelSource`]'s bytes through a [`WorldBuilder`].
pub fn load(source: &dyn LevelSource, builder: &dyn WorldBuilder) -> Result<BuiltWorld, String> {
    let bytes = source.level_bytes()?;
    tracing::info!(
        "inf-player: loaded {} level byte(s) from '{}'",
        bytes.len(),
        source.label()
    );
    builder.build(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::components::Sprite;
    use inf_ecs::Vec2d;

    /// A hand-built world with one `CharacterController2D` entity proves the
    /// binding heuristic (the level format cannot carry a CC2D, so this is the
    /// only way to unit-test attachment).
    #[test]
    fn resolve_actors_binds_the_class_to_cc2d_entities() {
        let mut world = EcsWorld::new();
        let g = Uuid::from_u128(0x11);
        let e = world.spawn_with_guid(g, "Hero", None);
        world
            .world_mut()
            .entity_mut(e)
            .insert(CharacterController2D {
                max_slope_deg: 46.0,
                snap_to_ground: 0.3,
                offset: 0.02,
            });
        // A second entity with no controller must not bind.
        world.spawn_with_guid(Uuid::from_u128(0x22), "Prop", None);

        let class = BlueprintClass::new("act:x", "X");
        let bound = resolve_actors(&world, std::slice::from_ref(&class));
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].0, g);
    }

    #[test]
    fn resolve_actors_is_empty_without_classes_or_controllers() {
        let mut world = EcsWorld::new();
        world.spawn_with_guid(Uuid::from_u128(1), "A", None);
        assert!(resolve_actors(&world, &[]).is_empty());
        let class = BlueprintClass::new("act:x", "X");
        // No CC2D entity → still empty even with a class available.
        assert!(resolve_actors(&world, std::slice::from_ref(&class)).is_empty());
    }

    /// `populate_world` instantiates components + hierarchy from decoded entities.
    #[test]
    fn populate_world_inserts_components_and_parenting() {
        let parent_guid = Uuid::from_u128(0xA0);
        let child_guid = Uuid::from_u128(0xA1);
        let mut parent = RuntimeEntity {
            guid: parent_guid,
            name: "Parent".into(),
            parent: None,
            transform: inf_ecs::components::Transform::from_translation(glam::DVec3::new(
                1.0, 2.0, 0.0,
            )),
            visible: true,
            mesh: None,
            material: None,
            light: None,
            camera: None,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
        };
        parent.sprite = Some(Sprite {
            size: Vec2d::new(1.0, 1.0),
            ..Sprite::default()
        });
        let child = RuntimeEntity {
            guid: child_guid,
            name: "Child".into(),
            parent: Some(parent_guid),
            visible: false,
            ..parent.clone()
        };

        let mut world = populate_world(vec![parent, child]);
        world.propagate();

        let pe = world.entity_of(parent_guid).unwrap();
        let ce = world.entity_of(child_guid).unwrap();
        assert!(world.world().get::<Sprite>(pe).is_some());
        assert_eq!(world.parent_of(ce), Some(pe));
        // The child's own visibility toggle propagated.
        assert!(
            !world
                .world()
                .get::<inf_ecs::components::ComputedVisibility>(ce)
                .unwrap()
                .0
        );
    }
}
