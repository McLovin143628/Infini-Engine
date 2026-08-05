//! **`.inf_fracture` sourcing for the shipped player (P22.3).**
//!
//! The `voxel::VoxelRegistry` shape, one phase later, with one real difference:
//! a `.inf_fracture` is **derived**, not authored. It has no sidecar and no place
//! in a project's content root — the cook synthesises one per mesh that some
//! level's `Destructible` names, and stores it under
//! [`inf_mesh::derived_fracture_id`], which is a pure bijection on the mesh's own
//! GUID.
//!
//! So there are only two sources, not four:
//!
//! * a **cooked pack**, where the cook put it ([`from_pack`](FractureRegistry::from_pack));
//! * a **PIE payload**, where the editor put the identical bytes by running the
//!   identical derivation ([`from_payload`](FractureRegistry::from_payload)).
//!
//! There is deliberately **no `from_dir`**. A dev `--level` run has loose
//! `.inf_mesh` files and no fractures beside them, because nothing has cooked
//! yet; a level run that way previews as indestructible, which is honest — and
//! silently re-deriving on load would make a `--level` run and a `--pack` run
//! differ in *when* the chunking happened, which is the class of divergence this
//! whole phase is shaped to forbid. Cook, then play.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_mesh::FractureAsset;
use inf_physics::d3::FractureState;
use uuid::Uuid;

/// The derived `.inf_fracture` GUID for a source mesh — the one lookup rule, so
/// the pack path and the PIE path name the same asset.
///
/// A thin delegate to Ring 0 ([`inf_mesh::derived_fracture_id`]), on the
/// `vmesh::derived_vmesh_id` precedent: the salt used to be hand-copied into the
/// player and held to the cook's by a drift test, and it is not going to be
/// copied a second time.
pub fn derived_fracture_id(mesh_id: Uuid) -> Uuid {
    inf_mesh::derived_fracture_id(AssetId(mesh_id)).uuid()
}

enum Source {
    /// No fracture content — every world before something was marked
    /// destructible, and every `--level` (uncooked) run.
    None,
    /// A cooked pack. Shared as an `Arc` with the other pack-backed stores so the
    /// mapping is opened once (the P18.2 rule).
    Pack(Arc<PackReader>),
    /// Bytes that arrived over the PIE wire. The editor derived them from the
    /// project's meshes and put them on the wire; the player subprocess has no
    /// filesystem handle to the project at all.
    Memory(HashMap<Uuid, Vec<u8>>),
}

/// Resolves a **mesh** GUID to its derived `.inf_fracture` chunk set.
///
/// Note the asymmetry with [`voxel::VoxelRegistry`](crate::voxel::VoxelRegistry):
/// that one is keyed by the asset a component *names*, this one by the mesh the
/// fracture is *derived from*, because `Destructible` deliberately names no asset
/// (the strength memo's §5 — a reference would be a second authority for the same
/// fact, and a dangling one to advise about).
pub struct FractureRegistry {
    source: Source,
    /// How many fracture assets the source can serve (a startup log / stat).
    count: usize,
}

impl Default for FractureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FractureRegistry {
    /// A registry that serves nothing.
    pub fn new() -> Self {
        Self {
            source: Source::None,
            count: 0,
        }
    }

    /// Serve `.inf_fracture` payloads out of a cooked pack (the `--pack` path).
    pub fn from_pack(reader: Arc<PackReader>) -> Self {
        let count = reader
            .index()
            .filter(|e| e.kind == AssetKind::Fracture)
            .count();
        Self {
            source: Source::Pack(reader),
            count,
        }
    }

    /// Serve the `.inf_fracture` payloads a PIE [`ScenePayload`] carried
    /// (schema v6).
    ///
    /// The editor derived these from the project's `.inf_mesh` bytes with the
    /// same `inf_mesh::fracture_mesh` the cook runs and put them on the wire; the
    /// player has no other route to them, because `--pie` discards every content
    /// path argument. This is what makes the PIE half of the phase-22 gate a real
    /// comparison rather than two empty maps agreeing — the P21.4 lesson, applied
    /// before it could be re-learned.
    ///
    /// [`ScenePayload`]: inf_runtime::pie::ScenePayload
    pub fn from_payload(fractures: &[(Uuid, Vec<u8>)]) -> Self {
        let map: HashMap<Uuid, Vec<u8>> = fractures.iter().cloned().collect();
        let count = map.len();
        Self {
            source: Source::Memory(map),
            count,
        }
    }

    /// How many fracture assets this registry can serve.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether this registry can serve nothing.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The chunk set derived from `mesh_id`, or `None` when this world has none.
    ///
    /// Decoded on every call rather than cached, exactly like
    /// `VoxelRegistry::load`: the caller ([`crate::attach_fractures`]) calls it
    /// once per destructible mesh at level load and never again, so a cache here
    /// would hold a level's worth of chunk geometry for the lifetime of the
    /// process to save work nobody repeats.
    pub fn load(&self, mesh_id: Uuid) -> Option<Arc<FractureAsset>> {
        let id = derived_fracture_id(mesh_id);
        let bytes: Vec<u8> = match &self.source {
            Source::None => return None,
            Source::Memory(map) => map.get(&id).cloned()?,
            Source::Pack(reader) => reader.read(AssetId(id)).ok()?,
        };
        match inf_asset::decode::<FractureAsset>(&bytes) {
            Ok(asset) => Some(Arc::new(asset)),
            Err(e) => {
                // A corrupt derived asset is an advisory, not a load failure: the
                // actor previews as indestructible, which is what a level with no
                // fracture at all does, and the level still runs.
                tracing::warn!("inf-player: undecodable .inf_fracture for mesh {mesh_id}: {e}");
                None
            }
        }
    }
}

/// Seed the **simulation's** fracture states from `assets` (P22.3), so
/// `destruct.apply_damage` has something to break.
///
/// A no-op for a world with no destructible content, in which case the step is
/// bit-identical to its pre-P22.3 self.
///
/// Called from **every** boot path that has a fracture source: `--pack` through
/// [`FractureRegistry::from_pack`], and `--pie` through
/// [`crate::sim_from_payload`] over the payload's own bytes. `--level` has no
/// source and is documented as such on [`FractureRegistry`].
///
/// Resolution goes through the ONE Ring-0 walk
/// ([`inf_physics::d3::resolve_fracture_states`]) that the editor's Simulate path
/// also calls, so "which actors can break, and into what" is answered in one
/// place rather than in two hosts that must be kept in step.
pub fn attach_fractures(sim: &mut crate::runtime_sim::RuntimeSim, assets: &FractureRegistry) {
    if assets.is_empty() {
        return;
    }
    let states: BTreeMap<Uuid, FractureState> =
        inf_physics::d3::resolve_fracture_states(sim.world(), |mesh| assets.load(mesh));
    sim.set_fractures(states);
}
