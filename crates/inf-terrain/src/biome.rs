//! The **`BiomeSet` asset** (`.inf_biomes`, P19.2): the level's named biomes.
//!
//! A biome is a *label* an author paints onto terrain — "boreal forest", "salt
//! flat", "old town" — plus the data every downstream system needs to act on that
//! label. The set is an asset rather than a component field for the same reason a
//! material is: it is authored once and shared by every terrain in the level (and
//! across levels), and the PCG binding in P19.3 wants to resolve a biome by GUID.
//!
//! ```text
//! BiomeSet
//!   ├ schema_version : u32
//!   ├ name           : String
//!   └ biomes         : Vec<BiomeDef>
//!                        ├ id              : u8      (1..=255; 0 is RESERVED)
//!                        ├ name            : String
//!                        ├ color           : [f32; 4] linear RGBA, the overlay swatch
//!                        ├ splat_layer     : Option<u8>     → which terrain layer it shades as
//!                        ├ pcg_graph       : Option<Uuid>   → the P19.3 hook
//!                        ├ water_hint      : Option<f32>    → still-water level, metres (P20)
//!                        └ structure_hint  : Option<String> → building-palette name (P19.5)
//! ```
//!
//! # Id 0 is reserved, and that is what makes the storage sparse
//!
//! [`UNASSIGNED_BIOME`](crate::UNASSIGNED_BIOME) is `0`. A tile that has never
//! been biome-painted stores **no** per-sample ids at all
//! ([`TerrainTile::biomes`](crate::TerrainTile::biomes)), which only works if the
//! default value means something coherent — so `0` is permanently *"no biome"* and
//! a set may not define it. [`BiomeSet::validate`] enforces that, and the tile
//! layer's sparse default is priced by test.
//!
//! # The hint fields are plain data, deliberately
//!
//! `water_hint` and `structure_hint` are inert here: nothing in P19.2 reads them.
//! They are declared now because they are what P20 (water) and P19.5 (building
//! grammar) will ask a biome for, and adding a field to a bincode payload later
//! costs a schema migration on every `.inf_biomes` in every project. Storing them
//! as *hints* — advisory scalars a consumer may ignore — rather than as bound
//! references keeps them honest: an empty hint is not a dangling edge.
//!
//! `pcg_graph` **is** a real reference (it becomes a cook dependency edge in
//! P19.3), which is why it is a `Uuid` and not a name.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use inf_asset::{AssetKind, AssetPayload};

use crate::splat::SPLAT_LAYERS;
use crate::tile::{MAX_BIOMES, UNASSIGNED_BIOME};

/// Current `.inf_biomes` payload schema version.
pub const BIOME_SET_SCHEMA_VERSION: u32 = 1;

/// The colour the **unassigned** biome (id `0`) draws as in the biome overlay: a
/// neutral mid grey. Not a [`BiomeDef`] — id 0 cannot be defined — so the one
/// place it is stated is here.
pub const UNASSIGNED_BIOME_COLOR: [f32; 4] = [0.22, 0.22, 0.24, 1.0];

/// One named biome.
///
/// `Clone + PartialEq` so the editor can diff a draft against the saved value;
/// no `Copy` (it owns strings).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiomeDef {
    /// Stable id, `1..=255`. This is the value stored per terrain sample, so it
    /// is the *identity* of the biome — renaming is free, renumbering repaints
    /// the world. `0` is [reserved](UNASSIGNED_BIOME) and rejected by
    /// [`BiomeSet::validate`].
    pub id: u8,
    /// Display name. Free text; uniqueness is *not* enforced (two biomes may
    /// legitimately share a name in different sets, and ids are the identity).
    pub name: String,
    /// Linear RGBA swatch — what the Biomes view mode tints this biome's terrain,
    /// and what the toolbar picker shows. Linear, not sRGB: it is uploaded
    /// straight into the shader palette, which writes into a linear HDR target.
    pub color: [f32; 4],
    /// Which of the terrain's four [splat layers](crate::SPLAT_LAYERS) this biome
    /// shades as, if any. Advisory in P19.2 — painting a biome does **not** paint
    /// the splat weights (they are independent authored layers, and silently
    /// rewriting one from the other would make a paint stroke unpredictably
    /// destructive). It is stored so a future "apply biome splat" action, and
    /// P19.3's evaluation, have a declared mapping instead of a convention.
    pub splat_layer: Option<u8>,
    /// The `.inf_pcg` graph this biome scatters with — **the P19.3 hook**. `None`
    /// means the biome contributes no procedural content.
    pub pcg_graph: Option<Uuid>,
    /// Still-water level for this biome, in **metres** of absolute world height
    /// (SI doctrine: 1 world unit = 1 metre). `None` = no standing water. Plain
    /// data in P19.2; P20's water system is its first reader.
    pub water_hint: Option<f32>,
    /// Building/structure palette name for this biome (P19.5's `office`,
    /// `house`, … module sets). Plain data in P19.2.
    pub structure_hint: Option<String>,
}

impl BiomeDef {
    /// A biome with the given id and name, a mid-grey swatch and no hints.
    pub fn new(id: u8, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            color: [0.5, 0.5, 0.5, 1.0],
            splat_layer: None,
            pcg_graph: None,
            water_hint: None,
            structure_hint: None,
        }
    }
}

/// Why a [`BiomeSet`] is not valid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BiomeSetError {
    /// A definition claimed the reserved id `0`.
    #[error(
        "biome id 0 is reserved for 'unassigned' and cannot be defined \
         (it is the sparse per-sample default)"
    )]
    ReservedId,
    /// Two definitions share an id — the per-sample value would be ambiguous.
    #[error("duplicate biome id {0}")]
    DuplicateId(u8),
    /// More definitions than ids exist.
    #[error("a biome set holds at most {MAX_BIOMES} biomes (ids 1..=255), got {0}")]
    TooMany(usize),
    /// A biome named a splat layer outside the four the terrain has.
    #[error("biome {id} maps to splat layer {layer}, but a terrain has {SPLAT_LAYERS} layers")]
    BadSplatLayer { id: u8, layer: u8 },
    /// A biome has a blank name — it would be unpickable in the toolbar.
    #[error("biome {0} has an empty name")]
    EmptyName(u8),
}

/// The `.inf_biomes` payload: a level's named biomes.
///
/// Deterministic bincode via [`AssetPayload`]. **No `skip_serializing_if`
/// anywhere** — that desyncs a non-self-describing stream (the engine-wide law);
/// `Option` fields encode their tag byte unconditionally, which is exactly what
/// keeps this round-trippable.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiomeSet {
    /// On-disk schema version (see [`BIOME_SET_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Display name of the set.
    pub name: String,
    /// The definitions. Order is authoring order (what the editor lists and the
    /// toolbar shows); **lookup is by [`id`](BiomeDef::id)**, never by index.
    pub biomes: Vec<BiomeDef>,
}

impl Default for BiomeSet {
    /// An empty, valid set at the current schema version.
    fn default() -> Self {
        Self {
            schema_version: BIOME_SET_SCHEMA_VERSION,
            name: "Biomes".into(),
            biomes: Vec::new(),
        }
    }
}

impl BiomeSet {
    /// A named, empty set at the current schema version.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// The starter set a freshly created `.inf_biomes` carries — three biomes so
    /// the paint tool and the overlay are usable the moment the asset exists.
    pub fn starter() -> Self {
        let mut set = Self::new("Biomes");
        let mut push = |id, name: &str, color: [f32; 4], layer: u8| {
            set.biomes.push(BiomeDef {
                color,
                splat_layer: Some(layer),
                ..BiomeDef::new(id, name)
            });
        };
        push(1, "Grassland", [0.29, 0.45, 0.19, 1.0], 0);
        push(2, "Rock", [0.42, 0.40, 0.37, 1.0], 1);
        push(3, "Snow", [0.86, 0.89, 0.94, 1.0], 3);
        set
    }

    /// The definition with `id`, or `None` (including for
    /// [`UNASSIGNED_BIOME`](crate::UNASSIGNED_BIOME), which is never defined).
    pub fn get(&self, id: u8) -> Option<&BiomeDef> {
        self.biomes.iter().find(|b| b.id == id)
    }

    /// `true` when `id` is defined by this set.
    pub fn contains(&self, id: u8) -> bool {
        self.get(id).is_some()
    }

    /// The lowest unused id in `1..=255`, or `None` when the set is full — what
    /// the editor's "Add Biome" uses so ids are never recycled by accident.
    pub fn next_free_id(&self) -> Option<u8> {
        (1..=u8::MAX).find(|&id| !self.contains(id))
    }

    /// Check every rule a biome set must satisfy. See [`BiomeSetError`].
    ///
    /// Called on save (the editor refuses to write an invalid set) **and** on
    /// load through [`migrate`](AssetPayload::migrate) — a hand-edited or
    /// corrupted payload must not become an ambiguous per-sample lookup.
    pub fn validate(&self) -> Result<(), BiomeSetError> {
        if self.biomes.len() > MAX_BIOMES {
            return Err(BiomeSetError::TooMany(self.biomes.len()));
        }
        let mut seen = [false; 256];
        for b in &self.biomes {
            if b.id == UNASSIGNED_BIOME {
                return Err(BiomeSetError::ReservedId);
            }
            if seen[b.id as usize] {
                return Err(BiomeSetError::DuplicateId(b.id));
            }
            seen[b.id as usize] = true;
            if b.name.trim().is_empty() {
                return Err(BiomeSetError::EmptyName(b.id));
            }
            if let Some(layer) = b.splat_layer {
                if (layer as usize) >= SPLAT_LAYERS {
                    return Err(BiomeSetError::BadSplatLayer { id: b.id, layer });
                }
            }
        }
        Ok(())
    }

    /// The asset GUIDs this set references — the `.inf_pcg` graphs its biomes
    /// scatter with. Deduplicated and sorted, so the sidecar's `dependencies`
    /// list is deterministic.
    ///
    /// These are the edges the cook follows *out of* a biome set (the edge *into*
    /// one is `Terrain.biome_set`). The hint fields are deliberately not here:
    /// they are advisory scalars, not references, so a set with a
    /// `structure_hint` naming a palette that does not exist is not a dangling
    /// edge — it is a hint nothing has claimed yet.
    pub fn dependencies(&self) -> Vec<inf_asset::AssetId> {
        let mut out: Vec<inf_asset::AssetId> = self
            .biomes
            .iter()
            .filter_map(|b| b.pcg_graph)
            .map(inf_asset::AssetId)
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// The overlay palette, indexed by biome id: `palette[0]` is
    /// [`UNASSIGNED_BIOME_COLOR`], `palette[id]` is that biome's
    /// [`color`](BiomeDef::color), and every undefined id is the unassigned
    /// colour too (an id painted before its definition was deleted reads as
    /// *unassigned*, which is the honest rendering).
    ///
    /// Length is `1 + the highest defined id` — never a fixed 256 — so an empty
    /// set costs one entry and the renderer pads. Pure, so two builds of one set
    /// give byte-identical uploads.
    pub fn palette(&self) -> Vec<[f32; 4]> {
        let top = self.biomes.iter().map(|b| b.id as usize).max().unwrap_or(0);
        let mut out = vec![UNASSIGNED_BIOME_COLOR; top + 1];
        for b in &self.biomes {
            out[b.id as usize] = b.color;
        }
        out
    }

    /// The [`water_hint`](BiomeDef::water_hint)s, **id-indexed** exactly as
    /// [`palette`](Self::palette) is (P20.4) — `out[id]` is that biome's
    /// still-water level in metres, or `None` for a biome that names none.
    ///
    /// Built here rather than in the editor so the two id-indexed projections a
    /// terrain sends the viewport (colours, water levels) are the same shape and
    /// the same length rule; the reserved id `0` is always `None`, because
    /// *unassigned* is not a biome and cannot carry a hint.
    ///
    /// `f64` at the boundary: the field is `f32` (it is authored through a
    /// number input), and everything downstream of it — a water level, a lake
    /// preview, `WaterBody::level_m` — is `f64` by the world-space rule.
    pub fn water_hints(&self) -> Vec<Option<f64>> {
        let top = self.biomes.iter().map(|b| b.id as usize).max().unwrap_or(0);
        let mut out = vec![None; top + 1];
        for b in &self.biomes {
            out[b.id as usize] = b.water_hint.map(|h| h as f64);
        }
        out
    }
}

impl AssetPayload for BiomeSet {
    const KIND: AssetKind = AssetKind::BiomeSet;
    const SCHEMA_VERSION: u32 = BIOME_SET_SCHEMA_VERSION;

    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current (the default rule), then **validate**: a payload
    /// whose ids are ambiguous or reserved would silently mis-resolve every
    /// per-sample lookup, so it fails at the door rather than downstream.
    fn migrate(self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        self.validate()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid biome set: {e}")))?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_starter_set_is_valid_and_round_trips() {
        let set = BiomeSet::starter();
        set.validate().unwrap();
        let bytes = inf_asset::encode(&set).unwrap();
        let back: BiomeSet = inf_asset::decode(&bytes).unwrap();
        assert_eq!(set, back);
        // Deterministic: re-encoding an unchanged payload is byte-identical.
        assert_eq!(bytes, inf_asset::encode(&back).unwrap());
    }

    /// **Id 0 is reserved.** The whole sparse storage rests on it, so a set that
    /// defines it must not decode.
    #[test]
    fn id_zero_is_rejected_on_validate_and_on_decode() {
        let mut set = BiomeSet::starter();
        set.biomes.push(BiomeDef::new(UNASSIGNED_BIOME, "Nothing"));
        assert_eq!(set.validate(), Err(BiomeSetError::ReservedId));

        // …and it cannot sneak in through a hand-written payload either: the
        // encode side has no validation, so this is exactly the corrupt-file case.
        let bytes = inf_asset::encode(&set).unwrap();
        let err = inf_asset::decode::<BiomeSet>(&bytes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("reserved"), "unhelpful rejection: {err}");
    }

    #[test]
    fn duplicate_ids_empty_names_and_bad_layers_are_rejected() {
        let mut dup = BiomeSet::starter();
        dup.biomes.push(BiomeDef::new(2, "Also two"));
        assert_eq!(dup.validate(), Err(BiomeSetError::DuplicateId(2)));

        let mut blank = BiomeSet::starter();
        blank.biomes[0].name = "   ".into();
        assert_eq!(blank.validate(), Err(BiomeSetError::EmptyName(1)));

        let mut layer = BiomeSet::starter();
        layer.biomes[0].splat_layer = Some(SPLAT_LAYERS as u8);
        assert_eq!(
            layer.validate(),
            Err(BiomeSetError::BadSplatLayer {
                id: 1,
                layer: SPLAT_LAYERS as u8
            })
        );
    }

    #[test]
    fn a_full_set_holds_255_biomes_and_no_more() {
        let mut set = BiomeSet::new("Full");
        for id in 1..=u8::MAX {
            set.biomes.push(BiomeDef::new(id, format!("b{id}")));
        }
        assert_eq!(set.biomes.len(), MAX_BIOMES);
        set.validate().unwrap();
        assert_eq!(set.next_free_id(), None, "a full set has no free id");

        // A 256th entry can only be a duplicate (ids are u8 and 0 is reserved),
        // so both guards are live; the count guard is what catches it first.
        set.biomes.push(BiomeDef::new(1, "overflow"));
        assert_eq!(set.validate(), Err(BiomeSetError::TooMany(256)));
    }

    #[test]
    fn next_free_id_skips_gaps_and_never_returns_zero() {
        let mut set = BiomeSet::new("Gaps");
        set.biomes.push(BiomeDef::new(1, "one"));
        set.biomes.push(BiomeDef::new(3, "three"));
        assert_eq!(set.next_free_id(), Some(2));
        set.biomes.push(BiomeDef::new(2, "two"));
        assert_eq!(set.next_free_id(), Some(4));
    }

    /// The palette is id-indexed, pads undefined ids with the unassigned colour,
    /// and is a pure function of the set.
    #[test]
    fn palette_is_id_indexed_and_pure() {
        let mut set = BiomeSet::new("P");
        set.biomes.push(BiomeDef {
            color: [1.0, 0.0, 0.0, 1.0],
            ..BiomeDef::new(5, "five")
        });
        let p = set.palette();
        assert_eq!(p.len(), 6, "1 + highest id");
        assert_eq!(p[0], UNASSIGNED_BIOME_COLOR, "id 0 is always unassigned");
        assert_eq!(
            p[3], UNASSIGNED_BIOME_COLOR,
            "an undefined id reads neutral"
        );
        assert_eq!(p[5], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(p, set.palette(), "palette is pure");

        // An empty set still yields the one unassigned entry (never an empty
        // upload the shader would index out of).
        assert_eq!(BiomeSet::new("e").palette(), vec![UNASSIGNED_BIOME_COLOR]);
    }

    /// The P20.4 twin: `water_hints` is the same shape as `palette`, id-indexed,
    /// pure, and the reserved id 0 never carries one.
    #[test]
    fn water_hints_are_id_indexed_and_the_reserved_id_has_none() {
        let mut set = BiomeSet::new("W");
        set.biomes.push(BiomeDef {
            water_hint: Some(12.5),
            ..BiomeDef::new(2, "lake")
        });
        set.biomes.push(BiomeDef::new(4, "dry"));
        let h = set.water_hints();
        assert_eq!(h.len(), 5, "1 + highest id, like the palette");
        assert_eq!(h.len(), set.palette().len(), "the two projections agree");
        assert_eq!(h[0], None, "id 0 is unassigned and can never be defined");
        assert_eq!(h[1], None, "an undefined id has no hint");
        assert_eq!(h[2], Some(12.5));
        assert_eq!(h[4], None, "a biome that names no hint reads None");
        assert_eq!(h, set.water_hints(), "water_hints is pure");
        // An empty set yields the single unassigned slot, never an empty vec a
        // caller would index out of.
        assert_eq!(BiomeSet::new("e").water_hints(), vec![None]);
    }

    #[test]
    fn a_newer_schema_is_refused() {
        let mut set = BiomeSet::starter();
        set.schema_version = BIOME_SET_SCHEMA_VERSION + 1;
        let bytes = inf_asset::encode(&set).unwrap();
        assert!(matches!(
            inf_asset::decode::<BiomeSet>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
    }

    #[test]
    fn the_kind_is_wired() {
        assert_eq!(BiomeSet::KIND, AssetKind::BiomeSet);
        assert_eq!(AssetKind::BiomeSet.extension(), Some("inf_biomes"));
        assert_eq!(
            AssetKind::from_extension("inf_biomes"),
            AssetKind::BiomeSet,
            "the extension must round-trip"
        );
    }
}
