//! The **per-sample terrain layers** seam (P19.3) — the second half of the
//! terrain contract, beside [`HeightProvider`](crate::height::HeightProvider).
//!
//! Scattering has always asked the terrain two questions: *how high* and *which
//! way does it face*. P19.1 and P19.2 gave a terrain two more layers with an
//! opinion about where things belong — the erosion **data maps** (flow /
//! deposition / wear) and the painted **biome ids** — and P19.3's mask samplers
//! read them. They do not fit [`HeightProvider`]: a purely procedural height
//! function has no data maps and no biomes, and forcing it to answer would make
//! every `FnHeight` in the codebase lie.
//!
//! So they get their own trait, and the default answer is *"I have none"*:
//!
//! ```text
//!   HeightProvider   →  height / normal        (every terrain, procedural or authored)
//!   TerrainFields    →  data_map / biome_id    (an AUTHORED terrain; NoFields otherwise)
//! ```
//!
//! [`NoFields`] is the honest empty implementation — every query is `None`, which
//! the samplers read as *off-terrain* and score `0`. It is what
//! [`evaluate`](crate::rules::evaluate) passes when a caller supplies no terrain
//! data, so a document that uses a mask sampler without a terrain places nothing
//! rather than silently placing everywhere.
//!
//! # Why `TerrainData` implements it directly (and `TerrainHeight` does not)
//!
//! [`TerrainHeight`](crate::height::TerrainHeight) is generic over
//! `inf_terrain::HeightSource`, and a `HeightSource` is exactly the thing that
//! may be a bare procedural field — it has no layers to offer. The layer trait is
//! therefore implemented on the concrete
//! [`inf_terrain::TerrainData`] (a local trait on a foreign type: allowed, and no
//! coherence conflict, because there is no blanket impl over a generic bound
//! here — the mistake `height.rs` documents at length).

use inf_terrain::{DataMapKind, TerrainData};

/// Read-only access to a terrain's **per-sample layers**: the P19.1 erosion data
/// maps and the P19.2 painted biome ids.
///
/// `Send + Sync` for the same reason [`HeightProvider`](crate::height::HeightProvider)
/// is: a density field built over these is evaluated in parallel across scatter
/// cells.
///
/// Every method has a default that answers `None` / "nothing here", so a source
/// that only has *some* of the layers implements only those.
pub trait TerrainFields: Send + Sync {
    /// The **raw** value of one erosion data map at world `(x, z)`, in that
    /// channel's own SI unit ([`DataMapKind`]) — never normalized, because
    /// normalization is a view the reader chooses (the P19.1 doctrine). `None`
    /// outside the terrain's authored extent.
    fn data_map(&self, _kind: DataMapKind, _x: f64, _z: f64) -> Option<f64> {
        None
    }

    /// The painted biome id at world `(x, z)`, or
    /// [`UNASSIGNED_BIOME`](inf_terrain::UNASSIGNED_BIOME) (`0`) where nothing has
    /// been painted. `None` outside the terrain's authored extent — which the
    /// biome mask reads as *outside*, exactly as it reads a different id.
    fn biome_id(&self, _x: f64, _z: f64) -> Option<u8> {
        None
    }

    /// The terrain's sample spacing in **metres** — the lattice step the biome
    /// mask's feather search walks (see
    /// [`BiomeMask`](crate::sampler::BiomeMask)). Sources with no lattice return
    /// the `1.0` default, which is a spacing, not a claim.
    fn sample_spacing(&self) -> f64 {
        1.0
    }
}

/// A reference to a field source is itself a field source (so a `&dyn
/// TerrainFields` satisfies the trait and one source can back a whole sampler
/// tree).
impl<T: TerrainFields + ?Sized> TerrainFields for &T {
    fn data_map(&self, kind: DataMapKind, x: f64, z: f64) -> Option<f64> {
        (**self).data_map(kind, x, z)
    }
    fn biome_id(&self, x: f64, z: f64) -> Option<u8> {
        (**self).biome_id(x, z)
    }
    fn sample_spacing(&self) -> f64 {
        (**self).sample_spacing()
    }
}

/// The empty source: **no** data maps and **no** biomes, anywhere.
///
/// Not a stub — it is the correct answer for a purely procedural terrain, and it
/// is what makes a mask sampler fail *closed* (score `0`, place nothing) rather
/// than open when a document is evaluated without terrain layers.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoFields;

impl TerrainFields for NoFields {}

/// An authored terrain answers every layer query.
///
/// Both reads go through `TerrainData`'s own nearest-sample resolution, which is
/// shared with `biome_at` — so the PCG mask and the biome overlay can never
/// disagree about which tile owns a seam.
impl TerrainFields for TerrainData {
    fn data_map(&self, kind: DataMapKind, x: f64, z: f64) -> Option<f64> {
        self.data_map_at(kind, glam::DVec2::new(x, z))
            .map(|v| v as f64)
    }

    fn biome_id(&self, x: f64, z: f64) -> Option<u8> {
        self.biome_at(glam::DVec2::new(x, z))
    }

    fn sample_spacing(&self) -> f64 {
        self.meters_per_sample()
    }
}

/// A terrain sampled through a **world-space offset** — the shape both evaluation
/// call sites (the editor command and the player's load-time pass) actually need,
/// because a `Terrain` component's tiles are authored in terrain-local space and
/// the entity carries a world transform.
///
/// It exists here rather than at each call site so the editor and the player
/// cannot drift about what "the terrain is at `origin`" means — the parity the
/// P19.3 gate rests on. `origin` is the terrain entity's world translation; a
/// query at world `(x, z)` reads terrain-local `(x − origin.x, z − origin.z)`.
pub struct OffsetTerrain<'a> {
    pub data: &'a TerrainData,
    pub origin: glam::DVec3,
}

impl<'a> OffsetTerrain<'a> {
    /// Wrap `data` anchored at world `origin`.
    pub fn new(data: &'a TerrainData, origin: glam::DVec3) -> Self {
        Self { data, origin }
    }

    /// The world-space height query this terrain answers, as a plain closure —
    /// the exact expression both evaluation paths build their
    /// [`FnHeight`](crate::height::FnHeight) from.
    #[inline]
    pub fn height_at(&self, x: f64, z: f64) -> Option<f64> {
        self.data
            .height_at(glam::DVec2::new(x - self.origin.x, z - self.origin.z))
            .map(|h| h + self.origin.y)
    }
}

impl TerrainFields for OffsetTerrain<'_> {
    fn data_map(&self, kind: DataMapKind, x: f64, z: f64) -> Option<f64> {
        self.data
            .data_map_at(kind, glam::DVec2::new(x - self.origin.x, z - self.origin.z))
            .map(|v| v as f64)
    }

    fn biome_id(&self, x: f64, z: f64) -> Option<u8> {
        self.data
            .biome_at(glam::DVec2::new(x - self.origin.x, z - self.origin.z))
    }

    fn sample_spacing(&self) -> f64 {
        self.data.meters_per_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn painted() -> TerrainData {
        let mut t = TerrainData::new(5, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        let res = t.tile_resolution();
        let tile = t.get_tile_mut((0, 0)).unwrap();
        tile.set_biome_sample(res, 2, 2, 7);
        tile.set_map_texel(res, 2, 2, [3.5, 1.0, 0.25]);
        t
    }

    /// The empty source fails **closed**: every query is `None`, which the mask
    /// samplers score as `0`.
    #[test]
    fn no_fields_answers_nothing() {
        let n = NoFields;
        assert_eq!(n.data_map(DataMapKind::Flow, 0.0, 0.0), None);
        assert_eq!(n.biome_id(0.0, 0.0), None);
        assert_eq!(n.sample_spacing(), 1.0);
    }

    #[test]
    fn terrain_data_answers_both_layers() {
        let t = painted();
        assert_eq!(t.biome_id(2.0, 2.0), Some(7));
        assert_eq!(t.biome_id(0.0, 0.0), Some(0), "unpainted is id 0, not None");
        assert_eq!(t.biome_id(1e6, 1e6), None, "off-extent is None");
        assert_eq!(t.data_map(DataMapKind::Flow, 2.0, 2.0), Some(3.5));
        assert_eq!(t.data_map(DataMapKind::Wear, 2.0, 2.0), Some(0.25));
        assert_eq!(t.sample_spacing(), 1.0);
        // The reference impl forwards.
        let r: &dyn TerrainFields = &t;
        assert_eq!(TerrainFields::biome_id(&r, 2.0, 2.0), Some(7));
    }

    /// The offset wrapper shifts **every** layer and the height by the same
    /// origin — the property that keeps the editor and the player in parity.
    #[test]
    fn offset_terrain_shifts_every_layer_by_the_same_origin() {
        let t = painted();
        let o = DVec3::new(100.0, 5.0, -50.0);
        let f = OffsetTerrain::new(&t, o);
        assert_eq!(f.biome_id(102.0, -48.0), Some(7));
        assert_eq!(f.data_map(DataMapKind::Flow, 102.0, -48.0), Some(3.5));
        assert_eq!(
            f.height_at(102.0, -48.0),
            Some(5.0),
            "height rides origin.y"
        );
        // The un-shifted position is off the (shifted) terrain entirely.
        assert_eq!(f.biome_id(2.0, 2.0), None);
        assert_eq!(f.sample_spacing(), 1.0);
    }
}
