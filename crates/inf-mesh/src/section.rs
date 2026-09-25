//! **The drawn SECTIONS of a multi-material rigid mesh** (wave VEH3f.2a).
//!
//! # The gap this closes, measured
//!
//! A rigid `MeshRef.asset` draws through the meshlet path as ONE instance with
//! ONE surface: the entity's own `Material` and the virtual-texture set its
//! `asset` resolves to. A mesh with twenty-one material slots — the calibration
//! sedan: paint, glass, leather, alcantara, chrome, rubber, a speedometer —
//! therefore drew in one flat colour, and so did every construction machine the
//! bridge imported in wave VEH3f (its body hung the row's paint over a mesh
//! whose textures were imported and never sampled). A skinned body solved the
//! same problem at CHAR1a.3 with per-slot sections; a rigid mesh had nothing.
//!
//! # The arrangement: computed identities, no index
//!
//! An importer that wants a mesh drawn per slot writes, beside the mesh `A`:
//!
//! * one single-slot `.inf_mesh` per slot at [`section_mesh_id`]`(A, slot)` —
//!   an ordinary rigid mesh, so the cook derives its meshlet DAG through the
//!   door every mesh takes;
//! * the slot's material at [`section_material_id`]`(A, slot)`, or NOTHING for
//!   a slot that should wear the entity's own surface (a car's paint, whose
//!   colour is the roster row's and not the pack's).
//!
//! Both hosts find a mesh's sections by COMPUTING the ids (the
//! `derived_vmesh_id` precedent — no side table to ship or drift) and draw one
//! instance per section; a mesh with no sections draws exactly as it always
//! did, which is every mesh in this repository and every mesh on CI.

use inf_asset::AssetId;

/// The most sections one mesh may have — the slot indices a host probes. The
/// widest pack this bridge imports has 25 slots on one body.
pub const MAX_SECTIONS: u32 = 48;

/// `"INFSECTIONMESH00"` in ASCII.
const SECTION_MESH_SALT: u128 = 0x494e_4653_4543_5449_4f4e_4d45_5348_3030;
/// `"INFSECTIONMATL00"` in ASCII.
const SECTION_MATERIAL_SALT: u128 = 0x494e_4653_4543_5449_4f4e_4d41_544c_3030;

fn mix(mesh: AssetId, slot: u32, salt: u128) -> AssetId {
    let mut x = mesh.uuid().as_u128() ^ salt;
    x = x.rotate_left(29)
        ^ (u128::from(slot) + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835);
    x ^= x >> 67;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    x ^= x >> 59;
    AssetId(uuid::Builder::from_random_bytes(x.to_be_bytes()).into_uuid())
}

/// **The `.inf_mesh` that draws slot `slot` of `mesh`**, a pure function of the
/// two. Distinct from [`section_material_id`] for every input (different salts).
pub fn section_mesh_id(mesh: AssetId, slot: u32) -> AssetId {
    mix(mesh, slot, SECTION_MESH_SALT)
}

/// **The `.inf_mat` slot `slot` of `mesh` is drawn with**, when it has one.
pub fn section_material_id(mesh: AssetId, slot: u32) -> AssetId {
    mix(mesh, slot, SECTION_MATERIAL_SALT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ids are pure, distinct across slots and kinds, and never the
    /// parent's own — a section that collided with its parent would draw the
    /// parent twice.
    #[test]
    fn section_ids_are_pure_and_distinct() {
        let a = AssetId(uuid::Uuid::from_u128(
            0x1234_5678_9abc_def0_0fed_cba9_8765_4321,
        ));
        let mut seen = std::collections::BTreeSet::new();
        for s in 0..MAX_SECTIONS {
            assert_eq!(section_mesh_id(a, s), section_mesh_id(a, s));
            assert!(seen.insert(section_mesh_id(a, s)));
            assert!(seen.insert(section_material_id(a, s)));
        }
        assert!(!seen.contains(&a));
    }
}
