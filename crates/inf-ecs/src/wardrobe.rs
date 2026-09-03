//! **THE WARDROBE** (wave EMS3) — the one place in this engine where a player
//! can change what they look like, and therefore the one place a wanted level
//! can be walked away from.
//!
//! # A grammar module becomes an interaction, and the pattern is the doorway's
//!
//! `inf-pcg` has placed a `Wardrobe` in four archetypes' bedrooms since P19
//! (`Apartment`, `House`, `Estate`, `Hotel`) and it has been furniture the whole
//! time: a drawn instance and a solid collider, with nothing to press. Making it
//! pressable is exactly what [`crate::door`] did for a `PcgDoorway` — a
//! synthetic guid carved out of the scene's own guid space, a band filter before
//! anything is allocated, and an [`InteractCandidate`] unioned into the one
//! resolution site. **No entity is spawned and no component is inserted**, so
//! this reaches no schema, no save file and no `RuntimeEntityGen` slot.
//!
//! # What identifies one, and why a family had to be split for it
//!
//! The only per-instance identity a placed module carries into the world is
//! `ScatteredInstance::mesh`, which is its shape family's GUID — and until this
//! wave `Cabinet`, `Locker`, `Wardrobe`, `Units`, `Shelf`, `Rack`, `Counter`,
//! `Basin` and `FrontDesk` all resolved to one `ModuleShape::Carcass`. Keying on
//! that would have offered a change of clothes at a shop counter and a
//! reception desk, in every building in the world. So EMS3 gave the wardrobe its
//! own family (`inf_pcg::building::modules::ModuleShape::Wardrobe`), which is the
//! smallest honest way to make it findable.
//!
//! # THE GUID IS MIRRORED, on `ScatteredSolid`'s own terms
//!
//! `inf-pcg` is a **dev**-dependency of `inf-physics` and is not a dependency of
//! this crate at all — that direction is stated in both manifests and is not
//! something this wave gets to move for one constant. So
//! [`WARDROBE_MESH_GUID`] is a mirror of
//! `inf_pcg::building::modules::module_mesh_guid(ModuleShape::Wardrobe)`, exactly as
//! [`crate::components::ScatteredSolid`] mirrors `inf_pcg::PcgCollider` and
//! [`crate::components::DoorwaySlot`] mirrors `PcgDoorway`. Two hand-written
//! copies of one value is a drift hazard, so the copy is **pinned against the
//! original** by `inf_physics`'s `wardrobe_3d`, which has the dev-dependency
//! that can see both.
//!
//! # THE HONEST BOUND: this changes a palette swap, not a garment
//!
//! What a press changes is [`crate::crowd::Appearance`] — the index into
//! `CROWD_LOOKS` both projectors already tint a body with. It does **not** bind
//! a different `.inf_cloth` on the character's `ClothSim`, and that is a
//! refusal rather than an omission: [`crate::cloth::step_cloth_simulation`]
//! re-seeds on an asset change and has waited for a caller since P24, but a
//! caller needs *content* — a set of garment assets a wardrobe could offer — and
//! this engine has none. Binding `ClothSim::asset` to a guid nothing resolves
//! would make a coat **vanish** (rule 2: an unresolvable garment is skipped, and
//! rule 4 rebuilds the store from the survivors), which is worse than not
//! offering it. Carried by name.

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use crate::band::SimBand;
use crate::components::PcgVolume;
use crate::interact::{InteractCandidate, InteractVerb};
use crate::world::EcsWorld;

/// **The mesh GUID `ModuleShape::Wardrobe` draws under** — a mirror of
/// `inf_pcg::building::modules::module_mesh_guid`'s answer for the family, on
/// [`crate::components::ScatteredSolid`]'s own terms.
///
/// Pinned against the original by `inf_physics::wardrobe_3d`, which has the
/// dev-dependency on `inf-pcg` that this crate deliberately does not.
pub const WARDROBE_MESH_GUID: Uuid = Uuid::from_u128(0x4a23_5cc6_ecfc_ae13_ae85_742e_4dc9_bd78);

/// The module name the mirror above is derived from — the string
/// `inf_pcg::building::modules::shape_of` maps, kept beside the guid so the pin can
/// re-derive it.
pub const WARDROBE_MODULE: &str = "Wardrobe";

/// The salt that carves a PCG volume's wardrobes out of the scene's own GUID
/// space. Its own constant, so a wardrobe can never alias a doorway, a
/// structure, a shell or a leaf — [`crate::door`]'s rule with a different
/// number.
const PCG_WARDROBE_SALT: u128 = 0x5741_5244_524f_4245_454d_5333_0000_0001;

/// The synthetic identity of the wardrobe at `index` inside the volume on entity
/// `volume`.
pub fn wardrobe_guid(volume: Uuid, index: usize) -> Uuid {
    let mut x = volume.as_u128() ^ PCG_WARDROBE_SALT;
    x ^= (index as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5cec_c5c3);
    x = x.rotate_left(37) ^ x.wrapping_mul(0xff51_afd7_ed55_8ccd_c4ce_b9fe_1a85_ec53);
    Uuid::from_u128(x)
}

/// What the prompt calls it — the **noun**, so the sentence reads *"Change
/// clothes"* rather than naming the cupboard. What a player is choosing is the
/// outfit.
pub const WARDROBE_LABEL: &str = "clothes";

/// How close a character has to be to open one, metres.
///
/// `crate::interact::DEFAULT_REACH_M`, reused by name: a wardrobe is an
/// ordinary piece of authored furniture as far as reach is concerned, and a
/// second number here would be a second opinion about how long an arm is.
pub const WARDROBE_REACH_M: f64 = crate::interact::DEFAULT_REACH_M;

/// How wide a cone one is offered within, degrees — *"roughly in front of me"*.
pub const WARDROBE_VIEW_CONE_DEG: f64 = crate::interact::DEFAULT_VIEW_CONE_DEG;

/// **Every wardrobe near enough to matter**, in `Guid` order.
///
/// # The cost, and the two filters that bound it
///
/// A volume's `evaluated` list is every instance its grammar placed — a city
/// block is thousands of them and a city is millions — and this runs on **every
/// frame**, because the interaction prompt is asked once a frame through the
/// same resolver a press goes through (the I5 rule: what the player is told and
/// what the press does cannot come apart).
///
/// So there are two gates, in order, and neither allocates:
///
/// 1. **the volume**, against the sim band — a block the simulation is not
///    running is not a block anybody is standing in;
/// 2. **the instance**, against `feet` and [`WARDROBE_REACH_M`] — a squared
///    distance and a mesh-guid compare per instance, which is the same walk both
///    projectors already make over the same list every frame to build their
///    scatter batches.
///
/// The result is at most a handful of candidates and usually none: a character
/// standing in a street is inside no bedroom.
pub fn candidates(world: &EcsWorld, band: &SimBand, feet: DVec3) -> Vec<InteractCandidate> {
    let reach2 = WARDROBE_REACH_M * WARDROBE_REACH_M;
    let mut out: Vec<InteractCandidate> = Vec::new();
    let w = world.world();
    let Some(mut q) = w.try_query::<(&crate::components::Guid, &PcgVolume)>() else {
        return out;
    };
    let mut volumes: Vec<(Uuid, &PcgVolume)> = q.iter(w).map(|(g, v)| (g.0, v)).collect();
    volumes.sort_by_key(|(g, _)| *g);
    for (volume, v) in volumes {
        for (i, inst) in v.evaluated.iter().enumerate() {
            if inst.mesh != Some(WARDROBE_MESH_GUID) {
                continue;
            }
            if !inst.position.is_finite() {
                continue;
            }
            // The band, per INSTANCE rather than per volume: a volume is a whole
            // city block and its own centre says nothing about whether the room
            // a character is standing in is resident. `Tier::Out` is the one
            // answer that refuses — the band's own vocabulary, so a wardrobe
            // becomes pressable at exactly the distance the colliders around it
            // become solid.
            if band.tier(inst.position, DVec3::splat(0.5), glam::DQuat::IDENTITY)
                == inf_math::Tier::Out
            {
                continue;
            }
            // The prompt is measured from the piece's own middle, lifted to
            // about waist height, because that is where a person reaches — the
            // same argument `door::prompt_position` makes about a leaf's opening
            // rather than its hinge.
            let at = inst.position + DVec3::Y * 0.9;
            if (at - feet).length_squared() > reach2 * 4.0 {
                continue;
            }
            out.push(InteractCandidate {
                guid: wardrobe_guid(volume, i),
                verb: InteractVerb::Change,
                label: WARDROBE_LABEL.to_string(),
                position: at,
                range_m: WARDROBE_REACH_M,
                view_cone_deg: WARDROBE_VIEW_CONE_DEG,
                // A door handle, because that is what is on the front of it and
                // because it is the affordance a rig's grip catalogue already
                // names. A rig with no fingers just does not close a hand, which
                // is `door::candidates`' own honest answer.
                grip: Some(inf_anim::GRIP_HANDLE.to_string()),
            });
        }
    }
    out.sort_by_key(|c| c.guid);
    out
}

/// **PUT ON SOMETHING ELSE** — the one door a change of clothes goes through.
///
/// # The next outfit is the next one, and that is deliberate
///
/// It cycles: `(current + 1) % CROWD_LOOKS.len()`. The alternatives were a draw
/// on the wardrobe's own guid — which can answer the outfit you are already
/// wearing, so a press does nothing and reads as a broken control — and a
/// picker UI, which is a panel this wave is not building. A cycle **always
/// changes the description**, which is the property the mandate's evasion route
/// needs, and pressing E repeatedly walks the whole rail.
///
/// Returns whether anything changed, so a caller can count presses that did
/// something. `false` is reachable only if the appearance was somehow already
/// the next one, which the cycle makes impossible — it is
/// `set_appearance`'s contract showing through rather than a case anybody hits.
pub fn change_clothes(world: &mut EcsWorld, actor: Uuid) -> bool {
    let now = crate::crowd::appearance_of(world, actor);
    let next = crate::crowd::Appearance {
        outfit: (now.outfit as usize + 1)
            .rem_euclid(crate::crowd::CROWD_LOOKS.len())
            .try_into()
            .unwrap_or(0),
    };
    crate::crowd::set_appearance(world, actor, next)
}

/// Whether this guid is a wardrobe the world holds — the press's own check, so
/// a `Change` hit on something that is not one does nothing rather than
/// dressing somebody out of a filing cabinet.
///
/// Answered by re-deriving the guid rather than by storing a set, which is this
/// tree's own rule about derived identities: `wardrobe_guid` is a pure function,
/// so asking it again is cheaper than keeping a second copy that can drift.
pub fn is_wardrobe(world: &EcsWorld, guid: Uuid) -> bool {
    let w = world.world();
    let Some(mut q) = w.try_query::<(&crate::components::Guid, &PcgVolume)>() else {
        return false;
    };
    for (g, v) in q.iter(w) {
        for (i, inst) in v.evaluated.iter().enumerate() {
            if inst.mesh == Some(WARDROBE_MESH_GUID) && wardrobe_guid(g.0, i) == guid {
                return true;
            }
        }
    }
    false
}

/// Every wardrobe guid the world holds, in `Guid` order — a diagnostic and a
/// gate's own reader, for the same reason `door::volume_doorways` exists.
pub fn wardrobes(world: &EcsWorld) -> BTreeSet<Uuid> {
    let mut out = BTreeSet::new();
    let w = world.world();
    let Some(mut q) = w.try_query::<(&crate::components::Guid, &PcgVolume)>() else {
        return out;
    };
    for (g, v) in q.iter(w) {
        for (i, inst) in v.evaluated.iter().enumerate() {
            if inst.mesh == Some(WARDROBE_MESH_GUID) {
                out.insert(wardrobe_guid(g.0, i));
            }
        }
    }
    out
}
