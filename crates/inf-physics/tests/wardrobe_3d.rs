//! **THE WARDROBE** (wave EMS3) — the mirrored GUID, and the press that changes
//! what a wanted man looks like.
//!
//! This file lives in `inf-physics` for one reason, and it is the whole of the
//! first arm: `inf-pcg` is a **dev**-dependency here and is not a dependency of
//! `inf-ecs` at all, so this is the only place in the tree that can hold
//! `inf_ecs::wardrobe::WARDROBE_MESH_GUID` up against the
//! `inf_pcg::building::module_mesh_guid` it mirrors. A mirror nothing compares
//! is the triplication hazard `ScatteredSolid`'s own round-trip test exists to
//! refuse.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::band::SimBand;
use inf_ecs::components::{PcgVolume, ScatteredInstance, Transform};
use inf_ecs::crowd::{appearance_of, CROWD_LOOKS};
use inf_ecs::interact::InteractVerb;
use inf_ecs::math::Vec2d;
use inf_ecs::wardrobe::{self, WARDROBE_MESH_GUID, WARDROBE_MODULE};
use inf_ecs::EcsWorld;

const BLOCK: Uuid = Uuid::from_u128(0x0E54_0001);
const HERO: Uuid = Uuid::from_u128(0x0E54_0002);

/// **THE MIRROR IS THE ORIGINAL**, both ways round.
///
/// The GUID and the family it comes from, so a rename of `ModuleShape::Wardrobe`
/// or a change to `MODULE_MESH_SALT` fails here instead of silently making every
/// wardrobe in the world unpressable — which is a defect with **no symptom at
/// all** short of a player walking up to one and getting no prompt.
#[test]
fn the_mirrored_wardrobe_guid_is_the_one_the_palette_draws() {
    let from_pcg = inf_pcg::building::modules::module_guid_for(WARDROBE_MODULE)
        .expect("the palettes still declare a `Wardrobe`");
    println!("`{WARDROBE_MODULE}` draws under {from_pcg}");
    assert_eq!(
        WARDROBE_MESH_GUID, from_pcg,
        "`inf_ecs::wardrobe::WARDROBE_MESH_GUID` no longer mirrors \
         `inf_pcg::building::modules::module_guid_for({WARDROBE_MODULE:?})` — every \
         wardrobe in every bedroom in the world just stopped offering a prompt, \
         and nothing else in this tree would have said so"
    );
    // …and it is its OWN family. This is the assertion the whole clause rests
    // on: before wave EMS3 nine module names shared one carcass GUID, so keying
    // on it would have offered a change of clothes at a shop counter, a
    // reception desk and a bathroom basin.
    for other in [
        "Cabinet",
        "Counter",
        "FrontDesk",
        "Basin",
        "Shelf",
        "Locker",
    ] {
        assert_ne!(
            inf_pcg::building::modules::module_guid_for(other),
            Some(WARDROBE_MESH_GUID),
            "`{other}` still draws under the wardrobe's GUID"
        );
    }
}

/// A block with `n` wardrobes in a row, `2 m` apart down `+X`.
fn town(n: usize) -> EcsWorld {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(BLOCK, "block", None);
    let mut v = PcgVolume {
        extent: Vec2d::new(40.0, 40.0),
        ..Default::default()
    };
    let piece = |at: DVec3, mesh: Option<Uuid>| ScatteredInstance {
        position: at,
        rotation: glam::DQuat::IDENTITY,
        scale: 1.0,
        kind: 0,
        mesh,
        extent: None,
        glow: 0.0,
        surface: Default::default(),
    };
    let mut instances: Vec<ScatteredInstance> = Vec::new();
    for i in 0..n {
        instances.push(piece(
            DVec3::new(i as f64 * 2.0, 0.0, 0.0),
            Some(WARDROBE_MESH_GUID),
        ));
        // A carcass beside each one — the module family the wardrobe was split
        // out of — so every arm below is measured against a world that contains
        // the thing it must NOT offer a prompt at.
        instances.push(piece(
            DVec3::new(i as f64 * 2.0, 0.0, 1.0),
            inf_pcg::building::modules::module_guid_for("Counter"),
        ));
    }
    v.evaluated = instances;
    world
        .world_mut()
        .entity_mut(e)
        .insert((Transform::IDENTITY, v));
    let h = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(h).insert((
        Transform::from_translation(DVec3::ZERO),
        inf_ecs::components::StreamingSource { radius_m: 512.0 },
    ));
    world.mark_dirty();
    world.propagate();
    world
}

/// **A WARDROBE OFFERS A CHANGE AND A COUNTER DOES NOT**, and only the one you
/// are standing at.
#[test]
fn only_a_wardrobe_in_reach_offers_a_change_of_clothes() {
    let world = town(6);
    let band = SimBand::unbounded();
    // Standing at the first one.
    let near = wardrobe::candidates(&world, &band, DVec3::new(0.0, 0.0, 0.0));
    println!(
        "6 wardrobes and 6 counters, standing at the first: {} candidate(s)",
        near.len()
    );
    assert!(!near.is_empty(), "no wardrobe was offered at all");
    assert!(
        near.iter().all(|c| c.verb == InteractVerb::Change),
        "something that is not a change of clothes got into the list"
    );
    assert!(
        near.len() < 6,
        "the reach filter admitted every wardrobe in the block"
    );
    // The prompt reads as the mechanic rather than as the furniture.
    let text = inf_ecs::interact::prompt_text(near[0].verb, &near[0].label, "E");
    println!("the prompt reads {text:?}");
    assert!(text.contains("Change"), "the prompt hides the verb");
    assert!(text.contains("clothes"), "the prompt names the cupboard");
    // Every guid it offers is one the world agrees is a wardrobe, and the
    // counters' are not.
    let all = wardrobe::wardrobes(&world);
    assert_eq!(all.len(), 6);
    for c in &near {
        assert!(all.contains(&c.guid));
        assert!(wardrobe::is_wardrobe(&world, c.guid));
    }
    assert!(
        !wardrobe::is_wardrobe(&world, Uuid::from_u128(0xdead_beef)),
        "a guid nothing placed was called a wardrobe"
    );
    // …and a hundred metres away there is nothing to press.
    assert!(
        wardrobe::candidates(&world, &band, DVec3::new(1000.0, 0.0, 0.0)).is_empty(),
        "a wardrobe was offered from a kilometre away"
    );
}

/// **PRESSING IT CHANGES THE DESCRIPTION, EVERY TIME** — the property the
/// mandate's evasion route needs, and the one a random draw would not have.
#[test]
fn a_press_always_changes_what_the_police_would_be_told() {
    let mut world = town(1);
    let mut seen: Vec<u8> = vec![appearance_of(&world, HERO).outfit];
    let mut digests: Vec<u64> = vec![inf_ecs::witness::look_digest(&world, HERO)];
    for _ in 0..CROWD_LOOKS.len() {
        assert!(
            wardrobe::change_clothes(&mut world, HERO),
            "a press did nothing — a wardrobe that can refuse is a broken control"
        );
        let now = appearance_of(&world, HERO);
        assert_ne!(
            Some(&now.outfit),
            seen.last(),
            "the outfit did not move at all"
        );
        seen.push(now.outfit);
        digests.push(inf_ecs::witness::look_digest(&world, HERO));
    }
    println!("pressing E eight times walks the rail: {seen:?}");
    // The whole rail, and back to where it started — so a player can get back
    // into the coat they like.
    assert_eq!(seen[0], seen[CROWD_LOOKS.len()], "the cycle did not close");
    let mut distinct = seen[..CROWD_LOOKS.len()].to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        CROWD_LOOKS.len(),
        "the cycle skipped part of the rail"
    );
    // And every step of it is a different description, which is what the police
    // are actually reading.
    let mut d = digests[..CROWD_LOOKS.len()].to_vec();
    d.sort_unstable();
    d.dedup();
    assert_eq!(d.len(), CROWD_LOOKS.len(), "two outfits describe alike");
    // The appearance is a RESOURCE, so a Simulate session's twin wipes it.
    inf_ecs::crowd::clear_appearance(&mut world);
    assert_eq!(
        appearance_of(&world, HERO).outfit,
        inf_ecs::crowd::derived_outfit(HERO),
        "clearing the appearance did not return the derived draw"
    );
}
