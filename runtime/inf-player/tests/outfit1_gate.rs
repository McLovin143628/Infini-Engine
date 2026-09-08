//! **WAVE OUTFIT1 — CLOTHES, HAIR AND WHAT A WEARABLE COSTS.**
//!
//! The sentence this wave exists for is a photograph, not a feature: every frame
//! this campaign has taken since CHAR1a shows a bare body in white underwear.
//! What closes it is one derived rule — a `SkeletalMesh` child on its wearer's
//! own rig is worn by it — and every arm here reads the WORLD or the PIXELS that
//! rule produces, never the rule itself.
//!
//! The arms:
//!
//! * `a_wearables_joints_are_its_wearers_over_a_walk` — the palette a garment
//!   draws with is the palette the body under it draws with, matrix for matrix,
//!   while the body is MOVING.
//! * `a_wearable_shares_its_wearers_palette_by_pointer` — and it is one
//!   allocation, so a dressed character uploads one 342-joint palette.
//! * `a_wearable_is_culled_with_its_wearer`
//! * `a_wearable_fades_with_its_wearer`
//! * `a_wearable_follows_a_pose_nothing_authored` — the ragdoll's claim, made of
//!   the thing a ragdoll actually publishes.
//! * `a_wearable_is_not_surveyed_as_a_person` — the defect this door created.
//! * `the_crowds_wardrobe_is_deterministic_and_far_wears_nothing`
//! * `the_committed_default_character_is_dressed`
//! * `the_islands_wearables_carry_their_licence_on_disk`
//! * `both_authoring_shapes_resolve_to_the_same_wearer` — a hierarchy child and
//!   an `AttachedTo`, through the one Ring-0 door.
//! * `what_a_wearable_costs` — draws, palette bytes and geometry.
//! * `a_thousand_dressed_agents_cost_what_the_tier_says`
//!
//! …and the five the OUTFIT1 AUDIT added, each closing something the wave
//! carried:
//!
//! * `a_fitted_garment_encloses_the_body_it_is_on` — item 166, on the geometry.
//! * `the_eyes_have_a_section_of_their_own_and_it_is_not_skin` — item 167. It
//!   REPLACES `the_combined_bodys_eyes_have_no_section_of_their_own`, which
//!   asserted the refusal; the refusal rested on a box around a 289-vertex
//!   island and the island splits.
//! * `the_hair_cards_are_masked_by_the_grooms_own_coverage` — item 164.
//! * `a_driver_wears_what_the_level_wears` — finding F2, the one the island's
//!   own street showed and the gate's synthetic crowd could not.
//! * `the_two_projectors_take_the_wearable_door` — finding F8's host half,
//!   named for what it reads: two SOURCES.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{AnimStateMachine, Material, SkeletalMesh, Transform, Visibility};
use inf_ecs::math::Vec3d;
use inf_ecs::wearable::{wearable_guid, HAIR_SLOT, OUTFIT_SLOT};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_player::skinned::SkinnedRegistry;
use inf_player::vmesh::VmeshRegistry;
use inf_render::RenderScene;

const HZ: f64 = 60.0;
const HERO: Uuid = Uuid::from_u128(0x0FF1_7000_0000_1001);

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The committed starter character's folder — our own, licence-free, and on
/// every checkout, so every arm below this line runs on CI.
fn starter_dir() -> PathBuf {
    repo().join("samples").join("starter-character")
}

/// The island's built project, or `None` on a machine that has not built it.
fn island_project() -> Option<PathBuf> {
    let p = repo().join("../island-build/project/Content");
    p.is_dir().then(|| p.canonicalize().unwrap_or(p))
}

fn ids() -> inf_editor_core::character::CharacterIds {
    inf_editor_core::samples::starter_character_ids()
}

fn asset(id: Option<inf_asset::AssetId>) -> Uuid {
    id.expect("every starter id is fixed").0
}

/// **A dressed character, built the way a level's bytes describe one**: a body
/// entity with two child entities naming their own meshes and the wearer's own
/// skeleton.
///
/// Deliberately raw ECS rather than `SceneDoc::edit_dress_character` — this file
/// is the shipped player's gate and the shape it has to certify is the shape a
/// LEVEL puts in front of it, whatever door wrote it.
fn dressed_world() -> EcsWorld {
    let i = ids();
    let mut world = EcsWorld::new();
    let hero = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(hero).insert((
        Transform {
            translation: Vec3d::new(0.0, 1.0, 0.0),
            ..Transform::IDENTITY
        },
        SkeletalMesh {
            mesh: Some(asset(i.mesh)),
            skeleton: Some(asset(i.skeleton)),
        },
        AnimStateMachine {
            sm: Some(asset(i.machine)),
            ..Default::default()
        },
        // The PAWN, so `movement::camera_subject` answers this hero and the
        // near-fade arm has a subject to fade.
        inf_ecs::components::CharacterMovement {
            player_controlled: true,
            ..Default::default()
        },
        Material::default(),
    ));
    for (slot, mesh) in [(OUTFIT_SLOT, i.outfit), (HAIR_SLOT, i.hair)] {
        let e = world.spawn_with_guid(wearable_guid(HERO, slot), slot, None);
        inf_ecs::hierarchy::set_parent(world.world_mut(), e, Some(hero));
        world.world_mut().entity_mut(e).insert((
            Transform::IDENTITY,
            SkeletalMesh {
                mesh: Some(asset(mesh)),
                skeleton: Some(asset(i.skeleton)),
            },
            Material::default(),
        ));
    }
    world.mark_dirty();
    world.propagate();
    world
}

/// The sim that world runs in, with the committed rig, clips and machine bound
/// so the pose step really poses.
fn dressed_sim() -> (RuntimeSim, SkinnedRegistry) {
    let dir = starter_dir();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(&dir);
    let mut sim = RuntimeSim::new(dressed_world(), Vec::new(), DVec2::new(0.0, -9.81), HZ);
    sim.set_skeletons(skeletons.into_iter().collect());
    sim.set_pose_clips(clips.into_iter().map(|(g, a)| (g, a.clip)).collect());
    sim.set_state_machines(machines.into_iter().map(|(g, a)| (g, a.machine)).collect());
    (sim, SkinnedRegistry::from_dir(&dir))
}

fn project(sim: &RuntimeSim, skinned: &SkinnedRegistry) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    inf_player::render::project_scene_with_skinned(
        &mut scene,
        sim,
        0.0,
        &VmeshRegistry::new(),
        skinned,
        &inf_voxel::VoxelVolumes::default(),
    );
    scene
}

/// The skinned instance whose mesh slot resolves to `mesh`, by the store's own
/// key — a projection lists them in `Guid` order and this file must not care.
fn instance_of<'a>(
    scene: &'a RenderScene,
    skinned: &SkinnedRegistry,
    sm: &SkeletalMesh,
) -> Option<&'a inf_render::SkinnedInstance> {
    let want = skinned.resolve_skinned(sm, None, None, None)?.key;
    scene.skinned.iter().find(|i| {
        scene.skinned_meshes.get(i.mesh).is_some() && skinned_key(scene, skinned, i) == Some(want)
    })
}

/// The `(mesh, skeleton)` an instance was built from — recovered by pointer,
/// because that is the identity the projector deduplicates its slots on.
fn skinned_key(
    scene: &RenderScene,
    skinned: &SkinnedRegistry,
    inst: &inf_render::SkinnedInstance,
) -> Option<(Uuid, Uuid)> {
    let arc = scene.skinned_meshes.get(inst.mesh)?;
    let i = ids();
    for (mesh, skel) in [
        (asset(i.mesh), asset(i.skeleton)),
        (asset(i.outfit), asset(i.skeleton)),
        (asset(i.hair), asset(i.skeleton)),
    ] {
        let sm = SkeletalMesh {
            mesh: Some(mesh),
            skeleton: Some(skel),
        };
        if let Some(d) = skinned.resolve_skinned(&sm, None, None, None) {
            if std::sync::Arc::ptr_eq(&d.mesh, arc) {
                return Some(d.key);
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) THE POSE
// ─────────────────────────────────────────────────────────────────────────────

/// **A wearable's joints are its wearer's, while the wearer is moving.**
///
/// The claim the whole door rests on, and the two halves are equally load
/// bearing: the palettes must be EQUAL, and the body's own hand must have MOVED
/// between the two reads — a garment that agrees with a body standing still
/// agrees with nothing.
#[test]
fn a_wearables_joints_are_its_wearers_over_a_walk() {
    let (mut sim, skinned) = dressed_sim();
    let i = ids();
    let body = SkeletalMesh {
        mesh: Some(asset(i.mesh)),
        skeleton: Some(asset(i.skeleton)),
    };
    let outfit = SkeletalMesh {
        mesh: Some(asset(i.outfit)),
        skeleton: Some(asset(i.skeleton)),
    };
    if skinned.resolve_skinned(&body, None, None, None).is_none() {
        eprintln!(
            "SKIP: the committed starter body did not resolve out of {:?}",
            starter_dir()
        );
        return;
    }
    // The hand joint, by name, so the number below is a HAND and not index 42.
    let rig: inf_anim::SkeletonAsset = inf_asset::decode(
        &std::fs::read(starter_dir().join("Starter.inf_skel")).expect("the committed rig"),
    )
    .expect("the rig decodes");
    let hand = rig
        .skeleton
        .index_of("hand_l")
        .expect("the committed rig has a left hand") as usize;

    let mut first: Option<glam::Mat4> = None;
    let mut moved = 0.0f32;
    for step in 0..90 {
        sim.step_once(RuntimeInput::default());
        if step % 30 != 29 {
            continue;
        }
        let scene = project(&sim, &skinned);
        let b = instance_of(&scene, &skinned, &body).expect("the body draws");
        let o = instance_of(&scene, &skinned, &outfit).expect("the outfit draws");
        assert_eq!(
            b.palette.len(),
            o.palette.len(),
            "the outfit is drawn with a palette of a different length from the body's"
        );
        assert_eq!(
            b.palette[hand], o.palette[hand],
            "at step {step} the outfit's `hand_l` is not the body's"
        );
        assert_eq!(
            &b.palette[..],
            &o.palette[..],
            "at step {step} the outfit's palette is not the body's, matrix for matrix"
        );
        match first {
            None => first = Some(b.palette[hand]),
            Some(f) => {
                moved =
                    moved.max((f.w_axis.truncate() - b.palette[hand].w_axis.truncate()).length());
            }
        }
    }
    println!("the hand moved {:.4} m between the samples", moved);
    assert!(
        moved > 1e-4,
        "the body's own hand never moved ({moved:.6} m), so `the outfit agrees \
         with the body` is a comparison of two bind poses"
    );
}

/// **And it is ONE palette**, not two identical ones.
///
/// The skinned pass's palette atlas deduplicates by `Arc` pointer, so this is the
/// difference between one upload of a rig-length palette per dressed character
/// and one per garment.
#[test]
fn a_wearable_shares_its_wearers_palette_by_pointer() {
    let (mut sim, skinned) = dressed_sim();
    let i = ids();
    let body = SkeletalMesh {
        mesh: Some(asset(i.mesh)),
        skeleton: Some(asset(i.skeleton)),
    };
    if skinned.resolve_skinned(&body, None, None, None).is_none() {
        eprintln!("SKIP: the committed starter body did not resolve");
        return;
    }
    sim.step_once(RuntimeInput::default());
    let scene = project(&sim, &skinned);
    let b = instance_of(&scene, &skinned, &body).expect("the body draws");
    let o = instance_of(
        &scene,
        &skinned,
        &SkeletalMesh {
            mesh: Some(asset(i.outfit)),
            skeleton: Some(asset(i.skeleton)),
        },
    )
    .expect("the outfit draws");
    let h = instance_of(
        &scene,
        &skinned,
        &SkeletalMesh {
            mesh: Some(asset(i.hair)),
            skeleton: Some(asset(i.skeleton)),
        },
    )
    .expect("the hair draws");
    println!(
        "one dressed character: {} skinned draws, palette {} joints, {} bytes ONCE",
        scene.skinned.len(),
        b.palette.len(),
        b.palette.len() * std::mem::size_of::<glam::Mat4>()
    );
    assert!(
        std::sync::Arc::ptr_eq(&b.palette, &o.palette),
        "the outfit carries its own copy of the body's palette"
    );
    assert!(
        std::sync::Arc::ptr_eq(&b.palette, &h.palette),
        "the hair carries its own copy of the body's palette"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) VISIBILITY AND THE CAMERA
// ─────────────────────────────────────────────────────────────────────────────

/// **Hiding a character hides its clothes**, and it does so through the
/// `ComputedVisibility` AND-chain rather than through anything this wave wrote.
#[test]
fn a_wearable_is_culled_with_its_wearer() {
    let (mut sim, skinned) = dressed_sim();
    if skinned
        .resolve_skinned(
            &SkeletalMesh {
                mesh: Some(asset(ids().mesh)),
                skeleton: Some(asset(ids().skeleton)),
            },
            None,
            None,
            None,
        )
        .is_none()
    {
        eprintln!("SKIP: the committed starter body did not resolve");
        return;
    }
    sim.step_once(RuntimeInput::default());
    let before = project(&sim, &skinned).skinned.len();
    assert_eq!(before, 3, "a dressed character is a body and two wearables");
    let hero = sim.world().entity_of(HERO).expect("the hero");
    sim.world_mut()
        .world_mut()
        .entity_mut(hero)
        .insert(Visibility { visible: false });
    sim.world_mut().mark_dirty();
    sim.world_mut().propagate();
    let after = project(&sim, &skinned).skinned.len();
    println!("skinned draws: {before} visible, {after} hidden");
    assert_eq!(
        after, 0,
        "hiding the character left {after} of its own draws in the scene"
    );
}

/// **A garment fades with the body it is on**, through the wearer's own
/// `subject_fade` — the first thing in this tree that reads a wearable's WEARER
/// rather than its own entity.
#[test]
fn a_wearable_fades_with_its_wearer() {
    let (mut sim, skinned) = dressed_sim();
    if skinned
        .resolve_skinned(
            &SkeletalMesh {
                mesh: Some(asset(ids().mesh)),
                skeleton: Some(asset(ids().skeleton)),
            },
            None,
            None,
            None,
        )
        .is_none()
    {
        eprintln!("SKIP: the committed starter body did not resolve");
        return;
    }
    sim.step_once(RuntimeInput::default());
    // Nothing faded: every instance keeps its material's own pair.
    for inst in &project(&sim, &skinned).skinned {
        assert_ne!(
            inst.blend,
            inf_render::BLEND_NEAR_FADE,
            "an instance is fading before the camera asked for it"
        );
    }
    // The camera pulls in on the hero. The subject is the sim's own answer
    // (`movement::camera_subject`, refreshed by the step above); the fade is
    // what a short boom publishes, written here rather than driven so the arm
    // measures the PROJECTION and not the boom.
    assert_eq!(
        sim.camera_subject(),
        Some(HERO),
        "the hero is not the camera's subject, so no fade would reach it"
    );
    sim.camera_mut().subject_fade = 0.25;
    let scene = project(&sim, &skinned);
    assert_eq!(scene.skinned.len(), 3);
    for inst in &scene.skinned {
        assert_eq!(
            inst.blend,
            inf_render::BLEND_NEAR_FADE,
            "one of the dressed character's three draws is still solid at a \
             0.25 fade — a coat drawn over a body that is not there"
        );
        assert!(
            (inst.cutoff - 0.25).abs() < 1e-6,
            "the fade threshold reached {} rather than the camera's 0.25",
            inst.cutoff
        );
    }
    println!("all {} draws fade at 0.25", scene.skinned.len());
}

/// **A wearable follows a pose nothing authored** — the ragdoll's claim, made of
/// what a ragdoll actually publishes.
///
/// `apply_ragdoll_pose` writes an `EvaluatedPose` into the sim's pose store for
/// the ragdolling character and nothing else; the wearable reads that store
/// through its WEARER's guid. So the claim "the garment follows the ragdoll" is
/// exactly "the garment follows whatever is in the wearer's pose slot, including
/// a pose no clip and no machine produced" — which is what this writes and reads
/// back, without needing a physics solve to make the point.
#[test]
fn a_wearable_follows_a_pose_nothing_authored() {
    let (mut sim, skinned) = dressed_sim();
    let i = ids();
    let body = SkeletalMesh {
        mesh: Some(asset(i.mesh)),
        skeleton: Some(asset(i.skeleton)),
    };
    if skinned.resolve_skinned(&body, None, None, None).is_none() {
        eprintln!("SKIP: the committed starter body did not resolve");
        return;
    }
    sim.step_once(RuntimeInput::default());
    let rig: inf_anim::SkeletonAsset = inf_asset::decode(
        &std::fs::read(starter_dir().join("Starter.inf_skel")).expect("the committed rig"),
    )
    .expect("the rig decodes");
    let hand = rig.skeleton.index_of("hand_l").expect("a left hand") as usize;
    // A pose nothing in the sim can produce: every joint rolled 40 degrees, the
    // shape a ragdoll's read-back has (an arbitrary set of local rotations that
    // no clip contains).
    let mut pose = inf_anim::Pose::rest(&rig.skeleton);
    for l in pose.locals.iter_mut() {
        l.rotation = (glam::Quat::from_array(l.rotation)
            * glam::Quat::from_rotation_z(40f32.to_radians()))
        .to_array();
    }
    let before = project(&sim, &skinned);
    let was = instance_of(&before, &skinned, &body)
        .expect("the body draws")
        .palette[hand];
    {
        let world = sim.world_mut();
        let mut store = world
            .world_mut()
            .remove_resource::<inf_ecs::pose::PoseStoreRes>()
            .unwrap_or_default();
        store.0.insert(
            HERO,
            inf_ecs::pose::EvaluatedPose {
                skeleton: asset(i.skeleton),
                pose,
                ..store
                    .0
                    .get(&HERO)
                    .cloned()
                    .expect("the hero was posed by its own machine")
            },
        );
        world.world_mut().insert_resource(store);
    }
    let scene = project(&sim, &skinned);
    let b = instance_of(&scene, &skinned, &body).expect("the body draws");
    let o = instance_of(
        &scene,
        &skinned,
        &SkeletalMesh {
            mesh: Some(asset(i.outfit)),
            skeleton: Some(asset(i.skeleton)),
        },
    )
    .expect("the outfit draws");
    let moved = (was.w_axis.truncate() - b.palette[hand].w_axis.truncate()).length();
    println!("the imposed pose moved `hand_l` {moved:.4} m");
    assert!(
        moved > 1e-3,
        "the imposed pose did not move the body's hand ({moved:.6} m), so the \
         comparison below is between two identical bind poses"
    );
    assert_eq!(
        b.palette[hand], o.palette[hand],
        "the body took the imposed pose and its outfit did not"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE CROWD
// ─────────────────────────────────────────────────────────────────────────────

/// **A wearable is not a person**, and the survey that decides what a crowd
/// wears has to say so.
///
/// Without the exclusion a dressed character offers THREE archetypes — itself, a
/// shirt and a haircut — and a third of the island's pedestrians is a walking
/// shirt.
#[test]
fn a_wearable_is_not_surveyed_as_a_person() {
    let world = dressed_world();
    let bodies = inf_ecs::society::level_archetypes(&world);
    println!(
        "the level offers {} archetype(s): {:?}",
        bodies.len(),
        bodies.iter().map(|b| b.mesh).collect::<Vec<_>>()
    );
    assert_eq!(
        bodies.len(),
        1,
        "a dressed character offers more than one body to the crowd — its \
         clothes are being read as people"
    );
    let a = bodies[0];
    assert_eq!(
        a.mesh,
        Some(asset(ids().mesh)),
        "the archetype is not the body"
    );
    // …and the clothes came WITH it, which is the crowd's whole wardrobe.
    assert_eq!(
        a.outfit.map(|w| w.mesh),
        Some(asset(ids().outfit)),
        "the archetype carries no outfit, so the crowd has nothing to put on"
    );
    assert_eq!(a.hair.map(|w| w.mesh), Some(asset(ids().hair)));
}

/// **The crowd's wardrobe is deterministic, and `Far` wears nothing.**
///
/// Two independent worlds built from the same guids assign the same clothes to
/// the same agents; and the tier rule is a measurement, not a sentence — the
/// same agent at `Far` has no wearable entities at all.
#[test]
fn the_crowds_wardrobe_is_deterministic_and_far_wears_nothing() {
    use inf_ecs::crowd::{CrowdArchetype, CrowdRecord, CrowdTier};
    let build = || {
        let mut world = dressed_world();
        let bodies = inf_ecs::society::level_archetypes(&world);
        let mut pop: BTreeMap<Uuid, CrowdRecord> = BTreeMap::new();
        for n in 0..24u128 {
            let g = Uuid::from_u128(0x0FF1_7000_C0DE_0000 + n);
            let a: CrowdArchetype = inf_ecs::society::level_archetype_for(&bodies, g);
            let mut rec = CrowdRecord::standing(a, DVec3::new(n as f64 * 0.5, 1.0, 0.0));
            rec.tier = CrowdTier::Near;
            pop.insert(g, rec);
        }
        world.mark_dirty();
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
        sim.set_crowd_population(pop);
        sim.step_once(RuntimeInput::default());
        sim
    };
    let worn = |sim: &RuntimeSim| -> Vec<(Uuid, Option<Uuid>)> {
        (0..24u128)
            .map(|n| {
                let g = Uuid::from_u128(0x0FF1_7000_C0DE_0000 + n);
                let m = sim
                    .world()
                    .entity_of(wearable_guid(g, OUTFIT_SLOT))
                    .and_then(|e| sim.world().world().get::<SkeletalMesh>(e))
                    .and_then(|s| s.mesh);
                (g, m)
            })
            .collect()
    };
    let a = build();
    let b = build();
    let (wa, wb) = (worn(&a), worn(&b));
    let dressed = wa.iter().filter(|(_, m)| m.is_some()).count();
    println!(
        "{dressed} of 24 near agents are dressed; two boots agree: {}",
        wa == wb
    );
    assert_eq!(wa, wb, "two boots dressed the same crowd differently");
    assert_eq!(
        dressed, 24,
        "only {dressed} of 24 posing agents put anything on"
    );

    // …and the same population at `Far` wears nothing at all.
    //
    // **The tier is a DISTANCE, not a field**: `step_crowd` re-derives it from
    // the ladder every step, so a record written with `tier: Far` is promoted
    // back the moment the pawn is standing next to it — measured, all 24 of
    // them, on this arm's first cut. So the agents go 300 m away and the ladder
    // is tightened around them, and the tier they actually got is asserted
    // before the clothes are counted: a DESPAWNED agent wears nothing
    // vacuously.
    let mut far = {
        let mut world = dressed_world();
        let bodies = inf_ecs::society::level_archetypes(&world);
        let mut pop: BTreeMap<Uuid, CrowdRecord> = BTreeMap::new();
        for n in 0..24u128 {
            let g = Uuid::from_u128(0x0FF1_7000_C0DE_0000 + n);
            let a: CrowdArchetype = inf_ecs::society::level_archetype_for(&bodies, g);
            pop.insert(
                g,
                CrowdRecord::standing(a, DVec3::new(300.0 + n as f64 * 0.5, 1.0, 0.0)),
            );
        }
        world.mark_dirty();
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
        assert!(
            sim.set_crowd_radii((8.0, 24.0, 4000.0)),
            "the ladder is legal"
        );
        sim.set_crowd_population(pop);
        sim
    };
    // **A ladder needs an ANCHOR.** `CrowdBand::from_anchors` falls open to
    // `Full` on an empty anchor set, so a world whose pawn is not a
    // `StreamingSource` tiers nothing at all — which is what the first cut of
    // this arm measured (all 24 at `Full`, 300 m away, on an 8/24/4000 ladder).
    {
        let hero = far.world().entity_of(HERO).expect("the hero");
        far.world_mut()
            .world_mut()
            .entity_mut(hero)
            .insert(inf_ecs::components::StreamingSource { radius_m: 256.0 });
        far.world_mut().mark_dirty();
        far.world_mut().propagate();
    }
    far.step_once(RuntimeInput::default());
    let tiers: Vec<CrowdTier> = (0..24u128)
        .filter_map(|n| {
            let g = Uuid::from_u128(0x0FF1_7000_C0DE_0000 + n);
            far.world()
                .entity_of(g)
                .and_then(|e| far.world().world().get::<inf_ecs::crowd::CrowdAgent>(e))
                .map(|a| a.tier)
        })
        .collect();
    let n = worn(&far).iter().filter(|(_, m)| m.is_some()).count();
    println!(
        "far agents: {} materialized at {:?}, wearing something: {n}",
        tiers.len(),
        tiers.first()
    );
    assert_eq!(tiers.len(), 24, "the far agents did not materialize at all");
    assert!(
        tiers.iter().all(|t| *t == CrowdTier::Far),
        "the arm's agents are not at Far, so `Far wears nothing` is untested"
    );
    assert_eq!(
        n, 0,
        "{n} `Far` agents are still carrying a garment draw — the tier rule is \
         the crowd's whole clothing budget"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE CONTENT
// ─────────────────────────────────────────────────────────────────────────────

/// **The public repository's default character is dressed** — on disk, in the
/// committed folder, and in the committed level a fresh project scaffolds.
#[test]
fn the_committed_default_character_is_dressed() {
    for (dir, stem) in [
        (starter_dir(), "Starter"),
        (
            repo().join("samples").join("starter-character-f"),
            "Starter_F",
        ),
    ] {
        for f in [
            format!("{stem}_Outfit.inf_mesh"),
            format!("{stem}_Outfit_Top.inf_mat"),
            format!("{stem}_Outfit_Bottom.inf_mat"),
            format!("{stem}_Hair_Mesh.inf_mesh"),
            format!("{stem}_Hair.inf_mat"),
        ] {
            let p = dir.join(&f);
            assert!(p.is_file(), "{} is not committed", p.display());
        }
        let outfit: inf_mesh::MeshAsset =
            inf_asset::decode(&std::fs::read(dir.join(format!("{stem}_Outfit.inf_mesh"))).unwrap())
                .expect("the committed outfit decodes");
        println!(
            "{stem}: outfit {} tris in {} slots, hair {} tris",
            outfit.triangle_count(),
            outfit.material_slots.len(),
            inf_asset::decode::<inf_mesh::MeshAsset>(
                &std::fs::read(dir.join(format!("{stem}_Hair_Mesh.inf_mesh"))).unwrap()
            )
            .unwrap()
            .triangle_count()
        );
        assert!(
            outfit.triangle_count() > 100,
            "the committed outfit is {} triangles",
            outfit.triangle_count()
        );
        assert_eq!(
            outfit.material_slots.len(),
            2,
            "the outfit is a tee AND a pair of trousers, which is two slots"
        );
        assert!(
            outfit.submeshes.iter().all(|s| s.is_skinned()),
            "a committed garment lost its skin stream and will draw in bind pose"
        );
    }
    // …and the level a `blank-3d` project scaffolds spawns it wearing them.
    let lvl = repo()
        .join("templates")
        .join("blank-3d")
        .join("Blank.inf_lvl");
    let doc = inf_editor_core::scene::serialize::load(&lvl).expect("the committed level loads");
    let pawn = inf_ecs::movement::camera_subject(doc.world()).expect("the template has a pawn");
    let worn = inf_ecs::wearable::wearables_of(doc.world(), pawn);
    println!("the blank-3d pawn wears {} thing(s)", worn.len());
    assert_eq!(
        worn.len(),
        2,
        "the committed starter level's pawn is wearing {} thing(s), not an \
         outfit and a head of hair",
        worn.len()
    );
}

/// **What an import wrote at a committed GUID carries its licence** (carried 110,
/// and the half of it nobody had measured).
#[test]
fn the_islands_wearables_carry_their_licence_on_disk() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — it is local-only content and CI has none");
        return;
    };
    let mut checked = 0;
    for stem in [
        "Starter_Outfit.inf_mesh",
        "Starter_Hair_Mesh.inf_mesh",
        "Starter_F_Outfit.inf_mesh",
        "Starter_F_Hair_Mesh.inf_mesh",
        "Starter_Body.inf_mesh",
        "Starter.inf_skel",
    ] {
        let p = content.join(format!("{stem}.toml"));
        if !p.is_file() {
            eprintln!("SKIP {stem}: not in this project");
            continue;
        }
        let text = std::fs::read_to_string(&p).expect("the sidecar reads");
        assert!(
            text.contains("licence_pack"),
            "{stem} carries no licence row, and it is the asset the level \
             REFERENCES — the ones that ship were the ones with none"
        );
        checked += 1;
    }
    println!("{checked} committed-GUID assets carry a licence row");
    assert!(checked >= 4, "only {checked} were there to check");
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) PIE == SHIPPING
// ─────────────────────────────────────────────────────────────────────────────

/// **Both authoring shapes resolve to the same wearer** — a hierarchy child and
/// an `AttachedTo`.
///
/// # Its name used to claim more than it reads (wave OUTFIT1 AUDIT, finding F8)
///
/// It was called `pie_equals_shipping_on_a_dressed_character`, and it calls ONE
/// Ring-0 function on two worlds. That is worth holding — the editor writes a
/// child and a runtime equip writes an `AttachedTo`, and a door that answered
/// differently for the two would draw a garment in its own bind pose on exactly
/// one of the paths — but it is not a comparison of two HOSTS and no name in
/// this file may say it is. The host half is
/// `the_two_projectors_take_the_wearable_door`, below.
#[test]
fn both_authoring_shapes_resolve_to_the_same_wearer() {
    let world = dressed_world();
    let hero = world.entity_of(HERO).expect("the hero");
    for slot in [OUTFIT_SLOT, HAIR_SLOT] {
        let e = world.entity_of(wearable_guid(HERO, slot)).expect("worn");
        assert_eq!(
            inf_ecs::wearable::pose_source(&world, e, wearable_guid(HERO, slot)),
            (hero, HERO),
            "{slot} does not resolve to its wearer"
        );
    }
    // The runtime shape: the same rule over `AttachedTo`, which is what a
    // runtime equip spawns and what no level contains.
    let mut w2 = EcsWorld::new();
    let i = ids();
    let body = w2.spawn_with_guid(HERO, "Hero", None);
    w2.world_mut().entity_mut(body).insert((
        Transform::IDENTITY,
        SkeletalMesh {
            mesh: Some(asset(i.mesh)),
            skeleton: Some(asset(i.skeleton)),
        },
    ));
    let coat = Uuid::from_u128(0x0FF1_7000_0000_2002);
    let e = w2.spawn_with_guid(coat, "Coat", None);
    w2.world_mut().entity_mut(e).insert((
        Transform::IDENTITY,
        SkeletalMesh {
            mesh: Some(asset(i.outfit)),
            skeleton: Some(asset(i.skeleton)),
        },
        inf_ecs::components::AttachedTo::new(HERO, "", Vec3d::ZERO),
    ));
    w2.mark_dirty();
    w2.propagate();
    assert_eq!(
        inf_ecs::wearable::pose_source(&w2, e, coat),
        (body, HERO),
        "an attached garment does not resolve to its wearer, so a runtime equip \
         would draw in its own bind pose"
    );
    println!("both authoring shapes resolve to the same wearer");
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) COST
// ─────────────────────────────────────────────────────────────────────────────

/// **What a wearable costs**, in the three currencies the renderer spends:
/// draws, palette bytes and geometry.
#[test]
fn what_a_wearable_costs() {
    let (mut sim, skinned) = dressed_sim();
    let i = ids();
    let body = SkeletalMesh {
        mesh: Some(asset(i.mesh)),
        skeleton: Some(asset(i.skeleton)),
    };
    if skinned.resolve_skinned(&body, None, None, None).is_none() {
        eprintln!("SKIP: the committed starter body did not resolve");
        return;
    }
    sim.step_once(RuntimeInput::default());
    let scene = project(&sim, &skinned);
    let tris: usize = scene
        .skinned_meshes
        .iter()
        .map(|m| m.indices.len() / 3)
        .sum();
    let palettes: usize = {
        let mut seen: Vec<*const Vec<glam::Mat4>> = Vec::new();
        for inst in &scene.skinned {
            let p = std::sync::Arc::as_ptr(&inst.palette);
            if !seen.contains(&p) {
                seen.push(p);
            }
        }
        seen.len()
    };
    let bytes = scene
        .skinned
        .first()
        .map(|i| i.palette.len() * std::mem::size_of::<glam::Mat4>())
        .unwrap_or(0);
    println!(
        "\n=== ONE DRESSED CHARACTER ===\n  \
         skinned draws {}, distinct palettes {} ({} bytes each), \
         geometry {} triangles over {} mesh slots",
        scene.skinned.len(),
        palettes,
        bytes,
        tris,
        scene.skinned_meshes.len()
    );
    assert_eq!(scene.skinned.len(), 3, "a dressed character is three draws");
    assert_eq!(
        palettes, 1,
        "a dressed character uploaded {palettes} palettes; the sharing is what \
         makes the door cheap"
    );
}

/// **A THOUSAND AGENTS, DRESSED** — what the wardrobe costs the crowd, per tier.
///
/// The number that matters is not the total, it is the SHAPE: the tiers that pose
/// pay two entities and two draws apiece and the tier that does not pays nothing
/// at all, so a crowd's clothing bill is a function of how many of it are near
/// the camera rather than of how many of it there are.
///
/// Printed with the profile it was taken in, because a debug step is not a
/// shipped one; the assertion is the RATIO, which is profile-independent.
#[test]
fn a_thousand_dressed_agents_cost_what_the_tier_says() {
    use inf_ecs::crowd::CrowdRecord;
    let n = 1000u128;
    let build = |far: bool, dress: bool| {
        let mut world = dressed_world();
        let mut bodies = inf_ecs::society::level_archetypes(&world);
        if !dress {
            // THE CONTROL: the same thousand agents, the same tier, the same
            // rig — and nothing on. Without it the number below is the cost of
            // posing a thousand characters, which is not what this arm claims.
            for b in bodies.iter_mut() {
                b.outfit = None;
                b.hair = None;
            }
        }
        let hero = world.entity_of(HERO).expect("the hero");
        world
            .world_mut()
            .entity_mut(hero)
            .insert(inf_ecs::components::StreamingSource { radius_m: 256.0 });
        let mut pop: BTreeMap<Uuid, CrowdRecord> = BTreeMap::new();
        for k in 0..n {
            let g = Uuid::from_u128(0x0FF1_7000_C000_0000 + k);
            let a = inf_ecs::society::level_archetype_for(&bodies, g);
            let x = if far { 300.0 } else { 1.0 } + (k % 40) as f64 * 0.4;
            let z = (k / 40) as f64 * 0.4;
            pop.insert(g, CrowdRecord::standing(a, DVec3::new(x, 1.0, z)));
        }
        world.mark_dirty();
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ);
        assert!(
            sim.set_crowd_radii((40.0, 60.0, 4000.0)),
            "the ladder is legal"
        );
        sim.set_crowd_population(pop);
        sim
    };
    let mut out = Vec::new();
    for (label, far, dress) in [
        ("near", false, true),
        ("near/bare", false, false),
        ("far", true, true),
    ] {
        let mut sim = build(far, dress);
        for _ in 0..3 {
            sim.step_once(RuntimeInput::default());
        }
        // The CROWD's clothes, not the world's: the hero is dressed at every
        // tier because it is not a crowd agent at all.
        let worn = inf_ecs::wearable::worn_count(sim.world())
            - inf_ecs::wearable::wearables_of(sim.world(), HERO).len();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            sim.step_once(RuntimeInput::default());
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / 10.0;
        println!(
            "  {label:<5} {n} agents: {worn} wearables, {:.1} us/step, {:.3} us/agent",
            us,
            us / n as f64
        );
        out.push((worn, us));
    }
    let (near_worn, near_us) = out[0];
    let (bare_worn, bare_us) = out[1];
    let (far_worn, far_us) = out[2];
    println!(
        "
=== A THOUSAND AGENTS (debug profile; the SHAPE is the claim) ===
           near, dressed  {near_worn} wearables at {near_us:.1} us/step
           near, bare     {bare_worn} wearables at {bare_us:.1} us/step
           far, dressed   {far_worn} wearables at {far_us:.1} us/step
           the wardrobe's own share of a near step: {:.1} us ({:+.1}%)",
        near_us - bare_us,
        100.0 * (near_us - bare_us) / bare_us
    );
    assert_eq!(bare_worn, 0, "the bare control is wearing something");
    assert_eq!(
        far_worn, 0,
        "{far_worn} wearables at Far — the tier rule IS the crowd's clothing budget"
    );
    assert!(
        near_worn > 0,
        "no wearables at all near the camera, so the far number means nothing"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (7) THE THREE CLOSURES (wave OUTFIT1 AUDIT — carried 164, 166, 167)
// ─────────────────────────────────────────────────────────────────────────────

/// The connected islands of one submesh, over its triangle graph.
fn islands(sub: &inf_mesh::SubMesh) -> Vec<Vec<u32>> {
    let nv = sub.vertices.len();
    let mut parent: Vec<u32> = (0..nv as u32).collect();
    fn root(p: &mut [u32], mut a: u32) -> u32 {
        while p[a as usize] != a {
            p[a as usize] = p[p[a as usize] as usize];
            a = p[a as usize];
        }
        a
    }
    for t in sub.indices.chunks_exact(3) {
        let (a, b, c) = (
            root(&mut parent, t[0]),
            root(&mut parent, t[1]),
            root(&mut parent, t[2]),
        );
        parent[b as usize] = a;
        parent[c as usize] = a;
    }
    let mut by: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for i in 0..nv as u32 {
        let r = root(&mut parent, i);
        by.entry(r).or_default().push(i);
    }
    by.into_values().collect()
}

/// The file an asset GUID lives in, found by its sidecar under `content`.
fn find_asset(content: &Path, id: inf_asset::AssetId) -> Option<PathBuf> {
    fn walk(dir: &Path, want: &str, out: &mut Option<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            if out.is_some() {
                return;
            }
            let p = e.path();
            if p.is_dir() {
                walk(&p, want, out);
            } else if p.extension().is_some_and(|x| x == "toml") {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    // The sidecar's OWN guid line, not any mention: a sidecar
                    // lists its dependencies' guids too, and matching one of
                    // those hands back the wrong file.
                    if t.lines().any(|l| l.trim() == format!("guid = \"{want}\"")) {
                        *out = Some(p.with_extension(""));
                    }
                }
            }
        }
    }
    let mut out = None;
    walk(content, &id.0.to_string(), &mut out);
    out
}

/// **The body is INSIDE the garment on it** — carried item 166, closed, and
/// measured on the geometry rather than on a pixel box.
///
/// The rule is the importer's own (`fit_wearable_over_wearer`) read back off the
/// asset: for each garment vertex, the deepest a body vertex reaches along that
/// vertex's own normal inside a narrow cylinder. A positive number is the body
/// standing OUTSIDE the shirt, which in the pixels is a brown patch on a white
/// tee.
///
/// Runs on the committed starter character on CI and on the island's rebound
/// MetaHuman where the project is there — two different pieces of content and
/// one property. Wave OUTFIT1's flat 4 mm lift left **11.60 %** of the island
/// garment's vertices with the body outside them; the fit leaves 3.57 %.
#[test]
fn a_fitted_garment_encloses_the_body_it_is_on() {
    let mut pairs: Vec<(String, PathBuf, PathBuf)> = vec![(
        "the committed starter character".to_string(),
        starter_dir().join("Starter_Body.inf_mesh"),
        starter_dir().join("Starter_Outfit.inf_mesh"),
    )];
    match island_project() {
        Some(c) => pairs.push((
            "the island's rebound MetaHuman".to_string(),
            c.join("Starter_Body.inf_mesh"),
            c.join("Starter_Outfit.inf_mesh"),
        )),
        None => eprintln!("SKIP the island half: no island project — local-only content"),
    }
    // The importer's own constants, restated rather than imported: this arm has
    // to fail when the RULE moves, and a shared constant would move with it.
    const REACH: f32 = 0.060;
    const TANGENT: f32 = 0.008;
    for (label, body_path, outfit_path) in pairs {
        let (Ok(bb), Ok(ob)) = (std::fs::read(&body_path), std::fs::read(&outfit_path)) else {
            eprintln!("SKIP {label}: {} is not there", body_path.display());
            continue;
        };
        let body: inf_mesh::MeshAsset = inf_asset::decode(&bb).expect("the body decodes");
        let outfit: inf_mesh::MeshAsset = inf_asset::decode(&ob).expect("the outfit decodes");
        let key = |p: [f32; 3]| {
            [
                (p[0] / REACH).floor() as i32,
                (p[1] / REACH).floor() as i32,
                (p[2] / REACH).floor() as i32,
            ]
        };
        let mut grid: BTreeMap<[i32; 3], Vec<[f32; 3]>> = BTreeMap::new();
        for sub in &body.submeshes {
            for v in &sub.vertices {
                grid.entry(key(v.position)).or_default().push(v.position);
            }
        }
        let (mut n, mut outside, mut worst) = (0usize, 0usize, 0.0f32);
        for sub in &outfit.submeshes {
            for v in &sub.vertices {
                let raw = v.normal;
                let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
                if len <= 1e-6 {
                    continue;
                }
                let nrm = [raw[0] / len, raw[1] / len, raw[2] / len];
                let k = key(v.position);
                let mut deepest = 0.0f32;
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let Some(list) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                                continue;
                            };
                            for b in list {
                                let d = [
                                    b[0] - v.position[0],
                                    b[1] - v.position[1],
                                    b[2] - v.position[2],
                                ];
                                let along = d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2];
                                if !(0.0..=REACH).contains(&along) {
                                    continue;
                                }
                                let t = [
                                    d[0] - along * nrm[0],
                                    d[1] - along * nrm[1],
                                    d[2] - along * nrm[2],
                                ];
                                if t[0] * t[0] + t[1] * t[1] + t[2] * t[2] > TANGENT * TANGENT {
                                    continue;
                                }
                                deepest = deepest.max(along);
                            }
                        }
                    }
                }
                n += 1;
                if deepest > 0.001 {
                    outside += 1;
                    worst = worst.max(deepest);
                }
            }
        }
        let pct = 100.0 * outside as f64 / n.max(1) as f64;
        println!(
            "{label}: {outside} of {n} garment vertices have the body outside them ({pct:.2} %), worst {:.1} mm",
            worst * 1000.0
        );
        // ANTI-VACUITY: a garment with no vertices, or one nowhere near a body,
        // would report 0 % for the wrong reason.
        assert!(
            n > 1_000,
            "{label}: only {n} garment vertices were examined"
        );
        assert!(
            pct < 5.0,
            "{label}: {pct:.2} % of the garment has the body standing outside it \
             (worst {:.1} mm) — that is a shirt with the character's own chest \
             showing through it",
            worst * 1000.0
        );
    }
}

/// **The eyes have a section of their own, and it is not skin** — carried item
/// 167, closed by the import-side split (door (d)).
///
/// Wave OUTFIT1 refused clause 3 and left a trip-wire asserting the refusal.
/// This is its inverse, and it reads three things off the world:
///
/// * the combined body has **two eye sections**, each still one closed round
///   island under 50 mm across — the shape the importer selects on;
/// * each is bound to a material of its own (`MI_EyeL/R_Baked`), which is the
///   address clause 3 said did not exist;
/// * and that material's base colour is an EYE and not skin, measured in its own
///   texels: the sclera ring is much brighter than the iris disc, and the head
///   atlas **at the same uvs** — which is exactly what those triangles used to
///   draw with — separates the same two radii by almost nothing.
#[test]
fn the_eyes_have_a_section_of_their_own_and_it_is_not_skin() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — it is local-only content and CI has none");
        return;
    };
    let mut checked = 0;
    for stem in ["Starter_Body", "Starter_F_Body"] {
        let p = content.join(format!("{stem}.inf_mesh"));
        let Ok(bytes) = std::fs::read(&p) else {
            eprintln!("SKIP {stem}: not in this project");
            continue;
        };
        let mesh: inf_mesh::MeshAsset = inf_asset::decode(&bytes).expect("the body decodes");
        if mesh.triangle_count() < 50_000 {
            eprintln!(
                "SKIP {stem}: {} triangles — the committed low-poly body, not a rebound MetaHuman",
                mesh.triangle_count()
            );
            continue;
        }
        let eye_slots: Vec<usize> = mesh
            .material_slots
            .iter()
            .enumerate()
            .filter(|(_, n)| n.contains("MI_EyeL_Baked") || n.contains("MI_EyeR_Baked"))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            eye_slots.len(),
            2,
            "{stem}: {} material slot(s) name an eye — the split did not run, and the \
             eyeballs are drawing the head atlas at their own uvs, which is skin",
            eye_slots.len()
        );
        let eye_sections: Vec<&inf_mesh::SubMesh> = mesh
            .submeshes
            .iter()
            .filter(|s| eye_slots.contains(&(s.material_slot.unwrap_or(0) as usize)))
            .collect();
        assert_eq!(
            eye_sections.len(),
            2,
            "{stem}: the eye slots have no sections"
        );
        for sub in &eye_sections {
            let isl = islands(sub);
            assert_eq!(
                isl.len(),
                1,
                "{stem}: an eye section holds {} islands — the split took more than an \
                 eyeball with it",
                isl.len()
            );
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for v in &sub.vertices {
                for a in 0..3 {
                    lo[a] = lo[a].min(v.position[a]);
                    hi[a] = hi[a].max(v.position[a]);
                }
            }
            let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            let widest = ext.iter().cloned().fold(0.0f32, f32::max);
            let thinnest = ext.iter().cloned().fold(f32::MAX, f32::min);
            println!(
                "{stem}: eye section {} verts, {} tris, {:.1} x {:.1} x {:.1} mm",
                sub.vertices.len(),
                sub.triangle_count(),
                ext[0] * 1000.0,
                ext[1] * 1000.0,
                ext[2] * 1000.0
            );
            assert!(
                widest < 0.050 && thinnest / widest > 0.5,
                "{stem}: the eye section is {:.1} x {:.1} x {:.1} mm, which is not a ball",
                ext[0] * 1000.0,
                ext[1] * 1000.0,
                ext[2] * 1000.0
            );
            assert!(
                sub.is_skinned(),
                "{stem}: the eye section lost its skin stream and will draw in bind pose"
            );
        }
        let eye_mat =
            mesh.material_slot_assets[eye_slots[0]].expect("the eye slot resolved to a material");
        let head_mat = mesh.material_slot_assets[0].expect("the head slot resolved to a material");
        let load = |id: inf_asset::AssetId| -> Option<inf_material::TextureAsset> {
            let mat: inf_material::MaterialAsset =
                inf_asset::decode(&std::fs::read(find_asset(&content, id)?).ok()?).ok()?;
            let tex = mat.base_color_texture?;
            inf_material::TextureAsset::from_payload(
                &std::fs::read(find_asset(&content, tex)?).ok()?,
            )
            .ok()
        };
        let (Some(eye), Some(head)) = (load(eye_mat), load(head_mat)) else {
            panic!("{stem}: the eye or the head material has no base colour to read");
        };
        let ring = |t: &inf_material::TextureAsset, px: &[u8], r: f32| -> f64 {
            let mut sum = 0.0f64;
            let n = 512;
            for k in 0..n {
                let a = k as f32 / n as f32 * std::f32::consts::TAU;
                let u = 0.5 + 0.5 * r * a.cos();
                let v = 0.5 + 0.5 * r * a.sin();
                let x = ((u * (t.width - 1) as f32) as usize).min(t.width as usize - 1);
                let y = ((v * (t.height - 1) as f32) as usize).min(t.height as usize - 1);
                let i = (y * t.width as usize + x) * 4;
                sum +=
                    0.2126 * px[i] as f64 + 0.7152 * px[i + 1] as f64 + 0.0722 * px[i + 2] as f64;
            }
            sum / n as f64
        };
        let ep = eye.level_rgba8(0).expect("the eye albedo decodes");
        let hp = head.level_rgba8(0).expect("the head albedo decodes");
        // r = 0.10 is inside the iris disc (the limbus the importer derives sits
        // at 0.30..0.36 of the square); r = 0.75 is the white of the eye.
        let iris = ring(&eye, &ep, 0.10);
        let sclera = ring(&eye, &ep, 0.75);
        let skin_a = ring(&head, &hp, 0.10);
        let skin_b = ring(&head, &hp, 0.75);
        println!(
            "{stem}: EYE iris {iris:.1} sclera {sclera:.1} (delta {:.1}); the HEAD atlas at \
             the same uvs {skin_a:.1} / {skin_b:.1} (delta {:.1})",
            sclera - iris,
            skin_b - skin_a
        );
        assert!(
            sclera - iris > 20.0,
            "{stem}: the eye's sclera is only {:.1} brighter than its iris — this is not \
             an eye, it is one flat colour",
            sclera - iris
        );
        assert!(
            (sclera - iris) > 3.0 * (skin_b - skin_a).abs().max(1.0),
            "{stem}: the eye material separates its iris from its sclera by {:.1} and the \
             HEAD atlas separates the same two uvs by {:.1} — the eyes are still drawing skin",
            sclera - iris,
            skin_b - skin_a
        );
        checked += 1;
    }
    assert!(checked > 0, "no rebound body was there to check");
}

/// **A hair card is a cut-out, not a ribbon** — carried item 164, closed.
///
/// Three facts, none of them a table:
///
/// * the committed hair material is **masked**, at UE's own clip value, and it
///   names a base colour — before this it was the glTF's own white, opaque
///   `WorldGridMaterial`, because a groom's cards mesh carries Unreal's
///   checkerboard in its slot table and nothing bound the manifest's answer;
/// * its base colour is the groom's **melanin**, not white;
/// * and the ALPHA of that texture is a strand mask: a sixth of the atlas
///   survives the cutoff and the rest is a hole. A solid ribbon is all of it.
#[test]
fn the_hair_cards_are_masked_by_the_grooms_own_coverage() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — it is local-only content and CI has none");
        return;
    };
    let mut checked = 0;
    for stem in ["Starter_Hair", "Starter_F_Hair"] {
        let p = content.join(format!("{stem}.inf_mat"));
        let Ok(bytes) = std::fs::read(&p) else {
            eprintln!("SKIP {stem}: not in this project");
            continue;
        };
        let mat: inf_material::MaterialAsset =
            inf_asset::decode(&bytes).expect("the hair material decodes");
        let lum =
            0.2126 * mat.base_color[0] + 0.7152 * mat.base_color[1] + 0.0722 * mat.base_color[2];
        let Some(tex_id) = mat.base_color_texture else {
            panic!(
                "{stem}: the hair material names no base colour, so there is no alpha \
                 channel for a cut-out to live in"
            );
        };
        let tex = inf_material::TextureAsset::from_payload(
            &std::fs::read(find_asset(&content, tex_id).expect("the hair albedo is on disk"))
                .expect("the hair albedo reads"),
        )
        .expect("the hair albedo decodes");
        let px = tex.level_rgba8(0).expect("the hair albedo has a mip 0");
        let n = px.len() / 4;
        let open = px
            .chunks_exact(4)
            .filter(|p| p[3] as f32 / 255.0 >= mat.alpha_cutoff)
            .count();
        let frac = 100.0 * open as f64 / n as f64;
        println!(
            "{stem}: blend {:?} cutoff {:.3}, base colour luminance {lum:.4}, {frac:.1} % of \
             {}x{} survives the cutoff",
            mat.blend, mat.alpha_cutoff, tex.width, tex.height
        );
        assert_eq!(
            mat.blend,
            inf_material::MatBlend::Masked,
            "{stem}: the hair is {:?}, so every card draws as a solid quad",
            mat.blend
        );
        assert!(
            lum < 0.3,
            "{stem}: the hair's base colour has luminance {lum:.4} — that is the glTF's own \
             white material, not the groom's melanin"
        );
        assert!(
            (5.0..40.0).contains(&frac),
            "{stem}: {frac:.1} % of the atlas survives the cutoff — a strand mask is a sixth \
             of its atlas and a solid ribbon is all of it"
        );
        checked += 1;
    }
    assert!(checked > 0, "no hair material was there to check");
}

/// **A DRIVER WEARS WHAT THE LEVEL WEARS** — finding F2 of the OUTFIT1 audit,
/// closed.
///
/// # The defect, measured on the island before the fix
///
/// `step_crowd_banded` walks `CrowdPopulationRes::records` and calls
/// `set_tier_wearables` for each — so every RESIDENT is dressed by its tier. A
/// traffic driver is deliberately **not** a record (`crowd::spawn_body`'s own
/// doc: it has no route, no schedule and no tier of its own), so that walk never
/// reached it and nothing put its clothes on. Measured on
/// `VancouverIsland.inf_lvl` at 900 steps: 239 society agents, every one of them
/// `Far` and correctly wearing nothing, and ONE `CrowdAgent` at `Full` 106.9 m
/// from the pawn — named `Driver`, `in_population false` — with **no wearable
/// entities at all**. Photographed at three metres: a MetaHuman body in white
/// underwear with no hair, walking down the showcase's own street.
///
/// # What this arm reads
///
/// The WORLD and the PROJECTOR, on both halves:
///
/// * the committed starter character (CI): a driver built from the level's own
///   archetype has both wearable children, they are visible, and the projector's
///   skinned instance list really contains their meshes — 3 draws for the hero
///   alone, 6 with the driver beside it;
/// * the island's own document, through `society::level_archetype` — the same
///   door `sync_society` plans a resident's day with — so the arm fails if the
///   island ever stops offering a dressed body.
///
/// Mutation: delete `set_tier_wearables` from `crowd::spawn_body` and both
/// halves go red.
#[test]
fn a_driver_wears_what_the_level_wears() {
    let i = ids();
    let driver = Uuid::from_u128(0x0FF1_7000_D817_0001);

    // ── the committed half, on CI ────────────────────────────────────────────
    let (mut sim, skinned) = dressed_sim();
    if skinned
        .resolve_skinned(
            &SkeletalMesh {
                mesh: Some(asset(i.mesh)),
                skeleton: Some(asset(i.skeleton)),
            },
            None,
            None,
            None,
        )
        .is_none()
    {
        eprintln!("SKIP: the committed starter body did not resolve");
        return;
    }
    sim.step_once(RuntimeInput::default());
    let before = project(&sim, &skinned).skinned.len();
    let archetype = inf_ecs::society::level_archetype(sim.world());
    assert!(
        archetype.outfit.is_some() && archetype.hair.is_some(),
        "the level's own archetype carries no clothes, so a dressed driver would \
         be vacuous"
    );
    inf_ecs::crowd::spawn_body(
        sim.world_mut(),
        driver,
        &archetype,
        DVec3::new(2.0, 1.0, 0.0),
    );
    sim.world_mut().mark_dirty();
    sim.world_mut().propagate();
    let worn = inf_ecs::wearable::wearables_of(sim.world(), driver);
    println!(
        "a driver built from the level's archetype wears {} thing(s); skinned \
         draws {before} -> {}",
        worn.len(),
        project(&sim, &skinned).skinned.len()
    );
    assert_eq!(
        worn.len(),
        2,
        "a traffic driver is a person made of the level's own archetype and it \
         went out in {} thing(s) — `spawn_body` is not taking the same second \
         step `step_crowd_banded` takes",
        worn.len()
    );
    // …and they are DRAWN, which is the half a world assertion cannot see.
    let scene = project(&sim, &skinned);
    assert_eq!(
        scene.skinned.len(),
        before + 3,
        "the driver added {} skinned draw(s) rather than a body and its two \
         wearables — the clothes exist in the world and the projector is not \
         drawing them",
        scene.skinned.len() - before
    );
    for (label, mesh) in [("outfit", i.outfit), ("hair", i.hair)] {
        let sm = SkeletalMesh {
            mesh: Some(asset(mesh)),
            skeleton: Some(asset(i.skeleton)),
        };
        let want = skinned
            .resolve_skinned(&sm, None, None, None)
            .expect("the committed garment resolves")
            .key;
        let n = scene
            .skinned
            .iter()
            .filter(|inst| skinned_key(&scene, &skinned, inst) == Some(want))
            .count();
        assert_eq!(
            n, 2,
            "{n} instance(s) draw the {label} — the hero's and the driver's is two"
        );
    }

    // ── and the island's own document ────────────────────────────────────────
    let Some(content) = island_project() else {
        eprintln!("SKIP the island half: no island project — local-only content");
        return;
    };
    let lvl = content.join("VancouverIsland.inf_lvl");
    let Ok(bytes) = std::fs::read(&lvl) else {
        eprintln!("SKIP the island half: no built level");
        return;
    };
    let level = inf_scene::decode(&bytes).expect("the island level decodes");
    let mut world = inf_player::level::populate_world(level.entities);
    world.propagate();
    let a = inf_ecs::society::level_archetype(&world);
    println!(
        "the island offers outfit {:?} hair {:?}",
        a.outfit.map(|w| w.mesh),
        a.hair.map(|w| w.mesh)
    );
    assert!(
        a.outfit.is_some() && a.hair.is_some(),
        "the island's own archetype carries no clothes, so every driver and every \
         resident on it goes out bare"
    );
    inf_ecs::crowd::spawn_body(&mut world, driver, &a, DVec3::new(0.0, 0.0, 0.0));
    world.mark_dirty();
    world.propagate();
    let worn = inf_ecs::wearable::wearables_of(&world, driver);
    let visible = worn
        .iter()
        .filter(|g| {
            world
                .entity_of(**g)
                .and_then(|e| {
                    world
                        .world()
                        .get::<inf_ecs::components::ComputedVisibility>(e)
                })
                .map(|c| c.0)
                .unwrap_or(false)
        })
        .count();
    println!(
        "the island's driver wears {} thing(s), {visible} visible",
        worn.len()
    );
    assert_eq!(
        worn.len(),
        2,
        "the island's own driver is wearing {} thing(s)",
        worn.len()
    );
    assert_eq!(
        visible, 2,
        "the island's driver has its clothes and {visible} of them are visible — \
         a wearable the visibility chain drops is a wearable no projector draws"
    );
}

/// **THE TWO HOSTS TAKE THE WEARABLE DOOR, AND READ THE WEARER** — the host half
/// of finding F8, named for what it reads: two SOURCES.
///
/// # Why a source pin and not a value comparison
///
/// The player's projector is `inf_player::render::project_scene_full`, a pure
/// function of a `RuntimeSim`; the editor's is
/// `inf_viewport::host::EngineHost::rebuild_scene`, a method on a live GPU host
/// with a surface, a device and a renderer. There is no headless door onto the
/// second, so the two cannot be handed one world and compared value for value
/// from a test — which is exactly why `projector_mirror` exists and why it pins
/// SOURCE. This arm is that pin's wearable clause, restated where a reader of
/// the wearables gate will meet it: if either host stopped taking the Ring-0
/// door, or read `entity`/`guid` where the other reads `pose_entity`/`pose_guid`,
/// one host would draw a garment in its own bind pose, at its own
/// un-interpolated position, on a body the other host was drawing it on
/// correctly.
///
/// Five fragments, because five separate reads follow from the door and each of
/// them is a divergence on its own: the door itself, the model-space lift, the
/// pose store, the animation player and the crowd tier.
#[test]
fn the_two_projectors_take_the_wearable_door() {
    // Whitespace-STRIPPED, because rustfmt decides where a call wraps and a pin
    // a formatter can break is a pin that reports a divergence nobody made.
    let read = |rel: &str| -> String {
        std::fs::read_to_string(repo().join(rel))
            .unwrap_or_else(|e| panic!("{rel}: {e}"))
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect()
    };
    let player = read("runtime/inf-player/src/render.rs");
    let editor = read("editor/crates/inf-viewport/src/host.rs");
    const SHARED: [&str; 5] = [
        "inf_ecs::wearable::pose_source(world,entity,guid)",
        "inf_ecs::pose::model_to_world(world,pose_entity)",
        "inf_ecs::pose::evaluated_pose(world,pose_guid)",
        "get::<inf_ecs::components::AnimPlayer>(pose_entity)",
        "get::<inf_ecs::crowd::CrowdAgent>(pose_entity)",
    ];
    for f in SHARED {
        assert!(
            player.contains(f),
            "the PLAYER's projector no longer contains `{f}` — a wearable it \
             draws is a wearable the editor is drawing on a different body"
        );
        assert!(
            editor.contains(f),
            "the EDITOR's projector no longer contains `{f}` — a wearable it \
             draws is a wearable the player is drawing on a different body"
        );
    }
    // ANTI-VACUITY: a pin over a file it could not read passes for the wrong
    // reason, and a pin over a fragment that is nowhere at all is worse.
    assert!(
        player.len() > 10_000 && editor.len() > 10_000,
        "one of the two projector sources did not read"
    );
    println!(
        "both projectors carry all {} wearable fragments ({} and {} bytes of source)",
        SHARED.len(),
        player.len(),
        editor.len()
    );
}
