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
//! * `pie_equals_shipping_on_a_dressed_character`
//! * `what_a_wearable_costs` — draws, palette bytes, and the crowd at 1 000.
//! * `the_combined_bodys_eyes_have_no_section_of_their_own` — clause 3's
//!   REFUSAL, as a measurement that will go red the day the combine changes.

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
        "{n} `Far` agents are still carrying a garment draw — the tier rule is          the crowd's whole clothing budget"
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

/// **The two hosts wear the same clothes the same way.**
///
/// The MIRROR pin `projector_mirror` makes is a source one; this is the value.
/// Both hosts resolve a wearable's pose through the ONE Ring-0 door
/// `inf_ecs::wearable::pose_source`, so the question a gate can ask is whether
/// that door answers the same thing about the same world twice — including
/// through the `AttachedTo` shape the editor never writes and a runtime equip
/// does.
#[test]
fn pie_equals_shipping_on_a_dressed_character() {
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
         skinned draws        {}\n  \
         distinct palettes    {} ({} bytes each)\n  \
         geometry             {} triangles over {} mesh slots",
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

// ─────────────────────────────────────────────────────────────────────────────
// (7) CLAUSE 3'S REFUSAL, AS A MEASUREMENT
// ─────────────────────────────────────────────────────────────────────────────

/// **The combined body's eyes have no section of their own** — clause 3, refused
/// with the measurement that refuses it.
///
/// Three facts, all read off the island's own rebound hero mesh and none
/// inferred:
///
/// * **the eye GEOMETRY is there.** A 40 × 40 × 50 mm box at the right eye holds
///   2 478 vertices, whose distances from their own centroid run 2.93 … 34.59 mm
///   with a plateau at 8–17 mm over a 40.1 mm z-span. That is a closed eyeball.
/// * **its own UV square SURVIVED the combine.** Those vertices span
///   u 0.0029 … 0.9922 — a full tile, which is what an eye texture is addressed
///   with, and not the sliver a repack would have left.
/// * **and it is in the same SECTION as the head skin.** The combined mesh has
///   one section per UV *tile* (the import's UDIM split) and the eye island is in
///   tile 0 with the face, overlapping it. The material a triangle draws with in
///   this engine is a property of its section, so there is no address at which to
///   bind `MI_EyeL_Baked` — and the head albedo at those UVs is skin
///   (luminance 59.6, sd 6.9, against a cheek control of 64.5; an iris and a
///   sclera in one box would be a standard deviation three times that).
///
/// Separating them needs a rule the combine destroyed and this wave did not
/// build: a connected-component split of the tile-0 section, matched against the
/// FACE mesh's own primitives. Clause 3 is REFUSED on that, and the ledger names
/// the three doors.
///
/// The arm asserts the state, so it goes RED the day a combine — or an
/// import-side split — gives the eyes a section of their own, which is the day
/// clause 3 becomes cheap.
#[test]
fn the_combined_bodys_eyes_have_no_section_of_their_own() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — it is local-only content and CI has none");
        return;
    };
    let p = content.join("Starter_Body.inf_mesh");
    if !p.is_file() {
        eprintln!("SKIP: no rebound body in this project");
        return;
    }
    let mesh: inf_mesh::MeshAsset = inf_asset::decode(&std::fs::read(&p).unwrap()).unwrap();
    if mesh.triangle_count() < 50_000 {
        eprintln!(
            "SKIP: the body at the committed GUID is {} triangles — this is the              committed low-poly one, not a rebound MetaHuman",
            mesh.triangle_count()
        );
        return;
    }
    // Which SECTION each eye vertex is in, and how wide its uv island is.
    let mut per_section: Vec<usize> = vec![0; mesh.submeshes.len()];
    let (mut umin, mut umax) = (f32::MAX, f32::MIN);
    for (si, sub) in mesh.submeshes.iter().enumerate() {
        for v in &sub.vertices {
            let [x, y, z] = v.position;
            if (0.015..0.055).contains(&x)
                && (1.645..1.685).contains(&y)
                && (0.085..0.135).contains(&z)
            {
                per_section[si] += 1;
                umin = umin.min(v.uv[0]);
                umax = umax.max(v.uv[0]);
            }
        }
    }
    let n: usize = per_section.iter().sum();
    let sections_with_eye = per_section.iter().filter(|c| **c > 0).count();
    println!(
        "
=== CLAUSE 3, REFUSED ===
           the body has {} sections over {} triangles
           the right eye is {n} vertices, u {umin:.4}..{umax:.4}
           and lives in {sections_with_eye} section(s): {per_section:?}",
        mesh.submeshes.len(),
        mesh.triangle_count()
    );
    assert!(
        n > 500,
        "only {n} vertices in the eye box — there is no eyeball in this body,          and clause 3's whole refusal rests on there being one"
    );
    assert!(
        umax - umin > 0.5,
        "the eye's uv island spans only {:.4} of u — the combine repacked it          after all, and the refusal's second fact has changed",
        umax - umin
    );
    // THE REFUSAL ITSELF: the eyes share a section with the skin, so there is no
    // address at which to bind an eye material.
    let biggest = per_section
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| **c)
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        mesh.submeshes[biggest].triangle_count() > 10_000,
        "the eye's own section is only {} triangles — it HAS a section of its          own now, which is clause 3 become cheap: read the ledger and bind          MI_EyeL_Baked to it",
        mesh.submeshes[biggest].triangle_count()
    );
}
