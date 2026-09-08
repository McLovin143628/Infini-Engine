//! **WAVE COV1 — TAKING COVER**, on the island the showcase runs.
//!
//! `inf_ecs::cover`'s tests pin the rules as numbers and
//! `inf-physics/tests/cover_3d.rs` pins them against a three-box fixture. This
//! file asks the only question those two cannot: **is there anything on this
//! island to take cover behind, and does a character actually get behind it.**
//!
//! That question is the wave's own first measurement and it is here at the top
//! of the file for the CHAR1b.2 audit's reason: the mantle's ledge probe found
//! **zero ledges over 289 grounded stances x 8 bearings** around the spawn, and
//! `mantle_high` has never played in the world. A cover probe asks a different
//! question of the same sweeps — a wall FACE rather than a ledge TOP — so it
//! ought to answer where the ledge probe did not, and "ought to" is not a
//! measurement. [`the_cover_census_over_the_island`] is.
//!
//! **Every arm reads the WORLD or the JOINTS.** The measured surface top, the
//! capsule's half-height, the head joint's world position, the ray's hit
//! entity, the transform's distance from the face. None of them reads a state
//! name, and a bind-posed hero snapped to a wall passes none of them.
//!
//! Arms that need the island SKIP with a printed reason when it is not built —
//! `char1a3_gate`'s rule, because the ALS clips and the MetaHumans are licensed
//! content that never enters this repository and CI must not have a red gate
//! about it.

use std::path::{Path, PathBuf};

use inf_ecs::components::{CharacterMovement, MovementMode, MovementRefusal, Transform};
use inf_ecs::cover::{CoverClass, CoverSide};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repository root")
}

/// The island project this machine builds locally, or `None`.
fn island_project() -> Option<PathBuf> {
    let p = repo().join("../island-build/project/Content");
    p.is_dir().then(|| p.canonicalize().unwrap_or(p))
}

/// The island level, loaded loose out of the local project — `char1b_gate`'s own
/// door, mirrored so this file can stand its hero on the real ground.
fn loose_sim(content: &Path, slug: &str) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_cloth_assets(inf_player::level::load_cloth_assets_from_dir(content))
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

fn hero_pos(sim: &RuntimeSim, hero: uuid::Uuid) -> glam::DVec3 {
    let w = sim.world();
    w.entity_of(hero)
        .and_then(|e| w.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .expect("the hero has a transform")
}

fn hero_cm(sim: &RuntimeSim, hero: uuid::Uuid) -> CharacterMovement {
    let w = sim.world();
    w.entity_of(hero)
        .and_then(|e| w.world().get::<CharacterMovement>(e))
        .cloned()
        .expect("the hero has a movement component")
}

fn hero_half(sim: &RuntimeSim, hero: uuid::Uuid) -> f64 {
    let w = sim.world();
    w.entity_of(hero)
        .and_then(|e| w.world().get::<inf_ecs::components::Collider3D>(e))
        .map(|c| c.half_extents.y)
        .expect("the hero has a capsule")
}

fn hero_radius(sim: &RuntimeSim, hero: uuid::Uuid) -> f64 {
    let w = sim.world();
    w.entity_of(hero)
        .and_then(|e| w.world().get::<inf_ecs::components::Collider3D>(e))
        .map(|c| c.radius)
        .unwrap_or(0.3)
}

/// Put the hero at `x, z` a metre and a fifth above the ground under it, facing
/// `bearing`, and let it settle — `char1b_gate::hero_to`, mirrored, with the
/// settle exposed because the census pays it 282 times.
fn hero_to(sim: &mut RuntimeSim, hero: uuid::Uuid, x: f64, z: f64, bearing: f64) {
    hero_to_settling(sim, hero, x, z, bearing, 180);
}

/// [`hero_to`], with the settle named.
///
/// 180 steps is `char1b_gate`'s own number and what every station arm uses. The
/// census uses **90**, and the difference is priced rather than assumed: a
/// station that has not grounded in 1.5 s is skipped by the `grounded` check
/// either way, so the shorter settle costs coverage of the stations on the
/// steepest ground and buys back seven minutes of a fourteen-minute arm.
fn hero_to_settling(
    sim: &mut RuntimeSim,
    hero: uuid::Uuid,
    x: f64,
    z: f64,
    bearing: f64,
    settle: usize,
) {
    let y = hero_pos(sim, hero).y + 1.2;
    {
        let w = sim.world_mut();
        let e = w.entity_of(hero).expect("the hero is in the world");
        if let Some(mut t) = w.world_mut().get_mut::<Transform>(e) {
            t.translation.x = x;
            t.translation.y = y;
            t.translation.z = z;
            t.rotation.y = bearing;
        }
        if let Some(mut c) = w.world_mut().get_mut::<CharacterMovement>(e) {
            c.runtime.velocity = inf_ecs::math::Vec3d::ZERO;
            c.runtime.aim_yaw_deg = bearing;
            c.runtime.body_yaw_deg = bearing;
            c.runtime.target_yaw_deg = bearing;
            c.runtime.cover = inf_ecs::cover::CoverState::default();
            c.mode = MovementMode::Grounded;
        }
    }
    for _ in 0..settle {
        sim.step_once(RuntimeInput::default());
    }
}

/// **The exclusion set the ENGINE's own cover press uses** — ALS's
/// `IgnoreOnlyPawn`, made out of what this engine has.
///
/// Every character's collider, not only the hero's. A gate helper that excluded
/// only the hero measures a different question from the one the movement step
/// asks, and the difference is not academic: the island's crowd walks past the
/// spawn as kinematic capsules 2.27 m tall, and the first cut of `find_station`
/// duly reported a PERSON as HIGH cover at the spawn — then the press refused
/// it, because `try_cover` excludes every character, and four arms failed
/// pointing at a station that was a pedestrian.
fn pawn_exclusion(
    sim: &RuntimeSim,
    hero: uuid::Uuid,
) -> std::collections::BTreeSet<inf_physics::d3::ColliderId3D> {
    let mut e = std::collections::BTreeSet::new();
    if let Some(c) = sim.bridge3d().collider_of(hero) {
        e.insert(c);
    }
    for g in inf_ecs::movement::movement_targets(sim.world()) {
        if let Some(c) = sim.bridge3d().collider_of(g) {
            e.insert(c);
        }
    }
    e
}

/// **The head joint's world position**, off the evaluated pose — the joint every
/// exposure claim in this file is measured on.
///
/// `None` for a character whose skeleton has not resolved, which is what a
/// failure message needs to be able to say: "the head is behind the cover" and
/// "there is no head" are different facts.
fn head_world(sim: &RuntimeSim, hero: uuid::Uuid) -> Option<glam::DVec3> {
    let content = island_project()?;
    let (rigs, _, _) = inf_player::level::load_anim_assets_from_dir(&content);
    let p = inf_ecs::pose::evaluated_pose(sim.world(), hero)?;
    let rig = rigs.get(&p.skeleton)?;
    let idx = rig
        .role_index()
        .first(inf_anim::BoneRoleKind::Head, inf_anim::BoneSide::Center)?;
    let to_world = inf_ecs::pose::model_to_world_of(sim.world(), hero)?;
    let g = inf_anim::pose::global_transforms(&rig.skeleton, &p.pose);
    let t = g[idx as usize].to_scale_rotation_translation().2;
    Some(to_world.transform_point3(glam::DVec3::new(
        f64::from(t.x),
        f64::from(t.y),
        f64::from(t.z),
    )))
}

/// Hold `keys` for `steps` fixed steps.
fn go(sim: &mut RuntimeSim, steps: usize, keys: &[&str], axes: &[(&str, f32)]) {
    let ax: std::collections::BTreeMap<String, f32> =
        axes.iter().map(|(n, v)| ((*n).to_string(), *v)).collect();
    for _ in 0..steps {
        sim.step_once(RuntimeInput::with_down(keys.to_vec()).with_axes(ax.clone()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (0) THE CENSUS — measured FIRST, and written down before anything is built
//     on it
// ─────────────────────────────────────────────────────────────────────────────

/// **HOW MUCH COVER IS THERE ON THE ISLAND, AND WHAT IS IT?**
///
/// The wave's first measurement, and the CHAR1b.2 audit's method: a grid of
/// grounded stances around the spawn, eight bearings at each, one
/// `probe_cover` per bearing, tallied by CLASS and by the LABEL of the thing it
/// found. The mantle's own census over the same stations found **zero ledges**;
/// this one asks a different question of the same sweeps and prints what it
/// gets, so the number is in the ledger rather than in a hope.
///
/// It asserts three things and each is deliberately weak, because a census is a
/// measurement rather than a claim:
///
/// * the probe RAN — the sweep count is not zero, so a census reporting nothing
///   is a census of an empty world and not of a switched-off probe;
/// * a kerb is never `Low` or `High` — the floor's refusal, which holds
///   whatever the island contains;
/// * and every classified surface has a LABEL that names a family, which is
///   carried 162's door doing its job: a façade with no ECS entity still says
///   what it is.
///
/// The rest is printed. A reader wanting "how much cover does Harbour City
/// have" reads the table this prints, and the implementer's report carries it.
#[test]
fn the_cover_census_over_the_island() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let spawn = hero_pos(&sim, hero);
    println!(
        "\n=== COV1 CENSUS — cover around the island spawn ({:.1}, {:.1}, {:.1}) ===",
        spawn.x, spawn.y, spawn.z
    );

    // The same lattice the CHAR1b.2 ledge census walked: 17 x 17 stations at
    // 8 m, eight bearings each. Only the stations the terrain actually grounds
    // count, so the denominator is what the world offered rather than what the
    // loop asked for.
    let spacing = 8.0;
    let half = 8i32;
    let bearings = 8;
    let mut grounded = 0usize;
    let mut probes = 0usize;
    let mut sweeps = 0u64;
    let mut low = 0usize;
    let mut high = 0usize;
    let mut by_family: std::collections::BTreeMap<String, usize> = Default::default();
    let mut refusals: std::collections::BTreeMap<String, usize> = Default::default();
    let settings = inf_physics::d3::CoverSettings::default();

    for ix in -half..=half {
        for iz in -half..=half {
            let x = spawn.x + f64::from(ix) * spacing;
            let z = spawn.z + f64::from(iz) * spacing;
            hero_to_settling(&mut sim, hero, x, z, 0.0, 90);
            let cm = hero_cm(&sim, hero);
            if !cm.runtime.grounded {
                continue;
            }
            grounded += 1;
            let here = hero_pos(&sim, hero);
            let radius = hero_radius(&sim, hero);
            let halfh = cm.half_height_for(MovementMode::Grounded);
            let feet = here - glam::DVec3::Y * (halfh + radius);
            let exclude = pawn_exclusion(&sim, hero);
            for b in 0..bearings {
                let a = std::f64::consts::TAU * f64::from(b) / f64::from(bearings);
                let dir = glam::DVec3::new(inf_math::psin64(a), 0.0, inf_math::pcos64(a));
                let p = {
                    let bridge = sim.bridge3d_mut();
                    inf_physics::d3::probe_cover(
                        bridge,
                        feet,
                        dir,
                        radius,
                        halfh,
                        cm.slope_limit_deg,
                        &settings,
                        &exclude,
                    )
                };
                probes += 1;
                sweeps += u64::from(p.sweeps);
                match p.class {
                    CoverClass::Low => {
                        low += 1;
                        *by_family
                            .entry(format!("Low  / {:?}", p.label.family))
                            .or_default() += 1;
                    }
                    CoverClass::High => {
                        high += 1;
                        *by_family
                            .entry(format!("High / {:?}", p.label.family))
                            .or_default() += 1;
                    }
                    CoverClass::None => {
                        *refusals.entry(format!("{:?}", p.refusal)).or_default() += 1;
                        // **A kerb is never cover**, whatever else the island
                        // holds. The floor's refusal, asserted on every station.
                        assert!(
                            p.label.family != inf_physics::d3::ColliderFamily::Kerb
                                || !p.class.is_cover(),
                            "a kerb classified as cover at ({x:.1}, {z:.1})"
                        );
                    }
                }
            }
        }
    }
    println!(
        "  stations {grounded} grounded x {bearings} bearings = {probes} probes, \
         {sweeps} shape casts"
    );
    println!("  LOW {low}   HIGH {high}   NONE {}", probes - low - high);
    println!("  by class and family:");
    for (k, v) in &by_family {
        println!("    {k:>34} : {v}");
    }
    println!("  refusals:");
    for (k, v) in &refusals {
        println!("    {k:>34} : {v}");
    }
    println!("=== end census ===\n");

    assert!(
        grounded > 0,
        "no station grounded — the census measured nothing"
    );
    assert!(
        sweeps > 0,
        "the probe spent no shape casts over {probes} probes — it is switched \
         off, and a census of a switched-off probe reads exactly like a census \
         of an empty island"
    );
    // Every classified surface names a family (carried 162's door). `Entity` is
    // a legal answer — a parked car IS an entity — and what must never happen is
    // a classified surface the label table cannot speak about at all, which is
    // what the `describe` below would print as a bare uuid.
    assert!(
        by_family.keys().all(|k| !k.ends_with("Unknown")),
        "a classified cover surface has no family: {by_family:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (1) FINDING A STATION — the arms search the island rather than naming a
//     coordinate, so a level edit moves them instead of breaking them
// ─────────────────────────────────────────────────────────────────────────────

/// One place on the island where a cover surface of a wanted class is, and the
/// bearing that faces it.
#[derive(Clone, Copy, Debug)]
struct Station {
    x: f64,
    z: f64,
    bearing_deg: f64,
    class: CoverClass,
    top_m: f64,
    left_m: f64,
    right_m: f64,
    family: inf_physics::d3::ColliderFamily,
}

/// What an arm is looking for.
///
/// A struct rather than six positional arguments, and every field is a fact
/// about the GEOMETRY an arm needs rather than a coordinate: "a façade with at
/// least a metre and a half of run and a real end to it" is what the slide and
/// the peek are about, and the island is asked where one is.
#[derive(Clone, Copy, Debug)]
struct StationWant {
    class: CoverClass,
    family: Option<inf_physics::d3::ColliderFamily>,
    /// How much surface, left plus right, the station must have. `0` for any.
    min_run_m: f64,
    /// Whether the surface must actually **END** on one side — a free corner
    /// with nothing behind it.
    ///
    /// This is the difference between a corner and a seam. Harbour City's
    /// façades are runs of ADJACENT structural boxes, so the edge of one box is
    /// the edge of nothing: a character leaning around it is still behind the
    /// next box, and the wave's first exposure arm measured exactly that (the
    /// ray hit `structure #6063` after leaning past `#6062`). A free corner is
    /// one where a probe a metre beyond the end finds no cover at all.
    free_corner: bool,
    /// How far out to search, in lattice cells.
    reach: i32,
    /// How far apart the cells are, metres.
    spacing: f64,
}

impl StationWant {
    fn of(class: CoverClass) -> Self {
        Self {
            class,
            family: None,
            min_run_m: 0.0,
            free_corner: false,
            reach: 8,
            spacing: 8.0,
        }
    }
    fn structure(mut self) -> Self {
        self.family = Some(inf_physics::d3::ColliderFamily::Structure);
        self
    }
    fn with_run(mut self, m: f64) -> Self {
        self.min_run_m = m;
        self
    }
    fn with_free_corner(mut self) -> Self {
        self.free_corner = true;
        self
    }
}

/// **Walk a lattice around the spawn and find cover of `want`.**
///
/// Deliberately a SEARCH and not a coordinate: a hard-coded `(x, z)` is a claim
/// about a level's contents that goes stale the day a wave moves a car, and
/// this campaign has already been caught by one of those. The arms below say
/// what KIND of thing they want and the island says where it is; a failure is
/// then "the island has no LOW cover within 64 m of the spawn", which is a
/// finding rather than a broken test.
///
/// `family` narrows it — a parked car is a document `Entity`, a façade is a
/// `Structure` with no entity at all (carried 162), and an arm that wants one
/// must not accidentally measure the other.
fn find_station(sim: &mut RuntimeSim, hero: uuid::Uuid, want: StationWant) -> Option<Station> {
    let (reach, spacing) = (want.reach, want.spacing);
    let spawn = hero_pos(sim, hero);
    let settings = inf_physics::d3::CoverSettings::default();
    // Rings outward from the spawn, so the nearest station wins and the arm's
    // own drive is short.
    let mut cells: Vec<(i32, i32)> = Vec::new();
    for ix in -reach..=reach {
        for iz in -reach..=reach {
            cells.push((ix, iz));
        }
    }
    cells.sort_by_key(|(a, b)| a.abs().max(b.abs()));
    for (ix, iz) in cells {
        let x = spawn.x + f64::from(ix) * spacing;
        let z = spawn.z + f64::from(iz) * spacing;
        hero_to(sim, hero, x, z, 0.0);
        let cm = hero_cm(sim, hero);
        if !cm.runtime.grounded {
            continue;
        }
        let here = hero_pos(sim, hero);
        let radius = hero_radius(sim, hero);
        let halfh = cm.half_height_for(MovementMode::Grounded);
        let feet = here - glam::DVec3::Y * (halfh + radius);
        let exclude = pawn_exclusion(sim, hero);
        for b in 0..16 {
            let deg = 360.0 * f64::from(b) / 16.0;
            let a = deg.to_radians();
            let dir = glam::DVec3::new(inf_math::psin64(a), 0.0, inf_math::pcos64(a));
            let p = {
                let bridge = sim.bridge3d_mut();
                inf_physics::d3::probe_cover(
                    bridge,
                    feet,
                    dir,
                    radius,
                    halfh,
                    cm.slope_limit_deg,
                    &settings,
                    &exclude,
                )
            };
            if p.class != want.class {
                continue;
            }
            if want.family.is_some_and(|f| p.label.family != f) {
                continue;
            }
            if p.left_m + p.right_m < want.min_run_m {
                continue;
            }
            if want.free_corner && !ends_freely(sim, hero, &p, dir, &exclude, &settings) {
                continue;
            }
            // **Leave the hero standing here, facing it.** The arms do NOT
            // re-place afterwards, and that is a repair rather than a
            // convenience: a second 180-step settle is three seconds of island,
            // and in three seconds the traffic fleet drives away from the
            // station the search just found. Four arms failed that way, at a
            // station whose surface was a moving car.
            hero_to(sim, hero, here.x, here.z, deg);
            return Some(Station {
                x: here.x,
                z: here.z,
                bearing_deg: deg,
                class: p.class,
                top_m: p.top_m,
                left_m: p.left_m,
                right_m: p.right_m,
                family: p.label.family,
            });
        }
    }
    None
}

/// **Does this surface actually END on one side**, with nothing behind it?
///
/// A probe taken a metre past the shorter extent, in the same direction. A
/// façade made of adjacent boxes answers YES to "is there still cover here" and
/// is therefore a SEAM rather than a corner; a building's actual end answers no.
///
/// Costs one probe per candidate, and only for the arms that ask.
fn ends_freely(
    sim: &mut RuntimeSim,
    hero: uuid::Uuid,
    p: &inf_physics::d3::CoverProbe,
    dir: glam::DVec3,
    exclude: &std::collections::BTreeSet<inf_physics::d3::ColliderId3D>,
    settings: &inf_physics::d3::CoverSettings,
) -> bool {
    let cm = hero_cm(sim, hero);
    let radius = hero_radius(sim, hero);
    let halfh = cm.half_height_for(MovementMode::Grounded);
    let left = inf_ecs::cover::tangent_left(p.normal_v()).to_dvec3();
    // The shorter side is the one a slide will reach first.
    let (side, run) = if p.left_m <= p.right_m {
        (left, p.left_m)
    } else {
        (-left, p.right_m)
    };
    // **Three samples, not one.** Harbour City's shops are rows of adjacent
    // structural boxes with narrow gaps between them, and a single probe a
    // metre past the run lands in a gap and calls it a corner. Measured: a
    // station that passed the one-sample test put a peeking head 0.425 m past
    // its own wall's end and straight into `structure #6059` — the next box
    // along. A real building end is clear for several metres.
    for d in [0.5, 1.0, 2.0, 3.0] {
        let past = p.anchor + side * (run + d);
        let beyond = {
            let bridge = sim.bridge3d_mut();
            inf_physics::d3::probe_cover(
                bridge,
                past,
                dir,
                radius,
                halfh,
                cm.slope_limit_deg,
                settings,
                exclude,
            )
        };
        if beyond.class.is_cover() {
            return false;
        }
    }
    true
}

/// Take cover from where the hero is standing, and let the snap finish.
///
/// Answers the refusal recorded on the step the press was MADE. Reading it
/// afterwards would read `None` on every run: `MovementRuntime::refusal` is this
/// step's answer and the idle steps that let the snap finish clear it, which is
/// the shape every refusal in this engine has.
fn take_cover(sim: &mut RuntimeSim, hero: uuid::Uuid, steps_after: usize) -> MovementRefusal {
    go(sim, 1, &["cover"], &[]);
    let refusal = hero_cm(sim, hero).runtime.refusal;
    go(sim, steps_after, &[], &[]);
    refusal
}

// ─────────────────────────────────────────────────────────────────────────────
// (2) THE CLASSIFICATION, ON THE ISLAND, OFF THE JOINTS
// ─────────────────────────────────────────────────────────────────────────────

/// **THE HERO CROUCHES BEHIND A LOW SURFACE AND STANDS AGAINST A HIGH ONE** —
/// clause 1 and clause 2, on the world the game is set in.
///
/// The class comes off the WORLD (the probe's measured top against the two
/// thresholds) and the stance off the CAPSULE and the JOINTS (the collider's
/// half-height, and the head joint's height above the feet). A cover system
/// that set a mode and moved nothing passes neither.
#[test]
fn the_islands_hero_crouches_behind_low_cover_and_stands_against_high() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let stand_half = hero_cm(&sim, hero).stand_half_height_m;
    let crouch_half = hero_cm(&sim, hero).crouch_half_height_m;

    // **A STRUCTURE**, on both rows. The census says the island's cover is 231
    // façade boxes and 5 low grammar walls against 11 vehicle entities, and a
    // vehicle is a body that DRIVES: an arm that measured one would be
    // measuring the traffic timetable. The vehicle rows are real cover and the
    // demo loop is where they are photographed.
    let mut measured: Vec<(CoverClass, f64, f64, f64, String)> = Vec::new();
    for want in [CoverClass::High, CoverClass::Low] {
        let Some(st) = find_station(&mut sim, hero, StationWant::of(want).structure()) else {
            println!("  no {want:?} STRUCTURE cover within 64 m of the spawn");
            continue;
        };
        let refusal = take_cover(&mut sim, hero, 60);
        let cm = hero_cm(&sim, hero);
        let half = hero_half(&sim, hero);
        let feet = hero_pos(&sim, hero).y - half - hero_radius(&sim, hero);
        let head = head_world(&sim, hero)
            .map(|h| h.y - feet)
            .unwrap_or(f64::NAN);
        println!(
            "  {want:?} station ({:.1}, {:.1}) bearing {:.0} — family {:?}, top {:.4} m, \
             mode {:?}, class {:?}, capsule half {:.4}, head {:.4} m above the feet",
            st.x,
            st.z,
            st.bearing_deg,
            st.family,
            st.top_m,
            cm.mode,
            cm.runtime.cover.class,
            half,
            head
        );
        assert_eq!(
            cm.mode,
            MovementMode::Cover,
            "the press at a measured {want:?} surface ({:?}, top {:.4} m) did not take cover: \
             refusal {refusal:?}",
            st.family,
            st.top_m,
        );
        assert_eq!(cm.runtime.cover.class, want);
        // The STANCE, off the collider rather than off the class.
        let wanted_half = if want == CoverClass::Low {
            crouch_half
        } else {
            stand_half
        };
        assert!(
            (half - wanted_half).abs() < 1.0e-6,
            "behind {want:?} cover the capsule is {half:.4} and the stance wants {wanted_half:.4}"
        );
        measured.push((want, st.top_m, half, head, format!("{:?}", st.family)));
    }
    assert!(
        !measured.is_empty(),
        "the island offered NO cover of either class within 64 m of the spawn — the census \
         above is the number, and this arm is what says the wave has nothing to stand behind"
    );
    // …and if both classes were found, the two stances are DIFFERENT, which is
    // the whole of "crouching or standing depending on the object". An arm that
    // measured one class could not say that.
    if measured.len() == 2 {
        let a = measured[0].2;
        let b = measured[1].2;
        assert!(
            (a - b).abs() > 0.2,
            "the two classes produced the same capsule ({a:.4} and {b:.4}) — the stance is not \
             coming from the surface"
        );
    }
}

/// **A KERB IS NOT COVER, AND THE PRESS SAYS SO BY NAME** — clause 6's refusal.
///
/// The island's streets carry `ROAD1b`'s kerb slabs, which have **no ECS entity
/// at all**: the label door (carried 162) is what lets this arm say "a kerb" in
/// its failure message rather than a uuid, and `ColliderFamily::is_coverable`
/// is what refuses them by NAME rather than only by height.
#[test]
fn a_kerb_on_the_island_is_never_cover_and_the_refusal_names_it() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let _hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    // Every kerb slab the bridge has admitted near the spawn, by label.
    let kerbs: Vec<uuid::Uuid> = sim
        .bridge3d()
        .labelled_guids()
        .into_iter()
        .filter(|(_, l)| l.family == inf_physics::d3::ColliderFamily::Kerb)
        .map(|(g, _)| g)
        .collect();
    println!("  the bridge holds {} labelled kerb slabs", kerbs.len());
    assert!(
        !kerbs.is_empty(),
        "no kerb slab is labelled near the spawn — either ROAD1b's footways are not admitted \
         here or the label door stopped recording, and this arm cannot tell which"
    );
    // …and none of them is coverable, by NAME.
    for g in &kerbs {
        let l = sim.bridge3d().label_of_guid(*g);
        assert!(
            !l.family.is_coverable(),
            "{} classified as coverable",
            l.describe(sim.world())
        );
    }
    println!(
        "  e.g. {} — refused by family, before any height is compared",
        sim.bridge3d().label_of_guid(kerbs[0]).describe(sim.world())
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (3) THE SLIDE, THE PEEK AND WHAT A BULLET REACHES
// ─────────────────────────────────────────────────────────────────────────────

/// **THE SLIDE STOPS AT THE CORNER, AND THE PEEK PUTS THE HEAD PAST IT** —
/// clauses 3 and 4, measured on the transform and on the HEAD JOINT.
#[test]
fn the_hero_slides_to_a_corner_and_leans_its_head_past_it() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let Some(st) = find_station(
        &mut sim,
        hero,
        StationWant::of(CoverClass::High)
            .structure()
            .with_run(1.5)
            .with_free_corner(),
    ) else {
        eprintln!(
            "SKIP: the island has no HIGH façade with 1.5 m of run and a free corner within \
             64 m of the spawn — see the census"
        );
        return;
    };
    take_cover(&mut sim, hero, 60);
    assert_eq!(hero_cm(&sim, hero).mode, MovementMode::Cover);
    println!(
        "  the wall at ({:.1}, {:.1}): {:?}, extents L {:.3} m / R {:.3} m",
        st.x, st.z, st.family, st.left_m, st.right_m
    );

    // Slide toward whichever end is nearer, so the drive is short and the arm
    // reaches a corner rather than a timeout.
    let (axis, toward_left) = if st.left_m <= st.right_m {
        (-1.0f32, true)
    } else {
        (1.0f32, false)
    };
    let before = hero_pos(&sim, hero);
    let mut at_corner = false;
    for _ in 0..40 {
        go(&mut sim, 15, &[], &[("move_x", axis)]);
        if hero_cm(&sim, hero).runtime.cover.at_corner {
            at_corner = true;
            break;
        }
        if hero_cm(&sim, hero).mode != MovementMode::Cover {
            break;
        }
    }
    let after = hero_pos(&sim, hero);
    let travelled = ((after.x - before.x).powi(2) + (after.z - before.z).powi(2)).sqrt();
    let cm = hero_cm(&sim, hero);
    println!(
        "  slid {travelled:.4} m toward the {} corner; at_corner {} mode {:?} along {:.4}",
        if toward_left { "left" } else { "right" },
        cm.runtime.cover.at_corner,
        cm.mode,
        cm.runtime.cover.along_m
    );
    assert_eq!(
        cm.mode,
        MovementMode::Cover,
        "the slide left cover on its own"
    );
    assert!(
        at_corner,
        "the slide never reached a corner after {travelled:.4} m — the extents said L {:.3} / \
         R {:.3}",
        st.left_m, st.right_m
    );
    assert!(
        travelled > 0.2,
        "the slide covered {travelled:.4} m, which is nothing"
    );

    // ── the PEEK, on the head joint.
    let Some(tucked) = head_world(&sim, hero) else {
        eprintln!("  the hero's skeleton has not resolved — no head to measure");
        return;
    };
    go(&mut sim, 60, &["aim"], &[]);
    let cm = hero_cm(&sim, hero);
    let Some(out) = head_world(&sim, hero) else {
        panic!("the head resolved before the peek and not after it");
    };
    let moved = ((out.x - tucked.x).powi(2) + (out.z - tucked.z).powi(2)).sqrt();
    println!(
        "  peek: side {:?} peek {:.3}, the head moved {moved:.4} m laterally",
        cm.runtime.cover.side, cm.runtime.cover.peek
    );
    assert!(
        cm.runtime.cover.side.is_out(),
        "aiming at a corner did not lean: side {:?}",
        cm.runtime.cover.side
    );
    assert!(
        moved > 0.25,
        "the head moved {moved:.4} m — the peek is a state name and not a displacement"
    );
    // …and it comes back.
    go(&mut sim, 60, &[], &[]);
    let back = head_world(&sim, hero).expect("the head is still there");
    let residual = ((back.x - tucked.x).powi(2) + (back.z - tucked.z).powi(2)).sqrt();
    println!("  un-peek: the head came back to within {residual:.4} m");
    assert!(
        residual < 0.15,
        "releasing aim left the head {residual:.4} m out of cover"
    );
}

/// **A SHOT AT A HERO TUCKED INTO A FAÇADE HITS THE FAÇADE** — clause 4's
/// exposure claim, the half the island can answer, with a RAY through the
/// physics world and the LABEL door naming what it hit.
///
/// # What this arm proves, and what it deliberately hands to the fixture
///
/// The full claim is two rays: tucked in the shot hits the cover, leaning out
/// it reaches the head. Both are proven in
/// `inf-physics/tests/cover_3d.rs::a_shot_at_a_crouched_character_hits_the_cover_and_a_peeking_one_is_exposed`,
/// on a free-standing wall.
///
/// **The island cannot answer the second half, and the reason is geometry
/// rather than the cover system.** Harbour City's shops are rows of ADJACENT
/// structural boxes with more boxes behind them, so "the far side" of a façade
/// is *inside a shop*: a ray fired from six metres beyond the wall starts in
/// the shop's interior and, when the character leans out, simply hits the next
/// box along instead of the one it was behind (measured: `structure #6142`
/// tucked, `structure #6059` leaning). There is no free-standing wall within
/// 64 m of the spawn whose far side is outdoors. That is a real fact about the
/// island the wave found, it is in the report, and PAR1's street furniture and
/// the P19 grammar's garden walls are where free-standing cover arrives.
///
/// So what is measured here is the half that IS true on the island and that a
/// fixture cannot claim: on the REAL geometry, with the REAL hero, the wall it
/// pressed against stops the bullet — and the head is behind that wall's own
/// face, measured on the JOINT.
#[test]
fn a_shot_at_a_hero_tucked_into_a_facade_hits_the_facade() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let Some(st) = find_station(
        &mut sim,
        hero,
        StationWant::of(CoverClass::High).structure(),
    ) else {
        eprintln!("SKIP: the island has no HIGH façade within 64 m of the spawn — see the census");
        return;
    };
    take_cover(&mut sim, hero, 60);
    let cm = hero_cm(&sim, hero);
    assert_eq!(cm.mode, MovementMode::Cover);
    let normal = cm.runtime.cover.normal.to_dvec3();
    let anchor = cm.runtime.cover.anchor.to_dvec3();

    let Some(head) = head_world(&sim, hero) else {
        eprintln!("  the hero's skeleton has not resolved — no head to measure");
        return;
    };
    // **The head is behind the wall's own face**, off the JOINT: the face is
    // `radius + standoff` in along the normal from the anchor, and the head must
    // be on the anchor's side of it.
    let radius = hero_radius(&sim, hero);
    let face = anchor - normal * (radius + inf_ecs::cover::STANDOFF_M);
    let depth = (head - face).dot(normal);
    println!(
        "  the façade at ({:.1}, {:.1}): {:?}, top {:.4} m; the head is {depth:.4} m out from \
         its face",
        st.x, st.z, st.family, st.top_m
    );
    assert!(
        depth > 0.0,
        "the head is {depth:.4} m INSIDE the wall the character is pressed against"
    );

    // The shot, from the far side, aimed at the head's own height.
    let from = head - normal * 6.0;
    let exclude = Default::default();
    let hit = sim
        .bridge3d_mut()
        .world_mut()
        .cast_ray_excluding(from, normal, 20.0, &exclude);
    let Some(hit) = hit else {
        panic!("the ray hit nothing at all — neither the façade nor the character");
    };
    let said = sim.bridge3d().describe_collider(sim.world(), hit.collider);
    let who = sim.bridge3d().guid_of_collider(hit.collider);
    println!("  tucked in, the shot hit {said}");
    assert_ne!(
        who,
        Some(hero),
        "a shot at a character tucked into a façade reached it: the ray hit {said}"
    );
    assert!(
        sim.bridge3d()
            .label_of(hit.collider)
            .is_some_and(|l| l.family == inf_physics::d3::ColliderFamily::Structure),
        "the shot was stopped by {said}, which is not the building"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (4) THE POSE, THE BUDGET, THE MACHINE AND THE CAMERA
// ─────────────────────────────────────────────────────────────────────────────

/// **THE COVER POSE IS NOT THE CROUCH IDLE RENAMED** — the anti-vacuity arm the
/// CHAR1b.1 law demands.
///
/// A character in `MovementMode::Cover` behind a low surface and one merely
/// crouching are in two different stances, and the difference is measured
/// JOINT BY JOINT off the two evaluated poses. Without this arm the whole
/// clause could be satisfied by binding the cover states to the crouch clips.
#[test]
fn the_cover_pose_is_not_the_crouch_idle_wearing_a_different_name() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let Some(st) = find_station(&mut sim, hero, StationWant::of(CoverClass::Low).structure())
        .or_else(|| {
            find_station(
                &mut sim,
                hero,
                StationWant::of(CoverClass::High).structure(),
            )
        })
    else {
        eprintln!("SKIP: the island offered no façade cover within 64 m — see the census");
        return;
    };
    // The control: crouch at the same place, facing the same way, WITHOUT cover.
    hero_to(&mut sim, hero, st.x, st.z, st.bearing_deg);
    go(&mut sim, 1, &["crouch"], &[]);
    go(&mut sim, 120, &[], &[]);
    let crouch_mode = hero_cm(&sim, hero).mode;
    let crouched = inf_ecs::pose::evaluated_pose(sim.world(), hero).map(|p| p.pose.clone());
    // …and the cover pose at the same place.
    hero_to(&mut sim, hero, st.x, st.z, st.bearing_deg);
    take_cover(&mut sim, hero, 120);
    let cover_mode = hero_cm(&sim, hero).mode;
    let covered = inf_ecs::pose::evaluated_pose(sim.world(), hero).map(|p| p.pose.clone());
    let state = inf_ecs::anim_bridge::anim_state(sim.world(), hero).map(|s| s.name.clone());
    println!("  control mode {crouch_mode:?}; cover mode {cover_mode:?}; machine state {state:?}");
    assert_eq!(
        cover_mode,
        MovementMode::Cover,
        "the press did not take cover"
    );
    let (Some(a), Some(b)) = (crouched, covered) else {
        eprintln!("  the hero has no evaluated pose — nothing to compare");
        return;
    };
    assert_eq!(
        a.locals.len(),
        b.locals.len(),
        "two poses of different rigs"
    );
    let mut worst = 0.0f64;
    let mut moved = 0usize;
    for (x, y) in a.locals.iter().zip(b.locals.iter()) {
        let qa = glam::Quat::from_array(x.rotation);
        let qb = glam::Quat::from_array(y.rotation);
        let d = 2.0 * inf_math::pacos64(f64::from(qa.dot(qb).abs().clamp(-1.0, 1.0))).to_degrees();
        if d > 1.0 {
            moved += 1;
        }
        worst = worst.max(d);
    }
    println!(
        "  the cover pose differs from the crouch idle on {moved} joints, worst {worst:.3} deg"
    );
    assert!(
        worst > 5.0 && moved >= 3,
        "the cover pose is {worst:.3} deg from the crouch idle on {moved} joints — it is the \
         crouch idle wearing a different name"
    );
    // …and the machine is IN one of the cover states, which is the other half:
    // a pose that differs because the character is standing somewhere else is
    // not a cover clip playing.
    assert!(
        state.as_deref().is_some_and(|s| s.starts_with("cover_")),
        "the machine is in {state:?} rather than a cover state — the clip is not bound"
    );
}

/// **A CHARACTER THAT IS NOT IN COVER SPENDS ZERO SHAPE CASTS LOOKING FOR IT** —
/// the budget arm, on the island and on the step's own counter.
#[test]
fn nothing_on_the_island_probes_for_cover_until_somebody_asks() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    // Walk about for a while, near whatever the spawn is next to.
    let mut spent = 0u64;
    for i in 0..600 {
        let ax: std::collections::BTreeMap<String, f32> = if i % 2 == 0 {
            [("move_y".to_string(), 1.0f32)].into()
        } else {
            Default::default()
        };
        sim.step_once(RuntimeInput::with_down(Vec::<&str>::new()).with_axes(ax));
        spent += u64::from(hero_cm(&sim, hero).runtime.cover.sweeps);
    }
    println!("  600 steps of walking about: {spent} cover shape casts");
    assert_eq!(
        spent, 0,
        "a hero that never pressed cover spent {spent} shape casts looking for it"
    );
    // …and the press costs something, so the zero is a measurement rather than a
    // probe that is switched off.
    go(&mut sim, 1, &["cover"], &[]);
    let pressed = hero_cm(&sim, hero).runtime.cover.sweeps;
    println!("  the press spent {pressed} shape casts");
    assert!(
        pressed > 0,
        "the press spent nothing either — the probe is not running at all"
    );
}

/// **THE ISLAND'S BAKED MACHINE CARRIES THE COVER STATES**, and a regenerate
/// would say so if it did not — carried 141's arm.
///
/// The island's `.inf_sm` files are BAKED CONTENT: `als.rs` may be edited freely
/// and every island arm stays green until somebody re-runs `inf-import --into
/// ../island-build/project --rebind-graph m f`. This arm is what makes that
/// visible: it rebuilds the locomotion graph from the CURRENT `als.rs` with the
/// SAME resolver the importer uses, and compares the state NAMES against what
/// is on disk. A stale machine fails by name, and the message says the verb.
#[test]
fn the_islands_machine_has_the_cover_states_a_rebuild_would_produce() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    let (_, _, machines) = inf_player::level::load_anim_assets_from_dir(&content);
    if machines.is_empty() {
        eprintln!("SKIP: the island project has no .inf_sm");
        return;
    }
    // What a rebuild WOULD produce, from this source tree's own table. The
    // resolver answers every name, because what is being compared is the state
    // set and not the bindings.
    let (fresh, _) = inf_anim::als::build_locomotion_graph(&|_| Some([0u8; 16]));
    let want: std::collections::BTreeSet<&str> = fresh
        .states
        .iter()
        .map(|s| s.name.as_str())
        .filter(|n| n.starts_with("cover_"))
        .collect();
    assert_eq!(
        want.len(),
        4,
        "the table itself no longer produces four cover states: {want:?}"
    );
    let mut checked = 0usize;
    let mut stale: Vec<String> = Vec::new();
    for (guid, m) in &machines {
        // Only the locomotion graphs: a machine with no `idle` is somebody
        // else's.
        if !m.machine.states.iter().any(|s| s.name == "idle") {
            continue;
        }
        checked += 1;
        let have: std::collections::BTreeSet<&str> =
            m.machine.states.iter().map(|s| s.name.as_str()).collect();
        let missing: Vec<&&str> = want.iter().filter(|n| !have.contains(*n)).collect();
        if !missing.is_empty() {
            stale.push(format!(
                "{guid}: missing {missing:?} of {} states",
                have.len()
            ));
        }
    }
    println!(
        "  {checked} locomotion machines on the island; {} stale",
        stale.len()
    );
    assert!(checked > 0, "no locomotion machine on the island to check");
    assert!(
        stale.is_empty(),
        "the island's baked `.inf_sm` is STALE — `als.rs` has cover states this content does \
         not. Re-run `inf-import --into ../island-build/project --rebind-graph m f`.\n  {}",
        stale.join("\n  ")
    );
    // …and the parameters came with them, because an edge that compares against
    // a parameter the machine does not declare is an edge that never fires.
    for (guid, m) in &machines {
        if !m.machine.states.iter().any(|s| s.name == "idle") {
            continue;
        }
        for p in [
            inf_anim::als::COVER_VAR,
            inf_anim::als::COVER_SIDE_VAR,
            inf_anim::als::COVER_PEEK_VAR,
        ] {
            assert!(
                m.machine.params.iter().any(|q| q.name == p),
                "{guid} declares no `{p}` — the cover edges compare against nothing"
            );
        }
    }
}

/// **THE COVER CAMERA IS A CLAIM ON THE DIRECTOR, AND NEVER A SECOND CAMERA** —
/// clause 4's camera half.
///
/// Measured on the director's own holder: while the hero is in cover the camera
/// is held by the `Override` layer under the cover tag, and the step it leaves
/// the holder goes back to gameplay. A second camera would show up as a pose
/// nothing on the stack asked for.
#[test]
fn the_cover_camera_is_an_override_claim_that_stops_by_not_asking() {
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content");
        return;
    };
    if !content.join("VancouverIsland.inf_lvl").is_file() {
        eprintln!("SKIP: no VancouverIsland.inf_lvl");
        return;
    }
    let mut sim = loose_sim(&content, "VancouverIsland");
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island has a pawn");
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
    }
    let Some(st) = find_station(
        &mut sim,
        hero,
        StationWant::of(CoverClass::High).structure(),
    ) else {
        eprintln!("SKIP: the island offered no façade cover within 64 m — see the census");
        return;
    };
    let _ = st;
    let before = sim.camera().director.holder().map(|h| h.0);
    take_cover(&mut sim, hero, 60);
    let cm = hero_cm(&sim, hero);
    let holder = sim.camera().director.holder();
    let arm_in = sim.camera().arm_m;
    println!("  before {before:?}; in cover the camera is held by {holder:?}, arm {arm_in:.4} m");
    assert_eq!(cm.mode, MovementMode::Cover);
    assert_eq!(
        holder.map(|h| h.0),
        Some(inf_ecs::camera::CameraLayer::Override),
        "the cover camera is not on the director at all"
    );
    assert_eq!(
        holder.map(|h| h.2),
        Some(inf_ecs::camera::CAMERA_TAG_COVER),
        "something else is holding the Override layer"
    );
    // Leave, and it hands back by not asking.
    go(&mut sim, 1, &["cover"], &[]);
    go(&mut sim, 120, &[], &[]);
    let after = sim.camera().director.holder().map(|h| h.0);
    println!("  after leaving cover the camera is held by {after:?}");
    assert_ne!(hero_cm(&sim, hero).mode, MovementMode::Cover);
    assert!(
        after != Some(inf_ecs::camera::CameraLayer::Override),
        "the cover claim outlived the cover"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (5) NPCs UNDER FIRE
// ─────────────────────────────────────────────────────────────────────────────

/// **AN OFFICER UNDER FIRE REACHES COVER, AND ONE THAT IS NOT UNDER FIRE PROBES
/// NOTHING** — clause 5, both halves, on a fixture rather than on the island.
///
/// A fixture and not the island because the claim is about the PASS: a
/// responder, a wall, and a shot beside it. The island half of clause 5 is the
/// demo loop's, where a real dispatch sends a real crew.
///
/// The seam is stated in the arm as it is in the code: this wave points the
/// officer at the threat and leans it out of cover on a duty cycle, and
/// **WPN2e wires the firing**.
#[test]
fn an_officer_under_fire_takes_cover_and_one_that_is_not_probes_nothing() {
    use inf_ecs::components::{
        BodyKind3D, CharacterController3D, Collider3D, ColliderShape3DKind, RigidBody3D,
    };
    use inf_ecs::math::Vec3d;

    const OFFICER: uuid::Uuid = uuid::Uuid::from_u128(0xC0_5001);
    const GROUND: uuid::Uuid = uuid::Uuid::from_u128(0xC0_5002);
    const WALL: uuid::Uuid = uuid::Uuid::from_u128(0xC0_5003);
    const DT: f64 = 1.0 / 60.0;

    let build = || {
        let mut w = inf_ecs::EcsWorld::new();
        let mut b = inf_physics::PhysicsBridge3D::new(glam::DVec3::new(0.0, -9.81, 0.0));
        // A floor and a wall the officer can get behind.
        for (guid, centre, half) in [
            (
                GROUND,
                glam::DVec3::new(0.0, -0.5, 0.0),
                glam::DVec3::new(60.0, 0.5, 60.0),
            ),
            (
                WALL,
                glam::DVec3::new(0.0, 1.5, 6.0),
                glam::DVec3::new(6.0, 1.5, 0.3),
            ),
        ] {
            let e = w.spawn_with_guid(guid, "Block", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::from_dvec3(centre);
            w.world_mut().entity_mut(e).insert((
                RigidBody3D {
                    kind: BodyKind3D::Static,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: Vec3d::from_dvec3(half),
                    ..Default::default()
                },
                t,
            ));
        }
        // The officer, five metres south of the wall.
        let cm = CharacterMovement::default();
        let e = w.spawn_with_guid(OFFICER, "Officer", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, cm.stand_half_height_m + 0.3, 1.0);
        w.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(0.3, cm.stand_half_height_m, 0.3),
                radius: 0.3,
                ..Default::default()
            },
            CharacterController3D::default(),
            cm,
            t,
        ));
        w.mark_dirty();
        w.propagate();
        inf_ecs::dispatch::set_responder(&mut w, OFFICER, true);
        (w, b)
    };

    // ── the control: nobody is shooting.
    let (mut w, mut b) = build();
    let mut quiet: inf_physics::d3::NpcCoverReport = Default::default();
    for i in 0..600 {
        b.sync_from_world(&w);
        let r = inf_physics::d3::step_npc_cover(&mut w, &mut b, &[], 45.0, i, DT);
        quiet.probes += r.probes;
        quiet.under_fire += r.under_fire;
        quiet.considered = r.considered;
        inf_physics::d3::step_character_movement(&mut w, &mut b, DT);
    }
    println!(
        "\n=== the NPC cover pass ===\n  quiet: {} responders considered, {} under fire, \
         {} shape casts",
        quiet.considered, quiet.under_fire, quiet.probes
    );
    assert_eq!(
        quiet.probes, 0,
        "an officer nobody is shooting at spent {} shape casts looking for cover",
        quiet.probes
    );
    assert!(
        quiet.considered > 0,
        "the pass never saw the officer at all"
    );

    // ── under fire: a shot from the far side of the wall.
    let (mut w, mut b) = build();
    let source = glam::DVec3::new(0.0, 1.4, 20.0);
    let mut reached: Option<u64> = None;
    let mut peeked = 0usize;
    let mut probes = 0u32;
    for i in 0..900u64 {
        b.sync_from_world(&w);
        let r = inf_physics::d3::step_npc_cover(&mut w, &mut b, &[source], 45.0, i, DT);
        probes += r.probes;
        if r.peeking > 0 {
            peeked += 1;
        }
        inf_physics::d3::step_character_movement(&mut w, &mut b, DT);
        let e = w.entity_of(OFFICER).unwrap();
        let mode = w.world().get::<CharacterMovement>(e).unwrap().mode;
        if mode == MovementMode::Cover && reached.is_none() {
            reached = Some(i);
        }
    }
    let e = w.entity_of(OFFICER).unwrap();
    let cm = w.world().get::<CharacterMovement>(e).unwrap().clone();
    let at = w
        .world()
        .get::<Transform>(e)
        .unwrap()
        .translation
        .to_dvec3();
    println!(
        "  under fire: reached cover at step {reached:?} ({:.2} s), class {:?}, at \
         ({:.2}, {:.2}), {peeked} steps leaned out, {probes} shape casts",
        reached.map(|s| s as f64 * DT).unwrap_or(f64::NAN),
        cm.runtime.cover.class,
        at.x,
        at.z,
    );
    let Some(step) = reached else {
        panic!(
            "the officer never reached cover in 15 s: mode {:?}, at ({:.2}, {:.2}), \
             {probes} shape casts spent",
            cm.mode, at.x, at.z
        );
    };
    assert!(
        (step as f64) * DT < 8.0,
        "the officer took {:.2} s to reach cover",
        step as f64 * DT
    );
    assert_eq!(cm.mode, MovementMode::Cover, "and it did not stay there");
    // It got BEHIND the wall, which is the world's answer rather than the
    // mode's: the wall is at z = 6 and the shooter beyond it.
    assert!(
        at.z > 4.5,
        "the officer is at z = {:.3} and the wall it should be behind is at 6.0",
        at.z
    );
    assert!(
        peeked > 0,
        "the officer never leaned out of cover in 15 s — the peek cadence is not running"
    );
    assert!(probes > 0, "…and it found its cover without probing for it");

    // ── **A CIVILIAN DOES NOT TAKE COVER** (clause 5's last sentence).
    //
    //    The same fixture, the same shot, one more body — and it is NOT on
    //    `RespondersRes`. The pass walks the responder set and nothing else, so
    //    a bystander is untouched by construction; this is the arm that says so
    //    rather than the comment. WPN1's `flee_from` is what a civilian does
    //    instead, and EMS2's exemption is what keeps the officer beside it from
    //    doing the same.
    const CIVILIAN: uuid::Uuid = uuid::Uuid::from_u128(0xC0_5004);
    let (mut w, mut b) = build();
    {
        let cm = CharacterMovement::default();
        let e = w.spawn_with_guid(CIVILIAN, "Bystander", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(1.0, cm.stand_half_height_m + 0.3, 1.0);
        w.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(0.3, cm.stand_half_height_m, 0.3),
                radius: 0.3,
                ..Default::default()
            },
            CharacterController3D::default(),
            cm,
            t,
        ));
        w.mark_dirty();
        w.propagate();
    }
    let mut civ_covered = 0usize;
    for i in 0..600u64 {
        b.sync_from_world(&w);
        inf_physics::d3::step_npc_cover(&mut w, &mut b, &[source], 45.0, i, DT);
        inf_physics::d3::step_character_movement(&mut w, &mut b, DT);
        let e = w.entity_of(CIVILIAN).unwrap();
        if w.world().get::<CharacterMovement>(e).unwrap().mode == MovementMode::Cover {
            civ_covered += 1;
        }
    }
    let e = w.entity_of(CIVILIAN).unwrap();
    let civ = w.world().get::<CharacterMovement>(e).unwrap().clone();
    println!(
        "  the civilian beside it: mode {:?} over 600 steps ({civ_covered} of them in cover),          {} cover shape casts",
        civ.mode, civ.runtime.cover.sweeps
    );
    assert_eq!(
        civ_covered, 0,
        "a bystander took cover — the pass is walking more than the responder set"
    );
    assert_eq!(civ.runtime.cover.sweeps, 0, "…and probed for it");
    // …while the officer in the same world still did.
    let e = w.entity_of(OFFICER).unwrap();
    assert_eq!(
        w.world().get::<CharacterMovement>(e).unwrap().mode,
        MovementMode::Cover,
        "the officer stopped taking cover once a civilian was in the world, which would          make the civilian's zero meaningless"
    );
}

/// **SWAT PREFERS HIGH COVER** — the EMS3 carried item becoming a behaviour.
///
/// The rule is `inf_ecs::cover::prefers_high`, and the arm is on the RULE rather
/// than on a firefight: at the top response rung a unit takes the class it can
/// shoot around an edge from, and at every other rung it takes what is nearest.
/// Which units arrive at all is EMS3's ladder and is measured there.
#[test]
fn swat_prefers_the_cover_it_can_shoot_around() {
    use inf_ecs::crime::Response;
    assert!(inf_ecs::cover::prefers_high(Response::Swat));
    for r in [Response::Cold, Response::Patrol, Response::MultiUnit] {
        assert!(
            !inf_ecs::cover::prefers_high(r),
            "{r:?} prefers high cover, which makes the preference meaningless"
        );
    }
    // …and the peek cadence is a cycle, not a latch: a unit that leaned out once
    // and stayed out is a unit standing in the open.
    let (out_at_zero, _) = inf_ecs::cover::peek_cycle(0.0);
    assert!(out_at_zero, "the cycle starts leaned out");
    let (out_late, _) = inf_ecs::cover::peek_cycle(inf_ecs::cover::NPC_PEEK_OUT_S + 0.1);
    assert!(!out_late, "the cycle never goes back in");
    let period = inf_ecs::cover::NPC_PEEK_OUT_S + inf_ecs::cover::NPC_PEEK_IN_S;
    let (again, _) = inf_ecs::cover::peek_cycle(period + 0.05);
    assert!(again, "the cycle does not repeat");
    // And a unit is behind its cover more of the time than out of it.
    assert!(
        inf_ecs::cover::NPC_PEEK_IN_S > inf_ecs::cover::NPC_PEEK_OUT_S,
        "an officer leans out for longer than it hides, which is not taking cover"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (6) DETERMINISM
// ─────────────────────────────────────────────────────────────────────────────

/// **The bytes a cover trace folds** — the mode, the snapped transform and the
/// peek, which is exactly what the brief calls the wave's sim state.
fn cover_trace(world: &inf_ecs::EcsWorld, hero: uuid::Uuid) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    let Some(e) = world.entity_of(hero) else {
        return out;
    };
    if let Some(t) = world.world().get::<Transform>(e) {
        for v in [
            t.translation.x,
            t.translation.y,
            t.translation.z,
            t.rotation.y,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    if let Some(cm) = world.world().get::<CharacterMovement>(e) {
        out.push(cm.mode as u8);
        let c = &cm.runtime.cover;
        out.push(u8::from(c.active));
        out.push(u8::from(c.crouched));
        out.push(u8::from(c.at_corner));
        out.push(c.class as u8);
        out.push(c.side as u8);
        for v in [
            c.peek, c.along_m, c.top_m, c.left_m, c.right_m, c.anchor.x, c.anchor.z, c.yaw_deg,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// **PIE == SHIPPING ON A COVER TRACE** (clause 7).
///
/// The editor's Simulate and the shipped player build the same three-body
/// fixture — a floor, a wall and a character — run the same scripted drive, and
/// their cover traces are compared step for step. The trace is the mode, the
/// snapped transform and the peek, which is the whole of what this wave adds to
/// the simulation.
///
/// **A fixture and not a committed level**, for `movement_parity`'s reason: no
/// committed level carries a `CharacterMovement`, and the phase-29 course this
/// file's camera sibling uses has nothing on it a cover press can take. A drive
/// that never entered cover would make two hosts agree about a character
/// standing in a field, so the arm counts the steps it spent in `Cover` and
/// **fails** if there are none.
///
/// What it proves that `two_identical_worlds_move_byte_for_byte` cannot: that
/// the editor FEEDS the cover verb the same way the player does. Dropping
/// `cover` from either host's action set fails this file and nothing else.
#[test]
fn pie_equals_shipping_on_a_cover_trace() {
    use inf_ecs::components::{
        BodyKind3D, CharacterController3D, Collider3D, ColliderShape3DKind, RigidBody3D,
    };
    use inf_ecs::math::Vec3d;
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::simulate::{SimInput, SimSession};
    use std::collections::BTreeMap;

    const HERO: uuid::Uuid = uuid::Uuid::from_u128(0xC0_9001);
    const GROUND: uuid::Uuid = uuid::Uuid::from_u128(0xC0_9002);
    const WALL: uuid::Uuid = uuid::Uuid::from_u128(0xC0_9003);
    const HZ: f64 = 60.0;
    const STEPS: u32 = 500;
    let gravity = glam::DVec2::new(0.0, -9.81);

    let block = |centre: glam::DVec3, half: glam::DVec3| {
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::from_dvec3(centre);
        (
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::from_dvec3(half),
                ..Default::default()
            },
            t,
        )
    };
    let hero_parts = || {
        let cm = CharacterMovement {
            player_controlled: true,
            ..Default::default()
        };
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, cm.stand_half_height_m + 0.3, 0.4);
        (
            RigidBody3D {
                kind: BodyKind3D::Kinematic,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: Vec3d::new(0.3, cm.stand_half_height_m, 0.3),
                radius: 0.3,
                ..Default::default()
            },
            CharacterController3D::default(),
            cm,
            t,
        )
    };
    // A wall two metres wide, so the slide reaches a corner inside the drive.
    let bodies = [
        (
            GROUND,
            block(
                glam::DVec3::new(0.0, -0.5, 0.0),
                glam::DVec3::new(40.0, 0.5, 40.0),
            ),
        ),
        (
            WALL,
            block(
                glam::DVec3::new(0.0, 1.5, 1.3),
                glam::DVec3::new(2.0, 1.5, 0.3),
            ),
        ),
    ];
    // Walk in, take cover, slide left into the corner, aim (which leans out),
    // release, and leave. A pure function of the step index.
    let script = |i: u32| -> (Vec<&'static str>, BTreeMap<String, f32>) {
        let ax = |p: &[(&str, f32)]| -> BTreeMap<String, f32> {
            p.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
        };
        match i {
            0..=59 => (vec![], ax(&[])),
            60..=62 => (vec!["cover"], ax(&[])),
            63..=240 => (vec![], ax(&[("move_x", -1.0)])),
            241..=360 => (vec!["aim"], ax(&[])),
            361..=420 => (vec![], ax(&[])),
            421..=423 => (vec!["cover"], ax(&[])),
            _ => (vec![], ax(&[("move_y", 1.0)])),
        }
    };

    let ship: Vec<Vec<u8>> = {
        let mut world = inf_ecs::EcsWorld::new();
        for (guid, parts) in bodies.clone() {
            let e = world.spawn_with_guid(guid, "Block", None);
            world.world_mut().entity_mut(e).insert(parts);
        }
        let e = world.spawn_with_guid(HERO, "Hero", None);
        world.world_mut().entity_mut(e).insert(hero_parts());
        world.mark_dirty();
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), gravity, HZ);
        (0..STEPS)
            .map(|i| {
                let (held, ax) = script(i);
                sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
                cover_trace(sim.world(), HERO)
            })
            .collect()
    };

    let pie: Vec<Vec<u8>> = {
        let mut doc = SceneDoc::new();
        for (guid, parts) in bodies.clone() {
            let e =
                doc.create_with_guid(guid, inf_editor_core::ipc::SpawnKind::Empty, "Block", None);
            doc.world_mut().world_mut().entity_mut(e).insert(parts);
        }
        let e = doc.create_with_guid(HERO, inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(hero_parts());
        doc.world_mut().mark_dirty();
        doc.world_mut().propagate();
        let mut session = SimSession::enter(&mut doc, Vec::new(), gravity, HZ);
        let out: Vec<Vec<u8>> = (0..STEPS)
            .map(|i| {
                let (held, ax) = script(i);
                session.step_once(&mut doc, SimInput::with_down(held).with_axes(ax));
                cover_trace(doc.world(), HERO)
            })
            .collect();
        session.exit(&mut doc);
        out
    };

    // **Non-vacuity FIRST.** `cover_trace` writes four `f64`s and then the mode,
    // so the mode byte is at offset 32 — pinned here against the writer rather
    // than assumed, which is `movement_parity::record_len`'s own lesson.
    const MODE_AT: usize = 4 * 8;
    let in_cover = ship
        .iter()
        .filter(|b| b.len() > MODE_AT && b[MODE_AT] == MovementMode::Cover as u8)
        .count();
    let peeked = ship
        .iter()
        .filter(|b| b.len() > MODE_AT + 5 && b[MODE_AT + 5] != CoverSide::Behind as u8)
        .count();
    let distinct: std::collections::BTreeSet<&Vec<u8>> = ship.iter().collect();
    println!(
        "\n=== PIE == shipping on a cover trace ===\n  {STEPS} steps: {in_cover} in Cover, \
         {peeked} leaning out, {} distinct traces on the shipping side",
        distinct.len()
    );
    assert!(
        in_cover > 200,
        "only {in_cover} of {STEPS} steps were in cover — the drive never took it, and two \
         hosts agreeing about a character standing in a field is not this arm's claim"
    );
    assert!(
        peeked > 30,
        "only {peeked} steps leaned out — the peek columns of this trace are all zero"
    );
    assert!(
        distinct.len() > STEPS as usize / 4,
        "the character barely moved ({} distinct traces)",
        distinct.len()
    );
    for (i, (a, b)) in ship.iter().zip(pie.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {i}: the editor's Simulate and the shipped player put the same character \
             in a different place, mode or peek"
        );
    }
    assert_eq!(ship.len(), pie.len());
}
