//! **WAVE PAR1a — BUILDING ILLUMINATION** — the gate.
//!
//! Every room of every archetype on every floor of the island hangs a real
//! fixture (a fitting you can see + a real `Light` from one vocabulary per
//! `RoomType`), exteriors carry porch / facade / sign / forecourt lights, and
//! the occupancy half of the night schedule decides which are lit. The arms:
//!
//! * the zero-unlit-rooms census over the WHOLE shipped island, by archetype
//!   and floor, read off the one volume door both hosts pass
//!   (`inf_pcg::compose_volume`) — and the same census WORLD-side, off the
//!   `ScatteredLight`s a loaded island actually holds;
//! * the 24 h schedule sweep: lit-room count by hour against the society's
//!   own day, PIE == shipping on the projected light list at every hour;
//! * the party wall: a lit room's fixture lights nothing in the room beside it
//!   (its box), with the unboxed control that does;
//! * (off CI) the luminance arms and the island's 21:00 light census.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use inf_pcg::building::fixtures::{self, CensusRow, FixtureRow};
use inf_pcg::building::FixtureTag;
use inf_pcg::ArchetypeId;

/// The shipped island's recipe.
fn shipped_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island/island.toml")
}

/// One evaluated settlement block, through the door the hosts use.
struct BlockEval {
    archetype: ArchetypeId,
    plans: Vec<inf_pcg::building::BuildingPlan>,
    lights: Vec<inf_pcg::building::PcgLight>,
}

/// Evaluate one block exactly as `inf_player::level` does: its zone graph's
/// building pass over its own extent, salted by its volume's guid, joined by
/// `compose_volume`. The ground is flat (a datum moves no room).
fn eval_block(block: &inf_editor_core::settlement::Block, guid: uuid::Uuid) -> BlockEval {
    let graph = inf_editor_core::settlement::zone_graph(block.archetype);
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert!(lowered.ok, "{:?}: {:?}", block.archetype, lowered.issues);
    let cx = inf_pcg::GrammarContext {
        entity: Some(guid),
        center: glam::DVec3::new(block.centre.x, 0.0, block.centre.y),
        extent: block.half,
        seed_offset: u64::from(block.seed),
    };
    let height = inf_pcg::FnHeight::new(|_, _| Some(0.0));
    let out = inf_pcg::evaluate_buildings(&lowered.buildings, &inf_pcg::NoSplines, &height, &cx);
    let plans = inf_pcg::building::plans_of(&lowered.buildings, &inf_pcg::NoSplines, &height, &cx);
    let vol = inf_pcg::compose_volume(Vec::new(), out);
    BlockEval {
        archetype: block.archetype,
        plans,
        lights: vol.lights,
    }
}

fn lit_rooms<'a>(tags: impl Iterator<Item = (u32, u32)> + 'a) -> BTreeSet<(u32, u32)> {
    tags.filter(|(_, r)| *r != FixtureTag::EXTERIOR).collect()
}

/// **THE ZERO-UNLIT-ROOMS CENSUS, OVER THE WHOLE SHIPPED ISLAND** (clause 2).
///
/// Reads: every block of every settlement of `samples/island`, evaluated
/// through `compose_volume`; for every room of every plan, whether a fixture
/// names it. The base tree (venue rigs only) fails it on every archetype but
/// the three venues' rig rooms.
#[test]
fn the_island_census_reads_zero_unlit_rooms_by_archetype_and_floor() {
    let recipe =
        inf_island::IslandRecipe::load(&shipped_recipe()).expect("the island recipe loads");
    let design = inf_island::read_design(&recipe).expect("the island design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let mut table: BTreeMap<(ArchetypeId, u32), CensusRow> = BTreeMap::new();
    let mut by_row: BTreeMap<FixtureRow, usize> = BTreeMap::new();
    let mut exterior_by_arch: BTreeMap<ArchetypeId, (usize, usize)> = BTreeMap::new();
    let (mut blocks, mut buildings, mut lights) = (0usize, 0usize, 0usize);
    for s in &plans {
        for b in &s.blocks {
            let guid = inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row);
            let e = eval_block(b, guid);
            blocks += 1;
            buildings += e.plans.len();
            lights += e.lights.len();
            for l in &e.lights {
                *by_row.entry(l.tag.row).or_default() += 1;
            }
            let lit = lit_rooms(e.lights.iter().map(|l| (l.tag.building, l.tag.room)));
            fixtures::census(&e.plans, &lit, &mut table);
            // A porch light on every building's street door.
            let porches: BTreeSet<u32> = e
                .lights
                .iter()
                .filter(|l| l.tag.row == FixtureRow::Porch)
                .map(|l| l.tag.building)
                .collect();
            let x = exterior_by_arch.entry(e.archetype).or_default();
            x.0 += e.plans.len();
            x.1 += porches.len();
        }
    }
    println!("PAR1a census: {blocks} blocks, {buildings} buildings, {lights} fixtures");
    println!(
        "{:<14} {:>5} {:>7} {:>6}",
        "archetype", "floor", "rooms", "unlit"
    );
    let mut unlit = 0u32;
    let mut rooms = 0u32;
    for ((a, f), r) in &table {
        println!("{:<14} {:>5} {:>7} {:>6}", a.name(), f, r.rooms, r.unlit);
        unlit += r.unlit;
        rooms += r.rooms;
    }
    for (row, n) in &by_row {
        println!("  row {:<18} {n}", row.name());
    }
    for (a, (b, p)) in &exterior_by_arch {
        println!("  {:<14} {b} buildings, {p} with a porch light", a.name());
        assert_eq!(b, p, "{a:?}: {} buildings have no porch light", b - p);
    }
    println!("PAR1a census: {rooms} rooms, {unlit} unlit");
    assert!(rooms > 1_000, "the census saw only {rooms} rooms");
    assert!(
        table.keys().filter(|(_, f)| *f >= 3).count() > 3,
        "the census saw no fourth floor anywhere"
    );
    assert_eq!(
        unlit, 0,
        "{unlit} of {rooms} rooms on the island hang no fixture"
    );
}

// ── the rendered arms ────────────────────────────────────────────────────────

use glam::{DVec2, DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_pcg::building::{BuildingPlan, PcgLight};
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightBound, LightClip, LightKind, MeshInstance,
    RenderLight, RenderScene, RenderSettings, RenderView, HEADLESS_FORMAT,
};

const W: u32 = 640;
const H: u32 = 360;
/// Frames before a readback (the warm-up `par0b_night_gate` uses).
const WARM: u32 = 6;
/// The fixed exposure every rendered arm reads at (see `render`).
const NIGHT_EXPOSURE: f32 = 8.0;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP par1a_rooms_gate: no GPU adapter ({e})");
            None
        }
    }
}

fn look(eye: DVec3, target: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn render(gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    render_with(gpu, scene, view, true)
}

/// [`render`], with the GI probe march's local-light bounce on or off — the
/// attribution control for a leak the box cannot see (the bounce is GI's).
fn render_with(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    gi_local: bool,
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    let mut s = RenderSettings::default();
    // The shipped ambient (the GI door: the night sky's irradiance + the probe
    // field, so an unlit room is black, not the GI-off hemispheric constant);
    // no sun shadow and no local shadows — so what the arms measure is the BOX.
    s.gi.enabled = true;
    s.gi.local_lights = gi_local;
    s.shadows.enabled = false;
    s.vsm.enabled = false;
    s.lights.local_shadow_budget = 0;
    // A fixed exposure of 8 — inside the eight stops the shipped night eye
    // opens (PAR0b measured x29 indoors at 21:00) — so a leak a viewer would
    // see at night is above one code here, and nothing auto-exposes it away.
    s.exposure = NIGHT_EXPOSURE;
    r.set_settings(s);
    for _ in 0..WARM {
        r.render(gpu, scene, view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

/// With `PAR1A_DUMP` naming a directory, write a frame there as raw RGBA.
fn dump(name: &str, img: &[u8]) {
    if let Some(dir) = std::env::var_os("PAR1A_DUMP") {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(format!("{name}.rgba")), img);
    }
}

fn luma(img: &[u8], x: u32, y: u32) -> f64 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f64::from(img[i]) + 0.7152 * f64::from(img[i + 1]) + 0.0722 * f64::from(img[i + 2])
}

fn p50(img: &[u8], x0: u32, y0: u32, w: u32, h: u32) -> f64 {
    let mut v: Vec<f64> = Vec::with_capacity((w * h) as usize);
    for y in y0..(y0 + h).min(H) {
        for x in x0..(x0 + w).min(W) {
            v.push(luma(img, x, y));
        }
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn project(view: &RenderView, p: DVec3) -> (u32, u32) {
    let c = view.view_proj() * (p - view.origin.origin()).as_vec3().extend(1.0);
    let n = c.truncate() / c.w;
    (
        ((n.x * 0.5 + 0.5) * W as f32).clamp(0.0, W as f32 - 1.0) as u32,
        ((0.5 - n.y * 0.5) * H as f32).clamp(0.0, H as f32 - 1.0) as u32,
    )
}

/// Median luma of a `px` x `px` patch around a world point.
fn patch_at(img: &[u8], view: &RenderView, p: DVec3, px: u32) -> f64 {
    let (x, y) = project(view, p);
    p50(
        img,
        x.saturating_sub(px / 2),
        y.saturating_sub(px / 2),
        px,
        px,
    )
}

/// A real grammar building of `arch`, `floors` storeys, on an axis-aligned
/// lot at the origin, ground at `y = 0`.
fn building(
    arch: ArchetypeId,
    floors: u32,
    seed: u64,
    furnish: bool,
) -> inf_pcg::building::BuildingOutput {
    let p = inf_pcg::building::BuildingParams {
        floors,
        ..inf_pcg::building::BuildingParams::new(
            arch,
            inf_pcg::building::Rect2::new(DVec2::ZERO, DVec2::new(24.0, 16.0)),
            0.0,
            seed,
        )
    };
    inf_pcg::building::build(&p, seed, furnish)
}

/// The building's solids as boxes (a transmitting solid — a shopfront — as
/// glass), its panes as glass, its fittings, the ground, a near-zero night
/// fill, and the fixtures `on` admits as real lights, each with its box when
/// `boxed`.
fn building_scene(
    out: &inf_pcg::building::BuildingOutput,
    on: &dyn Fn(&PcgLight) -> bool,
    boxed: bool,
) -> RenderScene {
    // A moonless night: the sun well below the horizon under the P17
    // atmosphere, so the sky's irradiance is the dark it is at 01:00.
    let night = Vec3::new(0.3, -0.45, 0.84).normalize();
    let mut scene = RenderScene {
        grid_enabled: false,
        sun: inf_render::SunParams {
            direction: night,
            intensity: 3.0,
            ..inf_render::SunParams::default()
        },
        atmosphere: inf_render::AtmosphereParams {
            enabled: true,
            aerial_perspective: 0.0,
            ..inf_render::AtmosphereParams::default()
        },
        ..Default::default()
    };
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        intensity: 1.0e-4,
        direction: Vec3::new(0.2, 0.9, 0.3).normalize(),
        cast_shadows: false,
        ..RenderLight::default()
    });
    let mut ground = MeshInstance::lit(
        DVec3::new(12.0, -0.5, 8.0),
        Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.4, 0.4, 0.4, 1.0],
        1,
    );
    ground.roughness = 1.0;
    scene.instances.push(ground);
    let mut glass: Vec<inf_render::ScatterInstance> = Vec::new();
    for (i, inst) in out.instances.iter().enumerate() {
        let Some(e) = inst.extent else { continue };
        let scale = Vec3::new(e[0] * 2.0, e[1] * 2.0, e[2] * 2.0);
        let rot = inst.rotation.as_quat();
        if inst.surface.transmission > 0.0 {
            glass.push(inf_render::ScatterInstance {
                position: inst.pos,
                rotation: rot,
                scale,
                color: [0.86, 0.93, 0.92, 1.0],
            });
            continue;
        }
        // Solids (the aligned prefix) and fittings draw; other decor (signs)
        // is irrelevant to a room's light and skipped.
        let solid = i < out.colliders.len();
        let fitting = inst.kind_index == inf_pcg::building::assemble::FIXTURE_KIND_INDEX;
        if !(solid || fitting) {
            continue;
        }
        let colour = inst.surface.tint.unwrap_or([0.7, 0.7, 0.7, 1.0]);
        let mut m = MeshInstance::lit(inst.pos, rot, scale, colour, 2 + i as u32);
        m.roughness = 1.0;
        scene.instances.push(m);
    }
    if !glass.is_empty() {
        let anchor = glass[0].position;
        let mut b = inf_render::ScatterBatch::lit(
            std::sync::Arc::new(inf_render::ScatterData::build(
                inf_render::PrimMesh::Cube,
                anchor,
                glass,
            )),
            anchor,
            0.05,
            9_000,
        );
        b.transmission = 0.85;
        b.casts_shadows = false;
        scene.scatter.push(b);
    }
    for l in out.lights.iter().filter(|l| on(l)) {
        if boxed {
            if let Some(c) = l.clip {
                scene.light_bounds.push(LightBound {
                    light: scene.lights.len() as u32,
                    clip: Some(LightClip {
                        center: c.center,
                        half: c.half.as_vec3(),
                        u: [c.u.x as f32, c.u.y as f32],
                    }),
                    draw_m: 0.0,
                });
            }
        }
        let point = l.outer_deg >= 180.0;
        scene.lights.push(RenderLight {
            kind: if point {
                LightKind::Point
            } else {
                LightKind::Spot
            },
            color: l.sweep.0,
            intensity: l.intensity,
            direction: (-l.dir).as_vec3(),
            position: l.at,
            range: l.range_m,
            inner_cos: l.inner_deg.to_radians().cos(),
            outer_cos: l.outer_deg.to_radians().cos(),
            cast_shadows: false,
        });
    }
    scene.mark_dirty();
    scene
}

/// A room on an upper storey with an exterior window, a neighbour across a
/// door-less party wall, and a room below it: `(room, window, neighbour,
/// shared wall, below)`.
fn pick_room(plan: &BuildingPlan) -> (usize, usize, usize, usize, usize) {
    use inf_pcg::building::{OpeningKind, RoomType};
    let ordinary = |k: RoomType| !matches!(k, RoomType::Stair | RoomType::Corridor);
    let doors: BTreeSet<usize> = plan
        .openings
        .iter()
        .filter(|o| o.kind == OpeningKind::Door)
        .map(|o| o.wall)
        .collect();
    for (ri, r) in plan.rooms.iter().enumerate() {
        if r.floor < 3 || !ordinary(r.kind) {
            continue;
        }
        let Some(win) = plan.openings.iter().position(|o| {
            o.kind == OpeningKind::Window
                && plan.walls[o.wall].inside == ri
                && plan.walls[o.wall].is_exterior()
        }) else {
            continue;
        };
        let neighbour = plan.walls.iter().enumerate().find_map(|(wi, w)| {
            if w.floor != r.floor || doors.contains(&wi) || w.length() < 2.0 {
                return None;
            }
            let other = match (w.inside, w.outside) {
                (i, Some(o)) if i == ri => o,
                (i, Some(o)) if o == ri => i,
                _ => return None,
            };
            ordinary(plan.rooms[other].kind).then_some((other, wi))
        });
        let c = r.rect.center();
        let below = plan
            .rooms
            .iter()
            .position(|b| b.floor + 1 == r.floor && b.rect.contains(c));
        if let (Some((nj, wj)), Some(bi)) = (neighbour, below) {
            return (ri, win, nj, wj, bi);
        }
    }
    panic!("no upper-floor room with a window, a party wall and a room below");
}

/// The building's slab thickness (its archetype's).
fn archetype_slab(plan: &BuildingPlan) -> f64 {
    inf_pcg::building::archetype(plan.archetype).slab_thickness
}

/// **A LIT ROOM READS ITS LAMP; THE FLAT NEXT DOOR AND THE FLAT BELOW DO
/// NOT** (clauses 1 and 6; the audit's priority (f)).
///
/// Reads: a real fourth-floor apartment room, lit by its OWN vocabulary
/// fixture(s) and nothing else, at night: (a) its floor under the lamp vs the
/// same room with the fixture removed; (b) the neighbour's floor 0.7 m past
/// the shared party wall, boxed vs the UNBOXED control; (c) the ceiling of the
/// room below, boxed vs unboxed. The base tree has no such fixture (a) and no
/// box (b, c — the control is the base renderer's behaviour).
#[test]
fn a_lit_room_reads_its_lamp_and_the_flats_beside_and_below_do_not() {
    let Some(gpu) = gpu() else { return };
    let out = building(ArchetypeId::Apartment, 5, 41, false);
    let plan = &out.plan;
    let (ri, _win, nj, wj, bi) = pick_room(plan);
    let room = plan.rooms[ri];
    let mine = |l: &PcgLight| l.tag.room == ri as u32;
    let lamps = out.lights.iter().filter(|l| mine(l)).count();
    assert!(lamps > 0, "room {ri} hangs no fixture");
    let y = plan.floor_y(room.floor);
    let c = room.rect.center();
    // (a) in the room: the floor under its lamp(s).
    let inside = look(
        DVec3::new(c.x + 0.8, y + 1.6, c.y + 0.8),
        DVec3::new(c.x, y, c.y),
    );
    let lit = render(&gpu, &building_scene(&out, &mine, true), &inside);
    let dark = render(&gpu, &building_scene(&out, &|_| false, true), &inside);
    let floor_lit = patch_at(&lit, &inside, DVec3::new(c.x, y, c.y), 24);
    let floor_dark = patch_at(&dark, &inside, DVec3::new(c.x, y, c.y), 24);
    dump("room_lit", &lit);
    dump("room_dark", &dark);
    // (b) next door: 0.7 m past the party wall, across from the room centre.
    let w = plan.walls[wj];
    let along = w.direction();
    let t = (c - w.a).dot(along).clamp(0.5, w.length() - 0.5);
    let on_wall = w.a + along * t;
    let into = (plan.rooms[nj].rect.center() - on_wall).normalize_or_zero();
    // The neighbour's CEILING 0.4 m past the wall: it faces down at a lamp
    // hung just under the same ceiling, so it is the surface next door an
    // unboxed lamp reaches first (the party wall's far face turns away).
    let probe = on_wall + into * 0.4;
    let ceil = y + inf_pcg::building::archetype(plan.archetype).floor_height - archetype_slab(plan);
    let probe3 = DVec3::new(probe.x, ceil, probe.y);
    let nc = plan.rooms[nj].rect.center();
    let next_view = look(DVec3::new(nc.x + into.x, y + 1.2, nc.y + into.y), probe3);
    let next_boxed = render(&gpu, &building_scene(&out, &mine, true), &next_view);
    let next_open = render(&gpu, &building_scene(&out, &mine, false), &next_view);
    let next_dark = render(&gpu, &building_scene(&out, &|_| false, true), &next_view);
    dump("next_boxed", &next_boxed);
    dump("next_unboxed", &next_open);
    let nb = patch_at(&next_boxed, &next_view, probe3, 16);
    let no = patch_at(&next_open, &next_view, probe3, 16);
    let nd = patch_at(&next_dark, &next_view, probe3, 16);
    // (c) below: the inner face of the room-below's wall nearest the lamp,
    // 0.4 m under its ceiling — a face that turns TOWARD a lamp above the
    // slab (a ceiling faces away from it and could never show the leak).
    let yb = plan.floor_y(plan.rooms[bi].floor);
    let br = plan.rooms[bi].rect;
    let half_t = inf_pcg::building::archetype(plan.archetype).wall_thickness * 0.5;
    let x_face = if (c.x - br.min.x) <= (br.max.x - c.x) {
        br.min.x + half_t + 0.01
    } else {
        br.max.x - half_t - 0.01
    };
    let ceil_below = DVec3::new(
        x_face,
        y - archetype_slab(plan) - 0.4,
        c.y.clamp(br.min.y + 0.5, br.max.y - 0.5),
    );
    let bc = br.center();
    let below_view = look(DVec3::new(bc.x, yb + 1.5, bc.y), ceil_below);
    let below_boxed = render(&gpu, &building_scene(&out, &mine, true), &below_view);
    let below_open = render(&gpu, &building_scene(&out, &mine, false), &below_view);
    let below_dark = render(&gpu, &building_scene(&out, &|_| false, true), &below_view);
    let below_nobounce = render_with(&gpu, &building_scene(&out, &mine, true), &below_view, false);
    dump("below_boxed", &below_boxed);
    dump("below_unboxed", &below_open);
    let bb = patch_at(&below_boxed, &below_view, ceil_below, 16);
    let bo = patch_at(&below_open, &below_view, ceil_below, 16);
    let bd = patch_at(&below_dark, &below_view, ceil_below, 16);
    let bn = patch_at(&below_nobounce, &below_view, ceil_below, 16);
    println!(
        "PAR1a ROOM: {lamps} fixture(s) in a {:?} on floor {}; floor under the lamp {floor_lit:.1} vs no fixture {floor_dark:.1}; \
         next door ({:?}) its ceiling 0.4 m past the wall: boxed {nb:.1}, UNBOXED {no:.1}, all dark {nd:.1}; \
         the wall of the room below: boxed {bb:.1}, UNBOXED {bo:.1}, all dark {bd:.1}, boxed with GI's local bounce off {bn:.1}",
        room.kind, room.floor, plan.rooms[nj].kind
    );
    assert!(
        floor_lit >= 40.0,
        "the room's lamp lights its floor to {floor_lit:.1}"
    );
    assert!(floor_dark <= 3.0, "the unlit room reads {floor_dark:.1}");
    assert!(
        nb <= nd + 1.0,
        "the lamp lights the flat next door: {nb:.1} vs dark {nd:.1}"
    );
    assert!(
        no >= nb + 4.0,
        "the unboxed control does not leak ({no:.1} vs {nb:.1}) — the arm reads nothing"
    );
    // Below: the box removes the DIRECT leak (the unboxed control's extra);
    // what is left boxed is the GI probe field carrying the room's bounce
    // through a slab thinner than a voxel — measured by the bounce-off
    // control reading dark, and carried to the GI cascade (PAR0b's item).
    assert!(
        bo >= bb + 4.0,
        "the room below: boxed {bb:.1}, unboxed {bo:.1}"
    );
    assert!(
        bn <= bd + 1.0,
        "boxed with GI's bounce off the room below reads {bn:.1} vs dark {bd:.1}"
    );
}

/// **AN UPPER-FLOOR ROOM IS LIT AND SEEN FROM THE STREET** (clauses 2 and 6).
///
/// Reads: from the pavement 14 m out, the fourth-floor room's window (its pane
/// a transmitting leaf, PAR0's glass) with the room's own fixture on, vs the
/// parapet wall beside it and vs the same window with the fixture off. The
/// base tree hangs no fixture in an apartment: its window reads the wall.
#[test]
fn an_upper_floor_window_is_lit_from_the_street_by_its_rooms_own_lamp() {
    let Some(gpu) = gpu() else { return };
    let out = building(ArchetypeId::Apartment, 5, 41, false);
    let plan = &out.plan;
    let (ri, win, ..) = pick_room(plan);
    let o = plan.openings[win];
    let w = plan.walls[o.wall];
    let mid = w.point_at((o.start + o.end) * 0.5);
    let rc = plan.rooms[ri].rect.center();
    let n = {
        let d = w.direction();
        let n = DVec2::new(-d.y, d.x);
        if (mid - rc).dot(n) >= 0.0 {
            n
        } else {
            -n
        }
    };
    let y = plan.floor_y(plan.rooms[ri].floor);
    let window = DVec3::new(mid.x, y + (o.sill + o.head) * 0.5, mid.y);
    let parapet = DVec3::new(mid.x, y + o.sill * 0.45, mid.y) + DVec3::new(n.x, 0.0, n.y) * 0.2;
    let eye = DVec3::new(mid.x + n.x * 14.0, 1.7, mid.y + n.y * 14.0);
    let view = look(eye, window);
    let mine = |l: &PcgLight| l.tag.room == ri as u32;
    let lit = render(&gpu, &building_scene(&out, &mine, true), &view);
    let dark = render(&gpu, &building_scene(&out, &|_| false, true), &view);
    dump("street_window_lit", &lit);
    dump("street_window_dark", &dark);
    let (wl, wd) = (
        patch_at(&lit, &view, window, 6),
        patch_at(&dark, &view, window, 6),
    );
    let pl = patch_at(&lit, &view, parapet, 4);
    println!(
        "PAR1a STREET: floor {} window from 14 m: lit {wl:.1}, the same window dark {wd:.1}, the parapet beside it {pl:.1}",
        plan.rooms[ri].floor
    );
    assert!(wl >= wd + 15.0, "the lit window {wl:.1} vs dark {wd:.1}");
    assert!(
        wl >= pl + 15.0,
        "the lit window {wl:.1} vs the wall beside it {pl:.1}"
    );
}
