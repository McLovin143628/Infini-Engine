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
                        interior: l.tag.room != FixtureTag::EXTERIOR,
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

// ── the loaded island: the census world-side, the 24 h sweep, PIE == shipping ─

use inf_player::runtime_sim::RuntimeSim;

fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

/// Build + cook the CI-scale island; the shipping sim (the pack) and the PIE
/// sim (the loose level the author saved), both with their streamers, built
/// the way `island_gate`'s two hosts are.
fn fixture_hosts(tmp: &std::path::Path) -> (RuntimeSim, RuntimeSim) {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    inf_island::write_content(&build, &content).expect("the island's content writes");
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    let source = inf_player::level::PackLevelSource::open(&out).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut ship = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut ship, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut ship,
        &inf_player::TerrainContent::Pack(source.clone()),
    );

    let slug = inf_island::slug(&recipe.name);
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(&content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(&content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(&content),
    )
    .with_bindings(inf_player::level::load_actor_classes_by_guid_from_dir(
        &content,
    ))
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        &content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(
        &content,
    ))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(&content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut pie = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut pie, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut pie, &inf_player::TerrainContent::Dir(terrains));
    (ship, pie)
}

/// Put the hero at `at`, freeze the clock at local `hour`, and step `steps`.
fn stand(sim: &mut RuntimeSim, at: DVec3, hour: f64, steps: usize) {
    let hero = sim
        .world()
        .world()
        .iter_entities()
        .find(|e| {
            e.get::<inf_ecs::components::CharacterMovement>()
                .is_some_and(|m| m.player_controlled)
        })
        .map(|e| e.id());
    if let Some(e) = hero {
        if let Some(mut t) = sim
            .world_mut()
            .world_mut()
            .get_mut::<inf_ecs::components::Transform>(e)
        {
            t.translation = inf_ecs::math::Vec3d::new(at.x, at.y + 2.0, at.z);
        }
    }
    {
        let w = sim.world_mut().world_mut();
        let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
        for mut tod in q.iter_mut(w) {
            tod.seconds = (hour * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
            tod.rate = 0.0;
        }
    }
    sim.world_mut().mark_dirty();
    for _ in 0..steps {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
}

fn project_world(sim: &RuntimeSim) -> RenderScene {
    let mut scene = RenderScene::default();
    let voxels = inf_voxel::VoxelVolumes::default();
    let mut meshes = inf_render::ScatterMeshes::new();
    inf_player::scatter_mesh::add_building_modules(&mut meshes);
    inf_player::render::project_scene_full(
        &mut scene,
        sim,
        1.0,
        &inf_player::vmesh::VmeshRegistry::default(),
        &inf_player::skinned::SkinnedRegistry::new(),
        &voxels,
        &mut inf_render::DebrisCache::default(),
        None,
        &meshes,
        &std::collections::HashMap::new(),
    );
    scene
}

/// The projected light list and its bounds as bytes — what the two hosts
/// must agree on.
fn light_bytes(scene: &RenderScene) -> Vec<u8> {
    let mut b = Vec::new();
    for l in &scene.lights {
        b.push(l.kind as u8);
        for f in l
            .color
            .iter()
            .chain([l.intensity, l.range, l.inner_cos, l.outer_cos].iter())
        {
            b.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        for f in [l.position.x, l.position.y, l.position.z] {
            b.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        b.push(u8::from(l.cast_shadows));
    }
    for lb in &scene.light_bounds {
        b.extend_from_slice(&lb.light.to_le_bytes());
        b.extend_from_slice(&lb.draw_m.to_bits().to_le_bytes());
        if let Some(c) = lb.clip {
            for f in [c.center.x, c.center.y, c.center.z] {
                b.extend_from_slice(&f.to_bits().to_le_bytes());
            }
        }
    }
    b
}

/// Every resident volume's fixtures: `(volume guid, centre, the lights)`.
fn resident_fixtures(sim: &RuntimeSim) -> Vec<(DVec3, Vec<inf_ecs::components::ScatteredLight>)> {
    let w = sim.world().world();
    let mut out = Vec::new();
    for e in w.iter_entities() {
        let Some(v) = e.get::<inf_ecs::components::PcgVolume>() else {
            continue;
        };
        if v.lights.is_empty() {
            continue;
        }
        let c = e
            .get::<inf_ecs::components::GlobalTransform>()
            .map(|g| g.translation())
            .unwrap_or(DVec3::ZERO);
        out.push((c, v.lights.clone()));
    }
    out.sort_by(|a, b| a.0.x.total_cmp(&b.0.x).then(a.0.z.total_cmp(&b.0.z)));
    out
}

/// Whether `p` lies inside the oriented shell box `s`, padded by `pad` metres.
fn in_shell(s: &inf_ecs::components::ScatteredSolid, p: DVec3, pad: f64) -> bool {
    let l = s.rotation.inverse() * (p - s.center);
    l.x.abs() <= s.half_extents.x + pad
        && l.y.abs() <= s.half_extents.y + pad
        && l.z.abs() <= s.half_extents.z + pad
}

/// **Every resident fixture is where its room is** (audit PAR1a, b'): over the
/// loaded world's volumes, a room light lies inside its own clip box, the box's
/// centre and the light lie inside ONE building shell of the volume, and the
/// light is above the terrain under it; an exterior light lies within a metre
/// of a shell. A fixture UNDER the ground is reported apart (`buried`): it is
/// in its room, and the room is a ground-floor room of a building whose floor
/// 0 sits below the grade of its slope (the building datum's ruling, not the
/// vocabulary's) — terrain skips a room's lights (`LIGHT_SKIP_ROOMS`), so it
/// lights no grass. Returns `(checked, misplaced, buried)`.
fn fixtures_in_their_buildings(sim: &mut RuntimeSim) -> (usize, Vec<String>, Vec<(u32, String)>) {
    let mut vols: Vec<(
        Vec<inf_ecs::components::ScatteredLight>,
        Vec<inf_ecs::StructureGroup>,
    )> = Vec::new();
    for e in sim.world().world().iter_entities() {
        if let Some(v) = e.get::<inf_ecs::components::PcgVolume>() {
            if !v.lights.is_empty() {
                vols.push((v.lights.clone(), v.structure_groups.clone()));
            }
        }
    }
    let (mut checked, mut bad, mut buried) = (0usize, Vec::new(), Vec::new());
    for (lights, groups) in &vols {
        for l in lights {
            if fixtures::FixtureRow::from_code(l.row).is_some_and(|r| r.is_rig()) {
                continue;
            }
            checked += 1;
            let ground = sim.terrain_height_at(l.at.x, l.at.z);
            let why = if l.room == u32::MAX {
                if !groups.iter().any(|g| in_shell(&g.shell, l.at, 1.0)) {
                    Some("an exterior light more than 1 m from every shell")
                } else {
                    None
                }
            } else if let Some(c) = l.clip {
                let clip = LightClip {
                    center: c.center,
                    half: c.half.as_vec3(),
                    u: [c.u.x as f32, c.u.y as f32],
                    interior: true,
                };
                if !inf_render::lights::clip_holds(&clip, l.at) {
                    Some("a room light outside its own box")
                } else if !groups
                    .iter()
                    .any(|g| in_shell(&g.shell, c.center, 0.25) && in_shell(&g.shell, l.at, 0.25))
                {
                    Some("a room's box outside every building shell")
                } else {
                    None
                }
            } else if !groups.iter().any(|g| in_shell(&g.shell, l.at, 0.25)) {
                Some("a task lamp outside every building shell")
            } else {
                None
            };
            let line = |w: &str| {
                format!(
                    "{w}: row {} building {} floor {} room {} at ({:.1}, {:.2}, {:.1}), ground {ground:.2}",
                    l.row, l.building, l.floor, l.room, l.at.x, l.at.y, l.at.z
                )
            };
            if let Some(w) = why {
                bad.push(line(w));
            } else if l.at.y < ground {
                buried.push((l.floor, line("under the ground")));
            }
        }
    }
    (checked, bad, buried)
}

/// The occupancy class a census table groups a fixture under.
fn class_of(l: &inf_ecs::components::ScatteredLight) -> &'static str {
    use inf_ecs::components::FixtureOccupancy as O;
    if l.room == u32::MAX {
        return "exterior";
    }
    match (l.occupancy, l.schedule) {
        (O::Crew, inf_ecs::components::LightSchedule::Always) => "never closes",
        (O::Crew, _) => "venue",
        (O::Work, _) => "work",
        (O::Shop, _) => "shop",
        (O::Home(inf_ecs::components::HomeRoom::Living), _) => "home living",
        (O::Home(inf_ecs::components::HomeRoom::Bedroom), _) => "home bedroom",
        (O::Home(_), _) => "home other",
    }
}

/// **THE CENSUS WORLD-SIDE, THE 24 H SWEEP AND PIE == SHIPPING** (clauses 2
/// and 4).
///
/// Reads: the CI island loaded by BOTH hosts (the cooked pack and the loose
/// level), the hero in the first settlement's core. (1) Every resident
/// volume's `ScatteredLight`s against the plans its block builds — zero rooms
/// without a fixture, world-side. (2) Every hour of a day: the rooms lit
/// (`fixture_occupancy` × `fixture_level`, the projector's own door) by
/// occupancy class, against the society's day: workplaces full at noon and
/// dark but for a minority at 02:00, homes lit in the evening and dark at
/// noon (the Dusk half) and in the small hours, the never-closing rooms the
/// same at every hour, porches dusk to dawn. (3) The projected light list
/// byte-identical between the hosts at every hour. (1b, audit PAR1a b') Every
/// resident fixture inside its own box and one building shell, with the REAL
/// terrain: 0 misplaced of 2 281, 81 under the grade (stated, ratcheted).
/// Mutations: occupancy frozen at 1 reds (2); a room box offset 30 m reds (1b)
/// (2 245 misplaced). The base tree has no room fixtures: (1) and (2) fail.
#[test]
fn the_24h_sweep_lights_rooms_by_who_is_in_them_and_pie_equals_shipping() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let (mut ship, mut pie) = fixture_hosts(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let core = plans[0].centre;
    let at = DVec3::new(core.x, 0.0, core.y);
    for sim in [&mut ship, &mut pie] {
        stand(sim, at, 21.0, 600);
    }

    // (1) the census, world-side.
    let resident = resident_fixtures(&ship);
    assert!(!resident.is_empty(), "no volume with fixtures streamed in");
    let mut table: BTreeMap<(ArchetypeId, u32), CensusRow> = BTreeMap::new();
    let mut matched = 0usize;
    for (c, lights) in &resident {
        let Some(block) = plans
            .iter()
            .flat_map(|s| s.blocks.iter())
            .find(|b| (b.centre.x - c.x).abs() < 1e-6 && (b.centre.y - c.z).abs() < 1e-6)
        else {
            continue;
        };
        matched += 1;
        let guid =
            inf_editor_core::settlement::block_guid(&recipe.name, block.site, block.col, block.row);
        let e = eval_block(block, guid);
        let lit = lit_rooms(lights.iter().map(|l| (l.building, l.room)));
        fixtures::census(&e.plans, &lit, &mut table);
    }
    let (rooms, unlit) = table
        .values()
        .fold((0, 0), |a, r| (a.0 + r.rooms, a.1 + r.unlit));
    println!(
        "PAR1a WORLD CENSUS: {matched} resident block volume(s), {rooms} rooms, {unlit} unlit"
    );
    assert!(
        matched > 0 && rooms > 100,
        "{matched} volumes, {rooms} rooms"
    );
    assert_eq!(
        unlit, 0,
        "{unlit} of {rooms} resident rooms hold no fixture"
    );

    // (1b) AUDIT PAR1a (b'): every fixture is where its room is, on the loaded
    // island with its REAL terrain — a room light inside its own box, the box's
    // centre and the light inside one building shell of the volume, the light
    // above the ground under it; an exterior light within a metre of a shell.
    let (placed, misplaced, buried) = fixtures_in_their_buildings(&mut ship);
    println!(
        "PAR1a PLACEMENT: {placed} resident fixtures, {} misplaced, {} under the ground",
        misplaced.len(),
        buried.len()
    );
    for m in misplaced.iter().chain(buried.iter().map(|b| &b.1)).take(12) {
        println!("  {m}");
    }
    // The buried are the lowest storeys of buildings on a slope (the datum's
    // ruling), a minority: measured 81 of 2 281 at the audit, 80 on floor 0
    // and one on floor 1 of a steep lot (0.26 m under). A ratchet, not a law.
    let mut by_floor: BTreeMap<u32, usize> = BTreeMap::new();
    for (f, _) in &buried {
        *by_floor.entry(*f).or_default() += 1;
    }
    println!("PAR1a PLACEMENT: under the ground by floor {by_floor:?}");
    assert!(
        buried.iter().all(|(f, _)| *f <= 1),
        "a fixture above the first floor is under the terrain: {:?}",
        buried.iter().find(|(f, _)| *f > 1)
    );
    assert!(
        buried.len() * 20 <= placed,
        "{} of {placed} fixtures are under the ground",
        buried.len()
    );
    assert!(placed > 500, "only {placed} resident fixtures were checked");
    assert!(
        misplaced.is_empty(),
        "{} of {placed} fixtures lie outside their building or under the ground: {:?}",
        misplaced.len(),
        &misplaced[..misplaced.len().min(4)]
    );

    // (2) + (3) the sweep.
    let mut by_hour: Vec<BTreeMap<&'static str, (usize, usize)>> = Vec::new();
    let mut projected = Vec::new();
    for hour in 0..24 {
        let h = f64::from(hour) + 0.5;
        for sim in [&mut ship, &mut pie] {
            stand(sim, at, h, 2);
        }
        let (a, b) = (project_world(&ship), project_world(&pie));
        assert_eq!(
            light_bytes(&a),
            light_bytes(&b),
            "{h:.1} h: the two hosts projected different light lists ({} vs {})",
            a.lights.len(),
            b.lights.len()
        );
        projected.push((a.lights.len(), a.light_bounds.len()));
        let hour_now = inf_ecs::sky::local_hour(ship.world());
        let sun_y = inf_ecs::sky::resolve_sky(ship.world())
            .map(|s| s.sun.y as f32)
            .unwrap_or(1.0);
        let mut row: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        let mut seen: BTreeSet<(usize, u32, u32)> = BTreeSet::new();
        for (vi, (_, lights)) in resident_fixtures(&ship).iter().enumerate() {
            for l in lights {
                if !seen.insert((
                    vi,
                    l.building,
                    if l.room == u32::MAX {
                        u32::MAX - l.row as u32
                    } else {
                        l.room
                    },
                )) {
                    continue;
                }
                let occ = inf_ecs::sky::fixture_occupancy(l.occupancy, l.seed, l.room, hour_now);
                let lvl = inf_ecs::sky::fixture_level(l.schedule, hour_now, sun_y, occ);
                let e = row.entry(class_of(l)).or_default();
                e.0 += 1;
                if lvl > 0.5 {
                    e.1 += 1;
                }
            }
        }
        by_hour.push(row);
    }
    println!("PAR1a SWEEP (rooms lit / rooms, by hour; projected lights, bounds):");
    for (h, row) in by_hour.iter().enumerate() {
        let cells: Vec<String> = row
            .iter()
            .map(|(k, (n, l))| format!("{k} {l}/{n}"))
            .collect();
        println!(
            "  {h:02}:30  {}  | {} lights, {} bounds",
            cells.join(", "),
            projected[h].0,
            projected[h].1
        );
    }
    let share = |h: usize, k: &str| {
        by_hour[h]
            .get(k)
            .map_or(f64::NAN, |(n, l)| *l as f64 / (*n).max(1) as f64)
    };
    assert!(
        share(12, "work") > 0.95,
        "workplaces at noon {}",
        share(12, "work")
    );
    assert!(
        share(2, "work") < 0.15,
        "workplaces at 02:30 {}",
        share(2, "work")
    );
    assert!(
        share(19, "work") > share(2, "work") + 0.1,
        "the evening tail"
    );
    // Homes: lit when it is dark AND the household is up — the living room
    // through the evening, the bedroom going to bed; none at noon (the Dusk
    // half), a minority in the small hours.
    // (The CI island's resident blocks hold apartments' and a hotel's
    // bedrooms; living rooms are held by `inf_ecs`'s own occupancy arm.)
    let held = |k: &str| by_hour[0].get(k).is_some_and(|(n, _)| *n >= 20);
    if held("home living") {
        assert!(share(12, "home living") < 0.05, "living rooms at noon");
        assert!(share(21, "home living") > 0.45, "living rooms at 21:30");
        assert!(share(4, "home living") < 0.12, "living rooms at 04:30");
    }
    assert!(
        held("home bedroom"),
        "the resident blocks hold no dwelling bedrooms"
    );
    assert!(share(12, "home bedroom") < 0.05, "bedrooms lit at noon");
    assert!(
        share(23, "home bedroom") > 1.5 * share(4, "home bedroom"),
        "bedrooms at 23:30 {} vs 04:30 {}",
        share(23, "home bedroom"),
        share(4, "home bedroom")
    );
    if by_hour[0].contains_key("never closes") {
        for h in 0..24 {
            assert_eq!(
                share(h, "never closes"),
                1.0,
                "a never-closing room dark at {h}:30"
            );
        }
    }
    assert!(
        share(12, "exterior") < 0.05 && share(22, "exterior") > 0.95,
        "porches dusk to dawn"
    );
    assert!(
        projected[21].0 > 100 && projected[21].1 + 1 >= projected[21].0,
        "21:30 projects {} lights, {} of them bounded",
        projected[21].0,
        projected[21].1
    );
}

/// **A LIT SHOP IS SEEN FROM THE STREET THROUGH ITS GLASS** (clause 6; FIX3's
/// shop is the proof target).
///
/// Reads: a real ground-floor shop at night from the far pavement, 10 m out
/// on the entrance's axis at eye height: the glazed frontage beside the door
/// (its shopfront modules and panes drawn as PAR0's transmitting glass) with
/// the shop's own fixtures on vs off, and against the solid fascia above it.
/// The base tree hangs no fixture in a shop: its frontage reads the dark.
#[test]
fn a_lit_shop_is_seen_from_the_street_through_its_glass() {
    use inf_pcg::building::OpeningKind;
    let Some(gpu) = gpu() else { return };
    let out = building(ArchetypeId::Shop, 1, 7, true);
    let plan = &out.plan;
    let wi = plan.entrance.expect("a shop has an entrance");
    let w = plan.walls[wi];
    let door = plan
        .openings
        .iter()
        .find(|o| o.wall == wi && o.kind == OpeningKind::Door)
        .copied()
        .expect("the entrance has a door");
    let mid = w.point_at((door.start + door.end) * 0.5);
    let rc = plan.rooms[w.inside].rect.center();
    let d = w.direction();
    let n = {
        let n = DVec2::new(-d.y, d.x);
        if (mid - rc).dot(n) >= 0.0 {
            n
        } else {
            -n
        }
    };
    let y = plan.floor_y(0);
    let eye = DVec3::new(mid.x + n.x * 10.0, y + 1.6, mid.y + n.y * 10.0);
    let target = DVec3::new(mid.x, y + 1.3, mid.y);
    let view = look(eye, target);
    let lit = render(
        &gpu,
        &building_scene(&out, &|l| l.tag.room != FixtureTag::EXTERIOR, true),
        &view,
    );
    let dark = render(&gpu, &building_scene(&out, &|_| false, true), &view);
    dump("shop_lit", &lit);
    dump("shop_dark", &dark);
    // The frontage: the door itself and the glazing either side of it, at
    // shop-window height; the fascia: the wall above the door's head.
    let front = |img: &[u8]| {
        [-1.6f64, 0.0, 1.6]
            .iter()
            .map(|s| {
                let p = mid + d * *s;
                patch_at(img, &view, DVec3::new(p.x, y + 1.3, p.y), 8)
            })
            .fold(0.0f64, f64::max)
    };
    let fascia = patch_at(
        &lit,
        &view,
        DVec3::new(mid.x, y + door.head + 0.4, mid.y) + DVec3::new(n.x, 0.0, n.y) * 0.2,
        6,
    );
    let (fl, fd) = (front(&lit), front(&dark));
    let lamps = out
        .lights
        .iter()
        .filter(|l| l.tag.room != FixtureTag::EXTERIOR)
        .count();
    println!("PAR1a SHOP: {lamps} interior fixture(s); the frontage from 10 m: lit {fl:.1}, dark {fd:.1}; the fascia above the door {fascia:.1}");
    assert!(
        fl >= fd + 15.0,
        "the lit shop's frontage {fl:.1} vs dark {fd:.1}"
    );
    assert!(
        fl >= fascia + 10.0,
        "the frontage {fl:.1} vs the fascia {fascia:.1}"
    );
}

/// **THE FAR BAND, MEASURED** (clause 5): a room fixture is a light only
/// inside its draw distance (`fixtures::ROOM_DRAW_M`); past it `plan_lights`
/// drops it (`LightPlan::culled_distance`). This arm reads the plan's
/// counters on both sides of the swap and the lit window's luminance at the
/// swap distance — the step a viewer sees when the light leaves — and REPORTS
/// it (the far band that would hide the step is carried, priced, in the
/// report: no pane glow stands in for it).
#[test]
fn a_room_fixture_is_dropped_past_its_draw_distance_and_the_swap_is_measured() {
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
    let mine = |l: &PcgLight| l.tag.room == ri as u32;
    let mut scene = building_scene(&out, &mine, true);
    for b in &mut scene.light_bounds {
        b.draw_m = fixtures::ROOM_DRAW_M;
    }
    let lamp = out.lights.iter().find(|l| mine(l)).expect("a lamp");
    let mut counts = Vec::new();
    for metres in [
        fixtures::ROOM_DRAW_M as f64 - 5.0,
        fixtures::ROOM_DRAW_M as f64 + lamp.range_m as f64 + 5.0,
    ] {
        let eye = window + DVec3::new(n.x, 0.0, n.y) * metres;
        let view = look(eye, window);
        let plan_now = inf_render::lights::plan_lights(
            &scene,
            &view,
            &[],
            &inf_render::lights::LightSettings::default(),
            1.0,
        );
        counts.push((metres, plan_now.local(), plan_now.culled_distance));
    }
    println!("PAR1a FAR BAND: (metres, local lights, culled by distance) {counts:?}");
    // The draw distance IS the structure LOD both hosts swap a building's
    // parts for its shell at: past it no window is drawn to see a room through.
    assert_eq!(
        f64::from(fixtures::ROOM_DRAW_M),
        inf_render::STRUCTURE_LOD_M
    );
    assert_eq!(
        counts[0].1, 1,
        "inside its draw distance the fixture is a light"
    );
    assert_eq!(
        (counts[1].1, counts[1].2),
        (0, 1),
        "past it the fixture is dropped"
    );
    if let Some(gpu) = gpu() {
        let dark_scene = building_scene(&out, &|_| false, true);
        for metres in [20.0, 45.0, fixtures::ROOM_DRAW_M as f64 - 5.0] {
            let eye = window + DVec3::new(n.x, 0.0, n.y) * metres;
            let view = look(eye, window);
            let lit = render(&gpu, &scene, &view);
            let dark = render(&gpu, &dark_scene, &view);
            dump(&format!("far_{metres:.0}_lit"), &lit);
            let (l, d) = (
                patch_at(&lit, &view, window, 3),
                patch_at(&dark, &view, window, 3),
            );
            println!(
                "PAR1a FAR BAND: the lit window at {metres:.0} m reads {l:.1} at x{NIGHT_EXPOSURE}, dark {d:.1} -- the step the light's departure would leave"
            );
        }
    }
}
