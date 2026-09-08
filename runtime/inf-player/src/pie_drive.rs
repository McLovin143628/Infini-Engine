//! **Driving and reading a headless PIE session** (wave FIX1).
//!
//! Two doors, and they are the whole of what protocol 3 added:
//!
//! * [`PieInputHost`] turns an `InputFrame` into a stepped simulation. It is
//!   the headless twin of `PlayerApp::frame`'s input half and it is spelled the
//!   same way on purpose — the key-code diff, the dialog's first refusal, the
//!   `apply_dt`, the panel edges read from the *resolved* state, the sim pause,
//!   `held_actions`, `step_once`. Nothing here writes an action name or an intent
//!   field; a gate presses a **key** and the shipped binding table decides what
//!   it means, which is the only form in which a swapped binding can red an arm.
//! * [`world_probe`] answers "what does the world look like now" by reading the
//!   shipped components. A `state_hash` can say *that* something changed and
//!   never *what*, and "the hero moved 3.75 m/s to its right" is not a claim a
//!   `u64` can carry.
//!
//! # The step boundary is load-bearing
//!
//! CP-C6's launch figures are exact to 1e-6 because the expected speed is
//! `tuning + GRAVITY.y·DT`: the verb writes the speed and *the same fixed step*
//! integrates one step of the fall. So a wire frame reaches `RuntimeSim` through
//! `step_once` and through nothing else — an input applied on one side of the
//! step and read on the other would move every one of those numbers by a step of
//! gravity, and the arms would still pass, wrongly.

use std::path::PathBuf;

use inf_input::{InputEvent, InputMap, InputState};
use inf_runtime::pie::{ActorProbe, InputFrame, WorldProbe};
use uuid::Uuid;

use crate::runtime_sim::RuntimeSim;

/// The fixed-step seconds a frame carries when it does not say.
const FALLBACK_DT: f64 = 1.0 / 60.0;

/// **A headless PIE session's input**: the shipped table, the shipped dialog and
/// the shipped reduction, driven by wire frames instead of by winit events.
pub struct PieInputHost {
    state: InputState,
    ui: crate::ui::PlayerUi,
    /// What the last frame resolved to. A session that sends input and then
    /// `Resume` would otherwise auto-advance with an EMPTY input and silently
    /// drop the keys it is holding — the failure mode of a wire that carries
    /// levels but a loop that forgets them between frames.
    held: crate::runtime_sim::RuntimeInput,
    /// The key codes the previous frame held — the diff that makes a level into
    /// a pair of edges.
    keys: Vec<String>,
    /// …and the mouse buttons, for the same reason.
    buttons: Vec<u8>,
}

impl PieInputHost {
    /// Open one over `map` (the LEVEL's table) with the player's own settings
    /// read from `settings_dir` — the same pair `PlayerApp::new` opens, so a
    /// rebinding a session made is one this host reads too.
    pub fn open(settings_dir: PathBuf, map: InputMap, sim: &mut RuntimeSim) -> Self {
        let (ui, map) = crate::ui::PlayerUi::open(settings_dir, map);
        if let Some(e) = &ui.load_error {
            tracing::warn!("inf-player: {e}");
        }
        ui.apply_to_sim(sim);
        Self {
            state: InputState::new(map),
            ui,
            held: crate::runtime_sim::RuntimeInput::default(),
            keys: Vec::new(),
            buttons: Vec::new(),
        }
    }

    /// The input the last applied frame resolved to — what an auto-advancing
    /// session steps with between frames.
    pub fn held(&self) -> crate::runtime_sim::RuntimeInput {
        self.held.clone()
    }

    /// The in-game UI this host drives — read by [`world_probe`] for the panel
    /// flags, which are decisions about the *session* rather than about the
    /// character and therefore live nowhere in the world.
    pub fn ui(&self) -> &crate::ui::PlayerUi {
        &self.ui
    }

    /// **Apply one wire frame and advance the simulation.**
    ///
    /// Returns the number of fixed steps taken (`frame.steps`, or zero when the
    /// frame only sets input up). Every line below has a twin in
    /// `PlayerApp::frame`; where the two differ it is stated.
    pub fn apply(&mut self, sim: &mut RuntimeSim, frame: &InputFrame) -> u32 {
        let dt = if frame.dt.is_finite() && frame.dt > 0.0 {
            frame.dt
        } else {
            FALLBACK_DT
        };

        // 1. The key LEVEL becomes a pair of edges against the last frame.
        let mut events: Vec<InputEvent> = Vec::new();
        let mut key_edges: Vec<(String, bool)> = Vec::new();
        for code in &frame.keys {
            if !self.keys.iter().any(|k| k == code) {
                key_edges.push((code.clone(), true));
            }
        }
        for code in &self.keys {
            if !frame.keys.iter().any(|k| k == code) {
                key_edges.push((code.clone(), false));
            }
        }
        self.keys = frame.keys.clone();

        // 2. **The dialog gets it first, and what it takes never reaches the
        //    game** — the same rule and the same reason as the windowed host: the
        //    press that moves a menu cursor must not also fire a weapon, and a
        //    key being captured for a rebinding must not fire the verb it is
        //    being taken from.
        for (code, pressed) in key_edges {
            let mut map = self.state.map().clone();
            let verdict = self.ui.key(&code, pressed, &mut map);
            if verdict.changed() {
                // Rebuilt from the level's table rather than patched: the look
                // tuning is a multiplier on what the project authored, and
                // applying it to the live map would compound it once a frame.
                self.state.set_map(self.ui.tuned_map());
                self.ui.apply_to_sim(sim);
            }
            if verdict.consumed {
                continue;
            }
            events.push(InputEvent::Key { code, pressed });
        }

        // 3. The mouse. Motion and wheel are deltas; buttons are a level, so they
        //    diff exactly as the keys do.
        if frame.motion != [0.0, 0.0] {
            events.push(InputEvent::MouseMotion {
                delta: frame.motion,
            });
        }
        if frame.wheel != [0.0, 0.0] {
            events.push(InputEvent::MouseWheel { delta: frame.wheel });
        }
        let mut button_edges: Vec<(u8, bool)> = Vec::new();
        for b in &frame.buttons {
            if !self.buttons.contains(b) {
                button_edges.push((*b, true));
            }
        }
        for b in &self.buttons {
            if !frame.buttons.contains(b) {
                button_edges.push((*b, false));
            }
        }
        self.buttons = frame.buttons.clone();
        for (index, pressed) in button_edges {
            let Some(button) = wire_mouse_button(index) else {
                continue;
            };
            if pressed {
                // A mouse button is a bindable source, so a running capture takes
                // it. A release is never consumed — see `PlayerUi::key` for the
                // stuck-key measurement that says why.
                let mut map = self.state.map().clone();
                let verdict = self.ui.mouse(button, &mut map);
                if verdict.changed() {
                    self.state.set_map(self.ui.tuned_map());
                    self.ui.apply_to_sim(sim);
                }
                if verdict.consumed {
                    continue;
                }
            }
            events.push(InputEvent::MouseButton { button, pressed });
        }

        // 4. Resolve. `apply_dt`, not `apply`: the frame time is what makes
        //    `InputState::hold_s` a duration rather than a zero, and that is the
        //    in-game UI's clock — a long press is timed by it.
        self.state.apply_dt(&events, dt);

        // 5. The panel edges, read from the RESOLVED state rather than from the
        //    key, so a player who rebound the menu opens it with what they bound.
        if self.state.just_pressed(inf_input::actions::MENU) {
            self.ui.toggle();
        }
        if self.state.just_pressed(inf_input::actions::INVENTORY) {
            self.ui.toggle_inventory();
        }
        self.ui.set_bag(bag_view(sim));
        for verb in self.ui.take_inventory_verbs() {
            sim.apply_inventory_verb(verb);
        }
        // The pause is on the SIM rather than on this host.
        sim.set_sim_paused(self.ui.pauses_sim());
        self.ui.report_unconsumed(&self.state);

        // 6. …and the step. `step_once`, once per requested step, with the input
        //    resolved once — see this module's header for why the boundary is
        //    load-bearing.
        let input = crate::input::held_actions(&self.state, dt);
        self.held = input.clone();
        for _ in 0..frame.steps {
            sim.step_once(input.clone());
        }
        frame.steps
    }
}

/// The wire's mouse-button index (`0` Left … `4` Forward).
fn wire_mouse_button(index: u8) -> Option<inf_input::MouseButton> {
    Some(match index {
        0 => inf_input::MouseButton::Left,
        1 => inf_input::MouseButton::Right,
        2 => inf_input::MouseButton::Middle,
        3 => inf_input::MouseButton::Back,
        4 => inf_input::MouseButton::Forward,
        _ => return None,
    })
}

/// What the inventory panel is showing — projected out of the sim exactly as the
/// windowed host projects it.
fn bag_view(sim: &RuntimeSim) -> inf_ui::InventoryView {
    let Some(hero) = inf_ecs::movement::camera_subject(sim.world()) else {
        return inf_ui::InventoryView::default();
    };
    let world = sim.world();
    let Some(inv) = inf_ecs::item::inventory_of(world, hero) else {
        return inf_ui::InventoryView::default();
    };
    let defs = inf_ecs::item::item_defs(world);
    inf_ui::InventoryView {
        slots: inv
            .slots
            .iter()
            .enumerate()
            .map(|(i, slot)| match slot {
                Some(s) => {
                    let def = defs.and_then(|d| d.get(&s.id));
                    inf_ui::InventorySlot {
                        label: def.map(|d| d.label.clone()).unwrap_or_else(|| s.id.clone()),
                        count: s.count,
                        equipped: inv.equipped == Some(i),
                        equippable: def.is_some_and(|d| d.is_weapon()),
                    }
                }
                None => inf_ui::InventorySlot::default(),
            })
            .collect(),
    }
}

/// **Read one actor out of the world.** `None` when the guid names nothing.
pub fn actor_probe(sim: &RuntimeSim, guid: Uuid) -> Option<ActorProbe> {
    use inf_ecs::components::{CharacterMovement, Name, Transform};
    let ecs = sim.world();
    let entity = ecs.entity_of(guid)?;
    let w = ecs.world();
    let transform = w.get::<Transform>(entity)?;
    let cm = w.get::<CharacterMovement>(entity);
    let (velocity, local, speed, vertical, grounded, mode, gait, rotation, aim) = match cm {
        Some(cm) => {
            let v = cm.runtime.velocity;
            let local = inf_ecs::movement::rotate_into_frame(
                inf_ecs::math::Vec2d::new(v.x, v.z),
                cm.runtime.aim_yaw_deg,
            );
            (
                [v.x, v.y, v.z],
                [local.x, local.y],
                (v.x * v.x + v.z * v.z).sqrt(),
                v.y,
                cm.runtime.grounded,
                format!("{:?}", cm.mode),
                format!("{:?}", cm.runtime.actual_gait),
                format!("{:?}", cm.rotation_mode),
                cm.runtime.aim_yaw_deg,
            )
        }
        None => (
            [0.0; 3],
            [0.0; 2],
            0.0,
            0.0,
            false,
            String::new(),
            String::new(),
            String::new(),
            0.0,
        ),
    };
    let inv = inf_ecs::item::inventory_of(ecs, guid);
    // **The pose, and whether it is the REST pose.** This is the T-pose question
    // and it is answered here rather than inferred from a byte count: a state
    // machine whose clips do not resolve publishes a full, correctly-sized pose
    // that IS the bind pose, so every "the pose store is N bytes" assertion in
    // this repository is green while the character stands in a T.
    let (pose_is_rest, pose_joints, pose_max_delta) = match inf_ecs::pose::evaluated_pose(ecs, guid)
    {
        Some(p) => match sim.skeleton_of(p.skeleton) {
            Some(asset) => {
                let rest = inf_anim::Pose::rest(&asset.skeleton);
                let d = pose_departure(&rest, &p.pose);
                (d == 0.0, p.pose.len() as u32, d)
            }
            None => (false, p.pose.len() as u32, f64::NAN),
        },
        None => (true, 0, 0.0),
    };
    Some(ActorProbe {
        guid: *guid.as_bytes(),
        name: w
            .get::<Name>(entity)
            .map(|n| n.0.clone())
            .unwrap_or_default(),
        position: [
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ],
        velocity,
        local_velocity: local,
        speed,
        vertical_speed: vertical,
        grounded,
        movement_mode: mode,
        gait,
        rotation_mode: rotation,
        aim_yaw_deg: aim,
        magazine: w
            .get::<inf_ecs::weapon::WeaponState>(entity)
            .map(|s| s.magazine)
            .unwrap_or(0),
        equipped: inv
            .and_then(|i| i.equipped_id())
            .map(str::to_string)
            .unwrap_or_default(),
        bag: inv
            .map(|i| {
                i.slots
                    .iter()
                    .flatten()
                    .map(|s| (s.id.clone(), s.count))
                    .collect()
            })
            .unwrap_or_default(),
        pose_is_rest,
        pose_joints,
        pose_max_delta,
    })
}

/// **How far a pose has departed from the bind pose**, as one scalar: the largest
/// per-joint (translation metres + quaternion component distance). Exactly `0.0`
/// is the rest pose, and that exactness is the point — the arm that matters is
/// "not rest", not "close to rest".
fn pose_departure(rest: &inf_anim::Pose, pose: &inf_anim::Pose) -> f64 {
    let n = rest.locals.len().min(pose.locals.len());
    let mut worst = 0.0f64;
    for i in 0..n {
        let a = &rest.locals[i];
        let b = &pose.locals[i];
        let dt = (0..3)
            .map(|k| (f64::from(a.translation[k]) - f64::from(b.translation[k])).powi(2))
            .sum::<f64>()
            .sqrt();
        let dr = (0..4)
            .map(|k| (f64::from(a.rotation[k]) - f64::from(b.rotation[k])).powi(2))
            .sum::<f64>()
            .sqrt();
        worst = worst.max(dt + dr);
    }
    worst
}

/// **The whole session, read.** `named` also reports that actor by guid.
pub fn world_probe(
    sim: &RuntimeSim,
    ui: Option<&crate::ui::PlayerUi>,
    frame: u64,
    named: Option<Uuid>,
) -> WorldProbe {
    let hero = inf_ecs::movement::camera_subject(sim.world());
    let camera = sim.camera();
    let pose = camera.pose;
    let focus = sim.camera_focus();
    WorldProbe {
        steps: sim.steps(),
        frame,
        shots: sim.gameplay().shots,
        menu_open: ui.is_some_and(|u| u.menu.open),
        inventory_open: ui.is_some_and(|u| u.inventory.open),
        camera_eye: [pose.position.x, pose.position.y, pose.position.z],
        camera_focus: [focus.x, focus.y, focus.z],
        camera_pull_in_m: camera.collision_pull_m,
        // **Scene entities**, not bevy's allocation high-water mark: a despawn
        // frees an id but `Entities::len` does not always fall, and "the pickup
        // is gone from the world" is a claim that needs a count that does.
        entities: {
            let w = sim.world().world();
            match w.try_query::<&inf_ecs::components::Guid>() {
                Some(mut q) => q.iter(w).count() as u32,
                None => w.entities().len(),
            }
        },
        hero: hero.and_then(|g| actor_probe(sim, g)),
        named: named.and_then(|g| actor_probe(sim, g)),
    }
}

/// **The env var that turns on the demo loop's hero log** (wave FIX1).
pub const HERO_LOG_ENV: &str = "INF_PIE_HERO_LOG";

/// **Where a PREVIEW session should put the hero, and when** (CHAR1b.2 audit) —
/// `"x,y,z@t"` world metres and seconds, `;`-separated, and nothing at all when
/// unset. The `@t` is optional and defaults to [`SPAWN_DELAY_S`].
///
/// # Why this exists, and why it is not a feature
///
/// Wave CHAR1b.2 turned on the mantle, the swim set and landing-by-height, and
/// filmed **none** of them: the demo loop drives the game through the keyboard,
/// the ledge course, the island's water and a drop from a measured height are
/// none of them reachable on foot from where the level puts the hero inside a
/// ninety-second session, and the wave wrote that down as a limitation of the
/// player ("it has no teleport"). It is a limitation of the *loop*: the same
/// sentence would excuse never photographing anything more than a minute's walk
/// from a player start.
///
/// So this is the loop's own stopwatch, in the same shape as [`HERO_LOG_ENV`]:
/// an env door read **once**, applied **once**, in a **preview** session only
/// (`--pie`, which is what the editor's Play button launches), inert in every
/// session that did not ask for it, and never written into a level. A shipped
/// player boot does not consult it — see [`SpawnOverride::tick`]'s caller.
///
/// It is not a movement mode, not an action, not a key, and no gate reads it:
/// a gate that needed a character somewhere puts it there through the ECS the
/// way `char1b_gate` already does.
///
/// # The grammar, and the heading (COV1 audit, carried 186)
///
/// `x,y,z[@seconds][/yaw_deg]`, semicolon-separated. The `/yaw` suffix is the
/// audit's addition and the reason for it is a measured one: a placement set a
/// POSITION and nothing else, so a scripted leg reached wherever the hero
/// happened to be looking — and a character standing still in
/// `VelocityDirection` does not turn its body under the mouse, so the demo
/// loop's cover leg had to sweep sixteen presses and hold `W` to face a wall
/// four metres in front of it. With a heading the leg is deterministic.
pub const SPAWN_AT_ENV: &str = "INF_PIE_SPAWN_AT";

/// **A garment to put on the hero in a PREVIEW session** (CHAR1b.2 audit) — the
/// asset GUID of a `.inf_cloth`, and nothing at all when unset.
///
/// The cape wave CHAR1b.2 authored is written into the island project's Content
/// and worn **in the gate**; putting it on the hero in the committed `.inf_lvl`
/// is a level edit (carried 137, OUTFIT1's). This is how the loop photographs it
/// without making that edit — the same one-shot, preview-only, env-gated door
/// [`SPAWN_AT_ENV`] is.
///
/// **It moved to the wire crate at wave OUTFIT1** (carried 142) and this is a
/// re-export, not a copy. The player inserts a `ClothSim` naming this GUID; the
/// EDITOR decides which garments the payload carries, and it decided by walking
/// the document — which never names a runtime-worn one. Two doors, one string:
/// `inf_editor_core::pie` reads the same constant when it builds the payload, so
/// the garment the player is about to put on is in the bytes it is handed.
pub use inf_runtime::pie::WEAR_CLOTH_ENV;

/// How long a preview waits before applying [`SPAWN_AT_ENV`], seconds.
///
/// The island streams; a hero teleported on frame zero arrives before the
/// terrain under it does and falls through the world. Two seconds is past the
/// first cell load on this machine and far inside the loop's own settle.
const SPAWN_DELAY_S: f64 = 2.0;

/// **The one-shot placement [`SPAWN_AT_ENV`] and [`WEAR_CLOTH_ENV`] describe.**
///
/// Inert unless the variables are set: `from_env` returns a value whose `tick`
/// returns immediately, so a session that did not ask for one pays one
/// `Option` check per frame, which is what the hero log costs too.
#[derive(Default)]
pub struct SpawnOverride {
    /// `(x, y, z, at_seconds, facing_deg)`, in the order given; each fires
    /// once. `facing_deg` is `None` when the entry gave no heading, which is
    /// every entry written before the COV1 audit.
    at: Vec<([f64; 3], f64, Option<f64>)>,
    cloth: Option<Uuid>,
    accum: f64,
    next: usize,
    cloth_done: bool,
}

impl SpawnOverride {
    /// Read [`SPAWN_AT_ENV`] and [`WEAR_CLOTH_ENV`], or an inert override.
    ///
    /// A malformed value is a **refusal with a reason on stderr**, not a
    /// silently ignored one: the whole point of the door is that the operator
    /// finds out whether it took.
    pub fn from_env() -> Self {
        let mut at: Vec<([f64; 3], f64, Option<f64>)> = Vec::new();
        if let Ok(v) = std::env::var(SPAWN_AT_ENV) {
            for entry in v.split(';').map(str::trim).filter(|e| !e.is_empty()) {
                // **`/yaw` is the HEADING** (COV1 audit, carried 186). A
                // placement set a position and nothing else, so the facing was
                // whatever the hero happened to have — and a character standing
                // still in `VelocityDirection` does not turn under the mouse, so
                // a scripted leg that wanted to face a wall had to WALK at one.
                // The demo loop's cover leg is the caller that needed it and
                // `demo.ps1`'s `-FaceAt` is the switch that writes it.
                let (entry_body, facing) = match entry.split_once('/') {
                    Some((b, y)) => (b.trim(), y.trim().parse::<f64>().ok()),
                    None => (entry, None),
                };
                let (coords, when) = match entry_body.split_once('@') {
                    Some((c, t)) => (c, t.trim().parse::<f64>().unwrap_or(SPAWN_DELAY_S)),
                    None => (entry_body, SPAWN_DELAY_S),
                };
                let parts: Vec<f64> = coords
                    .split(',')
                    .filter_map(|p| p.trim().parse::<f64>().ok())
                    .collect();
                let facing = facing.filter(|f| f.is_finite());
                if parts.len() == 3 && parts.iter().all(|p| p.is_finite()) && when.is_finite() {
                    at.push((
                        [parts[0], parts[1], parts[2]],
                        when.max(SPAWN_DELAY_S),
                        facing,
                    ));
                } else {
                    eprintln!(
                        "inf-player: {SPAWN_AT_ENV} entry `{entry}` is not `x,y,z[@s][/yaw]`"
                    );
                }
            }
            at.sort_by(|a, b| a.1.total_cmp(&b.1));
        }
        // The same read the editor's payload builder does, through the same
        // door: a garment worn here that the payload did not carry resolves to
        // nothing and draws nothing, which is what carried 142 was.
        let cloth = inf_runtime::pie::preview_worn_cloth();
        Self {
            at,
            cloth,
            ..Self::default()
        }
    }

    /// The placements this override still has to make.
    pub fn pending(&self) -> usize {
        self.at.len().saturating_sub(self.next)
    }

    /// Apply whichever placement is due, at most one per frame. Returns a line
    /// for the hero log when one fires, so a frame taken afterwards can be read
    /// against a record of where the hero was put.
    pub fn tick(&mut self, sim: &mut RuntimeSim, dt: f64) -> Option<String> {
        let done = self.next >= self.at.len();
        if done && (self.cloth.is_none() || self.cloth_done) {
            return None;
        }
        self.accum += dt;
        if self.accum < SPAWN_DELAY_S {
            return None;
        }
        let due = (!done && self.accum >= self.at[self.next].1).then(|| self.at[self.next]);
        let wear = self.cloth.filter(|_| !self.cloth_done);
        if due.is_none() && wear.is_none() {
            return None;
        }
        let hero = inf_ecs::movement::camera_subject(sim.world())?;
        let mut said = String::new();
        {
            let w = sim.world_mut();
            let entity = w.entity_of(hero)?;
            if let Some((at, when, facing)) = due {
                if let Some(mut t) = w
                    .world_mut()
                    .get_mut::<inf_ecs::components::Transform>(entity)
                {
                    t.translation.x = at[0];
                    t.translation.y = at[1];
                    t.translation.z = at[2];
                    if let Some(y) = facing {
                        t.rotation.y = y;
                    }
                }
                if let Some(mut cm) = w
                    .world_mut()
                    .get_mut::<inf_ecs::components::CharacterMovement>(entity)
                {
                    cm.runtime.velocity = inf_ecs::math::Vec3d::ZERO;
                    // **The heading is the BODY's, the AIM's and the target's**
                    // — all three, because the smoother turns the body toward
                    // `target_yaw_deg` and the cover probe reaches in the body's
                    // facing when no stick is held. Setting one of the three
                    // would have the character turn back on the next step.
                    if let Some(y) = facing {
                        cm.runtime.body_yaw_deg = y;
                        cm.runtime.target_yaw_deg = y;
                        cm.runtime.aim_yaw_deg = y;
                    }
                }
                self.next += 1;
                said.push_str(&format!(
                    "{SPAWN_AT_ENV} #{} at t={when:.0}s placed the hero at {:.2},{:.2},{:.2}",
                    self.next, at[0], at[1], at[2]
                ));
                if let Some(y) = facing {
                    said.push_str(&format!(" facing {y:.0} deg"));
                }
            }
            if let Some(guid) = wear {
                w.world_mut()
                    .entity_mut(entity)
                    .insert(inf_ecs::components::ClothSim {
                        asset: Some(guid),
                        enabled: true,
                        ..Default::default()
                    });
                self.cloth_done = true;
                if !said.is_empty() {
                    said.push_str("; ");
                }
                said.push_str(&format!("{WEAR_CLOTH_ENV} put {guid} on the hero"));
            }
        }
        (!said.is_empty()).then_some(said)
    }
}

/// How often a windowed PIE session appends a line, seconds.
const HERO_LOG_PERIOD_S: f64 = 0.25;

/// **Where the hero is, written down while a person is watching** (wave FIX1).
///
/// The demo loop's whole job is to end a wave with something the author can
/// judge, and "the hero moved" is not a claim two screenshots can make on their
/// own — a camera that drifts looks the same as a character that walks. A
/// windowed PIE session appends
/// `t,frame,x,y,z,mode,speed,camera_pull,aim_yaw,head_yaw,head_pitch,anim_state,foot_mm`
/// here four times a second and the script prints the first and last lines
/// beside its frames.
///
/// The last four columns arrived with wave CHAR1b.1 and are what makes the
/// look-at frames a measurement rather than a picture: `aim_yaw` is where the
/// mouse has put the camera relative to the body, `head_yaw`/`head_pitch` are
/// what the chain actually drew (`inf_anim::LookAtReport`), and `anim_state` is
/// the machine state the pose came out of — so "the head follows the mouse" and
/// "the character is walking" are both answerable from the file rather than from
/// the frame.
///
/// **Only when the variable names a path**, and the file is opened once: a
/// shipped player writes nothing, opens nothing and pays one `Option` check per
/// frame. It is an instrument, not a feature, and it deliberately does not go
/// through `tracing` — the `--pie` entry installs no subscriber, because one that
/// teed to stdout would corrupt the protocol stream.
#[derive(Default)]
pub struct HeroLog {
    file: Option<std::fs::File>,
    accum: f64,
}

impl HeroLog {
    /// Open the log named by [`HERO_LOG_ENV`], or an inert one.
    pub fn from_env() -> Self {
        let Ok(path) = std::env::var(HERO_LOG_ENV) else {
            return Self::default();
        };
        if path.trim().is_empty() {
            return Self::default();
        }
        match std::fs::File::create(&path) {
            Ok(file) => Self {
                file: Some(file),
                accum: 0.0,
            },
            Err(e) => {
                eprintln!("inf-player: cannot open the hero log at {path}: {e}");
                Self::default()
            }
        }
    }

    /// **Write one diagnostic line into the same log**, immediately.
    ///
    /// The demo loop reads this file and the editor's Output Log is behind the
    /// game's own window, so a session driven by a script has exactly one place
    /// to say what the operating system told it. Used by the keyboard grab.
    pub fn note(&mut self, text: &str) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        use std::io::Write as _;
        let _ = file.write_all(
            format!(
                "# {text}
"
            )
            .as_bytes(),
        );
        let _ = file.flush();
    }

    /// Append a line if enough wall clock has passed. Inert when no path was set.
    pub fn tick(&mut self, sim: &RuntimeSim, frame: u64, dt: f64) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        self.accum += dt;
        if self.accum < HERO_LOG_PERIOD_S {
            return;
        }
        self.accum = 0.0;
        let probe = world_probe(sim, None, frame, None);
        // **What the look-at chain drew, and what asked for it** (wave
        // CHAR1b.1). Read off the animation bridge, which is where the pose step
        // publishes it — not recomputed here, or the log would be a second
        // opinion rather than a record.
        let guid = inf_ecs::movement::camera_subject(sim.world());
        let look = guid.and_then(|g| inf_ecs::anim_bridge::look_report(sim.world(), g));
        let aim_yaw = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
            })
            .map(|cm| {
                inf_ecs::movement::angle_delta_deg(cm.runtime.aim_yaw_deg, cm.runtime.body_yaw_deg)
            })
            .unwrap_or(0.0);
        let state = guid
            .and_then(|g| inf_ecs::anim_bridge::anim_state(sim.world(), g))
            .map(|s| s.name.clone())
            .unwrap_or_default();
        // **The foot-IK residual, in millimetres**, worst of the two — the one
        // number the mandate's "the feet of characters sink in through the
        // surface a little bit" is answerable with in a RUNNING GAME. Negative
        // is a sole below the ground the probe found. Blank for a character with
        // no goals, which is what "the mechanism did not run" has to look like.
        let foot_mm = guid
            .and_then(|g| inf_ecs::anim_bridge::foot_error(sim.world(), g))
            .map(|e| {
                e.iter()
                    .flatten()
                    .fold(0.0f64, |a, v| if v.abs() > a.abs() { *v } else { a })
                    * 1000.0
            });
        // **THE CAMERA'S OWN FOUR NUMBERS** (wave CHAR1c), APPENDED so every
        // column index a script already reads keeps its meaning. Column 7 has
        // carried the clip since P29.6; these are the boom it left, how much of
        // the hero's body is being drawn, how far the whisker fan steered, and
        // who is holding the camera. A frame captioned "the boom at its authored
        // length" wants a number beside it, and this is where the number is.
        let cam = sim.camera();
        let holder = match cam.director.holder().map(|h| h.0) {
            Some(inf_ecs::camera::CameraLayer::Scripted) => "scripted",
            Some(inf_ecs::camera::CameraLayer::Override) => "override",
            Some(inf_ecs::camera::CameraLayer::Gameplay) | None => "gameplay",
        };
        // **THE COVER COLUMNS** (wave COV1), APPENDED for the same reason the
        // camera's four were: every column index a script already reads keeps
        // its meaning. Column 5 already carries the MODE, so `Cover` shows
        // there; these are the three things the mode cannot say — which class
        // of surface, which way the body is leaning out of it, and how far. A
        // character not in cover writes `-`, `-` and `0.000`, which is what a
        // predicate over them reads as "not in cover".
        let cover = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
            })
            .map(|cm| cm.runtime.cover)
            .unwrap_or_default();
        let (cover_class, cover_side) = if cover.active {
            (format!("{:?}", cover.class), format!("{:?}", cover.side))
        } else {
            ("-".to_string(), "-".to_string())
        };
        let line = match &probe.hero {
            Some(h) => format!(
                "{:.3},{},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.2},{:.2},{:.2},{},{},\
                 {:.4},{:.4},{:.2},{},{},{},{:.3}\n",
                sim.steps() as f64 / 60.0,
                probe.frame,
                h.position[0],
                h.position[1],
                h.position[2],
                if h.movement_mode.is_empty() {
                    "-"
                } else {
                    &h.movement_mode
                },
                h.speed,
                probe.camera_pull_in_m,
                aim_yaw,
                look.map(|l| l.total_yaw_deg).unwrap_or(0.0),
                look.map(|l| l.head_pitch_deg).unwrap_or(0.0),
                if state.is_empty() {
                    "-".to_string()
                } else {
                    state
                },
                match foot_mm {
                    Some(mm) => format!("{mm:.3}"),
                    None => String::new(),
                },
                cam.arm_m,
                cam.subject_fade,
                cam.whisker_steer_deg,
                holder,
                cover_class,
                cover_side,
                if cover.active { cover.peek } else { 0.0 }
            ),
            None => format!(
                "{:.3},{},,,,,no-hero,,,,,,,,,,,,,\n",
                sim.steps() as f64 / 60.0,
                probe.frame
            ),
        };
        use std::io::Write as _;
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
}
