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
        // The demo loop's probe reads the BAG; the attachment bench is drawn by
        // the panel and has no probe of its own. Empty rather than projected,
        // stated so the omission is a decision: what `hero.csv` carries about
        // attachments is the FOLD's own summary column (wave WPN2d), which is
        // the thing a frame is triggered on.
        bench: Vec::new(),
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

/// **Where a session writes a WAV of what its audio engine plays** (wave
/// VEH3e), or unset for none — see `RuntimeSim::capture_audio_to`. An
/// instrument like [`HERO_LOG_ENV`]: a capture session plays nothing out loud,
/// because its mixer renders to the file instead of to a device.
pub const RENDER_AUDIO_ENV: &str = "INF_RENDER_AUDIO";

/// **`INF_AUDIO_DEVICE=off`** (VEH3e audit) keeps a windowed session off the
/// output device: a demo run that should not make a noise, a machine whose
/// default device misbehaves. Anything else, or unset, opens it.
pub const AUDIO_DEVICE_ENV: &str = "INF_AUDIO_DEVICE";

/// Whether [`AUDIO_DEVICE_ENV`] turns the device off.
pub fn audio_device_off() -> bool {
    std::env::var(AUDIO_DEVICE_ENV)
        .map(|v| {
            let v = v.trim();
            v.eq_ignore_ascii_case("off") || v == "0"
        })
        .unwrap_or(false)
}

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

/// **Weapons to put in the hero's hands in a PREVIEW session** (wave WPN2a) —
/// a `;`-separated list of ids from `inf_ecs::weapon::WEAPON_REGISTRY_TOML`, and
/// nothing at all when unset.
///
/// **All of them go in the bag and the FIRST is equipped**, which is what makes
/// one session able to photograph one weapon of each class: the demo loop cycles
/// the rest in with the scroll wheel, through `Inventory::cycle_equipped` — the
/// shipped `weapon_switch` verb — rather than through a second door.
///
/// # Why an env door and not a level edit
///
/// [`SPAWN_AT_ENV`]'s argument, verbatim, one wave along. The eighty-five-row
/// registry reaches a LEVEL through the `item.define` node — that is what the
/// `phase30-gameplay` fixture does, and `wpn2a_gate` fires one weapon of each
/// class off it — but the ISLAND has no Blueprint of its own to put the call in:
/// the only class on it is the wizard's committed character controller, shared
/// by every character the New Character wizard has ever made, and putting a
/// weapon catalogue in that would arm every one of them.
///
/// So this is the demo loop's own door, in exactly [`SPAWN_AT_ENV`]'s shape:
/// read **once**, applied **once**, in a **preview** session only, inert in
/// every session that did not ask for it, and never written into a level. No
/// gate reads it — `wpn2a_gate` arms its characters through the ECS.
///
/// It merges the registry through `ItemDefs::merge_toml`, which is the SAME
/// door the `item.define` node calls, so what the loop photographs is what a
/// level would define.
///
/// # `id@dwell` — the rotation, and why the wheel could not do this (carried 209)
///
/// An entry may carry a **dwell in seconds** (`glock_17@8`). When any entry
/// does, the list becomes a **rotation**: the door equips each id in turn
/// through [`inf_physics::d3::gameplay::equip_weapon`] — the ECS door, the same
/// one the `item.equip` node and the gate use — and wraps for ever.
///
/// It exists because `weapon_switch` is a **rate**: the wheel is a delta source,
/// `axis_snapshot` divides a 120-count notch by the frame time, and the movement
/// step cycles ONE SLOT PER STEP while the sign is non-zero. So one notch of a
/// wheel is not one slot of a bag, and a scripted leg that wanted a *particular*
/// weapon in the hand had to spin and check, spin and check — twenty-four
/// notches, reversing half way, and wave WPN2a still had a session that never
/// reached `remington_870`. A rotation is a *schedule*, so a leg that arrives at
/// any time at all sees every weapon inside one cycle and waits for the one it
/// wants; and because it wraps, it does not care WHEN the leg starts, which no
/// absolute schedule could survive (the ballistics leg begins several minutes
/// into a session, and how many minutes depends on how long the streaming took).
///
/// **It is still a preview-only door.** The rotation is the demo loop's
/// instrument, exactly as `INF_PIE_SPAWN_AT` is; no gate reads it and no level
/// contains it.
pub const ARM_HERO_ENV: &str = "INF_PIE_ARM_HERO";

/// **Preview-only: retune every vehicle in the level** (VEH3a's audit),
/// `name=value;name=value`.
///
/// [`ARM_HERO_ENV`]'s shape at the vehicle. It exists because three of wave
/// VEH3a's four tyre frames did not fire, and one of them could not: a
/// line-lock burnout on ASPHALT under ROAD tyres produces no slip at all in
/// this model, because the island car's brakes out-hold its engine (13 kN
/// against 8). The gate spins one on SAND under SLICK tyres and measures 229
/// slipping steps; the demo loop had no way to say "slicks".
///
/// The values go through `VehicleClass::set` — the by-name door the authored
/// catalogue uses — and are written onto the chassis ENTITY, so the physics
/// bridge installs them on its next sync exactly as it installs an authored
/// class. Nothing bypasses the model.
///
/// Applied **once**, to **every** chassis in the level (an island session has no
/// door to name one car, and the hero boards whichever is nearest), in a
/// **preview** session only. A name the door does not know is a refusal with a
/// reason on stderr, which is the whole point of an operator's switch.
pub const TUNE_VEHICLE_ENV: &str = "INF_PIE_TUNE_VEHICLE";

/// **The demo loop's slow motion** (wave VEH3d): a factor on the WALL time a
/// preview feeds its fixed-step accumulator, in `[0.05, 1]`, default `1`.
///
/// The steps are the same steps — same `dt`, same inputs, same world — there
/// are just fewer of them per wall second, so nothing the simulation does
/// depends on it. It exists because a boarding's beats are shorter than the
/// loop's screenshot: the hand holds the handle for 0.1 s, the seat warp is
/// 0.55 s, the log is sampled at 4 Hz and a frame takes about a second to
/// capture, and two sessions at full speed photographed the moment AFTER
/// three of the beats the wave is about. Read only in a `--pie` preview, for
/// [`SPAWN_AT_ENV`]'s reason.
pub const TIME_SCALE_ENV: &str = "INF_PIE_TIME_SCALE";

/// [`TIME_SCALE_ENV`], read and clamped; `1.0` when absent or unreadable.
pub fn time_scale_from_env() -> f64 {
    std::env::var(TIME_SCALE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.05, 1.0))
        .unwrap_or(1.0)
}

/// **How often [`TUNE_VEHICLE_ENV`] re-applies itself**, seconds (VEH3c audit).
///
/// Twenty is a measurement of what it has to outrun rather than a round number:
/// `traffic::TRAFFIC_FULL_M` is crossed by a car the hero is driving toward in a
/// few seconds, and a tier crossing despawns and respawns the rig at its
/// **authored** class. Twenty seconds is short enough that a car the hero walks
/// up to is tuned before it is boarded and long enough that the walk — every
/// entity in the level, filtered through `rig_of` — is a thousandth of the
/// session's own budget.
const TUNE_PERIOD_S: f64 = 20.0;

/// **The one name on [`TUNE_VEHICLE_ENV`] that is not a tunable** (VEH3c audit).
///
/// `doors_open=1` swings every hinged part of every chassis onto its motor and
/// `doors_open=0` shuts them again. It is a DIRECTIVE: it is filtered out of the
/// name/value pairs before `VehicleClass::set` and `Vehicle::tune` see them, so
/// it is never reported refused, and it reaches
/// [`RuntimeSim::open_vehicle_doors`](crate::runtime_sim::RuntimeSim::open_vehicle_doors)
/// rather than either tuner.
const DOORS_OPEN_NAME: &str = "doors_open";

/// **The second directive** (wave VEH3d): `occupy=1` seats an NPC driver and
/// passenger in the standing car nearest the hero (within
/// [`OCCUPY_REACH_M`]), once, through
/// [`RuntimeSim::occupy_nearest_car`](crate::runtime_sim::RuntimeSim::occupy_nearest_car)
/// — so the demo loop can walk up to an occupied car and CARJACK it. Filtered
/// out of the pairs for `doors_open`'s reason.
const OCCUPY_NAME: &str = "occupy";

/// How far from the hero the car [`OCCUPY_NAME`] seats people in may be,
/// metres.
const OCCUPY_REACH_M: f64 = 25.0;

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
    /// The registry ids [`ARM_HERO_ENV`] named, with each one's **dwell** in
    /// seconds when it named one, and whether they have been given.
    weapons: Vec<(String, Option<f64>)>,
    weapon_done: bool,
    /// The `name=value` pairs [`TUNE_VEHICLE_ENV`] named, and whether they have
    /// been installed **at least once**.
    tune: Vec<(String, f64)>,
    tune_done: bool,
    /// Whether the `occupy` directive has seated anybody yet (wave VEH3d).
    occupied: bool,
    /// **When the tuning is next re-applied**, seconds on this door's own clock
    /// (VEH3c audit).
    ///
    /// # Once was not enough, and the number that says so
    ///
    /// This door used to fire exactly once and set `tune_done`. Every island
    /// chassis is created on the first sync, so that reached the twenty-three
    /// that were resident then — and **not the car the hero actually boards**,
    /// because a parked or traffic car crosses a tier boundary by being
    /// despawned and respawned through `rig_nodes`, which re-mints its
    /// **authored** `VehicleClass` and wipes the tuning with it.
    ///
    /// Measured: a session that asked for `panel_health_j=2000` (a 8 000 J hull
    /// against the default 36 000) on twenty-three chassis, drove nine hundred
    /// rows and finished at **77.9 % of hull** — which is 7 956 J of a 36 000 J
    /// hull, i.e. the DEFAULT, and within a tenth of a percent of the untuned
    /// session before it. Three of wave VEH3c's five frames could not fire
    /// because of it.
    ///
    /// So it re-applies on a cadence. It is idempotent — `VehicleClass::set`
    /// and `Vehicle::tune` both write a value rather than accumulate one — and
    /// it is preview-only, like everything else on this door.
    tune_next_s: f64,
    /// How many chassis the last re-tune found, so a pass that finds the same
    /// number stays quiet in the log.
    tune_seen: usize,
    /// Which id the rotation is holding, and when it hands over. Both are `0`
    /// on a list that named no dwell, which never rotates.
    equip_at: usize,
    equip_next_s: f64,
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
        // **The vehicle tuning** (VEH3a's audit), on the weapon list's own
        // shape: a malformed entry is a refusal with a reason on stderr.
        let mut tune: Vec<(String, f64)> = Vec::new();
        if let Ok(v) = std::env::var(TUNE_VEHICLE_ENV) {
            for entry in v.split(';').map(str::trim).filter(|e| !e.is_empty()) {
                match entry.split_once('=') {
                    Some((name, value)) => match value.trim().parse::<f64>() {
                        Ok(x) if x.is_finite() => tune.push((name.trim().to_string(), x)),
                        _ => eprintln!(
                            "inf-player: {TUNE_VEHICLE_ENV} entry `{entry}` has no finite value"
                        ),
                    },
                    None => eprintln!(
                        "inf-player: {TUNE_VEHICLE_ENV} entry `{entry}` is not `name=value`"
                    ),
                }
            }
        }
        // The same read the editor's payload builder does, through the same
        // door: a garment worn here that the payload did not carry resolves to
        // nothing and draws nothing, which is what carried 142 was.
        let cloth = inf_runtime::pie::preview_worn_cloth();
        // A malformed value is a refusal with a reason on stderr, exactly as the
        // placement's is: the whole point of the door is that the operator finds
        // out whether it took.
        let mut weapons: Vec<(String, Option<f64>)> = Vec::new();
        if let Ok(v) = std::env::var(ARM_HERO_ENV) {
            let mut defs = inf_ecs::item::ItemDefs::default();
            match defs.merge_toml(inf_ecs::weapon::WEAPON_REGISTRY_TOML) {
                Ok(_) => {
                    for entry in v.split(';') {
                        // `id` or `id@dwell_seconds` — the placement's own
                        // `x,y,z@s` shape, one door along.
                        let (name, dwell) = match entry.split_once('@') {
                            Some((n, d)) => match d.trim().parse::<f64>() {
                                Ok(d) if d.is_finite() && d > 0.0 => (n, Some(d)),
                                _ => {
                                    eprintln!(
                                        "inf-player: {ARM_HERO_ENV} entry `{entry}` has no readable dwell; treating it as `{n}`"
                                    );
                                    (n, None)
                                }
                            },
                            None => (entry, None),
                        };
                        let id = inf_ecs::item::canonical_id(name);
                        if id.is_empty() {
                            continue;
                        }
                        if defs.get(&id).is_some_and(|d| d.weapon.is_some()) {
                            weapons.push((id, dwell));
                        } else {
                            eprintln!(
                                "inf-player: {ARM_HERO_ENV} entry `{entry}` is not a weapon in the registry"
                            );
                        }
                    }
                }
                Err(e) => eprintln!("inf-player: the weapon registry does not parse: {e}"),
            }
        }
        Self {
            at,
            cloth,
            weapons,
            tune,
            ..Self::default()
        }
    }

    /// The placements this override still has to make.
    pub fn pending(&self) -> usize {
        self.at.len().saturating_sub(self.next)
    }

    /// Whether [`ARM_HERO_ENV`]'s list named a dwell, and therefore rotates.
    fn rotates(&self) -> bool {
        self.weapons.len() > 1 && self.weapons.iter().any(|(_, d)| d.is_some())
    }

    /// How long entry `i` is held, seconds — its own dwell, or the nearest one
    /// named before it, or the list's first. A list that names one dwell means
    /// that dwell for all of them, which is what an operator writing
    /// `a@8;b;c` plainly means.
    fn dwell(&self, i: usize) -> f64 {
        self.weapons
            .iter()
            .take(i + 1)
            .rev()
            .find_map(|(_, d)| *d)
            .or_else(|| self.weapons.iter().find_map(|(_, d)| *d))
            .unwrap_or(0.0)
            .clamp(0.0, 600.0)
    }

    /// Apply whichever placement is due, at most one per frame. Returns a line
    /// for the hero log when one fires, so a frame taken afterwards can be read
    /// against a record of where the hero was put.
    pub fn tick(&mut self, sim: &mut RuntimeSim, dt: f64) -> Option<String> {
        let done = self.next >= self.at.len();
        if done
            && (self.cloth.is_none() || self.cloth_done)
            && (self.weapons.is_empty() || self.weapon_done)
            && self.tune.is_empty()
            && !self.rotates()
        {
            return None;
        }
        self.accum += dt;
        if self.accum < SPAWN_DELAY_S {
            return None;
        }
        let due = (!done && self.accum >= self.at[self.next].1).then(|| self.at[self.next]);
        let wear = self.cloth.filter(|_| !self.cloth_done);
        let arm = (!self.weapon_done && !self.weapons.is_empty()).then(|| self.weapons.clone());
        // **A CADENCE, not a one-shot** (VEH3c audit) — see `tune_next_s` for the
        // 77.9 %-of-a-36-000-J-hull measurement that decided it. The first pass
        // still runs at `SPAWN_DELAY_S`, and every `TUNE_PERIOD_S` after that
        // catches whatever the streamer, the traffic pass or a tier boundary has
        // minted since.
        let retune =
            (!self.tune.is_empty() && self.accum >= self.tune_next_s).then(|| self.tune.clone());
        // The rotation's own clock (carried 209). It is checked BEFORE the
        // early return, because a rotation is the only thing this door does
        // that is not one-shot.
        let rotate = self.rotates() && self.weapon_done && self.accum >= self.equip_next_s;
        if due.is_none() && wear.is_none() && arm.is_none() && retune.is_none() && !rotate {
            return None;
        }
        let hero = inf_ecs::movement::camera_subject(sim.world())?;
        // **The CAMERA turns with the character** (COV1 audit, carried 186).
        //
        // Without this the heading buys almost nothing: `intent_move` is in the
        // AIM frame and the aim yaw is written from the camera every step, so a
        // scripted `W` walks where the CAMERA points and `try_cover` probes
        // there too. Setting the body alone leaves the two disagreeing for as
        // long as the leg holds a key.
        if let Some((_, _, Some(y))) = due {
            let cam = sim.camera_mut();
            cam.yaw_deg = y;
            cam.pose.yaw_deg = y;
            cam.seeded = false;
            cam.arm_seeded = false;
        }
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
        // **The weapon** (wave WPN2a). After the placement and the garment,
        // because it goes through the world's own catalogue and inventory doors
        // rather than through a component write, and both of those want the hero
        // to be where it is going to stand.
        if let Some(ids) = arm {
            let w = sim.world_mut();
            let taken = inf_ecs::item::item_defs_mut(w)
                .merge_toml(inf_ecs::weapon::WEAPON_REGISTRY_TOML)
                .unwrap_or(0);
            // A bag big enough for the whole list, so the wheel has somewhere to
            // cycle between: `cycle_equipped` walks SLOTS, and a bag with fewer
            // slots than weapons would silently drop the tail of the list.
            if inf_ecs::item::inventory_of(w, hero).is_none() {
                inf_ecs::item::give_inventory(w, hero, ids.len().max(6));
            }
            let mut given: Vec<&str> = Vec::new();
            for (id, _) in &ids {
                if inf_ecs::item::give(w, hero, id, 1) == 0 {
                    given.push(id.as_str());
                }
            }
            let equipped = ids
                .first()
                .is_some_and(|(id, _)| inf_physics::d3::gameplay::equip_weapon(w, hero, id));
            self.weapon_done = true;
            self.equip_at = 0;
            self.equip_next_s = self.accum + self.dwell(0);
            if !said.is_empty() {
                said.push_str("; ");
            }
            said.push_str(&format!(
                "{ARM_HERO_ENV} merged {taken} registry row(s) and gave the hero {given:?} (equipped the first: {equipped})"
            ));
            if self.rotates() {
                said.push_str(&format!(
                    "; rotating every {:.1} s through {} weapon(s) via `equip_weapon`",
                    self.dwell(0),
                    ids.len()
                ));
            }
        }
        // **THE ROTATION** (carried 209). One `equip_weapon` — the ECS door —
        // per dwell, wrapping for ever, so a scripted leg that arrives at any
        // time sees every weapon inside one cycle and never touches the wheel.
        if rotate {
            let n = self.weapons.len();
            self.equip_at = (self.equip_at + 1) % n;
            let (id, _) = self.weapons[self.equip_at].clone();
            let ok = inf_physics::d3::gameplay::equip_weapon(sim.world_mut(), hero, &id);
            self.equip_next_s = self.accum + self.dwell(self.equip_at);
            if !said.is_empty() {
                said.push_str("; ");
            }
            said.push_str(&format!(
                "{ARM_HERO_ENV} rotation equipped `{id}` ({}/{n}, ok {ok}) at t={:.1}s",
                self.equip_at + 1,
                self.accum
            ));
        }
        // **THE VEHICLE TUNING** (VEH3a's audit). Last, because it walks the
        // level rather than the hero, and because a car retuned before the
        // placement would be retuned on a chassis the streamer has not paged in.
        if let Some(pairs) = retune {
            // **`doors_open` IS A DIRECTIVE, NOT A TUNABLE** (VEH3c audit). It
            // swings every hinged part of every chassis onto its motor, which is
            // the only way a *running* session can show a door on its hinge: the
            // shipped input map has no key for one (VEH3d owns that), so without
            // this the joints the wave built could be measured and never seen.
            // Filtered out of `pairs` before the tunables run, or every chassis
            // would report it refused.
            let doors: Option<bool> = pairs
                .iter()
                .find(|(n, _)| n == DOORS_OPEN_NAME)
                .map(|(_, v)| *v != 0.0);
            let occupy = pairs.iter().any(|(n, v)| n == OCCUPY_NAME && *v != 0.0);
            let pairs: Vec<(String, f64)> = pairs
                .into_iter()
                .filter(|(n, _)| n != DOORS_OPEN_NAME && n != OCCUPY_NAME)
                .collect();
            // **`occupy` is the second directive** (wave VEH3d): seat an NPC
            // driver and passenger in the standing car nearest the hero, ONCE.
            // Before the tunables, because a car seated here is a car the
            // tuner should find the same as any other.
            if occupy && !self.occupied {
                let near = inf_ecs::movement::camera_subject(sim.world())
                    .and_then(|h| sim.world().entity_of(h))
                    .and_then(|e| {
                        sim.world()
                            .world()
                            .get::<inf_ecs::components::Transform>(e)
                            .map(|t| t.translation.to_dvec3())
                    });
                if let Some(near) = near {
                    if let Some(chassis) = sim.occupy_nearest_car(near, OCCUPY_REACH_M) {
                        self.occupied = true;
                        if !said.is_empty() {
                            said.push_str("; ");
                        }
                        said.push_str(&format!(
                            "{TUNE_VEHICLE_ENV} occupy seated a driver and a passenger in {chassis} at t={:.1}s",
                            self.accum
                        ));
                    }
                }
            }
            let w = sim.world_mut();
            // Every chassis in the level, through the RECOGNISER rather than a
            // component query: a vehicle is a rig with wheels, and that is the
            // one definition both hosts already share.
            let chassis: Vec<Uuid> = w
                .world()
                .iter_entities()
                .filter_map(|e| e.get::<inf_ecs::components::Guid>().map(|g| g.0))
                .filter(|g| inf_ecs::vehicle::rig_of(w, *g).is_some_and(|r| !r.wheels.is_empty()))
                .collect();
            let mut took = 0usize;
            let mut refused: Vec<&str> = Vec::new();
            for guid in &chassis {
                let Some(entity) = w.entity_of(*guid) else {
                    continue;
                };
                let mut class = w
                    .world()
                    .get::<inf_ecs::components::VehicleClass>(entity)
                    .copied()
                    .unwrap_or_default();
                for (name, value) in &pairs {
                    if class.set(name, *value) {
                        took += 1;
                    } else if !refused.contains(&name.as_str()) {
                        refused.push(name.as_str());
                    }
                }
                w.world_mut().entity_mut(entity).insert(class);
            }
            // **AND THROUGH THE LIVE TUNER**, which is the half that reaches the
            // PHYSICS (wave VEH3b). `reconcile_vehicles` installs an authored
            // class ONCE, at creation, and says why in its own doc: *"the
            // component is the STARTING point; the tuner owns it from there."*
            // Every island chassis is created on the first sync, long before
            // this door fires, so writing the component alone changed what the
            // Details grid shows and nothing the car does.
            //
            // Measured, and it is what found this: a session that asked for
            // `turbo_boost_max=0.8` on twenty-three chassis drove six hundred
            // rows at **0.000 boost**. This door IS a tuner, so it tunes.
            let mut live = 0usize;
            for guid in &chassis {
                if let Some(v) = sim.bridge3d_mut().vehicle_mut(*guid) {
                    for (name, value) in &pairs {
                        if v.tune(name, *value) {
                            live += 1;
                        }
                    }
                }
            }
            // …and the directive, after the tuning, because a door swung open on
            // a chassis the tier ladder has just re-minted would be swung on the
            // OLD rig. `set_part_open` is idempotent — it writes a target angle
            // rather than adding one — so the cadence re-opening a door that is
            // already open costs a motor re-aim that the bodywork skips.
            let mut swung = 0usize;
            if let Some(open) = doors {
                for guid in &chassis {
                    swung += sim.open_vehicle_doors(*guid, open);
                }
            }
            for name in &refused {
                eprintln!("inf-player: {TUNE_VEHICLE_ENV} name `{name}` is not a tunable");
            }
            let first = !self.tune_done;
            self.tune_done = true;
            self.tune_next_s = self.accum + TUNE_PERIOD_S;
            // **Only the first pass and the passes that FIND something new are
            // said**, because a line every twenty seconds for twenty minutes is
            // a log nobody reads. `took` counts the component writes, which is
            // constant once every chassis is resident; what moves is the chassis
            // COUNT, and a session that re-tunes a car the tier ladder just
            // re-minted is exactly the event worth a line.
            if first || chassis.len() != self.tune_seen {
                self.tune_seen = chassis.len();
                if !said.is_empty() {
                    said.push_str("; ");
                }
                said.push_str(&format!(
                    "{TUNE_VEHICLE_ENV} set {took} tunable(s) on {} chassis and {live} on the RUNNING vehicles at t={:.1}s, swung {swung} part(s) onto their hinges (refused {refused:?})",
                    chassis.len(),
                    self.accum
                ));
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
/// — plus the camera's four (CHAR1c), the cover's three (COV1), the
/// ballistics' three (WPN2a: rounds in flight, how far the last round that hit
/// something had flown, and WHAT IS IN THE HAND), the feel's four (WPN2b) and
/// the sound's two (WPN2c: casings live, and which TAIL the last shot chose) —
/// here four times a second, and the script prints the
/// first and last lines beside its frames. **Columns are only ever APPENDED**,
/// so every index a script already reads keeps its meaning.
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
pub struct HeroLog {
    file: Option<std::fs::File>,
    accum: f64,
    /// The session's queued-command count at the last row (wave VEH3e), so
    /// the `voice_cmds` column is a difference of two readings of the stream.
    audio_seen: u64,
    /// The fixed-step count at the last row, same purpose.
    steps_seen: u64,
}

impl Default for HeroLog {
    fn default() -> Self {
        Self {
            file: None,
            accum: 0.0,
            audio_seen: 0,
            steps_seen: 0,
        }
    }
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
                ..Self::default()
            },
            Err(e) => {
                eprintln!("inf-player: cannot open the hero log at {path}: {e}");
                Self::default()
            }
        }
    }

    /// Whether this log writes anywhere — the switch the input-delivery probe
    /// (wave VEH3f.2a) rides, so a session nobody reads installs no hook.
    pub fn enabled(&self) -> bool {
        self.file.is_some()
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
        // **THE BALLISTICS COLUMNS** (wave WPN2a), APPENDED for the reason the
        // camera's four and the cover's three were: every column index a script
        // already reads keeps its meaning. `rounds` is how many bodies are in
        // the air RIGHT NOW — which is what a frame of a tracer has to be
        // triggered on, because a round crosses a hundred metres in a tenth of a
        // second — and `last_hit_m` is how far the last one that hit something
        // had flown, latched, so a frame of an impact has a distance beside it.
        // Both read the pool; both are 0 on a level that has never fired.
        let pool = inf_ecs::ballistics::round_pool(sim.world());
        let rounds_live = pool.map(|p| p.rounds.len()).unwrap_or(0);
        let last_hit_m = pool.map(|p| p.last_flight_m).unwrap_or(0.0);
        // **THE BRASS** (wave WPN2c) — how many casings exist right now, which
        // is what a frame of brass on the ground has to be triggered on: they
        // fall in under a second and live for eight, so a leg that fired and
        // then looked would photograph an empty floor if it read anything else.
        // Zero on a level that has never fired.
        let casings_live = inf_ecs::casing::casings_live(sim.world());
        // **WHICH TAIL THE LAST SHOT CHOSE**, read off the pool the sim latched
        // it on. The first draft latched it here, off `GameplayReport::shots`,
        // and the column read `-` through an eight-round burst on the island —
        // see `inf_ecs::casing::CasingPool::last_shot_indoors` for why a
        // host-side latch cannot see every step.
        let tail = match inf_ecs::casing::last_shot_indoors(sim.world()) {
            Some(true) => "indoor",
            Some(false) => "outdoor",
            None => "-",
        };
        // **WHAT IS ACTUALLY IN THE HAND** (wave WPN2a), and it is here because
        // a frame was captioned wrongly without it: the demo loop cycles weapons
        // with the scroll wheel and named each frame after the id it MEANT to
        // equip, while the HUD in the pixels showed a different magazine. One
        // notch of a wheel is not one slot of a bag. `-` when nothing is
        // equipped, which is every session before this wave.
        let equipped = guid
            .and_then(|g| inf_ecs::weapon::equipped_def(sim.world(), g))
            .map(|(id, _)| id)
            .unwrap_or_else(|| "-".to_string());
        // **THE FEEL COLUMNS** (wave WPN2b), appended for the ballistics four's
        // reason verbatim. Every one of them is a number a FRAME has to be
        // triggered on, and none of them is derivable from the columns in front:
        //
        // * `recoil_mm` -- how far the hold-point spring has the weapon off the
        //   aim line RIGHT NOW, millimetres. A shot's climb lasts four steps and
        //   66 ms, so a frame of a burst cannot be taken on a wall clock.
        // * `aim_recoil_deg` -- how far the AIM has been pushed off where the
        //   player is pointing, degrees, positive up. That is the layer the
        //   reticle follows, and it is what the burst table is made of.
        // * `spread_deg` -- the whole cone the NEXT round would leave through,
        //   after the bloom, the stance, the movement and the aim. What a
        //   pattern frame is triggered on.
        // * `ads` -- the aim-down-sights blend, `[0, 1]`. The ADS frames are
        //   triggered on it and on the boom in column 13.
        //
        // All four are 0 for a character with no weapon, which is every session
        // before this wave.
        let feel = guid.and_then(|g| inf_ecs::feel::feel_of(sim.world(), g));
        let recoil_mm = feel.map(|f| f.vm.position.length() * 1000.0).unwrap_or(0.0);
        let aim_recoil_deg = feel.map(|f| f.applied_pitch_deg).unwrap_or(0.0);
        let ads = feel.map(|f| f.ads_blend).unwrap_or(0.0);
        let spread_deg = guid
            .and_then(|g| {
                let (_, def) = inf_ecs::weapon::equipped_def(sim.world(), g)?;
                let e = sim.world().entity_of(g)?;
                let bloom = sim
                    .world()
                    .world()
                    .get::<inf_ecs::weapon::WeaponState>(e)
                    .map(|s| s.spread_bloom_deg)
                    .unwrap_or(0.0);
                let cm = sim
                    .world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)?;
                let stance = match cm.mode {
                    inf_ecs::components::MovementMode::Crouch => {
                        inf_ecs::feel::ShotStance::Crouched
                    }
                    inf_ecs::components::MovementMode::Prone => inf_ecs::feel::ShotStance::Prone,
                    _ => inf_ecs::feel::ShotStance::Standing,
                };
                let v = cm.runtime.velocity.to_dvec3();
                let speed = (v.x * v.x + v.z * v.z).sqrt();
                Some(inf_ecs::feel::resolved_cone_deg(
                    &def, bloom, stance, speed, ads,
                ))
            })
            .unwrap_or(0.0);
        // **THE CLASS COLUMNS** (wave WPN2d), APPENDED for the reason every
        // block above them was: every column index a script already reads keeps
        // its meaning. None of the three is derivable from the columns in front:
        //
        // * `class` — what KIND of gun is in the hand (`ar`, `shotgun`,
        //   `launcher`, …). Column 23 carries the equipped ID, which is a row
        //   name; a frame captioned "the shotgun" wants the class, and a leg
        //   that rotates through one weapon per class has to be able to WAIT
        //   for one.
        // * `attach` — how many attachments are bolted on, and the fold's own
        //   loudness multiplier, as `n@x.xx`. Two numbers because they answer
        //   the two questions a frame of an attachment asks: is anything fitted,
        //   and did it CHANGE anything. `0@1.00` is a bare rail.
        // * `lock` — how far into a lock-on the launcher is, `[0, 1]`, and
        //   whether it has completed (`+`). What a frame of the lock indicator
        //   is triggered on, because a lock takes 1.2–1.6 s to acquire and
        //   releases the instant the cone loses it.
        //
        // All three are inert for a character with no weapon, which is every
        // session before this wave.
        let class = guid
            .and_then(|g| inf_ecs::weapon::equipped_def(sim.world(), g))
            .map(|(_, d)| d.audio_class().name().to_string())
            .unwrap_or_else(|| "-".to_string());
        let attach = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::weapon::WeaponState>(e)
                    .map(|s| {
                        let n = s
                            .attach
                            .iter()
                            .filter(|i| **i != inf_ecs::attachment::NO_ATTACHMENT)
                            .count();
                        let fold = inf_ecs::attachment::fold(&s.attach);
                        format!("{n}@{:.2}", fold.loudness_mult)
                    })
            })
            .unwrap_or_else(|| "-".to_string());
        let lock = guid
            .and_then(|g| {
                let (_, def) = inf_ecs::weapon::equipped_def(sim.world(), g)?;
                let e = sim.world().entity_of(g)?;
                let st = sim.world().world().get::<inf_ecs::weapon::WeaponState>(e)?;
                Some(format!(
                    "{:.2}{}",
                    st.lock_fraction(&def),
                    if st.locked_on(&def).is_some() {
                        "+"
                    } else {
                        ""
                    }
                ))
            })
            .unwrap_or_else(|| "-".to_string());
        // **THE SHOOTOUT COLUMNS** (wave WPN2e), APPENDED for the reason every
        // block above them was: every column index a script already reads keeps
        // its meaning. Neither is derivable from the columns in front, and both
        // read the WORLD rather than a report:
        //
        // * `engaged` — how many responding units are pointing a weapon at
        //   somebody **right now** (`inf_ecs::engage::engaged_units`, which
        //   counts slots with a live target). A frame of an officer aiming has
        //   to be triggered on this: the police arrive over tens of seconds and
        //   the aim itself is a ray-gated decision that can go away between two
        //   screenshots.
        // * `incoming` — rounds in the air that the hero did **not** fire. The
        //   pool carries a `shooter` per round, so this is "somebody is shooting
        //   at me" as a number, which is what a frame of the hero under fire is
        //   waited on. Zero on every level nobody has fired at the hero on.
        //
        // Both are zero for a session in which nobody is wanted, which is every
        // session before this wave.
        let engaged = inf_ecs::engage::engaged_units(sim.world());
        // **AND THE WANTED SYSTEM ITSELF** (wave WPN2e audit), appended on the
        // same rule. Two waves failed to diagnose the shootout leg because the
        // loop's only instrument stopped at `engaged`: a run that photographs
        // nothing cannot tell "nobody heard the shot", "the file went cold",
        // "the car never arrived" and "the officer arrived and could not see
        // you" apart, and those are four different bugs in four different
        // crates.
        //
        // * `heat` — the hero's own `CrimeRes` heat, which is what
        //   `Response::for_heat` reads and therefore the whole rung ladder;
        // * `on_scene` — units whose `UnitRun::state` is `OnScene`, which is the
        //   link between "a car was sent" and "an officer is standing here".
        //
        // Both are zero for every session before this wave and cost one absent
        // resource read on a level with no crime.
        let heat = guid
            .map(|g| inf_ecs::crime::heat_of(sim.world(), g))
            .unwrap_or(0);
        // * `responder_m` — how far the NEAREST responding crew is from the hero.
        //   `engaged 0` with a unit on scene is two more different bugs: an
        //   officer that cannot SEE you and an officer that is not NEAR you.
        //   `inf_ecs::engage::ENGAGE_RANGE_M` is 35 m and a unit stops within
        //   `ON_SCENE_M` of an incident **or where its road runs out**, which on
        //   a street can be a long way further.
        let responder_m = guid
            .and_then(|g| {
                let here = sim
                    .world()
                    .entity_of(g)
                    .and_then(|e| sim.world().world().get::<inf_ecs::components::Transform>(e))
                    .map(|t| t.translation.to_dvec3())?;
                inf_ecs::dispatch::responders(sim.world())
                    .into_iter()
                    .filter_map(|crew| {
                        let at = sim
                            .world()
                            .entity_of(crew)
                            .and_then(|e| {
                                sim.world().world().get::<inf_ecs::components::Transform>(e)
                            })
                            .map(|t| t.translation.to_dvec3())?;
                        let d = (at - here).length();
                        d.is_finite().then_some(d)
                    })
                    .fold(None::<f64>, |best, d| Some(best.map_or(d, |b| b.min(d))))
            })
            .unwrap_or(-1.0);
        let on_scene = inf_ecs::dispatch::dispatch_of(sim.world())
            .map(|r| {
                r.runs
                    .values()
                    .filter(|u| u.state == inf_ecs::dispatch::UnitState::OnScene)
                    .count()
            })
            .unwrap_or(0);
        let incoming = guid
            .and_then(|g| {
                inf_ecs::ballistics::round_pool(sim.world())
                    .map(|p| p.rounds.iter().filter(|r| r.shooter != g).count())
            })
            .unwrap_or(0);
        // **THE TYRE COLUMNS** (wave VEH3a), APPENDED for the reason the camera's
        // four, the cover's three and the ballistics' two were: every column
        // index a script already reads keeps its meaning. Four temperatures, the
        // surface the most wheels are on, the driven axle's two slips and the µ
        // the contact is worth — the numbers a frame of a burnout, a verge or a
        // kerb has to be TRIGGERED on, because none of them is visible in a
        // position column. All eight read `0` / `-` on a level with nobody
        // driving, which is what a predicate over them reads as such.
        let tyres: Vec<inf_ecs::vehicle::WheelState> = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
                    .map(|cm| cm.runtime.seat)
            })
            .filter(|seat| seat.is_seated())
            .and_then(|seat| sim.bridge3d().vehicle_of(seat.vehicle))
            .map(|v| v.wheels().to_vec())
            .unwrap_or_default();
        let temp_at = |i: usize| tyres.get(i).map(|w| w.temp_c).unwrap_or(0.0);
        // **Through Ring 0's own census** (`audit:` VEH3b), which is where the
        // "no wheel is on the ground" answer lives: this block used to run its
        // own copy of the tie-break and count wheels whose raycast MISSED, so a
        // beached car logged `asphalt` and a µ left over from its last contact
        // -- the three columns the VEH3a audit read a stuck island car off.
        let (tyre_surface, tyre_mu) = if tyres.is_empty() {
            ("-".to_string(), 0.0)
        } else {
            let (s, mu) = inf_ecs::vehicle::surface_census(&tyres);
            (s.to_string(), mu)
        };
        let driven = tyres.last().copied().unwrap_or_default();
        // **THE DRIVETRAIN COLUMNS** (wave VEH3b), APPENDED for the reason the
        // eight tyre columns before them were: a frame of a launch flare, a
        // downshift blip or a limiter bounce cannot be triggered on a position
        // column, and "the clutch is slipping", "the turbo is spooling" and "the
        // limiter is cutting" are three different things that all read the same
        // in `speed`. All four read `0` on a level with nobody driving.
        let drivetrain = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
                    .map(|cm| cm.runtime.seat)
            })
            .filter(|seat| seat.is_seated())
            .and_then(|seat| sim.bridge3d().vehicle_of(seat.vehicle))
            .and_then(|v| v.drivetrain())
            .unwrap_or_default();
        // **THE BODYWORK COLUMNS** (wave VEH3c), APPENDED for the drivetrain
        // columns' reason verbatim: a frame of a shed bumper, a shattered pane,
        // a flat tyre or a burning car cannot be triggered on a position column,
        // and every one of those is a different thing that reads the same in
        // `speed`. All five read `0` on a level where nobody is driving.
        let (car_health, engine_scale, flats, panes_broken, parts_shed) = {
            let seated = guid
                .and_then(|g| sim.world().entity_of(g))
                .and_then(|e| {
                    sim.world()
                        .world()
                        .get::<inf_ecs::components::CharacterMovement>(e)
                        .map(|cm| cm.runtime.seat)
                })
                .filter(|seat| seat.is_seated())
                .map(|seat| seat.vehicle);
            match seated {
                Some(v) => {
                    let world = sim.world();
                    let limits = inf_ecs::bodywork::DamageLimits::of(world, v);
                    let d = inf_ecs::bodywork::damage_of(world)
                        .and_then(|r| r.rows.get(&v))
                        .cloned()
                        .unwrap_or_default();
                    (
                        100.0 * d.hull_frac(limits.hull_capacity_j()),
                        100.0 * d.engine_scale(),
                        d.flat_count(),
                        d.panes_broken(),
                        d.parts_shed(),
                    )
                }
                None => (0.0, 0.0, 0, 0, 0),
            }
        };
        // **THE BOARDING COLUMNS** (wave VEH3d), APPENDED for the reason every
        // block before them was: a frame of a hand on a door handle, a door on
        // its hinge, hands on a rim through a lock or feet on the pedals cannot
        // be triggered on a position column, and "walking up to a car",
        // "opening its door" and "sitting in it" are three things that all read
        // `Grounded` or `Driving` in the mode column.
        //
        // * `board` — the machine's phase (`-` when nobody is boarding and
        //   nobody is at a wheel; `driving` at the wheel), with a `!` on the
        //   end while it is a CARJACK;
        // * `seat` — which seat (`driver`, `passenger`, `rear`, or `-`);
        // * `hand_m` — the hand residual on a DOOR (outer handle, inner pull),
        //   metres, off the hand pass's own report, while a hand is HOLDING
        //   one at full weight; `-1` otherwise, so a trigger can tell "on the
        //   handle" from "nowhere near it";
        // * `hinge_deg` — the door's hinge angle, read off VEH3c's joint;
        // * `wheel_m` — the hand residual on the RIM while driving at full
        //   weight, `-1` otherwise;
        // * `pedal_m` — the worst foot residual on its pedal (or floor) goal
        //   while seated, `-1` otherwise;
        // * `rim_deg` — how far the steering wheel has turned, degrees of RIM
        //   (450 at full lock), the angle the hands' grips turn by.
        //
        // All seven read `-` / `-1` / `0` for a session in which nobody boards,
        // which is every session before this wave.
        let (board_phase, board_seat, hand_m, hinge_deg, wheel_m, pedal_m, rim_deg) = guid
            .and_then(|g| sim.world().entity_of(g).map(|e| (g, e)))
            .and_then(|(g, e)| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
                    .map(|cm| (cm.runtime.boarding, cm.runtime.seat, g))
            })
            .map(|(b, seat, g)| {
                // The POSED joints against the live sockets (VEH3d audit) —
                // never the IK solver's `reach_error` on its own target, which
                // these columns carried and which read 0.0 whatever the drawn
                // arm was doing.
                let res = sim.boarding_residuals(g);
                use inf_ecs::boarding::BoardPhase;
                let holding = b.hand_weight >= 0.999;
                let phase = if b.phase == BoardPhase::Idle && seat.is_seated() {
                    BoardPhase::Driving
                } else {
                    b.phase
                };
                let seat_name = if phase == BoardPhase::Idle {
                    "-"
                } else {
                    inf_ecs::boarding::SeatIndex::from_u8(if seat.is_seated() {
                        seat.seat
                    } else {
                        b.seat
                    })
                    .name()
                };
                let on_door = matches!(
                    phase,
                    BoardPhase::OpeningDoor
                        | BoardPhase::Seated
                        | BoardPhase::Exiting
                        | BoardPhase::ClosingDoor
                );
                let rim = if seat.is_driving() {
                    inf_physics::d3::boarding::vehicle_steer(
                        sim.world(),
                        sim.bridge3d(),
                        seat.vehicle,
                    )
                } else {
                    0.0
                };
                (
                    if b.carjack && phase != BoardPhase::Idle {
                        format!("{}!", phase.name())
                    } else {
                        phase.name().to_string()
                    },
                    seat_name,
                    res.and_then(|r| r.handle_m)
                        .filter(|_| on_door && b.handle_weight >= 0.999)
                        .unwrap_or(-1.0),
                    b.door_deg,
                    res.and_then(|r| r.grips_m)
                        .filter(|_| phase == BoardPhase::Driving && holding)
                        .unwrap_or(-1.0),
                    res.and_then(|r| r.feet_m)
                        .filter(|_| matches!(phase, BoardPhase::Driving | BoardPhase::Seated))
                        .unwrap_or(-1.0),
                    rim,
                )
            })
            .unwrap_or(("-".to_string(), "-", -1.0, 0.0, -1.0, -1.0, 0.0));
        // **THE AUDIO COLUMNS** (wave VEH3e), APPENDED for the reason every
        // block before them was: a frame of a shift, a burnout or a kerb has to
        // be triggered on a column that can see it. The telemetry four (gear,
        // load, |slip| per axle and the surface the rear axle's squeal is
        // chosen by) are what the vehicle door PUBLISHED; the other five are
        // what the AUDIO ENGINE HOLDS — the loudest grain's pitch, the whine's
        // pitch, the two squeals' volumes, read back off its voices with
        // `voice_params` — and how many commands the car's keys queued a step
        // since the last row, read off the stream — and `thumps`, the surface
        // impulses (a kerb, a landing) the stream PLAYED since the last row,
        // which is what a frame of a kerb is triggered on. None of these is
        // what the planner meant to send.
        let queued = sim.audio_commands_queued();
        let fresh = queued.saturating_sub(self.audio_seen) as usize;
        let stepped = sim.steps().saturating_sub(self.steps_seen).max(1);
        self.audio_seen = queued;
        self.steps_seen = sim.steps();
        let voice = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
                    .map(|cm| cm.runtime.seat)
            })
            .filter(|seat| seat.is_seated())
            .and_then(|seat| {
                sim.vehicles()
                    .iter()
                    .find(|o| o.chassis == seat.vehicle)
                    .and_then(|o| o.voice.map(|t| (seat.vehicle, t)))
            });
        let (
            v_gear,
            v_load,
            v_slip_f,
            v_slip_r,
            v_surface,
            v_grain,
            v_whine,
            v_sq_f,
            v_sq_r,
            v_cmds,
            v_thumps,
        ) = match voice {
            Some((car, t)) => {
                use inf_ecs::vehicle_audio::{entity_key, voice_key, SurfaceVoice, VoiceLayer};
                let key = entity_key(car);
                let params = |l| sim.voice_params(voice_key(key, l));
                let grain = [
                    VoiceLayer::GrainIdle,
                    VoiceLayer::GrainMid,
                    VoiceLayer::GrainFull,
                ]
                .into_iter()
                .filter_map(params)
                .fold(
                    (0.0f64, 0.0f64),
                    |a, (v, p)| if v > a.0 { (v, p) } else { a },
                )
                .1;
                let keys: Vec<u64> = VoiceLayer::ALL.iter().map(|l| voice_key(key, *l)).collect();
                let log = sim.audio_command_log();
                let tail = &log[log.len().saturating_sub(fresh)..];
                let n = tail
                    .iter()
                    .filter(|c| c.source().is_some_and(|s| keys.contains(&s)))
                    .count();
                let impulses = [
                    voice_key(key, VoiceLayer::ImpulseFront),
                    voice_key(key, VoiceLayer::ImpulseRear),
                ];
                let thumps = tail
                        .iter()
                        .filter(|c| matches!(c, inf_audio::AudioCommand::Play(p) if impulses.contains(&p.source)))
                        .count();
                (
                    t.gear,
                    t.load(),
                    t.axles[0].slip,
                    t.axles[1].slip,
                    SurfaceVoice::of(t.axles[1].surface).name(),
                    grain,
                    params(VoiceLayer::Whine).map(|x| x.1).unwrap_or(0.0),
                    params(VoiceLayer::SquealFront).map(|x| x.0).unwrap_or(0.0),
                    params(VoiceLayer::SquealRear).map(|x| x.0).unwrap_or(0.0),
                    n as f64 / stepped as f64,
                    thumps,
                )
            }
            None => (0, 0.0, 0.0, 0.0, "-", 0.0, 0.0, 0.0, 0.0, 0.0, 0),
        };
        // **THE ROSTER COLUMNS** (wave VEH3f): which class and which lore row
        // the car the hero sits in is, what its body is drawn with, the hitch
        // angle when a trailer rides its fifth wheel, and the yaw rate when it
        // steers by skid -- read off the WORLD (`inf_ecs::roster::row_of`,
        // `body_kind`, `hitch_angle_deg`) and the chassis's own rapier body,
        // never off a table the driver chose from.
        let seated_car = guid
            .and_then(|g| sim.world().entity_of(g))
            .and_then(|e| {
                sim.world()
                    .world()
                    .get::<inf_ecs::components::CharacterMovement>(e)
                    .map(|cm| cm.runtime.seat)
            })
            .filter(|seat| seat.is_seated())
            .map(|seat| seat.vehicle);
        let (r_class, r_row, r_body, r_hitch, r_track) = match seated_car {
            Some(car) => {
                let row = inf_ecs::roster::row_of(sim.world(), car);
                let class = row
                    .and_then(|id| inf_ecs::roster::roster().get(id))
                    .and_then(|d| d.roster_class)
                    .map(|c| c.name())
                    .unwrap_or("-");
                let skid = row
                    .and_then(|id| inf_ecs::roster::roster().get(id))
                    .is_some_and(|d| {
                        d.class.max_steer_deg.is_nan() || d.class.max_steer_deg <= 0.0
                    });
                let yaw = if skid {
                    let b = sim.bridge3d();
                    b.body_of(car)
                        .and_then(|body| b.world().body_angvel(body))
                        .map(|w| w.y)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                (
                    class,
                    row.unwrap_or("-"),
                    inf_ecs::roster::body_kind(sim.world(), car),
                    inf_ecs::roster::hitch_angle_deg(sim.world(), car),
                    yaw,
                )
            }
            None => ("-", "-", "-", None, 0.0),
        };
        // **THE CRAFT COLUMNS** (wave VEH3g): the altitude of the craft the hero
        // is in (its chassis's world height), the wing's airspeed, angle of
        // attack, lift coefficient and spool, the winch cable's tension, the
        // hull's draught off its world position, and the craft's heading against
        // the P17 wind (the angle between where it points and where the wind
        // comes from) -- every one off the sim, the cert's (VEH3h) inputs.
        let (c_alt, c_ias, c_alpha, c_cl, c_spool, c_winch, c_draught, c_wind) = match seated_car {
            Some(car) => {
                let b = sim.bridge3d();
                let (pos, rot) = b
                    .body_of(car)
                    .and_then(|body| {
                        Some((
                            b.world().body_translation(body)?,
                            b.world().body_rotation(body)?,
                        ))
                    })
                    .unwrap_or((glam::DVec3::ZERO, glam::DQuat::IDENTITY));
                let v = b.vehicle_of(car);
                let f = v.and_then(|v| v.flight()).unwrap_or_default();
                let m = v.and_then(|v| v.marine());
                let (_, (wx, wz)) = inf_ecs::sky::water_environment(sim.world());
                let fwd = rot * glam::DVec3::Z;
                let wind_from = inf_math::patan2_64(-wx, -wz);
                let heading = inf_math::patan2_64(fwd.x, fwd.z);
                let rel = if wx == 0.0 && wz == 0.0 {
                    0.0
                } else {
                    let d = (heading - wind_from).to_degrees();
                    d - 360.0 * ((d + 180.0) / 360.0).floor()
                };
                (
                    pos.y,
                    f.airspeed_mps,
                    f.alpha_deg,
                    f.cl,
                    f.spool,
                    sim.winch_tension_n(car).unwrap_or(0.0),
                    m.map(|m| m.draught_m).unwrap_or(0.0),
                    rel,
                )
            }
            None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        };
        let line = match &probe.hero {
            Some(h) => format!(
                "{:.3},{},{:.4},{:.4},{:.4},{},{:.4},{:.4},{:.2},{:.2},{:.2},{},{},{:.4},{:.4},{:.2},{},{},{},{:.3},{},{:.3},{},{:.3},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{},{:.4},{:.4},{:.3},{:.0},{:.3},{:.3},{},{:.1},{:.1},{},{},{},{},{},{:.4},{:.1},{:.4},{:.4},{:.1},{},{:.2},{:.3},{:.3},{},{:.3},{:.3},{:.3},{:.3},{:.2},{},{},{},{},{},{:.4},{:.2},{:.2},{:.2},{:.3},{:.3},{:.0},{:.4},{:.1}\n",
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
                if cover.active { cover.peek } else { 0.0 },
                rounds_live,
                last_hit_m,
                equipped,
                recoil_mm,
                aim_recoil_deg,
                spread_deg,
                ads,
                casings_live,
                tail,
                class,
                attach,
                lock,
                engaged,
                incoming,
                heat,
                on_scene,
                responder_m,
                temp_at(0),
                temp_at(1),
                temp_at(2),
                temp_at(3),
                tyre_surface,
                driven.slip_ratio,
                driven.slip_lat,
                tyre_mu,
                drivetrain.rpm,
                drivetrain.clutch_lock,
                drivetrain.boost,
                u8::from(drivetrain.fuel_cut),
                car_health,
                engine_scale,
                flats,
                panes_broken,
                parts_shed,
                board_phase,
                board_seat,
                hand_m,
                hinge_deg,
                wheel_m,
                pedal_m,
                rim_deg,
                v_gear,
                v_load,
                v_slip_f,
                v_slip_r,
                v_surface,
                v_grain,
                v_whine,
                v_sq_f,
                v_sq_r,
                v_cmds,
                v_thumps,
                r_class,
                r_row,
                r_body,
                match r_hitch {
                    Some(a) => format!("{a:.2}"),
                    None => String::new(),
                },
                r_track,
                c_alt,
                c_ias,
                c_alpha,
                c_cl,
                c_spool,
                c_winch,
                c_draught,
                c_wind
            ),
            // **`no-hero` NAMES THE MODE COLUMN** (WPN2b audit, carried 224).
            //
            // It used to sit one field to the right -- five commas after the
            // frame instead of four -- which put the word at index 6, the
            // SPEED column, and left index 5, the MODE, empty. Every predicate
            // in `tools/demo/demo.ps1` reads the mode at `$c[5]`, so a
            // hero-less row was a blank mode rather than a named one, and a
            // driver waiting for a mode could not tell "the hero has not
            // spawned yet" from "this column is missing". Pre-existing since
            // wave FIX1 and harmless only because no predicate happened to
            // match either spelling.
            //
            // The row is 85 fields wide since wave VEH3g — 72 at VEH3e, the
            // five roster columns of VEH3f and the eight craft columns — which
            // the gate asserts against the armed branch above it and against
            // the demo README.
            None => format!(
                "{:.3},{},,,,no-hero,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n",
                sim.steps() as f64 / 60.0,
                probe.frame
            ),
        };
        use std::io::Write as _;
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
    }
}

/// **The demo loop's HOLD on a boarding beat** (VEH3d audit), `INF_PIE_BOARD_HOLD`
/// = `seconds[@metres]`: a PREVIEW session freezes its fixed steps for
/// `seconds` of wall time the first time the camera subject reaches each beat of
/// a boarding, and — with `@metres` — draws that frozen frame from a camera
/// `metres` off the beat's joint, looking at it.
///
/// It exists because the beats the wave is about are shorter than a
/// screenshot: the hand holds the outer handle for 0.1 s, the inner pull for a
/// handful of steps, and the implementer's frame `108-veh3d-hand-on-handle`
/// was taken with the hand already released (its HUD row carried no `HAND`)
/// — a frame of a claim that could not be seen. The steps are the same steps:
/// a hold runs NO step, it does not change one, so nothing the simulation does
/// depends on it. Read only in a `--pie` preview, for [`SPAWN_AT_ENV`]'s reason.
///
/// The beats, once each per boarding (re-armed when the subject is standing
/// idle again): `take` (the outer handle at weight 1 before the latch), `pull`
/// (the inner handle at weight 1, seated), `lock` (both hands on the rim past
/// 400 deg of rim), `throttle` (driving on more than 0.9 of throttle, faster
/// than [`THROTTLE_BEAT_MPS`]), `push` (the inner handle at weight 1, getting
/// out).
pub const BOARD_HOLD_ENV: &str = "INF_PIE_BOARD_HOLD";

/// The beat names [`BOARD_HOLD_ENV`] fires on, in bit order.
pub const BOARD_HOLD_BEATS: [&str; 5] = ["take", "pull", "lock", "throttle", "push"];

/// How fast the car must be going for the `throttle` beat, m/s — the demo
/// loop's own throttle frame fires on a hero moving faster than 1.5 m/s, and a
/// beat held at a standstill would be over before that row existed.
pub const THROTTLE_BEAT_MPS: f64 = 1.5;

/// See [`BOARD_HOLD_ENV`].
#[derive(Debug, Default, Clone)]
pub struct BoardHold {
    hold_s: f64,
    close_m: f64,
    left_s: f64,
    fired: u8,
    focus: Option<glam::DVec3>,
    /// The horizontal across the body at the beat (perpendicular to pelvis ->
    /// joint): a close-up from BEHIND a body reaching forward sees its back,
    /// so the close-up is taken from the side.
    aside: Option<glam::DVec3>,
}

impl BoardHold {
    /// Read [`BOARD_HOLD_ENV`]; inert when absent or unreadable.
    pub fn from_env() -> Self {
        let Ok(v) = std::env::var(BOARD_HOLD_ENV) else {
            return Self::default();
        };
        let mut it = v.trim().splitn(2, '@');
        let hold_s = it
            .next()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(0.0)
            .min(30.0);
        let close_m = it
            .next()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(0.0)
            .min(10.0);
        Self {
            hold_s,
            close_m,
            ..Default::default()
        }
    }

    /// A hold of `hold_s` seconds with a close-up `close_m` off the joint (`0`
    /// for none) — what [`Self::from_env`] builds, for a gate.
    pub fn new(hold_s: f64, close_m: f64) -> Self {
        Self {
            hold_s,
            close_m,
            ..Default::default()
        }
    }

    /// Whether this frame's fixed steps are held.
    pub fn holding(&self) -> bool {
        self.left_s > 0.0
    }

    /// `(the beat's joint, how far off it to draw from)` while a close-up hold
    /// is running.
    pub fn close_up(&self) -> Option<(glam::DVec3, f64)> {
        if !self.holding() || self.close_m <= 0.0 {
            return None;
        }
        self.focus.map(|f| (f, self.close_m))
    }

    /// The side the close-up looks from — see the field.
    pub fn close_up_aside(&self) -> Option<glam::DVec3> {
        self.aside
    }

    /// One display frame: count a running hold down, or start one on a beat not
    /// yet held this boarding. Answers a log line when a hold starts.
    pub fn tick(&mut self, sim: &RuntimeSim, dt: f64) -> Option<String> {
        use inf_ecs::boarding::BoardPhase;
        if self.hold_s <= 0.0 {
            return None;
        }
        if self.left_s > 0.0 {
            self.left_s = (self.left_s - dt.max(0.0)).max(0.0);
            return None;
        }
        let world = sim.world();
        let hero = inf_ecs::movement::camera_subject(world)?;
        let cm = world
            .world()
            .get::<inf_ecs::components::CharacterMovement>(world.entity_of(hero)?)?;
        let b = cm.runtime.boarding;
        if b.phase == BoardPhase::Idle && !cm.runtime.seat.is_seated() {
            self.fired = 0;
            return None;
        }
        let r = sim.boarding_residuals(hero);
        let (role, side) = (inf_anim::BoneRoleKind::Hand, b.hand_side);
        let joint = |role: inf_anim::BoneRoleKind, side: u8| {
            let s = if side == 0 {
                inf_anim::BoneSide::Left
            } else {
                inf_anim::BoneSide::Right
            };
            sim.posed_joint(hero, role, s)
        };
        let mid = |role: inf_anim::BoneRoleKind| match (joint(role, 0), joint(role, 1)) {
            (Some(a), Some(b)) => Some((a + b) * 0.5),
            _ => None,
        };
        let driving = b.phase == BoardPhase::Driving
            || (b.phase == BoardPhase::Idle && cm.runtime.seat.is_seated());
        let rim = if driving {
            inf_physics::d3::boarding::vehicle_steer(world, sim.bridge3d(), cm.runtime.seat.vehicle)
        } else {
            0.0
        };
        let beat =
            if b.phase == BoardPhase::OpeningDoor && b.mark_s < 0.0 && b.handle_weight >= 0.999 {
                Some((0u8, joint(role, side), r.and_then(|r| r.handle_m)))
            } else if b.phase == BoardPhase::Seated && b.handle_weight >= 0.999 {
                Some((1, joint(role, side), r.and_then(|r| r.handle_m)))
            } else if driving && rim.abs() > 400.0 && b.hand_weight >= 0.999 {
                Some((2, mid(role), r.and_then(|r| r.grips_m)))
            } else if driving && b.throttle_in > 0.9 && {
                let car = inf_physics::d3::boarding::car_frame(
                    world,
                    sim.bridge3d(),
                    cm.runtime.seat.vehicle,
                    false,
                );
                car.is_some_and(|c| c.vel.length() > THROTTLE_BEAT_MPS)
            } {
                Some((
                    3,
                    mid(inf_anim::BoneRoleKind::Foot),
                    r.and_then(|r| r.feet_m),
                ))
            } else if b.phase == BoardPhase::Exiting && b.handle_weight >= 0.999 {
                Some((4, joint(role, side), r.and_then(|r| r.handle_m)))
            } else {
                None
            };
        let (bit, focus, residual) = beat?;
        if self.fired & (1 << bit) != 0 {
            return None;
        }
        self.fired |= 1 << bit;
        self.left_s = self.hold_s;
        self.focus = focus;
        // The side of the body the beat's joint is on: its offset from the
        // pelvis across the body's own facing, so the close-up looks at the
        // reaching arm and not at the back hiding it.
        let (_, _, fwd) = inf_ecs::camera::basis(cm.runtime.body_yaw_deg, 0.0);
        let fwd = glam::DVec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
        self.aside = focus
            .zip(sim.posed_joint(
                hero,
                inf_anim::BoneRoleKind::Pelvis,
                inf_anim::BoneSide::Center,
            ))
            .map(|(f, p)| {
                let d = glam::DVec3::new(f.x - p.x, 0.0, f.z - p.z);
                (d - fwd * d.dot(fwd)).normalize_or_zero()
            })
            .filter(|a| a.length_squared() > 0.5);
        Some(format!(
            "{BOARD_HOLD_ENV} held `{}` for {:.1}s at step {} (residual {}; close-up {})",
            BOARD_HOLD_BEATS[bit as usize],
            self.hold_s,
            sim.steps(),
            residual
                .map(|m| format!("{:.1} mm", m * 1000.0))
                .unwrap_or_else(|| "-".to_string()),
            if self.close_m > 0.0 && focus.is_some() {
                format!("{:.2} m", self.close_m)
            } else {
                "off".to_string()
            }
        ))
    }
}

/// **The demo loop's CLEAR ROAD** (VEH3e audit), `INF_PIE_PLACE_CAR` =
/// `x,y,z/yaw@seconds`: in a PREVIEW session, at `seconds` of sim time, the car
/// NEAREST the camera subject is put at `(x, y, z)` facing `yaw` degrees
/// (`atan2(dx, dz)`, the placement's own convention), through
/// [`RuntimeSim::place_vehicle`] — the rig as a unit. Pair it with
/// `INF_PIE_SPAWN_AT` a moment later to stand the hero at its door.
///
/// It exists because three sessions of the VEH3e audio leg launched the
/// island's saloon from the Harbour City crossroads into the traffic queued at
/// the crossing, and the kerb and the slide were never reached: the car is
/// where the level parked it, and the cert needs a straight, empty street.
/// One placement, applied once; it moves a car, never a rule.
pub const PLACE_CAR_ENV: &str = "INF_PIE_PLACE_CAR";

/// See [`PLACE_CAR_ENV`].
#[derive(Debug, Default, Clone)]
pub struct CarPlacement {
    at: Option<(glam::DVec3, f64, f64)>,
    done: bool,
}

impl CarPlacement {
    /// Read [`PLACE_CAR_ENV`]; inert when absent, a refusal on stderr when
    /// malformed.
    pub fn from_env() -> Self {
        let Ok(v) = std::env::var(PLACE_CAR_ENV) else {
            return Self::default();
        };
        let parsed = (|| {
            let (body, when) = match v.trim().split_once('@') {
                Some((b, t)) => (b, t.trim().parse::<f64>().ok()?),
                None => (v.trim(), 1.0),
            };
            let (coords, yaw) = match body.split_once('/') {
                Some((c, y)) => (c, y.trim().parse::<f64>().ok()?),
                None => (body, 0.0),
            };
            let p: Vec<f64> = coords
                .split(',')
                .filter_map(|x| x.trim().parse::<f64>().ok())
                .collect();
            (p.len() == 3 && p.iter().all(|x| x.is_finite()) && yaw.is_finite() && when.is_finite())
                .then(|| (glam::DVec3::new(p[0], p[1], p[2]), yaw, when.max(0.0)))
        })();
        if parsed.is_none() {
            eprintln!("inf-player: {PLACE_CAR_ENV} `{v}` is not `x,y,z/yaw@seconds`");
        }
        Self {
            at: parsed,
            done: false,
        }
    }

    /// Place the car once its time has come. Answers a log line when it does.
    pub fn tick(&mut self, sim: &mut RuntimeSim) -> Option<String> {
        let (at, yaw, when) = self.at?;
        if self.done || (sim.steps() as f64) / 60.0 < when {
            return None;
        }
        self.done = true;
        let world = sim.world();
        let hero = inf_ecs::movement::camera_subject(world)?;
        let from = world
            .world()
            .get::<inf_ecs::components::Transform>(world.entity_of(hero)?)?
            .translation
            .to_dvec3();
        let car = sim
            .vehicles()
            .iter()
            .filter(|o| o.voice.is_some())
            .filter_map(|o| {
                let e = world.entity_of(o.chassis)?;
                let p = world
                    .world()
                    .get::<inf_ecs::components::Transform>(e)?
                    .translation
                    .to_dvec3();
                Some((o.chassis, (p - from).length()))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))?;
        let rot = glam::DQuat::from_rotation_y(yaw.to_radians());
        let placed = sim.place_vehicle(car.0, at, rot);
        Some(format!(
            "{PLACE_CAR_ENV} at t={when:.1}s put the car {} ({:.1} m from the hero) at {:.2},{:.2},{:.2} facing {yaw:.0} deg: {}",
            car.0,
            car.1,
            at.x,
            at.y,
            at.z,
            if placed { "placed" } else { "REFUSED (no body)" }
        ))
    }
}

/// **The demo loop's ROSTER LINE-UP** (wave VEH3f), `INF_PIE_LINEUP` =
/// `WHERE@seconds[;row,row,...]`: in a PREVIEW session, at `seconds` of sim
/// time, a line of roster rows is spawned side by side, through the
/// `vehicle.spawn` Blueprint verb's own door (`inf_ecs::roster::spawn_defined`).
///
/// * `WHERE` is `x,y,z/yaw` (a line through that point, every car facing
///   `yaw` degrees, `y` the ground there), or `ahead:M` -- `M` metres in front
///   of the camera subject, across its heading, every car facing it, each one
///   stood on the terrain under its own spot.
/// * With no row list: one row of EACH of the doc's eighteen classes, the
///   shortest of each so the line stays a street's width. With one: those rows,
///   in that order.
///
/// One placement, applied once; the rows are the committed catalogue, and
/// nothing about them is special-cased.
///
/// It exists because the island parks the classes that fit a kerb and a lot of
/// the construction ones, and a frame of all eighteen has nowhere else to come
/// from.
pub const LINEUP_ENV: &str = "INF_PIE_LINEUP";

/// See [`LINEUP_ENV`].
#[derive(Debug, Default, Clone)]
pub struct Lineup {
    at: Option<(glam::DVec3, f64, f64)>,
    /// `ahead:M` -- metres in front of the camera subject, instead of `at`'s
    /// point and heading.
    ahead_m: Option<f64>,
    rows: Vec<String>,
    done: bool,
}

impl Lineup {
    /// Read [`LINEUP_ENV`]; inert when absent, a refusal on stderr when
    /// malformed.
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var(LINEUP_ENV) else {
            return Self::default();
        };
        let (v, rows) = match raw.split_once(';') {
            Some((a, r)) => (
                a.to_string(),
                r.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            ),
            None => (raw.clone(), Vec::new()),
        };
        if let Some(rest) = v.trim().strip_prefix("ahead:") {
            let (m, when) = match rest.split_once('@') {
                Some((m, t)) => (m.trim().parse::<f64>().ok(), t.trim().parse::<f64>().ok()),
                None => (rest.trim().parse::<f64>().ok(), Some(1.0)),
            };
            return match (m, when) {
                (Some(m), Some(t)) if m.is_finite() && t.is_finite() => Self {
                    at: Some((glam::DVec3::ZERO, 0.0, t.max(0.0))),
                    ahead_m: Some(m),
                    rows,
                    done: false,
                },
                _ => {
                    eprintln!("inf-player: {LINEUP_ENV} `{raw}` is not `ahead:M@seconds`");
                    Self::default()
                }
            };
        }
        let parsed = (|| {
            let (body, when) = match v.trim().split_once('@') {
                Some((b, t)) => (b, t.trim().parse::<f64>().ok()?),
                None => (v.trim(), 1.0),
            };
            let (coords, yaw) = match body.split_once('/') {
                Some((c, y)) => (c, y.trim().parse::<f64>().ok()?),
                None => (body, 0.0),
            };
            let p: Vec<f64> = coords
                .split(',')
                .filter_map(|x| x.trim().parse::<f64>().ok())
                .collect();
            (p.len() == 3 && p.iter().all(|x| x.is_finite()) && yaw.is_finite() && when.is_finite())
                .then(|| (glam::DVec3::new(p[0], p[1], p[2]), yaw, when.max(0.0)))
        })();
        if parsed.is_none() {
            eprintln!("inf-player: {LINEUP_ENV} `{v}` is not `x,y,z/yaw@seconds`");
        }
        Self {
            at: parsed,
            ahead_m: None,
            rows,
            done: false,
        }
    }

    /// Spawn the line once its time has come. Answers a log line when it does.
    pub fn tick(&mut self, sim: &mut RuntimeSim) -> Option<String> {
        use inf_ecs::roster::{self, RosterClass};
        let (mut at, mut yaw, when) = self.at?;
        if self.done || (sim.steps() as f64) / 60.0 < when {
            return None;
        }
        self.done = true;
        let on_terrain = self.ahead_m.is_some();
        if let Some(m) = self.ahead_m {
            let world = sim.world();
            let hero = inf_ecs::movement::camera_subject(world)?;
            let t = *world
                .world()
                .get::<inf_ecs::components::Transform>(world.entity_of(hero)?)?;
            // Where the CAMERA looks (a standing hero's body yaw need not be
            // the view's -- the first session put the line behind the lens).
            let view = sim.camera().pose.yaw_deg;
            let h = view.to_radians();
            let fwd = glam::DVec3::new(inf_math::psin64(h), 0.0, inf_math::pcos64(h));
            at = t.translation.to_dvec3() + fwd * m;
            yaw = view + 180.0;
        }
        let r = yaw.to_radians();
        // The line runs across the heading: +X of a car facing `yaw`.
        let side = glam::DVec3::new(inf_math::pcos64(r), 0.0, -inf_math::psin64(r));
        // The line: the rows asked for, or the shortest row of every class.
        let wanted: Vec<(String, String)> = if self.rows.is_empty() {
            RosterClass::ALL
                .into_iter()
                .filter(|c| *c != RosterClass::Trailer)
                .filter_map(|class| {
                    roster::rows_of(class)
                        .into_iter()
                        .filter_map(|id| roster::roster().get(id).map(|d| (id, *d)))
                        .min_by(|a, b| {
                            a.1.half_extents
                                .z
                                .total_cmp(&b.1.half_extents.z)
                                .then(a.0.cmp(b.0))
                        })
                        .map(|(id, _)| (class.name().to_string(), id.to_string()))
                })
                .collect()
        } else {
            self.rows
                .iter()
                .map(|id| ("row".to_string(), id.clone()))
                .collect()
        };
        // Centred on `at`: half the line's width to either side.
        let widths: Vec<f64> = wanted
            .iter()
            .map(|(_, id)| {
                roster::roster()
                    .get(id)
                    .map(|d| 2.0 * d.half_extents.x.abs() + 3.0)
                    .unwrap_or(0.0)
            })
            .collect();
        let mut offset = -0.5 * widths.iter().sum::<f64>();
        let mut placed = Vec::new();
        for (label, id) in &wanted {
            let Some(def) = roster::roster().get(id).copied() else {
                continue;
            };
            let hx = def.half_extents.x.abs();
            offset += hx;
            let mut p = at + side * offset;
            let ground = if on_terrain {
                let g = sim.terrain_height_at(p.x, p.z);
                if g.is_finite() {
                    g
                } else {
                    at.y
                }
            } else {
                at.y
            };
            p.y = inf_ecs::vehicle::resting_origin_y(&def, ground) + 0.05;
            if roster::spawn_defined(sim.world_mut(), id, p, yaw).is_some() {
                placed.push(format!("{label}={id}"));
            }
            offset += hx + 3.0;
        }
        sim.world_mut().propagate();
        Some(format!(
            "{LINEUP_ENV} at t={when:.1}s spawned {} rows along {:.1} m centred on {:.2},{:.2},{:.2} facing {yaw:.0} deg: {}",
            placed.len(),
            widths.iter().sum::<f64>(),
            at.x,
            at.y,
            at.z,
            placed.join(" ")
        ))
    }
}

/// **The demo loop's ROSTER GALLERY** (VEH3f audit, priority a'),
/// `INF_PIE_GALLERY` = `ahead:M/side@start/dwell[;entry,...]`: in a PREVIEW
/// session, from `start` seconds of sim time, ONE vehicle at a time is put
/// `M` metres ahead of the camera subject along the camera's heading and
/// `side` metres to its right, on the terrain there, turned a three-quarter
/// view to the lens -- the previous one taken away -- and held `dwell`
/// seconds, each with a note in the hero log the demo loop photographs on.
///
/// * an entry is a roster row id (spawned through `roster::spawn_defined`,
///   the `vehicle.spawn` door) or `hero:<Set>` (`Sedan`, `Coupe`, `Suv`,
///   `Pickup`, `Cruiser`): the level's own authored car that wears that
///   set's car SHELL (wave VEH3f.2b; `hero:shell_<key>` names one directly),
///   MOVED there through [`RuntimeSim::place_vehicle`] (the island row's own
///   proportions -- nothing is re-derived for the frame);
/// * with no list: the shortest row of each of the eighteen classes, then the
///   five hero sets.
///
/// It exists because the implementer's eighteen-class line was 98 m wide and
/// partly behind the street's buildings, and no frame showed a hero body
/// alone. One car at a time, close, labelled by the log.
pub const GALLERY_ENV: &str = "INF_PIE_GALLERY";

/// See [`GALLERY_ENV`].
#[derive(Debug, Default, Clone)]
pub struct Gallery {
    ahead_m: f64,
    side_m: f64,
    start_s: f64,
    dwell_s: f64,
    entries: Vec<String>,
    next: usize,
    /// The roster car on show, to take away before the next one.
    shown: Option<(uuid::Uuid, inf_ecs::vehicle::VehicleDef)>,
    /// **The car the gallery camera frames** (wave VEH3f.2b): the one on
    /// show, whichever door put it there.
    focus: Option<uuid::Uuid>,
    armed: bool,
}

impl Gallery {
    /// Read [`GALLERY_ENV`]; inert when absent, a refusal on stderr when
    /// malformed.
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var(GALLERY_ENV) else {
            return Self::default();
        };
        let (head, list) = match raw.split_once(';') {
            Some((a, r)) => (a.to_string(), Some(r.to_string())),
            None => (raw.clone(), None),
        };
        let parsed = (|| {
            let rest = head.trim().strip_prefix("ahead:")?;
            let (place, time) = rest.split_once('@')?;
            let (m, side) = match place.split_once('/') {
                Some((m, s)) => (m.trim().parse::<f64>().ok()?, s.trim().parse::<f64>().ok()?),
                None => (place.trim().parse::<f64>().ok()?, 0.0),
            };
            let (start, dwell) = time.split_once('/')?;
            let (start, dwell) = (
                start.trim().parse::<f64>().ok()?,
                dwell.trim().parse::<f64>().ok()?,
            );
            [m, side, start, dwell]
                .iter()
                .all(|x| x.is_finite())
                .then_some((m, side, start.max(0.0), dwell.max(0.5)))
        })();
        let Some((ahead_m, side_m, start_s, dwell_s)) = parsed else {
            eprintln!(
                "inf-player: {GALLERY_ENV} `{raw}` is not `ahead:M/side@start/dwell[;entry,...]`"
            );
            return Self::default();
        };
        let entries: Vec<String> = match list {
            Some(l) => l
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            None => {
                use inf_ecs::roster::{self, RosterClass};
                let mut v: Vec<String> = RosterClass::ALL
                    .into_iter()
                    .filter(|c| *c != RosterClass::Trailer)
                    .filter_map(|class| {
                        roster::rows_of(class)
                            .into_iter()
                            .filter_map(|id| roster::roster().get(id).map(|d| (id, *d)))
                            .min_by(|a, b| {
                                a.1.half_extents
                                    .z
                                    .total_cmp(&b.1.half_extents.z)
                                    .then(a.0.cmp(b.0))
                            })
                            .map(|(id, _)| id.to_string())
                    })
                    .collect();
                v.extend(
                    inf_ecs::vehicle::HERO_SETS
                        .iter()
                        .map(|(name, _)| format!("hero:{name}")),
                );
                v
            }
        };
        Self {
            ahead_m,
            side_m,
            start_s,
            dwell_s,
            entries,
            next: 0,
            shown: None,
            focus: None,
            armed: true,
        }
    }

    /// **The gallery camera** (wave VEH3f.2b, the VEH3f audit's carried
    /// "frames far, inside a scaffold"): while a car is on show, an eye at its
    /// three-quarter FRONT -- ahead of the nose and out to the right flank,
    /// a car-length and a quarter off, a little above the roof -- and the
    /// point it looks at, the car's middle. Read off the car's live transform
    /// and collider every frame, so it frames what the world holds.
    /// `None` when nothing is on show.
    pub fn framing(&self, sim: &RuntimeSim) -> Option<(glam::DVec3, glam::DVec3)> {
        let car = self.focus?;
        let w = sim.world();
        let e = w.entity_of(car)?;
        let t = w.world().get::<inf_ecs::components::GlobalTransform>(e)?;
        let half = w
            .world()
            .get::<inf_ecs::components::Collider3D>(e)
            .map(|c| c.half_extents.to_dvec3())
            .unwrap_or(glam::DVec3::new(1.0, 0.7, 2.3));
        let at = t.translation();
        // The affine's linear part carries no scale on a chassis.
        let fwd =
            t.0.transform_vector3(glam::DVec3::Z)
                .normalize_or(glam::DVec3::Z);
        let right =
            t.0.transform_vector3(glam::DVec3::X)
                .normalize_or(glam::DVec3::X);
        let d = (2.0 * half.z).max(4.0) * 1.25;
        let eye = at
            + (fwd * 0.74 + right * 0.67).normalize() * d
            + glam::DVec3::Y * (half.y * 1.6 + 0.5);
        Some((eye, at + glam::DVec3::Y * (0.15 * half.y)))
    }

    /// Show the next vehicle once its time has come. Answers a log line when
    /// it does.
    pub fn tick(&mut self, sim: &mut RuntimeSim) -> Option<String> {
        use inf_ecs::roster;
        if !self.armed || self.next >= self.entries.len() {
            return None;
        }
        let due = self.start_s + self.next as f64 * self.dwell_s;
        if (sim.steps() as f64) / 60.0 < due {
            return None;
        }
        let i = self.next;
        self.next += 1;
        if let Some((g, def)) = self.shown.take() {
            inf_ecs::vehicle::despawn_rig(sim.world_mut(), g, &def);
        }
        let world = sim.world();
        let hero = inf_ecs::movement::camera_subject(world)?;
        let t = *world
            .world()
            .get::<inf_ecs::components::Transform>(world.entity_of(hero)?)?;
        let view = sim.camera().pose.yaw_deg;
        let h = view.to_radians();
        let fwd = glam::DVec3::new(inf_math::psin64(h), 0.0, inf_math::pcos64(h));
        let right = glam::DVec3::new(inf_math::pcos64(h), 0.0, -inf_math::psin64(h));
        let mut p = t.translation.to_dvec3() + fwd * self.ahead_m + right * self.side_m;
        let ground = {
            let g = sim.terrain_height_at(p.x, p.z);
            if g.is_finite() {
                g
            } else {
                p.y
            }
        };
        // Facing the lens, turned 35 degrees: the three-quarter front.
        let yaw = view + 180.0 - 35.0;
        let entry = self.entries[i].clone();
        let n = self.entries.len();
        if let Some(set) = entry.strip_prefix("hero:") {
            // **The island's hero rows wear the car SHELLS** (wave VEH3f.2b):
            // `hero:Sedan` finds the level's car whose shell body is
            // `shell_sedan`'s (the five sets' names map one to one), and a
            // `shell:<key>` entry names a shell directly.
            let key = if set.starts_with("shell_") {
                set.to_string()
            } else {
                format!("shell_{}", set.to_ascii_lowercase())
            };
            let Some(art) = inf_ecs::roster::ArtKey::from_name(&key).filter(|k| k.shell()) else {
                return Some(format!(
                    "{GALLERY_ENV} {}/{n} hero={set} UNKNOWN SHELL",
                    i + 1
                ));
            };
            let body = inf_ecs::roster::art_body_guid(art);
            let w = sim.world();
            let car = w.world().iter_entities().find_map(|e| {
                let m = e.get::<inf_ecs::components::MeshRef>()?;
                if m.asset != Some(body) {
                    return None;
                }
                let parent = w.parent_of(e.id())?;
                w.guid_of(parent)
            });
            let Some(car) = car else {
                return Some(format!(
                    "{GALLERY_ENV} {}/{n} hero={set} NOT RESIDENT (no car wears it within the streamed cells)",
                    i + 1
                ));
            };
            let half = w
                .entity_of(car)
                .and_then(|e| w.world().get::<inf_ecs::components::Collider3D>(e))
                .map(|c| c.half_extents.to_dvec3())
                .unwrap_or(glam::DVec3::splat(1.0));
            let name = w
                .entity_of(car)
                .and_then(|e| w.name_of(e))
                .unwrap_or("?")
                .to_string();
            p.y = ground + half.y + 0.45;
            let placed = sim.place_vehicle(car, p, glam::DQuat::from_rotation_y(yaw.to_radians()));
            self.focus = Some(car);
            return Some(format!(
                "{GALLERY_ENV} {}/{n} class=hero row={set} label={name} body=shell:{key} half={:.2},{:.2},{:.2} at {:.1},{:.1},{:.1} {}",
                i + 1,
                half.x,
                half.y,
                half.z,
                p.x,
                p.y,
                p.z,
                if placed { "placed" } else { "REFUSED" }
            ));
        }
        let Some(def) = roster::roster().get(&entry).copied() else {
            return Some(format!(
                "{GALLERY_ENV} {}/{n} row={entry} UNKNOWN ROW",
                i + 1
            ));
        };
        p.y = inf_ecs::vehicle::resting_origin_y(&def, ground) + 0.05;
        let guid = roster::spawn_defined(sim.world_mut(), &entry, p, yaw)?;
        sim.world_mut().propagate();
        self.shown = Some((guid, def));
        self.focus = Some(guid);
        let body = if let Some(k) = def.art.filter(|k| k.shell()) {
            format!("shell:{}", k.name())
        } else if def.art.is_some() {
            "art".to_string()
        } else if def.body_mesh.is_some() {
            "dcc-hero-panels".to_string()
        } else {
            format!("{:?}", def.body).to_lowercase()
        };
        Some(format!(
            "{GALLERY_ENV} {}/{n} class={} row={entry} label={} body={body} half={:.2},{:.2},{:.2} at {:.1},{:.1},{:.1}",
            i + 1,
            def.roster_class.map(|c| c.name()).unwrap_or("-"),
            roster::roster_label(&entry).unwrap_or(&entry),
            def.half_extents.x,
            def.half_extents.y,
            def.half_extents.z,
            p.x,
            p.y,
            p.z
        ))
    }
}

/// **The demo loop's HOLD on an AUDIO beat** (VEH3e audit), `INF_PIE_AUDIO_HOLD`
/// = `seconds`: a PREVIEW session freezes its fixed steps for `seconds` of wall
/// time the first time, per boarding, the camera subject's car reaches each of
/// three beats a frame of the car's voice has to show:
///
/// * `burnout` — a squeal (read back off the AUDIO ENGINE's voice) louder than
///   0.2 below [`BURNOUT_BEAT_MPS`];
/// * `thump` — a surface impulse the stream PLAYED on the car's keys since the
///   last display frame, faster than 3 m/s;
/// * `slide` — a squeal louder than 0.3 above [`SLIDE_BEAT_MPS`].
///
/// It exists because the implementer's burnout frame was taken ~1 s after the
/// row that triggered it, with the HUD's squeal already back at zero — a frame
/// of a claim that could not be seen — and a kerb's thump is ONE step. The
/// steps are the same steps: a hold runs no step and changes none (the mixer
/// keeps playing the loops it was told), so nothing the simulation or the
/// command stream does depends on it. [`BOARD_HOLD_ENV`]'s own door, one
/// instrument over; its five beats are pinned by VEH3d's gate and these three
/// are not theirs.
pub const AUDIO_HOLD_ENV: &str = "INF_PIE_AUDIO_HOLD";

/// The beat names [`AUDIO_HOLD_ENV`] fires on, in bit order.
pub const AUDIO_HOLD_BEATS: [&str; 3] = ["burnout", "thump", "slide"];

/// The `burnout` beat's ceiling, m/s: a tyre screaming while the car barely
/// moves.
pub const BURNOUT_BEAT_MPS: f64 = 5.0;

/// The `slide` beat's floor, m/s.
pub const SLIDE_BEAT_MPS: f64 = 8.0;

/// See [`AUDIO_HOLD_ENV`].
#[derive(Debug, Default, Clone)]
pub struct AudioHold {
    hold_s: f64,
    left_s: f64,
    fired: u8,
    audio_seen: u64,
}

impl AudioHold {
    /// Read [`AUDIO_HOLD_ENV`]; inert when absent or unreadable.
    pub fn from_env() -> Self {
        let hold_s = std::env::var(AUDIO_HOLD_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(0.0)
            .min(30.0);
        Self::new(hold_s)
    }

    /// A hold of `hold_s` seconds — what [`Self::from_env`] builds, for a gate.
    pub fn new(hold_s: f64) -> Self {
        Self {
            hold_s,
            ..Default::default()
        }
    }

    /// Whether this frame's fixed steps are held.
    pub fn holding(&self) -> bool {
        self.left_s > 0.0
    }

    /// One display frame: count a running hold down, or start one on a beat not
    /// yet held this boarding. Answers a log line when a hold starts.
    pub fn tick(&mut self, sim: &RuntimeSim, dt: f64) -> Option<String> {
        use inf_ecs::vehicle_audio::{entity_key, voice_key, VoiceLayer};
        if self.hold_s <= 0.0 {
            return None;
        }
        // What the stream queued since the last display frame -- a frame can
        // run several fixed steps, and a kerb's Play is on one of them.
        let queued = sim.audio_commands_queued();
        let fresh = queued.saturating_sub(self.audio_seen) as usize;
        self.audio_seen = queued;
        if self.left_s > 0.0 {
            self.left_s = (self.left_s - dt.max(0.0)).max(0.0);
            return None;
        }
        let world = sim.world();
        let hero = inf_ecs::movement::camera_subject(world)?;
        let seat = world
            .world()
            .get::<inf_ecs::components::CharacterMovement>(world.entity_of(hero)?)?
            .runtime
            .seat;
        if !seat.is_seated() {
            self.fired = 0;
            return None;
        }
        let car = seat.vehicle;
        let speed = sim
            .vehicles()
            .iter()
            .find(|o| o.chassis == car)
            .and_then(|o| o.voice)
            .map(|t| t.speed_mps.abs())?;
        let key = entity_key(car);
        let squeal = [VoiceLayer::SquealFront, VoiceLayer::SquealRear]
            .iter()
            .filter_map(|l| sim.voice_params(voice_key(key, *l)))
            .fold(0.0f64, |m, (v, _)| m.max(v));
        let impulses = [
            voice_key(key, VoiceLayer::ImpulseFront),
            voice_key(key, VoiceLayer::ImpulseRear),
        ];
        let log = sim.audio_command_log();
        let thump = log[log.len().saturating_sub(fresh)..]
            .iter()
            .any(|c| matches!(c, inf_audio::AudioCommand::Play(p) if impulses.contains(&p.source)));
        let bit = if squeal > 0.2 && speed < BURNOUT_BEAT_MPS {
            0u8
        } else if thump && speed > 3.0 {
            1
        } else if squeal > 0.3 && speed > SLIDE_BEAT_MPS {
            2
        } else {
            return None;
        };
        if self.fired & (1 << bit) != 0 {
            return None;
        }
        self.fired |= 1 << bit;
        self.left_s = self.hold_s;
        Some(format!(
            "{AUDIO_HOLD_ENV} held `{}` for {:.1}s at step {} (squeal {squeal:.3} at {speed:.2} m/s, thump {thump})",
            AUDIO_HOLD_BEATS[bit as usize],
            self.hold_s,
            sim.steps()
        ))
    }
}

/// **The demo loop's see-through car** (VEH3d audit), `INF_PIE_CUTAWAY` = an
/// alpha in `(0, 1)`: in a PREVIEW session, every drawn part of the car the
/// camera subject is SEATED in (or climbing into) is drawn TRANSLUCENT at that
/// alpha, and put back as it was the moment the subject is out of it.
///
/// It is VEH3c's `doors_open` doctrine for the seated frames: the saloon's
/// primitive body is an opaque box, and the implementer's seated, full-lock and
/// throttle frames showed the camera looking over a blue roof with no body in
/// the car at all — the hands on the rim and the feet on the pedals were read
/// off the columns and never seen. It writes the car's `Material` components
/// only (`blend` and the base colour's alpha), which no fixed step reads and
/// no trace section folds. Read only in a `--pie` preview.
pub const CUTAWAY_ENV: &str = "INF_PIE_CUTAWAY";

/// See [`CUTAWAY_ENV`].
#[derive(Debug, Default, Clone)]
pub struct Cutaway {
    alpha: Option<f32>,
    faded: Option<Uuid>,
    kept: Vec<(Uuid, inf_ecs::components::Material)>,
}

impl Cutaway {
    /// Read [`CUTAWAY_ENV`]; inert when absent, unreadable, or not in `(0, 1)`.
    pub fn from_env() -> Self {
        Self {
            alpha: std::env::var(CUTAWAY_ENV)
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|a| a.is_finite() && *a > 0.0 && *a < 1.0),
            ..Default::default()
        }
    }

    /// A cutaway at `alpha` — what [`Self::from_env`] builds, for a gate.
    pub fn with_alpha(alpha: f32) -> Self {
        Self {
            alpha: Some(alpha),
            ..Default::default()
        }
    }

    /// One display frame: fade the subject's car, or put the last one back.
    /// Answers a log line when the faded car changes.
    pub fn tick(&mut self, sim: &mut RuntimeSim) -> Option<String> {
        let alpha = self.alpha?;
        let want = {
            let world = sim.world();
            inf_ecs::movement::camera_subject(world)
                .and_then(|h| world.entity_of(h))
                .and_then(|e| {
                    world
                        .world()
                        .get::<inf_ecs::components::CharacterMovement>(e)
                })
                .filter(|cm| cm.runtime.seat.is_seated())
                .map(|cm| cm.runtime.seat.vehicle)
        };
        if want == self.faded {
            return None;
        }
        let world = sim.world_mut();
        for (guid, m) in self.kept.drain(..) {
            if let Some(e) = world.entity_of(guid) {
                world.world_mut().entity_mut(e).insert(m);
            }
        }
        self.faded = want;
        let chassis = want?;
        let root = world.entity_of(chassis)?;
        let mut n = 0usize;
        for e in world.subtree(root) {
            let (Some(guid), Some(m)) = (
                world
                    .world()
                    .get::<inf_ecs::components::Guid>(e)
                    .map(|g| g.0),
                world
                    .world()
                    .get::<inf_ecs::components::Material>(e)
                    .copied(),
            ) else {
                continue;
            };
            if world
                .world()
                .get::<inf_ecs::components::MeshRef>(e)
                .is_none()
            {
                continue;
            }
            self.kept.push((guid, m));
            let mut faded = m;
            faded.blend = inf_ecs::components::BlendMode::Translucent;
            faded.base_color.a = alpha;
            world.world_mut().entity_mut(e).insert(faded);
            n += 1;
        }
        Some(format!(
            "{CUTAWAY_ENV} drew {n} part(s) of {chassis} at alpha {alpha:.2}"
        ))
    }
}
