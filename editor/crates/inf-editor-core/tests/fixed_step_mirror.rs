//! **THE TWO FIXED STEPS AGREE ABOUT WHEN A VEHICLE RUNS** (island wave VEH1a).
//!
//! P29.7 put `step_vehicles` inside the last statement of
//! `inf_physics::d3::step_character_movement`, and its reason is on the record:
//! *"a sibling function called separately by each host would be a hand-maintained
//! mirror, and this phase's own ledger records two defects of exactly that
//! shape."* It was a good reason and it bought a real thing — the two hosts could
//! not disagree about whether a car stepped, because neither of them named it.
//!
//! What it cost is the thing wave I4b existed to remove: a car's milliseconds
//! were charged to `character move`, where they could not be told from a crowd's,
//! and *"a step that cannot say where its milliseconds went"* is the defect the
//! whole `STEP_PHASES` instrument answers. So VEH1a gives the vehicle step its
//! own row and **pays for the mirror instead of avoiding it**.
//!
//! This file is that payment. It pins the two call sites character-for-character
//! through the same `MIRROR-BEGIN` fence instrument `projector_mirror` uses, and
//! then pins the two *neighbourhoods* — because an identical statement in two
//! places says nothing if one host runs it before the solver and the other after.
//!
//! # Why a second binary and not an arm in `projector_mirror`
//!
//! `projector_mirror` is about the two **projectors** (scene → `RenderScene`),
//! which is a different pair of files and a different question. A fixed-step
//! mirror that lived there would be found by nobody looking for it, and the day
//! a third fixed-step statement needs fencing the file it belongs in already
//! exists.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// The editor's fixed step.
const EDITOR: &str = "editor/crates/inf-editor-core/src/simulate.rs";
/// The shipped player's fixed step.
const PLAYER: &str = "runtime/inf-player/src/runtime_sim.rs";

/// One `// MIRROR-BEGIN <tag>` … `// MIRROR-END <tag>` region, whitespace
/// stripped.
///
/// The fences are **counted**, not found — `projector_mirror::fenced`'s own rule,
/// which is the I1 audit's law applied to a delimiter: a `contains` needle that
/// is a prefix of a declaration can never fail, and neither can a second fence.
/// Restated here rather than shared because the two binaries cannot see each
/// other's helpers and a `support` module for six lines is a worse trade than
/// six lines.
fn fenced(src: &str, tag: &str, who: &str) -> String {
    let (b, e) = (
        format!("// MIRROR-BEGIN {tag}"),
        format!("// MIRROR-END {tag}"),
    );
    assert_eq!(src.matches(&b).count(), 1, "{who}: {tag} begin fences");
    assert_eq!(src.matches(&e).count(), 1, "{who}: {tag} end fences");
    let (i, j) = (
        src.find(&b).expect("checked"),
        src.find(&e).expect("checked"),
    );
    assert!(j > i, "{who}: the {tag} fence is inverted");
    src[i..j].chars().filter(|c| !c.is_whitespace()).collect()
}

/// The byte offset of `needle`'s single occurrence in `src`.
///
/// Counted, for the same reason `fenced` counts: an ordering claim built on
/// `find` would silently be about whichever of two call sites came first.
fn only(src: &str, needle: &str, who: &str) -> usize {
    assert_eq!(
        src.matches(needle).count(),
        1,
        "{who}: expected exactly one `{needle}`"
    );
    src.find(needle).expect("counted")
}

/// **The two hosts run the same vehicle statement.**
#[test]
fn both_fixed_steps_step_their_vehicles_the_same_way() {
    let editor = fenced(&read(EDITOR), "vehicle_step", "the editor SimSession");
    let player = fenced(&read(PLAYER), "vehicle_step", "the shipped RuntimeSim");
    assert!(
        editor.len() > 40,
        "the `vehicle_step` fence is {} chars — an empty fence would make this \
         gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the vehicle step has drifted between the editor's Simulate and the \
         shipped player. P29.7 kept the two identical by refusing to let either \
         host name the call; VEH1a let both name it in exchange for a phase row, \
         and this is the whole of what that trade rests on"
    );
    for (needle, why) in [
        (
            "inf_physics::d3::step_vehicles(",
            "the one Ring-0 vehicle door — a host that inlined the wheel rays \
             would be a second vehicle model",
        ),
        (
            "self.vehicles=",
            "the outcomes are retained, so a two-host gate can compare what the \
             cars DID and not only that both hosts called something",
        ),
    ] {
        let n: String = needle.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            editor.contains(&n),
            "the vehicle step no longer carries `{needle}`: {why}"
        );
    }
}

/// **The two hosts make the same engine noise.**
///
/// The audio half of the same trade. `inf_ecs::vehicle::engine_cue` is Ring 0
/// and is the *decision*; this is the six lines of mapping onto the P12.3 queue,
/// and there is one copy of them per host by construction.
#[test]
fn both_audio_steps_drive_the_engine_loop_the_same_way() {
    let editor = fenced(
        &read(EDITOR),
        "vehicle_engine_audio",
        "the editor SimSession",
    );
    let player = fenced(
        &read(PLAYER),
        "vehicle_engine_audio",
        "the shipped RuntimeSim",
    );
    assert!(
        editor.len() > 300,
        "the `vehicle_engine_audio` fence is {} chars — an empty fence would \
         make this gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the engine loop has drifted between the editor's Simulate and the \
         shipped player — so a PIE preview and a shipped build would make \
         different sounds from the same drive"
    );
    for (needle, why) in [
        (
            "inf_ecs::vehicle::engine_cue(",
            "the DECISION is Ring 0 — a host that computed a pitch itself would \
             be the second authority the P12 doctrine exists to prevent",
        ),
        (
            "out.revs,out.load",
            "the cue is a function of THIS step's published outcome, not of \
             whatever the vehicle map holds when a later phase asks",
        ),
        (
            "src.pitch,src.volume",
            "the emitter's authored pitch and volume are the base the cue scales \
             — an author says a truck idles low on the `AudioSource`, not in \
             engine code",
        ),
        (
            "started.insert(out.chassis)",
            "the Play is emitted ONCE and the latch is the same set autoplay and \
             the despawn sweep use, or an engine would restart its clip sixty \
             times a second",
        ),
        (
            "AudioCommand::SetPitch",
            "the loop is SetPitch/SetVolume over a P12.3 command that already \
             existed — the wave added no audio API",
        ),
    ] {
        let n: String = needle.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            editor.contains(&n),
            "the engine loop no longer carries `{needle}`: {why}"
        );
    }
    // …and it adds no COMMAND. `AudioCommand` is an enum the whole tree matches
    // on exhaustively, and a new variant would be a new API in a wave whose
    // audio clause is explicitly zero-new-API.
    for forbidden in ["AudioCommand::SetPosition", "AudioCommand::Stop"] {
        let n: String = forbidden.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            !editor.contains(&n),
            "the engine loop reaches for `{forbidden}` — see the wave's carried \
             item: the emitter is spatialized where its `Play` was issued, and \
             the missing command is a decision with its own arms, not a footnote"
        );
    }
}

/// **Both hosts step their TRAFFIC the same way** (wave VEH2b).
///
/// The `vehicle_step` fence's argument, one system along and with a sharper
/// edge: the traffic step DERIVES a level's carriageway and its whole car
/// population. Two hosts that called it with different arguments — or one that
/// did not call it at all — would not merely diverge on a drive; one of them
/// would have an empty street.
#[test]
fn both_fixed_steps_step_their_traffic_the_same_way() {
    let editor = fenced(&read(EDITOR), "traffic_step", "the editor SimSession");
    let player = fenced(&read(PLAYER), "traffic_step", "the shipped RuntimeSim");
    assert_eq!(
        editor, player,
        "the two hosts' traffic steps differ character for character"
    );
    assert!(
        editor.contains("step_traffic("),
        "the traffic fence does not call the traffic step: {editor}"
    );
}

/// **…and they run it in the same PLACE.**
///
/// The traffic step writes a driver's INTENT and builds bodies. Run after the
/// character step it would write an intent nothing reads until the next step;
/// run after the physics sync it would build a car the bridge does not see
/// until the next step. Both are one-step lags rather than divergences — which
/// is exactly why an equality gate cannot see them, and why the order is
/// asserted as an order.
#[test]
fn the_traffic_step_sits_between_the_crowd_and_the_physics_sync_on_both_hosts() {
    for (who, rel) in [
        ("the editor SimSession", EDITOR),
        ("the shipped RuntimeSim", PLAYER),
    ] {
        let src = read(rel);
        let crowd = only(&src, "inf_ecs::crowd::step_crowd_banded(", who);
        let traffic = only(&src, "// MIRROR-BEGIN traffic_step", who);
        let mover = only(&src, "inf_physics::d3::step_character_movement(", who);
        assert!(
            crowd < traffic,
            "{who}: the traffic step runs BEFORE the crowd, so a car and a pedestrian would be tiered against two different bands"
        );
        assert!(
            traffic < mover,
            "{who}: the traffic step runs AFTER the character step, so a driver's intent is one step stale for ever"
        );
        // …and before the 3D sync, so a car built this step is a body the
        // solver has this step. The LAST occurrence, because both hosts also
        // name `sync_from_world_sim` in their one-shot seeding path hundreds of
        // lines above the fixed step, and the first hit is that one.
        let sync_at = src
            .rfind("sync_from_world_sim(")
            .unwrap_or_else(|| panic!("{who}: no 3D sync"));
        assert!(
            traffic < sync_at,
            "{who}: the traffic step runs after the 3D sync, so a car it builds is drawn a step before it is solid"
        );
    }
}

/// **…and they run it in the same PLACE.**
///
/// The fence pins the statement; this pins its neighbours. A vehicle step that
/// ran after the solver on one host and before it on the other would satisfy an
/// equality gate perfectly and diverge on the first step of a drive, because a
/// suspension force applied after `bridge.step` is a force the solver never saw.
///
/// The claim is an *order*, so it is asserted as one: character move, then the
/// vehicle step, then gameplay, then the solver — on both hosts, from their own
/// source.
#[test]
fn the_vehicle_step_sits_between_the_character_step_and_the_solver_on_both_hosts() {
    for (who, rel, solver) in [
        ("the editor SimSession", EDITOR, "self.bridge3d.step(dt)"),
        ("the shipped RuntimeSim", PLAYER, "self.bridge3d.step(dt)"),
    ] {
        let src = read(rel);
        let mover = only(&src, "inf_physics::d3::step_character_movement(", who);
        let vehicle = only(&src, "// MIRROR-BEGIN vehicle_step", who);
        let gameplay = only(&src, "inf_physics::d3::step_gameplay(", who);
        let step = only(&src, solver, who);
        assert!(
            mover < vehicle,
            "{who}: the vehicle step runs BEFORE the character step, so a \
             driver's controls would be last step's"
        );
        assert!(
            vehicle < gameplay,
            "{who}: the vehicle step runs after gameplay — the ordering P29.7 \
             shipped put it immediately after the mover, and moving it is a \
             behavioural change nothing here asked for"
        );
        assert!(
            gameplay < step,
            "{who}: the solver runs before gameplay, which is not this engine's \
             fixed step at all"
        );
        assert!(
            vehicle < step,
            "{who}: the vehicle step runs AFTER the solver, so every suspension \
             force it applies is a force the solver has already integrated past \
             — the car would fall through its own springs for one step, every \
             step"
        );
    }
}

/// **The vehicle step is not still inside the movement door.**
///
/// The falsifier for the trade above: if `step_character_movement` kept calling
/// `step_vehicles` as well, both hosts would step every vehicle **twice** — two
/// sets of suspension forces, two ray casts, and a `vehicle` row measuring half
/// of it. An equality fence cannot see that, because the duplicate is in a third
/// file.
#[test]
fn the_movement_door_no_longer_steps_the_vehicles_itself() {
    let src = read("crates/inf-physics/src/d3/movement.rs");
    // Read a SCOPE, not a spelling (the P23 law) — and the first cut of this arm
    // did neither (VEH1a audit). It counted two *qualified* spellings,
    // `vehicle::step_vehicles(` plus `super::vehicle::step_vehicles(`, of which
    // the second **contains** the first, so one call would have been reported as
    // two; and it counted them over the whole file INCLUDING its comments, so it
    // passed only because this module's own prose happens not to put a paren
    // after the name it stopped calling. Neither of those is the failure that
    // matters. The cheapest way this regression actually comes back is
    // `use super::vehicle::step_vehicles;` at the top and a bare
    // `step_vehicles(world, bridge, dt);` in the body, and the old arm could not
    // see it at all.
    //
    // So the ban is on the NAME, in CODE, however it is qualified — with comment
    // lines dropped first, because a module is allowed to say what it stopped
    // calling and this one does.
    let hits: Vec<(usize, String)> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim_start();
            !t.starts_with("//") && !t.starts_with('*')
        })
        .filter(|(_, l)| l.contains("step_vehicles"))
        .map(|(i, l)| (i + 1, l.trim().to_string()))
        .collect();
    assert!(
        hits.is_empty(),
        "`inf_physics::d3::movement` names `step_vehicles` in code at {hits:?}; \
         both hosts now call it too, so every vehicle would be stepped twice a \
         step — two sets of suspension forces, two sets of rays, and a `vehicle` \
         row measuring half of it"
    );
}

/// **The two hosts hear the same doorway** (island wave VEN1b).
///
/// The re-evaluation half of the same trade, and it needs the same gate for the
/// same reason: `inf_physics::d3::audio::portal_gain` is Ring 0 and is the
/// *decision*; this is the mapping onto the queue, and there is one copy of it
/// per host by construction. Two copies that drifted would make a PIE preview
/// and a shipped build muffle a club by different amounts from the same
/// doorway — and the audio command stream is exactly what `physics_demo`'s
/// gate (c) compares between them.
#[test]
fn both_audio_steps_hear_the_same_doorway() {
    let editor = fenced(&read(EDITOR), "doorway_occlusion", "the editor SimSession");
    let player = fenced(&read(PLAYER), "doorway_occlusion", "the shipped RuntimeSim");
    assert!(
        editor.len() > 400,
        "the `doorway_occlusion` fence is {} chars — an empty fence would make \
         this gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the doorway occlusion has drifted between the editor's Simulate and \
         the shipped player"
    );
    for (needle, why) in [
        (
            "inf_physics::d3::audio::portal_gain_in(",
            "the DECISION is Ring 0 — a host that decided its own cut would be \
             the second authority the P12 doctrine exists to prevent, and it is \
             the shape this wave replaced (two copies of one raycast with a \
             -12 dB constant beside each)",
        ),
        (
            "inf_physics::d3::audio::portal_doors(world)",
            "the door list is built ONCE for the step and not once per source \
             (VEN1b audit). `d3::door::placements` allocates a label per door \
             over every doorway in the resident world, and the audio phase \
             measured 0.55 ms a step at the club rebuilding it five times over \
             — a host that dropped the hoist would put the cost back without \
             changing a single verdict",
        ),
        (
            "src.occlusion&&src.spatial&&src.looping",
            "only a LOOPING spatial source that opts in is re-evaluated — a \
             one-shot's gain is right when it is taken, and re-testing every \
             emitter in a settlement every step is the cost this predicate \
             refuses",
        ),
        (
            ">src.max_distance",
            "the emitter's own reach is the cull — past it the spatial model \
             has already taken the gain to zero, and this is the only distance \
             cull the audio step has at all",
        ),
        (
            "AudioCommand::SetOcclusion",
            "the per-step push is what makes a loop's occlusion live; \
             `PlayCommand::occlusion_gain` alone is taken once and never again",
        ),
        (
            "lowpass_hz:p.lowpass_hz",
            "the cutoff a shut door implies rides the command, so it is a \
             number two hosts are compared on rather than a claim in a doc",
        ),
    ] {
        let n: String = needle.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            editor.contains(&n),
            "the doorway loop no longer carries `{needle}`: {why}"
        );
    }
}

/// **THE GUNSHOT IS THE SAME NOISE ON BOTH HOSTS** (wave WPN1).
///
/// The report is a *command*, not an entity: there is nothing in the world to
/// compare, so the only thing that can keep the two hosts saying the same thing
/// about a shot is the source text. That is what this fence is — the
/// `vehicle_engine_audio` argument at a weapon, and the reason a report was
/// built inside `fire_weapon_audio` rather than as a second host-side loop.
#[test]
fn both_hosts_report_a_gunshot_the_same_way() {
    let editor = fenced(&read(EDITOR), "weapon_report", "the editor SimSession");
    let player = fenced(&read(PLAYER), "weapon_report", "the shipped RuntimeSim");
    assert!(
        editor.len() > 300,
        "the `weapon_report` fence is {} chars — an empty fence would make this \
         gate vacuous",
        editor.len()
    );
    assert_eq!(
        editor, player,
        "the gunshot report has drifted between the editor's Simulate and the \
         shipped player. A preview that made a different noise from the shipped \
         build is a bug no compiler and no screenshot finds"
    );
    for (needle, why) in [
        (
            "inf_ecs::weapon::report_source()",
            "the clip, the bus, the volume and the reach are ONE Ring-0 \
             description — the `venue_music_source` shape, and the alternative \
             is four constants in two host-side loops that have to be compared \
             character for character to stay in step",
        ),
        (
            "guid_source_key(hit.shooter)",
            "the report is keyed on the SHOOTER, so a barrel has one voice: \
             keyed per shot a 600 rpm burst is ten live voices a second, and \
             keyed on the target a miss would be silent",
        ),
        (
            "report.spatial.then_some(hit.from)",
            "a report is heard at the MUZZLE and the impact at the hit — the \
             two positions are the whole difference between the two commands, \
             and a report at `hit.to` would put a miss's noise 400 m away",
        ),
        (
            "guid_source_key(target)",
            "the impact is still the target's own emitter, unchanged by this \
             wave — a report that replaced it would have made every wall in \
             every level silent again",
        ),
        (
            "ifhit.loud{",
            "only a LOUD attack reports. A punch goes through this same list, \
             and a host that dropped the guard would fire a rifle's clip every \
             time somebody threw one",
        ),
    ] {
        let n: String = needle.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            editor.contains(&n),
            "the weapon report no longer carries `{needle}`: {why}"
        );
    }
    // …and the report really is FIRST: a queue is ordered, and a gate that
    // compares two command streams cannot tell an ordering it never asserted.
    let report = only(&editor, "guid_source_key(hit.shooter)", "the fence");
    let impact = only(&editor, "guid_source_key(target)", "the fence");
    assert!(
        report < impact,
        "the impact is queued before the report — a shot would be heard after \
         the thing it hit"
    );
}
