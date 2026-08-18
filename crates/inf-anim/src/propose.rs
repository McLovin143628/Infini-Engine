//! **Propose the graph** — pillar S3 (P29.5).
//!
//! §13's measured finding about the UE5 reference is that its author never
//! opened the animation graphs: an `AnimBlueprint` is a binary `.uasset`, so the
//! stock one gets shipped as delivered. Two things have to be true for that to
//! stop being the rational choice, and this module is the second of them. The
//! first is that a machine is **text** (pillar S1, `.inf_sm`'s TOML sidecar).
//! The second is that an author does not start from an empty canvas.
//!
//! # What "proposed" means, exactly
//!
//! It means a **normal `.inf_sm`**. Not a mode, not a black box, not a runtime
//! that infers a graph each frame — [`propose_machine`] returns a
//! [`StateMachine`] the author opens in the same canvas, edits with the same
//! tools and reviews as the same diff as one they built by hand. The proposal's
//! *reasoning* comes back beside it as [`Proposal::notes`], so the panel can say
//! why a threshold is 2.7 m/s instead of leaving the author to guess.
//!
//! # What it clusters on
//!
//! The facts [`crate::derive`] put on the clips at import, read back off them:
//! ground speed, gait tier, whether the feet cycle. Nothing here samples a pose
//! or opens a file — the derivation is the measurement and this is the decision,
//! which is why re-deriving a clip set and re-proposing over it are separate
//! buttons.
//!
//! # The shape it proposes
//!
//! * One **gait state** per non-empty tier (idle / walk / run / sprint), playing
//!   a single clip when the tier has one and a **1D blend space** over the speed
//!   parameter when it has more — which is exactly the case a hand-author gets
//!   wrong, because it needs each clip's own depicted speed as its axis position
//!   and that number is derived, not typed.
//! * A **chain** of transitions between adjacent tiers, thresholded at the
//!   midpoint of the two tiers' depicted speeds, so each state is used where its
//!   own stride is the closer match ([`crate::locomotion`]'s rule, generalised
//!   from three clips to a set).
//! * One **one-shot state** per clip whose feet do not cycle — a jump, a vault, a
//!   roll — entered from **any state** on a declared [`SmParamKind::Trigger`] and
//!   leaving on `exit_time`, at a priority above the gait chain. That is four v2
//!   features an author would otherwise have to know exist.
//!
//! # Determinism
//!
//! Inputs are sorted by name before anything is decided, every map is a
//! `BTreeMap`, and the arithmetic is adds, multiplies and comparisons. Two
//! proposals over the same clip set are the same machine, and the same bytes.

use std::collections::BTreeMap;

use crate::blend_space::{BlendEntry1D, BlendSpace1D, ClipRef};
use crate::channels::als;
use crate::clip::AnimClip;
use crate::derive::gait_of;
use crate::state_machine::{
    BlendCurve, CmpOp, Motion, SmCond, SmParam, SmParamKind, SmState, SmTransition, StateMachine,
};

/// The parameter a proposed machine reads its ground speed from.
///
/// [`crate::locomotion::SPEED_VAR`] by construction, so a proposed machine and a
/// wizard-generated one expect the same variable from gameplay and an actor that
/// drives one drives the other.
pub const SPEED_PARAM: &str = crate::locomotion::SPEED_VAR;

/// The names the four gait tiers are given, in tier order.
pub const TIER_NAMES: [&str; 4] = ["idle", "walk", "run", "sprint"];

/// What a proposal knows about one clip — everything it clusters on, and nothing
/// else.
///
/// Built by [`facts_of`] from a derived clip. A caller that has its own numbers
/// (a re-import, a test) can fill this in directly; the proposal never reaches
/// back into the clip.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipFacts {
    /// The clip's display name — the tie-break for every ordering decision below.
    pub name: String,
    /// Which clip.
    pub clip: ClipRef,
    /// Clip length, seconds.
    pub duration_s: f32,
    /// Depicted ground speed, m/s.
    pub speed_mps: f32,
    /// The 0–3 gait scale ([`crate::derive::gait_of`]).
    pub gait: f32,
    /// How many foot plants the derivation found. **Two or more is a cycle** — a
    /// clip whose feet come down twice is a gait, and one whose feet come down
    /// once or never is a one-shot.
    pub plants: usize,
}

impl ClipFacts {
    /// Whether this clip depicts a repeating gait.
    pub fn is_cycle(&self) -> bool {
        self.plants >= 2
    }

    /// The tier index this clip belongs to: 0 idle, 1 walk, 2 run, 3 sprint.
    pub fn tier(&self) -> usize {
        if !self.gait.is_finite() || self.gait < 0.5 {
            0
        } else if self.gait < 1.5 {
            1
        } else if self.gait < 2.5 {
            2
        } else {
            3
        }
    }
}

/// Read a derived clip's facts back off it.
///
/// Prefers the derived channels ([`crate::derive::MOVE_DATA_SPEED`],
/// [`als::W_GAIT`]) and falls
/// back to the distance track and then to nothing — so a clip that was never
/// derived proposes as an idle rather than as a refusal. The fallback is a
/// **value**: a set with one underived clip in it still proposes.
pub fn facts_of(name: impl Into<String>, clip_ref: ClipRef, clip: &AnimClip) -> ClipFacts {
    let duration_s = clip.duration;
    let distance = clip
        .distance
        .as_ref()
        .and_then(|d| d.distance_m.last().copied())
        .unwrap_or(0.0);
    let speed_mps = if crate::positive(duration_s) {
        distance / duration_s
    } else {
        0.0
    };
    let gait = clip
        .curve(als::W_GAIT)
        .and_then(|c| c.sample(0.0))
        .filter(|v| v.is_finite())
        .unwrap_or_else(|| gait_of(speed_mps, ProposalOptions::default().gait_speeds_mps));
    let plants = clip
        .markers
        .iter()
        .filter(|m| m.is_sync() && m.name.starts_with(crate::derive::PLANT_PREFIX))
        .count();
    ClipFacts {
        name: name.into(),
        clip: clip_ref,
        duration_s,
        speed_mps,
        gait,
        plants,
    }
}

/// The knobs a proposal turns.
#[derive(Clone, Debug, PartialEq)]
pub struct ProposalOptions {
    /// The gait ladder, m/s — the same three [`crate::derive::DeriveOptions`]
    /// maps `W_Gait` against, so a re-derive at different speeds and a re-propose
    /// agree.
    pub gait_speeds_mps: [f32; 3],
    /// Cross-fade duration for a gait-chain transition, seconds.
    ///
    /// The default is [`crate::locomotion`]'s 0.15 s, and it is a *cross-fade
    /// length* rather than a blend mode: the mode is inertialization, which P29.2
    /// made this engine's default for every state transition, so a proposal that
    /// said so explicitly would be restating the engine rather than deciding
    /// anything.
    pub blend_s: f64,
    /// Cross-fade duration for entering and leaving a one-shot, seconds.
    pub oneshot_blend_s: f64,
    /// The fraction of a one-shot that must play before it may leave.
    pub oneshot_exit_time: f64,
    /// The priority a one-shot's any-state entry is given, so it beats the gait
    /// chain when both are ready on the same step.
    pub oneshot_priority: i32,
    /// The speed below which a clip is **standing still**, m/s.
    ///
    /// It exists because "does not cycle" is not the same question as "is a
    /// one-shot", and the first version of this module conflated them: an idle
    /// has no foot plants *because the feet never leave the floor*, and turning
    /// it into a trigger-fired one-shot named `play_idle` is exactly the wrong
    /// answer. A clip that neither cycles nor travels is the base state.
    pub idle_speed_mps: f32,
}

impl Default for ProposalOptions {
    fn default() -> Self {
        Self {
            gait_speeds_mps: [1.65, 3.75, 6.5],
            blend_s: 0.15,
            oneshot_blend_s: 0.1,
            oneshot_exit_time: 0.9,
            oneshot_priority: 10,
            idle_speed_mps: 0.05,
        }
    }
}

/// A proposed machine and the reasoning that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    /// The machine — a normal one, valid, and the author's from here on.
    pub machine: StateMachine,
    /// One line per decision, in the order they were taken. Shown by the panel;
    /// **not** stored in the asset, because a note is a fact about the proposal
    /// and not about the machine.
    pub notes: Vec<String>,
    /// The trigger parameters the one-shots declared, so the caller can tell
    /// gameplay what to arm.
    pub triggers: Vec<String>,
}

/// Why a clip set could not be proposed over. A **value**, like every other
/// refusal in this crate.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ProposeError {
    #[error("a proposal needs at least one clip")]
    NoClips,
    #[error("{count} clips would declare more parameters than a machine may hold")]
    TooManyClips { count: usize },
}

/// **Propose a state machine over a clip set.**
///
/// The whole rule, in the order it is applied — every step of it visible in
/// [`Proposal::notes`] as well.
pub fn propose_machine(
    clips: &[ClipFacts],
    opts: &ProposalOptions,
) -> Result<Proposal, ProposeError> {
    if clips.is_empty() {
        return Err(ProposeError::NoClips);
    }
    if clips.len() > crate::state_machine::MAX_PARAMS {
        return Err(ProposeError::TooManyClips { count: clips.len() });
    }
    // Sorted by name once, here: every decision below reads this order, so two
    // proposals over the same set in different orders are the same machine.
    let mut sorted: Vec<&ClipFacts> = clips.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut notes = Vec::new();
    let mut tiers: BTreeMap<usize, Vec<&ClipFacts>> = BTreeMap::new();
    let mut oneshots: Vec<&ClipFacts> = Vec::new();
    for f in &sorted {
        if f.is_cycle() {
            tiers.entry(f.tier()).or_default().push(f);
        } else if !crate::greater(f.speed_mps, opts.idle_speed_mps) {
            // Neither cycles nor travels: an idle, a breathe, an aim pose. The
            // base state, not a trigger-fired one-shot.
            notes.push(format!(
                "`{}` neither cycles nor travels ({:.3} m/s), so it stands with the idle",
                f.name, f.speed_mps
            ));
            tiers.entry(0).or_default().push(f);
        } else {
            oneshots.push(f);
        }
    }
    // A set with no cycle in it at all still needs somewhere to stand, and the
    // slowest clip is the honest choice — it is what an author would pick.
    if tiers.is_empty() {
        let idle = oneshots.remove(
            oneshots
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.speed_mps.total_cmp(&b.1.speed_mps).then(a.0.cmp(&b.0)))
                .map(|(i, _)| i)
                .unwrap_or(0),
        );
        notes.push(format!(
            "no clip's feet cycle, so `{}` ({:.2} m/s, the slowest) is the base state",
            idle.name, idle.speed_mps
        ));
        tiers.entry(0).or_default().push(idle);
    }

    let mut states: Vec<SmState> = Vec::new();
    let mut tier_index: BTreeMap<usize, usize> = BTreeMap::new();
    let mut tier_speed: BTreeMap<usize, f32> = BTreeMap::new();
    for (tier, members) in &tiers {
        let name = TIER_NAMES[(*tier).min(TIER_NAMES.len() - 1)];
        let mean = members.iter().map(|f| f.speed_mps).sum::<f32>() / members.len() as f32;
        let x = 240.0 * states.len() as f32;
        let motion = if members.len() == 1 {
            notes.push(format!(
                "`{}` depicts {:.2} m/s (gait {:.2}) and is the whole of `{name}`",
                members[0].name, members[0].speed_mps, members[0].gait
            ));
            Motion::Clip(members[0].clip)
        } else {
            let mut entries: Vec<BlendEntry1D> = members
                .iter()
                .map(|f| BlendEntry1D {
                    pos: f.speed_mps as f64,
                    clip: f.clip,
                })
                .collect();
            entries.sort_by(|a, b| a.pos.total_cmp(&b.pos));
            notes.push(format!(
                "`{name}` blends {} clips on `{SPEED_PARAM}` at {} — each clip sits at the speed its own stride depicts, which is a derived number and not a typed one",
                members.len(),
                members
                    .iter()
                    .map(|f| format!("{} = {:.2}", f.name, f.speed_mps))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            Motion::Blend1D(BlendSpace1D {
                param: SPEED_PARAM.to_string(),
                entries,
            })
        };
        tier_index.insert(*tier, states.len());
        tier_speed.insert(*tier, mean);
        states.push(SmState {
            name: name.to_string(),
            motion,
            looping: true,
            speed: 1.0,
            position: (x, 0.0),
            on_enter: Vec::new(),
            on_exit: Vec::new(),
        });
    }

    // The gait chain: one threshold per adjacent pair, at the midpoint of what
    // the two tiers' own clips depict.
    let mut transitions: Vec<SmTransition> = Vec::new();
    let ladder: Vec<usize> = tier_index.keys().copied().collect();
    for pair in ladder.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        let (a, b) = (tier_index[&lo], tier_index[&hi]);
        let at = ((tier_speed[&lo] + tier_speed[&hi]) * 0.5) as f64;
        notes.push(format!(
            "`{}` <-> `{}` at {at:.2} m/s, the midpoint of {:.2} and {:.2} — so each state is used where its own stride is the closer match",
            states[a].name, states[b].name, tier_speed[&lo], tier_speed[&hi]
        ));
        transitions.push(
            SmTransition::on(a, b, opts.blend_s, SPEED_PARAM, CmpOp::Gt, at)
                .with_curve(BlendCurve::EaseInOut),
        );
        transitions.push(
            SmTransition::on(b, a, opts.blend_s, SPEED_PARAM, CmpOp::Le, at)
                .with_curve(BlendCurve::EaseInOut),
        );
    }

    // The one-shots: an any-state entry on a trigger, and an exit-timed return.
    let base = *tier_index.values().next().expect("at least one tier");
    let mut params = vec![SmParam::float(SPEED_PARAM)];
    let mut triggers = Vec::new();
    for f in &oneshots {
        let index = states.len();
        let trigger = trigger_name(&f.name);
        notes.push(format!(
            "`{}` does not cycle ({} foot plant{}), so it is a one-shot entered from ANY state on the trigger `{trigger}` and leaving after {:.0}% of it has played",
            f.name,
            f.plants,
            if f.plants == 1 { "" } else { "s" },
            opts.oneshot_exit_time * 100.0
        ));
        states.push(SmState {
            name: f.name.clone(),
            motion: Motion::Clip(f.clip),
            looping: false,
            speed: 1.0,
            position: (240.0 * (index as f32), 200.0),
            on_enter: Vec::new(),
            on_exit: Vec::new(),
        });
        params.push(SmParam::new(trigger.clone(), SmParamKind::Trigger));
        transitions.push(
            SmTransition::any(index, opts.oneshot_blend_s)
                .when(SmCond::Trigger(trigger.clone()))
                .with_priority(opts.oneshot_priority)
                .with_curve(BlendCurve::EaseIn),
        );
        transitions.push(
            SmTransition::new(index, base, opts.oneshot_blend_s)
                .with_exit_time(opts.oneshot_exit_time)
                .with_curve(BlendCurve::EaseOut),
        );
        triggers.push(trigger);
    }

    let machine = StateMachine {
        states,
        transitions,
        entry: base,
        params,
        profiles: Vec::new(),
    };
    Ok(Proposal {
        machine,
        notes,
        triggers,
    })
}

/// The trigger a one-shot clip is armed with: the clip's name, lowercased, with
/// every run of non-alphanumerics collapsed to one underscore.
///
/// A parameter name crosses to gameplay as a string, so it has to be typeable —
/// and a clip called `Hero Jump (start)` must not produce a parameter nobody can
/// write in a Blueprint node.
pub fn trigger_name(clip_name: &str) -> String {
    let mut out = String::with_capacity(clip_name.len() + 5);
    out.push_str("play_");
    let mut underscore = false;
    for ch in clip_name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            underscore = false;
        } else if !underscore {
            out.push('_');
            underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out == "play" {
        out.push_str("_clip");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::{SmContext, SmRuntime};

    fn facts(name: &str, speed: f32, plants: usize, tiers: [f32; 3]) -> ClipFacts {
        ClipFacts {
            name: name.into(),
            clip: [name.len() as u8; 16],
            duration_s: 1.0,
            speed_mps: speed,
            gait: gait_of(speed, tiers),
            plants,
        }
    }

    fn locomotion_set() -> Vec<ClipFacts> {
        let t = ProposalOptions::default().gait_speeds_mps;
        vec![
            facts("Idle", 0.0, 0, t),
            facts("Walk", 1.6, 2, t),
            facts("Jog", 3.2, 2, t),
            facts("Run", 4.0, 2, t),
            facts("Jump", 2.0, 1, t),
        ]
    }

    /// **The arm the brief asks for**: a proposal VALIDATES and RUNS. Not that it
    /// is good — nobody can assert that — but that the author who opens it opens
    /// a machine, not a shape the reader refuses.
    #[test]
    fn a_proposal_validates_and_runs_a_fixed_step_smoke() {
        let p = propose_machine(&locomotion_set(), &ProposalOptions::default()).unwrap();
        p.machine.validate().expect("a proposal must validate");

        // …and it runs. Drive the speed parameter up the ladder and back down at
        // a fixed step, and assert it visits every gait state it proposed.
        let mut rt = SmRuntime::default();
        let mut visited = std::collections::BTreeSet::new();
        let schedule = [0.0f64, 1.6, 1.6, 4.0, 4.0, 4.0, 1.6, 0.0, 0.0];
        for speed in schedule {
            for _ in 0..30 {
                let vars = |name: &str| (name == SPEED_PARAM).then_some(speed);
                let ctx = SmContext::new(&vars);
                rt.advance(&p.machine, &ctx, 1.0 / 60.0);
                visited.insert(p.machine.states[rt.current].name.clone());
            }
        }
        let gaits: std::collections::BTreeSet<String> = p
            .machine
            .states
            .iter()
            .filter(|s| s.looping)
            .map(|s| s.name.clone())
            .collect();
        assert_eq!(visited, gaits, "a proposed gait chain must be walkable");
    }

    /// The one-shot half: a declared trigger, an any-state entry above the gait
    /// chain's priority, and an exit-timed return.
    #[test]
    fn a_one_shot_is_entered_from_anywhere_on_its_trigger_and_leaves_on_exit_time() {
        let p = propose_machine(&locomotion_set(), &ProposalOptions::default()).unwrap();
        assert_eq!(p.triggers, vec!["play_jump".to_string()]);
        let jump = p
            .machine
            .states
            .iter()
            .position(|s| s.name == "Jump")
            .expect("a Jump state");
        assert!(!p.machine.states[jump].looping);
        let entry = p
            .machine
            .transitions
            .iter()
            .find(|t| t.to == jump)
            .expect("an entry edge");
        assert!(matches!(
            entry.from,
            crate::state_machine::SmSource::Any { exclude_self: true }
        ));
        assert!(entry.priority > 0);
        let exit = p
            .machine
            .transitions
            .iter()
            .find(|t| t.from == crate::state_machine::SmSource::State(jump))
            .expect("an exit edge");
        assert_eq!(exit.exit_time, Some(0.9));

        // The trigger really drives it: armed, the machine leaves the gait chain
        // for the one-shot; unarmed, it never does.
        let mut rt = SmRuntime::default();
        let vars = |name: &str| (name == SPEED_PARAM).then_some(1.6f64);
        for _ in 0..40 {
            let ctx = SmContext::new(&vars);
            rt.advance(&p.machine, &ctx, 1.0 / 60.0);
        }
        assert_ne!(rt.current, jump, "no trigger, no jump");
        rt.arm_trigger(&p.machine, "play_jump");
        let ctx = SmContext::new(&vars);
        rt.advance(&p.machine, &ctx, 1.0 / 60.0);
        assert_eq!(rt.current, jump, "the trigger did not fire the one-shot");
    }

    /// Two clips in one tier become a **blend space** on the derived speeds, in
    /// ascending order — the case an author gets wrong because the axis
    /// positions are measurements.
    #[test]
    fn a_tier_with_two_clips_becomes_a_blend_space_on_their_own_speeds() {
        let p = propose_machine(&locomotion_set(), &ProposalOptions::default()).unwrap();
        let run = p
            .machine
            .states
            .iter()
            .find(|s| s.name == "run")
            .expect("a run tier");
        let Motion::Blend1D(bs) = &run.motion else {
            panic!("expected a blend space, got {:?}", run.motion);
        };
        assert_eq!(bs.param, SPEED_PARAM);
        assert_eq!(bs.entries.len(), 2);
        assert!(bs.entries[0].pos < bs.entries[1].pos);
        assert!((bs.entries[0].pos - 3.2).abs() < 1e-4);
    }

    /// The proposal explains itself. Every threshold it chose is in the notes
    /// with the numbers it came from — this is the half that makes it an
    /// authoring aid rather than an oracle.
    #[test]
    fn every_threshold_is_explained_with_the_numbers_it_came_from() {
        let p = propose_machine(&locomotion_set(), &ProposalOptions::default()).unwrap();
        let all = p.notes.join("\n");
        assert!(all.contains("midpoint"), "{all}");
        assert!(all.contains("Jump"), "{all}");
        assert!(all.contains("blends 2 clips"), "{all}");
        // The chain's threshold really is the midpoint of the two tiers' means:
        // walk 1.6 and run (3.2 + 4.0)/2 = 3.6 → 2.60.
        assert!(all.contains("2.60 m/s"), "{all}");
    }

    /// The proposal is a **function of the set**, not of the order it arrives in.
    #[test]
    fn the_same_set_in_a_different_order_proposes_the_same_machine() {
        let mut a = locomotion_set();
        let p1 = propose_machine(&a, &ProposalOptions::default()).unwrap();
        a.reverse();
        let p2 = propose_machine(&a, &ProposalOptions::default()).unwrap();
        assert_eq!(p1, p2);
    }

    /// A set with nothing cyclic in it still proposes something standable.
    #[test]
    fn a_set_with_no_cycle_still_gets_a_base_state() {
        let t = ProposalOptions::default().gait_speeds_mps;
        let p = propose_machine(
            &[facts("Vault", 3.0, 1, t), facts("Aim", 0.0, 0, t)],
            &ProposalOptions::default(),
        )
        .unwrap();
        p.machine.validate().unwrap();
        let base = &p.machine.states[p.machine.entry];
        assert_eq!(base.name, TIER_NAMES[0]);
        assert_eq!(base.motion, Motion::Clip([3u8; 16]), "the base plays `Aim`");
        assert!(base.looping, "an idle loops; it is not a one-shot");
        assert_eq!(p.triggers, vec!["play_vault".to_string()]);
        assert_eq!(
            propose_machine(&[], &ProposalOptions::default()),
            Err(ProposeError::NoClips)
        );
    }

    /// A clip name a Blueprint could not type becomes a parameter it can.
    #[test]
    fn a_trigger_name_is_typeable() {
        assert_eq!(trigger_name("Hero Jump (start)"), "play_hero_jump_start");
        assert_eq!(trigger_name("Roll"), "play_roll");
        assert_eq!(trigger_name("!!!"), "play_clip");
    }

    /// Facts are read back off a **derived** clip, which is the seam between S2
    /// and S3: derivation measures, the proposal decides.
    #[test]
    fn the_facts_come_off_the_derived_clip() {
        let sk = crate::template::build_template(
            crate::template::BodyPlan::Biped,
            &crate::template::BodyParams::default(),
        )
        .unwrap();
        let set = crate::locomotion::build_locomotion(
            crate::template::BodyPlan::Biped,
            &sk,
            &crate::locomotion::GaitParams::default(),
        )
        .unwrap();
        let (walk, report) = crate::derive::derive_clip(
            &set.walk,
            &sk.skeleton,
            &crate::derive::DeriveOptions::default(),
        )
        .unwrap();
        let f = facts_of("Walk", [1u8; 16], &walk);
        assert_eq!(f.plants, report.plants.len());
        assert!(f.is_cycle(), "a generated walk cycles: {f:?}");
        assert!((f.gait - report.gait).abs() < 1e-6);
        // An UNDERIVED clip is still a fact, just a dull one — no refusal.
        let bare = facts_of("Bare", [2u8; 16], &set.walk);
        assert_eq!(bare.plants, 2, "the generator's own markers still count");
        assert_eq!(bare.speed_mps, 0.0, "an unbaked clip depicts no travel");
    }
}
