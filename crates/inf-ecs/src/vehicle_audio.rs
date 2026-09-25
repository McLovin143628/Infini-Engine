//! **What a car sounds like, as a pure function of sim state** (wave VEH3e).
//!
//! Since VEH1a a car has been ONE looping clip pitched by its revs
//! ([`crate::vehicle::engine_cue`]). This module replaces that with the research
//! doc's layer stack — combustion grains crossfaded by load, gear whine, turbo
//! whine and blow-off, tyre squeal by slip per surface, surface impulses on
//! suspension spikes — and the door and body sounds boarding makes, all as ONE
//! Ring-0 planner both hosts call inside the `vehicle_engine_audio` MIRROR fence.
//! What a host does with the answer is a `match` from [`VoiceCue`] onto the
//! P12.3 queue.
//!
//! # The contract is the command stream
//!
//! [`VoiceMemory::plan`] is a pure function of this step's sim state and of what
//! it has already told the queue: the memory holds what each voice was last
//! SET to (so an unchanged value is not re-sent), the previous step's strut
//! compressions (a kerb is a velocity), the previous throttle (a blow-off is an
//! edge) and each boarding body's previous phase and hinge angle (a latch and a
//! slam are edges). Both hosts start a session with an empty memory and feed it
//! the same outcomes, so they emit the same stream; nothing reads a clock and
//! nothing draws a random number — a grain's phase is the device's playback of
//! a baked loop, and the loop's firing period is in its bytes.
//!
//! # Who sings
//!
//! A car with an [`AudioSource`] whose engine is RUNNING: somebody is in it, or
//! its drivetrain is not quiet. A parked car nobody is in is silent (its loops
//! are stopped), which is also what keeps an island of parked cars out of the
//! audio log.
//!
//! **Two tiers** (VEH3e audit). A car the level authored — the hero's, a
//! fleet's — sings the FULL stack. A TRAFFIC car (a record of
//! [`crate::traffic::TrafficPopulationRes`]) gets an emitter only while its tier
//! is `Full` (a real rig inside `TRAFFIC_FULL_M` with an AI driver: the cars
//! near the hero) and sings the NEAR stack: ONE grain (the half-load one,
//! pitched by its revs and voiced by its level), ONE squeal (its worse
//! axle's slip) and its rolling road, each loop re-told at most every
//! [`NEAR_EVERY`] steps on a phase spread by its key. A crossing's traffic is heard from the kerb; a
//! queue of it does not flood the log.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use uuid::Uuid;

use crate::boarding::{BoardPhase, DOOR_SHUT_DEG};
use crate::components::{AudioSource, CharacterMovement, GlobalTransform, MovementMode, Transform};
use crate::vehicle::SurfaceClass;
use crate::EcsWorld;

// ── the clips ───────────────────────────────────────────────────────────────

/// **The GUID prefix every vehicle clip carries** — `"VEH3"` in ASCII, as the
/// gunshot library carries `"WPN2"`, so a GUID in a log says where it came from.
pub const VEHICLE_CLIP_BASE: u128 = 0x5645_4833_0000_0000;

/// **The rpm every combustion grain is baked at** — `inf_audio::vehicle_synth::
/// GRAIN_REF_RPM`, restated because this crate cannot name that one;
/// `veh3e_gate` asserts the two agree.
pub const GRAIN_REF_RPM: f64 = 2_400.0;

/// **The input-shaft rpm at which the whine plays at a pitch of one.**
pub const WHINE_REF_RPM: f64 = 3_000.0;

/// **Every vehicle clip, by index** — `inf_audio::vehicle_synth::
/// VEHICLE_CLIP_NAMES` restated, and asserted equal by `veh3e_gate`.
pub const VEHICLE_CLIP_NAMES: [&str; 36] = [
    "Grain_P4_Idle",
    "Grain_P4_Mid",
    "Grain_P4_Full",
    "Grain_P6_Idle",
    "Grain_P6_Mid",
    "Grain_P6_Full",
    "Grain_P8X_Idle",
    "Grain_P8X_Mid",
    "Grain_P8X_Full",
    "Grain_P8F_Idle",
    "Grain_P8F_Mid",
    "Grain_P8F_Full",
    "Grain_D6_Idle",
    "Grain_D6_Mid",
    "Grain_D6_Full",
    "Whine",
    "Turbo",
    "BlowOff",
    "Squeal_Asphalt",
    "Squeal_Gravel",
    "Squeal_Soft",
    "Impulse_Kerb",
    "Impulse_Gravel",
    "Impulse_Soft",
    "Door_Latch",
    "Door_Creak",
    "Door_Slam",
    "Body_Thud",
    "Roll_Asphalt",
    "Roll_Gravel",
    "Roll_Soft",
    // ── the craft (wave VEH3g) -- appended.
    "Rotor_BladePass",
    "Prop_BladePass",
    "Jet_Spool",
    "Hull_Slap",
    "Hull_Spray",
];

/// **A rotor's blade-pass at a pitch of one**, hertz --
/// `inf_audio::vehicle_synth::ROTOR_REF_HZ` restated (wave VEH3g).
pub const ROTOR_REF_HZ: f64 = 15.0;

/// **A propeller's blade-pass at a pitch of one**, hertz --
/// `inf_audio::vehicle_synth::PROP_REF_HZ` restated (wave VEH3g).
pub const PROP_REF_HZ: f64 = 90.0;

/// The clip at `index` into [`VEHICLE_CLIP_NAMES`].
pub fn vehicle_clip(index: u8) -> Uuid {
    Uuid::from_u128(VEHICLE_CLIP_BASE | u128::from(index))
}

/// Every vehicle clip GUID, in index order — what
/// [`crate::audio::engine_spawned_clips`] closes a pack over.
pub fn vehicle_clips() -> Vec<Uuid> {
    (0..VEHICLE_CLIP_NAMES.len() as u8)
        .map(vehicle_clip)
        .collect()
}

/// **Which grain set an engine sings from.**
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GrainFamily {
    /// An inline four.
    P4,
    /// A six.
    P6,
    /// A cross-plane V8 — the burble.
    P8Cross,
    /// A flat-plane V8 — the wail.
    P8Flat,
    /// A six-cylinder diesel.
    Diesel6,
}

impl GrainFamily {
    /// Every family, in clip order.
    pub const ALL: [GrainFamily; 5] = [
        GrainFamily::P4,
        GrainFamily::P6,
        GrainFamily::P8Cross,
        GrainFamily::P8Flat,
        GrainFamily::Diesel6,
    ];

    /// Its index into [`Self::ALL`].
    pub fn index(self) -> u8 {
        self as u8
    }

    /// How many cylinders its clip was baked firing.
    pub fn cylinders(self) -> f64 {
        match self {
            GrainFamily::P4 => 4.0,
            GrainFamily::P6 | GrainFamily::Diesel6 => 6.0,
            GrainFamily::P8Cross | GrainFamily::P8Flat => 8.0,
        }
    }

    /// **The family an engine's three v28 tunables choose**, or `None` for an
    /// engine with no combustion events — a turbine, a motor, a propeller —
    /// which keeps VEH1a's single pitched loop (VEH3g's voices).
    ///
    /// `voice_kind` `1` is a diesel whatever its count; otherwise the count
    /// picks the nearest family, and `firing_order` `2` makes an eight flat-plane.
    pub fn for_engine(cylinders: f64, voice_kind: f64, firing_order: f64) -> Option<Self> {
        if !cylinders.is_finite() || cylinders < 1.0 {
            return None;
        }
        let kind = voice_kind.round();
        if kind == 1.0 {
            return Some(GrainFamily::Diesel6);
        }
        if kind != 0.0 {
            return None;
        }
        Some(if cylinders <= 4.5 {
            GrainFamily::P4
        } else if cylinders <= 6.5 {
            GrainFamily::P6
        } else if firing_order.round() == 2.0 {
            GrainFamily::P8Flat
        } else {
            GrainFamily::P8Cross
        })
    }
}

/// **The doc's three loads**: 0 %, 50 % and 100 % throttle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EngineLoad {
    /// No load — a coasting or idling engine.
    Idle,
    /// Half.
    Mid,
    /// Full.
    Full,
}

impl EngineLoad {
    /// Every load, in clip order.
    pub const ALL: [EngineLoad; 3] = [EngineLoad::Idle, EngineLoad::Mid, EngineLoad::Full];

    /// Its index into [`Self::ALL`].
    pub fn index(self) -> u8 {
        self as u8
    }
}

/// **What a tyre sounds like on a surface**: a sealed one squeals, a loose one
/// hisses, a soft one scrubs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceVoice {
    /// Asphalt and concrete.
    #[default]
    Sealed,
    /// Gravel and sand.
    Loose,
    /// Grass and mud.
    Soft,
}

impl SurfaceVoice {
    /// The voice of a [`SurfaceClass`].
    pub fn of(surface: SurfaceClass) -> Self {
        match surface {
            SurfaceClass::Asphalt | SurfaceClass::Concrete => SurfaceVoice::Sealed,
            SurfaceClass::Gravel | SurfaceClass::Sand => SurfaceVoice::Loose,
            SurfaceClass::Grass | SurfaceClass::Mud => SurfaceVoice::Soft,
        }
    }

    /// Its index: `0` sealed, `1` loose, `2` soft.
    pub fn index(self) -> u8 {
        self as u8
    }

    /// Its name, for a readout.
    pub fn name(self) -> &'static str {
        match self {
            SurfaceVoice::Sealed => "sealed",
            SurfaceVoice::Loose => "loose",
            SurfaceVoice::Soft => "soft",
        }
    }
}

/// The grain a family sings at a load.
pub fn grain_clip(family: GrainFamily, load: EngineLoad) -> Uuid {
    vehicle_clip(family.index() * 3 + load.index())
}

/// The gear whine's loop.
pub fn whine_clip() -> Uuid {
    vehicle_clip(15)
}

/// The turbo's whistle.
pub fn turbo_clip() -> Uuid {
    vehicle_clip(16)
}

/// The blow-off valve's one-shot.
pub fn blow_off_clip() -> Uuid {
    vehicle_clip(17)
}

/// A sliding tyre's loop on a surface.
pub fn squeal_clip(voice: SurfaceVoice) -> Uuid {
    vehicle_clip(18 + voice.index())
}

/// A ROLLING tyre's loop on a surface (VEH3e audit).
pub fn roll_clip(voice: SurfaceVoice) -> Uuid {
    vehicle_clip(28 + voice.index())
}

/// A suspension spike's one-shot on a surface.
pub fn impulse_clip(voice: SurfaceVoice) -> Uuid {
    vehicle_clip(21 + voice.index())
}

/// A door or body sound.
pub fn door_clip(layer: DoorLayer) -> Uuid {
    vehicle_clip(24 + layer as u8)
}

/// A rotor's blade-pass loop (wave VEH3g).
pub fn rotor_clip() -> Uuid {
    vehicle_clip(31)
}

/// A propeller's blade-pass loop (wave VEH3g).
pub fn prop_clip() -> Uuid {
    vehicle_clip(32)
}

/// A turbine's spool loop (wave VEH3g).
pub fn jet_clip() -> Uuid {
    vehicle_clip(33)
}

/// A hull's slap on the water (wave VEH3g).
pub fn hull_slap_clip() -> Uuid {
    vehicle_clip(34)
}

/// A planing hull's spray (wave VEH3g).
pub fn hull_spray_clip() -> Uuid {
    vehicle_clip(35)
}

// ── the keys ────────────────────────────────────────────────────────────────

/// **The layers one car sings on** — one voice per layer, one key per voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VoiceLayer {
    /// The no-load grain.
    GrainIdle,
    /// The half-load grain.
    GrainMid,
    /// The full-load grain.
    GrainFull,
    /// The gearbox.
    Whine,
    /// The turbocharger.
    Turbo,
    /// The front axle's tyres.
    SquealFront,
    /// The rear axle's.
    SquealRear,
    /// The blow-off valve (a one-shot).
    BlowOff,
    /// A front-axle strut spike (a one-shot).
    ImpulseFront,
    /// A rear-axle one.
    ImpulseRear,
    /// The tyres ROLLING on the road (VEH3e audit) — appended, so every key
    /// above keeps its salt.
    Roll,
    /// A rotor's blade-pass (wave VEH3g) -- appended, as are the four below.
    Rotor,
    /// A propeller's blade-pass.
    Prop,
    /// A turbine's spool.
    Jet,
    /// A hull's slap on the water.
    HullSlap,
    /// A planing hull's spray.
    HullSpray,
}

impl VoiceLayer {
    /// Every layer, in the order the planner addresses them.
    pub const ALL: [VoiceLayer; 16] = [
        VoiceLayer::GrainIdle,
        VoiceLayer::GrainMid,
        VoiceLayer::GrainFull,
        VoiceLayer::Whine,
        VoiceLayer::Turbo,
        VoiceLayer::SquealFront,
        VoiceLayer::SquealRear,
        VoiceLayer::BlowOff,
        VoiceLayer::ImpulseFront,
        VoiceLayer::ImpulseRear,
        VoiceLayer::Roll,
        VoiceLayer::Rotor,
        VoiceLayer::Prop,
        VoiceLayer::Jet,
        VoiceLayer::HullSlap,
        VoiceLayer::HullSpray,
    ];

    /// Its index into [`Self::ALL`].
    pub fn index(self) -> usize {
        self as usize
    }
}

/// **The salts a car's source keys are spread over** — `crate::weapon::
/// LAYER_SALTS`' construction and its reason: eleven voices need eleven keys
/// (the rolling road's the eleventh, VEH3e audit), none of which may be the
/// chassis's own emitter key.
pub const VOICE_SALTS: [u64; 16] = [
    0x5645_4833_0000_0001,
    0x5645_4833_0000_0002,
    0x5645_4833_0000_0003,
    0x5645_4833_0000_0004,
    0x5645_4833_0000_0005,
    0x5645_4833_0000_0006,
    0x5645_4833_0000_0007,
    0x5645_4833_0000_0008,
    0x5645_4833_0000_0009,
    0x5645_4833_0000_000a,
    0x5645_4833_0000_000b,
    // The craft (wave VEH3g) -- five WIDELY SPREAD salts, not five in a row.
    // A key is `guid ^ salt`, so two voices of two craft collide whenever the
    // two guids differ by what the two salts differ by -- and fixtures mint
    // guids a few apart. Measured twice: `…000c..0010` put a car's hull spray
    // on the VEH3e course hero's creak (guids two apart), and `…0001_0001..5`
    // put a helicopter's rotor on a jet's spool (guids two apart, salts two
    // apart). These differ from each other, and from every salt above, in
    // dozens of bits.
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
    0x85EB_CA77_C2B2_AE63,
    0x27D4_EB2F_1656_67C5,
];

/// The key one layer of one car plays on.
pub fn voice_key(chassis_key: u64, layer: VoiceLayer) -> u64 {
    chassis_key ^ VOICE_SALTS[layer.index()]
}

/// **The door and body sounds, by layer.**
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DoorLayer {
    /// The latch let go — the outer handle pulled, or the inner one pushed.
    Latch,
    /// The hinge moving (a loop, voiced by the hinge's angular rate).
    Creak,
    /// The door arriving shut.
    Slam,
    /// A body landing out of a moving car.
    Thud,
}

/// The salts a body's door keys are spread over.
pub const DOOR_SALTS: [u64; 4] = [
    0x5645_4833_0000_0011,
    0x5645_4833_0000_0012,
    0x5645_4833_0000_0013,
    0x5645_4833_0000_0014,
];

/// The key one door layer of one body plays on.
pub fn door_key(body_key: u64, layer: DoorLayer) -> u64 {
    body_key ^ DOOR_SALTS[layer as usize]
}

// ── the telemetry ───────────────────────────────────────────────────────────

/// **One axle's tyres, as the ear hears them.**
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AxleVoice {
    /// The largest NORMALISED slip over the axle's grounded wheels:
    /// `hypot(slip_ratio / long_peak, slip_lat / lat_peak)`, so `1` is the peak
    /// of the tyre curve on either axis. Zero with nothing on the ground.
    pub slip: f64,
    /// The surface under the wheel that slip was measured on.
    pub surface: SurfaceClass,
    /// The largest strut compression on the axle, metres (`0` with nothing on
    /// the ground) — what a kerb is read off, as a VELOCITY between steps.
    pub compression_m: f64,
    /// Whether any wheel on the axle has a contact.
    pub grounded: bool,
}

/// **Everything a car's voice is a function of**, published by the class once
/// per fixed step on `inf_physics::d3::VehicleOutcome` — never read back off
/// the model by a later phase.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceTelemetry {
    /// Crank speed, rpm.
    pub rpm: f64,
    /// The class's idle, rpm.
    pub idle_rpm: f64,
    /// The class's redline, rpm.
    pub redline_rpm: f64,
    /// The throttle this step was solved with, `[0, 1]`.
    pub throttle: f64,
    /// The turbo's boost, `[0, 1]` of the class's peak.
    pub boost: f64,
    /// Whether the class carries a turbo at all.
    pub turbocharged: bool,
    /// Whether the limiter is cutting fuel this step.
    pub fuel_cut: bool,
    /// The gear engaged.
    pub gear: i32,
    /// The gearbox INPUT shaft's speed as the driven wheels imply it, rpm:
    /// their mean `|ω|` times this gear's total ratio — the doc's "wheel speed ×
    /// gear ratio". A shift steps it; a clutch that slips does not move it.
    pub shaft_rpm: f64,
    /// The three v28 voice tunables.
    pub cylinders: f64,
    /// See [`GrainFamily::for_engine`].
    pub voice_kind: f64,
    /// See [`GrainFamily::for_engine`].
    pub firing_order: f64,
    /// Whether somebody commanded the car this step.
    pub occupied: bool,
    /// Whether the drivetrain is quiet (`DrivetrainState::is_quiet`).
    pub quiet: bool,
    /// Forward speed, m/s — PUBLISHED so an arm can show the squeal does NOT
    /// read it.
    pub speed_mps: f64,
    /// Front axle, then rear.
    pub axles: [AxleVoice; 2],
    /// **What a CRAFT sings** (wave VEH3g) -- a rotor, a propeller, a turbine,
    /// a hull -- all zero for a car.
    pub craft: CraftVoice,
}

/// **A craft's voice** (wave VEH3g): the rotor / jet / hull voice kinds on the
/// VEH3e planner's own shape -- each a function of what the class published
/// this step, never of a clock.
///
/// | layer | voiced by | pitched by |
/// |---|---|---|
/// | rotor | the collective | blade-pass: `rpm x blades / 60` over [`ROTOR_REF_HZ`] |
/// | propeller | the throttle | blade-pass over [`PROP_REF_HZ`] |
/// | turbine | the spool | the spool |
/// | hull slap | speed through the water, fading as the hull planes | speed |
/// | hull spray | the planing share | fixed |
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CraftVoice {
    /// A rotor's blade-pass frequency, hertz; `0` for no rotor.
    pub rotor_hz: f64,
    /// A propeller's blade-pass frequency, hertz; `0` for no propeller.
    pub prop_hz: f64,
    /// Whether a turbine turns.
    pub jet: bool,
    /// The turbine's spool, `[0, 1]`.
    pub jet_spool: f64,
    /// How hard the rotor or the propeller is working, `[0, 1]`.
    pub load: f64,
    /// Whether a hull is in the water.
    pub hull_wet: bool,
    /// Speed through the water, m/s.
    pub hull_speed_mps: f64,
    /// The share of the weight the planing lift carries, `[0, 1]`.
    pub planing: f64,
    /// **The craft has no wheels** -- a hull or a rotorcraft -- so its stack
    /// carries no gearbox whine, no squeal and no rolling road: four voices a
    /// car's stack would hold silent for ever on a boat.
    pub wheelless: bool,
}

impl CraftVoice {
    /// Whether there is anything to sing.
    pub fn any(&self) -> bool {
        self.rotor_hz > 0.0 || self.prop_hz > 0.0 || self.jet || self.hull_wet
    }
}

/// How loud a rotor is at full collective.
pub const ROTOR_GAIN: f64 = 0.6;
/// How loud a propeller is at full throttle.
pub const PROP_GAIN: f64 = 0.45;
/// How loud a turbine is at full spool.
pub const JET_GAIN: f64 = 0.5;
/// How loud a hull slaps at speed.
pub const HULL_SLAP_GAIN: f64 = 0.4;
/// How loud a planing hull's spray is.
pub const HULL_SPRAY_GAIN: f64 = 0.35;

/// **A craft's loops** this step: (layer, clip, volume, pitch), before the
/// emitter's own volume -- the rotor, propeller, turbine and hull voice kinds.
pub fn craft_voices(c: &CraftVoice) -> Vec<(VoiceLayer, Uuid, f64, f64)> {
    let mut out = Vec::with_capacity(4);
    let load = finite01(c.load);
    if c.rotor_hz > 0.0 && c.rotor_hz.is_finite() {
        out.push((
            VoiceLayer::Rotor,
            rotor_clip(),
            ROTOR_GAIN * (0.4 + 0.6 * load),
            (c.rotor_hz / ROTOR_REF_HZ).clamp(PITCH_MIN, PITCH_MAX),
        ));
    }
    if c.prop_hz > 0.0 && c.prop_hz.is_finite() {
        out.push((
            VoiceLayer::Prop,
            prop_clip(),
            PROP_GAIN * (0.35 + 0.65 * load),
            (c.prop_hz / PROP_REF_HZ).clamp(PITCH_MIN, PITCH_MAX),
        ));
    }
    if c.jet {
        let sp = finite01(c.jet_spool);
        out.push((
            VoiceLayer::Jet,
            jet_clip(),
            JET_GAIN * (0.2 + 0.8 * sp),
            0.5 + 0.7 * sp,
        ));
    }
    if c.hull_wet {
        let v = if c.hull_speed_mps.is_finite() {
            c.hull_speed_mps.abs()
        } else {
            0.0
        };
        let plane = finite01(c.planing);
        out.push((
            VoiceLayer::HullSlap,
            hull_slap_clip(),
            HULL_SLAP_GAIN * (v / 10.0).clamp(0.0, 1.0) * (1.0 - 0.6 * plane),
            0.8 + 0.4 * (v / 20.0).clamp(0.0, 1.0),
        ));
        out.push((
            VoiceLayer::HullSpray,
            hull_spray_clip(),
            HULL_SPRAY_GAIN * plane * (v / 25.0).clamp(0.0, 1.0),
            1.0,
        ));
    }
    out
}

impl VoiceTelemetry {
    /// Revs as `[0, 1]` between idle and the redline.
    pub fn revs(&self) -> f64 {
        let span = (self.redline_rpm - self.idle_rpm).max(1.0);
        ((self.rpm - self.idle_rpm) / span).clamp(0.0, 1.0)
    }

    /// **The load the grains are crossfaded by**: the throttle, and nothing
    /// while the limiter cuts fuel — so a limiter bounce is heard as the
    /// full-load grain dropping out, which is what a fuel cut is.
    pub fn load(&self) -> f64 {
        if self.fuel_cut {
            0.0
        } else {
            finite01(self.throttle)
        }
    }

    /// Whether the engine is running: somebody is in the car, or its
    /// drivetrain has something to say.
    pub fn running(&self) -> bool {
        self.occupied || !self.quiet
    }
}

fn finite01(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ── the per-layer functions (each one an arm) ───────────────────────────────

/// **The load crossfade** — the three grains' shares of the engine at `load`:
/// triangles centred on 0, ½ and 1 that always sum to one.
pub fn load_weights(load: f64) -> [f64; 3] {
    let l = finite01(load);
    [
        (1.0 - 2.0 * l).max(0.0),
        1.0 - (2.0 * l - 1.0).abs(),
        (2.0 * l - 1.0).max(0.0),
    ]
}

/// How loud the whole engine is before the crossfade: louder with revs and
/// with load.
///
/// The layer gains are set for HEADROOM, measured through the render-to-file
/// door: with the listener in the driver's seat the first set (engine 0.8,
/// squeal 0.8, turbo 0.3, whine 0.16) summed past full scale under a full-
/// throttle wheelspin and clipped 1 723 of 796 800 samples of the course; this
/// set is the one the capture was re-measured at.
pub const ENGINE_GAIN: f64 = 0.55;

/// The engine's level at this step, before the emitter's own volume.
pub fn engine_level(t: &VoiceTelemetry) -> f64 {
    ENGINE_GAIN * (0.55 + 0.45 * t.revs()) * (0.7 + 0.3 * t.load())
}

/// **The grain's playback rate**: `rpm / GRAIN_REF_RPM`, times the cylinder
/// count over the family's own — so any count fires at `(rpm / 60)·(cyl / 2)`.
pub fn grain_pitch(t: &VoiceTelemetry, family: GrainFamily) -> f64 {
    let cyl = if t.cylinders.is_finite() && t.cylinders > 0.0 {
        t.cylinders
    } else {
        family.cylinders()
    };
    let rpm = if t.rpm.is_finite() {
        t.rpm.max(0.0)
    } else {
        0.0
    };
    (rpm / GRAIN_REF_RPM * cyl / family.cylinders()).clamp(PITCH_MIN, PITCH_MAX)
}

/// The lowest and highest playback rate any vehicle voice is asked for.
pub const PITCH_MIN: f64 = 0.05;
/// See [`PITCH_MIN`].
pub const PITCH_MAX: f64 = 4.0;

/// How loud the whine gets.
pub const WHINE_GAIN: f64 = 0.12;

/// **The gear whine**: pitched by the input shaft (`shaft_rpm /
/// WHINE_REF_RPM`), louder with shaft speed and with load.
pub fn whine_voice(t: &VoiceTelemetry) -> (f64, f64) {
    let shaft = if t.shaft_rpm.is_finite() {
        t.shaft_rpm.max(0.0)
    } else {
        0.0
    };
    let vol = WHINE_GAIN * (shaft / 5_000.0).clamp(0.0, 1.0) * (0.5 + 0.5 * t.load());
    let pitch = (shaft / WHINE_REF_RPM).clamp(PITCH_MIN, PITCH_MAX);
    (vol, pitch)
}

/// How loud the turbo gets at full boost.
pub const TURBO_GAIN: f64 = 0.2;

/// **The turbo's whistle**: pitched and voiced by boost.
pub fn turbo_voice(t: &VoiceTelemetry) -> (f64, f64) {
    let b = finite01(t.boost);
    (TURBO_GAIN * b * b, 0.6 + 1.2 * b)
}

/// The throttle a blow-off needs to have come from, and to have fallen to.
pub const BLOW_OFF_FROM: f64 = 0.5;
/// See [`BLOW_OFF_FROM`].
pub const BLOW_OFF_TO: f64 = 0.2;
/// The boost there has to be to dump.
pub const BLOW_OFF_MIN_BOOST: f64 = 0.3;
/// How loud a blow-off at full boost is.
pub const BLOW_OFF_GAIN: f64 = 0.45;

/// Where the squeal starts, in normalised slip — a little before the tyre
/// curve's peak, because a tyre at its limit is already audible.
pub const SQUEAL_ONSET: f64 = 0.8;
/// Where it is at full voice.
pub const SQUEAL_FULL: f64 = 2.2;
/// How loud a tyre at full voice is.
pub const SQUEAL_GAIN: f64 = 0.5;

/// **THE SQUEAL, BY SLIP AND NOT BY SPEED** (the research doc: *"a car sliding
/// sideways at 10 km/h should squeal just as hard as one sliding at 100
/// km/h"*): a smoothstep in normalised slip for the volume, a gentle rise for
/// the pitch. Speed is not an argument, so it cannot be read.
pub fn squeal_voice(slip: f64) -> (f64, f64) {
    let s = if slip.is_finite() { slip.max(0.0) } else { 0.0 };
    let x = ((s - SQUEAL_ONSET) / (SQUEAL_FULL - SQUEAL_ONSET)).clamp(0.0, 1.0);
    let vol = SQUEAL_GAIN * x * x * (3.0 - 2.0 * x);
    let pitch = 0.9 + 0.25 * ((s - SQUEAL_ONSET) / 4.0).clamp(0.0, 1.0);
    (vol, pitch)
}

/// **The rolling road's full voice**, m/s — 108 km/h (VEH3e audit).
pub const ROLL_FULL_MPS: f64 = 30.0;
/// How loud the rolling road is at [`ROLL_FULL_MPS`].
pub const ROLL_GAIN: f64 = 0.35;
/// Below this a rolling tyre is silent, m/s.
pub const ROLL_ONSET_MPS: f64 = 0.5;

/// **THE ROLLING ROAD** (VEH3e audit, the research doc's "continuous surface
/// roll noise"): voiced by SPEED — rising faster than linear, as tyre noise
/// does — and silent with no wheel on the ground; pitched a little higher as
/// the tread blocks come round faster. The SURFACE picks the clip
/// ([`roll_clip`]), so a car that leaves the tarmac for gravel changes its
/// roll, not its level.
pub fn roll_voice(t: &VoiceTelemetry) -> (f64, f64) {
    let speed = if t.speed_mps.is_finite() {
        t.speed_mps.abs()
    } else {
        0.0
    };
    let grounded = t.axles.iter().any(|a| a.grounded);
    let x = (speed / ROLL_FULL_MPS).clamp(0.0, 1.0);
    let vol = if grounded && speed >= ROLL_ONSET_MPS {
        ROLL_GAIN * x * (0.35 + 0.65 * x)
    } else {
        0.0
    };
    (vol, 0.8 + 0.45 * x)
}

/// The strut closing speed an impulse needs, m/s.
pub const IMPULSE_ONSET_MPS: f64 = 0.8;
/// The speed at which it is at full voice.
pub const IMPULSE_FULL_MPS: f64 = 3.0;
/// How loud a full impulse is.
pub const IMPULSE_GAIN: f64 = 0.7;

/// **A suspension spike's volume**, or `None` below the onset.
pub fn impulse_volume(strut_mps: f64) -> Option<f64> {
    if !strut_mps.is_finite() || strut_mps < IMPULSE_ONSET_MPS {
        return None;
    }
    Some(IMPULSE_GAIN * (strut_mps / IMPULSE_FULL_MPS).clamp(0.25, 1.0))
}

/// Door sound levels.
pub const LATCH_GAIN: f64 = 0.7;
/// See [`LATCH_GAIN`].
pub const CREAK_GAIN: f64 = 0.45;
/// See [`LATCH_GAIN`].
pub const SLAM_GAIN: f64 = 1.0;
/// See [`LATCH_GAIN`].
pub const THUD_GAIN: f64 = 0.9;

/// **A hinge's creak** at `rate_dps` degrees a second: silent under 15, full
/// by 165, and a little higher as it swings faster.
pub fn creak_voice(rate_dps: f64) -> (f64, f64) {
    let r = if rate_dps.is_finite() {
        rate_dps.abs()
    } else {
        0.0
    };
    let x = ((r - 15.0) / 150.0).clamp(0.0, 1.0);
    (CREAK_GAIN * x, 0.85 + 0.3 * (r / 200.0).clamp(0.0, 1.0))
}

// ── the cues ────────────────────────────────────────────────────────────────

/// **One thing the queue is told**, which a host maps onto exactly one P12.3
/// `AudioCommand`.
#[derive(Clone, Debug, PartialEq)]
pub enum VoiceCue {
    /// Start (or restart) a voice. `emitter` is whose `AudioSource` lends the
    /// bus and the attenuation; `at` is `None` for a non-spatial emitter.
    Play {
        /// The voice's key.
        source: u64,
        /// The entity whose emitter settings it takes.
        emitter: Uuid,
        /// The clip.
        clip: Uuid,
        /// Base volume.
        volume: f64,
        /// Playback rate.
        pitch: f64,
        /// A loop or a one-shot.
        looping: bool,
        /// Where, or `None`.
        at: Option<DVec3>,
    },
    /// A live voice's volume.
    Volume {
        /// The voice's key.
        source: u64,
        /// Base volume.
        volume: f64,
    },
    /// A live voice's playback rate.
    Pitch {
        /// The voice's key.
        source: u64,
        /// Playback rate.
        pitch: f64,
    },
    /// A live voice's position.
    Move {
        /// The voice's key.
        source: u64,
        /// Where it is now.
        at: DVec3,
    },
    /// Stop a voice.
    Stop {
        /// The voice's key.
        source: u64,
    },
}

impl VoiceCue {
    /// The key this cue addresses.
    pub fn source(&self) -> u64 {
        match self {
            VoiceCue::Play { source, .. }
            | VoiceCue::Volume { source, .. }
            | VoiceCue::Pitch { source, .. }
            | VoiceCue::Move { source, .. }
            | VoiceCue::Stop { source } => *source,
        }
    }
}

/// **What the queue last told one looping voice** — so an unchanged value is
/// not told twice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopState {
    /// The clip it is playing.
    pub clip: Uuid,
    /// The last volume sent.
    pub volume: f64,
    /// The last pitch sent.
    pub pitch: f64,
    /// The last position sent.
    pub at: Option<DVec3>,
}

#[derive(Clone, Debug, Default)]
struct CarMemory {
    /// Seen before this step, so an edge has something to be an edge against.
    seen: bool,
    loops: BTreeMap<u64, LoopState>,
    throttle: f64,
    compression_m: [f64; 2],
    hot: [bool; 2],
    /// Sings the NEAR stack (a traffic car).
    near: bool,
}

#[derive(Clone, Debug)]
struct DoorMemory {
    phase: BoardPhase,
    mark_s: f64,
    door_deg: f64,
    door: Uuid,
    /// The car the door belongs to — where its measured hinge is read on the
    /// step a boarding ends (VEH3e audit).
    vehicle: Uuid,
    roll: bool,
    bail: bool,
    creak: Option<LoopState>,
}

/// **The planner's memory**, one per host session — see the module doc for
/// what it holds and why that keeps the stream a function of sim state.
#[derive(Clone, Debug, Default)]
pub struct VoiceMemory {
    cars: BTreeMap<Uuid, CarMemory>,
    doors: BTreeMap<Uuid, DoorMemory>,
    /// Plans made so far — the NEAR tier's update phase. A count of calls,
    /// the same in both hosts; not a clock.
    tick: u64,
}

/// Every car a player-controlled body is seated in, in guid order.
fn player_cars(world: &EcsWorld) -> BTreeSet<Uuid> {
    let w = world.world();
    let Some(mut q) = w.try_query::<&CharacterMovement>() else {
        return BTreeSet::new();
    };
    q.iter(w)
        .filter(|cm| cm.player_controlled && cm.runtime.seat.is_seated())
        .map(|cm| cm.runtime.seat.vehicle)
        .collect()
}

/// The source key of an entity — the P12.3 convention both hosts use.
pub fn entity_key(guid: Uuid) -> u64 {
    guid.as_u128() as u64
}

fn position_of(world: &EcsWorld, guid: Uuid) -> DVec3 {
    let Some(e) = world.entity_of(guid) else {
        return DVec3::ZERO;
    };
    world
        .world()
        .get::<GlobalTransform>(e)
        .map(|g| g.translation())
        .or_else(|| {
            world
                .world()
                .get::<Transform>(e)
                .map(|t| t.translation.to_dvec3())
        })
        .unwrap_or(DVec3::ZERO)
}

impl VoiceMemory {
    /// A fresh session's memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many cars are singing — started loops, right now.
    pub fn voiced_cars(&self) -> usize {
        self.cars.values().filter(|c| !c.loops.is_empty()).count()
    }

    /// How many of them sing the NEAR stack (traffic), right now.
    pub fn near_voiced_cars(&self) -> usize {
        self.cars
            .values()
            .filter(|c| c.near && !c.loops.is_empty())
            .count()
    }

    /// What the queue was last told about one looping voice, or `None`.
    pub fn loop_state(&self, source: u64) -> Option<LoopState> {
        self.cars
            .values()
            .find_map(|c| c.loops.get(&source).copied())
            .or_else(|| {
                self.doors
                    .iter()
                    .find(|(g, _)| door_key(entity_key(**g), DoorLayer::Creak) == source)
                    .and_then(|(_, d)| d.creak)
            })
    }

    /// **The whole step's cues**, cars in the order given (the vehicle step's
    /// own outcome order) and then boarding bodies in guid order.
    ///
    /// `cars` pairs a chassis with the telemetry its class published this step;
    /// a car with no `AudioSource` is voiceless and skipped. `dt` is the fixed
    /// step, the denominator of every rate.
    pub fn plan(
        &mut self,
        world: &EcsWorld,
        cars: &[(Uuid, VoiceTelemetry)],
        dt: f64,
    ) -> Vec<VoiceCue> {
        let mut cues = Vec::new();
        let dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            1.0 / 60.0
        };
        let mut seen: BTreeSet<Uuid> = BTreeSet::new();
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;
        let traffic = crate::traffic::traffic_of(world).map(|t| &t.records);
        // The cars a PLAYER sits in sing the full stack whoever's they are —
        // a stolen traffic car is the player's car now.
        let played = player_cars(world);
        for (chassis, t) in cars {
            let Some(e) = world.entity_of(*chassis) else {
                continue;
            };
            let Some(src) = world.world().get::<AudioSource>(e).cloned() else {
                continue;
            };
            // A combustion engine sings its grains; a CRAFT (wave VEH3g) sings
            // its rotor, propeller, turbine or hull -- and a boat with a petrol
            // engine sings both. Neither is a car the stack cannot voice.
            let family = GrainFamily::for_engine(t.cylinders, t.voice_kind, t.firing_order);
            if family.is_none() && !t.craft.any() {
                continue;
            }
            seen.insert(*chassis);
            let at = src.spatial.then(|| position_of(world, *chassis));
            let near =
                traffic.is_some_and(|r| r.contains_key(chassis)) && !played.contains(chassis);
            let mem = self.cars.entry(*chassis).or_default();
            mem.near = near;
            plan_car(&mut cues, mem, *chassis, t, family, &src, at, dt, tick);
        }
        // A car that has gone — despawned, or lost its emitter — is silenced.
        let gone: Vec<Uuid> = self
            .cars
            .keys()
            .filter(|g| !seen.contains(*g))
            .copied()
            .collect();
        for g in gone {
            if let Some(mem) = self.cars.remove(&g) {
                for key in mem.loops.keys() {
                    cues.push(VoiceCue::Stop { source: *key });
                }
            }
        }
        self.plan_doors(&mut cues, world, dt);
        cues
    }

    fn plan_doors(&mut self, cues: &mut Vec<VoiceCue>, world: &EcsWorld, dt: f64) {
        let mut present: BTreeSet<Uuid> = BTreeSet::new();
        for body in crate::movement::movement_targets(world) {
            let Some(e) = world.entity_of(body) else {
                continue;
            };
            let Some(cm) = world.world().get::<CharacterMovement>(e) else {
                continue;
            };
            let b = cm.runtime.boarding;
            let roll = cm.mode == MovementMode::Roll;
            if b.phase == BoardPhase::Idle && !b.bail && !self.doors.contains_key(&body) {
                continue;
            }
            present.insert(body);
            let mem = self.doors.entry(body).or_insert(DoorMemory {
                phase: BoardPhase::Idle,
                mark_s: -1.0,
                door_deg: b.door_deg,
                door: b.door,
                vehicle: b.vehicle,
                roll,
                bail: b.bail,
                creak: None,
            });
            let key = entity_key(body);
            let at = if b.door.is_nil() {
                position_of(world, body)
            } else {
                position_of(world, b.door)
            };
            let one_shot = |layer: DoorLayer, volume: f64| VoiceCue::Play {
                source: door_key(key, layer),
                emitter: body,
                clip: door_clip(layer),
                volume,
                pitch: 1.0,
                looping: false,
                at: Some(at),
            };
            // THE LATCH: the outer handle's pull (the door phase's mark, set
            // on the step the motor is told) or the inner handle's push (the
            // first step of an exit).
            let latched = b.phase == BoardPhase::OpeningDoor
                && b.mark_s >= 0.0
                && !(mem.phase == BoardPhase::OpeningDoor && mem.mark_s >= 0.0);
            let pushed = b.phase == BoardPhase::Exiting && mem.phase != BoardPhase::Exiting;
            if !b.door.is_nil() && (latched || pushed) {
                cues.push(one_shot(DoorLayer::Latch, LATCH_GAIN));
            }
            // The hinge's rate, from two steps on which the machine READ the
            // hinge: `door_deg` is written in the four door phases and left
            // stale in the others, so a rate across a stale step is a jump
            // between two readings minutes apart, not a swing.
            let read = |p: BoardPhase| {
                matches!(
                    p,
                    BoardPhase::OpeningDoor
                        | BoardPhase::Seated
                        | BoardPhase::Exiting
                        | BoardPhase::ClosingDoor
                )
            };
            let rate = if read(mem.phase) && read(b.phase) {
                (b.door_deg - mem.door_deg) / dt
            } else {
                0.0
            };
            // THE SLAM: the hinge crossing shut while it was being closed —
            // including the step a closing boarding ENDS on, where the
            // machine has just measured the door shut and reset itself (its
            // `door` is nil by then, so the door is the one remembered).
            //
            // **The hinge as the JOINT has it on that end step** (VEH3e audit,
            // carried 5): a boarding that ends resets the machine's own
            // `door_deg` to 0 whether or not the door shut — a `ClosingDoor`
            // that TIMES OUT against something in the way ends with the door
            // standing open, and the reset read as a crossing was a slam out
            // of an open door. So the end step reads the part's measured angle
            // off the car's damage row (VEH3c folds the joint into it), and
            // only a door that is really shut slams.
            let closing = matches!(mem.phase, BoardPhase::Seated | BoardPhase::ClosingDoor);
            let slam_door = if b.door.is_nil() { mem.door } else { b.door };
            let now_deg = if b.door.is_nil() && !mem.door.is_nil() {
                let car = if b.vehicle.is_nil() {
                    mem.vehicle
                } else {
                    b.vehicle
                };
                crate::bodywork::damage_row(world, car)
                    .and_then(|r| r.parts.get(&mem.door))
                    .map(|p| p.angle_deg.abs())
                    .unwrap_or(b.door_deg)
            } else {
                b.door_deg
            };
            if !slam_door.is_nil()
                && closing
                && mem.door_deg > DOOR_SHUT_DEG
                && now_deg <= DOOR_SHUT_DEG
            {
                let rate = (now_deg - mem.door_deg) / dt;
                let v = SLAM_GAIN * (rate.abs() / 150.0).clamp(0.4, 1.0);
                cues.push(one_shot(DoorLayer::Slam, v));
            }
            // THE CREAK: a loop while the hinge is being worked — by the motor
            // opening it, by a hand pulling it shut (the rows where the hand is
            // ON the inner handle, never the motor-only rows before it), or by
            // the push out.
            let swinging = !b.door.is_nil()
                && match b.phase {
                    BoardPhase::OpeningDoor | BoardPhase::Exiting | BoardPhase::ClosingDoor => true,
                    BoardPhase::Seated => b.handle_weight > 0.0,
                    _ => false,
                };
            let creak_key = door_key(key, DoorLayer::Creak);
            if swinging {
                let (vol, pitch) = creak_voice(rate);
                match mem.creak {
                    None if vol > 0.0 => {
                        cues.push(VoiceCue::Play {
                            source: creak_key,
                            emitter: body,
                            clip: door_clip(DoorLayer::Creak),
                            volume: vol,
                            pitch,
                            looping: true,
                            at: Some(at),
                        });
                        mem.creak = Some(LoopState {
                            clip: door_clip(DoorLayer::Creak),
                            volume: vol,
                            pitch,
                            at: Some(at),
                        });
                    }
                    None => {}
                    Some(ref mut s) => update_loop(cues, creak_key, s, vol, pitch, Some(at)),
                }
            } else if mem.creak.take().is_some() {
                cues.push(VoiceCue::Stop { source: creak_key });
            }
            // THE THUD: a body out of a moving car, landing into its roll.
            if b.bail && roll && !mem.roll {
                cues.push(one_shot(DoorLayer::Thud, THUD_GAIN));
            }
            mem.phase = b.phase;
            mem.mark_s = b.mark_s;
            mem.door_deg = b.door_deg;
            if !b.door.is_nil() {
                mem.door = b.door;
            }
            if !b.vehicle.is_nil() {
                mem.vehicle = b.vehicle;
            }
            mem.roll = roll;
            mem.bail = b.bail;
        }
        // Forget a body that has finished — once its creak is stopped.
        let done: Vec<Uuid> = self
            .doors
            .iter()
            .filter(|(g, m)| {
                !present.contains(*g)
                    || (m.phase == BoardPhase::Idle && !m.bail && m.creak.is_none())
            })
            .map(|(g, _)| *g)
            .collect();
        for g in done {
            if let Some(m) = self.doors.remove(&g) {
                if m.creak.is_some() {
                    cues.push(VoiceCue::Stop {
                        source: door_key(entity_key(g), DoorLayer::Creak),
                    });
                }
            }
        }
    }
}

/// **The quietest volume a voice is asked for**: anything below it is sent as
/// silence. A thousandth is -60 dB under the emitter — and without a floor, a
/// whine at an idle's 2 rpm of shaft would be re-sent every step at 0.00006.
pub const VOICE_FLOOR: f64 = 1e-3;

/// **How far an emitter moves before it is moved**, metres. A centimetre: a
/// parked car's chassis settles by micrometres a step, and a `SetPosition` for
/// each of them is a command about nothing.
pub const MOVE_EPS_M: f64 = 0.01;

/// A volume below [`VOICE_FLOOR`] is silence.
fn floored(volume: f64) -> f64 {
    if volume.is_finite() && volume >= VOICE_FLOOR {
        volume
    } else {
        0.0
    }
}

/// Tell a live loop what changed — and nothing, while it is silent and stays
/// silent: a voice at volume zero is not re-pitched or moved until it is heard.
fn update_loop(
    cues: &mut Vec<VoiceCue>,
    source: u64,
    s: &mut LoopState,
    volume: f64,
    pitch: f64,
    at: Option<DVec3>,
) {
    let volume = floored(volume);
    if s.volume == 0.0 && volume == 0.0 {
        return;
    }
    if volume != s.volume {
        cues.push(VoiceCue::Volume { source, volume });
        s.volume = volume;
    }
    if pitch != s.pitch {
        cues.push(VoiceCue::Pitch { source, pitch });
        s.pitch = pitch;
    }
    if let Some(p) = at {
        let moved =
            s.at.is_none_or(|q| (p - q).length_squared() >= MOVE_EPS_M * MOVE_EPS_M);
        if moved {
            cues.push(VoiceCue::Move { source, at: p });
            s.at = Some(p);
        }
    }
}

/// The loops a running car sings this step: (layer, clip, volume, pitch).
fn car_loops(
    t: &VoiceTelemetry,
    family: GrainFamily,
    src: &AudioSource,
) -> Vec<(VoiceLayer, Uuid, f64, f64)> {
    let base = if src.volume.is_finite() {
        src.volume.max(0.0)
    } else {
        1.0
    };
    let authored_pitch = if src.pitch.is_finite() && src.pitch > 0.0 {
        src.pitch
    } else {
        1.0
    };
    let level = base * engine_level(t);
    let w = load_weights(t.load());
    let gp = (grain_pitch(t, family) * authored_pitch).clamp(PITCH_MIN, PITCH_MAX);
    let mut out = Vec::with_capacity(7);
    for (i, load) in EngineLoad::ALL.iter().enumerate() {
        let layer = VoiceLayer::ALL[i];
        // **A row's own clip OVERRIDES the grain** (wave VEH3f). A roster row
        // that names `engine_clip` has it written onto its emitter, and it plays
        // in all three load slots -- crossfaded by the same load weights and
        // pitched by the same grain pitch -- so an authored engine is heard
        // through the stack rather than beside it. Until this wave a clip on a
        // voiced car's emitter was silently ignored.
        let clip = src.clip.unwrap_or_else(|| grain_clip(family, *load));
        out.push((layer, clip, level * w[i], gp));
    }
    let (wv, wp) = whine_voice(t);
    out.push((VoiceLayer::Whine, whine_clip(), base * wv, wp));
    if t.turbocharged {
        let (tv, tp) = turbo_voice(t);
        out.push((VoiceLayer::Turbo, turbo_clip(), base * tv, tp));
    }
    for (axle, layer) in [
        (0usize, VoiceLayer::SquealFront),
        (1, VoiceLayer::SquealRear),
    ] {
        let a = t.axles[axle];
        let (sv, sp) = squeal_voice(a.slip);
        out.push((
            layer,
            squeal_clip(SurfaceVoice::of(a.surface)),
            base * sv,
            sp,
        ));
    }
    let (rv, rp) = roll_voice(t);
    out.push((
        VoiceLayer::Roll,
        roll_clip(SurfaceVoice::of(t.axles[1].surface)),
        base * rv,
        rp,
    ));
    out
}

/// The loops a craft with no combustion engine sings from its wheels (wave
/// VEH3g): the two squeals and the rolling road, exactly as a car's.
fn tyre_loops(t: &VoiceTelemetry, src: &AudioSource) -> Vec<(VoiceLayer, Uuid, f64, f64)> {
    let base = if src.volume.is_finite() {
        src.volume.max(0.0)
    } else {
        1.0
    };
    let mut out = Vec::with_capacity(3);
    for (axle, layer) in [
        (0usize, VoiceLayer::SquealFront),
        (1, VoiceLayer::SquealRear),
    ] {
        let a = t.axles[axle];
        let (sv, sp) = squeal_voice(a.slip);
        out.push((
            layer,
            squeal_clip(SurfaceVoice::of(a.surface)),
            base * sv,
            sp,
        ));
    }
    let (rv, rp) = roll_voice(t);
    out.push((
        VoiceLayer::Roll,
        roll_clip(SurfaceVoice::of(t.axles[1].surface)),
        base * rv,
        rp,
    ));
    out
}

/// **How often a NEAR voice is re-told**, fixed steps (VEH3e audit). A driven
/// traffic car's grain changes pitch and its emitter moves on every step, so a
/// loop told everything would cost up to three commands a step a car; told
/// every fourth step, on a phase its key spreads so a queue's cars do not all
/// speak on one step, it costs under one. A `Play` and a `Stop` are never
/// deferred.
pub const NEAR_EVERY: u64 = 4;

/// **The NEAR stack's level** under the full stack's (VEH3e audit) — a car
/// driving past is heard, not mixed like the one the player drives.
pub const NEAR_GAIN: f64 = 0.8;

/// The loops a NEAR (traffic) car sings: the half-load grain at the engine's
/// level and its revs' pitch, one squeal on its worse axle, and its rolling
/// road — which is most of what a passing car sounds like from the kerb.
fn near_loops(
    t: &VoiceTelemetry,
    family: GrainFamily,
    src: &AudioSource,
) -> Vec<(VoiceLayer, Uuid, f64, f64)> {
    let base = NEAR_GAIN
        * if src.volume.is_finite() {
            src.volume.max(0.0)
        } else {
            1.0
        };
    let authored_pitch = if src.pitch.is_finite() && src.pitch > 0.0 {
        src.pitch
    } else {
        1.0
    };
    let gp = (grain_pitch(t, family) * authored_pitch).clamp(PITCH_MIN, PITCH_MAX);
    let worse = if t.axles[0].slip > t.axles[1].slip {
        t.axles[0]
    } else {
        t.axles[1]
    };
    let (sv, sp) = squeal_voice(worse.slip);
    let (rv, rp) = roll_voice(t);
    vec![
        (
            VoiceLayer::GrainMid,
            grain_clip(family, EngineLoad::Mid),
            base * engine_level(t),
            gp,
        ),
        (
            VoiceLayer::SquealRear,
            squeal_clip(SurfaceVoice::of(worse.surface)),
            base * sv,
            sp,
        ),
        (
            VoiceLayer::Roll,
            roll_clip(SurfaceVoice::of(t.axles[1].surface)),
            base * rv,
            rp,
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn plan_car(
    cues: &mut Vec<VoiceCue>,
    mem: &mut CarMemory,
    chassis: Uuid,
    t: &VoiceTelemetry,
    family: Option<GrainFamily>,
    src: &AudioSource,
    at: Option<DVec3>,
    dt: f64,
    tick: u64,
) {
    let key = entity_key(chassis);
    let base = if src.volume.is_finite() {
        src.volume.max(0.0)
    } else {
        1.0
    };
    let first = !mem.seen;
    mem.seen = true;
    if t.running() {
        let mut loops = match (family, mem.near) {
            (Some(f), true) => near_loops(t, f, src),
            (Some(f), false) => car_loops(t, f, src),
            // A craft with no combustion engine still has its gear or its
            // skids: the tyre layers alone (a jet rolling out on the strip).
            (None, _) => tyre_loops(t, src),
        };
        // **THE CRAFT** (wave VEH3g) -- on top, at the emitter's own volume.
        // Never NEAR: an air or sea craft is never a traffic record, so the
        // rule is the full stack whenever it runs.
        let craft_base = if src.volume.is_finite() {
            src.volume.max(0.0)
        } else {
            1.0
        };
        if t.craft.wheelless {
            loops.retain(|l| {
                !matches!(
                    l.0,
                    VoiceLayer::Whine
                        | VoiceLayer::SquealFront
                        | VoiceLayer::SquealRear
                        | VoiceLayer::Roll
                )
            });
        }
        for (layer, clip, v, p) in craft_voices(&t.craft) {
            loops.push((layer, clip, craft_base * v, p));
        }
        // A loop the stack no longer has (a traffic car the player got out
        // of drops to the NEAR stack) is stopped, in key order.
        let keep: BTreeSet<u64> = loops.iter().map(|l| voice_key(key, l.0)).collect();
        let gone: Vec<u64> = mem
            .loops
            .keys()
            .filter(|k| !keep.contains(k))
            .copied()
            .collect();
        for source in gone {
            mem.loops.remove(&source);
            cues.push(VoiceCue::Stop { source });
        }
        for (layer, clip, volume, pitch) in loops {
            let source = voice_key(key, layer);
            // A NEAR loop is re-told on its own phase only (see `NEAR_EVERY`).
            let due = !mem.near || tick.wrapping_add(source).is_multiple_of(NEAR_EVERY);
            match mem.loops.get_mut(&source) {
                Some(s) if s.clip == clip && !due => {}
                // A new voice, or a squeal whose surface changed under it:
                // (re)start it with everything it needs in one command.
                Some(s) if s.clip == clip => update_loop(cues, source, s, volume, pitch, at),
                _ => {
                    let volume = floored(volume);
                    cues.push(VoiceCue::Play {
                        source,
                        emitter: chassis,
                        clip,
                        volume,
                        pitch,
                        looping: true,
                        at,
                    });
                    mem.loops.insert(
                        source,
                        LoopState {
                            clip,
                            volume,
                            pitch,
                            at,
                        },
                    );
                }
            }
        }
        // THE BLOW-OFF: the throttle SHUT on boost.
        if !first
            && !mem.near
            && t.turbocharged
            && mem.throttle >= BLOW_OFF_FROM
            && finite01(t.throttle) <= BLOW_OFF_TO
            && t.boost >= BLOW_OFF_MIN_BOOST
        {
            cues.push(VoiceCue::Play {
                source: voice_key(key, VoiceLayer::BlowOff),
                emitter: chassis,
                clip: blow_off_clip(),
                volume: base * BLOW_OFF_GAIN * finite01(t.boost),
                pitch: 1.0,
                looping: false,
                at,
            });
        }
    } else {
        // Engine off: every loop stops, and starts again with a Play.
        for source in std::mem::take(&mut mem.loops).into_keys() {
            cues.push(VoiceCue::Stop { source });
        }
    }
    // THE SURFACE IMPULSES: a strut closing faster than the onset, on its
    // rising edge only, with the axle's surface — whether or not the engine
    // runs (a parked car dropped on a kerb still thumps).
    for (i, layer) in [
        (0usize, VoiceLayer::ImpulseFront),
        (1, VoiceLayer::ImpulseRear),
    ] {
        let a = t.axles[i];
        let v = (a.compression_m - mem.compression_m[i]) / dt;
        if !first && !mem.hot[i] {
            if let Some(vol) = impulse_volume(v) {
                cues.push(VoiceCue::Play {
                    source: voice_key(key, layer),
                    emitter: chassis,
                    clip: impulse_clip(SurfaceVoice::of(a.surface)),
                    volume: base * vol,
                    pitch: 1.0,
                    looping: false,
                    at,
                });
            }
        }
        mem.hot[i] = v.is_finite() && v >= 0.5 * IMPULSE_ONSET_MPS;
        mem.compression_m[i] = a.compression_m;
    }
    mem.throttle = finite01(t.throttle);
}

/// **The audio HUD row** — what the mixer holds for one car's voices:
/// `AUDIO P8X  3120 rpm  LOAD 0.82  GEAR 3  WHINE 1.04  TURBO 0.55  SQUEAL F0.00 R0.71 sealed  CMDS 9`.
///
/// `whine`, `turbo` and `squeal` are what the HOST read back off its engine —
/// never the planner's own intent; `None` prints a dash.
#[allow(clippy::too_many_arguments)]
pub fn voice_readout(
    t: &VoiceTelemetry,
    whine_pitch: Option<f64>,
    turbo_pitch: Option<f64>,
    squeal: [Option<f64>; 2],
    commands: usize,
) -> String {
    let fam = GrainFamily::for_engine(t.cylinders, t.voice_kind, t.firing_order)
        .map(|f| match f {
            GrainFamily::P4 => "P4",
            GrainFamily::P6 => "P6",
            GrainFamily::P8Cross => "P8X",
            GrainFamily::P8Flat => "P8F",
            GrainFamily::Diesel6 => "D6",
        })
        .unwrap_or("-");
    let num = |v: Option<f64>| v.map(|v| format!("{v:.2}")).unwrap_or_else(|| "-".into());
    format!(
        "AUDIO {fam}  {:.0} rpm  LOAD {:.2}  GEAR {}  WHINE {}  TURBO {}  SQUEAL F{} R{} {}  CMDS {commands}",
        t.rpm,
        t.load(),
        t.gear,
        num(whine_pitch),
        num(turbo_pitch),
        num(squeal[0]),
        num(squeal[1]),
        SurfaceVoice::of(t.axles[1].surface).name(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn telemetry() -> VoiceTelemetry {
        VoiceTelemetry {
            rpm: 2_400.0,
            idle_rpm: 800.0,
            redline_rpm: 7_000.0,
            throttle: 0.5,
            boost: 0.0,
            turbocharged: false,
            fuel_cut: false,
            gear: 2,
            shaft_rpm: 2_400.0,
            cylinders: 8.0,
            voice_kind: 0.0,
            firing_order: 1.0,
            occupied: true,
            quiet: false,
            speed_mps: 10.0,
            axles: [AxleVoice::default(); 2],
            craft: CraftVoice::default(),
        }
    }

    #[test]
    fn the_crossfade_sums_to_one_and_hits_each_grain_at_its_load() {
        for i in 0..=20 {
            let w = load_weights(i as f64 / 20.0);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(load_weights(0.0), [1.0, 0.0, 0.0]);
        assert_eq!(load_weights(0.5), [0.0, 1.0, 0.0]);
        assert_eq!(load_weights(1.0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_four_and_an_eight_fire_at_one_to_two_at_the_same_revs() {
        let mut t = telemetry();
        t.cylinders = 4.0;
        let four = grain_pitch(&t, GrainFamily::P4);
        t.cylinders = 8.0;
        let eight = grain_pitch(&t, GrainFamily::P8Cross);
        // Same playback rate — and the eight's CLIP fires twice as often.
        assert_eq!(four, eight);
        assert_eq!(four, 1.0);
        // A five on the four's clip is pitched up by 5/4.
        t.cylinders = 5.0;
        assert!((grain_pitch(&t, GrainFamily::P6) - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn the_squeal_is_a_function_of_slip_alone() {
        assert_eq!(squeal_voice(0.0).0, 0.0);
        assert_eq!(squeal_voice(SQUEAL_ONSET).0, 0.0);
        assert!((squeal_voice(SQUEAL_FULL).0 - SQUEAL_GAIN).abs() < 1e-12);
        assert!(squeal_voice(1.5).0 > squeal_voice(1.0).0);
        assert_eq!(squeal_voice(f64::NAN), squeal_voice(0.0));
    }

    #[test]
    fn a_parked_car_nobody_is_in_is_silent() {
        let mut t = telemetry();
        t.occupied = false;
        t.quiet = true;
        assert!(!t.running());
        t.quiet = false;
        assert!(t.running());
    }

    #[test]
    fn the_families_follow_the_three_tunables() {
        assert_eq!(
            GrainFamily::for_engine(4.0, 0.0, 0.0),
            Some(GrainFamily::P4)
        );
        assert_eq!(
            GrainFamily::for_engine(6.0, 0.0, 0.0),
            Some(GrainFamily::P6)
        );
        assert_eq!(
            GrainFamily::for_engine(8.0, 0.0, 1.0),
            Some(GrainFamily::P8Cross)
        );
        assert_eq!(
            GrainFamily::for_engine(8.0, 0.0, 2.0),
            Some(GrainFamily::P8Flat)
        );
        assert_eq!(
            GrainFamily::for_engine(4.0, 1.0, 0.0),
            Some(GrainFamily::Diesel6)
        );
        assert_eq!(GrainFamily::for_engine(0.0, 0.0, 0.0), None);
        assert_eq!(GrainFamily::for_engine(4.0, 3.0, 0.0), None);
        // The clip table is family-major, three loads each.
        assert_eq!(
            grain_clip(GrainFamily::Diesel6, EngineLoad::Full),
            vehicle_clip(14)
        );
        assert_eq!(vehicle_clips().len(), VEHICLE_CLIP_NAMES.len());
    }

    #[test]
    fn every_key_is_distinct_from_the_chassis_and_from_each_other() {
        let k = 0x1234_5678_9abc_def0u64;
        let mut keys: BTreeSet<u64> = VoiceLayer::ALL.iter().map(|l| voice_key(k, *l)).collect();
        for d in [
            DoorLayer::Latch,
            DoorLayer::Creak,
            DoorLayer::Slam,
            DoorLayer::Thud,
        ] {
            keys.insert(door_key(k, d));
        }
        // Sixteen layers (the five craft voices of wave VEH3g appended) and
        // four door layers.
        assert_eq!(keys.len(), 20);
        assert!(!keys.contains(&k));
        // …and no CRAFT voice is another NEARBY entity's door (wave VEH3g):
        // fixtures mint guids a few apart, and a craft salt in the door salts'
        // own neighbourhood made a car's hull spray read as a hero's creak.
        // Only the craft layers: VEH3e's own eleven sit `0x10` under the four
        // door salts, so `car ^ …0001 == hero ^ …0011` whenever two guids
        // differ by sixteen (carried in the wave's ledger). And no craft layer
        // of one craft is ANOTHER craft layer of a nearby craft.
        let craft = [
            VoiceLayer::Rotor,
            VoiceLayer::Prop,
            VoiceLayer::Jet,
            VoiceLayer::HullSlap,
            VoiceLayer::HullSpray,
        ];
        for delta in 1u64..=0x40 {
            for a in craft {
                for b in VoiceLayer::ALL {
                    assert_ne!(
                        voice_key(k, a),
                        voice_key(k ^ delta, b),
                        "{a:?} vs {b:?} at {delta}"
                    );
                }
            }
        }
        for delta in 1u64..=0x40 {
            for l in [
                VoiceLayer::Rotor,
                VoiceLayer::Prop,
                VoiceLayer::Jet,
                VoiceLayer::HullSlap,
                VoiceLayer::HullSpray,
            ] {
                for d in [
                    DoorLayer::Latch,
                    DoorLayer::Creak,
                    DoorLayer::Slam,
                    DoorLayer::Thud,
                ] {
                    assert_ne!(
                        voice_key(k, l),
                        door_key(k ^ delta, d),
                        "{l:?} vs {d:?} at {delta}"
                    );
                }
            }
        }
        for s in crate::weapon::LAYER_SALTS {
            assert!(!keys.contains(&(k ^ s)));
        }
    }
}
