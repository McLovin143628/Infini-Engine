//! The **audio command queue** (P12.3) — the determinism seam between the sim
//! world and the audio device.
//!
//! # Doctrine: audio is output-only, so the engine lives *outside* the sim world
//!
//! The [`AudioEngine`](crate::AudioEngine) is a **long-lived host resource**, not
//! sim state: it owns a device (or the no-device fallback), voice handles, and a
//! mixer — none of which is part of the deterministic world snapshot. Sim systems
//! must never touch it directly, because a device side effect is not reproducible.
//!
//! Instead, each fixed step the sim (autoplay [`AudioSource`]s on their first
//! tick, and Blueprint `audio.*` nodes during actor Ticks) **emits
//! [`AudioCommand`]s into a plain [`Vec`]**. That queue is a *pure function of sim
//! state*, so it is fully deterministic and can be asserted in a headless test.
//! Only *after* the step does the host drain the queue into the engine
//! ([`AudioEngine::drain`](crate::AudioEngine::drain)); the drain resolves clip
//! GUIDs to decoded [`SoundData`](crate::SoundData) and performs the device side
//! effects that are *not* sim state. Determinism is preserved because the
//! **command stream** — not the audible output — is the observable contract.
//!
//! A source is keyed by a caller-supplied `source` id (the ECS entity's bits),
//! so the engine tracks **one active voice per source**: a second `Play` for the
//! same source replaces the first, and `Stop`/`SetVolume`/`SetPitch` address it by
//! that key without the sim ever holding a [`SoundHandle`](crate::SoundHandle).

use glam::DVec3;
use uuid::Uuid;

use crate::spatial::{Attenuation, Listener};

/// Parameters for starting (or restarting) a source's voice. Self-contained: the
/// host needs only this + the resolved clip, never the ECS world, so the command
/// stream is a complete, assertable description of what will play.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayCommand {
    /// Source identity (ECS entity bits). One active voice per source.
    pub source: u64,
    /// The `.inf_audio` clip GUID; resolved to [`SoundData`](crate::SoundData)
    /// host-side at drain time. An unresolvable GUID is a silent no-op.
    pub clip: Uuid,
    /// Named mixer bus (see [`MixerConfig`](crate::mixer::MixerConfig)).
    pub bus: String,
    /// Base linear volume before bus/master/spatial/occlusion scaling.
    pub volume: f64,
    /// Playback-rate factor (`1.0` = normal pitch).
    pub pitch: f64,
    /// Loop the whole clip (`true`) or play it once (`false`).
    pub looping: bool,
    /// World emitter position for spatialization, or `None` for a 2D/UI sound.
    pub position: Option<DVec3>,
    /// Distance-attenuation curve, used only when `position` is `Some`.
    pub attenuation: Attenuation,
    /// Extra obstruction gain in `[0, 1]` the caller (the sim's occlusion
    /// raycast) supplies; `1.0` = unobstructed. The engine multiplies it in.
    pub occlusion_gain: f64,
    /// **The low-pass cutoff this voice is born with**, hertz, or `None` for a
    /// voice nothing filters (wave WPN2c).
    ///
    /// [`AudioCommand::SetOcclusion`]'s own doc used to say a `Play` carries no
    /// cutoff *"because the model that decides one re-evaluates every step"*.
    /// That is true of a DOOR and false of a one-shot: a gunshot heard four
    /// hundred metres away is a thump because a few hundred metres of air is a
    /// low-pass, and that cutoff is decided once, at the muzzle, and never
    /// changes for the thirty milliseconds the sound exists. So a one-shot's
    /// filter belongs on the command that starts it, and a loop's still belongs
    /// on the per-step one.
    ///
    /// **It is audible** — see [`crate::mixer::Effect::Lowpass`] and
    /// `SoundData::low_passed`: the engine filters the decoded frames through
    /// its own one-pole before the voice starts, so this is a sound and not a
    /// number in a log. `None` is the whole of the pre-WPN2c behaviour, which is
    /// what keeps every committed command stream comparing equal.
    pub lowpass_hz: Option<f64>,
}

impl PlayCommand {
    /// A non-spatial (2D) play command on `bus` at unity volume/pitch.
    pub fn new(source: u64, clip: Uuid, bus: impl Into<String>) -> Self {
        Self {
            source,
            clip,
            bus: bus.into(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            position: None,
            attenuation: Attenuation::default(),
            occlusion_gain: 1.0,
            lowpass_hz: None,
        }
    }
}

/// A deterministic audio command. Produced by sim systems into a queue and
/// drained host-side into the [`AudioEngine`](crate::AudioEngine).
#[derive(Clone, Debug, PartialEq)]
pub enum AudioCommand {
    /// Start (or restart) the voice for a source.
    Play(PlayCommand),
    /// Stop and forget the source's voice.
    Stop { source: u64 },
    /// Set a live source's base volume.
    SetVolume { source: u64, volume: f64 },
    /// Set a live source's pitch (playback-rate factor).
    SetPitch { source: u64, pitch: f64 },
    /// **Re-evaluate a live source's obstruction** (island wave VEN1b).
    ///
    /// [`PlayCommand::occlusion_gain`] is taken once, when the voice starts,
    /// which is the right shape for a one-shot and the wrong one for a loop: a
    /// venue's music began muffled behind its own door and stayed muffled for
    /// the whole session however far in the listener walked. This is the same
    /// factor, pushed by the sim every step it changes.
    ///
    /// `lowpass_hz` is the cutoff the sim's occlusion model decided on, or
    /// `None` for a path that only attenuates. It is **modelled and
    /// inspectable, not yet audible** — `backend.rs` has no filter code and the
    /// mixer's own `Effect::Lowpass` has been in the same state since P12 —
    /// so it is carried here, where the two hosts' command streams are
    /// compared, and read back through
    /// [`AudioEngine::effective_lowpass_hz`](crate::AudioEngine::effective_lowpass_hz).
    SetOcclusion {
        /// The source, as [`PlayCommand::source`].
        source: u64,
        /// The obstruction gain in `[0, 1]`; `1.0` is clear.
        gain: f64,
        /// The cutoff this path implies, hertz, or `None`.
        lowpass_hz: Option<f64>,
    },
    /// **Move a live source's emitter** (wave EMS2).
    ///
    /// [`PlayCommand::position`] is taken once, when the voice starts, which is
    /// the right shape for a footstep and the wrong one for anything that
    /// travels: a siren spatialized where its `Play` was issued stays outside
    /// the station it left, however far across town the ambulance gets.
    ///
    /// [`SetOcclusion`](Self::SetOcclusion)'s argument verbatim, one property
    /// over, and the second half of a debt this tree has carried since VEH2a:
    /// `RigSpawn::engine_voice`'s own doc names *"no `AudioCommand::SetPosition`,
    /// so a driving car's engine is spatialized where its `Play` was issued"* as
    /// the reason **traffic is silent**. This is that command. It does not by
    /// itself give traffic a voice — a dozen cars each pushing a position every
    /// step is a bounded log that evicts, which is what `inf_core::BoundedLog`
    /// is for and what the island's own drive gate caught — so the debt is paid
    /// for the system that needs it (a handful of responding units at a tenth of
    /// the step rate) and named for the one that still does not.
    ///
    /// (A code span rather than a link: `inf-audio` does not depend on
    /// `inf-core`, so an intra-doc link there resolves to nothing and costs a
    /// rustdoc warning for a reference a reader can follow by name.)
    ///
    /// A source that is not playing is a silent no-op, exactly as every other
    /// addressed command here is. A **non-spatial** voice becomes spatial from
    /// this command on, which is [`crate::AudioEngine::set_position`]'s own
    /// documented behaviour and is the honest answer for a caller that has
    /// decided a sound has a place.
    SetPosition {
        /// The source, as [`PlayCommand::source`].
        source: u64,
        /// Where the emitter is now, world metres.
        position: DVec3,
    },
    /// Update the listener pose (position + orientation) for spatial mixing.
    SetListener(Listener),
}

/// **How many commands a host's audio log keeps** (wave WPN2c) — the ring the
/// two hosts' `audio_log`s are built with, and the reason it is not
/// `inf_core::DEFAULT_LOG_CAPACITY` any more.
///
/// # The arithmetic, with its population named
///
/// The default was 8 192, priced at island wave VEN1b against a ONE-layer
/// gunshot: one shooter at 600 rpm reached the first eviction after 819 s and
/// eight shooters after 102 s. This wave multiplies the rate by four and adds
/// two more emitters, so the same firefight is:
///
/// ```text
///   8 shooters x 600 rpm  =  80 shots/s
///   x 4 layers            = 320 Play/s
///   + one casing bounce per shot     =  80/s
///   + one SetListener per fixed step =  60/s
///                                    -------
///                                      460/s
/// ```
///
/// The gate arm this wave owes is `dropped == 0` over **120 s at eight
/// shooters**, which is 55 200 commands, so 8 192 would have evicted after
/// **17.8 s** and every arm that reads the head of the stream would have been
/// reading a tail. Sixty-five thousand five hundred and thirty-six is the next
/// power of two above the arm, and it buys a **142 s** horizon at that rate.
///
/// # What it costs, measured
///
/// An `AudioCommand` is 152 bytes on a 64-bit target (`wpn2c_gate` prints
/// `size_of` beside this number rather than trusting it), so the ring is a
/// **9.5 MiB** ceiling in the shipped player, reached only by a session that
/// actually issues that many commands — the `Vec` grows to what is pushed. A
/// quiet hour of walking around is 60 SetListeners a second, which reaches the
/// ceiling in 18 minutes and stays there. That is the honest price of a log the
/// gates read, and the alternative — a small ring in the player and a large one
/// in a test — would mean the gates were not reading what ships.
pub const AUDIO_LOG_CAPACITY: usize = 65_536;

/// A tiny ordered queue of [`AudioCommand`]s. The sim pushes; the host drains.
/// Deterministic by construction (a `Vec`, drained in order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioCommandQueue {
    commands: Vec<AudioCommand>,
}

impl AudioCommandQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push one command.
    pub fn push(&mut self, cmd: AudioCommand) {
        self.commands.push(cmd);
    }

    /// The queued commands, in order.
    pub fn commands(&self) -> &[AudioCommand] {
        &self.commands
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Take the commands out, leaving the queue empty (drain the frame).
    pub fn take(&mut self) -> Vec<AudioCommand> {
        std::mem::take(&mut self.commands)
    }

    /// Clear without returning.
    pub fn clear(&mut self) {
        self.commands.clear();
    }
}
