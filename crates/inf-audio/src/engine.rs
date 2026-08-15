//! [`AudioEngine`]: the facade — a master + bus volume model (built-in enum buses
//! *and* the P12.3 named-bus [`MixerConfig`]), a spatial listener with distance
//! attenuation + occlusion, per-sound play/stop/pause handles, and the
//! command-queue [`drain`](AudioEngine::drain) that the sim host uses.
//!
//! All mixing is computed **engine-side** (buses fold into one linear gain per
//! voice; spatial position into gain + panning; occlusion into an extra gain
//! multiplier) and pushed to kira as a plain per-voice volume/panning/rate. That
//! keeps the kira surface tiny and means the entire mix model is pure,
//! inspectable, and testable without an audio device — which is exactly what the
//! no-device fallback and CI rely on.

use std::collections::BTreeMap;

use glam::DVec3;
use uuid::Uuid;

use crate::backend::Backend;
use crate::command::{AudioCommand, PlayCommand};
use crate::mixer::{MixerConfig, ResolvedBus};
use crate::sound::SoundData;
use crate::spatial::{Attenuation, Listener};

/// A built-in mixer bus (the P9.1b baseline, kept for back-compat). The general
/// named-bus model is [`MixerConfig`]; a voice can play on either (see
/// [`BusRef`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bus {
    /// The global bus — its volume scales everything.
    Master,
    /// Sound effects.
    Sfx,
    /// Music.
    Music,
}

/// Which bus a voice plays on: a built-in [`Bus`] (legacy P9.1b path) or a named
/// [`MixerConfig`] bus (P12.3; what the command queue / ECS `AudioSource.bus`
/// use).
#[derive(Clone, Debug, PartialEq)]
pub enum BusRef {
    /// A built-in enum bus, scaled by the engine's master/sfx/music volumes.
    Builtin(Bus),
    /// A named mixer bus; its folded gain comes from the [`MixerConfig`].
    Named(String),
}

/// An occlusion hook: given the listener position and an emitter position, return
/// an obstruction gain in `[0, 1]` (`1.0` = clear). The engine multiplies it into
/// a spatial voice's gain. A caller (e.g. the sim's physics raycast) supplies it;
/// see [`AudioEngine::set_occlusion_hook`].
pub type OcclusionHook = Box<dyn Fn(DVec3, DVec3) -> f64 + Send + Sync>;

/// An opaque handle to a playing (or paused) sound. Stable for the voice's
/// lifetime; a stopped handle reads back as not-playing rather than aliasing a
/// new voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoundHandle(u64);

/// How to start a sound. Build with [`PlaySettings::default`] (or the `on`/
/// `spatial` helpers) and hand to [`AudioEngine::play`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaySettings {
    /// Which built-in bus the sound plays on. (The command-queue path uses named
    /// mixer buses instead — see [`AudioEngine::drain`].)
    pub bus: Bus,
    /// Base linear volume (`1.0` = unity), before bus/master/spatial scaling.
    pub volume: f64,
    /// Playback-rate factor (`1.0` = normal pitch/speed).
    pub pitch: f64,
    /// Loop the whole clip (`true`) or play once (`false`).
    pub looping: bool,
    /// A world-space emitter position for spatialization, or `None` for a
    /// non-positional (2D/UI) sound played at `panning = 0`.
    pub position: Option<DVec3>,
    /// The distance-attenuation curve, used only when `position` is `Some`.
    pub attenuation: Attenuation,
}

impl Default for PlaySettings {
    fn default() -> Self {
        Self {
            bus: Bus::Sfx,
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            position: None,
            attenuation: Attenuation::default(),
        }
    }
}

impl PlaySettings {
    /// Non-positional settings on the given bus.
    pub fn on(bus: Bus) -> Self {
        Self {
            bus,
            ..Self::default()
        }
    }

    /// Positional settings: a bus, a world position, and an attenuation curve.
    pub fn spatial(bus: Bus, position: DVec3, attenuation: Attenuation) -> Self {
        Self {
            bus,
            position: Some(position),
            attenuation,
            ..Self::default()
        }
    }

    /// Set the base volume.
    pub fn volume(mut self, volume: f64) -> Self {
        self.volume = volume;
        self
    }

    /// Set the pitch (playback-rate factor).
    pub fn pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch;
        self
    }

    /// Set the loop flag.
    pub fn looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }
}

/// Per-voice engine-side state (independent of whether a device is active).
#[derive(Clone, Debug)]
struct Voice {
    bus: BusRef,
    base_volume: f64,
    pitch: f64,
    looping: bool,
    paused: bool,
    position: Option<DVec3>,
    attenuation: Attenuation,
    /// Whether the engine's [`OcclusionHook`] recomputes this voice's
    /// `occlusion_gain` (direct-API path). Command-driven voices set this
    /// `false` and carry the sim-supplied factor instead.
    occlusion: bool,
    /// Extra obstruction multiplier in `[0, 1]` (`1.0` = clear).
    occlusion_gain: f64,
}

/// The audio facade over kira. Construct with [`new`](Self::new) (opens a device
/// when the `cpal` feature is on and one is available) or
/// [`disabled`](Self::disabled) (always the no-op path).
pub struct AudioEngine {
    backend: Backend,
    master_volume: f64,
    sfx_volume: f64,
    music_volume: f64,
    listener: Listener,
    voices: BTreeMap<u64, Voice>,
    next_id: u64,
    /// The named-bus mixer (P12.3) + its folded gains/cutoffs cache.
    mixer: MixerConfig,
    resolved: BTreeMap<String, ResolvedBus>,
    /// Source id (ECS entity bits) → its single active voice, for the command
    /// queue (one voice per source).
    sources: BTreeMap<u64, SoundHandle>,
    /// Optional occlusion hook applied to spatial voices flagged `occlusion`.
    occlusion_hook: Option<OcclusionHook>,
    /// **How many `Play` commands named a clip that did not resolve** (C4-43),
    /// and which clips have already been complained about.
    ///
    /// A counter rather than a log line alone: the crate's doctrine is that the
    /// audio stream is a pure function of the sim state, and an unresolvable
    /// clip is precisely the case where it stops being one. A number is
    /// something a test can assert and a session report can print; a silent
    /// `return` was neither.
    unresolved_clips: u64,
    warned_clips: std::collections::BTreeSet<Uuid>,
    /// How many voices the backend refused outright (a full command queue, or a
    /// device that would not take the sound).
    refused_voices: u64,
}

impl AudioEngine {
    /// Bring up the engine, opening a real audio device if the `cpal` feature is
    /// enabled and one is present. If not (feature off, or a headless/CI machine),
    /// it comes up in the **no-device fallback**: every method is a consistent
    /// no-op for playback while all engine-side state stays live. Never fails.
    pub fn new() -> Self {
        Self::with_backend(Backend::new())
    }

    /// Construct the engine forced into the no-device fallback, regardless of
    /// build features — the deterministic target for the fallback tests and for a
    /// server/tool that wants the mix model without any audio output.
    pub fn disabled() -> Self {
        Self::with_backend(Backend::disabled())
    }

    fn with_backend(backend: Backend) -> Self {
        let mixer = MixerConfig::default();
        let resolved = mixer.resolve();
        Self {
            backend,
            unresolved_clips: 0,
            warned_clips: std::collections::BTreeSet::new(),
            refused_voices: 0,
            master_volume: 1.0,
            sfx_volume: 1.0,
            music_volume: 1.0,
            listener: Listener::default(),
            voices: BTreeMap::new(),
            next_id: 0,
            mixer,
            resolved,
            sources: BTreeMap::new(),
            occlusion_hook: None,
        }
    }

    /// Whether a real audio device is driving playback. `false` in the no-device
    /// fallback — a caller can surface "muted" UI, but every API still works.
    pub fn is_active(&self) -> bool {
        self.backend.is_active()
    }

    // ── Named-bus mixer (P12.3) ─────────────────────────────────────────────────

    /// The active named-bus mixer configuration.
    pub fn mixer(&self) -> &MixerConfig {
        &self.mixer
    }

    /// Install a named-bus [`MixerConfig`], re-fold its gains/cutoffs, and re-push
    /// every affected voice's mix. The command-queue path plays voices on named
    /// buses, so this is how gameplay volume/effect settings reach the engine.
    pub fn set_mixer(&mut self, mixer: MixerConfig) {
        self.resolved = mixer.resolve();
        self.mixer = mixer;
        self.refresh_all();
    }

    /// The folded linear gain for a named mixer bus (`1.0` for an unknown bus).
    pub fn named_bus_gain(&self, name: &str) -> f64 {
        self.resolved.get(name).map(|r| r.gain).unwrap_or(1.0)
    }

    /// The folded (most-restrictive) lowpass cutoff for a named bus, if any. This
    /// is a **device-side** DSP value: it is modelled + inspectable here, but only
    /// audible on a `cpal` build once per-bus kira sub-tracks are wired (the
    /// documented follow-up). `None` = no bus in the chain filters.
    pub fn named_bus_lowpass_hz(&self, name: &str) -> Option<f64> {
        self.resolved.get(name).and_then(|r| r.cutoff_hz)
    }

    // ── Occlusion (P12.3) ───────────────────────────────────────────────────────

    /// Install (or clear) the [`OcclusionHook`]. When set, every spatial voice
    /// flagged for occlusion (re)computes its obstruction gain from the hook, and
    /// the engine multiplies that into the voice's gain. The caller supplies the
    /// factor (e.g. a physics raycast); the engine just multiplies.
    pub fn set_occlusion_hook(&mut self, hook: Option<OcclusionHook>) {
        self.occlusion_hook = hook;
        let ids: Vec<u64> = self.voices.keys().copied().collect();
        for id in ids {
            self.recompute_occlusion(id);
        }
        self.refresh_all();
    }

    /// Manually set a voice's obstruction gain (`[0, 1]`) and re-push its mix.
    /// The command-queue path uses this to deliver a sim-computed occlusion
    /// factor. Returns `false` for an unknown handle.
    pub fn set_occlusion(&mut self, handle: SoundHandle, gain: f64) -> bool {
        if let Some(v) = self.voices.get_mut(&handle.0) {
            v.occlusion_gain = gain.clamp(0.0, 1.0);
            self.refresh(handle.0);
            true
        } else {
            false
        }
    }

    /// Recompute one voice's occlusion gain from the hook (no-op if the voice is
    /// not occlusion-flagged, non-spatial, or there is no hook).
    fn recompute_occlusion(&mut self, id: u64) {
        let Some(hook) = self.occlusion_hook.as_ref() else {
            return;
        };
        if let Some(v) = self.voices.get_mut(&id) {
            if v.occlusion {
                if let Some(pos) = v.position {
                    v.occlusion_gain = hook(self.listener.position, pos).clamp(0.0, 1.0);
                }
            }
        }
    }

    // ── Volume model (built-in buses) ──────────────────────────────────────────

    /// The volume of a built-in bus (`Master` returns the global volume).
    pub fn bus_volume(&self, bus: Bus) -> f64 {
        match bus {
            Bus::Master => self.master_volume,
            Bus::Sfx => self.sfx_volume,
            Bus::Music => self.music_volume,
        }
    }

    /// Set a built-in bus's volume (clamped non-negative) and re-push every
    /// affected voice's mix. Setting `Master` scales the whole engine.
    pub fn set_bus_volume(&mut self, bus: Bus, volume: f64) {
        let v = volume.max(0.0);
        match bus {
            Bus::Master => self.master_volume = v,
            Bus::Sfx => self.sfx_volume = v,
            Bus::Music => self.music_volume = v,
        }
        self.refresh_all();
    }

    /// The global master volume (shorthand for `bus_volume(Bus::Master)`).
    pub fn master_volume(&self) -> f64 {
        self.master_volume
    }

    /// Set the global master volume (shorthand for `set_bus_volume(Master, …)`).
    pub fn set_master_volume(&mut self, volume: f64) {
        self.set_bus_volume(Bus::Master, volume);
    }

    // ── Listener ──────────────────────────────────────────────────────────────

    /// The current listener pose.
    pub fn listener(&self) -> Listener {
        self.listener
    }

    /// Move/orient the listener and re-push every spatial voice's mix (occlusion
    /// is recomputed too, since it depends on the listener position).
    pub fn set_listener(&mut self, listener: Listener) {
        self.listener = listener;
        let ids: Vec<u64> = self.voices.keys().copied().collect();
        for id in ids {
            self.recompute_occlusion(id);
        }
        self.refresh_all();
    }

    // ── Playback (direct API) ──────────────────────────────────────────────────

    /// How many `Play` commands named a clip that did not resolve (C4-43).
    /// Zero on a healthy level; every one of them is a source whose later
    /// `Stop` / `SetVolume` / `SetPitch` commands are also silently ignored,
    /// because the source never entered the table those commands look in.
    pub fn unresolved_clips(&self) -> u64 {
        self.unresolved_clips
    }

    /// How many voices the backend refused outright (C4-43).
    pub fn refused_voices(&self) -> u64 {
        self.refused_voices
    }

    /// Start a sound and return its handle. In the no-device fallback this still
    /// allocates a handle and tracks the voice (so `is_playing`, `effective_*`,
    /// etc. behave consistently); only the audible output is skipped.
    ///
    /// `None` when the backend refused the voice: its command queue was full,
    /// or the device would not take the sound. Never `None` in the no-device
    /// fallback, which tracks the voice so the reap seam can complete it.
    pub fn play(&mut self, data: &SoundData, settings: PlaySettings) -> Option<SoundHandle> {
        self.play_voice(
            data,
            Voice {
                bus: BusRef::Builtin(settings.bus),
                base_volume: settings.volume.max(0.0),
                pitch: settings.pitch,
                looping: settings.looping,
                paused: false,
                position: settings.position,
                attenuation: settings.attenuation,
                occlusion: settings.position.is_some(),
                occlusion_gain: 1.0,
            },
        )
    }

    /// Insert a fully-built voice, compute its mix (incl. occlusion hook), and
    /// hand it to the backend. Shared by [`play`](Self::play) and the command
    /// queue.
    /// Start a voice, or say the backend refused it (C4-43).
    ///
    /// `None` means **no voice exists** — the backend's queue was full, or the
    /// device rejected the sound. The old spelling inserted into `self.voices`
    /// unconditionally and returned a handle either way, so the engine held a
    /// voice the backend did not: `is_playing()` reported `true` for ever,
    /// `voice_count()` drifted upward, `drain_finished` could never reap it, and
    /// the caller was told it worked.
    fn play_voice(&mut self, data: &SoundData, mut voice: Voice) -> Option<SoundHandle> {
        let id = self.next_id;
        self.next_id += 1;
        // Apply the occlusion hook up front for hook-driven voices.
        if voice.occlusion {
            if let (Some(hook), Some(pos)) = (self.occlusion_hook.as_ref(), voice.position) {
                voice.occlusion_gain = hook(self.listener.position, pos).clamp(0.0, 1.0);
            }
        }
        let (gain, pan) = self.compute(&voice);
        let looping = voice.looping;
        let pitch = voice.pitch;
        self.voices.insert(id, voice);
        if self.backend.play(id, data, gain, pan, pitch, looping) {
            Some(SoundHandle(id))
        } else {
            // Never hold a voice the backend does not.
            self.voices.remove(&id);
            self.refused_voices += 1;
            None
        }
    }

    /// Stop a sound and forget its handle. Returns `false` if the handle was
    /// already stopped/unknown.
    pub fn stop(&mut self, handle: SoundHandle) -> bool {
        if self.voices.remove(&handle.0).is_some() {
            self.backend.stop(handle.0);
            // Drop any source mapping that pointed at this handle.
            self.sources.retain(|_, h| *h != handle);
            true
        } else {
            false
        }
    }

    /// Pause a sound. Returns `false` for an unknown handle.
    pub fn pause(&mut self, handle: SoundHandle) -> bool {
        if let Some(v) = self.voices.get_mut(&handle.0) {
            v.paused = true;
            self.backend.pause(handle.0);
            true
        } else {
            false
        }
    }

    /// Resume a paused sound. Returns `false` for an unknown handle.
    pub fn resume(&mut self, handle: SoundHandle) -> bool {
        if let Some(v) = self.voices.get_mut(&handle.0) {
            v.paused = false;
            self.backend.resume(handle.0);
            true
        } else {
            false
        }
    }

    /// Whether the handle names a live voice (from `play` until `stop`, or until a
    /// one-shot ends and [`reap`](Self::reap) removes it). The per-frame
    /// [`reap`](Self::reap) is what turns natural end-of-sound into "not playing".
    pub fn is_playing(&self, handle: SoundHandle) -> bool {
        self.voices.contains_key(&handle.0)
    }

    /// Drop voices that have finished playing on their own and any `sources`
    /// mapping that pointed at them. The **host** calls this once per frame after
    /// [`drain`](Self::drain): it touches only device-side bookkeeping (the same
    /// side as the backend), never the command stream, so the deterministic sim
    /// contract is untouched. Without it, finished one-shots would accumulate for
    /// the whole life of the process (and every listener move would re-mix them).
    pub fn reap(&mut self) {
        for id in self.backend.drain_finished() {
            self.voices.remove(&id);
            self.sources.retain(|_, h| h.0 != id);
        }
    }

    /// Whether the handle is a live, currently-paused voice.
    pub fn is_paused(&self, handle: SoundHandle) -> bool {
        self.voices
            .get(&handle.0)
            .map(|v| v.paused)
            .unwrap_or(false)
    }

    /// Whether the handle names a looping voice.
    pub fn is_looping(&self, handle: SoundHandle) -> bool {
        self.voices
            .get(&handle.0)
            .map(|v| v.looping)
            .unwrap_or(false)
    }

    /// Set a voice's base (pre-bus) volume and re-push its mix.
    pub fn set_volume(&mut self, handle: SoundHandle, volume: f64) -> bool {
        if let Some(v) = self.voices.get_mut(&handle.0) {
            v.base_volume = volume.max(0.0);
            self.refresh(handle.0);
            true
        } else {
            false
        }
    }

    /// Set a voice's pitch (playback-rate factor) and re-push its mix.
    pub fn set_pitch(&mut self, handle: SoundHandle, pitch: f64) -> bool {
        if let Some(v) = self.voices.get_mut(&handle.0) {
            v.pitch = pitch;
            self.refresh(handle.0);
            true
        } else {
            false
        }
    }

    /// Move a spatial voice's emitter and re-push its mix. Returns `false` for an
    /// unknown handle; a non-spatial voice becomes spatial from now on.
    pub fn set_position(&mut self, handle: SoundHandle, position: DVec3) -> bool {
        if self.voices.contains_key(&handle.0) {
            if let Some(v) = self.voices.get_mut(&handle.0) {
                v.position = Some(position);
            }
            self.recompute_occlusion(handle.0);
            self.refresh(handle.0);
            true
        } else {
            false
        }
    }

    /// The effective linear gain the engine is applying to a voice right now
    /// (`bus × base × spatial × occlusion`), or `None` for an unknown handle.
    /// Pure — the primary observable the no-device tests assert against.
    pub fn effective_volume(&self, handle: SoundHandle) -> Option<f64> {
        self.voices.get(&handle.0).map(|v| self.compute(v).0)
    }

    /// The effective stereo panning (`-1..1`) for a voice, or `None`. `0` for a
    /// non-spatial voice.
    pub fn effective_panning(&self, handle: SoundHandle) -> Option<f64> {
        self.voices.get(&handle.0).map(|v| self.compute(v).1)
    }

    /// Number of live voices (played, not yet stopped).
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    // ── Command queue (P12.3 drain-the-queue doctrine) ─────────────────────────

    /// Drain a batch of [`AudioCommand`]s produced by the sim into the engine,
    /// resolving each [`PlayCommand::clip`] GUID to a [`SoundData`] via `resolve`
    /// (an unresolvable clip is a silent no-op). This is the **only** bridge from
    /// deterministic sim state to the device: the command stream is the contract,
    /// the device effects here are not sim state. See [`crate::command`].
    pub fn drain(&mut self, cmds: &[AudioCommand], resolve: &dyn Fn(Uuid) -> Option<SoundData>) {
        for cmd in cmds {
            match cmd {
                AudioCommand::Play(p) => self.apply_play(p, resolve),
                AudioCommand::Stop { source } => {
                    if let Some(h) = self.sources.remove(source) {
                        self.stop(h);
                    }
                }
                AudioCommand::SetVolume { source, volume } => {
                    if let Some(&h) = self.sources.get(source) {
                        self.set_volume(h, *volume);
                    }
                }
                AudioCommand::SetPitch { source, pitch } => {
                    if let Some(&h) = self.sources.get(source) {
                        self.set_pitch(h, *pitch);
                    }
                }
                AudioCommand::SetListener(l) => self.set_listener(*l),
            }
        }
    }

    /// The live handle for a source id (from a drained `Play`), if any.
    pub fn source_handle(&self, source: u64) -> Option<SoundHandle> {
        self.sources.get(&source).copied()
    }

    fn apply_play(&mut self, p: &PlayCommand, resolve: &dyn Fn(Uuid) -> Option<SoundData>) {
        // One voice per source: replace any existing.
        if let Some(h) = self.sources.remove(&p.source) {
            self.stop(h);
        }
        let Some(data) = resolve(p.clip) else {
            // **An unresolvable clip is counted and named once** (C4-43). This
            // used to return in silence — and because the source never reached
            // `self.sources`, every subsequent `Stop` / `SetVolume` / `SetPitch`
            // for it was *also* a silent no-op. Under this crate's own doctrine
            // ("the stream is a pure function of the sim state") the stream has
            // diverged from the simulation, and nothing measured it.
            self.unresolved_clips += 1;
            if self.warned_clips.insert(p.clip) {
                tracing::warn!(
                    clip = %p.clip,
                    "audio clip did not resolve; this source is silent and every later \
                     command for it will be ignored"
                );
            }
            return;
        };
        let Some(handle) = self.play_voice(
            &data,
            Voice {
                bus: BusRef::Named(p.bus.clone()),
                base_volume: p.volume.max(0.0),
                pitch: p.pitch,
                looping: p.looping,
                paused: false,
                position: p.position,
                attenuation: p.attenuation,
                // The sim supplies the occlusion factor via the command; the hook
                // does not clobber it.
                occlusion: false,
                occlusion_gain: p.occlusion_gain.clamp(0.0, 1.0),
            },
        ) else {
            return; // the backend refused it; `play_voice` counted that
        };
        self.sources.insert(p.source, handle);
    }

    // ── internal ──────────────────────────────────────────────────────────────

    /// Linear factor for a built-in bus: master always applies; a per-bus factor
    /// applies on top for the non-master buses.
    fn builtin_bus_factor(&self, bus: Bus) -> f64 {
        self.master_volume
            * match bus {
                Bus::Master => 1.0,
                Bus::Sfx => self.sfx_volume,
                Bus::Music => self.music_volume,
            }
    }

    /// Linear factor for any [`BusRef`].
    fn bus_factor(&self, bus: &BusRef) -> f64 {
        match bus {
            BusRef::Builtin(b) => self.builtin_bus_factor(*b),
            BusRef::Named(name) => self.named_bus_gain(name),
        }
    }

    /// The `(gain, panning)` for a voice under the current buses + listener +
    /// occlusion.
    fn compute(&self, voice: &Voice) -> (f64, f64) {
        let bus = self.bus_factor(&voice.bus);
        match voice.position {
            Some(pos) => {
                let (spatial_gain, pan) = self.listener.resolve(pos, &voice.attenuation);
                (
                    bus * voice.base_volume * spatial_gain * voice.occlusion_gain,
                    pan,
                )
            }
            None => (bus * voice.base_volume, 0.0),
        }
    }

    fn refresh(&mut self, id: u64) {
        if let Some(voice) = self.voices.get(&id).cloned() {
            let (gain, pan) = self.compute(&voice);
            self.backend.set_params(id, gain, pan, voice.pitch);
        }
    }

    fn refresh_all(&mut self) {
        let ids: Vec<u64> = self.voices.keys().copied().collect();
        for id in ids {
            self.refresh(id);
        }
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}
