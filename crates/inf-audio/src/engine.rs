//! [`AudioEngine`]: the facade — a master + named-bus volume model, a spatial
//! listener, and per-sound play/stop/pause handles over the kira [`Backend`].
//!
//! All mixing is computed **engine-side** (buses fold into one linear gain per
//! voice, spatial position into gain + panning) and pushed to kira as a plain
//! per-voice volume/panning/rate. That keeps the kira surface tiny (just "set this
//! voice's params") and means the entire mix model is pure, inspectable, and
//! testable without an audio device — which is exactly what the no-device fallback
//! and CI rely on.

use std::collections::BTreeMap;

use glam::DVec3;

use crate::backend::Backend;
use crate::sound::SoundData;
use crate::spatial::{Attenuation, Listener};

/// A mixer bus. Minimal by design (P9.1b baseline); the full mixer + effect
/// graph is P12.3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bus {
    /// The global bus — its volume scales everything.
    Master,
    /// Sound effects.
    Sfx,
    /// Music.
    Music,
}

/// An opaque handle to a playing (or paused) sound. Stable for the voice's
/// lifetime; a stopped handle reads back as not-playing rather than aliasing a
/// new voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoundHandle(u64);

/// How to start a sound. Build with [`PlaySettings::default`] (or the `on`/
/// `spatial` helpers) and hand to [`AudioEngine::play`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaySettings {
    /// Which bus the sound plays on.
    pub bus: Bus,
    /// Base linear volume (`1.0` = unity), before bus/master/spatial scaling.
    pub volume: f64,
    /// Playback-rate factor (`1.0` = normal pitch/speed).
    pub pitch: f64,
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
}

/// Per-voice engine-side state (independent of whether a device is active).
#[derive(Clone, Copy, Debug)]
struct Voice {
    bus: Bus,
    base_volume: f64,
    pitch: f64,
    paused: bool,
    position: Option<DVec3>,
    attenuation: Attenuation,
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
        Self {
            backend,
            master_volume: 1.0,
            sfx_volume: 1.0,
            music_volume: 1.0,
            listener: Listener::default(),
            voices: BTreeMap::new(),
            next_id: 0,
        }
    }

    /// Whether a real audio device is driving playback. `false` in the no-device
    /// fallback — a caller can surface "muted" UI, but every API still works.
    pub fn is_active(&self) -> bool {
        self.backend.is_active()
    }

    // ── Volume model ──────────────────────────────────────────────────────────

    /// The volume of a bus (`Master` returns the global volume).
    pub fn bus_volume(&self, bus: Bus) -> f64 {
        match bus {
            Bus::Master => self.master_volume,
            Bus::Sfx => self.sfx_volume,
            Bus::Music => self.music_volume,
        }
    }

    /// Set a bus's volume (clamped non-negative) and re-push every affected
    /// voice's mix. Setting `Master` scales the whole engine.
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

    /// Move/orient the listener and re-push every spatial voice's mix.
    pub fn set_listener(&mut self, listener: Listener) {
        self.listener = listener;
        self.refresh_all();
    }

    // ── Playback ──────────────────────────────────────────────────────────────

    /// Start a sound and return its handle. In the no-device fallback this still
    /// allocates a handle and tracks the voice (so `is_playing`, `effective_*`,
    /// etc. behave consistently); only the audible output is skipped.
    pub fn play(&mut self, data: &SoundData, settings: PlaySettings) -> SoundHandle {
        let id = self.next_id;
        self.next_id += 1;
        let voice = Voice {
            bus: settings.bus,
            base_volume: settings.volume.max(0.0),
            pitch: settings.pitch,
            paused: false,
            position: settings.position,
            attenuation: settings.attenuation,
        };
        let (gain, pan) = self.compute(&voice);
        self.voices.insert(id, voice);
        self.backend.play(id, data, gain, pan, voice.pitch);
        SoundHandle(id)
    }

    /// Stop a sound and forget its handle. Returns `false` if the handle was
    /// already stopped/unknown.
    pub fn stop(&mut self, handle: SoundHandle) -> bool {
        if self.voices.remove(&handle.0).is_some() {
            self.backend.stop(handle.0);
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

    /// Whether the handle names a live voice (from `play` until `stop`). This
    /// tracks the **facade lifecycle**, not natural end-of-sound — wiring kira's
    /// completion signal so a finished one-shot auto-reaps is a documented
    /// follow-up (it needs a per-frame poll the runtime loop will own).
    pub fn is_playing(&self, handle: SoundHandle) -> bool {
        self.voices.contains_key(&handle.0)
    }

    /// Whether the handle is a live, currently-paused voice.
    pub fn is_paused(&self, handle: SoundHandle) -> bool {
        self.voices
            .get(&handle.0)
            .map(|v| v.paused)
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
        if let Some(v) = self.voices.get_mut(&handle.0) {
            v.position = Some(position);
            self.refresh(handle.0);
            true
        } else {
            false
        }
    }

    /// The effective linear gain the engine is applying to a voice right now
    /// (`master × bus × base × spatial`), or `None` for an unknown handle. Pure —
    /// the primary observable the no-device tests assert against.
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

    // ── internal ──────────────────────────────────────────────────────────────

    fn bus_factor(&self, bus: Bus) -> f64 {
        // The global master always applies; a per-bus factor applies on top for
        // the non-master buses.
        self.master_volume
            * match bus {
                Bus::Master => 1.0,
                Bus::Sfx => self.sfx_volume,
                Bus::Music => self.music_volume,
            }
    }

    /// The `(gain, panning)` for a voice under the current buses + listener.
    fn compute(&self, voice: &Voice) -> (f64, f64) {
        let bus = self.bus_factor(voice.bus);
        match voice.position {
            Some(pos) => {
                let (spatial_gain, pan) = self.listener.resolve(pos, &voice.attenuation);
                (bus * voice.base_volume * spatial_gain, pan)
            }
            None => (bus * voice.base_volume, 0.0),
        }
    }

    fn refresh(&mut self, id: u64) {
        if let Some(voice) = self.voices.get(&id).copied() {
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
