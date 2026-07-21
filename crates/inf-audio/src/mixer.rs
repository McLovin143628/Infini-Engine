//! The named-bus **mixer** (P12.3): a serde [`MixerConfig`] of hierarchical buses
//! with per-bus volume and an effect chain, persisted deterministically at
//! `.infinity/mixer.toml`.
//!
//! # What is engine-side vs. device-side
//!
//! The whole config is **pure data** and folds to a per-bus linear gain +
//! most-restrictive lowpass cutoff by [`MixerConfig::resolve`] — no kira, no
//! device, fully deterministic and unit-tested. The [`AudioEngine`] applies the
//! folded **gain** (bus volume × parent chain × [`Effect::Gain`]) to every voice
//! on that bus, so the mix is correct even in the no-device fallback (and in CI).
//!
//! [`Effect::Lowpass`] is a **DSP** effect: it has no meaning without an output
//! device. kira 0.12 *does* expose it as a track effect (`FilterBuilder` with
//! `FilterMode::LowPass`), so the config models it and `resolve` folds the
//! cutoff; wiring per-bus kira sub-tracks so the filter is *audible* on the `cpal`
//! build is the documented device-side follow-up (see [`backend`](crate::backend)).
//! In the no-device path the cutoff is inspectable but silent, exactly like every
//! other device effect.
//!
//! [`AudioEngine`]: crate::AudioEngine

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The current [`MixerConfig`] schema version. Bumped when the on-disk shape
/// changes; `MixerConfig` accepts equal-or-older and rejects newer.
pub const MIXER_SCHEMA_VERSION: u32 = 1;

/// Relative path (under a project root) of the persisted mixer config.
pub const MIXER_REL_PATH: &str = ".infinity/mixer.toml";

/// One effect in a bus's chain. v1 is deliberately tiny: a compute-side
/// [`Gain`](Effect::Gain) (fully engine-side) and a device-side
/// [`Lowpass`](Effect::Lowpass) (modelled + folded; audible wiring is the `cpal`
/// follow-up). Reverb / delay / sends are P12.3+ (kira exposes them all).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Effect {
    /// A gain trim in decibels, folded into the bus's linear gain (engine-side,
    /// deterministic). `0 dB` = unity, `-6 dB` ≈ half amplitude.
    Gain { db: f64 },
    /// A low-pass filter cutoff in hertz (device-side DSP). Folded as the
    /// *most restrictive* (minimum) cutoff along the bus chain.
    Lowpass { cutoff_hz: f64 },
}

/// A single mixer bus: a name, an optional parent (for hierarchy), a linear
/// volume, and an ordered effect chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bus {
    /// Unique bus name (the key the ECS `AudioSource.bus` field references).
    pub name: String,
    /// Parent bus name; `None` for a root (typically `"master"`). A bus folds its
    /// gain up through this chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// Linear volume for this bus (`1.0` = unity), clamped non-negative on fold.
    pub volume: f64,
    /// Ordered effect chain applied on this bus.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<Effect>,
}

impl Bus {
    /// A plain bus with the given name/parent at unity volume, no effects.
    pub fn new(name: impl Into<String>, parent: Option<&str>) -> Self {
        Self {
            name: name.into(),
            parent: parent.map(str::to_string),
            volume: 1.0,
            effects: Vec::new(),
        }
    }

    /// This bus's own linear factor: `volume × Π(gain effects as linear)`.
    /// Lowpass effects do not affect gain.
    fn local_gain(&self) -> f64 {
        let mut g = self.volume.max(0.0);
        for e in &self.effects {
            if let Effect::Gain { db } = e {
                g *= db_to_linear(*db);
            }
        }
        g
    }

    /// The tightest (minimum) lowpass cutoff declared on this bus, if any.
    fn local_cutoff(&self) -> Option<f64> {
        self.effects
            .iter()
            .filter_map(|e| match e {
                Effect::Lowpass { cutoff_hz } => Some(*cutoff_hz),
                _ => None,
            })
            .fold(None, |acc, c| Some(acc.map_or(c, |a: f64| a.min(c))))
    }
}

/// The folded result for one bus: the linear gain to multiply every voice on it
/// by, and the most-restrictive lowpass cutoff along its chain (device-side).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedBus {
    /// Linear gain (bus volume × parent chain × gain effects). Non-negative.
    pub gain: f64,
    /// Folded lowpass cutoff in hertz, or `None` if no bus in the chain filters.
    pub cutoff_hz: Option<f64>,
}

/// A hierarchical mixer configuration — the serde/TOML face of the named-bus
/// model. Deterministic: buses serialize in declaration order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MixerConfig {
    /// On-disk schema version (see [`MIXER_SCHEMA_VERSION`]).
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// The buses, in declaration order (order is preserved on save).
    pub buses: Vec<Bus>,
}

fn default_schema_version() -> u32 {
    MIXER_SCHEMA_VERSION
}

impl Default for MixerConfig {
    /// The default project mixer: a `master` root with `sfx` and `music`
    /// children, all at unity, no effects. Matches the engine's built-in buses.
    fn default() -> Self {
        Self {
            schema_version: MIXER_SCHEMA_VERSION,
            buses: vec![
                Bus::new("master", None),
                Bus::new("sfx", Some("master")),
                Bus::new("music", Some("master")),
            ],
        }
    }
}

impl MixerConfig {
    /// Look up a bus by name.
    pub fn bus(&self, name: &str) -> Option<&Bus> {
        self.buses.iter().find(|b| b.name == name)
    }

    /// Fold every bus into its [`ResolvedBus`] (gain up the parent chain + folded
    /// cutoff). Pure and deterministic. Cycles / missing parents terminate the
    /// walk safely (a bus whose parent is missing folds as if it were a root).
    pub fn resolve(&self) -> BTreeMap<String, ResolvedBus> {
        let index: BTreeMap<&str, &Bus> = self.buses.iter().map(|b| (b.name.as_str(), b)).collect();
        let mut out = BTreeMap::new();
        for bus in &self.buses {
            let mut gain = 1.0f64;
            let mut cutoff: Option<f64> = None;
            let mut cur = Some(bus);
            // Depth guard = bus count; a cycle can visit at most `len` distinct
            // buses before repeating, and we break on revisit.
            let mut seen = std::collections::BTreeSet::new();
            let mut steps = 0usize;
            while let Some(b) = cur {
                if !seen.insert(b.name.as_str()) || steps > self.buses.len() {
                    break;
                }
                steps += 1;
                gain *= b.local_gain();
                if let Some(c) = b.local_cutoff() {
                    cutoff = Some(cutoff.map_or(c, |a: f64| a.min(c)));
                }
                cur = b.parent.as_deref().and_then(|p| index.get(p).copied());
            }
            out.insert(
                bus.name.clone(),
                ResolvedBus {
                    gain: gain.max(0.0),
                    cutoff_hz: cutoff,
                },
            );
        }
        out
    }

    /// Validate + accept an equal-or-older config (rejects a newer schema),
    /// mirroring the asset-payload migrate guard.
    pub fn migrate(self) -> Result<Self, String> {
        if self.schema_version > MIXER_SCHEMA_VERSION {
            return Err(format!(
                "mixer.toml schema v{} is newer than supported v{MIXER_SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        Ok(self)
    }

    /// Deserialize from a TOML string (with the migrate guard).
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        toml::from_str::<MixerConfig>(s)
            .map_err(|e| e.to_string())?
            .migrate()
    }

    /// Serialize to a deterministic pretty-TOML string.
    pub fn to_toml_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// The mixer path under a project root (`<root>/.infinity/mixer.toml`).
    pub fn path_in(root: &Path) -> PathBuf {
        root.join(MIXER_REL_PATH)
    }

    /// Load the mixer from `<root>/.infinity/mixer.toml`, or the [`Default`]
    /// config if the file is absent. A parse error is surfaced (not swallowed).
    pub fn load_or_default(root: &Path) -> Result<Self, String> {
        let path = Self::path_in(root);
        match std::fs::read_to_string(&path) {
            Ok(s) => Self::from_toml_str(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    /// Write the mixer to `<root>/.infinity/mixer.toml`, creating `.infinity/`.
    pub fn save(&self, root: &Path) -> Result<(), String> {
        let path = Self::path_in(root);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
        std::fs::write(&path, self.to_toml_string()?)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }
}

/// Convert decibels to a linear amplitude factor (`10^(db/20)`).
pub fn db_to_linear(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_folds_to_unity() {
        let cfg = MixerConfig::default();
        let r = cfg.resolve();
        assert!((r["master"].gain - 1.0).abs() < 1e-12);
        assert!((r["sfx"].gain - 1.0).abs() < 1e-12);
        assert!((r["music"].gain - 1.0).abs() < 1e-12);
        assert_eq!(r["sfx"].cutoff_hz, None);
    }

    #[test]
    fn hierarchy_folds_parent_volume() {
        let cfg = MixerConfig {
            schema_version: MIXER_SCHEMA_VERSION,
            buses: vec![
                Bus {
                    name: "master".into(),
                    parent: None,
                    volume: 0.5,
                    effects: vec![],
                },
                Bus {
                    name: "sfx".into(),
                    parent: Some("master".into()),
                    volume: 0.5,
                    effects: vec![],
                },
            ],
        };
        let r = cfg.resolve();
        // sfx folds master: 0.5 × 0.5 = 0.25.
        assert!((r["sfx"].gain - 0.25).abs() < 1e-12);
        assert!((r["master"].gain - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gain_effect_folds_as_linear_and_lowpass_takes_minimum() {
        let cfg = MixerConfig {
            schema_version: MIXER_SCHEMA_VERSION,
            buses: vec![
                Bus {
                    name: "master".into(),
                    parent: None,
                    volume: 1.0,
                    effects: vec![Effect::Lowpass { cutoff_hz: 2000.0 }],
                },
                Bus {
                    name: "sfx".into(),
                    parent: Some("master".into()),
                    volume: 1.0,
                    // -6 dB ≈ 0.501 linear; plus a tighter 800 Hz lowpass.
                    effects: vec![
                        Effect::Gain { db: -6.0 },
                        Effect::Lowpass { cutoff_hz: 800.0 },
                    ],
                },
            ],
        };
        let r = cfg.resolve();
        assert!((r["sfx"].gain - db_to_linear(-6.0)).abs() < 1e-12);
        // Most-restrictive cutoff along the chain (min of 800 and 2000).
        assert_eq!(r["sfx"].cutoff_hz, Some(800.0));
        assert_eq!(r["master"].cutoff_hz, Some(2000.0));
    }

    #[test]
    fn toml_round_trips_deterministically() {
        let cfg = MixerConfig::default();
        let s = cfg.to_toml_string().unwrap();
        let back = MixerConfig::from_toml_str(&s).unwrap();
        assert_eq!(cfg, back);
        // Stable across re-serialize.
        assert_eq!(s, back.to_toml_string().unwrap());
    }

    #[test]
    fn migrate_rejects_newer_schema() {
        let cfg = MixerConfig {
            schema_version: MIXER_SCHEMA_VERSION + 1,
            buses: vec![Bus::new("master", None)],
        };
        assert!(cfg.migrate().is_err());
    }

    #[test]
    fn missing_parent_folds_as_root() {
        let cfg = MixerConfig {
            schema_version: MIXER_SCHEMA_VERSION,
            buses: vec![Bus {
                name: "sfx".into(),
                parent: Some("ghost".into()),
                volume: 0.5,
                effects: vec![],
            }],
        };
        let r = cfg.resolve();
        assert!((r["sfx"].gain - 0.5).abs() < 1e-12);
    }
}
