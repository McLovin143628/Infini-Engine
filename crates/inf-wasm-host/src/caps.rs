//! Capability grants (P14.5, deliverable 1) — the deny-by-default surface.
//!
//! A mod is granted a set of [`ModCaps`]; only the host functions its granted
//! capabilities cover are linked into its sandbox. A module that imports a host
//! function whose capability was **not** granted fails to instantiate with a
//! clear, capability-anchored message (see [`crate::host`]). Nothing is granted
//! implicitly: an all-false [`ModCaps`] links *no* host functions at all.

use serde::{Deserialize, Serialize};

/// The four host-API capability groups a mod can be granted. Deny-by-default:
/// [`ModCaps::NONE`] (all `false`) links no host functions.
///
/// Each flag gates a fixed set of host imports (module `"env"`):
///
/// | cap        | host imports it unlocks                              |
/// |------------|------------------------------------------------------|
/// | `entities` | `entity_translation`, `set_entity_translation`       |
/// | `input`    | `input_is_down`                                       |
/// | `log`      | `log`                                                 |
/// | `spawn`    | `spawn_cube`                                          |
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModCaps {
    /// Read/write entity transforms (`entity_translation`, `set_entity_translation`).
    #[serde(default)]
    pub entities: bool,
    /// Query held input actions (`input_is_down`).
    #[serde(default)]
    pub input: bool,
    /// Emit log lines (`log`).
    #[serde(default)]
    pub log: bool,
    /// Spawn new entities (`spawn_cube`).
    #[serde(default)]
    pub spawn: bool,
}

impl ModCaps {
    /// The deny-by-default grant: nothing.
    pub const NONE: ModCaps = ModCaps {
        entities: false,
        input: false,
        log: false,
        spawn: false,
    };

    /// Every capability — for the editor's own trusted first-party mods / tests.
    pub const ALL: ModCaps = ModCaps {
        entities: true,
        input: true,
        log: true,
        spawn: true,
    };

    /// Whether `import` (a host-function name in module `"env"`) is a known host
    /// import, and if so which capability grants it. `None` → not a host import
    /// this ABI defines at all.
    pub fn cap_for_import(import: &str) -> Option<Capability> {
        Some(match import {
            "entity_translation" | "set_entity_translation" => Capability::Entities,
            "input_is_down" => Capability::Input,
            "log" => Capability::Log,
            "spawn_cube" => Capability::Spawn,
            _ => return None,
        })
    }

    /// Whether this grant includes `cap`.
    pub fn has(&self, cap: Capability) -> bool {
        match cap {
            Capability::Entities => self.entities,
            Capability::Input => self.input,
            Capability::Log => self.log,
            Capability::Spawn => self.spawn,
        }
    }
}

/// A single capability group, used in error messages + the import→cap map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Entities,
    Input,
    Log,
    Spawn,
}

impl Capability {
    /// The lowercase manifest/name spelling (`"entities"`, …).
    pub fn name(self) -> &'static str {
        match self {
            Capability::Entities => "entities",
            Capability::Input => "input",
            Capability::Log => "log",
            Capability::Spawn => "spawn",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
