//! Hot-reload test fixture, version 2: `counter` grew a `bonus` field
//! (schema migration — old snapshots default it) and now advances by 10 per
//! tick; `fragile` no longer panics (the reload re-enables it).

use inf_hotreload::guest::{HotComponent, Logger};
use serde::{Deserialize, Serialize};

fn default_bonus() -> f64 {
    1.5
}

#[derive(Serialize, Deserialize)]
pub struct Counter {
    ticks: i64,
    #[serde(default = "default_bonus")]
    bonus: f64,
}

impl Default for Counter {
    fn default() -> Self {
        Counter {
            ticks: 0,
            bonus: default_bonus(),
        }
    }
}

impl HotComponent for Counter {
    const NAME: &'static str = "counter";

    fn schema() -> &'static str {
        "counter { ticks: i64, bonus: f64 }"
    }

    fn tick(&mut self, _dt: f64, log: &mut Logger<'_>) {
        self.ticks += 10;
        if self.ticks == 10 {
            log.info("counter v2: first tick");
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct Fragile {
    count: i64,
}

impl HotComponent for Fragile {
    const NAME: &'static str = "fragile";

    fn schema() -> &'static str {
        "fragile { count: i64 }"
    }

    fn tick(&mut self, _dt: f64, _log: &mut Logger<'_>) {
        self.count += 1;
    }
}

inf_hotreload::export_plugin!("2.0.0", [Counter, Fragile]);
