//! Hot-reload test fixture, version 1: `counter` advances by 1 per tick and
//! `fragile` deliberately panics on its third tick (panic-containment case).

use inf_hotreload::guest::{HotComponent, Logger};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub struct Counter {
    ticks: i64,
}

impl HotComponent for Counter {
    const NAME: &'static str = "counter";

    fn schema() -> &'static str {
        "counter { ticks: i64 }"
    }

    fn tick(&mut self, _dt: f64, log: &mut Logger<'_>) {
        self.ticks += 1;
        if self.ticks == 1 {
            log.info("counter v1: first tick");
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
        if self.count == 3 {
            panic!("fragile v1 exploded at count 3");
        }
    }
}

inf_hotreload::export_plugin!("1.0.0", [Counter, Fragile]);
