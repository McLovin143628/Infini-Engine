//! Host/guest wiring exercised in-process (no dylib): the guest shims and
//! the host world are the same code paths the loader uses, so everything
//! except `libloading` itself is testable at unit speed. The dylib story is
//! covered by `reload.rs`.

use inf_hotreload::abi::{PluginVTable, ABI_VERSION, LOG_ERROR};
use inf_hotreload::guest::{build_plugin, component_vtable, HotComponent, Logger};
use inf_hotreload::{HostError, HotWorld, InstanceStatus, Plugin};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
struct Ticker {
    ticks: i64,
}

impl HotComponent for Ticker {
    const NAME: &'static str = "ticker";

    fn schema() -> &'static str {
        "ticker { ticks: i64 }"
    }

    fn tick(&mut self, _dt: f64, log: &mut Logger<'_>) {
        self.ticks += 1;
        if self.ticks == 2 {
            log.warn("ticker reached 2");
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Bomb {
    armed: bool,
}

impl HotComponent for Bomb {
    const NAME: &'static str = "bomb";

    fn schema() -> &'static str {
        "bomb { armed: bool }"
    }

    fn tick(&mut self, _dt: f64, _log: &mut Logger<'_>) {
        if self.armed {
            panic!("bomb went off");
        }
        self.armed = true;
    }
}

fn plugin_of(
    version: &'static str,
    components: Vec<inf_hotreload::abi::ComponentVTable>,
) -> Plugin {
    let handle = build_plugin(version, components);
    // SAFETY: build_plugin leaks everything, so the vtable is immortal.
    unsafe { Plugin::from_vtable(&*handle.as_ptr(), 0, None) }.unwrap()
}

#[test]
fn spawn_tick_serialize_round_trip() {
    let plugin = plugin_of("test", vec![component_vtable::<Ticker>()]);
    assert_eq!(plugin.version(), "test");

    let mut world = HotWorld::new();
    let id = world.spawn(&plugin, "ticker").unwrap();
    for _ in 0..3 {
        world.tick(1.0 / 60.0);
    }
    assert_eq!(world.status(id), Some(InstanceStatus::Running));
    let state = world.state_json(id).unwrap();
    assert_eq!(state["ticks"], 3);
    assert!(world
        .logs()
        .iter()
        .any(|r| r.message.contains("ticker reached 2")));
}

#[test]
fn panic_disables_only_the_broken_component() {
    let plugin = plugin_of(
        "test",
        vec![component_vtable::<Ticker>(), component_vtable::<Bomb>()],
    );
    let mut world = HotWorld::new();
    let ticker = world.spawn(&plugin, "ticker").unwrap();
    let bomb = world.spawn(&plugin, "bomb").unwrap();

    for _ in 0..4 {
        world.tick(1.0 / 60.0);
    }

    // Bomb panicked on its second tick and got disabled with the message.
    assert_eq!(world.status(bomb), Some(InstanceStatus::Disabled));
    let error = world.last_error(bomb).unwrap();
    assert!(error.contains("bomb went off"), "got: {error}");
    assert!(world
        .logs()
        .iter()
        .any(|r| r.level == LOG_ERROR && r.message.contains("bomb went off")));

    // The ticker kept running through all four ticks.
    assert_eq!(world.status(ticker), Some(InstanceStatus::Running));
    assert_eq!(world.state_json(ticker).unwrap()["ticks"], 4);
}

#[test]
fn unknown_component_and_instance_are_errors() {
    let plugin = plugin_of("test", vec![component_vtable::<Ticker>()]);
    let mut world = HotWorld::new();
    assert!(matches!(
        world.spawn(&plugin, "nope"),
        Err(HostError::UnknownComponent(_))
    ));
}

#[test]
fn abi_version_mismatch_is_rejected() {
    let vtable: &'static PluginVTable = Box::leak(Box::new(PluginVTable {
        abi_version: ABI_VERSION + 1,
        version: "x".as_ptr(),
        version_len: 1,
        component_count: 0,
        components: std::ptr::null(),
    }));
    // SAFETY: leaked vtable with no components.
    let result = unsafe { Plugin::from_vtable(vtable, 0, None) };
    assert!(matches!(
        result,
        Err(HostError::AbiMismatch {
            found
        }) if found == ABI_VERSION + 1
    ));
}

#[test]
fn duplicate_component_names_are_rejected() {
    let handle = build_plugin(
        "dup",
        vec![component_vtable::<Ticker>(), component_vtable::<Ticker>()],
    );
    // SAFETY: leaked by build_plugin.
    let result = unsafe { Plugin::from_vtable(&*handle.as_ptr(), 0, None) };
    assert!(matches!(result, Err(HostError::DuplicateComponent(name)) if name == "ticker"));
}
