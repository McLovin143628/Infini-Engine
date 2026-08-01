//! The browser (wasm32) entry point (P14.2).
//!
//! [`start_player`] is exported to JavaScript via wasm-bindgen. It fetches the
//! cooked pack over HTTP, builds the world through the **same**
//! [`build_world_from_pack`](crate::build_world_from_pack) path the desktop
//! `--pack` boot uses, and runs the winit web event loop over a WebGPU surface
//! bound to the page's `<canvas>` — so a game runs in Chrome exactly as it does
//! on the desktop player.
//!
//! # Pack streaming (v1)
//!
//! v1 fetches the **whole** pack file, then parses it in memory
//! ([`inf_asset::PackReader::from_bytes`], decompressing blobs with the pure-Rust
//! `ruzstd` on wasm). HTTP range-request streaming (load the index, then fetch
//! blobs on demand) is a documented follow-up.
//!
//! # Honest runtime status
//!
//! This module is **compile-checked** in CI (`cargo check --target
//! wasm32-unknown-unknown`); the actual in-browser run is **human-verified** (CI
//! has no headless WebGPU). One runtime seam remains before a live run: async GPU
//! adapter acquisition. `inf-render`'s `GpuContext` requests its adapter with
//! `pollster::block_on`, which cannot block the browser's main thread — a WebGPU
//! run needs the async-adapter path (tracked in `docs/web-player.md`). Everything
//! else here — fetch, pack decode, world build, the winit canvas surface, and the
//! fixed-step loop on `web-time` — is wired and compiled.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// **JS entry point.** Start the player on the `<canvas id="{canvas_id}">`
/// element, booting the cooked pack fetched from `pack_url`. Returns immediately;
/// the game then runs on the browser's `requestAnimationFrame`. Any error is
/// logged to the devtools console.
///
/// ```js
/// import init, { start_player } from "./inf_player.js";
/// await init();
/// start_player("game", "./content.inf_pack");
/// ```
#[wasm_bindgen]
pub fn start_player(canvas_id: String, pack_url: String) {
    console_error_panic_hook::set_once();
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(e) = run(canvas_id, pack_url).await {
            web_sys::console::error_1(&JsValue::from_str(&format!("inf-player: {e}")));
        }
    });
}

/// Fetch → decode → build → run. Split out from [`start_player`] so the error
/// path is a single `?`-chain.
async fn run(canvas_id: String, pack_url: String) -> Result<(), String> {
    let canvas = canvas_by_id(&canvas_id)?;
    let bytes = fetch_bytes(&pack_url).await?;
    tracing::info!(
        "inf-player(web): fetched {} bytes from {pack_url}",
        bytes.len()
    );
    let reader = inf_asset::PackReader::from_bytes(bytes).map_err(|e| e.to_string())?;
    let source = crate::level::PackLevelSource::from_reader(reader, "web")?;
    let built = crate::build_world_from_pack(&source)?;
    let title = format!("Infinity Engine (Web) — {}", built.label);
    let mut sim = crate::sim_from_built(built);
    // Terrain streaming (P16.3b2) works here too: the in-memory `PackReader` still
    // hands `read_ref` a borrowed slice for an uncompressed (streaming-class)
    // entry, so tiles are sub-sliced with no copy. The load batch takes the serial
    // path on wasm (no thread pool) and produces the identical result.
    crate::attach_terrain_streaming(&mut sim, &crate::TerrainContent::Pack(source));
    crate::window::run_web(title, canvas, sim, crate::input::default_map())
}

/// Resolve `<canvas id="{id}">` from the DOM.
fn canvas_by_id(id: &str) -> Result<web_sys::HtmlCanvasElement, String> {
    let document = web_sys::window()
        .ok_or("no window")?
        .document()
        .ok_or("no document")?;
    let el = document
        .get_element_by_id(id)
        .ok_or_else(|| format!("no element #{id}"))?;
    el.dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| format!("#{id} is not a <canvas>"))
}

/// Fetch a URL and return its bytes (whole-file). Range-request streaming for
/// large packs is a documented follow-up.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let resp_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|e| format!("fetch: {e:?}"))?;
    let resp: web_sys::Response = resp_value
        .dyn_into()
        .map_err(|_| "fetch did not return a Response".to_string())?;
    if !resp.ok() {
        return Err(format!("HTTP {} for {url}", resp.status()));
    }
    let buf = JsFuture::from(
        resp.array_buffer()
            .map_err(|e| format!("array_buffer: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("array_buffer await: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}
