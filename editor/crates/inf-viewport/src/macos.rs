//! macOS viewport host: a `CAMetalLayer` sublayer added to the Tauri
//! window's layer-backed contentView, rendered by wgpu (Metal) from a
//! dedicated thread.
//!
//! Status: COMPILE-VERIFIED ONLY (cross-checked from Windows against
//! aarch64-apple-darwin; CI compiles it on real macOS). Runtime behavior —
//! retina scale, coordinate flip, cross-thread CALayer geometry writes —
//! needs a hardware pass; see the Spike A memo.
//!
//! Layer-frame updates run on the render thread inside a `CATransaction`
//! with actions disabled. Core Animation tolerates off-main-thread layer
//! property writes on standalone sublayers, but this is exactly the kind of
//! claim the hardware pass must confirm; if it flickers, the fallback is
//! dispatching frame updates to the main queue.
//!
//! Flycam input is not wired on macOS yet (Windows raw-input only for the
//! spike); the camera holds its default pose.

use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};

use objc2::rc::Retained;
use objc2::MainThreadMarker;
use objc2_app_kit::NSView;
use objc2_core_foundation::{CGPoint, CGRect, CGSize};
use objc2_quartz_core::{CAMetalLayer, CATransaction};

use crate::camera::EditorCamera;
use crate::host::EngineHost;
use crate::{SharedScene, SurfaceTarget, ViewportEventSink, ViewportRect};

enum Cmd {
    SetRect(ViewportRect),
    SetVisible(bool),
    Drop { x: f32, y: f32, payload: String },
    Destroy,
}

/// Cheap-to-clone handle for controlling the viewport thread.
pub struct ViewportHandle {
    tx: Sender<Cmd>,
}

impl ViewportHandle {
    /// Move/resize the layer (physical pixels, top-left origin relative to
    /// the parent view — the same contract as Windows).
    pub fn set_rect(&self, rect: ViewportRect) {
        let _ = self.tx.send(Cmd::SetRect(rect));
    }

    /// Show/hide the layer (see the Windows twin for why: HTML overlays
    /// crossing the hole are otherwise occluded by the native surface).
    pub fn set_visible(&self, visible: bool) {
        let _ = self.tx.send(Cmd::SetVisible(visible));
    }

    /// Drag-drop handoff (see the Windows twin for the contract).
    pub fn drop_payload(&self, x: f32, y: f32, payload: &str) {
        let _ = self.tx.send(Cmd::Drop {
            x,
            y,
            payload: payload.to_owned(),
        });
    }

    /// Tear down the viewport thread and its layer.
    pub fn destroy(&self) {
        let _ = self.tx.send(Cmd::Destroy);
    }
}

/// Create the CAMetalLayer under `ns_view` (the Tauri contentView) and start
/// the render thread. MUST be called on the AppKit main thread; the Ring-2
/// caller dispatches via `run_on_main_thread`.
pub fn spawn(ns_view: isize, sink: ViewportEventSink, scene: SharedScene) -> ViewportHandle {
    let (tx, rx) = channel();

    if MainThreadMarker::new().is_none() {
        tracing::error!("inf-viewport: macOS spawn must run on the main thread");
        return ViewportHandle { tx };
    }

    // SAFETY: the caller passes the live contentView of the editor window
    // and we are on the main thread.
    let (layer_ptr, scale) = unsafe {
        let view = &*(ns_view as *const NSView);
        view.setWantsLayer(true);
        let scale = view.window().map(|w| w.backingScaleFactor()).unwrap_or(2.0);

        let metal = CAMetalLayer::new();
        metal.setContentsScale(scale);
        metal.setFrame(CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 64.0,
                height: 64.0,
            },
        });
        match view.layer() {
            Some(root) => root.addSublayer(&metal),
            None => {
                tracing::error!("inf-viewport: contentView has no backing layer");
                return ViewportHandle { tx };
            }
        }
        // Intentional +1 retain: the layer lives until Destroy releases it.
        (Retained::into_raw(metal) as isize, scale)
    };

    // macOS input (flycam/orbit + key forwarding) isn't wired yet — the camera
    // holds its default pose, so there are no events to surface. Kept in the
    // signature for parity with the Windows host; the hardware pass wires it.
    let _ = sink;

    std::thread::Builder::new()
        .name("inf-viewport".into())
        .spawn(move || thread_main(layer_ptr, scale, rx, scene))
        .expect("failed to spawn inf-viewport thread");
    ViewportHandle { tx }
}

fn apply_rect(layer_ptr: isize, scale: f64, r: ViewportRect) {
    // SAFETY: layer_ptr holds a +1 retain until Destroy.
    unsafe {
        let layer = &*(layer_ptr as *const CAMetalLayer);
        // Physical px (top-left origin) → points (bottom-left origin).
        let super_h = layer
            .superlayer()
            .map(|s| s.bounds().size.height)
            .unwrap_or(0.0);
        let w = r.width as f64 / scale;
        let h = r.height as f64 / scale;
        let x = r.x as f64 / scale;
        let y = super_h - (r.y as f64 / scale + h);
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        layer.setFrame(CGRect {
            origin: CGPoint { x, y },
            size: CGSize {
                width: w,
                height: h,
            },
        });
        CATransaction::commit();
    }
}

fn thread_main(layer_ptr: isize, scale: f64, rx: Receiver<Cmd>, scene: SharedScene) {
    let target = SurfaceTarget::MetalLayer { layer: layer_ptr };
    let mut host = match EngineHost::new(target, 64, 64) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!("inf-viewport: engine init failed: {e}");
            return;
        }
    };
    tracing::info!("inf-viewport: CAMetalLayer + engine renderer up");

    let camera = EditorCamera::default();

    'outer: loop {
        let mut latest_rect: Option<ViewportRect> = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::SetRect(r)) => latest_rect = Some(r),
                Ok(Cmd::SetVisible(v)) => unsafe {
                    // SAFETY: layer_ptr holds a +1 retain until Destroy.
                    let layer = &*(layer_ptr as *const CAMetalLayer);
                    CATransaction::begin();
                    CATransaction::setDisableActions(true);
                    layer.setHidden(!v);
                    CATransaction::commit();
                },
                Ok(Cmd::Drop { x, y, payload }) => {
                    tracing::info!(
                        "inf-viewport: drop '{payload}' at viewport-local ({x:.0}, {y:.0}) px"
                    );
                }
                Ok(Cmd::Destroy) | Err(TryRecvError::Disconnected) => break 'outer,
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(r) = latest_rect {
            apply_rect(layer_ptr, scale, r);
            host.resize(r.width.max(1), r.height.max(1));
        }

        // Project the shared world (read-only on macOS: no input wired yet, so
        // no picking/gizmo writeback — the editor still drives the scene).
        if let Ok(doc) = scene.lock() {
            host.sync_from_doc(&doc);
        }

        // FIFO present blocks at vsync and paces the loop.
        if let Err(e) = host.render_frame(&camera) {
            tracing::error!("inf-viewport: unrecoverable render failure: {e}");
            break;
        }
    }

    // SAFETY: reclaim the +1 retain taken in `spawn`; drop releases it.
    unsafe {
        let layer = Retained::from_raw(layer_ptr as *mut CAMetalLayer);
        if let Some(layer) = layer {
            CATransaction::begin();
            CATransaction::setDisableActions(true);
            layer.removeFromSuperlayer();
            CATransaction::commit();
        }
    }
    tracing::info!("inf-viewport: shutting down");
}
