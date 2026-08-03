//! Underwater post (P20.3) — the pass that fogs the frame when the camera is
//! inside a water body, and puts sun shafts through the surface above it.
//!
//! ## Where it sits, and why
//!
//! After [`super::water`] and [`super::translucent`], before the grid, the sprite
//! layer and the debug lines. Two boundaries, both deliberate:
//!
//! * **after the water surface**, because the surface is the brightest thing in
//!   an underwater frame and it is what the light shafts are gathered *from*. A
//!   pass that ran before it would have nothing to make a shaft out of.
//! * **before the overlays**, because the grid, the gizmos and the outline are
//!   editor furniture, not things in the water. Fogging a translate gizmo to a
//!   blue smear at ten metres would make the editor unusable underwater — which
//!   is the same reasoning that keeps them out of the depth-of-field of every
//!   other renderer that has one.
//!
//! ## Two render passes, and why the first one exists
//!
//! The identical arrangement [`super::water`] uses, and for the identical reason:
//! this pass needs the scene *colour* it is fogging, and the colour target being
//! rendered into is the MSAA one, which cannot be sampled while it is an
//! attachment. So the node first records a **resolve-only pass** (`color_msaa`
//! in, `scene_hdr` out, no draws) and then samples the single-sample copy.
//! `scene_hdr` is overwritten by the real [`super::resolve`] node later in the
//! frame, so nothing downstream sees the intermediate.
//!
//! ## Writing back through MSAA loses no antialiasing
//!
//! The fog is a full-screen triangle into `color_msaa`, so every sample of a
//! pixel receives the same value — the fogged *resolved* colour. **On the colour
//! path that loses nothing**: the resolved colour already contains the
//! antialiasing, and the final [`super::resolve`] averaging four identical
//! samples reproduces it exactly. What it buys is one fragment invocation per
//! pixel rather than per sample.
//!
//! The **depth** path is where it does cost something. `textureLoad`ing sample 0
//! gives one column length for the whole pixel, so a pixel straddling a
//! silhouette is fogged entirely at the near depth or entirely at the far one —
//! an unfiltered, aliased fog edge on the very silhouettes the colour path
//! antialiased. Inherited from the water pass's arrangement (which measures its
//! column the same way, for the same reason) rather than introduced here, and
//! sub-pixel at the distances where the fog is strong. Named in the P20.3
//! ledger.
//!
//! ## Off path
//!
//! The camera not being underwater ⇒ `run` returns **before touching the
//! encoder**: no resolve, no render pass, no pipeline bind, no draw. Every
//! pre-P20.3 golden therefore records the exact command stream it did before —
//! including the three P20.1 water goldens, whose cameras are all above their
//! water.
//!
//! Pinned by `underwater_off_path_never_engages` (golden.rs), which reads
//! [`UnderwaterReport`] — the node's own engagement counter. That is a stronger
//! claim than a pixel comparison can make: a pass that engaged but wrote the
//! colour back unchanged would satisfy any image assertion, and this it cannot.

use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::water::{
    camera_underwater, shaft_sun_fade, RenderWater, Underwater, SHAFT_DECAY, SHAFT_GLOW_POWER,
    SHAFT_INTENSITY, SHAFT_REACH, SHAFT_TINT_DEPTH_M, UNDERWATER_FAR_M,
};

/// How many frames the underwater pass has **engaged** on — i.e. actually
/// recorded a resolve, a render pass and a draw.
///
/// The house instrumentation pattern (`SharedStreamReport`, the vgeom/scatter
/// audits) applied to an off-path claim. It exists because "the node contributed
/// nothing" is a statement about the *command stream*, and a pixel comparison
/// cannot make it: a pass that engaged and wrote the scene back unchanged is
/// invisible to every image assertion in the repo. One relaxed increment on the
/// frames that engage and nothing at all on the frames that do not, so the off
/// path stays exactly as free as it claims to be.
pub type UnderwaterReport = std::sync::Arc<AtomicU64>;

/// The underwater uniform. Mirrors `struct Underwater` in `underwater.wgsl`; the
/// pair is pinned by `the_uniform_matches_the_shader_struct`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct UnderwaterUniform {
    params: [f32; 4],
    absorption: [f32; 4],
    deep: [f32; 4],
    shafts: [f32; 4],
}

/// The render-local Y of the plane the shader caps its column against.
///
/// **The DISPLACED surface over the eye**, not the body's still-water level:
/// [`Underwater::surface_y`] is what `WaterSurface::height_at` answered at the
/// camera's own XZ, wave included. Under a 0.6 m swell those differ by a wave
/// amplitude, and using `level_m` would misplace the cap by exactly the thing
/// that makes a sea a sea. (The *shader* still treats it as a plane — a
/// per-pixel Gerstner inverse in a post pass would be a second surface
/// evaluation — so the approximation is "one plane, placed correctly at the
/// camera" rather than "one plane at the mean level".)
///
/// Split out of `run` so the derivation itself is testable:
/// `the_column_cap_follows_the_displaced_surface` feeds a body whose
/// `surface_y != level_m` and catches a swap to either one.
pub fn surface_plane_local(under: &Underwater, origin: &inf_math::FloatingOrigin) -> f32 {
    origin
        .to_render(glam::DVec3::new(0.0, under.surface_y, 0.0))
        .y
}

/// Pack the uniform for one submerged camera. Split out of `run` so it is
/// testable without a GPU.
///
/// `level_local` comes from [`surface_plane_local`]. `sun_y` is the scene sun
/// direction's `y` — the shafts' only time-of-day coupling, via
/// [`shaft_sun_fade`].
pub fn pack_uniform(
    under: &Underwater,
    body: &RenderWater,
    level_local: f32,
    shafts: bool,
    sun_y: f32,
) -> UnderwaterUniform {
    // The shafts fade out as the sun sets and switch OFF once it has: the
    // shader's lobe knows only the angle between a ray and `view.sun_dir`, and
    // that angle stays small for a ray rising toward a sun well below the
    // horizon. Folding the fade in here rather than in the shader keeps it in the
    // one place a GPU-free test can read.
    let fade = if shafts { shaft_sun_fade(sun_y) } else { 0.0 };
    let shafts_on = fade > 0.0;
    UnderwaterUniform {
        params: [
            under.strength,
            under.depth_m as f32,
            level_local,
            UNDERWATER_FAR_M,
        ],
        // The BODY's own extinction and deep colour — not a second set of
        // underwater constants. One medium, one absorption, seen from both sides.
        absorption: [
            body.absorption[0],
            body.absorption[1],
            body.absorption[2],
            SHAFT_INTENSITY * fade,
        ],
        deep: [
            body.deep_color[0],
            body.deep_color[1],
            body.deep_color[2],
            SHAFT_DECAY,
        ],
        shafts: [
            SHAFT_GLOW_POWER,
            SHAFT_REACH,
            f32::from(u8::from(shafts_on)),
            SHAFT_TINT_DEPTH_M,
        ],
    }
}

pub struct UnderwaterNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    /// Keyed on the full [`super::ResourceKey`]: the bind group embeds the
    /// resolved scene colour and the scene depth, both recreated on a resize. The
    /// uniform buffer is created once and never resized, so it contributes no key
    /// component — see [`crate::wetness::WetnessResources`] for the same argument
    /// spelled out.
    bind_group: super::GenCache<super::ResourceKey, wgpu::BindGroup>,
    /// Incremented once per frame this node actually records commands.
    report: UnderwaterReport,
}

impl UnderwaterNode {
    pub fn new(
        gpu: &GpuContext,
        view_bgl: &wgpu::BindGroupLayout,
        report: UnderwaterReport,
    ) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("underwater"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("underwater").into()),
            });

        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("underwater"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vf,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The resolved scene colour (this pass's own resolve-only
                    // pass wrote it) + its sampler.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // The scene depth, `textureLoad`ed at sample 0 — the same
                    // multisampled binding the water pass measures its column
                    // with, and for the same reason: the surface carries no
                    // thickness, so the distance has to come from the buffer.
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                ],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("underwater"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("underwater"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        // REPLACE, not blend: the shader returns the finished
                        // pixel (it has already mixed against the unfogged scene
                        // by the waterline ramp), so blending would apply the
                        // ramp twice.
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                // No depth attachment at all: this is a full-screen post, and the
                // distance it needs comes from the SAMPLED depth, not from a test.
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: SCENE_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });

        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("underwater-uniform"),
            size: std::mem::size_of::<UnderwaterUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("underwater-scene"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            sampler,
            uniform,
            bind_group: super::GenCache::default(),
            report,
        }
    }
}

impl RenderNode for UnderwaterNode {
    fn name(&self) -> &'static str {
        "underwater"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        // OFF PATH: the camera is not in the water ⇒ the encoder is not touched.
        // The test reuses `inf-water`'s evaluator through
        // `crate::water::camera_underwater`, so the pixel that fogs and the
        // buoyancy that lifts a boat are answering the same surface.
        let Some(under) = camera_underwater(&frame.scene.waters, frame.view.eye_world) else {
            return;
        };
        let body = &frame.scene.waters[under.body];
        // Past this point the encoder WILL be touched, so this is where the
        // engagement counter belongs — not at the top, where the early return
        // above would already have bumped it.
        self.report.fetch_add(1, Ordering::Relaxed);

        let u = pack_uniform(
            &under,
            body,
            surface_plane_local(&under, &frame.view.origin),
            frame.settings.water.quality.light_shafts(),
            frame.scene.sun.unit_direction().y,
        );
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));

        // ── the colour this pass fogs ─────────────────────────────────────
        //
        // One resolve-only pass: `color_msaa` in, `scene_hdr` out, no draws. See
        // the module docs.
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("underwater-resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: Some(&frame.targets.scene_hdr),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let (bgl, sampler, uniform) = (&self.bgl, &self.sampler, &self.uniform);
        let bind_group = self
            .bind_group
            .get_or_build(super::resource_key(frame), || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("underwater"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.scene_hdr),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.depth),
                        },
                    ],
                })
            })
            .clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("underwater"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::{WaterQuality, WaveField, WaveSpec};
    use glam::DVec3;

    fn ocean() -> RenderWater {
        RenderWater {
            level_m: 4.0,
            waves: WaveField::from_spec(&WaveSpec::default()),
            time_s: 1_234.5,
            ..RenderWater::default()
        }
    }

    #[test]
    fn the_uniform_matches_the_shader_struct() {
        // `struct Underwater` in underwater.wgsl: four vec4s. A mismatch is
        // silent on the GPU — the shader reads the wrong 16 bytes for everything
        // after the divergence.
        assert_eq!(std::mem::size_of::<UnderwaterUniform>(), 4 * 16);
        assert_eq!(std::mem::align_of::<UnderwaterUniform>(), 4);
    }

    /// **The one-absorption-story gate.** The fog's extinction and deep colour
    /// are the SUBMERGING BODY's, byte for byte — not a second set of underwater
    /// constants that would have to be kept in step with the surface's.
    #[test]
    fn the_fog_absorbs_with_the_body_it_is_inside() {
        let mut body = ocean();
        body.absorption = [0.37, 0.11, 0.043];
        body.deep_color = [0.011, 0.062, 0.141];
        let under = camera_underwater(std::slice::from_ref(&body), DVec3::new(0.0, -6.0, 0.0))
            .expect("submerged");
        let u = pack_uniform(&under, &body, 4.0, true, 1.0);
        assert_eq!(&u.absorption[..3], &body.absorption[..]);
        assert_eq!(&u.deep[..3], &body.deep_color[..]);
        // …and the depth the downwelling term uses is the eye's real submersion.
        assert!((u.params[1] - under.depth_m as f32).abs() < 1e-6);
        assert_eq!(u.params[2], 4.0, "the surface plane must be render-local");
        assert_eq!(u.params[3], UNDERWATER_FAR_M);
    }

    #[test]
    fn the_shaft_flag_follows_the_quality_tier() {
        let body = ocean();
        let under =
            camera_underwater(std::slice::from_ref(&body), DVec3::new(0.0, -6.0, 0.0)).unwrap();
        for (q, want) in [
            (WaterQuality::Low, 0.0f32),
            (WaterQuality::Medium, 1.0),
            (WaterQuality::High, 1.0),
        ] {
            let u = pack_uniform(&under, &body, 0.0, q.light_shafts(), 1.0);
            assert_eq!(u.shafts[2], want, "{q:?}");
        }
        // The tier gates the SHAFTS, never the fog: the absorption is the
        // content, and a Low tier still has to render water you can drown in.
        let low = pack_uniform(&under, &body, 0.0, false, 1.0);
        let high = pack_uniform(&under, &body, 0.0, true, 1.0);
        assert_eq!(&low.absorption[..3], &high.absorption[..3]);
        assert_eq!(low.params, high.params);
    }

    /// **The column cap follows the DISPLACED surface, not the still-water
    /// level.** The forwarding test hands `pack_uniform` a literal, so nothing
    /// there would notice a swap to `body.level_m`; this drives the derivation
    /// `run` actually uses, with a body whose two candidates differ.
    #[test]
    fn the_column_cap_follows_the_displaced_surface() {
        let body = ocean(); // level 4.0, a real swell on top of it
        let eye = DVec3::new(3.0, -6.0, -2.0);
        let under = camera_underwater(std::slice::from_ref(&body), eye).expect("submerged");
        assert_ne!(
            under.surface_y, body.level_m,
            "pick a clock/XZ where the wave is not exactly zero, or this is vacuous"
        );

        let origin = inf_math::FloatingOrigin::default();
        let plane = surface_plane_local(&under, &origin);
        assert_eq!(
            plane, under.surface_y as f32,
            "the cap is not at the displaced surface"
        );
        assert_ne!(
            plane, body.level_m as f32,
            "the cap fell back to the still-water level"
        );

        // …and it rebases like every other water number: under a shifted origin
        // the plane moves with the world, so the shader's `level - eye.y` is
        // unchanged.
        let shifted = inf_math::FloatingOrigin::new(DVec3::new(0.0, 90.0, 0.0));
        let plane_b = surface_plane_local(&under, &shifted);
        let eye_a = origin.to_render(eye).y;
        let eye_b = shifted.to_render(eye).y;
        assert!(
            ((plane - eye_a) - (plane_b - eye_b)).abs() < 1e-3,
            "the cap does not survive a rebase: {plane} - {eye_a} vs {plane_b} - {eye_b}"
        );
    }

    /// The sun fade reaches the uniform: it scales the shaft intensity and, once
    /// the sun is down, clears the enable flag so the 24-tap loop is skipped
    /// outright. The fade curve itself is pinned in
    /// `water::tests::light_shafts_fade_out_as_the_sun_sets`.
    #[test]
    fn a_set_sun_switches_the_shafts_off_in_the_uniform() {
        let body = ocean();
        let under =
            camera_underwater(std::slice::from_ref(&body), DVec3::new(0.0, -6.0, 0.0)).unwrap();

        let noon = pack_uniform(&under, &body, 0.0, true, 1.0);
        assert_eq!(noon.shafts[2], 1.0);
        assert_eq!(noon.absorption[3], SHAFT_INTENSITY);

        // Straight-down sun — the value `SunParams::unit_moon_direction`'s
        // fallback produces, and one a projector really can hand over.
        let night = pack_uniform(&under, &body, 0.0, true, -1.0);
        assert_eq!(
            night.shafts[2], 0.0,
            "shafts ran with the sun below the world"
        );
        assert_eq!(night.absorption[3], 0.0);

        // Dusk is a fade, not a switch.
        let dusk = pack_uniform(&under, &body, 0.0, true, 0.0);
        assert!(
            dusk.absorption[3] > 0.0 && dusk.absorption[3] < SHAFT_INTENSITY,
            "the horizon is not mid-fade: {}",
            dusk.absorption[3]
        );
        assert_eq!(dusk.shafts[2], 1.0);

        // The fog is untouched by any of it — the absorption is the content.
        for u in [noon, night, dusk] {
            assert_eq!(&u.absorption[..3], &body.absorption[..]);
            assert_eq!(u.params, noon.params);
            assert_eq!(&u.deep[..3], &body.deep_color[..]);
        }
    }

    #[test]
    fn packing_is_deterministic() {
        let body = ocean();
        let under =
            camera_underwater(std::slice::from_ref(&body), DVec3::new(3.0, -2.0, -7.0)).unwrap();
        let a = pack_uniform(&under, &body, 1.5, true, 0.7);
        let b = pack_uniform(&under, &body, 1.5, true, 0.7);
        assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));
    }

    /// The waterline ramp reaches the shader: a camera at the surface packs zero
    /// strength (the shader then returns the scene untouched), a deep one packs
    /// full strength.
    #[test]
    fn the_waterline_ramp_reaches_the_uniform() {
        let body = ocean();
        let deep =
            camera_underwater(std::slice::from_ref(&body), DVec3::new(0.0, -20.0, 0.0)).unwrap();
        assert_eq!(pack_uniform(&deep, &body, 0.0, true, 1.0).params[0], 1.0);

        let shallow = camera_underwater(
            std::slice::from_ref(&body),
            DVec3::new(0.0, deep.surface_y - 0.01, 0.0),
        )
        .unwrap();
        let s = pack_uniform(&shallow, &body, 0.0, true, 1.0).params[0];
        assert!(s > 0.0 && s < 0.05, "the ramp is not soft at the line: {s}");
    }
}
