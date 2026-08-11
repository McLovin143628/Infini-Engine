//! **The WGSL mirror**: the same three stages, on the adapter.
//!
//! # The shape of the thing
//!
//! [`DenseGpu`] owns a device and three compute pipelines; [`GpuBackend`] is the
//! [`crate::dense::DenseBackend`] implementation that drives them. The pipeline
//! in [`crate::dense`] does not know which one it is running — that is the whole
//! point, because it means "the GPU agrees with the CPU" is a claim about three
//! functions and nothing underneath them can drift.
//!
//! # One adapter door
//!
//! The device comes from [`inf_render::GpuContext::headless`], which is the
//! workspace's single adapter constructor — hardware first, software fallback
//! second, a device-lost callback, and a `Result<_, String>` when there is
//! nothing at all. A second copy of that thirty-line function here would be a
//! second answer to "is there a GPU", and the house has had that argument
//! already. No adapter is [`DenseError::NoGpu`] carrying the door's own message,
//! and the tests skip on it rather than failing — the golden-harness doctrine,
//! applied to compute.
//!
//! # Shaders are files, and they are validated without a GPU
//!
//! Every `.wgsl` lives beside this module and is embedded with `include_str!`,
//! registered in [`SHADERS`]. `naga` parses and validates all of them in a unit
//! test that needs **no adapter**, so a shader that cannot compile is a red test
//! on every CI platform rather than a silent skip on the ones without a card.
//! That is the `SHADER_TABLE` law from `inf-render`, in its standalone-compute
//! form: add a shader by adding a row, or the validation gate does not see it.
//!
//! # Bindings, and the limit that shapes them
//!
//! The device is requested with default limits, which cap a stage at **8 storage
//! buffers**. The sweep uses seven and a uniform; fusion uses five and a
//! uniform. Nothing here is packed to dodge that cap, and if a future stage
//! needs a ninth buffer the answer is to pack fields, not to raise a limit that
//! then has to hold on every tier.

use inf_core::job::JobPool;
use inf_photo::gray::GrayImage;
use inf_render::GpuContext;
use wgpu::util::DeviceExt;

use crate::dense::DenseBackend;
use crate::sweep::{CensusConfig, DepthMap, SweepConfig, SweepGeometry};
use crate::tsdf::{FuseParams, TsdfGrid};
use crate::DenseError;

/// Every shader this crate ships, by label.
///
/// The naga validation gate iterates this table. A shader that is not in it is
/// a shader nothing checks — which is exactly how a broken pass reaches a
/// machine with a GPU and nowhere else.
pub const SHADERS: &[(&str, &str)] = &[
    ("census", include_str!("shaders/census.wgsl")),
    ("plane_sweep", include_str!("shaders/plane_sweep.wgsl")),
    ("tsdf", include_str!("shaders/tsdf.wgsl")),
];

/// The census shader's uniform block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CensusParams {
    width: u32,
    height: u32,
    dilation: u32,
    _pad: u32,
}

/// The sweep shader's uniform block. 48 bytes — a multiple of 16.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SweepParams {
    width: u32,
    height: u32,
    planes: u32,
    neighbours: u32,
    min_valid: u32,
    max_cost: u32,
    exclusion: u32,
    _pad0: u32,
    inv_near: f32,
    inv_step: f32,
    min_confidence: f32,
    _pad1: f32,
}

/// A device and the three pipelines the dense stage needs.
pub struct DenseGpu {
    /// The adapter door's context. Public so a test can name the adapter it
    /// measured a parity number on.
    pub gpu: GpuContext,
    census_bgl: wgpu::BindGroupLayout,
    census_pipeline: wgpu::ComputePipeline,
    sweep_bgl: wgpu::BindGroupLayout,
    sweep_pipeline: wgpu::ComputePipeline,
    fuse_bgl: wgpu::BindGroupLayout,
    fuse_pipeline: wgpu::ComputePipeline,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn source(label: &str) -> &'static str {
    SHADERS
        .iter()
        .find(|(l, _)| *l == label)
        .unwrap_or_else(|| panic!("unknown shader table entry: {label}"))
        .1
}

impl DenseGpu {
    /// Open the adapter and build the pipelines, or say why not.
    pub fn new() -> Result<Self, DenseError> {
        let gpu = GpuContext::headless().map_err(|message| DenseError::NoGpu { message })?;
        Ok(Self::from_context(gpu))
    }

    /// Build the pipelines on a context the caller already has.
    pub fn from_context(gpu: GpuContext) -> Self {
        let device = &gpu.device;
        let build = |label: &str, entries: Vec<wgpu::BindGroupLayoutEntry>, entry_point: &str| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source(label).into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            (bgl, pipeline)
        };

        let (census_bgl, census_pipeline) = build(
            "census",
            vec![
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
            "cs_census",
        );
        let (sweep_bgl, sweep_pipeline) = build(
            "plane_sweep",
            vec![
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
            "cs_sweep",
        );
        let (fuse_bgl, fuse_pipeline) = build(
            "tsdf",
            vec![
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
            "cs_fuse",
        );

        Self {
            gpu,
            census_bgl,
            census_pipeline,
            sweep_bgl,
            sweep_pipeline,
            fuse_bgl,
            fuse_pipeline,
        }
    }

    /// A short description of the adapter, for test output.
    pub fn adapter_name(&self) -> String {
        let info = self.gpu.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }

    fn upload<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Read a storage buffer back into a `Vec`.
    ///
    /// The house readback idiom: a staging buffer, one `map_async`, one
    /// `poll(wait_indefinitely)`, one `get_mapped_range`, `unmap`. There is no
    /// row padding here because these are buffers, not textures — the 256-byte
    /// `COPY_BYTES_PER_ROW_ALIGNMENT` rule applies to texture copies only.
    fn read_back<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
        let bytes = (count * std::mem::size_of::<T>()) as u64;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dense-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dense-readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.gpu.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv()
            .expect("map_async dropped")
            .expect("map_async failed");
        let data = slice.get_mapped_range().expect("mapped range");
        let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    /// The census transform of one image, on the GPU.
    pub fn census(&self, image: &GrayImage, cfg: &CensusConfig) -> Vec<u32> {
        let w = image.width();
        let h = image.height();
        let n = w as usize * h as usize;
        // One intensity per word; see the shader's comment for why.
        let widened: Vec<u32> = image.pixels().iter().map(|p| *p as u32).collect();
        let params = CensusParams {
            width: w,
            height: h,
            dilation: cfg.dilation.max(1),
            _pad: 0,
        };
        let params_buf = self.upload("census-params", &[params], wgpu::BufferUsages::UNIFORM);
        let image_buf = self.upload("census-image", &widened, wgpu::BufferUsages::STORAGE);
        let out_buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("census-out"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("census"),
                layout: &self.census_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: image_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("census"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("census"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.census_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        self.gpu.queue.submit([encoder.finish()]);
        self.read_back(&out_buf, n)
    }

    /// Sweep one reference view, on the GPU.
    pub fn sweep(
        &self,
        geometry: &SweepGeometry,
        reference_census: &[u32],
        census: &[u32],
        cfg: &SweepConfig,
    ) -> DepthMap {
        let n = geometry.pixels();
        let params = SweepParams {
            width: geometry.width,
            height: geometry.height,
            planes: geometry.depths.len() as u32,
            neighbours: geometry.params.len() as u32,
            min_valid: cfg.min_valid_neighbours.min(geometry.params.len()).max(1) as u32,
            max_cost: cfg.max_cost,
            exclusion: cfg.exclusion_planes,
            _pad0: 0,
            inv_near: geometry.inv_near,
            inv_step: geometry.inv_step,
            min_confidence: cfg.min_confidence,
            _pad1: 0.0,
        };
        let params_buf = self.upload("sweep-params", &[params], wgpu::BufferUsages::UNIFORM);
        let nb_buf = self.upload(
            "sweep-neighbours",
            &geometry.params,
            wgpu::BufferUsages::STORAGE,
        );
        let rays_buf = self.upload("sweep-rays", &geometry.rays, wgpu::BufferUsages::STORAGE);
        let depths_buf = self.upload(
            "sweep-depths",
            &geometry.depths,
            wgpu::BufferUsages::STORAGE,
        );
        let ref_buf = self.upload(
            "sweep-ref-census",
            reference_census,
            wgpu::BufferUsages::STORAGE,
        );
        let census_buf = self.upload("sweep-census", census, wgpu::BufferUsages::STORAGE);
        let make_out = |label: &str| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let depth_out = make_out("sweep-depth");
        let conf_out = make_out("sweep-confidence");
        let bind = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("plane_sweep"),
                layout: &self.sweep_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: nb_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rays_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: depths_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ref_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: census_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: depth_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: conf_out.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("plane_sweep"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("plane_sweep"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.sweep_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(geometry.width.div_ceil(8), geometry.height.div_ceil(8), 1);
        }
        self.gpu.queue.submit([encoder.finish()]);

        DepthMap {
            view: geometry.view,
            width: geometry.width,
            height: geometry.height,
            depth: self.read_back(&depth_out, n),
            confidence: self.read_back(&conf_out, n),
        }
    }

    /// Integrate one view into a grid, on the GPU. Returns how many voxels
    /// voted — the **engagement counter**, read out of an integer atomic, which
    /// is order-independent in a way a float sum is not.
    pub fn integrate(
        &self,
        grid: &mut TsdfGrid,
        params: &FuseParams,
        map: &DepthMap,
        weights: &[f32],
    ) -> usize {
        let n = grid.len();
        let params_buf = self.upload("fuse-params", &[*params], wgpu::BufferUsages::UNIFORM);
        let depth_buf = self.upload("fuse-depth", &map.depth, wgpu::BufferUsages::STORAGE);
        let weight_buf = self.upload("fuse-weights", weights, wgpu::BufferUsages::STORAGE);
        let acc_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let dist_buf = self.upload("fuse-distance", &grid.distance, acc_usage);
        let wacc_buf = self.upload("fuse-weight-acc", &grid.weight, acc_usage);
        let votes_buf = self.upload("fuse-votes", &[0u32], acc_usage);
        let bind = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tsdf"),
                layout: &self.fuse_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: depth_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dist_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wacc_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: votes_buf.as_entire_binding(),
                    },
                ],
            });
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tsdf"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tsdf"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fuse_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                grid.dims[0].div_ceil(4),
                grid.dims[1].div_ceil(4),
                grid.dims[2].div_ceil(4),
            );
        }
        self.gpu.queue.submit([encoder.finish()]);

        grid.distance = self.read_back(&dist_buf, n);
        grid.weight = self.read_back(&wacc_buf, n);
        self.read_back::<u32>(&votes_buf, 1)[0] as usize
    }
}

/// The [`DenseBackend`] that runs the pipeline on an adapter.
pub struct GpuBackend<'a> {
    /// The device and pipelines.
    pub gpu: &'a DenseGpu,
}

impl<'a> GpuBackend<'a> {
    /// Wrap a device.
    pub fn new(gpu: &'a DenseGpu) -> Self {
        Self { gpu }
    }
}

impl DenseBackend for GpuBackend<'_> {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn census_all(
        &mut self,
        images: &[&GrayImage],
        cfg: &CensusConfig,
        _pool: &JobPool,
    ) -> Vec<Vec<u32>> {
        images.iter().map(|img| self.gpu.census(img, cfg)).collect()
    }

    fn sweep(
        &mut self,
        geometry: &SweepGeometry,
        reference_census: &[u32],
        census: &[u32],
        cfg: &SweepConfig,
        _pool: &JobPool,
    ) -> DepthMap {
        self.gpu.sweep(geometry, reference_census, census, cfg)
    }

    fn integrate(
        &mut self,
        grid: &mut TsdfGrid,
        params: &FuseParams,
        map: &DepthMap,
        weights: &[f32],
        _pool: &JobPool,
    ) -> usize {
        self.gpu.integrate(grid, params, map, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(label: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!(
                "shader '{label}' failed to parse:\n{}",
                e.emit_to_string(source)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("shader '{label}' failed validation: {e:?}"));
    }

    #[test]
    fn every_shader_parses_and_validates_without_an_adapter() {
        // The `SHADER_TABLE` law in its standalone-compute form. This runs on
        // every CI platform, GPU or not, so a broken shader is red everywhere
        // rather than green on the machines that skip.
        assert_eq!(SHADERS.len(), 3, "a shader left the table");
        for (label, src) in SHADERS {
            validate(label, src);
        }
    }

    #[test]
    fn the_shader_table_and_the_pipelines_name_the_same_shaders() {
        // Anti-vacuity for the table: a shader could be validated and never
        // built, or built from an `include_str!` that bypassed the table
        // entirely. Both spellings resolve through `source`, and this asserts
        // every label the pipelines ask for is in it.
        for label in ["census", "plane_sweep", "tsdf"] {
            assert!(
                !source(label).is_empty(),
                "shader '{label}' resolved to nothing"
            );
        }
        // And the entry points the pipelines request really exist in the
        // sources, so a renamed entry point is a red test rather than a
        // runtime panic on a machine with a card.
        assert!(source("census").contains("fn cs_census"));
        assert!(source("plane_sweep").contains("fn cs_sweep"));
        assert!(source("tsdf").contains("fn cs_fuse"));
    }

    #[test]
    fn no_shader_calls_fma_or_round() {
        // The parity rules, enforced as source facts rather than as a comment
        // somebody has to remember. `fma()` would guarantee the contraction the
        // CPU reference does not perform, and `round()` is round-half-to-even
        // in WGSL against Rust's round-half-away-from-zero — the two functions
        // that would break the mirror on purpose.
        for (label, src) in SHADERS {
            // Read the CODE, not the file: this very rule is *written down* in
            // two of these shaders' headers, and a naive `contains` would fail
            // on the sentence explaining itself. Comment lines are stripped
            // first — which is also what makes the check a claim about what the
            // shader does rather than about what it says.
            let code: String = src
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                !code.contains("fma("),
                "shader '{label}' calls fma(), which the CPU reference does not"
            );
            assert!(
                !code.contains("round("),
                "shader '{label}' calls round(); use floor(x + 0.5), which is exact in both \
                 languages"
            );
            // Anti-vacuity: the stripped source must still be the shader, not
            // an empty string that trivially contains nothing.
            assert!(
                code.contains("@compute"),
                "stripping comments from '{label}' left no code to check"
            );
        }
    }

    #[test]
    fn the_uniform_blocks_are_sized_for_wgsl() {
        // A WGSL uniform block is laid out to a 16-byte multiple. Rust would
        // happily hand over 44 bytes and wgpu would reject the bind group at
        // runtime on a machine with an adapter — which is to say, never in CI.
        assert_eq!(std::mem::size_of::<CensusParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<SweepParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<FuseParams>() % 16, 0);
        assert_eq!(std::mem::size_of::<SweepParams>(), 48);
        assert_eq!(std::mem::size_of::<FuseParams>(), 112);
        // And the neighbour array's stride is what the shader's struct is.
        assert_eq!(std::mem::size_of::<crate::sweep::NeighbourParams>(), 48);
    }
}
