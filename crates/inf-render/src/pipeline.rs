//! WGSL shader-module cache. Modules compile once per device and are shared
//! by every pass; pipelines themselves live with their pass (formats and
//! sample counts are fixed per pass), except where a pass must specialize per
//! output format (blit) and keeps its own small map.

use std::collections::HashMap;

use crate::gpu::GpuContext;

#[derive(Default)]
pub struct ShaderCache {
    modules: HashMap<&'static str, wgpu::ShaderModule>,
}

impl ShaderCache {
    /// Get-or-compile the module registered under `name`. Naga validates at
    /// creation; a bad shader is a programmer error, so this panics loudly
    /// rather than limping on.
    pub fn module(
        &mut self,
        gpu: &GpuContext,
        name: &'static str,
        source: &'static str,
    ) -> &wgpu::ShaderModule {
        self.modules.entry(name).or_insert_with(|| {
            gpu.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                })
        })
    }
}
