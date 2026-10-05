// Instanced forward mesh pass: metallic-roughness PBR (Cook-Torrance GGX) lit by
// the scene lights uniform. `fs` shades; `fs_id` writes the pick id for the
// ID-buffer pass (same vertex path, R32Uint target). The selection-mask
// fragment lives in mask.wgsl so this module can own the lights bind group.

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // P26.5: the mesh's own uv. `crate::primitives` generates one per built-in
    // shape and `passes::classic_vgeom` copies the authored `VgeomVertex::uv`
    // across, so this path no longer projects a uv it was never given.
    @location(2) uv: vec2<f32>,
    // Instance data
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) nrm_0: vec4<f32>,
    @location(8) nrm_1: vec4<f32>,
    @location(9) nrm_2: vec4<f32>,
    @location(10) color: vec4<f32>,
    @location(11) misc: vec4<u32>, // x = pick id
    @location(12) pbr: vec4<f32>,      // x = metallic, y = roughness
    @location(13) emissive: vec4<f32>, // rgb = emissive
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
    @location(4) @interpolate(flat) pbr: vec4<f32>,
    @location(5) @interpolate(flat) emissive: vec3<f32>,
    // P26.5: the mesh's own uv, and the instance's virtual-texture set (albedo,
    // normal, ORM slots). `obj_pos`/`obj_nrm` — the object-space frame P26.3's
    // box projection was derived from — are gone with the projection.
    @location(6) uv: vec2<f32>,
    @location(8) @interpolate(flat) vt: vec3<u32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let nrm = mat3x3<f32>(in.nrm_0.xyz, in.nrm_1.xyz, in.nrm_2.xyz);
    var out: VsOut;
    let wp = model * vec4<f32>(in.pos, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    out.normal = nrm * in.normal;
    out.color = in.color;
    out.id = in.misc.x;
    out.pbr = in.pbr;
    out.emissive = in.emissive.rgb;
    out.uv = in.uv;
    out.vt = in.misc.yzw;
    return out;
}

// The uv this path samples with is the vertex stream's own (P26.5). The
// dominant-axis box projection that stood in for it through P26.3/P26.4 is gone
// from WGSL entirely; its last producer — deformed cloth and hair, which have no
// authored parametrization — takes `inf_render::box_uv` once per vertex instead.


// AO + cascaded shadows + dynamic GI ride the shared env bind group at @group(2)
// (declared in env_lighting.wgsl, prepended by `lit_scene_shader`): `ao_tex`/`ao_smp`
// (SSAO, white when off), `shadow_factor()`, and `ambient_irradiance()`.







@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // R-P5 masked alpha-test: blend code 1 (pbr.w) discards fragments whose base
    // color alpha is below the cutoff (pbr.z). Opaque (code 0) and translucent
    // (code 2) never take this branch, so every pre-R-P5 golden stays
    // byte-identical (the branch is present but always false for them). Runs
    // before the unlit short-circuit so masked cutouts show in every view mode.
    //
    // **AND THE COVERAGE MAY BE A TEXTURE** (wave OUTFIT1 AUDIT, carried 176) —
    // `skinned_mesh.wgsl`'s change, character for character, one pipeline over.
    // Until this line the alpha it read was `in.color.a` alone, the instance's
    // CONSTANT, so a masked RIGID surface could only be entirely there or
    // entirely gone and a cut-out that lives in a MAP could not exist on this
    // path either. The base colour's own alpha channel is where this engine
    // keeps a masked material's coverage, and `vt_sample_color` is the door that
    // reads it — the same slot, the same uv and the same derivatives
    // `vt_surface` uses below, so the mask and the shading cannot disagree about
    // which texel a fragment is.
    //
    // The derivatives are taken in UNIFORM control flow, before the branch, for
    // the reason the VT block below states. Every instance that binds no albedo
    // slot takes `vt_bound(0) == false` and keeps exactly the constant it had,
    // so every committed golden runs the same arithmetic.
    let mask_ddx = dpdx(in.uv);
    let mask_ddy = dpdy(in.uv);
    if (in.pbr.w > 0.5 && in.pbr.w < 1.5) {
        var mask_a = in.color.a;
        if (vt_bound(in.vt.x)) {
            let mask_uv = vt_scale_uv(in.vt, in.uv, mask_ddx, mask_ddy);
            mask_a = mask_a * vt_sample_color(in.vt.x, mask_uv[0], mask_uv[1], mask_uv[2]).a;
        }
        if (mask_a < in.pbr.z) {
            discard;
        }
    }
    // P26.5 RESIDENCY HEAT-MAP (`ViewMode::VtResidency`): every virtual-textured
    // surface painted by how far behind the streamer is at that pixel.
    //
    // **Before the unlit short-circuit, not after.** `VtResidency` sets
    // `flags.x` too (the `Biomes` precedent: everything that is not the overlay
    // renders flat so the ramp is readable), so a heat branch placed below the
    // unlit return would never execute — measured, and it is the whole defect a
    // real-frame arm catches and a source read does not.
    //
    // `flags.z` is 0.0 in every other mode, so this is present-and-false and
    // every golden runs the identical arithmetic. The derivatives are taken
    // here, in uniform control flow: `flags.z` is a uniform, so the return is
    // uniform, and `in.uv` is an interpolated input either way.
    if (view.flags.z > 0.5) {
        return vec4<f32>(vt_heat(in.vt, in.uv, dpdx(in.uv), dpdy(in.uv)), 1.0);
    }
    // P27.5 VsmPages: the shadow-page residency ramp, beside the texture one and
    // for the same reason. `flags.w` is 0.0 in every other mode, so this is
    // present-and-false and every golden runs the identical arithmetic. Above
    // the unlit short-circuit, which `VsmPages` also sets (the `VtResidency`
    // precedent, and the defect a source read does not catch).
    if (view.flags.w > 0.5) {
        return vec4<f32>(vsm_heat(in.world_pos, normalize(in.normal)), 1.0);
    }
    // Unlit view mode (R-P2): return albedo + emissive directly, skipping the
    // light loop entirely. Drives both Unlit and Wireframe; `flags.x` is 0 in the
    // default Lit mode so this branch is never taken there (goldens byte-stable).
    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + in.emissive, in.color.a);
    }
    var n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    var albedo = in.color.rgb;
    var metallic = clamp(in.pbr.x, 0.0, 1.0);
    var rough = clamp(in.pbr.y, 0.04, 1.0);
    // P26.3 VIRTUAL TEXTURES. `in.vt` is all zeros on every instance that names
    // no texture — which is every instance before this batch and every instance
    // of every committed golden — so `vt_surface` returns its arguments
    // unchanged and the arithmetic below is byte-identical.
    // The MESH'S OWN uv (P26.5).
    let uv = in.uv;
    // The screen derivatives, taken in UNIFORM control flow — a fragment shader
    // may only difference against its neighbours outside a divergent branch, and
    // the VT branch below is per instance. Cheap when nothing samples: a
    // derivative of an interpolated value is two subtractions.
    let vt_ddx = dpdx(uv);
    let vt_ddy = dpdy(uv);
    let vt_dpx = dpdx(in.world_pos);
    let vt_dpy = dpdy(in.world_pos);
    var vt_ao = 1.0;
    if (in.vt.x != 0u || in.vt.y != 0u || in.vt.z != 0u) {
        let s = vt_surface(in.vt, uv, vt_ddx, vt_ddy,
                           albedo, in.color.a, metallic, rough);
        albedo = s.albedo;
        metallic = clamp(s.metallic, 0.0, 1.0);
        rough = clamp(s.roughness, 0.04, 1.0);
        vt_ao = s.occlusion;
        if (s.has_normal) {
            n = vt_apply_normal(n, vt_dpx, vt_dpy, vt_ddx, vt_ddy, s.normal_ts);
        }
    }
    // P20.3 SHORELINE WETNESS: the same band the terrain takes, so a jetty, a
    // rock and the beach they sit on darken together at the waterline instead of
    // the ground alone changing colour. Applied before `f0` so a wet metal's
    // reflectance follows its wetted albedo. `wet.dims.x` is 0 on every scene
    // without water ⇒ the branch is present-but-false and the pre-P20.3 goldens
    // run the identical arithmetic.
    if (wet.dims.x > 0u) {
        let wetted = wet_apply(in.world_pos, albedo, rough);
        albedo = wetted.rgb;
        rough = clamp(wetted.a, 0.04, 1.0);
    }
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    // Wave PAR0: every directional light, then the local lights this
    // fragment's froxel lists — the shared library (`lights.wgsl`).
    lo += lights_direct(in.world_pos, in.pos.xy, n, v, view.sun_dir.xyz,
                        albedo, metallic, rough, f0, LIGHT_SUN_SHADOW | LIGHT_CLOUD_SHADOW | LIGHT_LOCAL_SHADOW);
    // Wave PAR0: the cluster-grid debug view paints the froxel's light count.
    if (lights_debug_view()) {
        return vec4<f32>(lights_debug_heat(in.world_pos, in.pos.xy), 1.0);
    }

    // Image-based ambient: hemispheric sky/ground irradiance by default, or the
    // dynamic-GI probe irradiance when GI is on — modulated by the screen-space AO
    // (ambient term only — never the direct light above).
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    // Wave FIX3: one door. The hemispheric constant is the fallback for a level
    // that asked for no computed ambient; otherwise the sky's own irradiance
    // plus the probe field's signed difference from it. See
    // `ambient_irradiance` / `sky_irradiance` in `env_lighting.wgsl`.
    let amb = ambient_irradiance(
        in.world_pos,
        n,
        mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up),
    );
    // The material's own occlusion map multiplies the screen-space AO: both
    // modulate the AMBIENT term only, never the direct light above.
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r
        * vt_ao;
    lo += amb * albedo * (1.0 - metallic) * ao;
    // P18.4: the ambient specular becomes a real directional term when GI is on
    // (SH radiance along the reflection vector, optionally re-anchored at an SSR
    // hit); otherwise it is exactly the constant it always was.
    lo += gi_ambient_specular(in.world_pos, n, v, rough, f0, amb) * ao;

    lo += in.emissive;

    // Distance haze toward the (linear) horizon colour — applied in HDR-linear;
    // the post tonemap pass (ACES + exposure) runs afterward on the whole buffer.
    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    // P17.2: with an atmosphere, the fixed haze is replaced wholesale by physical
    // aerial perspective + height fog. The branch is never taken for a scene
    // without a time-of-day authority, so the arithmetic above is exactly what
    // every pre-P17.2 golden ran.
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}

@fragment
fn fs_id(in: VsOut) -> @location(0) u32 {
    return in.id;
}
