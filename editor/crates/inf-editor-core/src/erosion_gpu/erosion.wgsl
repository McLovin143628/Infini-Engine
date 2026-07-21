// GPU hydraulic + thermal erosion — the WGSL port of `inf_terrain::erosion`
// (P10.3b). Each numbered CPU pass becomes one `@compute @workgroup_size(8,8)`
// entry point; the CPU module doc's WGSL-port map is the spec and the CPU code
// is the golden reference this shader is validated against (see the parity test
// in `erosion_gpu/mod.rs`).
//
// SUBSTRATE NOTE. The port map suggests r32float/rgba32float/rg32float storage
// TEXTURES. This port uses **storage buffers** instead — a deliberate,
// documented substrate choice (not a semantic deviation): the fields, ping-pong
// pairs, formulas and pass order are identical. Buffers give exact row-major
// index parity with the CPU (`i = r*nx + c`), need no read-only/read-write
// storage-texture feature (so it runs on any adapter incl. lavapipe/CI), and a
// headless upload->simulate->download needs no 256-row alignment. Ping-pong is
// realized as buffer pairs selected by the host via 4 prebuilt bind groups
// (terrain in/out, sediment in/out), so the shader always reads `*_in` / writes
// `*_out` and the host swaps which physical buffer is which between passes.
//
// Hole / boundary policy (matches the CPU reference exactly):
//   * unauthored (hole) cells are walls: no flux crosses to/from them, thermal
//     never moves material into them, their water/sediment stay 0, their terrain
//     is copied through unchanged;
//   * the region outer edge is open: an off-grid neighbour is a dry ghost whose
//     water surface equals the edge cell's terrain (Δh = d), so water drains out.

struct Params {
  nx: u32,
  nz: u32,
  dt: f32,
  rain_rate: f32,
  evaporation: f32,
  gravity: f32,
  pipe_area: f32,
  sediment_capacity: f32,
  dissolving: f32,
  deposition: f32,
  min_tilt: f32,
  max_erode_depth: f32,
  talus_tan: f32,
  thermal_rate: f32,
  max_thermal_depth: f32,
  l: f32,
};

@group(0) @binding(0) var<storage, read>       terrain_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> terrain_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> sediment_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> sediment_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> water: array<f32>;
// flux texel = (fl, fr, ft, fb) outflow to the -x / +x / -z / +z neighbour.
@group(0) @binding(5) var<storage, read_write> flux: array<vec4<f32>>;
// velocity = (u, v) water velocity.
@group(0) @binding(6) var<storage, read_write> velocity: array<vec2<f32>>;
// aux texel = (rain_factor, authored) — two read-only per-cell fields packed
// into one buffer to stay within the 8-storage-buffer downlevel limit.
@group(0) @binding(7) var<storage, read>       aux: array<vec2<f32>>;
@group(0) @binding(8) var<uniform>             P: Params;

const SQRT2: f32 = 1.4142135623730951;

// 8 neighbour offsets (orthogonal then diagonal) — the fixed thermal order.
const N8 = array<vec2<i32>, 8>(
  vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1),
  vec2<i32>(-1, -1), vec2<i32>(1, -1), vec2<i32>(-1, 1), vec2<i32>(1, 1),
);

fn idx(c: u32, r: u32) -> u32 { return r * P.nx + c; }
fn is_auth(i: u32) -> bool { return aux[i].y != 0.0; }

// ── 1. rain ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8)
fn rain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { return; }
  let add = P.dt * P.rain_rate;
  water[i] = water[i] + add * aux[i].x;
}

// New non-negative flux for one pipe (0 for a wall).
fn pipe(old: f32, scale: f32, dh: f32, wall: bool) -> f32 {
  if (wall) { return 0.0; }
  return max(old + scale * dh, 0.0);
}

// ── 2. flux update (in place: reads only terrain, water, own old flux) ────────
@compute @workgroup_size(8, 8)
fn flux_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { flux[i] = vec4<f32>(0.0); return; }

  let self_surf = terrain_in[i] + water[i];
  let scale = P.dt * P.pipe_area * P.gravity / P.l;
  let fo = flux[i];

  // left (-x)
  var dh_l = 0.0; var wall_l = false;
  if (c == 0u) { dh_l = water[i]; } else {
    let j = idx(c - 1u, r);
    if (is_auth(j)) { dh_l = self_surf - (terrain_in[j] + water[j]); } else { wall_l = true; }
  }
  // right (+x)
  var dh_r = 0.0; var wall_r = false;
  if (c == P.nx - 1u) { dh_r = water[i]; } else {
    let j = idx(c + 1u, r);
    if (is_auth(j)) { dh_r = self_surf - (terrain_in[j] + water[j]); } else { wall_r = true; }
  }
  // top (-z)
  var dh_t = 0.0; var wall_t = false;
  if (r == 0u) { dh_t = water[i]; } else {
    let j = idx(c, r - 1u);
    if (is_auth(j)) { dh_t = self_surf - (terrain_in[j] + water[j]); } else { wall_t = true; }
  }
  // bottom (+z)
  var dh_b = 0.0; var wall_b = false;
  if (r == P.nz - 1u) { dh_b = water[i]; } else {
    let j = idx(c, r + 1u);
    if (is_auth(j)) { dh_b = self_surf - (terrain_in[j] + water[j]); } else { wall_b = true; }
  }

  let nfl = pipe(fo.x, scale, dh_l, wall_l);
  let nfr = pipe(fo.y, scale, dh_r, wall_r);
  let nft = pipe(fo.z, scale, dh_t, wall_t);
  let nfb = pipe(fo.w, scale, dh_b, wall_b);

  let total = nfl + nfr + nft + nfb;
  if (total > 0.0) {
    let avail = water[i] * P.l * P.l;
    let k = min(avail / (total * P.dt), 1.0);
    flux[i] = vec4<f32>(nfl, nfr, nft, nfb) * k;
  } else {
    flux[i] = vec4<f32>(0.0);
  }
}

// ── 3 + 4. water update + velocity field ─────────────────────────────────────
@compute @workgroup_size(8, 8)
fn water_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { water[i] = 0.0; velocity[i] = vec2<f32>(0.0); return; }

  var fr_left = 0.0;
  if (c > 0u) { fr_left = flux[idx(c - 1u, r)].y; }
  var fl_right = 0.0;
  if (c + 1u < P.nx) { fl_right = flux[idx(c + 1u, r)].x; }
  var fb_top = 0.0;
  if (r > 0u) { fb_top = flux[idx(c, r - 1u)].w; }
  var ft_bottom = 0.0;
  if (r + 1u < P.nz) { ft_bottom = flux[idx(c, r + 1u)].z; }

  let fo = flux[i];
  let inflow = fr_left + fl_right + fb_top + ft_bottom;
  let outflow = fo.x + fo.y + fo.z + fo.w;
  let d_old = water[i];
  let d_new = max(d_old + P.dt * (inflow - outflow) / (P.l * P.l), 0.0);
  water[i] = d_new;

  let d_mean = 0.5 * (d_old + d_new);
  if (d_mean > 1e-6) {
    let dwx = 0.5 * ((fr_left - fo.x) + (fo.y - fl_right));
    let dwz = 0.5 * ((fb_top - fo.z) + (fo.w - ft_bottom));
    velocity[i] = vec2<f32>(dwx / (P.l * d_mean), dwz / (P.l * d_mean));
  } else {
    velocity[i] = vec2<f32>(0.0);
  }
}

// Terrain height (from terrain_in) of the authored neighbour at (c+dx, r+dz),
// or `here` when off-grid or a hole.
fn nb_b(c: u32, r: u32, dx: i32, dz: i32, here: f32) -> f32 {
  let nc = i32(c) + dx; let nr = i32(r) + dz;
  if (nc < 0 || nr < 0 || nc >= i32(P.nx) || nr >= i32(P.nz)) { return here; }
  let j = u32(nr) * P.nx + u32(nc);
  if (!is_auth(j)) { return here; }
  return terrain_in[j];
}

fn sin_tilt(c: u32, r: u32) -> f32 {
  let here = terrain_in[idx(c, r)];
  let bx0 = nb_b(c, r, -1, 0, here);
  let bx1 = nb_b(c, r, 1, 0, here);
  let bz0 = nb_b(c, r, 0, -1, here);
  let bz1 = nb_b(c, r, 0, 1, here);
  let gx = (bx1 - bx0) / (2.0 * P.l);
  let gz = (bz1 - bz0) / (2.0 * P.l);
  let slope = sqrt(gx * gx + gz * gz);
  return slope / sqrt(1.0 + slope * slope);
}

// ── 5. erosion / deposition (terrain ping-pong; sediment_in updated in place) ─
@compute @workgroup_size(8, 8)
fn erode_deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { terrain_out[i] = terrain_in[i]; return; } // hole: copy through

  let st = max(sin_tilt(c, r), P.min_tilt);
  let vel = velocity[i];
  let speed = sqrt(vel.x * vel.x + vel.y * vel.y);
  let capacity = P.sediment_capacity * st * speed;
  let s = sediment_in[i];
  if (capacity > s) {
    let amt = max(min(P.dissolving * (capacity - s), P.max_erode_depth), 0.0);
    terrain_out[i] = terrain_in[i] - amt;
    sediment_in[i] = s + amt;
  } else {
    let amt = max(min(P.deposition * (s - capacity), s), 0.0);
    terrain_out[i] = terrain_in[i] + amt;
    sediment_in[i] = s - amt;
  }
}

// Bilinear sample of sediment_in at fractional grid coords, clamped to the grid.
fn sample_s(px: f32, pz: f32) -> f32 {
  let cx = clamp(px, 0.0, f32(P.nx - 1u));
  let cz = clamp(pz, 0.0, f32(P.nz - 1u));
  let x0 = u32(floor(cx)); let z0 = u32(floor(cz));
  let x1 = min(x0 + 1u, P.nx - 1u); let z1 = min(z0 + 1u, P.nz - 1u);
  let fx = cx - f32(x0); let fz = cz - f32(z0);
  let s00 = sediment_in[z0 * P.nx + x0];
  let s10 = sediment_in[z0 * P.nx + x1];
  let s01 = sediment_in[z1 * P.nx + x0];
  let s11 = sediment_in[z1 * P.nx + x1];
  let sx0 = s00 + (s10 - s00) * fx;
  let sx1 = s01 + (s11 - s01) * fx;
  return sx0 + (sx1 - sx0) * fz;
}

// ── 6. semi-Lagrangian sediment advection (sediment ping-pong) ────────────────
@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { sediment_out[i] = 0.0; return; }
  let vel = velocity[i];
  let px = f32(c) - vel.x * P.dt / P.l;
  let pz = f32(r) - vel.y * P.dt / P.l;
  sediment_out[i] = sample_s(px, pz);
}

// ── 7. evaporation ────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8)
fn evaporate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { return; }
  let keep = 1.0 - P.evaporation * P.dt;
  water[i] = max(water[i] * keep, 0.0);
}

// How much cell (sc,sr) gives to its neighbour at (sc+tdx, sr+tdz) this step —
// the CPU thermal scatter re-expressed as a gather (recomputes the source's full
// 8-neighbourhood). Reads terrain_in.
fn give_from_to(sc: u32, sr: u32, tdx: i32, tdz: i32) -> f32 {
  let si = idx(sc, sr);
  if (!is_auth(si)) { return 0.0; }
  let bi = terrain_in[si];
  var sum_excess = 0.0;
  var max_diff = 0.0;
  var target_excess = 0.0;
  for (var k = 0; k < 8; k = k + 1) {
    let d = N8[k];
    let nc = i32(sc) + d.x; let nr = i32(sr) + d.y;
    if (nc < 0 || nr < 0 || nc >= i32(P.nx) || nr >= i32(P.nz)) { continue; }
    let j = u32(nr) * P.nx + u32(nc);
    if (!is_auth(j)) { continue; } // hole wall: never move material into a hole
    let diff = bi - terrain_in[j];
    if (diff <= 0.0) { continue; }
    var dist = P.l;
    if (d.x != 0 && d.y != 0) { dist = P.l * SQRT2; }
    if (diff / dist > P.talus_tan) {
      let e = diff - P.talus_tan * dist;
      if (e > 0.0) {
        sum_excess = sum_excess + e;
        if (diff > max_diff) { max_diff = diff; }
        if (d.x == tdx && d.y == tdz) { target_excess = e; }
      }
    }
  }
  if (sum_excess <= 0.0) { return 0.0; }
  let move_total = max(min(P.thermal_rate * 0.5 * max_diff, P.max_thermal_depth), 0.0);
  if (move_total <= 0.0 || target_excess <= 0.0) { return 0.0; }
  return move_total * target_excess / sum_excess;
}

// ── 8. thermal erosion (mass-conserving gather; terrain ping-pong) ────────────
@compute @workgroup_size(8, 8)
fn thermal(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let r = gid.y;
  if (c >= P.nx || r >= P.nz) { return; }
  let i = idx(c, r);
  if (!is_auth(i)) { terrain_out[i] = terrain_in[i]; return; } // hole: copy through

  var give_away = 0.0;
  var receive = 0.0;
  for (var k = 0; k < 8; k = k + 1) {
    let d = N8[k];
    // material this cell sheds toward neighbour c+d
    give_away = give_away + give_from_to(c, r, d.x, d.y);
    // material neighbour (c+d) sheds toward this cell (offset from it to us: -d)
    let sc = i32(c) + d.x; let sr = i32(r) + d.y;
    if (sc >= 0 && sr >= 0 && sc < i32(P.nx) && sr < i32(P.nz)) {
      receive = receive + give_from_to(u32(sc), u32(sr), -d.x, -d.y);
    }
  }
  terrain_out[i] = terrain_in[i] - give_away + receive;
}
