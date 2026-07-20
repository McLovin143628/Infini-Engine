//! Golden-image comparison helpers (P2.5). The renderer produces deterministic
//! output for a fixed scene+view, but exact pixels differ across GPUs
//! (rasterization, AA, filtering). We compare on a downscaled image with a
//! perceptual tolerance: tight enough to catch a dropped pass or a wrong
//! color, loose enough to survive adapter differences.
//!
//! The comparison lives in the library (unit-tested with synthetic images);
//! the harness that renders scenes and reads/writes PNGs lives in
//! `tests/golden.rs`.

/// Downscale an RGBA8 image to `dw`×`dh` by box-averaging. Small target sizes
/// wash out per-pixel GPU noise while preserving structure.
pub fn downscale(rgba: &[u8], w: u32, h: u32, dw: u32, dh: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (dw * dh * 3) as usize];
    for dy in 0..dh {
        for dx in 0..dw {
            let x0 = dx * w / dw;
            let x1 = ((dx + 1) * w / dw).max(x0 + 1).min(w);
            let y0 = dy * h / dh;
            let y1 = ((dy + 1) * h / dh).max(y0 + 1).min(h);
            let mut acc = [0.0f32; 3];
            let mut n = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = ((y * w + x) * 4) as usize;
                    acc[0] += rgba[i] as f32;
                    acc[1] += rgba[i + 1] as f32;
                    acc[2] += rgba[i + 2] as f32;
                    n += 1.0;
                }
            }
            let o = ((dy * dw + dx) * 3) as usize;
            for c in 0..3 {
                out[o + c] = acc[c] / n / 255.0;
            }
        }
    }
    out
}

/// Perceptual difference between two same-size RGBA8 images: (mean, max) of
/// per-channel absolute difference on a 64×36 downscale, in 0..1.
pub fn image_diff(a: &[u8], b: &[u8], w: u32, h: u32) -> (f32, f32) {
    const DW: u32 = 64;
    const DH: u32 = 36;
    let da = downscale(a, w, h, DW, DH);
    let db = downscale(b, w, h, DW, DH);
    let mut sum = 0.0;
    let mut max = 0.0f32;
    for (x, y) in da.iter().zip(db.iter()) {
        let d = (x - y).abs();
        sum += d;
        max = max.max(d);
    }
    (sum / da.len() as f32, max)
}

/// Default tolerances: mean ≤ 6 %, max single-region ≤ 35 %. Chosen to pass a
/// re-render of the same scene on the same adapter with room for AA jitter,
/// while failing a structurally different frame.
pub const GOLDEN_MEAN_TOLERANCE: f32 = 0.06;
pub const GOLDEN_MAX_TOLERANCE: f32 = 0.35;

pub fn within_tolerance(mean: f32, max: f32) -> bool {
    mean <= GOLDEN_MEAN_TOLERANCE && max <= GOLDEN_MAX_TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        v
    }

    #[test]
    fn identical_images_have_zero_diff() {
        let a = solid(80, 45, [30, 60, 120]);
        let (mean, max) = image_diff(&a, &a, 80, 45);
        assert_eq!(mean, 0.0);
        assert_eq!(max, 0.0);
        assert!(within_tolerance(mean, max));
    }

    #[test]
    fn different_colors_exceed_tolerance() {
        let a = solid(80, 45, [10, 10, 10]);
        let b = solid(80, 45, [200, 200, 200]);
        let (mean, max) = image_diff(&a, &b, 80, 45);
        assert!(!within_tolerance(mean, max), "mean {mean} max {max}");
    }

    #[test]
    fn small_noise_stays_within_tolerance() {
        let a = solid(80, 45, [128, 128, 128]);
        let mut b = a.clone();
        // Perturb a few pixels a little — should wash out under downscale.
        for i in (0..b.len()).step_by(4 * 7) {
            b[i] = b[i].saturating_add(6);
        }
        let (mean, max) = image_diff(&a, &b, 80, 45);
        assert!(within_tolerance(mean, max), "mean {mean} max {max}");
    }

    #[test]
    fn downscale_preserves_average() {
        // A left-black / right-white split averages to ~0.5 overall.
        let (w, h) = (64u32, 8u32);
        let mut img = Vec::new();
        for _ in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0 } else { 255 };
                img.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let ds = downscale(&img, w, h, 2, 1);
        assert!(ds[0] < 0.01, "left should be black: {}", ds[0]);
        assert!(ds[3] > 0.99, "right should be white: {}", ds[3]);
    }
}
