//! Sprite-sheet slicing (P8.2a).
//!
//! A texture asset can carry a *slice model* — a grid definition and/or a set of
//! named manual rectangles — that turns one atlas image into many addressable
//! sub-sprites. The model lives in the texture's TOML sidecar under the
//! [`SPRITE_SHEET_KEY`] key inside the importer `import` table (alongside the
//! [`inf_material::TextureImportSettings`]), so it round-trips deterministically
//! with the rest of the asset's metadata and never disturbs the texture payload.
//!
//! Authoring units are **texture pixels** (the natural unit in the slicing
//! editor). [`SpriteSheetSlices::resolve`] projects them to normalized
//! `[0,1]²` UV rects that drop straight into an ECS [`inf_ecs::AtlasRect`], with
//! a top-left origin matching the image/atlas convention.

use inf_asset::{AssetError, AssetId, Result};
use inf_material::TextureAsset;
use serde::{Deserialize, Serialize};

use super::AssetProject;
use crate::ipc::{SpriteGridDto, SpriteRectDto, SpriteSheetDto};

/// The sidecar `import`-table key the slice model is stored under.
pub const SPRITE_SHEET_KEY: &str = "sprite_sheet";

/// Uniform-grid slicing, in texture pixels. `margin_*` is a single top-left
/// offset before the first cell; `padding_*` is the gap between adjacent cells.
/// Cells are laid out row-major (left→right, top→bottom).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GridSlicing {
    pub columns: u32,
    pub rows: u32,
    #[serde(default)]
    pub margin_x: u32,
    #[serde(default)]
    pub margin_y: u32,
    #[serde(default)]
    pub padding_x: u32,
    #[serde(default)]
    pub padding_y: u32,
}

/// One named manual rectangle, in texture pixels (top-left origin).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedRect {
    pub name: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// The persisted slice model for one texture: an optional grid plus any number
/// of named manual rects. Empty fields serialize away so a texture with no
/// slicing keeps a clean sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SpriteSheetSlices {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<GridSlicing>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub manual: Vec<NamedRect>,
}

/// A concrete slice resolved against a texture's pixel dimensions: its pixel
/// rectangle plus the normalized UV corners (top-left origin, `[0,1]²`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedSlice {
    pub name: String,
    /// Pixel rect `[x, y, width, height]` (rounded).
    pub px: [u32; 4],
    /// Normalized UV min corner `(u, v)`.
    pub uv_min: [f64; 2],
    /// Normalized UV max corner `(u, v)`.
    pub uv_max: [f64; 2],
}

/// The largest grid a sheet may be sliced into (C4-15).
///
/// `columns` and `rows` come out of the **sidecar** — an editable TOML file — so
/// `100_000 × 100_000` is a value a text editor can produce. Without a ceiling
/// that is 10¹⁰ iterations, each pushing a heap `String`, behind a
/// `Vec::with_capacity` whose `u32` product wrapped *small*: the editor hangs
/// and then dies, silently in release, from a file somebody hand-edited.
///
/// 64 Ki cells is far past any real atlas (a 4096² sheet of 16-pixel tiles is
/// 65 536 cells exactly, which is the number this is) and is bounded work.
const MAX_GRID_CELLS: u32 = 1 << 16;

impl GridSlicing {
    /// Row-major tile count, bounded by [`MAX_GRID_CELLS`].
    ///
    /// **`u64`, not `u32`** (C4-15): `columns * rows` in `u32` wraps, and the
    /// release profile has no `overflow-checks`, so `65_536 × 65_536` used to
    /// report **zero** cells and `100_000 × 100_000` reported 1 410 065 408.
    pub fn count(&self) -> u32 {
        (u64::from(self.columns.max(1)) * u64::from(self.rows.max(1))).min(MAX_GRID_CELLS as u64)
            as u32
    }

    /// Whether this grid was clamped by [`MAX_GRID_CELLS`] — the fact an import
    /// advisory needs, since a truncated slice list otherwise looks like an
    /// author's choice.
    pub fn exceeds_ceiling(&self) -> bool {
        u64::from(self.columns.max(1)) * u64::from(self.rows.max(1)) > MAX_GRID_CELLS as u64
    }

    /// Resolve every cell to a [`ResolvedSlice`], named by its row-major index
    /// (`"0"`, `"1"`, …). Cell sizes are computed in `f64` so non-integer
    /// divisions stay precise in UV space; the reported pixel rect is rounded.
    ///
    /// Bounded by [`MAX_GRID_CELLS`]: the loop stops there rather than running
    /// for the rest of the session.
    pub fn resolve(&self, tex_w: u32, tex_h: u32) -> Vec<ResolvedSlice> {
        let cols = self.columns.max(1);
        let rows = self.rows.max(1);
        let budget = self.count() as usize;
        let tw = tex_w.max(1) as f64;
        let th = tex_h.max(1) as f64;

        // Space left for the cells after the top-left margin and inter-cell gaps.
        let usable_w =
            (tw - self.margin_x as f64 - (cols - 1) as f64 * self.padding_x as f64).max(0.0);
        let usable_h =
            (th - self.margin_y as f64 - (rows - 1) as f64 * self.padding_y as f64).max(0.0);
        let cell_w = usable_w / cols as f64;
        let cell_h = usable_h / rows as f64;

        let mut out = Vec::with_capacity(budget);
        'grid: for r in 0..rows {
            for c in 0..cols {
                if out.len() >= budget {
                    break 'grid;
                }
                let x0 = self.margin_x as f64 + c as f64 * (cell_w + self.padding_x as f64);
                let y0 = self.margin_y as f64 + r as f64 * (cell_h + self.padding_y as f64);
                let x1 = x0 + cell_w;
                let y1 = y0 + cell_h;
                out.push(ResolvedSlice {
                    // `u64`: `r * cols + c` in `u32` wraps at a large grid and
                    // produced DUPLICATE slice names — two cells with one
                    // identity (C4-15).
                    name: (u64::from(r) * u64::from(cols) + u64::from(c)).to_string(),
                    px: [
                        px_round(x0),
                        px_round(y0),
                        px_round(cell_w),
                        px_round(cell_h),
                    ],
                    uv_min: [x0 / tw, y0 / th],
                    uv_max: [x1 / tw, y1 / th],
                });
            }
        }
        out
    }
}

/// A pixel coordinate from an `f64` computation, as `u32`.
///
/// `as u32` on a float **saturates** (negative → 0, huge → `u32::MAX`) and maps
/// NaN to 0, so a negative or non-finite rect used to clamp silently to the
/// top-left corner and read as a deliberate one (C4-45). This spells the
/// arithmetic out so the clamp is a decision rather than a language rule nobody
/// re-derives at the call site — and it puts **every** non-finite value at the
/// origin, because an infinity is no more a pixel coordinate than a NaN is.
fn px_round(v: f64) -> u32 {
    if v.is_finite() {
        v.round().clamp(0.0, u32::MAX as f64) as u32
    } else {
        0
    }
}

impl NamedRect {
    fn resolve(&self, tex_w: u32, tex_h: u32) -> ResolvedSlice {
        let tw = tex_w.max(1) as f64;
        let th = tex_h.max(1) as f64;
        let x0 = self.x as f64;
        let y0 = self.y as f64;
        let x1 = x0 + self.width as f64;
        let y1 = y0 + self.height as f64;
        ResolvedSlice {
            name: self.name.clone(),
            px: [self.x, self.y, self.width, self.height],
            uv_min: [x0 / tw, y0 / th],
            uv_max: [x1 / tw, y1 / th],
        }
    }
}

impl SpriteSheetSlices {
    /// Resolve the whole model against a texture's pixel size: grid cells first
    /// (index-named), then the manual rects in author order.
    pub fn resolve(&self, tex_w: u32, tex_h: u32) -> Vec<ResolvedSlice> {
        let mut out = Vec::new();
        if let Some(g) = &self.grid {
            out.extend(g.resolve(tex_w, tex_h));
        }
        for m in &self.manual {
            out.push(m.resolve(tex_w, tex_h));
        }
        out
    }

    /// True when nothing is defined (no grid, no manual rects) — the caller
    /// drops the sidecar key entirely so untouched textures stay clean.
    pub fn is_empty(&self) -> bool {
        self.grid.is_none() && self.manual.is_empty()
    }

    /// Project to the frontend DTO, tagging it with the texture id + pixel size.
    pub fn to_dto(&self, texture_id: String, tex_width: u32, tex_height: u32) -> SpriteSheetDto {
        SpriteSheetDto {
            texture_id,
            grid: self.grid.as_ref().map(|g| SpriteGridDto {
                columns: g.columns,
                rows: g.rows,
                margin_x: g.margin_x,
                margin_y: g.margin_y,
                padding_x: g.padding_x,
                padding_y: g.padding_y,
            }),
            manual: self
                .manual
                .iter()
                .map(|m| SpriteRectDto {
                    name: m.name.clone(),
                    x: m.x,
                    y: m.y,
                    width: m.width,
                    height: m.height,
                })
                .collect(),
            tex_width,
            tex_height,
        }
    }

    /// Build the persisted model from an inbound DTO (pixel dims ignored — they
    /// come from the payload, not the editor).
    pub fn from_dto(dto: &SpriteSheetDto) -> Self {
        Self {
            grid: dto.grid.as_ref().map(|g| GridSlicing {
                columns: g.columns,
                rows: g.rows,
                margin_x: g.margin_x,
                margin_y: g.margin_y,
                padding_x: g.padding_x,
                padding_y: g.padding_y,
            }),
            manual: dto
                .manual
                .iter()
                .map(|m| NamedRect {
                    name: m.name.clone(),
                    x: m.x,
                    y: m.y,
                    width: m.width,
                    height: m.height,
                })
                .collect(),
        }
    }
}

/// Read the slice model + pixel dimensions for a texture asset. Missing/absent
/// slicing yields [`SpriteSheetSlices::default`]; the pixel size comes from the
/// texture payload so the caller can resolve UVs. Errors only on a bad id / a
/// non-texture asset / an unreadable payload.
pub fn read_slices(proj: &AssetProject, id: AssetId) -> Result<(SpriteSheetSlices, u32, u32)> {
    let entry = proj.db().get(id).ok_or(AssetError::UnknownAsset(id))?;
    let slices = entry
        .sidecar
        .import
        .as_ref()
        .and_then(|t| t.get(SPRITE_SHEET_KEY))
        .and_then(|v| v.clone().try_into::<SpriteSheetSlices>().ok())
        .unwrap_or_default();
    // `load_texture`, not `load_payload`: a `.inf_tex` has two container
    // versions since P26.1 and only one of them is bincode (see
    // `AssetProject::load_texture`).
    let tex: TextureAsset = proj.load_texture(id)?;
    Ok((slices, tex.width, tex.height))
}

/// Persist a texture's slice model into its sidecar `import` table (merging
/// beside the existing texture-import settings, never clobbering them) and
/// rewrite the sidecar deterministically. An empty model removes the key.
pub fn write_slices(
    proj: &mut AssetProject,
    id: AssetId,
    slices: &SpriteSheetSlices,
) -> Result<()> {
    let entry = proj.db().get(id).ok_or(AssetError::UnknownAsset(id))?;
    let mut sidecar = entry.sidecar.clone();
    let mut import = sidecar.import.take().unwrap_or_default();
    if slices.is_empty() {
        import.remove(SPRITE_SHEET_KEY);
    } else {
        let value = toml::Value::try_from(slices)
            .map_err(|e| AssetError::Import(format!("serialize slices: {e}")))?;
        import.insert(SPRITE_SHEET_KEY.to_string(), value);
    }
    sidecar.import = if import.is_empty() {
        None
    } else {
        Some(import)
    };
    proj.replace_sidecar(id, sidecar)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn plain_grid_no_margin_padding() {
        // 4×2 grid over a 64×32 atlas → 16×16 cells.
        let g = GridSlicing {
            columns: 4,
            rows: 2,
            ..Default::default()
        };
        let s = g.resolve(64, 32);
        assert_eq!(s.len(), 8);

        // Tile 0 (top-left).
        assert_eq!(s[0].name, "0");
        assert_eq!(s[0].px, [0, 0, 16, 16]);
        approx(s[0].uv_min[0], 0.0);
        approx(s[0].uv_min[1], 0.0);
        approx(s[0].uv_max[0], 0.25);
        approx(s[0].uv_max[1], 0.5);

        // Tile 5 = row 1, col 1.
        assert_eq!(s[5].name, "5");
        assert_eq!(s[5].px, [16, 16, 16, 16]);
        approx(s[5].uv_min[0], 0.25);
        approx(s[5].uv_min[1], 0.5);
        approx(s[5].uv_max[0], 0.5);
        approx(s[5].uv_max[1], 1.0);
    }

    #[test]
    fn grid_with_margin_and_padding() {
        // 2×2 over 40×40, 4px margin, 2px padding.
        // usable = 40 - 4 - 1*2 = 34 → cell = 17.
        let g = GridSlicing {
            columns: 2,
            rows: 2,
            margin_x: 4,
            margin_y: 4,
            padding_x: 2,
            padding_y: 2,
        };
        let s = g.resolve(40, 40);
        assert_eq!(s[0].px, [4, 4, 17, 17]);
        approx(s[0].uv_min[0], 4.0 / 40.0);
        approx(s[0].uv_max[0], 21.0 / 40.0); // 4 + 17

        // col 1 starts at margin + cell + padding = 4 + 17 + 2 = 23.
        assert_eq!(s[1].px, [23, 4, 17, 17]);
        approx(s[1].uv_min[0], 23.0 / 40.0);
        approx(s[1].uv_max[0], 40.0 / 40.0);
    }

    #[test]
    fn non_square_cells_keep_uv_precision() {
        // 3 columns over 100px → 33.333… px cells; UVs must stay exact thirds.
        let g = GridSlicing {
            columns: 3,
            rows: 1,
            ..Default::default()
        };
        let s = g.resolve(100, 10);
        approx(s[0].uv_min[0], 0.0);
        approx(s[1].uv_min[0], 1.0 / 3.0);
        approx(s[2].uv_min[0], 2.0 / 3.0);
        approx(s[2].uv_max[0], 1.0);
        // Pixel rects round to nearest.
        assert_eq!(s[0].px[2], 33);
    }

    #[test]
    fn manual_rects_resolve_after_grid() {
        let model = SpriteSheetSlices {
            grid: Some(GridSlicing {
                columns: 2,
                rows: 1,
                ..Default::default()
            }),
            manual: vec![NamedRect {
                name: "hero".into(),
                x: 10,
                y: 20,
                width: 30,
                height: 40,
            }],
        };
        let s = model.resolve(100, 100);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].name, "0");
        assert_eq!(s[1].name, "1");
        assert_eq!(s[2].name, "hero");
        approx(s[2].uv_min[0], 0.10);
        approx(s[2].uv_min[1], 0.20);
        approx(s[2].uv_max[0], 0.40);
        approx(s[2].uv_max[1], 0.60);
    }

    #[test]
    fn model_toml_round_trips_deterministically() {
        let model = SpriteSheetSlices {
            grid: Some(GridSlicing {
                columns: 8,
                rows: 4,
                margin_x: 1,
                padding_y: 2,
                ..Default::default()
            }),
            manual: vec![NamedRect {
                name: "flag".into(),
                x: 1,
                y: 2,
                width: 3,
                height: 4,
            }],
        };
        let a = toml::to_string_pretty(&model).unwrap();
        let b = toml::to_string_pretty(&model).unwrap();
        assert_eq!(a, b, "re-emit is byte-identical");
        let back: SpriteSheetSlices = toml::from_str(&a).unwrap();
        assert_eq!(back, model);
    }

    /// **C4-15 — the sidecar-fed grid that hung the editor.**
    ///
    /// `columns` and `rows` come out of an editable TOML sidecar. `Vec::with_capacity((cols * rows) as usize)`
    /// was a `u32` multiply, and the release profile has no `overflow-checks`, so
    /// `100_000 x 100_000` wrapped the capacity SMALL and the loop then ran
    /// 10^10 iterations, each pushing a heap `String`. `(r * cols + c).to_string()`
    /// wrapped in the same arithmetic and produced duplicate slice names.
    ///
    /// Un-fix mutation: restore either `u32` product and the corresponding
    /// assertion fails (the count one instantly; the loop one by not returning).
    #[test]
    fn an_absurd_grid_is_bounded_rather_than_run() {
        // The wrap the counts used to take: 65 536 x 65 536 is 2^32, i.e. ZERO
        // in u32, and 100 000^2 is 1 410 065 408.
        let g = |columns, rows| GridSlicing {
            columns,
            rows,
            margin_x: 0,
            margin_y: 0,
            padding_x: 0,
            padding_y: 0,
        };
        assert_eq!(g(65_536, 65_536).count(), MAX_GRID_CELLS);
        assert_eq!(g(100_000, 100_000).count(), MAX_GRID_CELLS);
        assert!(g(100_000, 100_000).exceeds_ceiling());
        assert!(!g(64, 64).exceeds_ceiling());
        // Ordinary grids are unchanged.
        assert_eq!(g(8, 4).count(), 32);
        assert_eq!(g(0, 0).count(), 1);

        // The resolve loop is bounded, and every name it emits is unique — which
        // the wrapped `r * cols + c` could not promise.
        let slices = g(100_000, 100_000).resolve(1024, 1024);
        assert_eq!(slices.len(), MAX_GRID_CELLS as usize);
        let mut names: Vec<&str> = slices.iter().map(|s| s.name.as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "two cells were given one name");
    }

    /// **C4-45 — a float-to-int cast saturates; it does not refuse.**
    ///
    /// `as u32` maps a negative to 0, a huge value to `u32::MAX`, and NaN to 0,
    /// so a nonsense rect used to clamp silently into the top-left corner and
    /// read as a deliberate one. Spelled out, the clamp is a decision.
    #[test]
    fn a_nonsense_pixel_rect_clamps_deliberately_rather_than_by_accident() {
        // Neither a NaN nor an infinity is a pixel coordinate, so both take the
        // origin rather than a saturating edge that would read as authored.
        assert_eq!(px_round(f64::NAN), 0);
        assert_eq!(px_round(f64::INFINITY), 0);
        assert_eq!(px_round(f64::NEG_INFINITY), 0);
        assert_eq!(px_round(-12.7), 0);
        assert_eq!(px_round(1e30), u32::MAX);
        assert_eq!(px_round(12.4), 12);
        assert_eq!(px_round(12.6), 13);
    }
}
