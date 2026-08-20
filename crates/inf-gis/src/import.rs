//! **The import door** — one function every front end goes through.
//!
//! Wave G left this crate with nine correct modules and no caller: the
//! disposition memo's plainest sentence is that every "the engine now does X"
//! in it should be read as "the engine now *can* do X, from Rust, when
//! something calls it". This module is the something. It is deliberately Ring 0
//! and deliberately free of the editor, so that the editor's wizard, the `inf
//! gis` CLI and a cook-time pipeline are three front ends onto **one** import,
//! not three imports that agree by inspection.
//!
//! # The shape
//!
//! ```text
//!   resolve_crs   ← the .prj sidecar, or what the author stated
//!        │
//!     probe       ← what is in the file: fields, counts, bounds, where on Earth
//!        │
//!   import_layer  ← reproject into the level's anchor, read, plan
//!        │
//!    SpawnPlan    ← a VALUE: exactly what will be created, and what will not be
//! ```
//!
//! The last step is the load-bearing one. A [`SpawnPlan`] is a plain value with
//! no world in it, so the CLI can produce one headlessly and the editor can
//! produce one and *apply* it — and the two can be compared byte for byte on the
//! same fixture, which is what "one door" has to mean if it is to be provable
//! rather than asserted. The applier (`inf_editor_core::gis`) contains no import
//! decision of its own: every count, every skip and every truncation is decided
//! here.
//!
//! # Refusals are values
//!
//! Nothing here panics or guesses. A file with no `.prj` and no stated CRS is a
//! refusal naming the remedy; a level with no geo-anchor is a refusal naming the
//! remedy; Web Mercator as a world anchor stays refused with its 1.53× number
//! (see [`crate::crs::Transform::new`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::{DVec3, Vec3Swizzles};
use inf_math::geo::GeoAnchor;

use crate::crs::Transform;
use crate::feature::{Attr, GeoFeature, GeoGeometry, GeoLayer, LayerKind};
use crate::{epsg, Advisory, GisError};

// ── the .prj sidecar ────────────────────────────────────────────────────────

/// What a `.prj` sidecar told us.
///
/// Every field is optional because a `.prj` is a text file written by whatever
/// tool exported the layer, and "we could not tell" is a real answer that has to
/// travel with the ones we could tell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PrjRead {
    /// A CRS spelling this engine can resolve (`"EPSG:26910"`), when one could be
    /// derived.
    pub crs: Option<String>,
    /// The projection's own name as the file spells it.
    pub name: Option<String>,
    /// Metres per unit of the **vertical** axis, from a `VERT_CS` unit. `1.0`
    /// when the file says nothing, which is also the right answer for the
    /// overwhelming majority of files.
    pub vertical_unit_m: f64,
    /// Hazards: an authority code we could not resolve, a name we had to pattern
    /// match, a vertical unit we had to assume.
    pub advisories: Vec<Advisory>,
}

/// The `.prj` beside a vector source — `roads.shp` → `roads.prj`.
///
/// Shapefile's own convention, and GeoJSON exporters that came from a Shapefile
/// pipeline write one too. Returns the path whether or not it exists; the caller
/// checks.
pub fn prj_path_for(path: &Path) -> PathBuf {
    path.with_extension("prj")
}

/// Read and parse the `.prj` beside `path`, if there is one.
///
/// Returns the raw WKT alongside the parse, because a wizard that cannot resolve
/// a CRS should show the author the text it failed on — that is the difference
/// between "unsupported" and "here is what your file says, is this it?".
pub fn read_prj(path: &Path) -> Option<(String, PrjRead)> {
    let p = prj_path_for(path);
    let wkt = std::fs::read_to_string(&p).ok()?;
    let read = parse_prj_wkt(&wkt);
    Some((wkt, read))
}

/// The value of the last `AUTHORITY["EPSG","<code>"]` in a WKT string.
///
/// **The last, not the first**, and that is the whole trick: WKT nests, and the
/// inner `GEOGCS`/`DATUM`/`SPHEROID` objects carry their own authority codes. A
/// `PROJCS` for NAD83 / UTM zone 10N contains `AUTHORITY["EPSG","4269"]` (its
/// geographic base), `AUTHORITY["EPSG","7019"]` (the ellipsoid) and, at the very
/// end where the outermost object closes, `AUTHORITY["EPSG","26910"]` — the one
/// that names the file. Taking the first would import every projected layer as
/// its own datum's geographic CRS, which reads coordinates in the hundreds of
/// thousands as degrees and refuses at the axis-order door with a confusing
/// message.
fn last_epsg_authority(wkt: &str) -> Option<u32> {
    let upper = wkt.to_ascii_uppercase();
    let mut found = None;
    let mut from = 0usize;
    while let Some(rel) = upper[from..].find("AUTHORITY") {
        let at = from + rel;
        // Scan forward to the closing bracket of this AUTHORITY[...] and read the
        // two quoted arguments out of it.
        let Some(close_rel) = upper[at..].find(']') else {
            break;
        };
        let body = &wkt[at..at + close_rel];
        let quoted: Vec<&str> = body
            .split('"')
            .skip(1)
            .step_by(2)
            .map(str::trim)
            .collect::<Vec<_>>();
        if quoted.len() >= 2 && quoted[0].eq_ignore_ascii_case("EPSG") {
            if let Ok(code) = quoted[1].parse::<u32>() {
                found = Some(code);
            }
        }
        from = at + close_rel;
    }
    found
}

/// The first quoted string inside the outermost `PROJCS[` / `GEOGCS[` — the
/// CRS's own name.
fn wkt_name(wkt: &str) -> Option<String> {
    let upper = wkt.to_ascii_uppercase();
    let at = upper
        .find("PROJCS[")
        .or_else(|| upper.find("GEOGCS["))
        .or_else(|| upper.find("PROJCRS["))
        .or_else(|| upper.find("GEOGCRS["))?;
    let rest = &wkt[at..];
    let mut parts = rest.split('"');
    parts.next()?;
    parts.next().map(|s| s.trim().to_string())
}

/// A UTM EPSG code derived from a CRS *name*, for files whose `.prj` carries no
/// authority code at all.
///
/// ESRI's exporter writes exactly such files — `PROJCS["NAD_1983_UTM_Zone_10N",…]`
/// with no `AUTHORITY` anywhere — and they are the single most common thing a
/// North-American government portal hands an author. Matching the name is a
/// guess, so it comes back with an advisory saying it was one.
fn utm_epsg_from_name(name: &str) -> Option<u32> {
    let folded: String = name
        .chars()
        .filter(|c| !(*c == '_' || *c == '-' || c.is_whitespace()))
        .flat_map(char::to_lowercase)
        .collect();
    let at = folded.find("utmzone")?;
    let tail = &folded[at + "utmzone".len()..];
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    let zone: u32 = digits.parse().ok()?;
    if !(1..=60).contains(&zone) {
        return None;
    }
    let south = tail[digits.len()..].starts_with('s');
    // NAD83 first: `nad1983` folds to contain "nad1983", and NAD83's own UTM
    // codes (269xx) are the ones a Canadian or US municipal layer is in.
    if folded.contains("nad1983") || folded.contains("nad83") {
        if south || zone > 23 {
            return None;
        }
        return Some(26900 + zone);
    }
    if folded.contains("wgs1984") || folded.contains("wgs84") {
        return Some(if south { 32700 + zone } else { 32600 + zone });
    }
    None
}

/// The metres-per-unit of a `VERT_CS`'s `UNIT[…]`, when the file has one.
fn vertical_unit_from_wkt(wkt: &str) -> Option<f64> {
    let upper = wkt.to_ascii_uppercase();
    let at = upper.find("VERT_CS[").or_else(|| upper.find("VERTCRS["))?;
    let rest_upper = &upper[at..];
    let rel = rest_upper
        .find("UNIT[")
        .or_else(|| rest_upper.find("LENGTHUNIT["))?;
    let body_start = at + rel;
    let close = upper[body_start..].find(']')?;
    let body = &wkt[body_start..body_start + close];
    // `UNIT["foot",0.3048]` — the factor is the token after the closing quote.
    let after_name = body.split('"').nth(2)?;
    let number: String = after_name
        .trim_start_matches(|c: char| c == ',' || c.is_whitespace())
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '-')
        .collect();
    number
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0)
}

/// Parse a `.prj`'s WKT into whatever of it this engine can honour.
///
/// This is **not** a WKT parser and does not pretend to be one — writing one is
/// a real project and the disposition memo says so. It is a scan for the three
/// facts an import needs: the authority code, the projection's name (as a
/// fallback), and the vertical unit. Everything it cannot answer becomes a
/// `None` plus an advisory, and the author states the CRS by hand.
pub fn parse_prj_wkt(wkt: &str) -> PrjRead {
    let mut out = PrjRead {
        vertical_unit_m: 1.0,
        ..Default::default()
    };
    out.name = wkt_name(wkt);

    if let Some(code) = last_epsg_authority(wkt) {
        if epsg::derived_proj4(code).is_some() {
            out.crs = Some(format!("EPSG:{code}"));
        } else {
            out.advisories.push(Advisory::new(
                "crs.prj_code_unknown",
                format!(
                    "the .prj names EPSG:{code}, which is not in this engine's \
                     curated CRS table (it ships no proj.db). Look the code up at \
                     epsg.io and paste its proj4 string as the source CRS — that is \
                     accepted verbatim and has no such limit."
                ),
            ));
        }
    }
    if out.crs.is_none() {
        if let Some(code) = out.name.as_deref().and_then(utm_epsg_from_name) {
            out.crs = Some(format!("EPSG:{code}"));
            out.advisories.push(Advisory::new(
                "crs.prj_name_matched",
                format!(
                    "the .prj carries no authority code, so its CRS was derived from \
                     the projection NAME {:?} as EPSG:{code}. That is a pattern \
                     match, not a lookup — check it before importing a city.",
                    out.name.as_deref().unwrap_or("")
                ),
            ));
        }
    }
    if out.crs.is_none() {
        out.advisories.push(Advisory::new(
            "crs.prj_unreadable",
            "the .prj sidecar could not be resolved to a coordinate system this \
             engine knows. State the source CRS by hand — an EPSG code \
             (\"EPSG:26910\") or a proj4 string pasted from epsg.io."
                .to_string(),
        ));
    }

    if let Some(v) = vertical_unit_from_wkt(wkt) {
        out.vertical_unit_m = v;
    }
    out
}

// ── the resolved CRS ────────────────────────────────────────────────────────

/// Where a source's CRS came from. Provenance, so an import report can say
/// whether an author chose it or a file did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrsOrigin {
    /// The caller (wizard field, `--crs` flag) stated it.
    Stated,
    /// Read out of the `.prj` sidecar's authority code.
    Prj,
    /// Derived from the `.prj`'s projection *name* — a pattern match.
    PrjName,
}

impl CrsOrigin {
    pub const fn label(self) -> &'static str {
        match self {
            CrsOrigin::Stated => "stated",
            CrsOrigin::Prj => "prj",
            CrsOrigin::PrjName => "prj-name",
        }
    }
}

/// A source CRS, resolved, with where it came from and what it cost.
#[derive(Clone, Debug, PartialEq)]
pub struct CrsChoice {
    /// The spelling to hand [`Transform`] — an EPSG code or a proj4 string.
    pub spec: String,
    pub origin: CrsOrigin,
    /// The file's own name for it, when a `.prj` gave one.
    pub name: Option<String>,
    /// Metres per unit of the source's vertical axis.
    pub vertical_unit_m: f64,
    /// The `.prj` text, when there was one — so a wizard can show it.
    pub prj_wkt: Option<String>,
    pub advisories: Vec<Advisory>,
}

/// Resolve a source's CRS: what the caller stated, else the `.prj`, else a
/// refusal naming the remedy.
///
/// A stated CRS always wins — the `.prj` is a hint from whatever tool wrote the
/// file, and an author who has typed a code has more information than it does.
/// The `.prj` is still read and still reported when they disagree, because a
/// silent override is how a layer lands in the wrong hemisphere.
pub fn resolve_crs(path: &Path, stated: Option<&str>) -> Result<CrsChoice, GisError> {
    let prj = read_prj(path);
    let stated = stated.map(str::trim).filter(|s| !s.is_empty());

    let (wkt, read) = match prj {
        Some((w, r)) => (Some(w), Some(r)),
        None => (None, None),
    };
    let vertical_unit_m = read.as_ref().map_or(1.0, |r| r.vertical_unit_m);
    let name = read.as_ref().and_then(|r| r.name.clone());

    if let Some(spec) = stated {
        // Validate the spelling here so the refusal names the field the author
        // typed into rather than surfacing three calls later.
        epsg::proj4_for(spec)?;
        let mut advisories = Vec::new();
        if let Some(r) = &read {
            if let Some(from_prj) = &r.crs {
                if !crs_specs_agree(from_prj, spec) {
                    advisories.push(Advisory::new(
                        "crs.overrides_prj",
                        format!(
                            "the .prj sidecar says {from_prj} and this import was told \
                             {spec}. The stated value wins. If the import lands \
                             somewhere unexpected, this is the first thing to check."
                        ),
                    ));
                }
            } else {
                advisories.extend(r.advisories.iter().cloned());
            }
        }
        return Ok(CrsChoice {
            spec: spec.to_string(),
            origin: CrsOrigin::Stated,
            name,
            vertical_unit_m,
            prj_wkt: wkt,
            advisories,
        });
    }

    match read {
        Some(r) => match r.crs.clone() {
            Some(spec) => {
                let origin = if r
                    .advisories
                    .iter()
                    .any(|a| a.code == "crs.prj_name_matched")
                {
                    CrsOrigin::PrjName
                } else {
                    CrsOrigin::Prj
                };
                Ok(CrsChoice {
                    spec,
                    origin,
                    name,
                    vertical_unit_m,
                    prj_wkt: wkt,
                    advisories: r.advisories,
                })
            }
            None => Err(GisError::Crs(format!(
                "{path:?} has a .prj sidecar this engine could not resolve to a \
                 coordinate system, and no CRS was stated. State one — an EPSG code \
                 (\"EPSG:26910\") or a proj4 string pasted from epsg.io, which is \
                 accepted verbatim. The .prj says: {}",
                truncate_for_message(wkt.as_deref().unwrap_or(""), 220)
            ))),
        },
        None => Err(GisError::Crs(format!(
            "{path:?} has no .prj sidecar and no CRS was stated, so there is nothing \
             that says what its coordinates mean. A Shapefile keeps its coordinate \
             system in a .prj beside it; a GeoJSON is WGS 84 (EPSG:4326) by the \
             format's own rule unless its producer says otherwise. State the source \
             CRS — an EPSG code (\"EPSG:4326\", \"EPSG:26910\") or a proj4 string."
        ))),
    }
}

/// Whether two CRS spellings name the same projection, compared through the one
/// door that resolves them ([`epsg::proj4_for`]) rather than by string equality —
/// `"EPSG:32610"`, `"32610"` and the proj4 string are three spellings of one CRS.
fn crs_specs_agree(a: &str, b: &str) -> bool {
    match (epsg::proj4_for(a), epsg::proj4_for(b)) {
        (Ok(x), Ok(y)) => x.trim() == y.trim(),
        _ => false,
    }
}

fn truncate_for_message(s: &str, max: usize) -> String {
    let one_line = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max {
        return one_line;
    }
    let head: String = one_line.chars().take(max).collect();
    format!("{head}…")
}

// ── the probe ───────────────────────────────────────────────────────────────

/// One attribute field, summarised over a whole layer.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSummary {
    pub name: String,
    /// How many features carry a value for it (a present-but-unset `Null` does
    /// not count).
    pub present: usize,
    /// How many of those values are usable as numbers.
    pub numeric: usize,
    /// The first non-empty value, as text — what a wizard shows beside the name.
    pub sample: Option<String>,
}

/// What is inside a vector source, before anything is imported.
#[derive(Clone, Debug, PartialEq)]
pub struct GisProbe {
    pub path: PathBuf,
    /// `"shapefile"` or `"geojson"`.
    pub format: &'static str,
    /// The layer's name — the file stem.
    pub layer_name: String,
    pub features: usize,
    pub points: usize,
    pub polylines: usize,
    pub polygons: usize,
    /// Field names in a deterministic order, with their coverage.
    pub fields: Vec<FieldSummary>,
    pub crs: CrsChoice,
    /// Bounds in the source's **own** coordinates, `(min, max)` packed
    /// `(x, elevation, y)`. `None` for an empty layer.
    pub bounds_source: Option<(DVec3, DVec3)>,
    /// The layer's centre as `(latitude, longitude)` in degrees, when it could
    /// be derived.
    pub centre_lat_lon: Option<(f64, f64)>,
    /// The UTM zone EPSG code this engine suggests anchoring a world at, derived
    /// from the centre.
    pub suggested_anchor_epsg: Option<u32>,
    /// Records the reader could not use, with the reason.
    pub skipped: Vec<String>,
    pub advisories: Vec<Advisory>,
}

impl GisProbe {
    /// The field names, in order — the shape a wizard's attribute picker wants.
    pub fn field_names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// The dominant geometry kind, which is what decides whether a layer can be
    /// roads (lines) or footprints (areas).
    pub fn dominant_kind(&self) -> &'static str {
        if self.polygons >= self.polylines && self.polygons >= self.points {
            "polygon"
        } else if self.polylines >= self.points {
            "polyline"
        } else {
            "point"
        }
    }
}

fn format_for(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .as_deref()
    {
        Some("shp") => Some("shapefile"),
        Some("geojson") | Some("json") => Some("geojson"),
        _ => None,
    }
}

/// Look inside a vector source **without** importing it: fields, counts,
/// bounds, and where on Earth it sits.
///
/// Reads in the source's own coordinate space ([`Transform::source_space`]), so
/// it works before the level has a geo-anchor — which is the whole point, since
/// what the probe finds is how an author decides what to anchor to.
pub fn probe(path: &Path, stated_crs: Option<&str>) -> Result<GisProbe, GisError> {
    let format = format_for(path).ok_or_else(|| {
        GisError::Format(format!(
            "this engine reads Shapefile (.shp, with its .dbf beside it) and GeoJSON \
             (.geojson, .json); {path:?} is neither. Other vector formats need a \
             conversion step on your machine — `ogr2ogr out.geojson in.gpkg` covers \
             most of them."
        ))
    })?;
    let crs = resolve_crs(path, stated_crs)?;
    let tf = Transform::source_space(&crs.spec)?;
    let layer = crate::vector::read_vector(path, LayerKind::Generic, &crs.spec, &tf)?;

    let mut points = 0usize;
    let mut polylines = 0usize;
    let mut polygons = 0usize;
    let mut fields: BTreeMap<String, FieldSummary> = BTreeMap::new();
    for f in &layer.features {
        match &f.geometry {
            GeoGeometry::Point(_) => points += 1,
            GeoGeometry::Polyline { .. } => polylines += 1,
            GeoGeometry::Polygon { .. } => polygons += 1,
        }
        for (k, v) in &f.attributes {
            let e = fields.entry(k.clone()).or_insert_with(|| FieldSummary {
                name: k.clone(),
                present: 0,
                numeric: 0,
                sample: None,
            });
            if matches!(v, Attr::Null) {
                continue;
            }
            e.present += 1;
            if v.as_number().is_some() {
                e.numeric += 1;
            }
            if e.sample.is_none() {
                let text = v.to_string();
                if !text.trim().is_empty() {
                    e.sample = Some(text);
                }
            }
        }
    }

    let bounds_source = layer.bounds_xz();
    let centre_lat_lon = bounds_source.and_then(|(lo, hi)| {
        let cx = (lo.x + hi.x) * 0.5;
        let cy = (lo.z + hi.z) * 0.5;
        source_centre_to_lat_lon(&crs.spec, cx, cy)
    });
    let suggested_anchor_epsg = centre_lat_lon.map(|(lat, lon)| epsg::suggested_utm_epsg(lat, lon));

    let mut advisories = crs.advisories.clone();
    advisories.extend(layer.advisories.iter().cloned());

    Ok(GisProbe {
        path: path.to_path_buf(),
        format,
        layer_name: layer.name.clone(),
        features: layer.features.len(),
        points,
        polylines,
        polygons,
        fields: fields.into_values().collect(),
        crs,
        bounds_source,
        centre_lat_lon,
        suggested_anchor_epsg,
        skipped: layer.skipped,
        advisories,
    })
}

/// A source-space centre → `(latitude, longitude)` in degrees.
///
/// Reuses [`crate::crs::anchor_at`] for the projected case rather than opening a
/// second inverse-projection path: the anchor door already derives the geodetic
/// position of a projected origin, and having two functions that invert a
/// projection is how two answers to "where is this" come about.
fn source_centre_to_lat_lon(spec: &str, x: f64, y: f64) -> Option<(f64, f64)> {
    if !(x.is_finite() && y.is_finite()) {
        return None;
    }
    let proj4 = epsg::proj4_for(spec).ok()?;
    if proj4.contains("+proj=longlat") {
        // A geographic source is already in degrees, in file order (x = lon).
        if !(-180.0..=180.0).contains(&x) || !(-90.0..=90.0).contains(&y) {
            return None;
        }
        return Some((y, x));
    }
    let anchor = crate::crs::anchor_at(spec, x, y, 0.0, "unknown").ok()?;
    Some((anchor.origin_latitude_deg, anchor.origin_longitude_deg))
}

// ── options, the plan, and the report ───────────────────────────────────────

/// The default ceiling on entities one vector layer may create.
///
/// **A guard, not a preference.** A county road layer is ~10⁵ features and a
/// metropolitan footprint layer more, and an author who points the wizard at one
/// by accident should get a bounded result and a sentence, not a hung editor. It
/// is *reported* whenever it bites ([`SpawnPlan::truncated`]) and it is raisable
/// at the door — see [`ISLAND_MAX_ENTITIES`].
pub const DEFAULT_MAX_ENTITIES: usize = 4_096;

/// The ceiling an island-scale import needs.
///
/// The certification measured 10 000 roads and 50 000 footprints truncating to
/// 4 096 with 5 904 and 45 904 dropped. This is the value that imports both
/// whole, with headroom: it is not a *default*, because raising the default
/// would turn a misdirected wizard into a hung editor, but it is the number an
/// author who means it asks for.
pub const ISLAND_MAX_ENTITIES: usize = 262_144;

/// A feature shorter than this is a digitising stub, not a road.
pub const MIN_FEATURE_LENGTH_M: f64 = 8.0;

/// Defaults for a stream whose attribute row does not state its channel.
pub const DEFAULT_STREAM_WIDTH_M: f64 = 3.0;
/// See [`DEFAULT_STREAM_WIDTH_M`].
pub const DEFAULT_STREAM_DEPTH_M: f64 = 0.8;
/// See [`DEFAULT_STREAM_WIDTH_M`].
pub const DEFAULT_STREAM_FLOW_M_S: f64 = 0.6;

/// What an import may create, and what it may not.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportOptions {
    /// Features whose plan-view length is below this are skipped as digitising
    /// stubs.
    pub min_length_m: f64,
    /// The hard ceiling on entities. See [`DEFAULT_MAX_ENTITIES`].
    pub max_entities: usize,
    /// Reverse every stream's flow direction (only meaningful for
    /// [`LayerKind::Streams`]).
    pub reverse_flow: bool,
    /// Prefix for generated entity names; empty falls back to the layer's name.
    pub name_prefix: String,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            min_length_m: MIN_FEATURE_LENGTH_M,
            max_entities: DEFAULT_MAX_ENTITIES,
            reverse_flow: false,
            name_prefix: String::new(),
        }
    }
}

/// What one planned entity is.
#[derive(Clone, Debug, PartialEq)]
pub enum PlannedKind {
    /// A polyline or a closed ring, as a `Spline`.
    Spline { closed: bool },
    /// A watercourse: a `WaterBody` + `Spline` pair.
    River {
        width_m: f64,
        depth_m: f64,
        flow_m_s: f64,
    },
}

/// One entity an import will create — named, positioned and typed, with no
/// world in it.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedEntity {
    pub name: String,
    /// World-metre positions, already past the CRS door.
    pub points: Vec<DVec3>,
    pub kind: PlannedKind,
    /// Which feature of the layer this came from — so a report can name it.
    pub feature: usize,
}

/// **The plan**: exactly what an import will create and exactly what it will
/// not, as a value.
///
/// Two front ends that produce equal plans from equal inputs are one importer.
/// That is the property this type exists to make testable, and
/// `inf_gis::import`'s own tests plus the `inf gis` CLI gate both assert it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpawnPlan {
    pub entities: Vec<PlannedEntity>,
    /// Features below [`ImportOptions::min_length_m`].
    pub too_short: usize,
    /// Features with no usable run at all (a lone point, a one-vertex line).
    pub unusable: usize,
    /// Features dropped by [`ImportOptions::max_entities`]. **Reported, never
    /// silent** — and the remedy is named in [`SpawnPlan::summary`].
    pub truncated: usize,
    /// The cap that produced `truncated`, carried so the report can name the
    /// number an author would have to raise.
    pub cap: usize,
    pub advisories: Vec<String>,
}

impl SpawnPlan {
    pub fn count(&self) -> usize {
        self.entities.len()
    }

    /// **A stable digest of everything this plan will create.**
    ///
    /// The one-door proof needs a comparison two *processes* can make: the
    /// editor's Ring-2 command and `inf gis` both build a plan, and "the same
    /// plan" has to mean something a test can check across a process boundary
    /// rather than something a comment claims.
    ///
    /// FNV-1a over the name bytes, the kind discriminant and each coordinate's
    /// **bit pattern** — `to_bits`, not a formatted decimal, because a plan is
    /// exact and a rounded comparison would pass for two imports that differ by
    /// a metre in the eighth digit. Deterministic across machines: this is
    /// integer arithmetic over IEEE bit patterns, with no float operation in it.
    ///
    /// The entity count and every name are **length-prefixed** (I2 audit), so
    /// what is folded is a self-delimiting encoding rather than a concatenation
    /// in which the only variable-length field — the name — runs straight into a
    /// fixed-width word it could have spelled. That costs two `u64`s and removes
    /// a whole class of question from the one-door proof's foundation.
    ///
    /// What it deliberately does **not** fold is [`SpawnPlan::advisories`]: the
    /// digest answers "will these two imports create the same things", and an
    /// advisory is a sentence about the file rather than a thing that is
    /// created. `the_digest_separates_plans_that_differ_by_one_bit` is what
    /// keeps every other field in.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let byte = |h: &mut u64, b: u8| {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        let word = |h: &mut u64, v: u64| {
            for b in v.to_le_bytes() {
                byte(h, b);
            }
        };
        word(&mut h, self.entities.len() as u64);
        for e in &self.entities {
            word(&mut h, e.name.len() as u64);
            for b in e.name.as_bytes() {
                byte(&mut h, *b);
            }
            word(&mut h, e.feature as u64);
            match e.kind {
                PlannedKind::Spline { closed } => {
                    word(&mut h, if closed { 1 } else { 0 });
                }
                PlannedKind::River {
                    width_m,
                    depth_m,
                    flow_m_s,
                } => {
                    word(&mut h, 2);
                    for v in [width_m, depth_m, flow_m_s] {
                        word(&mut h, v.to_bits());
                    }
                }
            }
            word(&mut h, e.points.len() as u64);
            for p in &e.points {
                for v in [p.x, p.y, p.z] {
                    word(&mut h, v.to_bits());
                }
            }
        }
        for v in [self.too_short, self.unusable, self.truncated, self.cap] {
            word(&mut h, v as u64);
        }
        h
    }

    /// A one-line summary an author can act on.
    ///
    /// The truncation clause names **the cap and the remedy**, not just the
    /// count: the certification's complaint about this cap was never that it was
    /// silent — it was that there was nowhere to raise it and nothing that said
    /// how.
    pub fn summary(&self, layer: &str) -> String {
        self.summary_with_count(layer, self.entities.len())
    }

    /// [`SpawnPlan::summary`] with the entity count supplied — the shape an
    /// *applied* plan needs, because by then the entities are GUIDs in a
    /// document and the plan has been consumed.
    ///
    /// One sentence, written once. The wizard's done state, the editor log and
    /// `inf gis import`'s stdout are three readers of it, and a second spelling
    /// is how two front ends come to describe one import differently.
    pub fn summary_with_count(&self, layer: &str, count: usize) -> String {
        let mut parts = vec![format!("{count} entities")];
        if self.too_short > 0 {
            parts.push(format!("{} too short", self.too_short));
        }
        if self.unusable > 0 {
            parts.push(format!("{} unusable", self.unusable));
        }
        if self.truncated > 0 {
            parts.push(format!(
                "{} NOT IMPORTED (the entity cap is {}; raise it in the import \
                 wizard, or pass --max on `inf gis import`, to take the whole layer)",
                self.truncated, self.cap
            ));
        }
        format!("{layer}: {}", parts.join(", "))
    }
}

/// The name to give a feature's entity: the source's own name field when it has
/// one, else the layer plus an index.
pub fn feature_name(f: &GeoFeature, fallback: &str, index: usize) -> String {
    const NAME_FIELDS: [&str; 8] = [
        "name",
        "street_name",
        "full_name",
        "fullname",
        "streetname",
        "gnis_name",
        "rd_name",
        "label",
    ];
    match f.attr_text(&NAME_FIELDS) {
        Some(n) => n.trim().to_string(),
        None if fallback.trim().is_empty() => format!("Feature {index}"),
        None => format!("{fallback} {index}"),
    }
}

/// The point runs a feature contributes: one for a polyline, one per ring for a
/// polygon, none for a point.
pub fn runs_of(f: &GeoFeature) -> Vec<Vec<DVec3>> {
    match &f.geometry {
        GeoGeometry::Polyline { points, .. } => vec![points.clone()],
        GeoGeometry::Polygon { exterior, holes } => {
            let mut out = vec![exterior.clone()];
            out.extend(holes.iter().cloned());
            out
        }
        GeoGeometry::Point(_) => Vec::new(),
    }
}

/// Plan-view length of a run, in metres.
pub fn run_length(pts: &[DVec3]) -> f64 {
    pts.windows(2)
        .map(|w| (w[1].xz() - w[0].xz()).length())
        .sum()
}

/// A stream's channel, read off its attribute row with the spellings real
/// hydrography layers use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamAttrs {
    pub width_m: f64,
    pub depth_m: f64,
    pub flow_m_s: f64,
}

/// Read a stream's channel, falling back to the module defaults.
pub fn stream_attrs(f: &GeoFeature) -> StreamAttrs {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    StreamAttrs {
        width_m: f
            .attr_number(&["width_m", "width", "chan_width", "bankfull_w"])
            .filter(|v| positive(*v))
            .unwrap_or(DEFAULT_STREAM_WIDTH_M),
        depth_m: f
            .attr_number(&["depth_m", "depth", "mean_depth"])
            .filter(|v| positive(*v))
            .unwrap_or(DEFAULT_STREAM_DEPTH_M),
        flow_m_s: f
            .attr_number(&["flow_m_s", "velocity", "flow"])
            .filter(|v| positive(*v))
            .unwrap_or(DEFAULT_STREAM_FLOW_M_S),
    }
}

/// Turn a layer into the plan of what it will create.
///
/// **Every import decision lives here** — naming, the stub filter, the entity
/// cap, the stream channel — so that a front end contains none of them and two
/// front ends cannot drift.
pub fn plan_spawn(layer: &GeoLayer, opts: &ImportOptions) -> SpawnPlan {
    let mut plan = SpawnPlan {
        cap: opts.max_entities,
        ..Default::default()
    };
    let fallback = if opts.name_prefix.trim().is_empty() {
        layer.name.as_str()
    } else {
        opts.name_prefix.trim()
    };
    let min_len = if opts.min_length_m.is_finite() {
        opts.min_length_m.max(0.0)
    } else {
        MIN_FEATURE_LENGTH_M
    };

    for (i, f) in layer.features.iter().enumerate() {
        let runs = runs_of(f);
        if runs.is_empty() {
            plan.unusable += 1;
            continue;
        }
        let name = feature_name(f, fallback, i);
        let stream = stream_attrs(f);
        for (r, pts) in runs.into_iter().enumerate() {
            if pts.len() < 2 {
                plan.unusable += 1;
                continue;
            }
            if run_length(&pts) < min_len {
                plan.too_short += 1;
                continue;
            }
            if plan.entities.len() >= opts.max_entities {
                plan.truncated += 1;
                continue;
            }
            let name = if r == 0 {
                name.clone()
            } else {
                format!("{name} (hole {r})")
            };
            let kind = match layer.kind {
                LayerKind::Streams => PlannedKind::River {
                    width_m: stream.width_m,
                    depth_m: stream.depth_m,
                    flow_m_s: stream.flow_m_s,
                },
                _ => PlannedKind::Spline {
                    closed: matches!(f.geometry, GeoGeometry::Polygon { .. }),
                },
            };
            let mut pts = pts;
            if opts.reverse_flow && layer.kind == LayerKind::Streams {
                pts.reverse();
            }
            plan.entities.push(PlannedEntity {
                name,
                points: pts,
                kind,
                feature: i,
            });
        }
    }

    plan.advisories
        .extend(layer.advisories.iter().map(|a| a.to_string()));
    if !layer.skipped.is_empty() {
        plan.advisories.push(format!(
            "{} record(s) were skipped by the reader; the first was: {}",
            layer.skipped.len(),
            layer.skipped[0]
        ));
    }
    plan
}

// ── the door ────────────────────────────────────────────────────────────────

/// Everything one import needs to be reproducible.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportRequest {
    pub path: PathBuf,
    pub kind: LayerKind,
    /// `None` reads the `.prj`; `Some` overrides it.
    pub source_crs: Option<String>,
    /// `None` reads the `.prj`'s `VERT_CS`; `Some` overrides it.
    pub vertical_unit_m: Option<f64>,
    pub options: ImportOptions,
}

impl ImportRequest {
    pub fn new(path: impl Into<PathBuf>, kind: LayerKind) -> Self {
        Self {
            path: path.into(),
            kind,
            source_crs: None,
            vertical_unit_m: None,
            options: ImportOptions::default(),
        }
    }
}

/// One imported layer: the features, the CRS it came through, and the plan.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerImport {
    pub layer: GeoLayer,
    pub crs: CrsChoice,
    pub plan: SpawnPlan,
}

/// **The import.** Resolve the CRS, reproject into the level's anchor, read,
/// and plan.
///
/// # Refusals, each naming its remedy
///
/// * **No geo-anchor.** [`Transform::new`] refuses a disabled anchor, and it has
///   to: without one there is no answer to "where does easting 491 234 go", and
///   guessing puts the whole import at the world origin where it looks imported.
/// * **A geographic or Web-Mercator anchor.** Refused with the 1.53× number, in
///   [`crate::crs`], where the reason lives.
/// * **A source CRS this engine cannot build.** Refused with the proj4 escape
///   hatch named.
/// * **A probe transform.** [`Transform::source_space`] is for looking, never
///   for importing, and this door says so rather than trusting a caller.
pub fn import_layer(req: &ImportRequest, anchor: &GeoAnchor) -> Result<LayerImport, GisError> {
    if format_for(&req.path).is_none() {
        return Err(GisError::Format(format!(
            "this engine reads Shapefile (.shp) and GeoJSON (.geojson, .json); \
             {:?} is neither.",
            req.path
        )));
    }
    let mut crs = resolve_crs(&req.path, req.source_crs.as_deref())?;
    if let Some(v) = req.vertical_unit_m {
        if !(v.is_finite() && v > 0.0) {
            return Err(GisError::NotFinite(format!(
                "the stated vertical unit {v} is not a positive finite number of \
                 metres per unit (feet are 0.3048)"
            )));
        }
        if (v - crs.vertical_unit_m).abs() > 1e-12 && crs.prj_wkt.is_some() {
            crs.advisories.push(Advisory::new(
                "vertical.overrides_prj",
                format!(
                    "the .prj implies {} m per vertical unit and this import was told \
                     {v}. The stated value wins.",
                    crs.vertical_unit_m
                ),
            ));
        }
        crs.vertical_unit_m = v;
    }

    let tf = Transform::with_vertical_unit(&crs.spec, anchor, crs.vertical_unit_m)?;
    debug_assert!(!tf.is_source_space());
    let mut layer = crate::vector::read_vector(&req.path, req.kind, &crs.spec, &tf)?;
    // The transform's own advisories reach the layer through the readers; the
    // CRS choice's (a .prj override, a name match) do not, and they are exactly
    // the ones an author needs.
    for a in &crs.advisories {
        if !layer.advisories.contains(a) {
            layer.advisories.push(a.clone());
        }
    }
    let plan = plan_spawn(&layer, &req.options);
    Ok(LayerImport { layer, crs, plan })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crs::anchor_at;

    const NAD83_UTM10_WKT: &str = r#"PROJCS["NAD83 / UTM zone 10N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","26910"]]"#;

    /// **The last authority wins, and it is not a style choice.** This WKT names
    /// five EPSG codes; four of them describe the ellipsoid, the prime meridian,
    /// the angular unit and the *geographic base*. Taking the first would import
    /// a UTM layer as EPSG:4269 — degrees — and every easting in the file is
    /// then a longitude of 491 234, which the axis-order door refuses with a
    /// message about the wrong continent.
    #[test]
    fn a_prj_resolves_to_its_outermost_authority_not_its_first() {
        let read = parse_prj_wkt(NAD83_UTM10_WKT);
        assert_eq!(read.crs.as_deref(), Some("EPSG:26910"), "{read:?}");
        assert_eq!(read.name.as_deref(), Some("NAD83 / UTM zone 10N"));
        assert_eq!(read.vertical_unit_m, 1.0);
        assert!(read.advisories.is_empty(), "{:?}", read.advisories);
        // The control: the first authority in the text is the ellipsoid's.
        assert!(NAD83_UTM10_WKT.find("7019").unwrap() < NAD83_UTM10_WKT.find("26910").unwrap());
    }

    /// ESRI writes `.prj` files with no authority code at all, and they are the
    /// most common thing a municipal portal hands an author. Matching the name
    /// is a guess and says so.
    #[test]
    fn an_esri_prj_with_no_authority_falls_back_to_its_name_and_advises() {
        let wkt = r#"PROJCS["NAD_1983_UTM_Zone_10N",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian",-123.0],UNIT["Meter",1.0]]"#;
        let read = parse_prj_wkt(wkt);
        assert_eq!(read.crs.as_deref(), Some("EPSG:26910"), "{read:?}");
        assert!(
            read.advisories
                .iter()
                .any(|a| a.code == "crs.prj_name_matched"),
            "a pattern match must say it was one: {:?}",
            read.advisories
        );
        // A WGS 84 zone resolves to the 326xx family, and a southern one carries
        // the +south false northing.
        assert_eq!(
            parse_prj_wkt(r#"PROJCS["WGS_1984_UTM_Zone_55S",GEOGCS["GCS_WGS_1984"]]"#)
                .crs
                .as_deref(),
            Some("EPSG:32755")
        );
    }

    /// A `.prj` we cannot resolve is a refusal with the remedy and the file's own
    /// text, not a silent guess.
    #[test]
    fn an_unresolvable_prj_refuses_and_shows_what_the_file_says() {
        let wkt = r#"PROJCS["Some_Local_Grid_1927",PROJECTION["Krovak"],UNIT["Meter",1.0]]"#;
        let read = parse_prj_wkt(wkt);
        assert_eq!(read.crs, None);
        assert!(read
            .advisories
            .iter()
            .any(|a| a.code == "crs.prj_unreadable"));

        let dir = tempfile::tempdir().unwrap();
        let shp = dir.path().join("roads.shp");
        std::fs::write(&shp, b"not a shapefile").unwrap();
        std::fs::write(dir.path().join("roads.prj"), wkt).unwrap();
        let e = resolve_crs(&shp, None).unwrap_err().to_string();
        assert!(
            e.contains("Krovak"),
            "the .prj's own text must be shown: {e}"
        );
        assert!(e.contains("epsg.io"), "the remedy must be named: {e}");

        // Stating one wins, and the disagreement is reported rather than hidden.
        let c = resolve_crs(&shp, Some("EPSG:26910")).unwrap();
        assert_eq!(c.spec, "EPSG:26910");
        assert_eq!(c.origin, CrsOrigin::Stated);
    }

    #[test]
    fn a_stated_crs_that_disagrees_with_the_prj_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let shp = dir.path().join("roads.shp");
        std::fs::write(&shp, b"x").unwrap();
        std::fs::write(dir.path().join("roads.prj"), NAD83_UTM10_WKT).unwrap();

        let same = resolve_crs(&shp, Some("EPSG:26910")).unwrap();
        assert!(
            same.advisories.is_empty(),
            "agreeing spellings raise nothing: {:?}",
            same.advisories
        );
        // The same CRS spelled as a bare number is still the same CRS — the
        // comparison goes through `proj4_for`, not through string equality.
        let bare = resolve_crs(&shp, Some("26910")).unwrap();
        assert!(bare.advisories.is_empty(), "{:?}", bare.advisories);

        let other = resolve_crs(&shp, Some("EPSG:32610")).unwrap();
        assert!(
            other
                .advisories
                .iter()
                .any(|a| a.code == "crs.overrides_prj"),
            "a silent override is how a layer lands in the wrong datum: {:?}",
            other.advisories
        );
    }

    /// A vertical unit is read once, applied once, and changes the elevation by
    /// exactly its factor.
    #[test]
    fn a_vertical_unit_is_read_from_the_prj_and_applied_exactly_once() {
        let wkt = format!(
            r#"COMPD_CS["NAD83 / UTM zone 10N + NAVD88 height (ftUS)",{NAD83_UTM10_WKT},VERT_CS["NAVD88 height (ftUS)",VERT_DATUM["North American Vertical Datum 1988",2005],UNIT["US survey foot",0.304800609601219]]]"#
        );
        let read = parse_prj_wkt(&wkt);
        assert_eq!(read.crs.as_deref(), Some("EPSG:26910"));
        assert!(
            (read.vertical_unit_m - 0.304_800_609_601_219).abs() < 1e-15,
            "got {}",
            read.vertical_unit_m
        );

        let anchor = anchor_at("EPSG:26910", 491_000.0, 5_459_000.0, 0.0, "NAVD88").unwrap();
        let metres = Transform::new("EPSG:26910", &anchor).unwrap();
        let feet = Transform::with_vertical_unit("EPSG:26910", &anchor, 0.3048).unwrap();
        let a = metres.to_world(491_100.0, 5_459_100.0, 100.0).unwrap();
        let b = feet.to_world(491_100.0, 5_459_100.0, 100.0).unwrap();
        assert!(
            (a.x - b.x).abs() < 1e-9 && (a.z - b.z).abs() < 1e-9,
            "{a:?} {b:?}"
        );
        assert!(
            (b.y - 30.48).abs() < 1e-9 && (a.y - 100.0).abs() < 1e-9,
            "100 feet is 30.48 m, not {}: {a:?} {b:?}",
            b.y
        );
        // A vertical unit of zero or a NaN is refused rather than flattening or
        // NaN-ing every elevation.
        assert!(Transform::with_vertical_unit("EPSG:26910", &anchor, 0.0).is_err());
        assert!(Transform::with_vertical_unit("EPSG:26910", &anchor, f64::NAN).is_err());
    }

    /// **BEFORE the projection — and an anchor at datum zero cannot see it.**
    ///
    /// [`Transform::with_vertical_unit`]'s doc says the conversion happens
    /// before `GeoAnchor::world_from_projected` "because it subtracts
    /// `origin_height_m`, which is in metres: scaling afterwards would scale the
    /// anchor's own datum height along with the sample". The arm above anchors
    /// at `origin_height_m = 0`, where the two orderings are algebraically
    /// identical — so moving the multiplication past the anchor **passed every
    /// test in this crate** (measured, I2 audit). This is the arm that fails
    /// when it moves.
    #[test]
    fn the_vertical_unit_is_applied_before_the_anchors_metric_datum_height() {
        // Fifty metres of datum, which is what an anchor above sea level looks
        // like — and the only thing that makes the two orders differ.
        let datum = anchor_at("EPSG:26910", 491_000.0, 5_459_000.0, 50.0, "NAVD88").unwrap();
        assert_eq!(datum.origin_height_m, 50.0, "the fixture must have a datum");
        let feet = Transform::with_vertical_unit("EPSG:26910", &datum, 0.3048).unwrap();
        let w = feet.to_world(491_100.0, 5_459_100.0, 100.0).unwrap();

        // Convert first: 100 ft is 30.48 m, and the anchor's origin is 50 m up.
        let before = 100.0 * 0.3048 - 50.0;
        assert!(
            (w.y - before).abs() < 1e-9,
            "the vertical unit must be applied BEFORE the anchor's metric datum \
             height is subtracted; got {} rather than {before}",
            w.y
        );
        // THE ALTERNATIVE, PRICED: converting after the anchor gives
        // (100 - 50) x 0.3048 = 15.24 m — 34.76 m of silent error on every
        // vertex of every layer, with nothing to see but a world that is in the
        // wrong place vertically.
        let after = (100.0 - 50.0) * 0.3048;
        assert!(
            (after - w.y).abs() > 30.0,
            "the two orderings differ by {:.2} m; if that is small this fixture \
             has no datum in it and the arm cannot fail",
            (after - w.y).abs()
        );
        println!(
            "IB-11 vertical unit: before-the-anchor {before:.2} m vs \
             after-the-anchor {after:.2} m ({:.2} m apart)",
            (after - before).abs()
        );
    }

    /// **What the digest can and cannot tell apart**, measured.
    ///
    /// [`SpawnPlan::digest`] is the foundation of the one-door proof — the CLI
    /// prints it and `tools/inf-cli/tests/cli.rs` compares it against a plan
    /// built in-process — and nothing was holding it to its own claims. Dropping
    /// two of the four counts out of the fold passed every test in this crate
    /// (measured, I2 audit), and the stated reason for hashing **bit patterns**
    /// rather than a formatted decimal had no arm at all.
    ///
    /// So: one ULP of one coordinate is a different plan, every count is part of
    /// the plan, and an advisory is not.
    #[test]
    fn the_digest_separates_plans_that_differ_by_one_bit() {
        let base = SpawnPlan {
            entities: vec![
                PlannedEntity {
                    name: "Main St".into(),
                    points: vec![DVec3::new(1.0, 2.0, 3.0), DVec3::new(4.0, 5.0, 6.0)],
                    kind: PlannedKind::Spline { closed: false },
                    feature: 0,
                },
                PlannedEntity {
                    name: "Still Creek".into(),
                    points: vec![DVec3::new(7.0, 8.0, 9.0)],
                    kind: PlannedKind::River {
                        width_m: 3.0,
                        depth_m: 0.8,
                        flow_m_s: 0.6,
                    },
                    feature: 1,
                },
            ],
            too_short: 2,
            unusable: 3,
            truncated: 4,
            cap: 4_096,
            advisories: vec!["the layer is in feet".into()],
        };
        let d = base.digest();
        assert_eq!(base.clone().digest(), d, "a plan is its own digest");

        // **ONE ULP of ONE coordinate** — the claim `to_bits` exists to make.
        let mut ulp = base.clone();
        ulp.entities[0].points[0].y = f64::from_bits(2.0f64.to_bits() + 1);
        assert_ne!(
            ulp.entities[0].points[0].y, 2.0,
            "the fixture has to actually move"
        );
        assert_eq!(
            format!("{:.12}", ulp.entities[0].points[0].y),
            format!("{:.12}", 2.0),
            "…and it has to be invisible to a formatted comparison, or this arm \
             measures rounding rather than bits"
        );

        let mut counts_moved = Vec::new();
        for i in 0..4 {
            let mut p = base.clone();
            match i {
                0 => p.too_short += 1,
                1 => p.unusable += 1,
                2 => p.truncated += 1,
                _ => p.cap += 1,
            }
            counts_moved.push((["too_short", "unusable", "truncated", "cap"][i], p.digest()));
        }

        let mut dropped = base.clone();
        dropped.entities.pop();
        let mut reordered = base.clone();
        reordered.entities.swap(0, 1);
        let mut renamed = base.clone();
        renamed.entities[0].name = "Main Street".into();
        let mut rekinded = base.clone();
        rekinded.entities[0].kind = PlannedKind::Spline { closed: true };
        let mut refeatured = base.clone();
        refeatured.entities[0].feature = 7;
        let mut rechannelled = base.clone();
        rechannelled.entities[1].kind = PlannedKind::River {
            width_m: 3.5,
            depth_m: 0.8,
            flow_m_s: 0.6,
        };

        let mut differ: Vec<(&str, u64)> = vec![
            ("one ulp of one coordinate", ulp.digest()),
            ("a dropped entity", dropped.digest()),
            ("a reordered plan", reordered.digest()),
            ("a renamed entity", renamed.digest()),
            ("a closed ring", rekinded.digest()),
            ("a different source feature", refeatured.digest()),
            ("a wider channel", rechannelled.digest()),
        ];
        differ.extend(counts_moved);
        for (what, other) in &differ {
            assert_ne!(*other, d, "the digest cannot see {what}");
        }

        // An ADVISORY is deliberately not folded: it is a sentence about the
        // file, not a thing the import creates.
        let mut chattier = base.clone();
        chattier.advisories.push("and its .prj guessed".into());
        assert_eq!(chattier.digest(), d);
    }

    fn write_geojson(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    const TWO_ROADS: &str = r#"{"type":"FeatureCollection","features":[
      {"type":"Feature","properties":{"name":"Main St","lanes":4},
       "geometry":{"type":"LineString","coordinates":[[-123.12,49.28],[-123.11,49.28]]}},
      {"type":"Feature","properties":{"name":"Stub","lanes":null},
       "geometry":{"type":"LineString","coordinates":[[-123.12,49.28],[-123.119995,49.28]]}}
    ]}"#;

    /// The probe reads a file **before** the level has an anchor, which is the
    /// whole reason it can exist: what it finds is how the author decides what
    /// to anchor to.
    #[test]
    fn the_probe_reads_a_file_with_no_anchor_and_suggests_one() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_geojson(dir.path(), "roads.geojson", TWO_ROADS);
        let probe = probe(&p, Some("EPSG:4326")).unwrap();
        assert_eq!(probe.format, "geojson");
        assert_eq!(probe.features, 2);
        assert_eq!(probe.polylines, 2);
        assert_eq!(probe.dominant_kind(), "polyline");
        assert_eq!(
            probe.field_names(),
            vec!["lanes".to_string(), "name".into()]
        );
        let lanes = probe.fields.iter().find(|f| f.name == "lanes").unwrap();
        assert_eq!(
            (lanes.present, lanes.numeric),
            (1, 1),
            "a JSON null is present-but-unset and must not count as a value"
        );
        assert_eq!(
            probe
                .fields
                .iter()
                .find(|f| f.name == "name")
                .unwrap()
                .sample
                .as_deref(),
            Some("Main St")
        );
        // Vancouver is UTM zone 10 north.
        assert_eq!(
            probe.suggested_anchor_epsg,
            Some(32610),
            "{:?}",
            probe.centre_lat_lon
        );
        let (lat, lon) = probe.centre_lat_lon.unwrap();
        assert!(
            (lat - 49.28).abs() < 1e-6 && (lon + 123.115).abs() < 1e-3,
            "{lat} {lon}"
        );
        // Source bounds are the file's own degrees, not world metres.
        let (lo, hi) = probe.bounds_source.unwrap();
        assert!(lo.x < -123.1 && hi.x < -123.1, "{lo:?} {hi:?}");
    }

    /// The plan is a value: the same request against the same file gives the
    /// same plan, and the cap is reported with the number to raise.
    #[test]
    fn the_plan_is_deterministic_and_names_the_cap_it_hit() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_geojson(dir.path(), "roads.geojson", TWO_ROADS);
        let anchor = anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008").unwrap();

        let mut req = ImportRequest::new(&p, LayerKind::Roads);
        req.source_crs = Some("EPSG:4326".into());
        let a = import_layer(&req, &anchor).unwrap();
        let b = import_layer(&req, &anchor).unwrap();
        assert_eq!(a.plan, b.plan, "two imports of one file are one plan");
        assert_eq!(a.plan.count(), 1, "the 0.4 m stub is below the 8 m floor");
        assert_eq!(a.plan.too_short, 1);
        assert_eq!(a.plan.entities[0].name, "Main St");
        assert!(matches!(
            a.plan.entities[0].kind,
            PlannedKind::Spline { closed: false }
        ));

        req.options.max_entities = 0;
        let capped = import_layer(&req, &anchor).unwrap().plan;
        assert_eq!((capped.count(), capped.truncated), (0, 1));
        let s = capped.summary("Roads");
        assert!(s.contains("NOT IMPORTED"), "{s}");
        assert!(
            s.contains("the entity cap is 0") && s.contains("--max"),
            "the remedy has to name the number and where to change it: {s}"
        );
    }

    /// A level with no geo-anchor is a refusal, not an import at the world
    /// origin — and Web Mercator stays refused as an anchor with its number.
    #[test]
    fn the_anchor_refusals_reach_the_import_door() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_geojson(dir.path(), "roads.geojson", TWO_ROADS);
        let mut req = ImportRequest::new(&p, LayerKind::Roads);
        req.source_crs = Some("EPSG:4326".into());

        let e = import_layer(&req, &GeoAnchor::default())
            .unwrap_err()
            .to_string();
        assert!(
            e.to_ascii_lowercase().contains("anchor"),
            "the refusal must name the anchor: {e}"
        );

        let mut merc = anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008").unwrap();
        merc.crs = "EPSG:3857".into();
        let e = import_layer(&req, &merc).unwrap_err().to_string();
        assert!(e.contains("1.53") && e.contains("Web Mercator"), "{e}");
        // And it stays legal as a SOURCE.
        req.source_crs = Some("EPSG:3857".into());
        let anchor = anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "EGM2008").unwrap();
        assert!(import_layer(&req, &anchor).is_ok());
    }

    /// A probe transform can look and can never import.
    #[test]
    fn a_source_space_transform_is_probe_only() {
        let tf = Transform::source_space("EPSG:26910").unwrap();
        assert!(tf.is_source_space());
        let w = tf.to_world(491_100.0, 5_459_100.0, 12.0).unwrap();
        assert_eq!(w, DVec3::new(491_100.0, 12.0, 5_459_100.0));
        // The axis-order guard still runs, which is exactly where an author
        // wants to hear about a transposed file.
        let geo = Transform::source_space("EPSG:4326").unwrap();
        assert!(geo.to_world(49.28, -123.12, 0.0).is_err());
    }
}
