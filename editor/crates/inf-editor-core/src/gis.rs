//! GIS vector layers → scene entities (Wave G, Ring 1).
//!
//! [`inf_gis`] turns a Shapefile or a GeoJSON into world-metre geometry. This
//! module is the other half: it turns that geometry into things the engine's
//! existing systems already know how to simulate and draw.
//!
//! # It spawns through the doors that already exist
//!
//! Nothing here invents a component or a persistence path. A stream becomes the
//! same `WaterBody` + `Spline` pair the hydrology tool creates by hand; a road
//! or a boundary becomes the same `Spline` a designer would draw. That is
//! deliberate and it is what makes the import *cheap*: the river validator, the
//! uphill cook advisory, the buoyancy solver, the flow-to-foam wiring and the
//! grammar's span source all work on GIS-imported water and GIS-imported roads
//! on the day they land, because none of them can tell where the spline came
//! from.
//!
//! It is also what keeps this module from being a schema event. Every spawn
//! below is an ordinary scene edit; nothing new is persisted, so nothing new is
//! versioned.
//!
//! # Flow direction comes from the vertex order
//!
//! A published stream layer digitises each watercourse **downstream** — that is
//! the near-universal convention, and it is the only flow information a polyline
//! carries. So the polyline's own order becomes the river's flow order, and
//! [`SpawnOptions::reverse_flow`] exists for the layers that got it backwards.
//! An import that guessed instead would produce rivers running uphill, which the
//! P20.4 cook advisory would then report by the hundred — correct, and useless.

use glam::{DVec3, Vec3Swizzles};
use inf_gis::feature::{GeoFeature, GeoGeometry, GeoLayer, LayerKind};
use uuid::Uuid;

use crate::scene::SceneDoc;

/// Default river width when a stream layer carries no width attribute, metres.
///
/// Small on purpose. A published hydrography layer is mostly headwater creeks,
/// and a default sized for a main stem would flood every valley in the import;
/// an author widening the handful of real rivers afterwards is a better first
/// hour than an author narrowing four thousand creeks.
pub const DEFAULT_STREAM_WIDTH_M: f64 = 3.0;
/// Default river depth, metres.
pub const DEFAULT_STREAM_DEPTH_M: f64 = 0.8;
/// Default surface flow speed, m/s — a walking-pace creek.
pub const DEFAULT_STREAM_FLOW_M_S: f64 = 0.6;

/// The shortest feature worth spawning, metres.
///
/// Vector layers are full of stubs — a two-metre driveway apron, a culvert
/// fragment, the residue of a clipped tile edge. Each would become an entity
/// with a spline of no useful length, and ten thousand of them would make a
/// level nobody can open. Named so an author can lower it deliberately.
pub const MIN_FEATURE_LENGTH_M: f64 = 8.0;

/// What to do with a layer, and how much of it.
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnOptions {
    /// Skip features shorter than this (metres). See [`MIN_FEATURE_LENGTH_M`].
    pub min_length_m: f64,
    /// Stop after this many entities. **A guard, not a preference**: a county
    /// road layer is ~10⁵ features and spawning all of them into a document that
    /// keeps every entity's record in memory is how an editor stops responding.
    /// The import reports what it left behind.
    pub max_entities: usize,
    /// Reverse each polyline before spawning — for a stream layer digitised
    /// upstream. Has no effect on non-flowing kinds.
    pub reverse_flow: bool,
    /// Prefix for generated entity names when a feature carries no name
    /// attribute.
    pub name_prefix: String,
}

impl Default for SpawnOptions {
    fn default() -> Self {
        Self {
            min_length_m: MIN_FEATURE_LENGTH_M,
            max_entities: 4096,
            reverse_flow: false,
            name_prefix: String::new(),
        }
    }
}

/// What a spawn produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpawnReport {
    /// The entities created, in spawn order.
    pub spawned: Vec<Uuid>,
    /// Features skipped for being shorter than `min_length_m`.
    pub too_short: usize,
    /// Features skipped because they carried no usable geometry.
    pub unusable: usize,
    /// Features left unspawned because `max_entities` was reached.
    ///
    /// **Reported, never silent.** An import that quietly stops at four thousand
    /// of forty thousand roads produces a city with a hard edge and no
    /// explanation.
    pub truncated: usize,
    /// Non-fatal advisories, including the layer's own.
    pub advisories: Vec<String>,
}

impl SpawnReport {
    pub fn count(&self) -> usize {
        self.spawned.len()
    }

    /// A one-line summary for the log and the wizard's done state.
    pub fn summary(&self, layer: &str) -> String {
        let mut s = format!("{}: {} entities", layer, self.spawned.len());
        if self.too_short > 0 {
            s.push_str(&format!(", {} too short", self.too_short));
        }
        if self.unusable > 0 {
            s.push_str(&format!(", {} unusable", self.unusable));
        }
        if self.truncated > 0 {
            s.push_str(&format!(", {} NOT IMPORTED (entity cap)", self.truncated));
        }
        s
    }
}

/// A feature's name, from whichever attribute the source used.
fn feature_name(f: &GeoFeature, fallback: &str, index: usize) -> String {
    f.attr_text(&[
        "name",
        "street_name",
        "full_name",
        "fullname",
        "streetname",
        "gnis_name",
        "label",
    ])
    .map(|s| s.trim().to_string())
    .filter(|s| !s.is_empty())
    .unwrap_or_else(|| {
        if fallback.is_empty() {
            format!("Feature {index}")
        } else {
            format!("{fallback} {index}")
        }
    })
}

/// Every polyline a feature contributes, as world-space runs.
fn runs_of(f: &GeoFeature) -> Vec<Vec<DVec3>> {
    match &f.geometry {
        GeoGeometry::Polyline { points, .. } => vec![points.clone()],
        GeoGeometry::Polygon { exterior, holes } => std::iter::once(exterior.clone())
            .chain(holes.iter().cloned())
            .collect(),
        GeoGeometry::Point(_) => Vec::new(),
    }
}

fn run_length(pts: &[DVec3]) -> f64 {
    pts.windows(2)
        .map(|w| (w[1].xz() - w[0].xz()).length())
        .sum()
}

/// Spawn a vector layer into the document.
///
/// The layer's [`kind`](GeoLayer::kind) decides what each feature becomes:
///
/// | kind | becomes |
/// |---|---|
/// | [`Streams`](LayerKind::Streams) | a `WaterBody::river` + `Spline` — the same pair the hydrology tool creates, so the river validator, the uphill advisory and the buoyancy solver all apply |
/// | everything else | a `Spline` entity, which the grammar's polyline span and the road builder both consume |
///
/// Areas (`Lakes`, `Biomes`, `Buildings`, `Parcels`) spawn their **boundaries**
/// as closed splines. Turning a boundary into a lake surface or a building lot
/// needs the polygon-interior work this wave deliberately deferred — see the
/// disposition memo — and a closed spline is the honest half that exists today.
pub fn spawn_layer(doc: &mut SceneDoc, layer: &GeoLayer, opts: &SpawnOptions) -> SpawnReport {
    let mut report = SpawnReport {
        advisories: layer.advisories.iter().map(|a| a.to_string()).collect(),
        ..Default::default()
    };
    if !layer.skipped.is_empty() {
        report.advisories.push(format!(
            "{} feature(s) of this layer could not be read and were skipped; the \
             first was: {}",
            layer.skipped.len(),
            layer.skipped.first().cloned().unwrap_or_default()
        ));
    }

    let prefix = if opts.name_prefix.is_empty() {
        layer.name.clone()
    } else {
        opts.name_prefix.clone()
    };

    for (i, f) in layer.features.iter().enumerate() {
        let runs = runs_of(f);
        if runs.is_empty() {
            report.unusable += 1;
            continue;
        }
        for run in runs {
            if run.len() < 2 {
                report.unusable += 1;
                continue;
            }
            if run_length(&run) < opts.min_length_m {
                report.too_short += 1;
                continue;
            }
            if report.spawned.len() >= opts.max_entities {
                report.truncated += 1;
                continue;
            }
            let mut pts = run;
            // Flow order is the vertex order — see the module docs.
            if opts.reverse_flow && layer.kind == LayerKind::Streams {
                pts.reverse();
            }
            let name = feature_name(f, &prefix, i);
            let guid = match layer.kind {
                LayerKind::Streams => {
                    let width = f
                        .attr_number(&["width_m", "width", "chan_width", "bankfull_w"])
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .unwrap_or(DEFAULT_STREAM_WIDTH_M);
                    let depth = f
                        .attr_number(&["depth_m", "depth", "mean_depth"])
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .unwrap_or(DEFAULT_STREAM_DEPTH_M);
                    let flow = f
                        .attr_number(&["flow_m_s", "velocity", "flow"])
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .unwrap_or(DEFAULT_STREAM_FLOW_M_S);
                    doc.edit_create_river(&name, &pts, width, depth, flow)
                }
                _ => spawn_spline(
                    doc,
                    &name,
                    &pts,
                    matches!(&f.geometry, GeoGeometry::Polygon { .. }),
                ),
            };
            report.spawned.push(guid);
        }
    }
    report
}

/// Spawn a bare `Spline` entity at the run's first point.
///
/// The points are stored in the entity's own frame — the same convention
/// `edit_create_river` uses — so an entity dragged in the viewport carries its
/// path with it instead of leaving it behind at the world origin.
fn spawn_spline(doc: &mut SceneDoc, name: &str, pts: &[DVec3], closed: bool) -> Uuid {
    use inf_ecs::components::{Spline, SplineInterp, Transform};
    use inf_ecs::math::Vec3d;

    let origin = pts.first().copied().unwrap_or(DVec3::ZERO);
    let guid = doc.edit_create(crate::ipc::SpawnKind::Empty, name, None);
    if let Some(entity) = doc.world().entity_of(guid) {
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(origin.x, origin.y, origin.z);
        doc.world_mut().world_mut().entity_mut(entity).insert((
            Spline {
                points: pts
                    .iter()
                    .map(|p| {
                        let l = *p - origin;
                        Vec3d::new(l.x, l.y, l.z)
                    })
                    .collect(),
                closed,
                // **Linear, not Catmull-Rom.** A surveyed centreline's vertices
                // ARE the road; smoothing through them invents curvature the
                // survey does not have and pulls the path off its own right of
                // way at every bend. A hand-drawn spline wants smoothing; an
                // imported one does not.
                interp: SplineInterp::Linear,
            },
            t,
        ));
    }
    guid
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_gis::feature::{Attr, GeoLayer};

    fn line(pts: &[(f64, f64)]) -> GeoGeometry {
        GeoGeometry::Polyline {
            points: pts.iter().map(|&(x, z)| DVec3::new(x, 0.0, z)).collect(),
            closed: false,
        }
    }

    fn layer(kind: LayerKind, features: Vec<GeoFeature>) -> GeoLayer {
        let mut l = GeoLayer::new("Test", kind, "EPSG:32610");
        l.features = features;
        l
    }

    /// **A stream layer becomes rivers the existing hydrology understands.**
    ///
    /// The point of the arm is the last assertion: the imported entity is
    /// indistinguishable from a hand-drawn one, so `river_path_of` — the door
    /// the validator, the cook advisory and the buoyancy solver all read
    /// through — resolves it with no GIS-specific code anywhere.
    #[test]
    fn a_stream_layer_becomes_rivers_the_hydrology_tools_can_read() {
        let mut doc = SceneDoc::new();
        let mut creek = GeoFeature::new(line(&[(0.0, 0.0), (0.0, 60.0), (30.0, 120.0)]));
        creek
            .attributes
            .insert("NAME".into(), Attr::Text("Still Creek".into()));
        creek.attributes.insert("WIDTH_M".into(), Attr::Number(6.5));

        let report = spawn_layer(
            &mut doc,
            &layer(LayerKind::Streams, vec![creek]),
            &SpawnOptions::default(),
        );
        assert_eq!(report.count(), 1, "{report:?}");
        let guid = report.spawned[0];

        // It is a real river as far as the hydrology layer is concerned.
        let path = crate::hydro::river_path_of(&doc, guid)
            .expect("the imported stream resolves through the hydrology door");
        assert!(
            (path.length_m - 127.08).abs() < 1.0,
            "the river is {} m long; the polyline is ~127",
            path.length_m
        );

        // The attribute reached the component rather than the default.
        let world = doc.world();
        let e = world.entity_of(guid).unwrap();
        let body = world
            .world()
            .get::<inf_ecs::components::WaterBody>(e)
            .expect("a WaterBody");
        assert_eq!(body.kind, inf_ecs::components::WaterKind::River);
        assert!((body.river_width_start_m - 6.5).abs() < 1e-9);
        // …and the ones it did NOT carry took the documented defaults.
        assert!((body.river_depth_start_m - DEFAULT_STREAM_DEPTH_M).abs() < 1e-9);
        assert_eq!(world.name_of(e).as_deref(), Some("Still Creek"));
    }

    /// **Flow direction is the vertex order**, and `reverse_flow` is the escape
    /// hatch for a layer that digitised it the other way.
    ///
    /// Un-fix mutation: ignore `reverse_flow` and the two paths below start at
    /// the same place.
    #[test]
    fn flow_direction_follows_the_polyline_and_can_be_reversed() {
        let pts = &[(0.0, 0.0), (0.0, 100.0)];
        let mk = |rev: bool| {
            let mut doc = SceneDoc::new();
            let r = spawn_layer(
                &mut doc,
                &layer(LayerKind::Streams, vec![GeoFeature::new(line(pts))]),
                &SpawnOptions {
                    reverse_flow: rev,
                    ..Default::default()
                },
            );
            let path = crate::hydro::river_path_of(&doc, r.spawned[0]).expect("a river");
            path.frames[0].center
        };
        let downstream = mk(false);
        let upstream = mk(true);
        assert!(
            (downstream.z - 0.0).abs() < 1e-6,
            "the river should start at the polyline's FIRST point, got {downstream:?}"
        );
        assert!(
            (upstream.z - 100.0).abs() < 1e-6,
            "reversed, it should start at the last, got {upstream:?}"
        );
    }

    /// Roads and boundaries become splines the grammar and the road builder can
    /// consume — with **linear** interpolation, because a survey's vertices are
    /// the road.
    #[test]
    fn roads_become_linear_splines_and_rings_close() {
        let mut doc = SceneDoc::new();
        let road = GeoFeature::new(line(&[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]));
        let parcel = GeoFeature::new(GeoGeometry::Polygon {
            exterior: vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(40.0, 0.0, 0.0),
                DVec3::new(40.0, 0.0, 40.0),
                DVec3::new(0.0, 0.0, 40.0),
            ],
            holes: vec![],
        });
        let mut l = layer(LayerKind::Roads, vec![road]);
        l.features.push(parcel);
        let report = spawn_layer(&mut doc, &l, &SpawnOptions::default());
        assert_eq!(report.count(), 2, "{report:?}");

        let world = doc.world();
        let splines: Vec<inf_ecs::components::Spline> = report
            .spawned
            .iter()
            .map(|g| {
                world
                    .world()
                    .get::<inf_ecs::components::Spline>(world.entity_of(*g).unwrap())
                    .cloned()
                    .expect("a Spline")
            })
            .collect();
        for s in &splines {
            assert_eq!(
                s.interp,
                inf_ecs::components::SplineInterp::Linear,
                "an imported centreline must not be smoothed — its vertices ARE the road"
            );
        }
        assert!(!splines[0].closed, "an open road stays open");
        assert!(splines[1].closed, "a polygon boundary comes in as a ring");
        // Points are entity-local, so the first is the origin.
        assert_eq!(splines[0].points[0], inf_ecs::math::Vec3d::ZERO);
        assert_eq!(splines[0].points.len(), 3);
    }

    /// **Stubs are skipped, the cap truncates, and both are REPORTED.**
    ///
    /// A vector layer is full of two-metre fragments and a county road file is
    /// 10⁵ features; an import that silently drops either produces a city with a
    /// hard edge and no explanation.
    #[test]
    fn stubs_and_the_entity_cap_are_reported_rather_than_silent() {
        let mut doc = SceneDoc::new();
        let mut features = vec![
            GeoFeature::new(line(&[(0.0, 0.0), (2.0, 0.0)])), // a 2 m stub
            GeoFeature::new(GeoGeometry::Point(DVec3::ZERO)), // no run at all
        ];
        for i in 0..5 {
            features.push(GeoFeature::new(line(&[
                (0.0, i as f64 * 10.0),
                (50.0, i as f64 * 10.0),
            ])));
        }
        let report = spawn_layer(
            &mut doc,
            &layer(LayerKind::Roads, features),
            &SpawnOptions {
                max_entities: 3,
                ..Default::default()
            },
        );
        assert_eq!(report.count(), 3, "the cap holds");
        assert_eq!(report.too_short, 1);
        assert_eq!(report.unusable, 1);
        assert_eq!(report.truncated, 2, "what the cap left behind is COUNTED");
        let s = report.summary("Roads");
        assert!(s.contains("NOT IMPORTED"), "{s}");
        assert!(s.contains("too short"), "{s}");
    }

    /// A layer's own advisories ride through to the import's report — an axis
    /// warning or a datum-error note must not stop at the reader.
    #[test]
    fn layer_advisories_reach_the_spawn_report() {
        let mut doc = SceneDoc::new();
        let mut l = layer(LayerKind::Roads, vec![]);
        l.advisories
            .push(inf_gis::Advisory::new("datum.no_grid", "a 20 m offset"));
        l.skipped.push("record 4: non-finite coordinate".into());
        let report = spawn_layer(&mut doc, &l, &SpawnOptions::default());
        assert_eq!(report.count(), 0);
        assert!(
            report.advisories.iter().any(|a| a.contains("20 m offset")),
            "{:?}",
            report.advisories
        );
        assert!(
            report.advisories.iter().any(|a| a.contains("record 4")),
            "{:?}",
            report.advisories
        );
    }
}
