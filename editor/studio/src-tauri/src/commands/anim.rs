//! Ring-2 command surface for **animation clip derivation** (P29.5, pillar S2).
//!
//! Two commands, and between them they are the panel's whole relationship with
//! what the import measured:
//!
//! * [`anim_clip_info`] reads a `.inf_anim` back — its curve channels (with a
//!   sparkline each), its markers, and whether it carries a root-motion track —
//!   so the Blend Space panel can show the derived data rather than assert that
//!   it exists;
//! * [`anim_rederive`] runs the derivation again over a clip already on disk,
//!   which is safe to put on a button because `inf_anim::derive_clip` un-bakes
//!   before it bakes and is therefore idempotent.
//!
//! Every rule is Ring 0 or Ring 1 (`inf_anim::derive`,
//! `inf_editor_core::assets::anim_derive`); this file is the string↔id hop and
//! the projection into a DTO, per the typed-IPC law.
//!
//! # Refusals are values
//!
//! An asset that is not a clip, a clip bound to no rig, a clip with no duration:
//! each comes back inside the DTO rather than as an `Err`, because this is a
//! panel that opens on whatever the author selected and a toast is not an
//! answer. The one `Err` is a malformed id — a caller bug, not a content state.

use inf_anim::{AnimClipAsset, DeriveOptions};
use inf_asset::{AssetId, AssetKind};
use serde::{Deserialize, Serialize};
use tauri::State;

use super::assets::{emit_changed, AssetState};

/// How many points a curve's sparkline carries.
///
/// Enough to read the shape of a foot-speed channel at panel width, few enough
/// that a clip with thirty channels is one small message rather than a copy of
/// its own keyframes.
const SPARK_SAMPLES: usize = 64;

/// One curve channel, projected for the inspector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimCurveDto {
    pub name: String,
    /// How many authored keys it has.
    pub keys: usize,
    pub min: f32,
    pub max: f32,
    /// `SPARK_SAMPLES` uniform samples over the clip, for the sparkline.
    pub samples: Vec<f32>,
    /// Whether this is a channel **this engine derived** rather than one an
    /// author wrote — the distinction a re-derive button has to make visible,
    /// because it replaces the derived ones and leaves the rest alone.
    pub derived: bool,
}

/// One marker, projected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimMarkerDto {
    pub time_s: f32,
    pub name: String,
    /// Empty for an event-only notify.
    pub group: String,
}

/// The baked root motion, projected.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimRootMotionDto {
    /// Total translation over the clip, metres, clip space.
    pub translation: [f32; 3],
    /// Total turn over the clip, **degrees** (the authoring convention; the
    /// track itself is radians).
    pub yaw_deg: f32,
    /// Total ground distance, metres.
    pub distance_m: f32,
    pub keys: usize,
}

/// What a clip carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimClipInfoDto {
    pub id: String,
    pub name: String,
    pub duration_s: f32,
    pub curves: Vec<AnimCurveDto>,
    pub markers: Vec<AnimMarkerDto>,
    pub root_motion: Option<AnimRootMotionDto>,
    /// The rig the clip is bound to, when one resolves — a re-derive needs it.
    pub skeleton: Option<String>,
    /// Why there is nothing to show. A **value**.
    pub refusal: Option<String>,
}

/// What a derivation found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimDeriveDto {
    /// Ground distance the clip travels, metres.
    pub distance_m: f32,
    /// The speed the clip depicts, m/s — the greater of what its root travels
    /// and what its stride says.
    pub avg_speed_mps: f32,
    /// `distance / duration` — what the **root** travels. Zero for an in-place
    /// cycle, which is most authored locomotion.
    pub travel_speed_mps: f32,
    /// `stride x cadence` — what the **feet** say. The number an in-place cycle
    /// answers with, and the one a proposal clusters on when nothing translates.
    pub stride_speed_mps: f32,
    /// How far a foot travels along the ground over one cycle, metres.
    pub stride_m: f32,
    /// The 0–3 `W_Gait` scale.
    pub gait: f32,
    /// Net rise, metres — what the traversal advisory keys on.
    pub rise_m: f32,
    pub plants: usize,
    pub markers: usize,
    pub curves: Vec<String>,
    /// Non-fatal things the author can act on.
    pub advisories: Vec<String>,
    /// Why nothing was derived. A **value**.
    pub refusal: Option<String>,
}

fn parse(id: &str) -> Result<AssetId, String> {
    id.parse::<AssetId>()
        .map_err(|e| format!("bad asset id: {e}"))
}

fn refused(id: &str, why: String) -> AnimClipInfoDto {
    AnimClipInfoDto {
        id: id.to_string(),
        name: String::new(),
        duration_s: 0.0,
        curves: Vec::new(),
        markers: Vec::new(),
        root_motion: None,
        skeleton: None,
        refusal: Some(why),
    }
}

/// Read a `.inf_anim` back: its channels, its markers and its baked root motion.
#[tauri::command]
pub async fn anim_clip_info(
    id: String,
    assets: State<'_, AssetState>,
) -> Result<AnimClipInfoDto, String> {
    let asset_id = parse(&id)?;
    let loaded = assets.with_project(|p| {
        let Some(entry) = p.db().get(asset_id) else {
            return Ok(Err(format!("no asset {asset_id}")));
        };
        if entry.kind() != AssetKind::AnimClip {
            return Ok(Err(format!("{} is not an animation clip", entry.name)));
        }
        let name = entry.name.clone();
        let payload = match p.load_payload::<AnimClipAsset>(asset_id) {
            Ok(a) => a,
            Err(e) => return Ok(Err(e.to_string())),
        };
        let rig = inf_editor_core::assets::anim_derive::skeleton_for(p, asset_id, &payload);
        Ok(Ok((name, payload, rig.is_some())))
    })?;
    let (name, payload, has_rig) = match loaded {
        Ok(v) => v,
        Err(why) => return Ok(refused(&id, why)),
    };
    let clip = &payload.clip;

    let mut curves: Vec<AnimCurveDto> = clip
        .curves
        .iter()
        .map(|c| {
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for v in &c.values {
                if v.is_finite() {
                    min = min.min(*v);
                    max = max.max(*v);
                }
            }
            let samples = (0..SPARK_SAMPLES)
                .map(|k| {
                    let t = clip.duration * (k as f32) / (SPARK_SAMPLES as f32 - 1.0);
                    c.sample(t).filter(|v| v.is_finite()).unwrap_or(0.0)
                })
                .collect();
            AnimCurveDto {
                name: c.name.clone(),
                keys: c.times.len(),
                min: if min.is_finite() { min } else { 0.0 },
                max: if max.is_finite() { max } else { 0.0 },
                samples,
                derived: inf_anim::derive::is_derived_curve(&c.name),
            }
        })
        .collect();
    curves.sort_by(|a, b| a.name.cmp(&b.name));

    let mut markers: Vec<AnimMarkerDto> = clip
        .markers
        .iter()
        .map(|m| AnimMarkerDto {
            time_s: m.time_s,
            name: m.name.clone(),
            group: m.group.clone(),
        })
        .collect();
    markers.sort_by(|a, b| {
        a.time_s
            .total_cmp(&b.time_s)
            .then_with(|| a.name.cmp(&b.name))
    });

    let root_motion = clip.root_motion.as_ref().and_then(|t| {
        let first = *t.translation.first()?;
        let last = *t.translation.last()?;
        let yaw =
            t.yaw_rad.last().copied().unwrap_or(0.0) - t.yaw_rad.first().copied().unwrap_or(0.0);
        Some(AnimRootMotionDto {
            translation: [last[0] - first[0], last[1] - first[1], last[2] - first[2]],
            yaw_deg: yaw.to_degrees(),
            distance_m: clip
                .distance
                .as_ref()
                .and_then(|d| d.distance_m.last().copied())
                .unwrap_or(0.0),
            keys: t.times.len(),
        })
    });

    Ok(AnimClipInfoDto {
        id,
        name,
        duration_s: clip.duration,
        curves,
        markers,
        root_motion,
        skeleton: payload
            .skeleton
            .filter(|_| has_rig)
            .map(|b| uuid::Uuid::from_bytes(b).to_string()),
        refusal: None,
    })
}

/// Re-run the derivation over a clip already on disk and write it back.
///
/// `traversal` picks [`inf_anim::DeriveOptions::traversal`] — the whole vertical
/// on the track and no loop wrap — which is the one decision a clip cannot make
/// for itself (see `inf_anim::VerticalPolicy`) and therefore the one this button
/// has to offer.
#[tauri::command]
pub async fn anim_rederive(
    app: tauri::AppHandle,
    id: String,
    traversal: bool,
    assets: State<'_, AssetState>,
) -> Result<AnimDeriveDto, String> {
    let asset_id = parse(&id)?;
    let opts = if traversal {
        DeriveOptions::traversal()
    } else {
        DeriveOptions::default()
    };
    let name = assets
        .with_project(|p| Ok(p.db().get(asset_id).map(|e| e.name.clone())))?
        .unwrap_or_else(|| id.clone());
    let outcome = assets.with_project(|p| {
        Ok(inf_editor_core::assets::anim_derive::rederive_asset(
            p, asset_id, &opts,
        ))
    })?;
    let outcome = match outcome {
        Ok(o) => o,
        Err(why) => {
            return Ok(AnimDeriveDto {
                distance_m: 0.0,
                avg_speed_mps: 0.0,
                travel_speed_mps: 0.0,
                stride_speed_mps: 0.0,
                stride_m: 0.0,
                gait: 0.0,
                rise_m: 0.0,
                plants: 0,
                markers: 0,
                curves: Vec::new(),
                advisories: Vec::new(),
                refusal: Some(why),
            })
        }
    };
    let advisories = inf_editor_core::assets::anim_derive::advisories(&name, &outcome);
    let dto = match outcome.report() {
        Some(r) => AnimDeriveDto {
            distance_m: r.distance_m,
            avg_speed_mps: r.avg_speed_mps,
            travel_speed_mps: r.travel_speed_mps,
            stride_speed_mps: r.stride_speed_mps,
            stride_m: r.stride_m,
            gait: r.gait,
            rise_m: r.translation[1],
            plants: r.plants.len(),
            markers: r.markers,
            curves: r.curves.clone(),
            advisories,
            refusal: None,
        },
        None => AnimDeriveDto {
            distance_m: 0.0,
            avg_speed_mps: 0.0,
            travel_speed_mps: 0.0,
            stride_speed_mps: 0.0,
            stride_m: 0.0,
            gait: 0.0,
            rise_m: 0.0,
            plants: 0,
            markers: 0,
            curves: Vec::new(),
            advisories,
            refusal: outcome.skipped().map(str::to_string),
        },
    };
    // The clip's bytes moved, so the drawer's thumbnails and dependency view are
    // stale. Only when something was actually written.
    if dto.refusal.is_none() {
        emit_changed(&app, &assets);
    }
    Ok(dto)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A malformed id is the one thing that is an `Err` — a caller bug rather
    /// than a content state.
    #[test]
    fn a_bad_id_is_an_error_and_a_good_one_parses() {
        assert!(parse("not-a-guid").is_err());
        assert!(parse(&AssetId::new().to_string()).is_ok());
    }

    /// The refusal projection is a whole DTO, not a half-filled one — a panel
    /// binding to `curves` must not have to check `refusal` first.
    #[test]
    fn a_refusal_is_a_complete_value() {
        let d = refused("abc", "no such thing".into());
        assert_eq!(d.id, "abc");
        assert!(d.curves.is_empty() && d.markers.is_empty());
        assert!(d.root_motion.is_none());
        assert_eq!(d.refusal.as_deref(), Some("no such thing"));
    }
}
