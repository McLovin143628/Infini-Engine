//! Ring-2 command surface for the 2D blend-space authoring panel (P29.2).
//!
//! **One command, and no state.** The panel owns its sample list and hands the
//! whole thing over on every change; the backend answers with everything the
//! canvas draws — the Delaunay triangulation, its hull, the barycentric weights
//! at the query point, the weight-blended cycle length, and the blended pose's
//! joints in model space.
//!
//! # Why the geometry crosses the wire at all
//!
//! Because a blend space's weights are **sim state**. Since P29.2 they are
//! barycentric weights over `inf_anim::delaunay`'s triangulation, they choose the
//! pose, and the pose is folded into the trace the PIE-==-shipping gate compares.
//! A triangulator written a second time in TypeScript would be free to pick the
//! other diagonal of a co-circular square, and the panel would then draw a blend
//! the game does not play. So the canvas asks and draws; it does not derive.
//!
//! # Why there is no `blendspace_save`
//!
//! A 2D blend space lives inside a `.inf_sm` state, and `sm_save` is the one door
//! that writes one. The panel pushes its edit into the machine document
//! `smStore` holds and that document is saved through the door it always was —
//! rather than this module growing a second way to author the same bytes, which
//! is the shape of defect the P23.6 "one door" ruling exists to prevent.

use serde::{Deserialize, Serialize};
use tauri::State;
use uuid::Uuid;

use inf_anim::{
    global_transforms, sample_blend_space_2d, triangulate, weights_2d, AnimClip, AnimClipAsset,
    BlendEntry2D, BlendSpace2D, ClipRef, Pose, Skeleton, SkeletonAsset,
};
use inf_asset::{AssetId, AssetKind};

use super::assets::AssetState;

// ── DTOs (camelCase wire mirror) ─────────────────────────────────────────────

/// One placed sample: a clip pinned at a point of the parameter plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlendSampleDto {
    pub x: f64,
    pub y: f64,
    /// Asset GUID of the `.inf_anim` clip, or `None` while unassigned.
    pub clip: Option<String>,
}

/// One sample's share of the blend at the query point.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlendWeightDto {
    pub index: usize,
    pub weight: f64,
}

/// One joint of the previewed pose, model space, metres.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlendJointDto {
    pub name: String,
    pub parent: Option<u16>,
    pub position: [f32; 3],
}

/// Everything the canvas draws, in one round trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlendPreviewDto {
    pub triangles: Vec<[usize; 3]>,
    pub hull: Vec<[usize; 2]>,
    /// True when the samples have no area, so there is no triangulation and the
    /// engine interpolates along the row instead. Stated rather than inferred
    /// from an empty `triangles`, because "empty because degenerate" and "empty
    /// because there are no samples" are different things to say to an author.
    pub degenerate: bool,
    pub weights: Vec<BlendWeightDto>,
    pub blended_duration_s: f64,
    pub joints: Vec<BlendJointDto>,
    pub skeleton: Option<String>,
    /// Why the pose is empty, when it is. A **value**, not an error: a panel that
    /// cannot preview yet is an ordinary state of authoring, and an `Err` here
    /// would take the triangulation down with it.
    pub refusal: Option<String>,
}

// ── the command ──────────────────────────────────────────────────────────────

/// Triangulate the samples, weight them at `(x, y)`, and pose the blend at
/// `time_s`.
///
/// Writes nothing. A sample with no clip still takes part in the *geometry* —
/// an author places points before assigning clips, and the triangulation is what
/// tells them whether the placement is the one they meant.
#[tauri::command]
pub async fn blendspace_preview(
    samples: Vec<BlendSampleDto>,
    x: f64,
    y: f64,
    time_s: f64,
    assets: State<'_, AssetState>,
) -> Result<BlendPreviewDto, String> {
    let points: Vec<glam::DVec2> = samples
        .iter()
        .map(|s| glam::DVec2::new(s.x, s.y))
        .collect();
    let tri = triangulate(&points);
    let weights: Vec<BlendWeightDto> = weights_2d(&points, glam::DVec2::new(x, y))
        .into_iter()
        .map(|(index, weight)| BlendWeightDto { index, weight })
        .collect();
    let degenerate = tri.is_empty() && points.len() >= 3;

    // Resolve the clips the weighted samples actually name. A sample with no
    // clip, or one that no longer resolves, simply does not contribute — which is
    // the same rule `sample_blend_space_2d` applies to an unresolved `ClipRef`.
    let mut resolved: Vec<(usize, AnimClip, Option<[u8; 16]>)> = Vec::new();
    let mut load_error: Option<String> = None;
    for (i, s) in samples.iter().enumerate() {
        let Some(id) = s.clip.as_deref().and_then(|g| Uuid::parse_str(g).ok()) else {
            continue;
        };
        let loaded = assets.with_project(|p| {
            match p.db().get(AssetId(id)).map(|e| e.kind()) {
                Some(AssetKind::AnimClip) => {}
                _ => return Ok(None),
            }
            match p.load_payload::<AnimClipAsset>(AssetId(id)) {
                Ok(a) => Ok(Some((a.clip, a.skeleton))),
                Err(e) => Err(e.to_string()),
            }
        });
        match loaded {
            Ok(Some((clip, skel))) => resolved.push((i, clip, skel)),
            Ok(None) => {}
            Err(e) => {
                load_error.get_or_insert(e);
            }
        }
    }

    // The blended cycle length — the denominator the play-head is normalised
    // against, and what an `exitTime` is a fraction of. Derived here from the
    // same weights the pose uses, so the readout and the pose cannot disagree.
    let blended_duration_s: f64 = weights
        .iter()
        .filter_map(|w| {
            resolved
                .iter()
                .find(|(i, _, _)| *i == w.index)
                .map(|(_, c, _)| c.duration as f64 * w.weight)
        })
        .sum();

    // The rig: whichever skeleton the contributing clips name, with the first
    // one winning. A blend across two rigs is not a blend, and saying so beats
    // posing the first one and calling it a preview.
    let mut skeleton_id: Option<Uuid> = None;
    let mut mixed_rigs = false;
    for (_, _, skel) in &resolved {
        let Some(bytes) = skel else { continue };
        let id = Uuid::from_bytes(*bytes);
        match skeleton_id {
            None => skeleton_id = Some(id),
            Some(prev) if prev != id => mixed_rigs = true,
            Some(_) => {}
        }
    }

    let mut refusal = load_error;
    let mut joints = Vec::new();
    if mixed_rigs {
        refusal.get_or_insert_with(|| {
            "these clips were authored against different skeletons — a blend across two rigs is not a blend".to_string()
        });
    } else if let Some(id) = skeleton_id {
        match assets
            .with_project(|p| p.load_payload::<SkeletonAsset>(AssetId(id)).map_err(|e| e.to_string()))
        {
            Ok(asset) if !asset.skeleton.is_empty() => {
                joints = pose_joints(&asset.skeleton, &samples, &resolved, x, y, time_s);
            }
            Ok(_) => {
                refusal.get_or_insert_with(|| "the skeleton these clips name is empty".to_string());
            }
            Err(e) => {
                refusal.get_or_insert(e);
            }
        }
    } else if resolved.is_empty() {
        refusal.get_or_insert_with(|| "assign a clip to a sample to preview the blend".to_string());
    } else {
        refusal.get_or_insert_with(|| {
            "these clips name no skeleton, so there is no rig to pose".to_string()
        });
    }

    Ok(BlendPreviewDto {
        triangles: tri.triangles,
        hull: tri.hull,
        degenerate,
        weights,
        blended_duration_s,
        joints,
        skeleton: skeleton_id.map(|u| u.to_string()),
        refusal,
    })
}

/// Sample the blend through the **engine's own door** and flatten the result into
/// model-space joint positions.
///
/// `sample_blend_space_2d` and not a hand-rolled weighting: it is what the fixed
/// step calls, so what the panel shows is what the character does — including the
/// phase sync and, since P29.2, the marker warping.
fn pose_joints(
    skeleton: &Skeleton,
    samples: &[BlendSampleDto],
    resolved: &[(usize, AnimClip, Option<[u8; 16]>)],
    x: f64,
    y: f64,
    time_s: f64,
) -> Vec<BlendJointDto> {
    // A synthetic `ClipRef` per sample index, so the space's entries and the
    // resolver agree without needing the real GUIDs (which a sample may not have).
    let key = |i: usize| -> ClipRef {
        let mut k = [0u8; 16];
        k[0..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
        k
    };
    let space = BlendSpace2D::new(
        "x",
        "y",
        samples
            .iter()
            .enumerate()
            .map(|(i, s)| BlendEntry2D {
                pos: [s.x, s.y],
                clip: key(i),
            })
            .collect(),
    );
    let lookup = |c: ClipRef| -> Option<&AnimClip> {
        resolved
            .iter()
            .find(|(i, _, _)| key(*i) == c)
            .map(|(_, clip, _)| clip)
    };
    let pose: Pose = sample_blend_space_2d(
        &space,
        skeleton,
        &lookup,
        glam::DVec2::new(x, y),
        time_s,
    );
    let globals = global_transforms(skeleton, &pose);
    skeleton
        .joints()
        .iter()
        .zip(globals)
        .map(|(j, g)| BlendJointDto {
            name: j.name.clone(),
            parent: j.parent,
            position: g.w_axis.truncate().to_array(),
        })
        .collect()
}
