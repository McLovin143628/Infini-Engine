//! **P28.2's invariant, as world state**: a resident cluster's tiles are
//! resident. Always. After every transaction of a churn.
//!
//! # What is being falsified
//!
//! The artifact this phase exists to make impossible is the one the direction
//! memo names in its first paragraph — *"high-poly geometry with a blurry
//! texture"*: three page systems that can disagree because nothing makes them
//! agree. The claim is not "the code calls both consumers"; it is that **there is
//! no reachable state** in which a meshlet page is in the pools and a tile its
//! materials sample at that page's detail level is not in the atlas.
//!
//! So this file drives the real machinery — `inf_vgeom::VgeomStreamer`,
//! `inf_vt::VtResidency`, and the `pair` door between them — through a seeded
//! churn of camera thresholds and budgets, and after **every** transaction
//! re-derives, from the geometry, what each resident page must hold, and asserts
//! the atlas holds it.
//!
//! # The oracle is independent, and that is the point
//!
//! A gate cannot see an error two subsystems share. The cook's pairing is a **uv
//! bound per page** turned into a tile rectangle; this file's oracle **samples
//! the actual triangles** — three corners and a centroid, per triangle, per
//! meshlet of the page — and places each sample with plain arithmetic, reading
//! the materialized mesh and the texture descriptor and never the container's
//! tiles section. The two derivations share no code:
//!
//! | | the cook's | the oracle's |
//! |---|---|---|
//! | footprint | axis-aligned bound over the page's referenced vertices | point samples of every triangle |
//! | tile placement | `pair_page_tiles` | `tile_of` below, written out |
//! | mip | `tile_mip_for_lod` | the memo's formula, re-derived here |
//! | source | the `.inf_vmesh` tiles section | `to_mesh()` + `VtTextureDesc` |
//!
//! Re-deriving the mip rule is deliberate: it is an **oracle**, not a unit pin,
//! so a change to the rule in one place has to be a change in both, and a
//! one-sided edit fails here. (The P28.1 audit's complaint about a re-implemented
//! `admit` was the opposite case — a *pin* that re-implemented its subject and
//! therefore could not see it move.)
//!
//! # No adapter
//!
//! Both halves are the GPU-free halves by construction, so this runs on every CI
//! leg and the churn can be exhaustive rather than sampled. What it does **not**
//! cover is the renderer's use of the mechanism — the order of the three steps
//! around the virtual-texture transaction — and that is pinned separately, by
//! `the_renderers_cluster_sync_seats_the_texture_half_between_the_plan_and_the_pair`
//! reading `renderer.rs`.

use std::collections::{BTreeMap, BTreeSet};

use inf_vgeom::{ClusterTexture, ClusterTextureSet, VgeomSource, VgeomStreamBudget, VgeomStreamer};
use inf_vt::{
    full_pyramid, TileCoord, VtPoolConfig, VtResidency, VtTextureDesc, VtTextureHandle, VtWant,
};

/// The asset id the churn streams (the streamer keys on a `u128`, not a GUID).
const ASSET: u128 = 0x2802_0000_0000_0001;

/// Texture GUIDs, distinct so a mixed-up pairing is visible rather than lucky.
const TEX_A: u128 = 0xA0A0_0000_0000_0000_0000_0000_0000_0001;
const TEX_B: u128 = 0xA0A0_0000_0000_0000_0000_0000_0000_0002;

fn asset_id(v: u128) -> inf_asset::AssetId {
    inf_asset::AssetId(uuid::Uuid::from_u128(v))
}

/// The two virtual textures the fixture's material samples: a 2 048² albedo and a
/// 1 024² second map, so the pairing has to get *two* different pyramids right
/// and a single-texture bug cannot pass by symmetry.
fn descs() -> Vec<(u128, VtTextureDesc)> {
    vec![
        (TEX_A, full_pyramid(2048, 2048, 128, 4, true)),
        (TEX_B, full_pyramid(1024, 1024, 128, 4, false)),
    ]
}

/// A paired `.inf_vmesh`, built from the shared displaced-grid fixture with its
/// analytic tangents on — so the asset under test is the shape a cook produces,
/// tangent channel and all, not a reduced stand-in for one.
fn paired_source() -> (VgeomSource, inf_vgeom::VgeomMesh) {
    let mesh = inf_vgeom::test_support::build_grid_tangented(
        24,
        0.3,
        inf_vgeom::test_support::GridNormals::Analytic,
        true,
    );
    let set = ClusterTextureSet {
        textures: descs()
            .iter()
            .map(|(g, d)| ClusterTexture::from_desc(asset_id(*g), d))
            .collect(),
    };
    let src = VgeomSource::from_mesh_paired(&mesh, &set).expect("build the paired image");
    (src, mesh)
}

/// A residency holding both textures, at `budget_bytes`.
fn residency(budget_bytes: u64) -> (VtResidency, BTreeMap<u128, VtTextureHandle>) {
    let (mut res, _adv) = VtResidency::new(VtPoolConfig {
        budget_bytes,
        ..Default::default()
    });
    let mut by_guid = BTreeMap::new();
    for (guid, desc) in descs() {
        let h = res.register_texture(desc).expect("the floor fits");
        by_guid.insert(guid, h);
    }
    (res, by_guid)
}

// ── the oracle ──────────────────────────────────────────────────────────────

/// Place a uv on a tile grid. Written out rather than called, so this file's
/// arithmetic is its own.
fn tile_of(u: f32, n: u32) -> u32 {
    let w = u - u.floor(); // the sampler's wrap, spelled here
    ((w * n as f32) as u32).min(n - 1)
}

/// **The mip rule, re-derived from the memo**: a LOD level halves a page's
/// triangles and a mip level quarters its texels, so one mip is worth two LOD
/// levels; the root page (`lod == u32::MAX`, spanning every level) takes the
/// coarsest level there is.
fn oracle_mip(lod: u32, mip_count: u32) -> u32 {
    let coarsest = mip_count - 1;
    if lod == u32::MAX {
        coarsest
    } else {
        (lod / 2).min(coarsest)
    }
}

/// Every tile a page's geometry actually samples, derived by walking its
/// triangles — never by reading the page's tiles section.
fn oracle_tiles(
    src: &VgeomSource,
    mesh: &inf_vgeom::VgeomMesh,
    page: usize,
) -> BTreeSet<(u128, TileCoord)> {
    let mut out = BTreeSet::new();
    let entry = src.pages()[page];
    let Some(globals) = src.with_page_sections(page, |s| {
        bytemuck::cast_slice::<u8, u32>(s.indices).to_vec()
    }) else {
        return out;
    };
    for g in globals {
        let ml = &mesh.meshlets[g as usize];
        for t in 0..ml.triangle_count as usize {
            let tri = mesh.triangle(g as usize, t);
            let uvs: Vec<[f32; 2]> = tri.iter().map(|&v| mesh.vertices[v as usize].uv).collect();
            let centroid = [
                (uvs[0][0] + uvs[1][0] + uvs[2][0]) / 3.0,
                (uvs[0][1] + uvs[1][1] + uvs[2][1]) / 3.0,
            ];
            for uv in uvs.iter().chain(std::iter::once(&centroid)) {
                for (guid, desc) in descs() {
                    let mip = oracle_mip(entry.lod, desc.mips.len() as u32);
                    let m = &desc.mips[mip as usize];
                    out.insert((
                        guid,
                        TileCoord::new(mip, tile_of(uv[0], m.tiles_x), tile_of(uv[1], m.tiles_y)),
                    ));
                }
            }
        }
    }
    out
}

// ── one step of the churn ───────────────────────────────────────────────────

/// The three-step page-in the renderer runs, with the virtual-texture half in the
/// middle. `couple` is the falsification switch: with it off the cluster tiles
/// never enter the want set, which is precisely the pre-P28.2 arrangement.
fn step(
    streamer: &mut VgeomStreamer,
    res: &mut VtResidency,
    by_guid: &BTreeMap<u128, VtTextureHandle>,
    src: &VgeomSource,
    threshold: f32,
    couple: bool,
) -> u32 {
    // 1. the geometry half.
    let plan = streamer.plan(&[inf_vgeom::VgeomWant {
        asset: ASSET,
        source: src,
        threshold,
    }]);

    // 2. the tiles every resident page samples, read off the container.
    let mut page_tiles: BTreeMap<usize, Vec<(u128, TileCoord)>> = BTreeMap::new();
    let resident_now = streamer.residency(ASSET).map_or(0, |r| r.resident_pages());
    for page in 0..resident_now {
        let refs = src
            .with_page_sections(page, |s| {
                s.tile_refs()
                    .iter()
                    .map(|t| (t.texture().uuid().as_u128(), t.coord()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        page_tiles.insert(page, refs);
    }

    // 3. ONE virtual-texture transaction, floor priority, in one want set.
    let mut wants: Vec<VtWant> = Vec::new();
    if couple {
        for refs in page_tiles.values() {
            for (guid, tile) in refs {
                if let Some(h) = by_guid.get(guid) {
                    wants.push(VtWant::new(*h, *tile));
                }
            }
        }
    }
    let _txn = res.apply_wants(&wants);

    // 4. pair, or hand the page back. Uncoupled, the seat is unconditional —
    // which is precisely the pre-P28.2 world: the geometry streamer admits what
    // its own budget allows and has no opinion about textures at all.
    let page_in = streamer.pair(plan, |_asset, page| {
        if !couple {
            return Some(Vec::new());
        }
        let refs = page_tiles.get(&page)?;
        let mut seated = Vec::with_capacity(refs.len());
        for (guid, tile) in refs {
            let h = by_guid.get(guid)?;
            if !res.is_resident(*h, *tile) {
                return None;
            }
            seated.push((*guid, *tile));
        }
        Some(seated)
    });
    // Every page handed back carries both halves, and the halves belong to each
    // other: the tiles are exactly the pairing of the page whose geometry it is.
    // The type makes the pair inseparable; this reads it back, so a `pair` that
    // handed out somebody else's tiles is visible here rather than nowhere.
    if couple {
        for p in page_in.pages() {
            let want = page_tiles
                .get(&p.geometry().page)
                .expect("a page the want pass never saw");
            assert_eq!(
                p.tiles(),
                want.as_slice(),
                "page {} carries tiles that are not its own",
                p.geometry().page
            );
        }
    }
    page_in.retracted
}

/// The invariant, checked against the WORLD: for every page the streamer says is
/// resident, every tile the geometry samples is in the atlas.
fn assert_invariant(
    streamer: &VgeomStreamer,
    res: &VtResidency,
    by_guid: &BTreeMap<u128, VtTextureHandle>,
    src: &VgeomSource,
    mesh: &inf_vgeom::VgeomMesh,
    at: &str,
) -> usize {
    let mut checked = 0usize;
    let resident = streamer.residency(ASSET).map_or(0, |r| r.resident_pages());
    for page in 0..resident {
        for (guid, tile) in oracle_tiles(src, mesh, page) {
            let h = by_guid[&guid];
            assert!(
                res.is_resident(h, tile),
                "{at}: page {page} is resident and the tile it samples is not — \
                 texture {guid:#x}, mip {} ({}, {}). This is the \"high-poly mesh, \
                 blurry texture\" state, and it is supposed to be unreachable.",
                tile.mip,
                tile.x,
                tile.y
            );
            checked += 1;
        }
    }
    checked
}

// ── the gate ────────────────────────────────────────────────────────────────

/// **THE INVARIANT.** A churn of camera distances over a comfortable texture
/// budget: residency grows, shrinks and grows again, and after every transaction
/// every resident page's tiles are in the atlas.
#[test]
fn a_resident_clusters_tiles_are_resident_after_every_transaction_of_a_churn() {
    let (src, mesh) = paired_source();
    let (mut res, by_guid) = residency(64 * 1024 * 1024);
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());

    // Coarse → fine → coarse → fine again, so pages are admitted, evicted and
    // re-admitted rather than only ever added.
    let thresholds: Vec<f32> = vec![
        4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.02, 0.005, 0.02, 0.25, 2.0, 8.0, 0.5, 0.001,
    ];
    let mut peak = 0usize;
    let mut checks = 0usize;
    for (i, t) in thresholds.iter().enumerate() {
        let retracted = step(&mut streamer, &mut res, &by_guid, &src, *t, true);
        assert_eq!(
            retracted, 0,
            "step {i}: a comfortable budget retracted a page"
        );
        checks += assert_invariant(
            &streamer,
            &res,
            &by_guid,
            &src,
            &mesh,
            &format!("step {i} (threshold {t})"),
        );
        peak = peak.max(streamer.residency(ASSET).map_or(0, |r| r.resident_pages()));
    }

    // Anti-vacuity, in both directions: the churn really did stream, and the
    // oracle really did have tiles to check.
    assert!(
        peak >= 3,
        "the churn never got past {peak} resident pages — it is not exercising residency"
    );
    assert_eq!(
        streamer.residency(ASSET).map(|r| r.resident_pages()),
        Some(src.pages().len()),
        "the finest threshold must reach full residency"
    );
    assert!(
        checks > 1000,
        "the oracle only checked {checks} (tile, page) pairs"
    );
}

/// **The falsification.** The identical churn with the cluster tiles removed from
/// the want set — which is exactly the pre-P28.2 arrangement, two systems with no
/// edge between them — must reach the forbidden state.
///
/// Without this arm the one above is satisfied by a fixture whose analytic floor
/// happens to cover every tile the geometry samples, and it would say nothing at
/// all about the coupling.
#[test]
fn without_the_coupling_a_resident_cluster_loses_its_tiles() {
    let (src, mesh) = paired_source();
    let (mut res, by_guid) = residency(64 * 1024 * 1024);
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());

    // No cluster wants at all: the residency holds only its mandatory floor (the
    // coarsest level of each texture, pinned at registration).
    for t in [4.0f32, 0.5, 0.02, 0.001] {
        step(&mut streamer, &mut res, &by_guid, &src, t, false);
    }
    let resident = streamer.residency(ASSET).map_or(0, |r| r.resident_pages());
    assert!(resident > 0, "nothing streamed — the control proves nothing");

    let mut missing = 0usize;
    let mut total = 0usize;
    for page in 0..resident {
        for (guid, tile) in oracle_tiles(&src, &mesh, page) {
            total += 1;
            if !res.is_resident(by_guid[&guid], tile) {
                missing += 1;
            }
        }
    }
    assert!(
        missing > 0,
        "uncoupled, all {total} sampled tiles were resident anyway — the invariant \
         arm is measuring a fixture, not a mechanism"
    );
    // And it is not a rounding-scale miss: the finest pages sample tiles the
    // coarsest-mip floor cannot possibly serve.
    assert!(
        missing * 2 > total,
        "only {missing} of {total} sampled tiles were missing without the coupling"
    );
}

/// **A texture budget that cannot hold the pairing retracts the GEOMETRY.**
///
/// This is the coupling's other direction, and the one that makes the invariant
/// survivable rather than merely true: when the atlas cannot seat a page's tiles,
/// the page is handed back — softer geometry AND softer texture, together — and
/// the invariant still holds over the reduced residency.
#[test]
fn a_texture_budget_that_cannot_seat_the_tiles_hands_the_geometry_back() {
    let (src, mesh) = paired_source();
    // Tight enough that the finest pages' tile sets do not fit, roomy enough that
    // the two mandatory floors do (`register_texture` refuses otherwise).
    let (mut res, by_guid) = residency(1024 * 1024);
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());

    let mut retracted_total = 0u32;
    for t in [4.0f32, 0.5, 0.05, 0.001, 0.001] {
        retracted_total += step(&mut streamer, &mut res, &by_guid, &src, t, true);
        assert_invariant(&streamer, &res, &by_guid, &src, &mesh, "tight budget");
    }
    assert!(
        retracted_total > 0,
        "a 1 MiB atlas seated every cluster page's tiles — the fixture is not \
         tight enough for this arm to be about anything"
    );
    // The geometry really did stop short, and page 0 — the floor that makes
    // "never a hole" true — is still there.
    let resident = streamer.residency(ASSET).map_or(0, |r| r.resident_pages());
    assert!(
        resident >= 1 && resident < src.pages().len(),
        "resident {resident} of {} — a retraction must cost detail, not the asset",
        src.pages().len()
    );
}

/// The renderer runs the three steps in the one order that makes the invariant
/// hold between transactions, and this reads its source to say so.
///
/// The churn above proves the *mechanism*; it cannot see the renderer, which is
/// where the three steps are actually sequenced. The order is load-bearing in a
/// way a comment cannot enforce: seat the texture half in a **second**
/// transaction and a resident page's tiles stop being protected on the very next
/// frame, because `apply_wants` protects what its want set names.
#[test]
fn the_renderers_cluster_sync_seats_the_texture_half_between_the_plan_and_the_pair() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/renderer.rs"),
    )
    .expect("read renderer.rs")
    .replace("\r\n", "\n");
    let at = |needle: &str| {
        src.find(needle)
            .unwrap_or_else(|| panic!("renderer.rs no longer contains `{needle}`"))
    };
    let plan = at("plan_cluster_pages(scene, view, &vsettings)");
    let wants = at("n.cluster_tile_wants(scene,");
    let vt = at("self.vt_stream(gpu, scene, view, &mut encoder, &cluster_wants)");
    let pair = at("n.commit_cluster_pages(gpu,");
    assert!(
        plan < wants && wants < vt && vt < pair,
        "the cluster page-in is out of order: plan {plan}, wants {wants}, \
         vt {vt}, pair {pair}"
    );
    // And the wants really are folded into the ONE transaction rather than
    // applied beside it.
    let fold = at("wants.push(inf_vt::VtWant::new(h, *tile))");
    let apply = at("lib.sync(&gpu.device, &gpu.queue, pools, &wants)");
    assert!(
        fold < apply,
        "the cluster wants are added after the transaction that would protect them"
    );
}
