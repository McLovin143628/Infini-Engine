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

/// How one step of the churn is wired — the falsification switches, named.
#[derive(Clone, Copy)]
struct Churn {
    /// Fold the cluster tiles into the want set and demand them at `pair`. With
    /// it **off** the cluster tiles never enter the want set, which is precisely
    /// the pre-P28.2 arrangement: two systems with no edge between them.
    couple: bool,
    /// Drop a cooked address the **registered image does not have** instead of
    /// asking for it and refusing the page when it does not arrive (P28.2 audit;
    /// `VtResidency::can_address` is the door the renderer uses). With it off,
    /// a pairing that meets another image of its texture retracts the asset on
    /// every frame for ever.
    filter_stale: bool,
}

impl Churn {
    /// The shipped arrangement.
    const COUPLED: Self = Self {
        couple: true,
        filter_stale: true,
    };
    /// Pre-P28.2: geometry streams, textures stream, nothing joins them.
    const UNCOUPLED: Self = Self {
        couple: false,
        filter_stale: false,
    };
    /// P28.2 as first landed: coupled, but unable to tell "not paged in" from
    /// "no such tile".
    const STALE_BLIND: Self = Self {
        couple: true,
        filter_stale: false,
    };
}

/// The three-step page-in the renderer runs, with the virtual-texture half in the
/// middle.
///
/// `competing` is a second want class at **refinement** priority — what the
/// feedback mask contributes in the real frame. It exists so the protection
/// order inside `apply_wants` (touch every resident want, THEN admit misses) is
/// under load rather than merely present.
fn step(
    streamer: &mut VgeomStreamer,
    res: &mut VtResidency,
    by_guid: &BTreeMap<u128, VtTextureHandle>,
    src: &VgeomSource,
    threshold: f32,
    churn: Churn,
    competing: &[VtWant],
) -> u32 {
    let couple = churn.couple;
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
                    .filter(|(g, t)| {
                        // The renderer's own filter, through the same door: a
                        // texture this level did not register, or an address the
                        // registered image does not have, is not part of the
                        // pairing.
                        !churn.filter_stale
                            || by_guid.get(g).is_some_and(|h| res.can_address(*h, *t))
                    })
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
    wants.extend_from_slice(competing);
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
        let retracted = step(
            &mut streamer,
            &mut res,
            &by_guid,
            &src,
            *t,
            Churn::COUPLED,
            &[],
        );
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
        step(
            &mut streamer,
            &mut res,
            &by_guid,
            &src,
            t,
            Churn::UNCOUPLED,
            &[],
        );
    }
    let resident = streamer.residency(ASSET).map_or(0, |r| r.resident_pages());
    assert!(
        resident > 0,
        "nothing streamed — the control proves nothing"
    );

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
        retracted_total += step(
            &mut streamer,
            &mut res,
            &by_guid,
            &src,
            t,
            Churn::COUPLED,
            &[],
        );
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

// ── the P28.2 audit's arms ──────────────────────────────────────────────────

/// A paired image cooked against `cook`, to be met at runtime by whatever the
/// caller registers — the constructed mismatch the staleness arm needs.
fn source_paired_against(cook: &[(u128, VtTextureDesc)]) -> VgeomSource {
    let mesh = inf_vgeom::test_support::build_grid_tangented(
        24,
        0.3,
        inf_vgeom::test_support::GridNormals::Analytic,
        true,
    );
    let set = ClusterTextureSet {
        textures: cook
            .iter()
            .map(|(g, d)| ClusterTexture::from_desc(asset_id(*g), d))
            .collect(),
    };
    VgeomSource::from_mesh_paired(&mesh, &set).expect("build the paired image")
}

/// **THE STALENESS ARM** (P28.2 audit). A `.inf_vmesh`'s tiles section holds
/// `(guid, mip, x, y)` **addresses** into another asset's address space, and
/// nothing in either container ties the two — no version, no digest, no extent.
/// So the question is not whether a pairing can meet another image of its
/// texture; it is what happens when it does.
///
/// The answer must not be "the asset disappears". `is_resident` reports *no such
/// tile* exactly as it reports *not paged in*, and the first address a shrunken
/// pyramid loses is the **root page's** coarsest mip — the page that makes
/// "never a hole" true. Refuse it and residency goes to zero, on every frame,
/// for ever, silently: no budget will ever seat a tile that does not exist.
///
/// The control is the arrangement as this batch first landed it, and it must
/// reach that state, or this arm is measuring a fixture rather than the door.
#[test]
fn a_pairing_cooked_against_another_image_degrades_instead_of_erasing_the_asset() {
    // Cooked against 2 048² / 1 024²; the session registers half-size images of
    // the same GUIDs — two mips fewer, so the cooked coarse addresses do not
    // exist at all and the surviving ones name a different level.
    let src = source_paired_against(&descs());
    let smaller: Vec<(u128, VtTextureDesc)> = vec![
        (TEX_A, full_pyramid(512, 512, 128, 4, true)),
        (TEX_B, full_pyramid(256, 256, 128, 4, false)),
    ];
    let register = || {
        let (mut res, _adv) = VtResidency::new(VtPoolConfig {
            budget_bytes: 64 * 1024 * 1024,
            ..Default::default()
        });
        let mut by_guid = BTreeMap::new();
        for (guid, desc) in &smaller {
            let h = res.register_texture(desc.clone()).expect("the floor fits");
            by_guid.insert(*guid, h);
        }
        (res, by_guid)
    };
    let thresholds = [4.0f32, 0.5, 0.02, 0.001, 0.001];

    // THE CONTROL: blind to the difference between "not seated" and "not there".
    let (mut res, by_guid) = register();
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());
    for t in thresholds {
        step(
            &mut streamer,
            &mut res,
            &by_guid,
            &src,
            t,
            Churn::STALE_BLIND,
            &[],
        );
    }
    assert_eq!(
        streamer.residency(ASSET).map_or(0, |r| r.resident_pages()),
        0,
        "the control must reach the forbidden state: a pairing asking for tiles \
         no budget can seat has to erase the asset — page 0 included, because the \
         root page's address is the first one a shrunken pyramid loses"
    );

    // THE DOOR: `can_address` separates the two answers, so a stale slot is
    // uncoupled — exactly as a texture this level does not bind is — and the
    // geometry streams.
    let (mut res, by_guid) = register();
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());
    let mut retracted = 0u32;
    for t in thresholds {
        retracted += step(
            &mut streamer,
            &mut res,
            &by_guid,
            &src,
            t,
            Churn::COUPLED,
            &[],
        );
    }
    assert_eq!(retracted, 0, "a stale address must not retract a page");
    assert_eq!(
        streamer.residency(ASSET).map(|r| r.resident_pages()),
        Some(src.pages().len()),
        "with the stale addresses dropped, the asset streams to full residency"
    );

    // Anti-vacuity, both ways: the mismatch is real (addresses were dropped) and
    // it is not total (the finer levels still exist in the smaller image, so the
    // coupling still has something left to protect).
    let mut dropped = 0usize;
    let mut kept = 0usize;
    for page in 0..src.pages().len() {
        src.with_page_sections(page, |s| {
            for t in s.tile_refs() {
                let h = by_guid[&t.texture().uuid().as_u128()];
                if res.can_address(h, t.coord()) {
                    kept += 1;
                } else {
                    dropped += 1;
                }
            }
        });
    }
    assert!(
        dropped > 0,
        "the two images address the same tiles — nothing was stale, so the arm \
         proves nothing"
    );
    assert!(
        kept > 0,
        "every address was stale — the coupling is inert here"
    );
}

/// **THE PROTECTION ORDER, UNDER LOAD** (P28.2 audit). `VtResidency::apply_wants`
/// touches every want that is already resident *before* it offers a slot to any
/// miss, and the memo calls that ordering the mechanism by which the invariant
/// survives **between** transactions rather than only at one.
///
/// The batch's own churn cannot see it. With one want class and a comfortable
/// budget nothing ever competes for a slot, so an `apply_wants` that protected
/// nothing at all passes every other arm in this file. This one puts a second
/// want class at **refinement** priority — a texture the mesh never samples,
/// disjoint by construction, because a competing class drawn from the pairing's
/// own tiles is deduped into the pairing and competes with nothing — against a
/// pool that cannot hold both.
///
/// **What it establishes, and the bound it measures.** The protection is
/// priority-**blind**: step 3 protects every want that is already resident, of
/// any class, before step 4 admits any miss, of any class. So a refinement that
/// got there first outranks a floor want that has not. The guarantee that
/// survives is the one the invariant needs — *a resident cluster page never
/// loses ground it already holds* — and the cost is that a competing class can
/// stop the pairing gaining more. Measured here rather than asserted away: on
/// this fixture the decoy costs the pairing its finest page.
#[test]
fn a_refinement_class_under_slot_pressure_cannot_cost_a_resident_page_its_tiles() {
    let (src, mesh) = paired_source();
    // A third texture, registered but NOT part of the pairing — a surface the
    // feedback mask asks for and this mesh never samples.
    const TEX_C: u128 = 0xA0A0_0000_0000_0000_0000_0000_0000_0003;
    let residency_with_decoy = || {
        let (mut res, _adv) = VtResidency::new(VtPoolConfig {
            budget_bytes: 4 * 1024 * 1024,
            ..Default::default()
        });
        let mut by_guid = BTreeMap::new();
        for (guid, desc) in descs() {
            by_guid.insert(guid, res.register_texture(desc).expect("the floor fits"));
        }
        let decoy = full_pyramid(2048, 2048, 128, 4, true);
        let h = res.register_texture(decoy.clone()).expect("the floor fits");
        let mut competing = Vec::new();
        let m = &decoy.mips[0];
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                competing.push(VtWant::refine(h, TileCoord::new(0, x, y)));
            }
        }
        by_guid.insert(TEX_C, h);
        (res, by_guid, competing)
    };
    // Strictly refining, so "residency went backwards" can only mean the texture
    // half took ground back — never that the camera asked for less.
    let thresholds: Vec<f32> = vec![4.0, 0.5, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

    let ladder = |competing: bool| -> (Vec<usize>, bool) {
        let (mut res, by_guid, decoy) = residency_with_decoy();
        let extra = if competing { decoy } else { Vec::new() };
        let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());
        let mut pages = Vec::new();
        let mut contended = false;
        for t in &thresholds {
            step(
                &mut streamer,
                &mut res,
                &by_guid,
                &src,
                *t,
                Churn::COUPLED,
                &extra,
            );
            contended |= res.stats().deferred > 0;
            assert_invariant(
                &streamer,
                &res,
                &by_guid,
                &src,
                &mesh,
                if competing {
                    "with a competing refinement class"
                } else {
                    "alone"
                },
            );
            pages.push(streamer.residency(ASSET).map_or(0, |r| r.resident_pages()));
        }
        (pages, contended)
    };

    let (alone, _) = ladder(false);
    let (contested, contended) = ladder(true);
    assert!(
        contended,
        "the competing class never exhausted a slot — this arm is not under load,          so it cannot see the protection it exists to check"
    );

    // THE GUARANTEE: under a want that only ever refines, a resident cluster page
    // is never handed back to seat somebody else's tile.
    for w in contested.windows(2) {
        assert!(
            w[1] >= w[0],
            "residency went backwards under a refining want: {contested:?}. A              competing class took a resident page's tiles, which is what the              protection order forbids."
        );
    }
    // THE BOUND, measured: it may still cost detail not yet gained.
    for (c, a) in contested.iter().zip(&alone) {
        assert!(
            c <= a,
            "a competing class made the pairing stream MORE: {contested:?} against              {alone:?} — the arm's premise is inverted"
        );
    }
    assert!(
        contested.last() < alone.last(),
        "the decoy cost the pairing nothing at all ({contested:?} against          {alone:?}) — then the pool is not contested enough for the bound above          to be the honest reading of this fixture"
    );
}
