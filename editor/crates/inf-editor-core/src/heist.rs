//! **Harbour Heist** — the InfiniScript arc's dogfood mission (wave SCRIPT3).
//!
//! The arc built a language, a cook, a hot-reload path, an editor and 132 verbs.
//! This is the thing that says whether any of it is usable: **a whole mission,
//! authored as one `.infini` file, with no Rust behind it.** The objectives, the
//! timer, the bolt on the door, the loot, the stakes and the outcome are member
//! variables and verbs; the level under it is a plaza, a vault and a hero, and
//! everything else the mission needs it makes for itself on `BeginPlay`.
//!
//! # What is generated here and what is authored
//!
//! The level, the vault's grammar graph and the README are **generated** —
//! `write_heist` writes them and `committed_sample_matches_generators` locks
//! them, like every other sample in this crate.
//!
//! `samples/harbour-heist/HarbourHeist.infini` is **not**. It is the artifact
//! the wave exists to produce: git-diffable text a designer edits, with the hot
//! reload path pointed at it. Generating it from a Rust `const` would make the
//! showcase a lie in the one place it is supposed to be true. What the generator
//! does write beside it is its **sidecar**, because a `.infini` with no sidecar
//! takes a GUID synthesised from its content hash and its own next save renames
//! it (the SCRIPT1b audit's HIGH), and the level binds it by GUID.
//!
//! The one thing that cannot be locked by a byte comparison is that the script's
//! coordinates and the level's agree. That is asserted by **behaviour** instead,
//! in `harbour_heist_gate.rs`: the hero starts outside the vault, walks in, and
//! the mission notices. A string comparison could not have said that anyway.
//!
//! # Small on purpose, and where it is set
//!
//! The fiction is Harbour City, on the island. The level is **not** the island:
//! `samples/island` is a level whose terrain is 550 MB and is built rather than
//! committed, and a mission gate that loaded it would spend its whole budget
//! travelling — the `phase30-gameplay` ruling, for the `phase30-gameplay`
//! reason. What is on the island and what is here are the same *vocabulary*:
//! the grammar-built building, the doors it hangs, the item catalogue, the
//! society that populates a volume with residents.

use std::path::PathBuf;

use uuid::Uuid;

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

/// The mission level.
pub const HEIST_LEVEL_GUID: Uuid = Uuid::from_u128(0x5C13_0000);
/// `HarbourHeist.infini` — the mission itself.
pub const HEIST_SCRIPT_GUID: Uuid = Uuid::from_u128(0x5C13_0001);
/// The vault building's grammar graph.
pub const HEIST_PCG_GUID: Uuid = Uuid::from_u128(0x5C13_0002);
/// The sun.
pub const HEIST_SUN_GUID: Uuid = Uuid::from_u128(0x5C13_0003);
/// The quayside slab everything stands on.
pub const HEIST_GROUND_GUID: Uuid = Uuid::from_u128(0x5C13_0004);
/// The volume the vault building is grown in.
pub const HEIST_VAULT_GUID: Uuid = Uuid::from_u128(0x5C13_0005);
/// The volume the apartment block across the plaza is grown in.
pub const HEIST_HOUSING_GUID: Uuid = Uuid::from_u128(0x5C13_0008);
/// The apartment block's grammar graph.
pub const HEIST_HOUSING_PCG_GUID: Uuid = Uuid::from_u128(0x5C13_0009);
/// The hero — **the actor the mission is bound to**, so the script's `entity` is
/// the player.
pub const HEIST_HERO_GUID: Uuid = Uuid::from_u128(0x5C13_0006);
/// `Alarm.inf_mesh` — **the asset the mission NAMES**, and the reason this
/// sample has a mesh in it at all.
///
/// `engine.spawn("Alarm")` is the node kit's only `StrRole::Asset` port, so the
/// cook resolves that string against the project and pulls what it finds into
/// the pack's dependency closure — and a name that resolves to nothing is a
/// **blocking** advisory, which is the shape SK1c stopped a wave over. So the
/// mission naming a prefab is not decoration: it is the one live consumer of the
/// asset-reference walk, and this asset is what the walk finds.
///
/// The bound, stated where a reader meets it: a `PackEntry` carries no name, so
/// the *runtime* cannot resolve `"Alarm"` back to this GUID and the spawned
/// entity is a placeholder cube carrying the name. Spelling the prefab as this
/// GUID instead would bind it (`inf_ecs::prefab::spawn_prefab`), at the cost of
/// a script a designer cannot read.
pub const HEIST_ALARM_MESH_GUID: Uuid = Uuid::from_u128(0x5C13_0007);

/// The vault building's footprint, metres — the settlement library's own Office
/// lot (30 x 34, rounded to the plaza's depth), because that is the size at
/// which `building.plan` cuts a floor into rooms an office WORKS in. A 14 x 10
/// office is a lobby with a corridor, and `occupancy` gives a lobby nobody.
pub const HEIST_VAULT_M: (f64, f64) = (30.0, 24.0);
/// The apartment block across the plaza, metres — the settlement library's
/// Apartment lot.
///
/// **It is here for the crowd, and the crowd is a mission mechanic.**
/// `inf_ecs::society` pairs a HOME with a WORK to make an agent, so a level with
/// forty desks and nowhere to live has forty slots and no people in it —
/// measured, before this block existed. The bank's staff live across the square.
pub const HEIST_HOUSING_M: (f64, f64) = (26.0, 30.0);
/// Where the apartment block stands, world metres.
pub const HEIST_HOUSING_AT: (f64, f64, f64) = (40.0, 0.0, 0.0);
/// Where the hero starts, world metres: on the plaza, outside the vault and
/// outside the boat zone, so the mission's first phase has somewhere to be.
pub const HEIST_HERO_START: (f64, f64, f64) = (0.0, 0.0, -20.0);
/// The hero's capsule half-height, metres.
pub const HEIST_HERO_HALF_H: f64 = 0.9;
/// The hero's capsule radius, metres.
pub const HEIST_HERO_RADIUS: f64 = 0.3;
/// The centre of the vault floor — inside `in_the_vault()`'s box.
pub const HEIST_VAULT_AT: (f64, f64, f64) = (0.0, 0.0, 0.0);
/// The quay the boat is tied to — inside `at_the_boat()`'s box.
pub const HEIST_BOAT_AT: (f64, f64, f64) = (0.0, 0.0, -30.0);
/// The vault door's hinge, world metres — the point the script's three `door.*`
/// calls name.
///
/// **A place, not a number the mission owns.** Every quantity the *mission*
/// decides — how long the vault clock runs, what being watched costs, how much
/// the hero's body is worth — is spelled in the `.infini` and nowhere else. A
/// Rust constant beside it would be a second opinion about a number only one of
/// them can change, and the one that can is the one a designer edits.
pub const HEIST_VAULT_DOOR_AT: (f64, f64, f64) = (0.0, 1.05, -12.0);

/// The repo-root `samples/harbour-heist/` directory.
pub fn heist_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../samples/harbour-heist")
}

/// The committed mission source, read off disk.
///
/// Through `std::fs` rather than `include_str!` deliberately: `include_str!`
/// would bake the mission into the binary at compile time, and the whole point
/// of clause 4 is that a designer edits this file and the running session picks
/// it up. A test that read a baked copy would measure the wrong file.
pub fn heist_script() -> Result<Vec<u8>, String> {
    let path = heist_dir().join("HarbourHeist.infini");
    std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))
}

/// The mission's scene: a sun, a quayside slab, the vault, and the hero the
/// mission is bound to.
///
/// **The hero has no `CharacterMovement`**, and that is deliberate rather than
/// an omission. This is a mission gate, not a locomotion gate: the trace it
/// produces has to be a function of the mission's own decisions, and a
/// player-controlled mover would fold gravity, ground-snap and a camera into
/// every step of it. The gate moves the hero the way a player would end up
/// moving them and the script reacts — which is the division of labour the
/// mission itself has.
pub fn heist_scene() -> SceneDoc {
    use inf_ecs::components::{
        ActorClass, BodyKind3D, Collider3D, ColliderShape3DKind, Light, LightKind, PcgVolume,
        RigidBody3D, StreamingSource, Transform,
    };
    use inf_ecs::math::{Color, Vec2d, Vec3d};

    macro_rules! insert {
        ($doc:expr, $guid:expr, $component:expr) => {{
            let e = $doc.entity_of($guid).expect("entity");
            $doc.world_mut()
                .world_mut()
                .entity_mut(e)
                .insert($component);
        }};
    }

    let mut doc = SceneDoc::new();
    doc.set_title("Harbour Heist");

    doc.create_with_guid(HEIST_SUN_GUID, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        HEIST_SUN_GUID,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-52.0, -34.0, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        HEIST_SUN_GUID,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.0,
            ..Default::default()
        }
    );

    // The quay: one static slab, because the level carries no terrain and a
    // mission that begins with the hero falling out of the world is a short one.
    doc.create_with_guid(HEIST_GROUND_GUID, SpawnKind::Empty, "Quay", None);
    insert!(
        doc,
        HEIST_GROUND_GUID,
        Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        HEIST_GROUND_GUID,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        HEIST_GROUND_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(40.0, 0.5, 40.0),
            ..Default::default()
        }
    );

    // The vault, grown by the grammar — the same graph `phase30-gameplay`'s
    // house is grown from, because the mission is about what a SCRIPT can do
    // with a building and not about the building.
    doc.create_with_guid(HEIST_VAULT_GUID, SpawnKind::Empty, "Vault", None);
    insert!(doc, HEIST_VAULT_GUID, Transform::IDENTITY);
    insert!(
        doc,
        HEIST_VAULT_GUID,
        PcgVolume {
            graph: Some(HEIST_PCG_GUID),
            extent: Vec2d::new(HEIST_VAULT_M.0 * 0.5, HEIST_VAULT_M.1 * 0.5),
            seed: 1,
            ..Default::default()
        }
    );

    // The apartment block across the plaza — the crowd's other half, and the
    // reason `crowd.population()` answers a number at all (see
    // `heist_vault_graph`'s table).
    doc.create_with_guid(HEIST_HOUSING_GUID, SpawnKind::Empty, "Housing", None);
    insert!(
        doc,
        HEIST_HOUSING_GUID,
        Transform {
            translation: Vec3d::new(HEIST_HOUSING_AT.0, HEIST_HOUSING_AT.1, HEIST_HOUSING_AT.2),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        HEIST_HOUSING_GUID,
        PcgVolume {
            graph: Some(HEIST_HOUSING_PCG_GUID),
            extent: Vec2d::new(HEIST_HOUSING_M.0 * 0.5, HEIST_HOUSING_M.1 * 0.5),
            seed: 2,
            ..Default::default()
        }
    );

    doc.create_with_guid(HEIST_HERO_GUID, SpawnKind::Empty, "Hero", None);
    insert!(
        doc,
        HEIST_HERO_GUID,
        Transform {
            translation: Vec3d::new(
                HEIST_HERO_START.0,
                HEIST_HERO_START.1 + HEIST_HERO_HALF_H + HEIST_HERO_RADIUS,
                HEIST_HERO_START.2
            ),
            rotation: Vec3d::ZERO,
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        HEIST_HERO_GUID,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        HEIST_HERO_GUID,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(HEIST_HERO_RADIUS, HEIST_HERO_HALF_H, HEIST_HERO_RADIUS),
            radius: HEIST_HERO_RADIUS,
            ..Default::default()
        }
    );
    // **The binding**: one component, and it names a `.infini` exactly as it
    // would name a `.inf_act`. That is the SCRIPT1b ruling made visible — a
    // script is content, and a level binds it the way it binds a Blueprint.
    insert!(doc, HEIST_HERO_GUID, ActorClass(HEIST_SCRIPT_GUID));
    insert!(doc, HEIST_HERO_GUID, StreamingSource { radius_m: 128.0 });

    doc.world_mut().propagate();
    doc
}

const HEIST_README: &str = concat!(
    "# samples/harbour-heist\n",
    "\n",
    "**Harbour Heist** - the InfiniScript arc's dogfood mission (wave SCRIPT3):\n",
    "a whole mission authored as one `.infini` file, with no Rust behind it.\n",
    "\n",
    "- `HarbourHeist.infini` - **the mission**. Objectives, a timer, a bolted\n",
    "  door, loot, stakes and an outcome, as member variables and verbs from the\n",
    "  shipped node kit. Hand-authored: this file is the artifact, not a\n",
    "  generated one.\n",
    "- `HarbourHeist.inf_lvl` - a quayside slab, a grammar-built bank, the\n",
    "  apartment block its staff live in, and one hero whose `ActorClass` names\n",
    "  the script above. Everything else the mission needs - the item catalogue,\n",
    "  the vault door, the bullion on the floor, the hero's own health - the\n",
    "  script makes on `BeginPlay`.\n",
    "- `HarbourVault.inf_pcg` / `HarbourHousing.inf_pcg` - the two buildings'\n",
    "  grammar graphs. The apartment block is not scenery: `inf_ecs::society`\n",
    "  pairs a HOME with a WORK to make an agent, so without it the bank has\n",
    "  forty desks, nobody in them, and a mission whose `crowd.population()`\n",
    "  branch could never run.\n",
    "- `Alarm.inf_mesh` - **the asset the mission NAMES**. `engine.spawn` takes\n",
    "  the node kit's only asset-naming string, so the cook resolves it and pulls\n",
    "  this into the pack's closure; a name that resolved to nothing would BLOCK\n",
    "  the build. A pack entry carries no name, so the runtime cannot resolve the\n",
    "  stem back - the spawned alarm is a placeholder cube carrying the name.\n",
    "  Spelling the prefab as this asset's GUID instead binds it.\n",
    "\n",
    "The gate over it is `runtime/inf-player/tests/harbour_heist_gate.rs`, which\n",
    "runs the whole mission twice on each of two routes - off a cooked\n",
    "`.ipack` the way a shipped build boots, and off the `ScenePayload` the\n",
    "editor really builds for PIE - and requires the traces byte-identical step\n",
    "for step. The two routes take the two exits from the vault and reach the\n",
    "two endings: loot the shelf and you are CLEAR; linger in the open, where\n",
    "the staff can see you and the clock runs double, and you are CAUGHT.\n",
    "\n",
    "The iteration claim the arc exists to make is measured on this file by\n",
    "`editor/crates/inf-editor-core/tests/script_iteration.rs`: edit the mission,\n",
    "and the running Simulate is running the new one.\n",
    "\n",
    "The level, the graph and this README are generated - do not hand-edit them.\n",
    "Regenerate with:\n",
    "\n",
    "```sh\n",
    "INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples\n",
    "```\n",
);

/// **The vault**: a two-storey `Office`, grown by the grammar.
///
/// An office and not the `phase30-gameplay` house, and the reason is the
/// mission's `witnesses` variable. `inf_pcg::building::society::occupancy` gives
/// a room a `Work` slot per so many square metres of Office, Workshop or Retail
/// and **nothing at all** to a Living room, a Kitchen or a Bath, so a house is a
/// building with nobody in it. Measured, in this order, which is worth recording
/// because each step looked like the answer:
///
/// | graph | residents | agents |
/// |---|---|---|
/// | the gameplay house, 14 x 10, one floor | 0 | 0 |
/// | an Office at 14 x 10, two floors | 0 | 0 |
/// | an Office at 30 x 24, two floors | 40 work | **0** |
/// | …with an Apartment block across the plaza | 40 work + homes | a crowd |
///
/// Two facts fall out of that table and both are the settlement library's, met
/// from the other side: an archetype has a **lot size** it plans properly at
/// (`settlement::zone_lots` gives Office 30 x 34), and `inf_ecs::society` makes
/// an agent by pairing a **home** with a work — so forty desks and nowhere to
/// live is forty slots and no people.
pub fn heist_vault_graph() -> inf_graph::Graph {
    building_graph(
        HEIST_VAULT_M,
        inf_pcg::ArchetypeId::Office,
        2,
        120.0,
        "vault",
    )
}

/// **The apartment block across the plaza** — where the bank's staff live, and
/// therefore the other half of the crowd. See [`heist_vault_graph`]'s table.
pub fn heist_housing_graph() -> inf_graph::Graph {
    building_graph(
        HEIST_HOUSING_M,
        inf_pcg::ArchetypeId::Apartment,
        3,
        100.0,
        "housing",
    )
}

/// One building on one lot: `grammar.footprint` -> `building.lots` ->
/// `building.plan`, with a `building.archetype` on the side.
///
/// The `phase30-gameplay` chain with the archetype and the lot rule lifted out,
/// because this sample grows two buildings and two copies of the same eighty
/// lines would be two places to fix a graph.
fn building_graph(
    size: (f64, f64),
    archetype: inf_pcg::ArchetypeId,
    floors: i64,
    min_area: f64,
    name: &str,
) -> inf_graph::Graph {
    let reg = inf_pcg::pcg_registry();
    let mut g = inf_graph::Graph::empty();
    use inf_graph::ParamValue as P;
    let add = |g: &mut inf_graph::Graph,
               n: u32,
               type_id: &str,
               params: &[(&str, inf_graph::ParamValue)]| {
        let node = inf_graph::NodeId(n);
        let mut m = inf_graph::ParamMap::new();
        for (k, v) in params {
            m.insert((*k).to_string(), v.clone());
        }
        inf_graph::apply_edits(
            g,
            &reg,
            &[inf_graph::GraphEdit::AddNode {
                id: node,
                type_id: type_id.into(),
                x: 0.0,
                y: 0.0,
                params: m,
            }],
        );
        node
    };
    let plot = add(
        &mut g,
        1,
        "grammar.footprint",
        &[("size_x", P::Float(size.0)), ("size_z", P::Float(size.1))],
    );
    // One lot filling the whole plot — the `phase30-gameplay` arrangement, and
    // for its reason: a frontage as wide as the footprint subdivides into
    // exactly one lot, which is what "one building" means here.
    let lots = add(
        &mut g,
        2,
        "building.lots",
        &[
            ("frontage", P::Float(size.0)),
            ("depth", P::Float(size.1)),
            ("jitter", P::Float(0.0)),
            ("setback", P::Float(0.0)),
            ("min_area", P::Float(min_area)),
        ],
    );
    let arch = add(
        &mut g,
        3,
        "building.archetype",
        &[
            ("archetype", P::Enum(archetype.name().into())),
            ("floors", P::Int(floors)),
            ("furnish", P::Bool(false)),
        ],
    );
    let plan = add(
        &mut g,
        4,
        "building.plan",
        &[
            ("name", P::Text(name.into())),
            ("seed", P::Int(11)),
            ("ground", P::Enum("Span".into())),
        ],
    );
    let out = add(&mut g, 5, "output.pcg", &[]);
    for (from, fp, to, tp) in [
        (plot, "out", lots, "block"),
        (lots, "out", plan, "lots"),
        (arch, "out", plan, "archetype"),
        (plan, "out", out, "scatter"),
    ] {
        inf_graph::apply_edits(
            &mut g,
            &reg,
            &[inf_graph::GraphEdit::Connect {
                link: inf_graph::Link {
                    from,
                    from_port: fp.into(),
                    to,
                    to_port: tp.into(),
                },
            }],
        );
    }
    g
}

/// **The asset the mission names**, and nothing more than that: a half-metre
/// box, six quads, 24 vertices.
///
/// Deliberately not `samples::phase22_box_mesh`, which tessellates at 16 × 16 per
/// face and would put **108 KB** in this folder for a prop nothing draws. What
/// this asset is *for* is being found by the cook's asset-reference walk, and a
/// closure edge does not care how many triangles are on the other end of it.
fn alarm_mesh() -> inf_mesh::MeshAsset {
    let h = 0.25f32;
    let top = 0.5f32;
    let mut vertices: Vec<inf_mesh::MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // (axis, sign) in a fixed order, so the asset is a pure function of nothing
    // at all and its bytes are the same on every host.
    for axis in 0..3usize {
        for sign in [-1.0f32, 1.0] {
            let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = vertices.len() as u32;
            let mut normal = [0.0f32; 3];
            normal[axis] = sign;
            let lo = |a: usize| if a == 1 { 0.0 } else { -h };
            let hi = |a: usize| if a == 1 { top } else { h };
            for (fu, fv) in [(0.0f32, 0.0f32), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let mut p = [0.0f32; 3];
                p[axis] = if sign < 0.0 { lo(axis) } else { hi(axis) };
                p[u_axis] = lo(u_axis) + fu * (hi(u_axis) - lo(u_axis));
                p[v_axis] = lo(v_axis) + fv * (hi(v_axis) - lo(v_axis));
                vertices.push(inf_mesh::MeshVertex {
                    position: p,
                    normal,
                    uv: [fu, fv],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
            }
            indices.extend_from_slice(&[base, base + 2, base + 1, base + 1, base + 2, base + 3]);
        }
    }
    inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: "alarm".into(),
            vertices,
            indices,
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec!["mat:alarm".into()],
    )
}

/// Write every committed mission file **except the mission**, which is authored
/// (see the module header).
pub fn write_heist() -> Result<(), String> {
    let dir = heist_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir: {e}"))?;

    crate::scene::serialize::save(
        &heist_scene(),
        &dir.join("HarbourHeist.inf_lvl"),
        Some(HEIST_LEVEL_GUID),
    )?;

    for (graph, guid, file) in [
        (heist_vault_graph(), HEIST_PCG_GUID, "HarbourVault.inf_pcg"),
        (
            heist_housing_graph(),
            HEIST_HOUSING_PCG_GUID,
            "HarbourHousing.inf_pcg",
        ),
    ] {
        let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
        if !lowered.ok {
            return Err(format!("{file} does not lower: {:?}", lowered.issues));
        }
        let pcg = inf_pcg::PcgAssetPayload::from_graph(&graph, lowered.document);
        let bytes = inf_asset::encode(&pcg).map_err(|e| format!("encode {file}: {e}"))?;
        let path = dir.join(file);
        std::fs::write(&path, &bytes).map_err(|e| format!("write {file}: {e}"))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(guid),
            inf_asset::AssetKind::Pcg,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .map_err(|e| format!("write {file} sidecar: {e}"))?;
    }

    // The asset the mission NAMES — see `HEIST_ALARM_MESH_GUID`.
    let alarm = inf_asset::encode(&alarm_mesh()).map_err(|e| format!("encode alarm: {e}"))?;
    let alarm_path = dir.join("Alarm.inf_mesh");
    std::fs::write(&alarm_path, &alarm).map_err(|e| format!("write alarm: {e}"))?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(HEIST_ALARM_MESH_GUID),
        inf_asset::AssetKind::Mesh,
        inf_asset::ContentHash::of(&alarm),
    )
    .save(&alarm_path)
    .map_err(|e| format!("write alarm sidecar: {e}"))?;

    // The mission's own sidecar. Written from the file on disk, so the hash it
    // records is the hash of what is committed — and NOT written by the
    // generator's idea of what the mission says, because the generator has none.
    let script = heist_script()?;
    let script_path = dir.join("HarbourHeist.infini");
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(HEIST_SCRIPT_GUID),
        inf_asset::AssetKind::Script,
        inf_asset::ContentHash::of(&script),
    )
    .save(&script_path)
    .map_err(|e| format!("write script sidecar: {e}"))?;

    std::fs::write(dir.join("README.md"), HEIST_README)
        .map_err(|e| format!("write readme: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The mission compiles, clean, through the one file door.**
    ///
    /// Not "the file exists": a committed mission that no longer parses is a
    /// sample nobody can open, and the door it goes through here
    /// (`compile_bytes`) is the same one the watcher, the cook and the PIE
    /// payload builder enter — so a mission that compiles here compiles in all
    /// three, by construction rather than by hope.
    #[test]
    fn the_committed_mission_compiles_clean() {
        let src = heist_script().expect("the mission is committed");
        let (class, warnings) =
            inf_script::compile_bytes(&src, "HarbourHeist.infini", "script:heist".to_string())
                .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
        assert!(
            warnings.is_empty(),
            "the committed mission should compile clean: {}",
            inf_script::render(&warnings)
        );
        // Anti-vacuity: an empty file compiles clean too. The mission is a
        // mission — two handlers and the six functions they are written out of.
        assert_eq!(class.events.len(), 2, "begin_play and tick");
        assert_eq!(class.functions.len(), 6);
    }

    /// **The mission's sidecar describes the mission**, so the level's
    /// `ActorClass` resolves and a save does not rename the script under it (the
    /// SCRIPT1b audit's HIGH).
    #[test]
    fn the_missions_sidecar_pins_its_identity_to_its_bytes() {
        let path = heist_dir().join("HarbourHeist.infini");
        let side = inf_asset::AssetSidecar::load(&path).expect("the sidecar is committed");
        assert_eq!(side.guid.0, HEIST_SCRIPT_GUID);
        assert_eq!(side.kind, inf_asset::AssetKind::Script);
        assert_eq!(
            side.content_hash,
            inf_asset::ContentHash::of(&heist_script().expect("the mission is committed")),
            "the committed sidecar's hash is not the committed script's. Re-bless \
             with INF_BLESS_SAMPLES=1"
        );
    }

    /// **The level binds the script**, which is the whole arrangement: one
    /// `ActorClass` component naming a `.infini` GUID.
    #[test]
    fn the_level_binds_the_mission_to_the_hero() {
        let doc = heist_scene();
        let e = doc.entity_of(HEIST_HERO_GUID).expect("the hero");
        let bound = doc
            .world()
            .world()
            .get::<inf_ecs::components::ActorClass>(e)
            .expect("the hero is bound to a class");
        assert_eq!(bound.0, HEIST_SCRIPT_GUID);
    }
}
