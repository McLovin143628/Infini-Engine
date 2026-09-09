//! **`inf-import`** — the headless Infini side of the Unreal bridge (wave
//! ASSET0, clause 2).
//!
//! ```text
//! inf-import --manifest <manifest.json> --into <project-dir>
//!            [--pack <name>]…            only these packs
//!            [--max-texture <n>]         ceiling on a texture's longest side (0 = source)
//!            [--dest <subfolder>]        under <project>/Content (default "UE")
//!            [--bind <Stem>=<key>]…      write a material at a committed GUID
//!            [--no-meshes]               materials and textures only
//!            [--character-lods <n>]      LOD rungs to store per character (default 3)
//!            [--retarget-to <objpath>]   the rig every clip is retargeted onto
//!            [--rebind-mesh <Stem>=<key>] write that RIGID mesh at <Stem>'s GUID
//!            [--rebind-character <key>]  write that body at the starter GUIDs
//!            [--rebind-character-f <key>] …and that one at the FEMALE starter's
//!            [--only <key substring>]…    import only meshes whose key matches
//!            [--wear-cloth <guid>]        put that `.inf_cloth` on every level's pawn
//!            [--wearable <m|f>:<outfit|hair>:<key>[:<joint>]]…  wear it
//!            [--dry-run]                 read the manifest, write nothing
//! ```
//!
//! Everything it does is [`inf_editor_core::assets::ue_import`]; this file is
//! argument parsing and a report. See that module for the PBR remap, the clamp
//! and what a rebind is for.

use std::path::PathBuf;
use std::process::ExitCode;

use inf_editor_core::assets::ue_import::{import_manifest, UeImportOptions};
use inf_editor_core::assets::AssetProject;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") || args.is_empty() {
        print_help();
        return ExitCode::SUCCESS;
    }
    // **`garment` is a VERB, not a flag** (wave CHAR1b.2, the CHAR1a audit's item
    // 87). It shares this binary rather than getting its own for the reason this
    // crate's own manifest gives for the binary existing at all:
    // `garment_from_files` is `inf_editor_core`'s, and `tools/inf-cli`
    // deliberately links Ring 0 and nothing else. Everything the verb decides
    // lives in Ring 1 where a test can reach it; this is argument parsing and a
    // report.
    if args.first().map(String::as_str) == Some("garment") {
        return match run_garment(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("inf-import garment: {e}");
                ExitCode::FAILURE
            }
        };
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inf-import: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!(
        "inf-import {} — Unreal → Infini asset bridge (the import side)\n\n\
         USAGE:\n  \
             inf-import --manifest <manifest.json> --into <project-dir>\n             \
             [--pack <name>]… [--max-texture <n>] [--dest <sub>]\n             \
             [--bind <Stem>=<material-key>]… [--no-meshes]\n             \
             [--character-lods <n>] [--retarget-to <objpath>] [--dry-run]\n             \
             [--only <key>]… [--wearable <m|f>:<outfit|hair>:<key>[:<joint>]]…\n  \
             inf-import --into <project-dir> --rebind-graph <m|f>\n  \
             inf-import garment --mesh <a.inf_mesh> --out <a.inf_cloth>\n             \
             [--skeleton <r.inf_skel>] [--pin-top <fraction>] [--body-radius <m>]\n\n\
         The manifest is written by tools/ue-export/export.py. A --bind writes\n\
         an imported material at the GUID the committed ground library assigns\n\
         that stem, so a committed level picks it up without naming licensed\n\
         content: e.g. --bind Road_Asphalt=<the asphalt material's key>.\n\n\
         NOTHING THIS WRITES MAY BE COMMITTED. It goes into a project's Content,\n\
         which for the island is outside this repository -- and since the ASSET0\n\
         audit that is a door rather than a sentence: an --into inside the engine\n\
         checkout is REFUSED before the first texture is decoded.",
        env!("CARGO_PKG_VERSION")
    );
}

fn run(args: &[String]) -> Result<(), String> {
    let mut manifest: Option<PathBuf> = None;
    let mut into: Option<PathBuf> = None;
    let mut dry = false;
    // **Rebuild a committed identity's locomotion graph and nothing else**
    // (wave CHAR1b.2). See `ue_import::rebind_graphs` for why this is its own
    // verb rather than a side effect of `--rebind-character`: the island's hero
    // wears a body from one manifest and plays clips from another, so re-running
    // a body rebind to pick up a new map row swaps the character.
    let mut rebind_graphs: Vec<String> = Vec::new();
    let mut opts = UeImportOptions::default();
    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--manifest" => manifest = Some(PathBuf::from(take(&mut i)?)),
            "--into" => into = Some(PathBuf::from(take(&mut i)?)),
            "--pack" => opts.packs.push(take(&mut i)?),
            "--dest" => opts.dest = take(&mut i)?,
            "--max-texture" => {
                let v = take(&mut i)?;
                opts.max_texture = v
                    .parse()
                    .map_err(|_| format!("--max-texture wants a number, got {v:?}"))?;
            }
            "--bind" => {
                let v = take(&mut i)?;
                let (stem, key) = v
                    .split_once('=')
                    .ok_or_else(|| format!("--bind wants <Stem>=<material-key>, got {v:?}"))?;
                opts.rebinds.push((stem.to_string(), key.to_string()));
            }
            // **Wave WPN2d.** `--bind` writes a MATERIAL at a committed stem's
            // GUID; this writes a rigid MESH at one. Two flags because the two
            // identities are derived by two different Ring-0 functions and a
            // single flag would have to guess which.
            "--rebind-mesh" => {
                let v = take(&mut i)?;
                let (stem, key) = v
                    .split_once('=')
                    .ok_or_else(|| format!("--rebind-mesh wants <Stem>=<mesh-key>, got {v:?}"))?;
                opts.rebind_meshes.push((stem.to_string(), key.to_string()));
            }
            "--no-meshes" => opts.meshes = false,
            "--character-lods" => {
                let v = take(&mut i)?;
                opts.character_lods = v
                    .parse()
                    .map_err(|_| format!("--character-lods wants a number, got {v:?}"))?;
            }
            "--retarget-to" => opts.retarget_to = Some(take(&mut i)?),
            "--rebind-character" => opts.rebind_character = Some(take(&mut i)?),
            "--rebind-character-f" => opts.rebind_character_f = Some(take(&mut i)?),
            "--rebind-graph" => rebind_graphs.push(take(&mut i)?),
            "--only" => opts.only.push(take(&mut i)?),
            "--wear-cloth" => {
                let v = take(&mut i)?;
                opts.wear_cloth =
                    Some(inf_asset::AssetId(v.parse().map_err(|e| {
                        format!("--wear-cloth wants a GUID, got {v:?} ({e})")
                    })?));
            }
            "--wearable" => {
                opts.wearables
                    .push(inf_editor_core::assets::ue_import::parse_wearable(&take(
                        &mut i,
                    )?)?)
            }
            "--dry-run" => dry = true,
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 1;
    }
    let into = into.ok_or("--into is required")?;
    let content = if into.join("Content").is_dir() {
        into.join("Content")
    } else {
        into.clone()
    };
    // The graph-only verb needs no manifest at all: every clip it binds is
    // already in the project.
    if !rebind_graphs.is_empty() {
        let mut project = AssetProject::open(&content).map_err(|e| e.to_string())?;
        let mut stems: Vec<(inf_editor_core::character::CharacterIds, &str)> = Vec::new();
        for which in &rebind_graphs {
            match which.as_str() {
                "m" | "male" | "Starter" => stems.push((
                    inf_editor_core::samples::starter_character_ids(),
                    "Starter_Locomotion",
                )),
                "f" | "female" | "Starter_F" => stems.push((
                    inf_editor_core::samples::starter_character_f_ids(),
                    "Starter_F_Locomotion",
                )),
                other => return Err(format!("--rebind-graph wants `m` or `f`, got {other:?}")),
            }
        }
        let report = inf_editor_core::assets::ue_import::rebind_graphs(&mut project, &stems);
        for a in &report.advisories {
            println!("inf-import: ADVISORY {a}");
        }
        for (stem, id) in &report.rebinds {
            println!("inf-import: REBOUND  {stem} -> {id}");
        }
        return Ok(());
    }
    // **The garment verb needs no manifest either** (carried item 137): the
    // `.inf_cloth` it names is already in the project, exactly as every clip the
    // graph verb binds is.
    if manifest.is_none() {
        if let Some(cloth) = opts.wear_cloth {
            let project = AssetProject::open(&content).map_err(|e| e.to_string())?;
            let mut report = inf_editor_core::assets::ue_import::UeImportReport::default();
            let n = inf_editor_core::assets::ue_import::wear_cloth_in_levels(
                &project,
                cloth,
                &mut report,
            );
            for a in &report.advisories {
                println!("inf-import: ADVISORY {a}");
            }
            println!("inf-import: {n} level(s) dressed with the garment {cloth}");
            return Ok(());
        }
    }
    let manifest = manifest.ok_or("--manifest is required")?;
    println!("inf-import: manifest {}", manifest.display());
    println!("inf-import: content  {}", content.display());
    println!(
        "inf-import: packs {} · max-texture {} · dest {} · meshes {} · binds {}",
        if opts.packs.is_empty() {
            "(all)".to_string()
        } else {
            opts.packs.join(",")
        },
        opts.max_texture,
        opts.dest,
        opts.meshes,
        opts.rebinds.len()
    );
    if dry {
        println!("inf-import: --dry-run, nothing written");
        return Ok(());
    }

    let started = std::time::Instant::now();
    let mut project = AssetProject::open(&content).map_err(|e| e.to_string())?;
    let report = import_manifest(&mut project, &manifest, &opts).map_err(|e| e.to_string())?;
    let secs = started.elapsed().as_secs_f64();

    for a in &report.advisories {
        println!("inf-import: ADVISORY {a}");
    }
    for (key, id) in &report.materials {
        println!("inf-import: material {id}  {key}");
    }
    for (key, id, rungs, tris) in &report.meshes {
        println!("inf-import: mesh     {id}  {tris:>7} tris, {rungs} source rungs  {key}");
    }
    for (key, mesh, skel, rungs, tris, joints) in &report.skeletal {
        println!(
            "inf-import: body     {mesh}  {tris:>7} tris, {rungs} rungs, {joints} joints, \
             skeleton {}  {key}",
            skel.map(|s| s.to_string()).unwrap_or_else(|| "NONE".into())
        );
    }
    for (key, id, tracks) in &report.clips {
        println!("inf-import: clip     {id}  {tracks:>4} tracks  {key}");
    }
    for (pack, licence, ship) in &report.licences {
        println!(
            "inf-import: LICENCE  {pack} [{}] {licence}",
            if *ship { "MAY SHIP" } else { "LOCAL ONLY" }
        );
    }
    for (stem, id) in &report.rebinds {
        println!("inf-import: REBOUND  {stem} -> {id}");
    }
    for f in &report.fixtures {
        println!(
            "inf-import: fixture  {} at ({:.2}, {:.2}, {:.2}) m, {:.0} cd, {:.1} m range, sRGB8 {:?}",
            f.name,
            f.offset_m[0],
            f.offset_m[1],
            f.offset_m[2],
            f.intensity,
            f.range_m,
            f.color_srgb8
        );
    }
    println!(
        "inf-import: {} materials, {} textures, {} meshes, {} bodies, {} clips, \
         {} fixtures, {:.1} MB, {secs:.1} s",
        report.materials.len(),
        report.textures.len(),
        report.meshes.len(),
        report.skeletal.len(),
        report.clips.len(),
        report.fixtures.len(),
        report.bytes as f64 / 1_048_576.0,
    );
    Ok(())
}

/// **`inf-import garment`** — author a `.inf_cloth` from files (wave CHAR1b.2).
///
/// The whole of what a garment IS lives in
/// `inf_editor_core::groom::garment_from_files`; this reads the arguments, the
/// two files and writes the payload with its sidecar through the same
/// `AssetProject` door every other written asset goes through.
fn run_garment(args: &[String]) -> Result<(), String> {
    let mut mesh: Option<PathBuf> = None;
    let mut skel: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut pin_top = 0.05_f64;
    let mut spec = inf_editor_core::groom::GarmentSpec::default();
    let mut i = 0;
    while i < args.len() {
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[*i - 1]))
        };
        match args[i].as_str() {
            "--mesh" => mesh = Some(PathBuf::from(take(&mut i)?)),
            "--skeleton" => skel = Some(PathBuf::from(take(&mut i)?)),
            "--out" => out = Some(PathBuf::from(take(&mut i)?)),
            "--pin-top" => {
                let v = take(&mut i)?;
                pin_top = v
                    .parse()
                    .map_err(|_| format!("--pin-top wants a fraction, got {v:?}"))?;
            }
            "--body-radius" => {
                let v = take(&mut i)?;
                spec.body_radius_m = v
                    .parse()
                    .map_err(|_| format!("--body-radius wants metres, got {v:?}"))?;
            }
            other => return Err(format!("unknown option {other:?}")),
        }
        i += 1;
    }
    let mesh_path = mesh.ok_or("--mesh is required")?;
    let out_path = out.ok_or("--out is required")?;
    let mesh_bytes =
        std::fs::read(&mesh_path).map_err(|e| format!("{}: {e}", mesh_path.display()))?;
    let skel_bytes = match &skel {
        Some(p) => Some(std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?),
        None => None,
    };
    // The garment's `source_mesh` is the mesh asset's own GUID, read off its
    // sidecar — the same id the Model Editor would have passed, so a cloth
    // authored here and one authored in a session name the same source.
    let (asset, report) = inf_editor_core::groom::garment_from_files(
        &mesh_bytes,
        skel_bytes.as_deref(),
        source_guid(&mesh_path),
        pin_top,
        spec,
    )
    .map_err(|e| e.to_string())?;
    let dir = out_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let name = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("--out needs a file name")?
        .to_string();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut project = AssetProject::open(&dir).map_err(|e| e.to_string())?;
    let id = project
        .write_asset(&dir, &name, &asset, None, Vec::new(), None)
        .map_err(|e| e.to_string())?;
    println!("inf-import: GARMENT {name}.inf_cloth -> {id}");
    println!(
        "  {} particles, {} triangles, {} stretch + {} bend constraints, {} pinned, {} capsules",
        report.particles,
        report.triangles,
        report.stretch,
        report.bend,
        report.pinned,
        report.capsules
    );
    Ok(())
}

/// The mesh asset's own GUID, read off its committed sidecar; sixteen zero bytes
/// when there is no sidecar to read, which is the same "no source" a scratch mesh
/// carries.
fn source_guid(mesh_path: &std::path::Path) -> [u8; 16] {
    let sidecar = {
        let mut p = mesh_path.as_os_str().to_os_string();
        p.push(".toml");
        PathBuf::from(p)
    };
    std::fs::read_to_string(&sidecar)
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
        .and_then(|t| t.get("guid").and_then(|v| v.as_str()).map(str::to_string))
        .and_then(|g| g.parse::<uuid::Uuid>().ok())
        .map(|u| *u.as_bytes())
        .unwrap_or([0u8; 16])
}
