//! ts-rs export harness (ROADMAP P0.1.5).
//!
//! Running this test (re)generates the committed TypeScript bindings under
//! `editor/studio/src/bindings/`. CI's bindings-drift job reruns it and fails
//! on `git diff`, so the committed bindings can never lag the Rust types.
//! Every new type in `inf_editor_core::ipc` must be added to `ROOTS` here.

use std::path::Path;

use inf_editor_core::ipc::CollisionLayerDto;
use inf_editor_core::ipc::HeightmapProbeDto;
use inf_editor_core::ipc::{
    AddableComponentDto, AssetChanged, AssetDto, AssetFolderDto, AssetRefDto, AssetSnapshot,
    BiomeDefDto, BiomeSetDto, BiomeSettingsDto, BodyParamsDto, CaptureCameraDto, CaptureImportDto,
    CaptureIssueDto, CapturePreviewDto, CaptureProgressDto, CaptureResultDto, CaptureSettingsDto,
    CaptureStatusDto, CharacterCreateDto, CharacterJointDto, CharacterLegDto, CharacterPreviewDto,
    CharacterRigDto, CharacterSpecDto, ComponentDto, CoverageDto, CoverageViewDto, DataAssetDto,
    DataFieldDto, DataMapExportDto, DccApplyDto, DccDocDto, DccDragBeginDto, DccDragDto,
    DccExportDto, DccGarmentDto, DccGizmoModeDto, DccGroomDto, DccGroomResultDto, DccGroomStatDto,
    DccHistoryEntryDto, DccImportDto, DccModeDto, DccNewDto, DccPaintModeDto, DccPreviewDto,
    DccSaveDto, DccSculptModeDto, DccSelectDto, DccToolDto, DccUnwrapDto, DeleteResult, DetailsDto,
    ErosionParamsDto, ErosionReportDto, FileEntryDto, FoliageSettingsDto, GaitParamsDto,
    GeneratedSourceDto, GitFileDto, GitStatusDto, GizmoModeDto, GizmoSpaceDto, ImportEventDto,
    LakePreviewDto, LayoutSummary, LevelSettingsDto, LogLine, PackageErrorDto, PackageKindCountDto,
    PackageResultDto, PartitionSettingsDto, PhotoEntryDto, ProjectInfoDto, ProjectSettingsDto,
    ProjectTemplateDto, PropFieldDto, PropValueDto, RecentProjectDto, RiverBedConflictDto,
    RiverClimbDto, RiverReportDto, SaveResultDto, SceneDelta, SceneNode, SceneSnapshot,
    ScriptDiagnosticDto, SculptFalloffDto, SculptOpDto, SculptSettingsDto, SearchHitDto,
    SearchOptsDto, SeqInterpDto, SeqKeyDto, SeqTrackDto, SequenceDto, SkelApplyDto, SkelDocDto,
    SkelJointDto, SkelSocketDto, SkyAtmosphereDto, Snap2DDto, Snap3DDto, SortingLayerDto,
    SpawnKind, SpoilModeDto, SpriteGridDto, SpriteRectDto, SpriteSheetDto, TerrainBiomesDto,
    TerrainImportPlanDto, TerrainImportResultDto, TerrainImportSettingsDto, TilemapCellDto,
    TilemapDto, TimeOfDayDto, ToolModeDto, ViewModeDto, ViewportDrop, ViewportGizmoDto,
    ViewportKey, ViewportModeDto, ViewportRect, ViewportToolStatusDto, VoxelOpModeDto,
    VoxelSettingsDto, VoxelStatusDto, VoxelToolKindDto, WaterDefaultsDto, WaterSettingsDto,
    WaterToolKindDto, WeatherDto, WeatherPresetDto,
};
use inf_editor_core::ipc::{CollectionDto, MatOverridesDto, MatValuesDto, MaterialInstanceDto};
use inf_editor_core::ipc::{MixerBusDto, MixerConfigDto, MixerEffectDto};
use ts_rs::{Config, TS};

#[test]
fn export_bindings() {
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../studio/src/bindings");
    let cfg = Config::new().with_out_dir(out);

    // export_all also exports each root's transitive dependencies.
    ViewportRect::export_all(&cfg).expect("export ViewportRect");
    ViewportDrop::export_all(&cfg).expect("export ViewportDrop");
    ViewportKey::export_all(&cfg).expect("export ViewportKey");
    LogLine::export_all(&cfg).expect("export LogLine");
    // The Model Editor (P23.4). `DccApplyDto` pulls in `DccDocDto`, which pulls
    // in the two report DTOs and the mode enum, so the roots are the four the
    // frontend actually receives plus the two it sends.
    DccApplyDto::export_all(&cfg).expect("export DccApplyDto");
    DccDocDto::export_all(&cfg).expect("export DccDocDto");
    DccExportDto::export_all(&cfg).expect("export DccExportDto");
    DccImportDto::export_all(&cfg).expect("export DccImportDto");
    DccModeDto::export_all(&cfg).expect("export DccModeDto");
    DccPreviewDto::export_all(&cfg).expect("export DccPreviewDto");
    DccSaveDto::export_all(&cfg).expect("export DccSaveDto");
    DccGarmentDto::export_all(&cfg).expect("export DccGarmentDto");
    DccGroomDto::export_all(&cfg).expect("export DccGroomDto");
    DccGroomResultDto::export_all(&cfg).expect("export DccGroomResultDto");
    DccGroomStatDto::export_all(&cfg).expect("export DccGroomStatDto");
    DccSelectDto::export_all(&cfg).expect("export DccSelectDto");
    DccToolDto::export_all(&cfg).expect("export DccToolDto");
    // Wave D: the New Mesh dialog's payload, and the two gizmo settings that
    // were hard-wired constants until this wave. All three are rooted because
    // the panel holds each of them in local state before any call carries one.
    DccNewDto::export_all(&cfg).expect("export DccNewDto");
    DccHistoryEntryDto::export_all(&cfg).expect("export DccHistoryEntryDto");
    // Wave D: the CODE TAB's reply.
    GeneratedSourceDto::export_all(&cfg).expect("export GeneratedSourceDto");
    inf_editor_core::dcc::DccPivotDto::export_all(&cfg).expect("export DccPivotDto");
    inf_editor_core::dcc::DccOrientDto::export_all(&cfg).expect("export DccOrientDto");
    // P23.5's drag surface: the sculpt brush and the component gizmo.
    DccDragDto::export_all(&cfg).expect("export DccDragDto");
    DccDragBeginDto::export_all(&cfg).expect("export DccDragBeginDto");
    DccSculptModeDto::export_all(&cfg).expect("export DccSculptModeDto");
    // P24.2: the weight brush's mode. Exported as a root because the panel
    // holds one in local state before any drag exists to carry it.
    DccPaintModeDto::export_all(&cfg).expect("export DccPaintModeDto");
    DccGizmoModeDto::export_all(&cfg).expect("export DccGizmoModeDto");
    DccUnwrapDto::export_all(&cfg).expect("export DccUnwrapDto");
    // The Skeleton Editor (P24.3). `SkelApplyDto` pulls in `SkelDocDto`, which
    // pulls in the joint and socket rows — all four are rooted anyway, per this
    // file's rule that every `inf_editor_core::ipc` type appears here so a future
    // consumer importing one directly cannot drift.
    SkelApplyDto::export_all(&cfg).expect("export SkelApplyDto");
    SkelDocDto::export_all(&cfg).expect("export SkelDocDto");
    SkelJointDto::export_all(&cfg).expect("export SkelJointDto");
    SkelSocketDto::export_all(&cfg).expect("export SkelSocketDto");
    // The New Character wizard (P24.5). Same rule: every type it names is rooted,
    // even the ones `CharacterSpecDto`/`CharacterPreviewDto` already pull in.
    BodyParamsDto::export_all(&cfg).expect("export BodyParamsDto");
    GaitParamsDto::export_all(&cfg).expect("export GaitParamsDto");
    CharacterSpecDto::export_all(&cfg).expect("export CharacterSpecDto");
    CharacterJointDto::export_all(&cfg).expect("export CharacterJointDto");
    CharacterLegDto::export_all(&cfg).expect("export CharacterLegDto");
    CharacterRigDto::export_all(&cfg).expect("export CharacterRigDto");
    CharacterPreviewDto::export_all(&cfg).expect("export CharacterPreviewDto");
    CharacterCreateDto::export_all(&cfg).expect("export CharacterCreateDto");
    // The capture wizard (P25.4). Same rule: every type it names is rooted, even
    // the ones `CaptureStatusDto` already pulls in, so a panel importing one
    // directly cannot drift.
    CaptureCameraDto::export_all(&cfg).expect("export CaptureCameraDto");
    CaptureSettingsDto::export_all(&cfg).expect("export CaptureSettingsDto");
    PhotoEntryDto::export_all(&cfg).expect("export PhotoEntryDto");
    CaptureIssueDto::export_all(&cfg).expect("export CaptureIssueDto");
    CoverageViewDto::export_all(&cfg).expect("export CoverageViewDto");
    CoverageDto::export_all(&cfg).expect("export CoverageDto");
    CaptureResultDto::export_all(&cfg).expect("export CaptureResultDto");
    CaptureStatusDto::export_all(&cfg).expect("export CaptureStatusDto");
    CaptureProgressDto::export_all(&cfg).expect("export CaptureProgressDto");
    CapturePreviewDto::export_all(&cfg).expect("export CapturePreviewDto");
    CaptureImportDto::export_all(&cfg).expect("export CaptureImportDto");
    LayoutSummary::export_all(&cfg).expect("export LayoutSummary");
    SceneNode::export_all(&cfg).expect("export SceneNode");
    SceneSnapshot::export_all(&cfg).expect("export SceneSnapshot");
    SceneDelta::export_all(&cfg).expect("export SceneDelta");
    SpawnKind::export_all(&cfg).expect("export SpawnKind");
    PropValueDto::export_all(&cfg).expect("export PropValueDto");
    PropFieldDto::export_all(&cfg).expect("export PropFieldDto");
    ComponentDto::export_all(&cfg).expect("export ComponentDto");
    DetailsDto::export_all(&cfg).expect("export DetailsDto");
    AddableComponentDto::export_all(&cfg).expect("export AddableComponentDto");
    AssetDto::export_all(&cfg).expect("export AssetDto");
    AssetFolderDto::export_all(&cfg).expect("export AssetFolderDto");
    AssetSnapshot::export_all(&cfg).expect("export AssetSnapshot");
    AssetRefDto::export_all(&cfg).expect("export AssetRefDto");
    DeleteResult::export_all(&cfg).expect("export DeleteResult");
    ImportEventDto::export_all(&cfg).expect("export ImportEventDto");
    ViewportToolStatusDto::export_all(&cfg).expect("export ViewportToolStatusDto");
    ViewportGizmoDto::export_all(&cfg).expect("export ViewportGizmoDto");
    SaveResultDto::export_all(&cfg).expect("export SaveResultDto");
    HeightmapProbeDto::export_all(&cfg).expect("export HeightmapProbeDto");
    inf_editor_core::ipc::GeoAnchorDto::export_all(&cfg).expect("export GeoAnchorDto");
    TerrainImportSettingsDto::export_all(&cfg).expect("export TerrainImportSettingsDto");
    TerrainImportPlanDto::export_all(&cfg).expect("export TerrainImportPlanDto");
    TerrainImportResultDto::export_all(&cfg).expect("export TerrainImportResultDto");
    // IB-3: the GIS import wizard's wire. Rooted here rather than mirrored by
    // hand in TypeScript — the `sm.rs` trade (hand-written mirrors, no drift
    // guard) is the one thing a wizard with a dozen fields must not repeat.
    inf_editor_core::ipc::GisFieldDto::export_all(&cfg).expect("export GisFieldDto");
    inf_editor_core::ipc::GisCrsDto::export_all(&cfg).expect("export GisCrsDto");
    inf_editor_core::ipc::GisProbeDto::export_all(&cfg).expect("export GisProbeDto");
    inf_editor_core::ipc::GisImportSettingsDto::export_all(&cfg)
        .expect("export GisImportSettingsDto");
    inf_editor_core::ipc::GisImportResultDto::export_all(&cfg).expect("export GisImportResultDto");
    AssetChanged::export_all(&cfg).expect("export AssetChanged");
    DataAssetDto::export_all(&cfg).expect("export DataAssetDto");
    DataFieldDto::export_all(&cfg).expect("export DataFieldDto");
    ProjectInfoDto::export_all(&cfg).expect("export ProjectInfoDto");
    RecentProjectDto::export_all(&cfg).expect("export RecentProjectDto");
    ProjectTemplateDto::export_all(&cfg).expect("export ProjectTemplateDto");
    FileEntryDto::export_all(&cfg).expect("export FileEntryDto");
    GitStatusDto::export_all(&cfg).expect("export GitStatusDto");
    GitFileDto::export_all(&cfg).expect("export GitFileDto");
    SearchOptsDto::export_all(&cfg).expect("export SearchOptsDto");
    SearchHitDto::export_all(&cfg).expect("export SearchHitDto");
    // The InfiniScript parse refusals (wave SCRIPT2b).
    ScriptDiagnosticDto::export_all(&cfg).expect("export ScriptDiagnosticDto");
    SpriteGridDto::export_all(&cfg).expect("export SpriteGridDto");
    SpriteRectDto::export_all(&cfg).expect("export SpriteRectDto");
    SpriteSheetDto::export_all(&cfg).expect("export SpriteSheetDto");
    SortingLayerDto::export_all(&cfg).expect("export SortingLayerDto");
    CollisionLayerDto::export_all(&cfg).expect("export CollisionLayerDto");
    ViewportModeDto::export_all(&cfg).expect("export ViewportModeDto");
    ViewModeDto::export_all(&cfg).expect("export ViewModeDto");
    Snap2DDto::export_all(&cfg).expect("export Snap2DDto");
    GizmoModeDto::export_all(&cfg).expect("export GizmoModeDto");
    GizmoSpaceDto::export_all(&cfg).expect("export GizmoSpaceDto");
    Snap3DDto::export_all(&cfg).expect("export Snap3DDto");
    ToolModeDto::export_all(&cfg).expect("export ToolModeDto");
    SculptOpDto::export_all(&cfg).expect("export SculptOpDto");
    SculptFalloffDto::export_all(&cfg).expect("export SculptFalloffDto");
    SculptSettingsDto::export_all(&cfg).expect("export SculptSettingsDto");
    // P19.2 biome painting: the brush push, the `.inf_biomes` editor's view, and
    // the toolbar's resolved vocabulary. `BiomeSetDto`/`TerrainBiomesDto` reach
    // `BiomeDefDto` transitively, but every root is listed explicitly (the
    // `PartitionSettingsDto` convention).
    BiomeSettingsDto::export_all(&cfg).expect("export BiomeSettingsDto");
    BiomeDefDto::export_all(&cfg).expect("export BiomeDefDto");
    BiomeSetDto::export_all(&cfg).expect("export BiomeSetDto");
    TerrainBiomesDto::export_all(&cfg).expect("export TerrainBiomesDto");
    FoliageSettingsDto::export_all(&cfg).expect("export FoliageSettingsDto");
    // P20.4 hydrology authoring: the brush push, the biome-hint defaults, the
    // lake fill preview and the river verdict. Every root listed explicitly.
    WaterToolKindDto::export_all(&cfg).expect("export WaterToolKindDto");
    WaterSettingsDto::export_all(&cfg).expect("export WaterSettingsDto");
    VoxelToolKindDto::export_all(&cfg).expect("export VoxelToolKindDto");
    VoxelOpModeDto::export_all(&cfg).expect("export VoxelOpModeDto");
    SpoilModeDto::export_all(&cfg).expect("export SpoilModeDto");
    VoxelSettingsDto::export_all(&cfg).expect("export VoxelSettingsDto");
    VoxelStatusDto::export_all(&cfg).expect("export VoxelStatusDto");
    WaterDefaultsDto::export_all(&cfg).expect("export WaterDefaultsDto");
    LakePreviewDto::export_all(&cfg).expect("export LakePreviewDto");
    RiverClimbDto::export_all(&cfg).expect("export RiverClimbDto");
    RiverBedConflictDto::export_all(&cfg).expect("export RiverBedConflictDto");
    RiverReportDto::export_all(&cfg).expect("export RiverReportDto");
    ProjectSettingsDto::export_all(&cfg).expect("export ProjectSettingsDto");
    // Wave E batch A: the app-level preferences file. `export_all` also emits
    // its two nested tables (`Snap3DDto`, `FoliageSettingsDto`), which are
    // rooted above in their own right.
    inf_editor_core::ipc::EditorSettings::export_all(&cfg).expect("export EditorSettings");
    // Island wave I5: the GAME's binding table, rendered by the preferences
    // panel from the same projection the in-game dialog uses.
    inf_editor_core::ipc::GameBindingRowDto::export_all(&cfg).expect("export GameBindingRowDto");
    inf_editor_core::ipc::GameBindingApplyDto::export_all(&cfg)
        .expect("export GameBindingApplyDto");
    // Wave E batch B: the answer to "which editors can open this object".
    inf_editor_core::ipc::EntityEditorsDto::export_all(&cfg).expect("export EntityEditorsDto");
    // Wave E batch C: the two native-viewport pointer gestures and the pointer
    // feel the preferences push into the input state machine.
    inf_editor_core::ipc::ViewportContextMenuDto::export_all(&cfg)
        .expect("export ViewportContextMenuDto");
    inf_editor_core::ipc::ViewportActivateDto::export_all(&cfg)
        .expect("export ViewportActivateDto");
    inf_editor_core::ipc::ViewportInteractionDto::export_all(&cfg)
        .expect("export ViewportInteractionDto");
    ErosionParamsDto::export_all(&cfg).expect("export ErosionParamsDto");
    ErosionReportDto::export_all(&cfg).expect("export ErosionReportDto");
    DataMapExportDto::export_all(&cfg).expect("export DataMapExportDto");
    TilemapCellDto::export_all(&cfg).expect("export TilemapCellDto");
    TilemapDto::export_all(&cfg).expect("export TilemapDto");
    PackageKindCountDto::export_all(&cfg).expect("export PackageKindCountDto");
    PackageResultDto::export_all(&cfg).expect("export PackageResultDto");
    PackageErrorDto::export_all(&cfg).expect("export PackageErrorDto");
    SequenceDto::export_all(&cfg).expect("export SequenceDto");
    SeqTrackDto::export_all(&cfg).expect("export SeqTrackDto");
    SeqKeyDto::export_all(&cfg).expect("export SeqKeyDto");
    SeqInterpDto::export_all(&cfg).expect("export SeqInterpDto");
    LevelSettingsDto::export_all(&cfg).expect("export LevelSettingsDto");
    // Nested inside `LevelSettingsDto`, so `export_all` already emits it — rooted
    // explicitly per this file's rule that every `inf_editor_core::ipc` type
    // appears here, so a future consumer importing it directly cannot drift.
    PartitionSettingsDto::export_all(&cfg).expect("export PartitionSettingsDto");
    TimeOfDayDto::export_all(&cfg).expect("export TimeOfDayDto");
    SkyAtmosphereDto::export_all(&cfg).expect("export SkyAtmosphereDto");
    WeatherDto::export_all(&cfg).expect("export WeatherDto");
    WeatherPresetDto::export_all(&cfg).expect("export WeatherPresetDto");
    MatValuesDto::export_all(&cfg).expect("export MatValuesDto");
    MatOverridesDto::export_all(&cfg).expect("export MatOverridesDto");
    MaterialInstanceDto::export_all(&cfg).expect("export MaterialInstanceDto");
    CollectionDto::export_all(&cfg).expect("export CollectionDto");
    MixerEffectDto::export_all(&cfg).expect("export MixerEffectDto");
    MixerBusDto::export_all(&cfg).expect("export MixerBusDto");
    MixerConfigDto::export_all(&cfg).expect("export MixerConfigDto");
}

/// **`#[ts(export)]` writes bindings where the drift job cannot see them.**
///
/// Wave D shipped six DTOs carrying the attribute, and `ts-rs` wrote a *second*
/// bindings directory beside the crate — `editor/crates/inf-editor-core/bindings/`
/// — six duplicate `.ts` files that were committed, that no consumer imports,
/// and that go stale silently. CI's drift job is scoped to
/// `editor/studio/src/bindings`, so it could not see them; the fix deleted them
/// and removed the attributes, and nothing stopped the next author doing it
/// again.
///
/// This is that something. The export door is `export_bindings` above and its
/// `Config::with_out_dir`; the attribute is a second door with no address.
#[test]
fn no_dto_exports_itself_behind_the_harness() {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&src, &mut files);
    assert!(files.len() > 5, "the sweep found no sources under {src:?}");
    let mut offenders = Vec::new();
    for f in &files {
        // The P22 CRLF law: normalize before searching a `.rs`.
        let text = std::fs::read_to_string(f)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        // The attribute, not the words: `ts(export` catches `#[ts(export)]` and
        // the same thing inside a longer `#[ts(export, rename = ...)]` list,
        // without matching an ordinary argument called `export`.
        if text.contains("ts(export") {
            offenders.push(f.display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "these types export themselves rather than through `export_bindings`, \
         so their `.ts` lands beside the crate where CI's drift job does not \
         look: {offenders:?}"
    );
    // Not vacuous: the directory that attribute created must also be gone.
    let stray = Path::new(env!("CARGO_MANIFEST_DIR")).join("bindings");
    assert!(
        !stray.exists(),
        "{stray:?} is back — a second bindings directory the drift job cannot see"
    );
}
