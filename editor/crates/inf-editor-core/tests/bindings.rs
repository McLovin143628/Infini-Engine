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
    BiomeDefDto, BiomeSetDto, BiomeSettingsDto, ComponentDto, DataAssetDto, DataFieldDto,
    DataMapExportDto, DeleteResult, DetailsDto, ErosionParamsDto, ErosionReportDto, FileEntryDto,
    FoliageSettingsDto, GitFileDto, GitStatusDto, GizmoModeDto, GizmoSpaceDto, ImportEventDto,
    LakePreviewDto, LayoutSummary, LevelSettingsDto, LogLine, PackageErrorDto, PackageKindCountDto,
    PackageResultDto, PartitionSettingsDto, ProjectInfoDto, ProjectSettingsDto, ProjectTemplateDto,
    PropFieldDto, PropValueDto, RecentProjectDto, RiverBedConflictDto, RiverClimbDto,
    RiverReportDto, SaveResultDto, SceneDelta, SceneNode, SceneSnapshot, SculptFalloffDto,
    SculptOpDto, SculptSettingsDto, SearchHitDto, SearchOptsDto, SeqInterpDto, SeqKeyDto,
    SeqTrackDto, SequenceDto, SkyAtmosphereDto, Snap2DDto, Snap3DDto, SortingLayerDto, SpawnKind,
    SpoilModeDto, SpriteGridDto, SpriteRectDto, SpriteSheetDto, TerrainBiomesDto,
    TerrainImportPlanDto, TerrainImportResultDto, TerrainImportSettingsDto, TilemapCellDto,
    TilemapDto, TimeOfDayDto, ToolModeDto, ViewModeDto, ViewportDrop, ViewportKey, ViewportModeDto,
    ViewportRect, ViewportToolStatusDto, VoxelOpModeDto, VoxelSettingsDto, VoxelStatusDto,
    VoxelToolKindDto, WaterDefaultsDto, WaterSettingsDto, WaterToolKindDto, WeatherDto,
    WeatherPresetDto,
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
    SaveResultDto::export_all(&cfg).expect("export SaveResultDto");
    HeightmapProbeDto::export_all(&cfg).expect("export HeightmapProbeDto");
    TerrainImportSettingsDto::export_all(&cfg).expect("export TerrainImportSettingsDto");
    TerrainImportPlanDto::export_all(&cfg).expect("export TerrainImportPlanDto");
    TerrainImportResultDto::export_all(&cfg).expect("export TerrainImportResultDto");
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
