//! ts-rs export harness (ROADMAP P0.1.5).
//!
//! Running this test (re)generates the committed TypeScript bindings under
//! `editor/studio/src/bindings/`. CI's bindings-drift job reruns it and fails
//! on `git diff`, so the committed bindings can never lag the Rust types.
//! Every new type in `inf_editor_core::ipc` must be added to `ROOTS` here.

use std::path::Path;

use inf_editor_core::ipc::CollisionLayerDto;
use inf_editor_core::ipc::{
    AssetChanged, AssetDto, AssetFolderDto, AssetRefDto, AssetSnapshot, ComponentDto, DataAssetDto,
    DataFieldDto, DeleteResult, DetailsDto, ErosionParamsDto, ErosionReportDto, FileEntryDto,
    GitFileDto, GitStatusDto, GizmoModeDto, GizmoSpaceDto, ImportEventDto, LayoutSummary,
    LevelSettingsDto, LogLine, PackageErrorDto, PackageKindCountDto, PackageResultDto,
    ProjectInfoDto, ProjectSettingsDto, ProjectTemplateDto, PropFieldDto, PropValueDto,
    RecentProjectDto, SceneDelta, SceneNode, SceneSnapshot, SculptFalloffDto, SculptOpDto,
    SculptSettingsDto, SearchHitDto, SearchOptsDto, SeqInterpDto, SeqKeyDto, SeqTrackDto,
    SequenceDto, Snap2DDto, Snap3DDto, SortingLayerDto, SpawnKind, SpriteGridDto, SpriteRectDto,
    SpriteSheetDto, TilemapCellDto, TilemapDto, ToolModeDto, ViewModeDto, ViewportDrop,
    ViewportKey, ViewportModeDto, ViewportRect,
};
use inf_editor_core::ipc::{CollectionDto, MatOverridesDto, MatValuesDto, MaterialInstanceDto};
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
    AssetDto::export_all(&cfg).expect("export AssetDto");
    AssetFolderDto::export_all(&cfg).expect("export AssetFolderDto");
    AssetSnapshot::export_all(&cfg).expect("export AssetSnapshot");
    AssetRefDto::export_all(&cfg).expect("export AssetRefDto");
    DeleteResult::export_all(&cfg).expect("export DeleteResult");
    ImportEventDto::export_all(&cfg).expect("export ImportEventDto");
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
    ProjectSettingsDto::export_all(&cfg).expect("export ProjectSettingsDto");
    ErosionParamsDto::export_all(&cfg).expect("export ErosionParamsDto");
    ErosionReportDto::export_all(&cfg).expect("export ErosionReportDto");
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
    MatValuesDto::export_all(&cfg).expect("export MatValuesDto");
    MatOverridesDto::export_all(&cfg).expect("export MatOverridesDto");
    MaterialInstanceDto::export_all(&cfg).expect("export MaterialInstanceDto");
    CollectionDto::export_all(&cfg).expect("export CollectionDto");
}
