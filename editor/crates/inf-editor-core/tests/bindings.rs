//! ts-rs export harness (ROADMAP P0.1.5).
//!
//! Running this test (re)generates the committed TypeScript bindings under
//! `editor/studio/src/bindings/`. CI's bindings-drift job reruns it and fails
//! on `git diff`, so the committed bindings can never lag the Rust types.
//! Every new type in `inf_editor_core::ipc` must be added to `ROOTS` here.

use std::path::Path;

use inf_editor_core::ipc::{
    AssetChanged, AssetDto, AssetFolderDto, AssetRefDto, AssetSnapshot, ComponentDto, DataAssetDto,
    DataFieldDto, DeleteResult, DetailsDto, FileEntryDto, GitFileDto, GitStatusDto, ImportEventDto,
    LayoutSummary, LogLine, ProjectInfoDto, ProjectTemplateDto, PropFieldDto, PropValueDto,
    RecentProjectDto, SceneDelta, SceneNode, SceneSnapshot, SearchHitDto, SearchOptsDto,
    SortingLayerDto, SpawnKind, SpriteGridDto, SpriteRectDto, SpriteSheetDto, ViewportDrop,
    ViewportKey, ViewportRect,
};
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
}
