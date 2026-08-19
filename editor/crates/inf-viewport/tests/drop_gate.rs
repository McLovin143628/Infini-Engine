//! **A dropped mesh must be BOUND to the entity it spawns** (Wave E audit, A6).
//!
//! From P4 to Wave E nothing in the editor ever wrote `MeshRef::asset`: a
//! dragged-in prop spawned a placeholder cube named after the asset, and "Edit
//! in Model Editor" had nothing to open on any prop the user had actually
//! placed. Wave E fixed it in `EngineHost::spawn_drop` and again, separately, in
//! `scene_spawn_asset` — and the arm that certified the fix called neither. It
//! called `SceneDoc::edit_create_mesh_asset` directly, so its own justification
//! ("delete the `mesh_asset()` branch in `spawn_drop` and the second assertion
//! fails") named a mutation that, measured, left it green.
//!
//! Both doors now route through `inf_editor_core::viewport_drop::spawn_asset_entity`,
//! whose two branches ARE arm-covered in that crate. What no runtime test can
//! reach is that this caller still goes through it: `spawn_drop` needs an
//! `EngineHost` — a GPU device, a window and an asset root — and the module is
//! `#[cfg(any(windows, target_os = "macos"))]` besides. So the pin is on the
//! source, and it reads the FUNCTION BODY rather than a spelling anywhere in
//! the file (the P23 law: ban a scope, not a string).
//!
//! `include_str!` is safe on a Windows checkout because `.rs` carries
//! `text eol=lf` in `.gitattributes` (P22.4); the CRLF strip is kept anyway,
//! since a locally-created file has whatever the editor wrote.

const HOST: &str = include_str!("../src/host.rs");

/// The body of `EngineHost::spawn_drop`, as text.
fn spawn_drop_body() -> String {
    let src = HOST.replace("\r\n", "\n");
    let start = src
        .find("pub fn spawn_drop(")
        .expect("EngineHost::spawn_drop is the viewport's drag-drop door");
    let rest = &src[start..];
    // The next `\n    /// ` or `\n    pub fn ` starts the following item.
    let end = rest[1..]
        .find("\n    pub fn ")
        .or_else(|| rest[1..].find("\n    /// "))
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

#[test]
fn the_viewport_drop_path_uses_the_shared_asset_spawn_door() {
    let body = spawn_drop_body();
    assert!(
        body.contains("spawn_asset_entity(doc, name, mesh_asset, None)"),
        "a dropped asset must be created through \
         `inf_editor_core::viewport_drop::spawn_asset_entity`, the one door \
         `scene_spawn_asset` also uses — otherwise the two drop paths can \
         disagree about whether a dropped mesh is bound, which is the state \
         this wave found the editor in. Body:\n{body}"
    );
    // …and it must still be told WHICH asset, from the payload's kind: binding
    // on the id alone would put a texture's GUID into a `MeshRef`.
    assert!(
        body.contains("parsed.mesh_asset()"),
        "the binding must come from the payload's KIND, not from any id it \
         happens to carry. Body:\n{body}"
    );
}
