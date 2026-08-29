//! **What a program names** — the one walk that tells a cook which assets a
//! Blueprint class, or an InfiniScript, reaches by name.
//!
//! # The SK1c blocker-4 lesson, closed
//!
//! `inf_packager`'s `asset_deps` has six arms — Level, Material, StateMachine,
//! AnimClip, BiomeSet, Pcg — and **no Blueprint among them**. Wave SK1c stopped
//! on exactly that: an item catalogue is authored in a `.inf_act`, so a mesh the
//! catalogue named would never enter the cook's dependency closure and would not
//! be packed at all. The wave was stopped rather than shipping *"three cubes and
//! a dangling reference"*.
//!
//! SCRIPT1b has to open that door anyway, because a `.infini` is a *third*
//! producer of the same IR and would arrive with the same hole. So the walk
//! lives **here**, in Ring 0 beside the IR it reads, and the cook and the PIE
//! payload builder both call it. Two implementations that agree by construction
//! is the arrangement this house keeps paying for (P22's *one door for three
//! paths*).
//!
//! # Enumerating, not guessing
//!
//! A string argument in this IR can be three different things and only one of
//! them is an asset:
//!
//! * an **asset name** — `engine.spawn("…")`'s prefab;
//! * a **gameplay id** — `item.give("rifle")`, whose meaning is a row in a
//!   catalogue the *same program* defines;
//! * **text** — a `debug.print` message;
//! * a whole **table** — `item.define`'s TOML catalogue, `door.spawn`'s door
//!   list — which is a string with structure inside it that nothing here reads.
//!
//! Resolving the second class against the asset database would attach an actor
//! to whatever asset happened to be called `rifle`; refusing it would advise on
//! every correct program. So each is written down, in [`STR_PORTS`], with a
//! **role**. The table is not a list of the ports somebody remembered: the census
//! arm walks the node kit and fails if a registered `Str` input port has no row,
//! so a verb added in SCRIPT2 cannot join the surface unclassified.
//!
//! # What the walk deliberately does not do
//!
//! It does not read *inside* a string. `item.define`'s argument is a whole TOML
//! catalogue and `door.spawn`'s is a door table; the day either grows a mesh
//! field, the parse belongs beside that table's own parser and the role of the
//! port becomes [`StrRole::Table`]'s problem rather than this walk's. Naming it
//! now is the difference between a known bound and a surprise.

use crate::{BlueprintClass, BlueprintFn, Expr, Lit, Stmt};

/// What a `Str` input port of a node means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrRole {
    /// Names an **asset**: a GUID, or an asset's file stem. The cook resolves it
    /// and pulls it into the pack.
    Asset,
    /// Names a **gameplay id** defined by the program itself (an item id, an
    /// input action, an animation state, a dispatched event). Never an asset.
    Id,
    /// Free **text**: a message, a label.
    Text,
    /// A **table** — a whole TOML document carried as one string. Not an asset
    /// today; if one of these grows an asset field, the edge belongs to the
    /// table's own parser and this row is where to start.
    Table,
}

/// Every `Str` input port in the node kit, and what its string means.
///
/// `(type_id, port, role)`. Kept sorted by `type_id` for reading; the census arm
/// treats it as a set.
///
/// **A row here is a decision, not a description.** Adding a verb with a `Str`
/// input and no row fails `every_string_port_in_the_node_kit_is_classified`,
/// which is the whole reason the table is spelled out instead of derived from a
/// naming convention (a convention would agree with any new port by
/// construction — the shape of gate this repository keeps having to repair).
pub const STR_PORTS: &[(&str, &str, StrRole)] = &[
    ("anim.consume_notify", "name", StrRole::Id),
    ("anim.query_state", "name", StrRole::Id),
    ("anim.set_param", "name", StrRole::Id),
    ("anim.set_trigger", "name", StrRole::Id),
    ("debug.print", "message", StrRole::Text),
    ("dispatch.bind", "handler", StrRole::Id),
    ("dispatch.bind", "name", StrRole::Id),
    ("dispatch.call", "name", StrRole::Id),
    ("dispatch.unbind", "handler", StrRole::Id),
    ("dispatch.unbind", "name", StrRole::Id),
    ("door.spawn", "toml", StrRole::Table),
    // **The one asset-naming port in the kit today.** "Spawn Prefab" takes the
    // name of a thing to instantiate, which is what an asset reference is.
    ("engine.spawn", "prefab", StrRole::Asset),
    ("input.is_down", "key", StrRole::Id),
    ("input.just_pressed", "key", StrRole::Id),
    ("item.count", "id", StrRole::Id),
    ("item.define", "toml", StrRole::Table),
    ("item.equip", "id", StrRole::Id),
    ("item.give", "id", StrRole::Id),
    ("item.spawn_pickup", "id", StrRole::Id),
    ("sky.set_weather", "preset", StrRole::Id),
];

/// One asset a program names, with the place it named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetRef {
    /// The node type that named it (`engine.spawn`).
    pub type_id: String,
    /// The port it sat on (`prefab`).
    pub port: String,
    /// The literal string, exactly as written.
    pub name: String,
}

/// The [`StrRole`] of a node's port, or `None` if the pair is not in the table.
pub fn str_role(type_id: &str, port: &str) -> Option<StrRole> {
    STR_PORTS
        .iter()
        .find(|(t, p, _)| *t == type_id && *p == port)
        .map(|(_, _, r)| *r)
}

/// The non-exec, non-param-pin input port names of a node, in the order the
/// lowerer builds arguments in (`lower::build_call`) and the order the parser
/// checks arity in (`inf_script::verbs`). Empty for an unregistered type.
fn data_input_ports(reg: &inf_graph::NodeRegistry, type_id: &str) -> Vec<String> {
    reg.get(type_id)
        .map(|def| {
            def.inputs
                .iter()
                .filter(|p| !p.ty.is_exec() && !p.param_pin)
                .map(|p| p.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

/// Every asset a class names, in program order, deduplicated by name.
///
/// The cook resolves each against the asset database; an unresolvable one is an
/// **advisory**, because a name is a human's spelling and a typo must say so
/// rather than silently ship a level with nothing behind it.
pub fn asset_refs(class: &BlueprintClass) -> Vec<AssetRef> {
    let reg = crate::nodekit::blueprint_registry();
    let mut out: Vec<AssetRef> = Vec::new();
    let mut push = |r: AssetRef| {
        if !out
            .iter()
            .any(|x| x.name == r.name && x.type_id == r.type_id)
        {
            out.push(r);
        }
    };
    for binding in &class.events {
        walk_fn(&reg, &binding.body, &mut push);
    }
    for f in &class.functions {
        walk_fn(&reg, f, &mut push);
    }
    out
}

/// The same walk over one handler — what the graph↔text bridge holds.
pub fn asset_refs_in_fn(f: &BlueprintFn) -> Vec<AssetRef> {
    let reg = crate::nodekit::blueprint_registry();
    let mut out: Vec<AssetRef> = Vec::new();
    walk_fn(&reg, f, &mut |r: AssetRef| {
        if !out
            .iter()
            .any(|x| x.name == r.name && x.type_id == r.type_id)
        {
            out.push(r);
        }
    });
    out
}

fn walk_fn(reg: &inf_graph::NodeRegistry, f: &BlueprintFn, push: &mut impl FnMut(AssetRef)) {
    walk_block(reg, &f.body, push);
}

fn walk_block(reg: &inf_graph::NodeRegistry, body: &[Stmt], push: &mut impl FnMut(AssetRef)) {
    for s in body {
        match s {
            Stmt::Let { value, .. } => walk_expr(reg, value, push),
            Stmt::Assign { value, .. } => walk_expr(reg, value, push),
            Stmt::ExprStmt(e) => walk_expr(reg, e, push),
            Stmt::Return(Some(e)) => walk_expr(reg, e, push),
            Stmt::Return(None) => {}
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                walk_expr(reg, cond, push);
                walk_block(reg, then_body, push);
                walk_block(reg, else_body, push);
            }
            Stmt::While { cond, body } => {
                walk_expr(reg, cond, push);
                walk_block(reg, body, push);
            }
            // A hand-written Rust snippet is opaque by construction — the
            // transpiler's own contract. A snippet that names an asset is
            // invisible here, and that is the same bound `lift` has.
            Stmt::Snippet(_) => {}
        }
    }
}

fn walk_expr(reg: &inf_graph::NodeRegistry, e: &Expr, push: &mut impl FnMut(AssetRef)) {
    match e {
        Expr::Call { path, args } => {
            let type_id = crate::lower::node_type_of_path(path);
            let ports = data_input_ports(reg, &type_id);
            for (i, arg) in args.iter().enumerate() {
                walk_expr(reg, arg, push);
                let Some(port) = ports.get(i) else { continue };
                if str_role(&type_id, port) != Some(StrRole::Asset) {
                    continue;
                }
                if let Expr::Lit(Lit::Str(name)) = arg {
                    push(AssetRef {
                        type_id: type_id.clone(),
                        port: port.clone(),
                        name: name.clone(),
                    });
                }
            }
        }
        Expr::Binary(_, a, b) => {
            walk_expr(reg, a, push);
            walk_expr(reg, b, push);
        }
        Expr::Unary(_, a) => walk_expr(reg, a, push),
        Expr::Lit(_) | Expr::Local(_) | Expr::Param(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::{EventBinding, EventKind};
    use crate::{Param, Ty};

    /// **The table covers the node kit, and nothing more.**
    ///
    /// Both directions on purpose: a `Str` port with no row would be an
    /// unclassified string the cook silently ignores, and a row naming a port
    /// that no longer exists is a decision about nothing that reads as coverage.
    #[test]
    fn every_string_port_in_the_node_kit_is_classified() {
        let reg = crate::nodekit::blueprint_registry();
        let mut found = 0usize;
        let mut missing: Vec<String> = Vec::new();
        for def in reg.ordered() {
            for p in &def.inputs {
                if p.ty != inf_graph::PortType::Str || p.param_pin {
                    continue;
                }
                found += 1;
                if str_role(&def.type_id, &p.name).is_none() {
                    missing.push(format!("{}.{}", def.type_id, p.name));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "unclassified string ports — every one is a string the cook cannot \
             know the meaning of, so decide each in `STR_PORTS`: {missing:?}"
        );
        assert_eq!(
            found,
            STR_PORTS.len(),
            "the table has {} rows and the node kit has {found} string ports — a \
             row naming a port that no longer exists reads as coverage and is not",
            STR_PORTS.len()
        );
        // Anti-vacuity: a kit with no string ports would satisfy both.
        assert!(found >= 20, "only {found} string ports");
        let assets = STR_PORTS
            .iter()
            .filter(|(_, _, r)| *r == StrRole::Asset)
            .count();
        assert_eq!(
            assets, 1,
            "exactly one asset-naming port today (`engine.spawn`'s prefab); when \
             that changes, this number and the cook's advisory move together"
        );
    }

    fn call(path: &[&str], args: Vec<Expr>) -> Expr {
        Expr::Call {
            path: path.iter().map(|s| (*s).to_string()).collect(),
            args,
        }
    }

    fn s(v: &str) -> Expr {
        Expr::Lit(Lit::Str(v.into()))
    }

    /// The walk finds an asset name through a branch, a loop and a nested
    /// expression, and does **not** report the gameplay ids beside it.
    #[test]
    fn the_walk_finds_asset_names_and_leaves_gameplay_ids_alone() {
        let body = vec![
            Stmt::ExprStmt(call(&["debug", "print"], vec![s("hello")])),
            Stmt::ExprStmt(call(
                &["item", "give"],
                vec![s("rifle"), Expr::Lit(Lit::Int(1))],
            )),
            Stmt::If {
                cond: Expr::Lit(Lit::Bool(true)),
                then_body: vec![Stmt::ExprStmt(call(&["engine", "spawn"], vec![s("Enemy")]))],
                else_body: vec![Stmt::While {
                    cond: Expr::Lit(Lit::Bool(false)),
                    body: vec![Stmt::ExprStmt(call(&["engine", "spawn"], vec![s("Crate")]))],
                }],
            },
            // Twice, so the dedupe is exercised rather than assumed.
            Stmt::ExprStmt(call(&["engine", "spawn"], vec![s("Enemy")])),
        ];
        let mut class = BlueprintClass::new("c", "C");
        class.events.push(EventBinding {
            event: EventKind::BeginPlay,
            body: BlueprintFn {
                id: "begin_play".into(),
                name: "begin_play".into(),
                params: vec![],
                ret: Ty::Unit,
                body,
            },
        });
        let refs = asset_refs(&class);
        let names: Vec<&str> = refs.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["Enemy", "Crate"], "{refs:?}");
        assert!(refs.iter().all(|r| r.type_id == "engine.spawn"));
        assert!(refs.iter().all(|r| r.port == "prefab"));
    }

    /// A non-literal argument is not a name. `engine.spawn(some_local)` names an
    /// asset the cook cannot know, and reporting the wrong one would be worse
    /// than reporting none.
    #[test]
    fn a_computed_prefab_name_is_not_reported() {
        let mut class = BlueprintClass::new("c", "C");
        class.events.push(EventBinding {
            event: EventKind::Tick,
            body: BlueprintFn {
                id: "tick".into(),
                name: "tick".into(),
                params: vec![Param {
                    name: "dt".into(),
                    ty: Ty::Float,
                }],
                ret: Ty::Unit,
                body: vec![Stmt::ExprStmt(call(
                    &["engine", "spawn"],
                    vec![Expr::Param("dt".into())],
                ))],
            },
        });
        assert!(asset_refs(&class).is_empty());
    }
}
