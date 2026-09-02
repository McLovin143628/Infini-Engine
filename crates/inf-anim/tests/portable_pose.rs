//! **The pose pipeline calls no `std` transcendental** (P24.2 audit M-SLERP).
//!
//! A grep, because the property is about *what the code is allowed to call* and
//! no numeric assertion in a single-process test can see it. The claim it
//! defends: an evaluated pose is folded into the sim's `state_bytes`, and
//! `state_bytes` is compared between the editor's Simulate and the shipped
//! player — and, through the replay and net paths, between two *machines*. The
//! P14 law says `std` `sin`/`cos`/`acos` are not bit-identical across targets, so
//! a single one of them on this path silently falsifies every determinism claim
//! the engine makes about animation.
//!
//! # It was not hypothetical
//!
//! The P24.2 audit found `Quat::slerp` — `acos_approx` plus three `sin`s — in
//! **both** of the two most-travelled blends in the crate:
//!
//! * `clip::QuatTrack::sample`, on every `Interpolation::Linear` track, which is
//!   the `#[default]` and what the glTF importer emits;
//! * `pose::blend_poses`, on every state-machine transition with a non-zero
//!   duration and in every blend space.
//!
//! Both are now `inf_math::pslerp`. This file is what stops them coming back —
//! and it covers `ik.rs`, whose module docs have claimed a trig-free property
//! since the day it was written with nothing enforcing it.
//!
//! # `sqrt` is deliberately absent from the ban
//!
//! IEEE-754 specifies it exactly, so it is bit-portable. That is why both IK
//! solvers are built out of it.

/// Files whose **whole item text** must be free of `std` transcendentals, with
/// the reason each one is on the list.
///
/// # The last two are not in this crate, and that is the point
///
/// P24.2 made `root_delta` portable and put `root_motion.rs` on this list. The
/// *application* of that delta — `DQuat::from_rotation_y(yaw) * local`, which is
/// `sin_cos` inside glam — is one call downstream, in the two fixed steps
/// themselves, and this list did not reach them. So the fix stopped exactly
/// where the gate's vision stopped, and stayed stopped for four phases (L6.F5).
/// The multiply now lives in `root_motion::root_delta_world`, one door for both
/// hosts; these two entries are what stops a future edit re-inlining it.
///
/// `include_str!` across crate boundaries is unusual and deliberate. It creates
/// no Cargo dependency and violates no ring rule — a gate that greps a file is
/// not a consumer of its types — and the alternative is a *second* copy of this
/// ban list somewhere else, which is how a list becomes two lists that disagree.
/// Both files are workspace members whose paths are as stable as this file's own.
const SIM_PATH: [(&str, &str, &str); 44] = [
    // ── the fixed step that PUBLISHES the pose (SK1b audit) ──
    //
    // The most surprising absence on this list. `crates/inf-ecs/src/pose.rs` is
    // the door every writer above reaches `pose_state_bytes` through — it runs
    // the machine, the layer stack, the drive pass, the chain solves, the foot
    // IK, SK1b's hand pass and the correction re-drive, and then folds the
    // result into the bytes both hosts compare. Two of its own crate's files
    // (`camera.rs`, `movement.rs`) were added by the P29.6 audit for exactly
    // this reason and this one was not, so SK1b's `apply_hand_ik` and
    // `solve_arm` landed on an uncovered path.
    //
    // It could not be noticed by `every_source_file_in_this_crate_is_covered_or_
    // ledgered` either: that arm enumerates `crates/inf-anim/src` and this file
    // is in another crate, which is the SK1a-audit lesson (a hand-maintained
    // list needs an arm that enumerates its domain) meeting a domain with two
    // halves.
    (
        "inf_ecs::pose",
        include_str!("../../inf-ecs/src/pose.rs"),
        "step_pose_evaluation is the door every pose writer reaches `pose_state_bytes` through, and both hosts compare those bytes",
    ),
    // ── VEN1b's postures ──
    //
    // A sit and a dance are poses. They are composed onto the machine's own
    // result inside `step_pose_evaluation`, so they land in `pose_state_bytes`
    // by the same route every writer above them does — and the club gate
    // compares those bytes between the editor's PIE and the shipped player.
    (
        "posture.rs",
        include_str!("../src/posture.rs"),
        "a seated or dancing body is a pose, folded into pose_state_bytes and compared between the two hosts",
    ),
    // ── SK1b's hand solver ──
    //
    // A curled finger is a pose. It reaches `pose_state_bytes` through the same
    // fixed step every other writer here does, and the wave's own grip gate
    // compares it between two processes.
    (
        "grip.rs",
        include_str!("../src/grip.rs"),
        "a finger curl is a pose, folded into pose_state_bytes and compared between the editor's PIE and the shipped player",
    ),
    // ── P29.6's text form, for the same reason the two below it are here ──
    //
    // It is not on the *runtime* path at all — it is an authoring door — and it
    // is on this list because of what it writes: the committed `.inf_sm` text
    // sidecar is content, so a value that printed or parsed differently on two
    // targets would fork the file rather than the frame, and every determinism
    // gate downstream would compare each machine with itself and pass.
    (
        "text.rs",
        include_str!("../src/text.rs"),
        "the .inf_sm text form is committed content — a threshold that printed differently on two targets would fork the file, not the frame",
    ),
    // ── P29.5's proposal, for the same reason its derivation is here ──
    (
        "propose.rs",
        include_str!("../src/propose.rs"),
        "a proposal writes a committed .inf_sm — its thresholds are bytes in a file two developers diff",
    ),
    // ── P29.5's derivation, which writes committed bytes ──
    //
    // The strongest case on this list rather than the weakest: everything else
    // here decides a pose at runtime, and this one decides what is *in the file*.
    // A `sin` here would not merely disagree between two machines at play time —
    // it would put two different `.inf_anim` byte streams in two developers'
    // checkouts, and every determinism gate downstream would compare them and
    // pass, because each machine would agree with itself.
    (
        "derive.rs",
        include_str!("../src/derive.rs"),
        "an import derivation writes the root-motion track, the markers and the curves INTO a committed .inf_anim, so a non-portable call here forks the content itself",
    ),
    // ── P29.4's three new modules ──
    (
        "warp.rs",
        include_str!("../src/warp.rs"),
        "a warped offset is written straight onto an entity's Transform by the movement step, on both hosts, and is folded into state_bytes",
    ),
    (
        "foot.rs",
        include_str!("../src/foot.rs"),
        "a foot lock decides where a foot IS, which is a pose, and the wave's own gate measures its slide in metres",
    ),
    (
        "ragdoll.rs",
        include_str!("../src/ragdoll.rs"),
        "the ragdoll's blend weights and its get-up choice are a pure function of sim state by doctrine, and the pose they choose rides the trace",
    ),
    (
        "pose.rs",
        include_str!("../src/pose.rs"),
        "blend_poses and skinning_matrices produce the pose that is folded into state_bytes",
    ),
    // ── P29.2's six new modules, plus the one this gate had always missed ──
    //
    // `blend_space.rs` is the "always missed" one: it has driven every 1D and 2D
    // blend since P11.2 and was never on this list, which was survivable only
    // because it delegated all of its arithmetic. P29.2 gave it a triangulation
    // and a marker warp of its own, so the omission stops being survivable.
    (
        "blend_space.rs",
        include_str!("../src/blend_space.rs"),
        "every blend space's weights and per-clip sample times, which choose the poses everything below blends",
    ),
    (
        "channels.rs",
        include_str!("../src/channels.rs"),
        "a `.inf_anim` v2 curve value reaches a blend weight, and a blend weight reaches the pose",
    ),
    (
        "delaunay.rs",
        include_str!("../src/delaunay.rs"),
        "the triangulation decides WHICH samples a 2D blend space weights, so a different answer here is a different pose",
    ),
    (
        "sync.rs",
        include_str!("../src/sync.rs"),
        "a warped sample time is a clip time, and a clip time is a pose",
    ),
    (
        "layers.rs",
        include_str!("../src/layers.rs"),
        "additive composition and the layer stack are the last thing applied to a pose before it is folded into state_bytes",
    ),
    (
        "inertialize.rs",
        include_str!("../src/inertialize.rs"),
        "the quintic decay is the DEFAULT for state transitions, so it is on the pose path of every machine-driven character",
    ),
    (
        "pose_match.rs",
        include_str!("../src/pose_match.rs"),
        "a match chooses the frame a state enters at, which is a play-head, which is committed sim state",
    ),
    (
        "clip.rs",
        include_str!("../src/clip.rs"),
        "QuatTrack::sample is the default interpolation and the most-travelled blend in the engine",
    ),
    (
        "ik.rs",
        include_str!("../src/ik.rs"),
        "IK is a post-pass over that same pose, and its module docs have claimed this property since it was written",
    ),
    (
        "root_motion.rs",
        include_str!("../src/root_motion.rs"),
        "root_delta writes an entity's Transform, from BOTH fixed steps, into state_bytes",
    ),
    (
        "state_machine.rs",
        include_str!("../src/state_machine.rs"),
        "eval_pose chooses and cross-fades the poses the two above produce",
    ),
    (
        "cloth.rs",
        include_str!("../src/cloth.rs"),
        "the XPBD solver's particle positions are folded into state_bytes beside the pose, and its whole numerical vocabulary was chosen to be sqrt-only for exactly this reason (see its module docs on why bending is a cross spring rather than a dihedral angle)",
    ),
    (
        "hair.rs",
        include_str!("../src/hair.rs"),
        "strand positions ride state_bytes beside the cloth's, and the ribbon builder runs in the projector on both hosts — a cross product and a length, and nothing that is not one of those",
    ),
    (
        "locomotion.rs",
        include_str!("../src/locomotion.rs"),
        "the P24.5 generator writes the KEYFRAMES of a committed `.inf_anim`, which is a strictly stronger claim than the rest of this list: everything above produces values that ride state_bytes for one session, and this produces bytes that go on disk, into a cook, into a pack, and are compared by a golden. `Quat::from_rotation_x` is `f32::sin_cos`, so the whole clip is written out through the half-angle identity by hand",
    ),
    (
        "drive.rs",
        include_str!("../src/drive.rs"),
        "SK1a's procedural drive writes twist and IK-handle transforms straight into the pose the layer stack produced, one call above the IK this list already certifies, and every one of them is folded into pose_state_bytes and compared between the editor's PIE and the shipped player",
    ),
    (
        "manny.rs",
        include_str!("../src/manny.rs"),
        "the 161-bone generator writes the BIND POSE of a committed `.inf_skel` — locomotion.rs's stronger claim, applied to the rig those keyframes are indexed against, and the reason its arms are a T-pose of pure axis offsets rather than the A-pose the shipped mannequin stands in",
    ),
    (
        "roles.rs",
        include_str!("../src/roles.rs"),
        "the role table decides which joints the drive, the ragdoll and the weight solver touch at all, so a transcendental here would move which bones move",
    ),
    // ── SK1a audit: the five files in this crate's own `src` that were on no
    //    list at all ──
    //
    // The completeness arm below is what found them; each writes a pose or a
    // bind pose and each was one edit away from a silent escape.
    (
        "template.rs",
        include_str!("../src/template.rs"),
        "the generator writes the BIND POSE of every committed `.inf_skel` this engine makes, manny.rs's claim applied to the other four body plans — and it is where the pure-translation law that keeps an inverse bind an exact negation is actually spelled",
    ),
    (
        "merge.rs",
        include_str!("../src/merge.rs"),
        "modular rigging writes the bind pose of the ASSEMBLED rig, which is what a weight table and an IK chain are then indexed against; the merged asset is committed content exactly as its parts were",
    ),
    (
        "retarget.rs",
        include_str!("../src/retarget.rs"),
        "the retarget writes local rotations into a destination pose bind-relatively — a pose that is folded into state_bytes like any other, and the one path in the crate where two DIFFERENT rigs' binds are composed",
    ),
    (
        "skeleton.rs",
        include_str!("../src/skeleton.rs"),
        "`JointTransform` and the inverse-bind arithmetic every pose in the crate is built out of; a transcendental at this level would be under all of the above at once",
    ),
    (
        "sockets.rs",
        include_str!("../src/sockets.rs"),
        "`socket_transforms` composes a socket's world frame out of the posed globals, and an attachment's transform rides state_bytes through `inf_ecs::attach`",
    ),
    (
        "inf_player::runtime_sim",
        include_str!("../../../runtime/inf-player/src/runtime_sim.rs"),
        "the SHIPPED fixed step: everything it writes into an entity's Transform is folded into state_bytes, and `apply_root_motion` was rotating the root delta with `DQuat::from_rotation_y` one call below a `root_delta` this gate had already certified",
    ),
    (
        "inf_editor_core::simulate",
        include_str!("../../../editor/crates/inf-editor-core/src/simulate.rs"),
        "the EDITOR fixed step, and the other half of every `PIE == shipping` claim in the repository — the two are compared byte for byte, so a libm call in either one is a libm call in the comparison",
    ),
    // ── P29.6 audit, A2: where the locomotion camera's trigonometry ACTUALLY is ──
    //
    // The wave put `inf_physics::d3::camera` on the *other* portable gate
    // (`inf-physics/tests/portable_character.rs`) with a note saying the camera
    // trace is compared byte for byte. True — and that file contains not one
    // trig call. Every sine the camera evaluates is in these two: `basis` builds
    // the yaw/pitch frame, and `axis_independent_lag` delegates its whole
    // rotation to `movement::rotate_into_frame`/`rotate_from_frame`. Neither
    // file was covered by any gate in the tree, so a `.sin()` added to either
    // would have failed nothing — and `phase29_gate`'s determinism arm compares
    // two runs on ONE machine with one libm, so it cannot see a portability
    // defect either. This repository's own law, met again: a gate must aim at
    // the thing it names.
    (
        "inf_ecs::camera",
        include_str!("../../inf-ecs/src/camera.rs"),
        "`basis` and `axis_independent_lag` are where the camera's sines are, and `LocomotionCamera::trace_bytes` is compared byte for byte across runs by `phase29_gate`",
    ),
    (
        "inf_ecs::movement",
        include_str!("../../inf-ecs/src/movement.rs"),
        "the movement MODEL: `rotate_into_frame`, the quadrant hysteresis and the gait ladder — every one of them folded into `state_bytes` by the fixed step above, and the camera's lag rotates through the first of them",
    ),
    // ── SK1c: the hand pass's PRODUCER ──
    //
    // `inf_physics::d3::gameplay` composes every character's `HandIk` request
    // and computes the point an aiming character brings its weapon up to. That
    // point is converted into model space inside `apply_hand_ik` and lands in
    // the solved pose, so it is folded into `pose_state_bytes` and compared
    // between two hosts and — through the replay and net paths — two machines.
    //
    // The SK1b audit added `inf_ecs::pose` for exactly this reason and recorded
    // the shape as a law: a completeness arm covers the directory it
    // enumerates, so a new file on the pose path in a *third* crate has to be
    // named here or nothing walks it. The direction itself goes through
    // `inf_ecs::weapon::aim_forward`, which is `psin64`/`pcos64`; this entry is
    // what stops the next edit inlining a `sin`.
    (
        "inf_physics::d3::gameplay",
        include_str!("../../inf-physics/src/d3/gameplay.rs"),
        "step_hand_ik composes the HandIk request every posed character's hands are solved against, and those bones are folded into pose_state_bytes",
    ),
    // ── NPC1a: the crowd, named on DAY ONE ──
    //
    // Every previous entry in this list was added after the module it covers
    // existed, and two of them were added because an audit went looking. This
    // one arrives with its module, because the module's output is the strongest
    // possible case for the law: `CrowdRoute::position_at` decides where a
    // thousand NPCs stand, that lands on a `Transform`, and the transform is
    // folded into `state_bytes` and compared between two hosts and — through
    // the replay and net paths — two machines. The route is deliberately built
    // out of `+ - * / sqrt %` alone for exactly this reason; the entry is what
    // stops the next edit reaching for `.atan2(` to make an agent face where it
    // is going.
    (
        "inf_ecs::crowd",
        include_str!("../../inf-ecs/src/crowd.rs"),
        "CrowdRoute::position_at decides where every crowd NPC stands and CrowdBand::tier decides what it costs; both reach a Transform, which state_bytes folds and two hosts compare",
    ),
    // ── NPC1c: the nav layer, all three of its files, on DAY ONE ──
    //
    // NPC1a's crowd entry arrived with its module and this is the same
    // discipline one wave on. The reason is stronger here, not weaker: a route
    // is the thing an agent's `Transform` is READ OFF, so every metre `inf-nav`
    // computes is a metre `state_bytes` folds — and unlike a pose, a route is
    // also compared BETWEEN the two graphs two hosts derived separately. Three
    // entries rather than one because the ban reads a file, and the arithmetic
    // is spread over three of them: the graph builds the costs, the path
    // interpolates along them and the search orders them.
    (
        "inf_nav::graph",
        include_str!("../../inf-nav/src/graph.rs"),
        "polyline_length is the cost every route is ordered by, and `nearest` is how an agent finds the node it starts from",
    ),
    (
        "inf_nav::path",
        include_str!("../../inf-nav/src/path.rs"),
        "position_at is where a Far agent's Transform comes from — one interpolation between two metres — and `project` is what a demotion re-phases against",
    ),
    (
        "inf_nav::route",
        include_str!("../../inf-nav/src/route.rs"),
        "the Dijkstra whose tie-break makes two hosts pick the same one of a grid's six equally short routes",
    ),
    // ── VEH2b: the lane, on DAY ONE, for the same reason its three siblings are
    //    here. A lane centreline is a spine offset sideways by a normal, and the
    //    normal is a ROTATION BY NINETY DEGREES — two component swaps and a
    //    negation rather than a sine, which is the whole reason this file can
    //    hold the module at all. The metres it produces become a traffic car's
    //    `Transform` and therefore the replay trace, so the law binds exactly as
    //    it binds on `position_at` one file over.
    (
        "inf_nav::lane",
        include_str!("../../inf-nav/src/lane.rs"),
        "offset_path's mitre is a reciprocal of a dot product and never a cosine, and the metres it lays down are the ones a traffic car's Transform is read off",
    ),
    // ── NPC1d: the society, BOTH halves, on DAY ONE ──
    //
    // The third wave running to name its modules with them, and the case is the
    // same one NPC1a made: a slot's metres become a nav node's position, the
    // node becomes a leg of a schedule, and the leg is where a scheduled agent's
    // `Transform` is read off. `inf_pcg`'s half decides WHERE a room's people
    // stand, `inf_ecs`'s decides where the pavement between two blocks runs and
    // which workplace is nearest — and a nearest-of is a comparison between two
    // distances, which is exactly the arithmetic the law is about.
    (
        "inf_pcg::building::society",
        include_str!("../../inf-pcg/src/building/society.rs"),
        "slots_of puts a room's people on the room's own nav node, and that node's metres become a schedule leg an agent's Transform is read off",
    ),
    (
        "inf_ecs::society",
        include_str!("../../inf-ecs/src/society.rs"),
        "the pavement ring's corners, the nearest-workplace pairing and the frontage join are all distances, and every one of them decides a metre a scheduled agent stands on",
    ),
];

/// Every `std` transcendental that is not bit-portable across targets.
///
/// `.sqrt()` is **not** here, on purpose: IEEE-754 specifies it exactly. Neither
/// is `.abs()`, `.floor()` or `.clamp()`, for the same reason. `.cbrt()` IS here
/// — the P22.4 widening: on `wasm32` the standard library routes it through the
/// `libm` crate, so a browser client and a native one differ by an ulp.
/// The canonical libm ban, **derived rather than hand-copied** (Wave-G audit).
///
/// This was a 36-entry literal restating `inf_math::libm_ban::METHODS` +
/// `UFCS`, kept honest only by `this_gate_bans_everything_the_canonical_list_does`
/// noticing when the two drifted apart. When the audit added the logarithm and
/// hyperbolic family to the canonical list, this copy went red — which is the
/// meta-arm working, and also the seventh demonstration that a copy is a copy.
/// Reading the list instead of restating it makes the drift impossible rather
/// than merely detectable, which is what R2.B's "one list, not six" asked for.
fn banned_calls() -> Vec<&'static str> {
    inf_math::libm_ban::METHODS
        .iter()
        .chain(inf_math::libm_ban::UFCS.iter())
        .copied()
        .collect()
}

/// glam constructors that reach `sin_cos` **inside another crate**, where no grep
/// of this one would ever see them.
///
/// This half of the list is the one that has actually caught things: P23.5 found
/// `DQuat::from_axis_angle` doing it, `ik::rotation_between` exists because
/// `Quat::from_rotation_arc`'s antiparallel branch does it, and M-SLERP found
/// `Quat::slerp` doing it twice.
const BANNED_GLAM: [&str; 8] = [
    ".slerp(",
    "from_rotation_arc",
    "from_axis_angle",
    "from_euler",
    "from_rotation_x(",
    "from_rotation_y(",
    "from_rotation_z(",
    "to_euler(",
];

/// Source with every `//` comment line blanked, and the `#[cfg(test)]` tail cut
/// off.
///
/// Two exclusions, both load-bearing:
///
/// * **comments** — the module docs of these files name `sin`, `slerp` and
///   `from_rotation_arc` repeatedly *while explaining why they are banned*, which
///   is the sentence such a gate is most likely to be written next to (the P24.1
///   F1 finding, in its general form);
/// * **`#[cfg(test)]`** — a fixture may build a rotation from an angle. It is not
///   on the sim path; it is the thing that *checks* the sim path, and forcing
///   `pslerp` into a test fixture would make the fixture agree with the code
///   under test by construction.
fn production_code(src: &str) -> String {
    // Carriage returns are FILTERED OUT rather than rewritten as a pair. The
    // P22 law only needs them gone before a newline-delimited search, and a
    // char filter carries no escape sequence for a scripted edit to mangle.
    let src: String = src.chars().filter(|c| *c != '\r').collect();
    // **Every** `#[cfg(test)]` region, not just the tail after the first one
    // (P24.2 re-audit minor 2). Cutting at the first was sound for these five
    // files -- each has one test module, last -- and that is a property of
    // today's files rather than of the rule, which is exactly the shape of
    // gate this repository keeps having to repair. Worse than unsound: an
    // EARLY test module would discard every production line after it, and the
    // ban would then scan almost nothing while still passing.
    //
    // A module is skipped from its attribute to the closing brace at ITS OWN
    // indentation, so a nested one inside an `impl` is handled too.
    //
    // **A brace-block item and a one-line item are told apart first** (P24.2
    // micro-round). `#[cfg(test)]` also decorates `use super::*;`, a `const`
    // fixture or a `mod tests;` declaration -- items with no block at all. Hunting
    // a closing brace from one of those runs on until it finds the NEXT item's
    // closing brace and discards every production line in between, which is a
    // silent shrink of what the ban scans: the gate would still pass, on less. So
    // the item's first line decides. A line ending in `{` opens a block and the
    // brace hunt runs; a line ending in `;` is the whole item and only it is cut.
    // Trailing `//` comments are cut off before that reading, since `mod tests {
    // // why` is otherwise classified as neither.
    let lines: Vec<&str> = src.lines().collect();
    let mut keep = vec![true; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        if lines[i].trim_start().starts_with("#[cfg(test)]") {
            let indent = lines[i].len() - lines[i].trim_start().len();
            let close = format!("{}}}", " ".repeat(indent));
            keep[i] = false;
            let mut j = i + 1;
            let mut opens_a_block = false;
            while j < lines.len() {
                keep[j] = false;
                let head = lines[j].split("//").next().unwrap_or("").trim_end();
                if head.ends_with('{') {
                    opens_a_block = true;
                    break;
                }
                if head.ends_with(';') {
                    break;
                }
                j += 1;
            }
            if opens_a_block {
                while j < lines.len() {
                    keep[j] = false;
                    if lines[j] == close {
                        break;
                    }
                    j += 1;
                }
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    lines
        .iter()
        .enumerate()
        .map(|(n, l)| {
            if !keep[n] || l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// **What this gate deliberately does NOT cover, and why** (P24.2 re-audit F1b).
///
/// `inf_ecs::components`' `GlobalTransform` propagation calls `Quat::from_euler`
/// / `to_euler` (components.rs:102 and :111) on every entity transform, and that
/// reaches `sin_cos` and `atan2` inside glam exactly as `to_euler` did in
/// `root_motion`. It is **not** fixed here, and the reason is structural rather
/// than an oversight:
///
/// * it is the **euler-degrees Transform doctrine** settled in Phase 3 -- the
///   authoring convention every entity transform in the engine has flowed through
///   since scene v1;
/// * re-plumbing it means changing what a `Transform` *is*, which is a schema and
///   an authoring decision, not a fix-round item.
///
/// The consequence, stated rather than glossed: **same-platform traces are
/// unaffected** (libm is deterministic per platform, which is what every gate in
/// this repository actually compares), and **cross-platform trace portability
/// would require revisiting the euler-conversion doctrine wholesale**. Written
/// down in ROADMAP section 12's P24 block so it is a decision rather than a gap.
const LEDGERED_EXCLUSIONS: [(&str, &str); 3] = [
    (
        "inf_ecs::components (GlobalTransform propagation)",
        "the Phase-3 euler-degrees Transform doctrine; see ROADMAP section 12",
    ),
    // ── SK1a audit: the two files in this crate's own `src` that really are
    //    not on the pose path, named rather than merely absent ──
    (
        "lib.rs",
        "a module list and a re-export list; it contains no arithmetic at all, and the completeness arm below would otherwise be satisfied by adding it to the pose path, which is a worse lie than an honest exclusion",
    ),
    (
        "asset.rs",
        "the bincode payload ladder: schema stamps, migration rungs and index bounds checks. It moves no value a pose is made of, and the numbers it does touch are u16 joint indices",
    ),
];

#[test]
fn the_ledgered_exclusions_are_named_rather_than_forgotten() {
    // A list this gate is *allowed* not to cover has to be short and reasoned; an
    // empty one would mean the gate claims total coverage, which it does not.
    //
    // **`>=`, not `==`** (SK1a audit): the arm pinned the length at 1, so the
    // only way to record a newly-found gap honestly was to make this red. A
    // ledger nobody can add to is a ledger that stops being written.
    assert!(
        (1..=8).contains(&LEDGERED_EXCLUSIONS.len()),
        "a list of things a gate does not cover has to stay short enough to read"
    );
    for (what, why) in LEDGERED_EXCLUSIONS {
        assert!(!what.is_empty() && why.len() > 20, "{what}: {why}");
    }
}

/// **The gate can see its own directory** (SK1a audit).
///
/// `SIM_PATH` is a hand-maintained enumeration, so until this arm existed a new
/// file in `crates/inf-anim/src` joined nothing: it was covered by no ban, named
/// on no ledger, and no arm could tell the difference between "deliberately out
/// of scope" and "nobody thought of it". Measured when this was written: **seven**
/// of the crate's thirty-one source files were on neither list, and three of them
/// (`retarget.rs`, `template.rs`, `merge.rs`) write poses or bind poses. None was
/// a live escape — which is exactly the shape `inf_math::libm_ban`'s own header
/// warns about, a list that enumerates what somebody thought of and passes for
/// years.
///
/// This is the P22 law at a gate: **a ban enumerates what you thought of, an
/// allowlist what is allowed.** Every `.rs` under this crate's `src` must be on
/// one list or the other, by name.
#[test]
fn every_source_file_in_this_crate_is_covered_or_ledgered() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files: Vec<String> = std::fs::read_dir(&src)
        .expect("this crate has a src directory")
        .filter_map(|e| {
            let p = e.expect("dir entry").path();
            (p.extension().and_then(|x| x.to_str()) == Some("rs"))
                .then(|| p.file_name()?.to_str().map(str::to_string))
                .flatten()
        })
        .collect();
    files.sort();
    assert!(
        files.len() >= 25,
        "the sweep found {} files under {}; it is not looking where it thinks it is",
        files.len(),
        src.display()
    );
    let covered: Vec<&str> = SIM_PATH.iter().map(|(n, _, _)| *n).collect();
    let ledgered: Vec<&str> = LEDGERED_EXCLUSIONS.iter().map(|(n, _)| *n).collect();
    let missing: Vec<&String> = files
        .iter()
        .filter(|f| !covered.contains(&f.as_str()) && !ledgered.contains(&f.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "{missing:?} are on neither `SIM_PATH` nor `LEDGERED_EXCLUSIONS` — a file \
         in this crate is either on the pose path and banned from `std` \
         transcendentals, or it is not and says why. Add it to one of the two."
    );
    println!(
        "{} source files: {} on the pose path, {} ledgered out",
        files.len(),
        files.len() - ledgered.len(),
        ledgered.len()
    );
}

#[test]
fn the_animation_blend_uses_no_platform_dependent_trigonometry() {
    for (name, src, why) in SIM_PATH {
        let code = production_code(src);
        for banned in banned_calls().iter().chain(BANNED_GLAM.iter()) {
            let hits: Vec<usize> = code
                .lines()
                .enumerate()
                .filter(|(_, l)| l.contains(banned))
                .map(|(i, _)| i + 1)
                .collect();
            assert!(
                hits.is_empty(),
                "{name} calls `{banned}` at line(s) {hits:?} — {why}, and `std` \
                 transcendentals are not bit-identical across targets (the P14 \
                 law). Use `inf_math`'s portable family: `pslerp`, `psin64`, \
                 `pcos64`, `pacos64`. `sqrt` is fine and is deliberately not \
                 banned."
            );
        }
    }
}

/// **NOT VACUOUS**: the files really are on the list, they really do contain
/// code, and the ban really does catch the thing it was written for.
#[test]
fn the_trig_ban_is_looking_at_real_code() {
    for (name, src, _) in SIM_PATH {
        let code = production_code(src);
        assert!(
            code.lines().filter(|l| !l.trim().is_empty()).count() > 40,
            "{name} reduced to nothing after stripping comments and tests — the \
             ban above is scanning an empty string"
        );
        // The pose path is reached, not merely present.
        assert!(
            code.contains("pub fn") || code.contains("pub(crate) fn"),
            "{name} exposes nothing"
        );
    }
    // The portable replacement really is used where the banned one was.
    //
    // **By NAME, not by index.** This used to read `SIM_PATH[0]` and `SIM_PATH[1]`
    // and it broke the moment P29.2 inserted a file between them — a positional
    // reference into a list whose whole purpose is to grow. A lookup that cannot
    // find its file fails loudly, which is the behaviour an index silently traded
    // away for asserting the wrong file's contents.
    let by_name = |want: &str| -> String {
        let (_, src, _) = SIM_PATH
            .iter()
            .find(|(n, _, _)| *n == want)
            .unwrap_or_else(|| panic!("{want} is not on the sim-path list"));
        production_code(src)
    };
    assert!(
        by_name("pose.rs").contains("inf_math::pslerp"),
        "pose.rs no longer blends through the portable arc"
    );
    assert!(
        by_name("clip.rs").contains("inf_math::pslerp"),
        "clip.rs no longer blends through the portable arc"
    );
    assert!(
        by_name("layers.rs").contains("inf_math::pslerp"),
        "layers.rs no longer scales its additive rotations through the portable arc"
    );
    // And the ban is a real filter: it fires on a string that contains one.
    let decoy = "let q = a.slerp(b, 0.5);";
    assert!(
        BANNED_GLAM.iter().any(|b| decoy.contains(b)),
        "the ban list would not catch a literal `slerp` call"
    );
}

/// **Both spellings of every banned function, at both widths** — the structural
/// property the list claims, rather than a count somebody kept in their head.
///
/// Six functions had a method ban and no UFCS twin until this test was written,
/// which is how the list was *sampled* rather than ruled. Adding a method form
/// without its twins now fails here instead of failing silently in five years.
#[test]
fn the_ban_covers_both_spellings_of_every_function() {
    let calls = banned_calls();
    let methods: Vec<&str> = calls
        .iter()
        .copied()
        .filter(|b| b.starts_with('.'))
        .collect();
    assert_eq!(
        methods.len(),
        inf_math::libm_ban::METHODS.len(),
        "the method half of the ban is no longer the canonical set: {methods:?}"
    );
    for m in &methods {
        let name = m.trim_start_matches('.').trim_end_matches(['(', ')']);
        for width in ["f32", "f64"] {
            let ufcs = format!("{width}::{name}(");
            assert!(
                calls.contains(&ufcs.as_str()),
                "`{m}` is banned as a method and `{ufcs}` is not banned at all — \
                 the same call written the other way walks straight past this \
                 gate, which is exactly the P24.2 minor-3 finding"
            );
        }
    }
    // …and there is nothing else in the list: each method contributes itself
    // plus its two UFCS twins.
    assert_eq!(calls.len(), methods.len() * 3);
}

/// **`#[cfg(test)]` on a one-line item cuts that item, not the rest of the file.**
///
/// The stripper hunts a closing brace at the attribute's own indentation. Applied
/// to `#[cfg(test)] use super::*;` — an item with no block — that hunt used to run
/// on to the next item's brace and discard every production line in between. The
/// gate would have stayed green while scanning less, which is the failure mode
/// this whole file exists to prevent.
#[test]
fn a_test_gated_one_line_item_does_not_swallow_the_code_after_it() {
    let src = "\
#[cfg(test)]
use super::*;

pub fn on_the_sim_path(q: DQuat) -> f64 {
    q.to_euler(EulerRot::YXZ).0
}

#[cfg(test)]
mod tests {
    fn fixture() -> f64 {
        (1.0f64).sin()
    }
}

pub fn also_on_it(x: f64) -> f64 {
    x.sqrt()
}
";
    let code = production_code(src);
    assert!(
        code.contains("q.to_euler(EulerRot::YXZ)"),
        "the production violation between the two test regions was stripped away \
         with them — the ban is scanning less than the file: {code:?}"
    );
    assert!(code.contains("pub fn also_on_it"), "{code:?}");
    // The block-form region is still removed in full, fixture and all.
    assert!(!code.contains("mod tests"), "{code:?}");
    assert!(!code.contains("(1.0f64).sin()"), "{code:?}");
    // …and the `use` line itself really did go.
    assert!(!code.contains("use super::*;"), "{code:?}");
    // The ban therefore fires on the violation, which is the point of all of it.
    assert!(
        BANNED_GLAM.iter().any(|b| code.contains(b)),
        "the surviving violation is invisible to the ban: {code:?}"
    );
}

/// **Round-2 finding R2.B**: this gate's list is a superset of the canonical
/// one, and passes the shared completeness meta-arm.
///
/// Six hand-copies of the libm ban existed and they had diverged — the erosion
/// mirror banned `f64::cos(` over an `f32` module, the fracture gate was
/// missing sixteen UFCS twins including `f64::cbrt(` and every glam entry, the
/// physics gate had no `.atan()`. None was a live escape, which is exactly the
/// problem with a list that enumerates what somebody thought of.
/// `inf_math::libm_ban::ALL` is the one list now; this arm is what keeps this
/// gate tied to it.
#[test]
fn this_gate_bans_everything_the_canonical_list_does() {
    let mine: Vec<&str> = banned_calls()
        .into_iter()
        .chain(BANNED_GLAM.iter().copied())
        .collect();
    inf_math::libm_ban::covers_both_spellings("inf-anim/tests/portable_pose.rs", &mine);
    let missing: Vec<&str> = inf_math::libm_ban::ALL
        .iter()
        .copied()
        .filter(|b| !mine.contains(b))
        .collect();
    assert!(
        missing.is_empty(),
        "{} is missing {missing:?} from `inf_math::libm_ban::ALL` — six copies of this list diverged once already",
        "inf-anim/tests/portable_pose.rs"
    );
}
