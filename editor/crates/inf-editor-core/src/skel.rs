//! **The skeleton editing session** (P24.3) — the Ring-1 half of the Skeleton
//! Editor.
//!
//! # Why this is snapshot-undo and the mesh editor is journal-undo
//!
//! `inf_dcc::MeshSession` is `base + Vec<Op> + cursor` because a mesh is a
//! hundred thousand vertices and snapshotting one per keystroke is not a design.
//! A [`SkeletonAsset`] is *tens of joints* — a biped template is nineteen — so a
//! snapshot is a few kilobytes and the whole apparatus a journal buys (op
//! discriminants frozen on a wire, replay, checkpoint intervals) would be
//! ceremony with nothing behind it. Same reasoning as `LayoutStore`'s, one domain
//! over: pick the representation the data's *size* argues for, not the one the
//! neighbouring module happens to use.
//!
//! The consequence is stated rather than left to be discovered: **a skeleton
//! edit is not replayable from a journal**, so there is no `.inf_skel` session
//! save and nothing to version. The asset on disk is the only durable artefact,
//! which is exactly what [`SkelSession::saved`] tracks.
//!
//! # The inverse binds are DERIVED, always
//!
//! `Joint::inverse_bind` is the matrix that takes a vertex out of mesh space and
//! into the joint's bind-local space; `Joint::local_bind` is the rest transform.
//! They are not independent — the first is the inverse of the *composed* second —
//! and a UI that let an author move a joint without recomputing them would
//! deform every skinned mesh bound to the rig into confetti while looking like it
//! had only moved a bone.
//!
//! So every edit that touches a rest transform runs [`recompute_inverse_binds`],
//! which is pure arithmetic (a parent-order walk composing `Mat4`s and one
//! `Mat4::inverse` per joint — no trigonometry, the P14 law).
//!
//! # Canonical names are WARNINGS, not refusals
//!
//! Renaming `upper_arm_l` to `arm1` is legal — it is the author's rig — and it
//! silently breaks two things: `RetargetMap::humanoid_identity`, which pairs by
//! the nineteen canonical names, and `inf_anim::mirror_joint_map`, which pairs by
//! the `_l`/`_r` suffix. Refusing would make the editor unable to author a
//! non-humanoid; saying nothing would make the breakage discoverable only much
//! later, in a retarget that quietly does nothing. So a rename returns a
//! [`RenameVerdict`] carrying what it cost, and the panel shows it.

use inf_anim::{
    merge_skeletons, mirrored_joint_name, unmatched_sided_joints, BodyParams, BodyPlan, Joint,
    JointLimit, JointTransform, Skeleton, SkeletonAsset, SkeletonMergeError, Socket,
};

/// Why a skeleton edit refused.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum SkelError {
    /// A joint index the skeleton does not have.
    #[error("no joint {index} (the skeleton has {joints})")]
    NoSuchJoint { index: u16, joints: usize },
    /// A socket name the skeleton does not author.
    #[error("no socket named {name:?}")]
    NoSuchSocket { name: String },
    /// A socket name already in use.
    ///
    /// Refused rather than de-duplicated, for `merge_skeletons`' reason:
    /// `find_socket` takes the first match, so two sockets with one name is an
    /// attach point whose meaning depends on authoring order.
    #[error("a socket named {name:?} already exists")]
    SocketExists { name: String },
    /// An empty or whitespace-only name.
    #[error("{what} cannot be empty")]
    EmptyName { what: &'static str },
    /// A non-finite number reached a transform.
    #[error("{what} is not finite")]
    NonFinite { what: &'static str },
    /// The generator refused to build a template.
    #[error("the template is invalid: {0}")]
    Template(String),
    /// A part could not be merged onto this rig.
    #[error("{0}")]
    Merge(String),
}

impl From<SkeletonMergeError> for SkelError {
    fn from(e: SkeletonMergeError) -> Self {
        SkelError::Merge(e.to_string())
    }
}

/// What a rename cost, beyond the rename.
///
/// Both fields are **warnings**: the rename happened. See the module docs for
/// why neither is a refusal.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenameVerdict {
    /// The joint left the canonical humanoid vocabulary
    /// (`inf_anim::humanoid_joint_names`), so `RetargetMap::humanoid_identity`
    /// will no longer pair it with anything.
    pub left_humanoid_set: bool,
    /// The joint had a left/right twin and the new name does not pair with one,
    /// so `mirror_joint_map` will now map it to itself — a mirror across this
    /// rig weights the copy to the wrong side.
    pub broke_mirror_pair: bool,
    /// The name it used to have, so a panel can offer to put it back.
    pub previous: String,
}

/// One open skeleton document.
///
/// Holds the asset being edited, a snapshot undo stack, and **the bytes last
/// written to disk** — so "is this dirty" is answered by comparing against the
/// file's content and not by a flag somebody has to remember to clear (the
/// `save_mesh_session` disk-truth pattern).
#[derive(Debug, Clone)]
pub struct SkelSession {
    asset: SkeletonAsset,
    saved: SkeletonAsset,
    undo: Vec<SkeletonAsset>,
    redo: Vec<SkeletonAsset>,
}

/// How many edits deep undo goes. The `SceneDoc` history's depth, for the same
/// reason: it is what an author's muscle memory expects, and a snapshot here is
/// kilobytes.
pub const MAX_UNDO: usize = 50;

impl SkelSession {
    /// Open `asset` for editing. It is considered **saved** — this is the state
    /// that came off disk.
    pub fn new(asset: SkeletonAsset) -> Self {
        Self {
            saved: asset.clone(),
            asset,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// The live asset.
    pub fn asset(&self) -> &SkeletonAsset {
        &self.asset
    }

    /// Whether the live asset differs from what is on disk.
    pub fn is_dirty(&self) -> bool {
        self.asset != self.saved
    }

    /// Mark the current state as the one on disk — called **after** a successful
    /// write, never before, so a failed write leaves the session dirty.
    pub fn mark_saved(&mut self) {
        self.saved = self.asset.clone();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Push the current state onto the undo stack — called by every mutator
    /// *before* it mutates, and only when the mutation is going to succeed.
    fn checkpoint(&mut self) {
        self.undo.push(self.asset.clone());
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
        // Any new edit invalidates the redo branch, exactly like `SceneDoc`.
        self.redo.clear();
    }

    /// Step back one edit. `false` when there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.asset, prev));
                true
            }
            None => false,
        }
    }

    /// Step forward one edit. `false` when there is nothing to redo.
    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.asset, next));
                true
            }
            None => false,
        }
    }

    fn joint_in_range(&self, index: u16) -> Result<(), SkelError> {
        if (index as usize) < self.asset.skeleton.len() {
            Ok(())
        } else {
            Err(SkelError::NoSuchJoint {
                index,
                joints: self.asset.skeleton.len(),
            })
        }
    }

    /// Rename a joint. Returns what the rename cost — see [`RenameVerdict`].
    pub fn rename_joint(&mut self, index: u16, name: &str) -> Result<RenameVerdict, SkelError> {
        self.joint_in_range(index)?;
        let name = name.trim();
        if name.is_empty() {
            return Err(SkelError::EmptyName {
                what: "a joint name",
            });
        }
        let previous = self.asset.skeleton.joints()[index as usize].name.clone();
        let was_canonical = inf_anim::humanoid_joint_names().contains(&previous.as_str());
        let is_canonical = inf_anim::humanoid_joint_names().contains(&name);
        // "Had a twin" is measured on the SKELETON, not on the name: a `hand_l`
        // in a rig with no `hand_r` was already unpaired, and renaming it breaks
        // nothing.
        let had_pair = mirrored_joint_name(&previous)
            .and_then(|t| self.asset.skeleton.index_of(&t))
            .is_some();
        let has_pair = mirrored_joint_name(name)
            .and_then(|t| self.asset.skeleton.index_of(&t))
            .is_some();

        self.checkpoint();
        let mut joints = self.asset.skeleton.joints().to_vec();
        joints[index as usize].name = name.to_string();
        self.asset.skeleton = Skeleton::new(joints).expect("a rename cannot break the topology");
        Ok(RenameVerdict {
            left_humanoid_set: was_canonical && !is_canonical,
            broke_mirror_pair: had_pair && !has_pair,
            previous,
        })
    }

    /// Move a joint's **rest transform** and recompute every inverse bind.
    ///
    /// The recompute is not optional (see the module docs): `inverse_bind` is a
    /// function of the composed rest pose, and leaving it stale deforms every
    /// mesh bound to the rig.
    pub fn set_joint_local(&mut self, index: u16, local: JointTransform) -> Result<(), SkelError> {
        self.joint_in_range(index)?;
        for (what, v) in [
            ("a joint translation", &local.translation[..]),
            ("a joint rotation", &local.rotation[..]),
            ("a joint scale", &local.scale[..]),
        ] {
            if !v.iter().all(|c| c.is_finite()) {
                return Err(SkelError::NonFinite { what });
            }
        }
        self.checkpoint();
        let mut joints = self.asset.skeleton.joints().to_vec();
        joints[index as usize].local_bind = local;
        recompute_inverse_binds(&mut joints);
        self.asset.skeleton = Skeleton::new(joints).expect("moving a joint cannot break topology");
        Ok(())
    }

    /// Set or clear a joint's rotation limit.
    ///
    /// `None` **removes** the row rather than writing a full-range one, which is
    /// the distinction `JointLimit`'s own doc rests on: an absent row means
    /// "unlimited", and a present full-range row would be indistinguishable from
    /// an authored one.
    pub fn set_limit(&mut self, index: u16, limit: Option<JointLimit>) -> Result<(), SkelError> {
        self.joint_in_range(index)?;
        if let Some(l) = limit {
            if !l.min_deg.iter().chain(&l.max_deg).all(|d| d.is_finite()) {
                return Err(SkelError::NonFinite {
                    what: "a joint limit",
                });
            }
        }
        self.checkpoint();
        self.asset.limits.retain(|l| l.joint != index);
        if let Some(mut l) = limit {
            l.joint = index;
            self.asset.limits.push(l);
            // Sorted by joint so the on-disk bytes are a property of the rig and
            // not of the order an author happened to click.
            self.asset.limits.sort_by_key(|l| l.joint);
        }
        Ok(())
    }

    /// Add a socket on `joint`.
    pub fn add_socket(&mut self, name: &str, joint: u16) -> Result<(), SkelError> {
        self.joint_in_range(joint)?;
        let name = name.trim();
        if name.is_empty() {
            return Err(SkelError::EmptyName {
                what: "a socket name",
            });
        }
        if self.asset.sockets.iter().any(|s| s.name == name) {
            return Err(SkelError::SocketExists { name: name.into() });
        }
        self.checkpoint();
        self.asset.sockets.push(Socket::new(name, joint));
        self.asset.sockets.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(())
    }

    /// Remove a socket by name.
    pub fn remove_socket(&mut self, name: &str) -> Result<(), SkelError> {
        if !self.asset.sockets.iter().any(|s| s.name == name) {
            return Err(SkelError::NoSuchSocket { name: name.into() });
        }
        self.checkpoint();
        self.asset.sockets.retain(|s| s.name != name);
        Ok(())
    }

    /// Move a socket: re-parent it to `joint` and/or set its local offset.
    pub fn place_socket(
        &mut self,
        name: &str,
        joint: u16,
        offset: JointTransform,
    ) -> Result<(), SkelError> {
        self.joint_in_range(joint)?;
        if !self.asset.sockets.iter().any(|s| s.name == name) {
            return Err(SkelError::NoSuchSocket { name: name.into() });
        }
        if !offset
            .translation
            .iter()
            .chain(&offset.rotation)
            .chain(&offset.scale)
            .all(|c| c.is_finite())
        {
            return Err(SkelError::NonFinite {
                what: "a socket offset",
            });
        }
        self.checkpoint();
        for s in self.asset.sockets.iter_mut().filter(|s| s.name == name) {
            s.joint = joint;
            s.local_offset = offset;
        }
        Ok(())
    }

    /// **Replace** the rig with a freshly generated template.
    ///
    /// Undoable like every other edit, so an author who was exploring
    /// proportions can get their hand-built rig back.
    pub fn instantiate_template(
        &mut self,
        plan: BodyPlan,
        params: &BodyParams,
    ) -> Result<(), SkelError> {
        let fresh = inf_anim::build_template(plan, params)
            .map_err(|e| SkelError::Template(e.to_string()))?;
        self.checkpoint();
        self.asset = fresh;
        Ok(())
    }

    /// **Replace the whole rig**, undoably — what an auto-fit lands through.
    ///
    /// Separate from [`SkelSession::instantiate_template`] because a fit's rig is
    /// *computed from a mesh*, not generated from parameters: routing it through
    /// the template door would mean re-deriving the parameters it was already
    /// past. Same undo semantics, so a fit an author dislikes is one Ctrl+Z away
    /// from the rig they had.
    pub fn replace_asset(&mut self, asset: SkeletonAsset) {
        self.checkpoint();
        self.asset = asset;
    }

    /// **Append a part** onto this rig at `attach` — the modular-rigging door
    /// (`inf_anim::merge_skeletons`), returning the joint offset the caller needs
    /// to remap the part's weight table.
    pub fn merge_part(&mut self, incoming: &SkeletonAsset, attach: u16) -> Result<u16, SkelError> {
        let merged = merge_skeletons(&self.asset, incoming, attach)?;
        self.checkpoint();
        self.asset = merged.asset;
        Ok(merged.joint_offset)
    }

    /// Joints whose name says they have a side and whose twin is missing — what
    /// the panel warns about, and what `mirror_with_joints` refuses on.
    pub fn unmatched_sided(&self) -> Vec<String> {
        unmatched_sided_joints(&self.asset.skeleton)
    }
}

/// Recompute every joint's `inverse_bind` from the current `local_bind` chain.
///
/// One parent-order walk (the list is topological, so a parent's global is
/// always already computed) composing `Mat4`s, then one `Mat4::inverse` per
/// joint. **Pure arithmetic** — no trigonometry — which is what keeps it inside
/// the P14 determinism law, the same discipline `template::Builder` follows with
/// its exact translation negation.
///
/// A singular joint (zero scale) inverts to a matrix of infinities in glam; that
/// is caught at the door instead, by [`SkelSession::set_joint_local`]'s finite
/// check plus the fact that a zero scale is authored deliberately. Left as-is
/// rather than clamped, because silently resizing an author's bone is worse than
/// a bone that visibly does nothing.
pub fn recompute_inverse_binds(joints: &mut [Joint]) {
    let mut globals: Vec<glam::Mat4> = Vec::with_capacity(joints.len());
    for j in joints.iter() {
        let local = j.local_bind.to_mat4();
        // The list is topological, so a parent's global is always already in
        // `globals` — which is what makes one pass enough.
        let g = match j.parent {
            Some(p) => globals[p as usize] * local,
            None => local,
        };
        globals.push(g);
    }
    for (j, g) in joints.iter_mut().zip(&globals) {
        j.inverse_bind = g.inverse().to_cols_array();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn biped() -> SkeletonAsset {
        inf_anim::build_template(BodyPlan::Biped, &BodyParams::default()).unwrap()
    }

    #[test]
    fn a_fresh_session_is_clean_and_has_no_history() {
        let s = SkelSession::new(biped());
        assert!(!s.is_dirty());
        assert!(!s.can_undo() && !s.can_redo());
    }

    /// **Dirty is measured against DISK**, not against a flag: an edit dirties,
    /// `mark_saved` cleans, and undoing back to the saved state cleans too —
    /// which a boolean flag would get wrong.
    #[test]
    fn dirty_is_a_comparison_with_the_saved_state() {
        let mut s = SkelSession::new(biped());
        s.add_socket("grip", 0).unwrap();
        assert!(s.is_dirty());
        s.undo();
        assert!(
            !s.is_dirty(),
            "undoing back to the saved state must not leave the document dirty"
        );
        s.redo();
        assert!(s.is_dirty());
        s.mark_saved();
        assert!(!s.is_dirty());
    }

    /// **The headline correctness gate**: moving a joint recomputes the inverse
    /// binds, so a mesh bound to the rig still deforms correctly.
    ///
    /// Asserted as the defining property — `inverse_bind * global_bind == I` —
    /// rather than against a literal, because the literal would just be this
    /// computation written twice.
    #[test]
    fn moving_a_joint_recomputes_every_inverse_bind() {
        let mut s = SkelSession::new(biped());
        let elbow = s.asset().skeleton.index_of("lower_arm_r").unwrap();
        let before = s.asset().skeleton.joints()[elbow as usize].inverse_bind;

        let mut local = s.asset().skeleton.joints()[elbow as usize].local_bind;
        local.translation[1] -= 0.15;
        s.set_joint_local(elbow, local).expect("moves");
        assert_ne!(
            s.asset().skeleton.joints()[elbow as usize].inverse_bind,
            before,
            "the inverse bind did not follow the joint"
        );

        // The property: for every joint, inverse_bind composed with its global
        // bind transform is the identity.
        let joints = s.asset().skeleton.joints();
        let mut globals: Vec<glam::Mat4> = Vec::new();
        for j in joints {
            let l = j.local_bind.to_mat4();
            globals.push(match j.parent {
                Some(p) => globals[p as usize] * l,
                None => l,
            });
        }
        for (j, g) in joints.iter().zip(&globals) {
            let id = glam::Mat4::from_cols_array(&j.inverse_bind) * *g;
            for (a, b) in id
                .to_cols_array()
                .iter()
                .zip(glam::Mat4::IDENTITY.to_cols_array().iter())
            {
                assert!(
                    (a - b).abs() < 1e-4,
                    "{} has a stale inverse bind: {id:?}",
                    j.name
                );
            }
        }
    }

    /// …and a CHILD's inverse bind follows when its PARENT moves — the half a
    /// per-joint recompute would miss.
    #[test]
    fn moving_a_parent_updates_its_childrens_inverse_binds() {
        let mut s = SkelSession::new(biped());
        let shoulder = s.asset().skeleton.index_of("shoulder_r").unwrap();
        let hand = s.asset().skeleton.index_of("hand_r").unwrap();
        let before = s.asset().skeleton.joints()[hand as usize].inverse_bind;
        let mut local = s.asset().skeleton.joints()[shoulder as usize].local_bind;
        local.translation[0] += 0.2;
        s.set_joint_local(shoulder, local).unwrap();
        assert_ne!(
            s.asset().skeleton.joints()[hand as usize].inverse_bind,
            before,
            "a child's inverse bind did not follow its parent"
        );
    }

    /// A rename off the canonical vocabulary is allowed and **reported** — both
    /// costs, separately, because they break different things.
    #[test]
    fn renaming_off_the_canonical_set_warns_without_refusing() {
        let mut s = SkelSession::new(biped());
        let i = s.asset().skeleton.index_of("upper_arm_l").unwrap();
        let v = s.rename_joint(i, "arm1").expect("renames");
        assert_eq!(v.previous, "upper_arm_l");
        assert!(v.left_humanoid_set, "it left the retarget vocabulary");
        assert!(v.broke_mirror_pair, "it lost its `upper_arm_r` twin");
        assert_eq!(s.asset().skeleton.joints()[i as usize].name, "arm1");
        // …and the mirror map now says so.
        assert!(s.unmatched_sided().contains(&"upper_arm_r".to_string()));
    }

    /// A rename that stays inside both conventions warns about neither — the
    /// anti-vacuity arm, without which a verdict that was always `true` would
    /// pass the test above.
    #[test]
    fn a_harmless_rename_warns_about_nothing() {
        let mut s = SkelSession::new(biped());
        // `pelvis` is deliberately NOT one of the canonical nineteen (see
        // `template::girdle_name`), so renaming it leaves nothing.
        let i = s.asset().skeleton.index_of("pelvis").unwrap();
        let v = s.rename_joint(i, "hip_anchor").expect("renames");
        assert_eq!(
            v,
            RenameVerdict {
                left_humanoid_set: false,
                broke_mirror_pair: false,
                previous: "pelvis".into(),
            }
        );
    }

    /// An empty name, and a joint index the rig does not have, are refusals.
    #[test]
    fn empty_names_and_bad_indices_refuse() {
        let mut s = SkelSession::new(biped());
        assert!(matches!(
            s.rename_joint(0, "   "),
            Err(SkelError::EmptyName { .. })
        ));
        assert!(matches!(
            s.rename_joint(9999, "x"),
            Err(SkelError::NoSuchJoint { index: 9999, .. })
        ));
        assert!(matches!(
            s.add_socket("", 0),
            Err(SkelError::EmptyName { .. })
        ));
        assert!(!s.is_dirty(), "a refusal must not have edited anything");
        assert!(!s.can_undo(), "a refusal must not push an undo step");
    }

    /// Sockets: add refuses a duplicate name, remove refuses an absent one, and
    /// place moves both the joint and the offset.
    #[test]
    fn socket_editing_round_trips_and_refuses_duplicates() {
        let mut s = SkelSession::new(biped());
        s.add_socket("scabbard", 0).expect("adds");
        assert_eq!(
            s.add_socket("scabbard", 1),
            Err(SkelError::SocketExists {
                name: "scabbard".into()
            })
        );
        let off = JointTransform::from_trs(
            glam::Vec3::new(0.1, 0.0, 0.0),
            glam::Quat::IDENTITY,
            glam::Vec3::ONE,
        );
        s.place_socket("scabbard", 2, off).expect("places");
        let sock = s
            .asset()
            .sockets
            .iter()
            .find(|x| x.name == "scabbard")
            .unwrap();
        assert_eq!(sock.joint, 2);
        assert_eq!(sock.local_offset, off);
        s.remove_socket("scabbard").expect("removes");
        assert_eq!(
            s.remove_socket("scabbard"),
            Err(SkelError::NoSuchSocket {
                name: "scabbard".into()
            })
        );
    }

    /// `None` **removes** the limit row rather than writing a full-range one —
    /// the distinction the absent-means-unlimited rule rests on.
    #[test]
    fn clearing_a_limit_removes_its_row() {
        let mut s = SkelSession::new(biped());
        let knee = s.asset().skeleton.index_of("lower_leg_r").unwrap();
        assert!(s.asset().limit(knee).is_some(), "the template limits knees");
        s.set_limit(knee, None).expect("clears");
        assert!(s.asset().limit(knee).is_none());
        assert!(
            !s.asset().limits.iter().any(|l| l.joint == knee),
            "a cleared limit left a full-range row behind"
        );
        s.set_limit(knee, Some(JointLimit::hinge_x(knee, -90.0, 0.0)))
            .expect("sets");
        assert_eq!(s.asset().limit(knee).unwrap().min_deg[0], -90.0);
        // The table stays sorted by joint, so the bytes are a property of the rig.
        let joints: Vec<u16> = s.asset().limits.iter().map(|l| l.joint).collect();
        let mut sorted = joints.clone();
        sorted.sort_unstable();
        assert_eq!(joints, sorted);
    }

    /// Undo walks back through several edits and redo walks forward; a new edit
    /// truncates the redo branch.
    #[test]
    fn undo_and_redo_walk_the_history() {
        let mut s = SkelSession::new(biped());
        s.add_socket("a", 0).unwrap();
        s.add_socket("b", 0).unwrap();
        assert_eq!(s.asset().sockets.len(), biped().sockets.len() + 2);
        assert!(s.undo() && s.undo());
        assert_eq!(s.asset().sockets.len(), biped().sockets.len());
        assert!(!s.undo(), "nothing left to undo");
        assert!(s.redo());
        assert_eq!(s.asset().sockets.len(), biped().sockets.len() + 1);
        // A fresh edit drops the redo branch.
        s.add_socket("c", 0).unwrap();
        assert!(!s.can_redo());
    }

    /// Template instantiation replaces the rig and is undoable.
    #[test]
    fn instantiating_a_template_is_undoable() {
        let mut s = SkelSession::new(biped());
        s.add_socket("keepsake", 0).unwrap();
        let before = s.asset().clone();
        let params = BodyParams {
            height_m: 2.4,
            ..Default::default()
        };
        s.instantiate_template(BodyPlan::Quadruped, &params)
            .expect("builds");
        assert!(
            s.asset().skeleton.index_of("upper_arm_l").is_none(),
            "a quadruped has no arms"
        );
        s.undo();
        assert_eq!(s.asset(), &before, "undo must restore the hand-built rig");
    }

    /// **Merging a part through the session** appends and reports the offset —
    /// the Ring-1 door onto `inf_anim::merge_skeletons`.
    #[test]
    fn merging_a_part_appends_and_reports_its_offset() {
        let mut s = SkelSession::new(biped());
        let before = s.asset().skeleton.len();
        let tail = {
            let j = |name: &str, parent: Option<u16>| Joint {
                name: name.into(),
                parent,
                inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            };
            SkeletonAsset::new(
                Skeleton::new(vec![j("tail_0", None), j("tail_1", Some(0))]).unwrap(),
            )
        };
        let attach = s.asset().skeleton.index_of("hips").unwrap();
        let offset = s.merge_part(&tail, attach).expect("merges");
        assert_eq!(offset as usize, before);
        assert_eq!(s.asset().skeleton.len(), before + 2);
        assert_eq!(
            s.asset().skeleton.joints()[before].parent,
            Some(attach),
            "the tail did not attach where it was asked to"
        );
        // A colliding name refuses, as a value, and leaves the rig alone.
        let dup = SkeletonAsset::new(
            Skeleton::new(vec![Joint {
                name: "hips".into(),
                parent: None,
                inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            }])
            .unwrap(),
        );
        let len = s.asset().skeleton.len();
        assert!(matches!(s.merge_part(&dup, 0), Err(SkelError::Merge(_))));
        assert_eq!(s.asset().skeleton.len(), len);
    }

    /// A non-finite transform never reaches the asset.
    #[test]
    fn a_non_finite_transform_is_refused() {
        let mut s = SkelSession::new(biped());
        let mut bad = JointTransform::IDENTITY;
        bad.translation[0] = f32::NAN;
        assert!(matches!(
            s.set_joint_local(0, bad),
            Err(SkelError::NonFinite { .. })
        ));
        assert!(!s.is_dirty());
    }
}
