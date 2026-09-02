//! **One door for what a `PcgVolume` evaluates to** (I3).
//!
//! A volume's population is three passes concatenated in a fixed order —
//! scatter rules, then grammars, then buildings — and until I3 that
//! concatenation was written out by hand in **two** places: the shipped player's
//! `evaluate_pcg_volumes_in` and the editor's `pcg_evaluate` command. Two
//! spellings of one order is the shape this repository has paid for at four
//! seams, and it became load-bearing the moment [`StructureGroup`] arrived:
//! a group is a pair of **index ranges**, and an index range is only meaningful
//! against the list it indexes. A host that appended the scatter instances after
//! the grammar's would produce ranges that name the wrong buildings, and nothing
//! would crash — a distant tower would simply draw somebody else's walls.
//!
//! So the composition is a function, both hosts call it, and the offsetting
//! happens exactly where the lists are joined.

use crate::building::StructureGroup;
use crate::grammar::expand::GrammarOutput;
use crate::scatter::{PcgCollider, PcgInstance};

/// **The most real lights one `PcgVolume` may contribute to a frame** (wave
/// VEN1a).
///
/// # The measurement that forces it
///
/// `inf_render::MAX_LIGHTS` is **16 for the whole scene**, and the truncation
/// is first-N in projection order with **no distance prioritization anywhere**
/// between the ECS and the uniform — so the seventeenth light is not dimmed,
/// it is deleted, and which sixteen survive depends on GUID order. A venue
/// hangs up to four fixtures (three stage spots and a bar glow), and a
/// settlement block is subdivided into LOTS: a 100 m city block at a
/// nightclub's 32 m frontage is **nine** nightclubs, which is thirty-six
/// lights, which is the sun going out.
///
/// # Why the cap is here and not in the settlement generator
///
/// Because the scarce thing is *lights in a frame*, and a lot rule is a
/// statement about *frontage*. Making venue lots whole-block does hold the
/// budget — it was tried — and it also made the wave's gate arm run
/// **11 s → 539 s**, because one 54 × 54 m building is a different shape of
/// problem from six 20 × 30 ones with the same floor area. A rule about the
/// scarce thing is smaller than a rule about a proxy for it.
///
/// # What it costs, stated
///
/// A block of six bars lights four of them. Which four is the volume's own
/// building order — deterministic, identical on both hosts, and **not** a
/// function of the camera, which is what keeps a level's content from
/// depending on where somebody stood. What it is NOT is a distance-prioritized
/// selection, which is the right long answer and belongs to the renderer,
/// where the eye is; see the wave ledger's carried list.
///
/// Four, because at most three venue blocks stand in one settlement (the
/// nightlife strip), so the worst frame is `3 × 4 + 2` sky lights = **14 of
/// 16**.
pub const VOLUME_LIGHT_CAP: usize = 4;

/// Everything one `PcgVolume` evaluates to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VolumeOutput {
    /// Every placed instance: the scatter rules' first, then the grammars' and
    /// the buildings'.
    pub instances: Vec<PcgInstance>,
    /// Every solid box. Scatter rules contribute none.
    pub colliders: Vec<PcgCollider>,
    /// Which runs of the two lists above are one building, and each building's
    /// shell. Ranges are into **this** struct's lists.
    pub groups: Vec<StructureGroup>,
    /// **Every door this volume's buildings want** (I6). Located in the world
    /// and naming no index, so composition neither shifts nor re-bases them.
    pub doorways: Vec<crate::building::PcgDoorway>,
    /// **Every place a person can be in this volume's buildings** (NPC1d).
    /// Located in the world and naming only its own building's ordinal, so
    /// composition neither shifts nor re-bases them.
    pub slots: Vec<crate::building::society::PcgSlot>,
    /// **The walkable interior of this volume's slot-bearing buildings**
    /// (NPC1d), already in the level's own namespace.
    pub interior: inf_nav::NavGraph,
    /// **Every real light this volume's buildings hang** (wave VEN1a). Located
    /// in the world and naming no index, so composition neither shifts nor
    /// re-bases them.
    pub lights: Vec<crate::building::PcgLight>,
    /// **Where this volume's buildings make a NOISE** (wave VEN1b) — the
    /// [`Music`](crate::building::StationUse::Music) stations, and nothing else.
    ///
    /// The occupied stations became [`slots`](Self::slots) in `pass.rs`, where
    /// the building's salt was in hand; these are the ones no body stands at,
    /// and they cross into the ECS as themselves because what reads them is the
    /// audio step rather than the society. World metres, no index — the terms
    /// [`doorways`](Self::doorways) and [`lights`](Self::lights) are on.
    pub emitters: Vec<crate::building::PcgStation>,
}

/// Join a volume's scatter instances with its grammar/building output.
///
/// `scatter` comes first because that is the order both hosts have always used
/// and a level's committed content hash depends on it. The grammar output's
/// group ranges are shifted by the scatter prefix — the collider ranges are
/// **not** shifted, because scatter rules contribute no colliders, and that
/// asymmetry is precisely why the two ranges are carried separately.
pub fn compose_volume(scatter: Vec<PcgInstance>, grammar: GrammarOutput) -> VolumeOutput {
    let shift = scatter.len() as u32;
    let mut instances = scatter;
    instances.extend(grammar.instances);
    VolumeOutput {
        instances,
        colliders: grammar.colliders,
        doorways: grammar.doorways,
        slots: grammar.slots,
        interior: grammar.interior,
        // **Capped here, at the one door**, so no host can produce a volume
        // that overruns the frame's light budget and no host can produce a
        // different four from the other. See `VOLUME_LIGHT_CAP`.
        lights: {
            let mut l = grammar.lights;
            l.truncate(VOLUME_LIGHT_CAP);
            l
        },
        // **The stations nobody stands at** (VEN1b). `pass.rs` has already
        // taken the occupied ones into `slots`; what is left on the list is the
        // music, and filtering here — at the one door both hosts pass, beside
        // the light cap — is what keeps a seat from reaching the audio step and
        // an emitter from reaching the society.
        emitters: grammar
            .stations
            .into_iter()
            .filter(|s| !s.use_kind.is_occupied_by_a_person())
            .collect(),
        groups: grammar
            .groups
            .into_iter()
            .map(|mut g| {
                g.inst_start += shift;
                g
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::lod::group_shell;
    use crate::building::LotFrame;
    use glam::{DQuat, DVec3};

    fn inst(i: u32) -> PcgInstance {
        PcgInstance {
            pos: DVec3::new(f64::from(i), 0.0, 0.0),
            rotation: DQuat::IDENTITY,
            scale: 1.0,
            kind_index: i,
            mesh: None,
            extent: None,
            glow: 0.0,
            surface: crate::scatter::PcgSurface::DEFAULT,
        }
    }
    fn col(i: u32) -> PcgCollider {
        PcgCollider {
            center: DVec3::new(f64::from(i), 1.0, 0.0),
            half_extents: DVec3::splat(0.5),
            rotation: DQuat::IDENTITY,
        }
    }

    /// **The defect the door exists to prevent, measured.** A group's instance
    /// range must name the same building after composition as before it.
    #[test]
    fn composition_rebases_the_instance_range_and_leaves_the_collider_range() {
        let colliders: Vec<PcgCollider> = (0..4).map(col).collect();
        let grammar = GrammarOutput {
            instances: (100..104).map(inst).collect(),
            colliders: colliders.clone(),
            groups: vec![StructureGroup {
                shell: group_shell(LotFrame::IDENTITY, &colliders).expect("a shell"),
                start: 0,
                len: 4,
                inst_start: 0,
                inst_len: 4,
            }],
            doorways: Vec::new(),
            decor: Vec::new(),
            slots: Vec::new(),
            interior: inf_nav::NavGraph::new(),
            lights: Vec::new(),
            stations: Vec::new(),
        };
        let scatter: Vec<PcgInstance> = (0..7).map(inst).collect();
        let out = compose_volume(scatter, grammar);

        assert_eq!(out.instances.len(), 11);
        assert_eq!(out.colliders.len(), 4);
        let g = out.groups[0];
        assert_eq!(g.inst_start, 7, "the scatter prefix must shift instances");
        assert_eq!(g.start, 0, "scatter contributes no colliders to shift past");
        // The range really names the building's own instances.
        for i in &out.instances[g.instance_range()] {
            assert!(i.kind_index >= 100, "range names a scatter instance");
        }
        for c in &out.colliders[g.range()] {
            assert!(c.center.y == 1.0);
        }
    }

    /// **A colliding `kind_index` cannot make a wall panel draw a grass tuft**
    /// (wave TER2b audit).
    ///
    /// Wave TER2b's clause 0 rests on a claim this module is the proof of: an
    /// index cannot name a mesh here, because composition interleaves a *scatter*
    /// palette's indices with a *grammar module* palette's into the same `u32`
    /// with no offset — so the GUID rides on the instance instead. The claim was
    /// argued in three doc comments and asserted nowhere, and what protects it is
    /// three `mesh: None` literals at the grammar and building placement sites,
    /// which a fourth site would not inherit.
    ///
    /// This collides the indices **deliberately**: both halves use `0..4`, so
    /// every grammar instance shares an index with a scatter instance that draws
    /// a real mesh. The mesh must follow the instance and not the index.
    #[test]
    fn a_colliding_kind_index_does_not_carry_a_mesh_across_the_join() {
        let tuft = uuid::Uuid::from_u128(0xC0FFEE);
        let scatter: Vec<PcgInstance> = (0..4)
            .map(|i| PcgInstance {
                mesh: Some(tuft),
                ..inst(i)
            })
            .collect();
        // The grammar's own placements, with the SAME indices and no mesh — which
        // is what `place_module` and `building::assemble` write.
        let grammar = GrammarOutput {
            instances: (0..4).map(inst).collect(),
            ..GrammarOutput::default()
        };
        let out = compose_volume(scatter, grammar);
        assert_eq!(out.instances.len(), 8);
        for (n, i) in out.instances.iter().enumerate() {
            let want = if n < 4 { Some(tuft) } else { None };
            assert_eq!(
                i.mesh, want,
                "instance {n} carries kind_index {} and mesh {:?}; after the join \
                 the first four are scatter and the last four are grammar modules, \
                 and their indices are the same four numbers",
                i.kind_index, i.mesh
            );
        }
        // …and the indices really did collide, or the arm proves nothing.
        let scatter_kinds: Vec<u32> = out.instances[..4].iter().map(|i| i.kind_index).collect();
        let grammar_kinds: Vec<u32> = out.instances[4..].iter().map(|i| i.kind_index).collect();
        assert_eq!(scatter_kinds, grammar_kinds);
    }

    /// **THE FRAME BUDGET, HELD AT THE ONE DOOR** (wave VEN1a).
    ///
    /// A block of venues is a block of rigs, and the frame has sixteen light
    /// slots for the whole scene. This is the arm that says the cap is applied
    /// where both hosts pass and that it keeps the SAME four.
    #[test]
    fn a_volume_never_contributes_more_than_its_share_of_the_frame() {
        let fixture = |n: usize| crate::building::PcgLight {
            at: DVec3::new(n as f64, 4.0, 0.0),
            dir: -DVec3::Y,
            sweep: ([3.0, 0.1, 0.2], [2.4, 0.2, 1.9]),
            intensity: 26.0,
            range_m: 8.0,
            inner_deg: 20.0,
            outer_deg: 36.0,
            cycle_hz: 0.11,
            phase: 0,
            phases: 3,
        };
        // Nine nightclubs' worth -- what a 100 m city block at a 32 m frontage
        // actually subdivides into.
        let grammar = GrammarOutput {
            lights: (0..36).map(fixture).collect(),
            ..GrammarOutput::default()
        };
        let out = compose_volume(Vec::new(), grammar);
        assert_eq!(
            out.lights.len(),
            VOLUME_LIGHT_CAP,
            "thirty-six fixtures reached the frame; `MAX_LIGHTS` is 16 for the whole scene"
        );
        // …and it is the FIRST four, in the volume's own building order, so two
        // hosts composing the same volume light the same four buildings.
        for (k, l) in out.lights.iter().enumerate() {
            assert_eq!(l.at.x, k as f64, "the cap kept a different four");
        }
        // A volume under the cap is untouched, so every level that predates the
        // venues composes byte-identically.
        let few = GrammarOutput {
            lights: (0..2).map(fixture).collect(),
            ..GrammarOutput::default()
        };
        assert_eq!(compose_volume(Vec::new(), few).lights.len(), 2);
        assert!(compose_volume(Vec::new(), GrammarOutput::default())
            .lights
            .is_empty());
        // …and the cap really is small enough for the strip: three venue
        // blocks plus the sky's two lights must clear the frame's sixteen.
        // Measured through the length the arm just took, so this is an
        // assertion about what `compose_volume` DID rather than one clippy can
        // fold to a constant.
        let worst = 3 * out.lights.len() + 2;
        assert!(
            worst <= 16,
            "three venue blocks at {} fixtures each is {worst} lights against a frame of 16",
            out.lights.len()
        );
    }

    #[test]
    fn a_volume_with_no_buildings_composes_to_no_groups() {
        let out = compose_volume((0..3).map(inst).collect(), GrammarOutput::default());
        assert_eq!(out.instances.len(), 3);
        assert!(out.groups.is_empty());
        assert!(out.colliders.is_empty());
    }
}
