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

    #[test]
    fn a_volume_with_no_buildings_composes_to_no_groups() {
        let out = compose_volume((0..3).map(inst).collect(), GrammarOutput::default());
        assert_eq!(out.instances.len(), 3);
        assert!(out.groups.is_empty());
        assert!(out.colliders.is_empty());
    }
}
