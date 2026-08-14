//! **The unified streamer's audit** (P28.3): the three page systems' residency,
//! budget and coupling in one place.
//!
//! `inf-stream` owns the arbitration and knows nothing about a GPU; this is the
//! renderer's side of it — the struct that gathers the live numbers, calls the
//! arbiter with the **real** floors, and hands a host or a gate one value to
//! read.
//!
//! # Why one struct and not three accessors
//!
//! Because the question the phase exists to answer has no answer in three:
//! *"how much streaming residency does this machine hold?"* was
//! `VgeomStreamStats::resident_bytes` plus `VtResidency::resident_bytes` plus
//! `VsmResidency::resident_bytes`, three numbers nothing summed, and the P28.2
//! ledger's own remainder says so — *"`resident_bytes` counts geometry only …
//! a combined number is literally P28.3's clause 2."* This is that number, and
//! it is a **sum and not a re-count**: a cluster page's texture tiles are spent
//! out of the texture pool by the same transaction, so they are counted once,
//! there, exactly as the pairing already refused to count them twice.

use inf_stream::{BudgetGrant, Consumer, RingLedger, StreamError, CONSUMERS};

/// **The coupling invariant**, re-exported here so a gate that reads a
/// [`StreamReport`] can check it without naming `inf-stream` — the same reason
/// the arbiter's vocabulary is re-exported from the crate root.
pub use inf_stream::breaches;

/// One frame's view of the whole streamer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamReport {
    /// The unified ceiling the three are divided from
    /// ([`RenderSettings::stream`](crate::RenderSettings)).
    pub budget_bytes: u64,
    /// What the arbiter grants each consumer **from the live floors** — or the
    /// refusal, when the budget cannot hold them.
    ///
    /// This is the audit, not the sizing: the pools were sized from
    /// [`RenderSettings::arbitrate_budgets`](crate::RenderSettings::arbitrate_budgets),
    /// which runs with zero floors because a settings struct knows no content.
    /// Here the floors are real, so an `Err` is a machine that genuinely cannot
    /// hold what this level pins.
    pub grant: Result<BudgetGrant, StreamError>,
    /// Bytes each consumer is holding right now, in [`Consumer`] order.
    pub resident: [u64; CONSUMERS],
    /// Bytes each consumer may never fall below, in [`Consumer`] order. The
    /// shadow atlas's is **zero by design** — a page nothing marked is a page
    /// nothing reads, which is `inf-vsm`'s load-bearing difference from
    /// `inf-vt` and not an omission here.
    pub floors: [u64; CONSUMERS],
    /// Cluster pages coupled to their texture tiles.
    pub coupled_groups: usize,
    /// Distinct tiles those groups name, deduplicated across pages.
    pub coupled_tiles: usize,
    /// Groups dropped since the renderer was built — the anti-vacuity number
    /// for cross-system aging.
    pub dropped_groups: u64,
    /// Cluster pages the texture half refused, cumulative (P28.2).
    pub retracted: u64,
    /// Cooked tile addresses the registered image does not have, cumulative
    /// (P28.2 audit). **Its reader is this struct** — the counter reached
    /// `VgeomStreamReport` in P28.2 with nothing consuming it, and the audit's
    /// remainder list says so by name.
    pub stale_tiles: u64,
    /// Textures whose cooked **grid digest** does not match the registered
    /// image's, cumulative (P28.3) — the *structural* half of the staleness
    /// class the P28.2 audit routed here. A non-zero value is a pack whose
    /// `.inf_vmesh` and `.inf_tex` came from two different cooks, and it is the
    /// one direction `stale_tiles` cannot see: an image that GREW still has
    /// every cooked address, so nothing is missing and the detail is silently
    /// wrong.
    pub mismatched_textures: u64,
    /// The readback ledger both feedback consumers report into.
    pub ring: RingLedger,
}

impl StreamReport {
    /// Bytes the three page systems hold **between them** — the number that did
    /// not exist before this phase.
    pub fn resident_bytes(&self) -> u64 {
        self.resident.iter().sum()
    }

    /// Bytes one consumer holds.
    pub fn resident_of(&self, c: Consumer) -> u64 {
        self.resident[c.index()]
    }

    /// Whether every consumer is inside the grant the live floors justify.
    ///
    /// `false` when the arbiter refused, because a residency measured against a
    /// budget that was refused is measured against nothing.
    pub fn within_grant(&self) -> bool {
        match &self.grant {
            Ok(g) => (0..CONSUMERS).all(|i| self.resident[i] <= g.bytes[i]),
            Err(_) => false,
        }
    }

    /// A one-line human summary, in the shape `VtStats::summary`,
    /// `VgeomStreamStats::summary` and `VsmStats::summary` already ship — so the
    /// four read alike in one log.
    pub fn summary(&self) -> String {
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        let mut s = format!(
            "stream: {:.1} MiB resident of {:.1} MiB — geometry {:.1}, texture {:.1}, \
             shadow {:.1}; {} groups / {} tiles coupled ({} dropped, {} retracted)",
            mib(self.resident_bytes()),
            mib(self.budget_bytes),
            mib(self.resident_of(Consumer::Geometry)),
            mib(self.resident_of(Consumer::Texture)),
            mib(self.resident_of(Consumer::Shadow)),
            self.coupled_groups,
            self.coupled_tiles,
            self.dropped_groups,
            self.retracted,
        );
        if self.stale_tiles > 0 {
            s.push_str(&format!(" [{} stale tile addresses]", self.stale_tiles));
        }
        if self.mismatched_textures > 0 {
            s.push_str(&format!(
                " [{} textures paired against another image]",
                self.mismatched_textures
            ));
        }
        if let Err(e) = &self.grant {
            s.push_str(&format!(" [REFUSED: {e}]"));
        }
        s.push_str("; ");
        s.push_str(&self.ring.summary());
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> StreamReport {
        StreamReport {
            budget_bytes: 100,
            grant: inf_stream::arbitrate(
                100,
                &[
                    inf_stream::BudgetRequest::want(40),
                    inf_stream::BudgetRequest::want(40),
                    inf_stream::BudgetRequest::want(20),
                ],
            ),
            resident: [10, 20, 5],
            floors: [4, 2, 0],
            coupled_groups: 3,
            coupled_tiles: 7,
            dropped_groups: 2,
            retracted: 1,
            stale_tiles: 0,
            mismatched_textures: 0,
            ring: RingLedger::default(),
        }
    }

    /// The combined number is a **sum**, and each consumer is still readable on
    /// its own — one number never wears two names.
    #[test]
    fn the_combined_residency_is_the_sum_of_three_readable_parts() {
        let r = report();
        assert_eq!(r.resident_bytes(), 35);
        assert_eq!(r.resident_of(Consumer::Geometry), 10);
        assert_eq!(r.resident_of(Consumer::Texture), 20);
        assert_eq!(r.resident_of(Consumer::Shadow), 5);
        assert!(r.within_grant());
    }

    /// A refusal is not "within the grant", however small residency is —
    /// otherwise a machine that cannot hold its own floor reports healthy.
    #[test]
    fn a_refused_budget_is_never_within_its_grant() {
        let mut r = report();
        r.resident = [0, 0, 0];
        r.grant = Err(StreamError::FloorExceedsBudget {
            floor_bytes: 900,
            total_bytes: 100,
        });
        assert!(!r.within_grant(), "an empty residency masked a refusal");
        let s = r.summary();
        assert!(s.contains("REFUSED"), "{s}");
        assert!(s.contains("mandatory floors"), "{s}");
    }

    /// The summary names every number it carries, and says nothing about stale
    /// addresses when there are none.
    #[test]
    fn the_stream_line_says_what_it_counts() {
        let r = report();
        let s = r.summary();
        for token in [
            "stream:",
            "geometry",
            "texture",
            "shadow",
            "3 groups",
            "7 tiles",
            "2 dropped",
            "1 retracted",
            "stream ring",
        ] {
            assert!(s.contains(token), "{token:?} missing from {s}");
        }
        assert!(!s.contains("stale"), "{s}");
        assert!(!s.contains("another image"), "{s}");
        let mut stale = r;
        stale.stale_tiles = 4;
        stale.mismatched_textures = 1;
        let s = stale.summary();
        assert!(s.contains("[4 stale tile addresses]"), "{s}");
        assert!(
            s.contains("[1 textures paired against another image]"),
            "{s}"
        );
    }

    /// Residency past the grant is visible — the ratchet's own predicate.
    #[test]
    fn residency_past_a_grant_is_reported_as_such() {
        let mut r = report();
        r.resident[Consumer::Texture.index()] = 41;
        assert!(!r.within_grant());
    }
}
