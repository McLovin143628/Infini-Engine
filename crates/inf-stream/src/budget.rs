//! **One budget arbiter** (P28.3, clause 2): one unified byte ceiling divided
//! into three grants.
//!
//! Before this batch the tree had three independent VRAM ceilings — the meshlet
//! pools' `budget_bytes` (256 MiB), the virtual-texture page pool's (24 MiB)
//! and the shadow atlas's (64 MiB) — each clamped by its own tier rule and
//! **nothing bounding their sum**. A host could be handed 344 MiB of streaming
//! residency by a settings struct that never says so anywhere, and the P28.2
//! ledger's own remainder says as much: *"`resident_bytes` counts geometry
//! only … a combined number is literally P28.3's clause 2."*
//!
//! # The rule
//!
//! [`arbitrate`] takes the unified ceiling and one [`BudgetRequest`] per
//! consumer, and returns one grant per consumer:
//!
//! 1. **Floors first.** Every consumer's floor is granted before any want is.
//!    A floor is what residency may never fall below — the virtual texture's
//!    pinned coarsest mips, the meshlet streamer's always-resident page 0 — and
//!    a total that cannot hold the floors is refused **by name**
//!    ([`StreamError::FloorExceedsBudget`]), not silently clamped, which is
//!    `VtError::MandatoryFloorExceedsBudget`'s rule lifted one level.
//! 2. **The remainder is water-filled**, evenly and by whole bytes, clamped at
//!    each consumer's want, in a fixed [`Consumer`] order. A consumer that
//!    wants less than its even share leaves the difference to the others rather
//!    than holding it.
//! 3. **It is an identity when everything fits.** `sum(want) <= total` grants
//!    every consumer its want exactly — which is what makes the shipped
//!    defaults byte-identical to the three independent budgets they replace,
//!    and is asserted rather than assumed
//!    ([`tests::the_shipped_default_is_an_identity`]).
//!
//! # Why an even split and not a proportional one
//!
//! A proportional split hands the largest *request* the largest share, so the
//! consumer that asks for the most under pressure gets the most — and the
//! meshlet pools' default request is ten times the virtual texture's. Under a
//! ceiling that binds, proportional gives geometry 74 % of a scarce budget and
//! textures 7 %, which is precisely the "high-poly mesh with a blurry texture"
//! this phase exists to make impossible. Even-with-clamping does the opposite:
//! the small requests are satisfied first (they saturate and stop taking), and
//! what remains goes to whoever can still use it. Measured on the shipped
//! numbers under a 64 MiB ceiling: even-split gives **(geometry 23.33 MiB,
//! texture 21.33, shadow 19.33)**; proportional gives **(47.24, 5.78, 10.98)**,
//! i.e. the texture pool falls **below its own Low-tier ceiling of 6 MiB** on a
//! machine that asked for High. The comparison is in
//! [`tests::an_even_split_does_not_starve_the_smallest_consumer`], run against a
//! proportional control written out in the test rather than described.
//!
//! # Determinism
//!
//! Integer arithmetic, a fixed consumer order for the sub-byte remainder, no
//! floats and no iteration over a hash. Two calls with equal arguments return
//! equal grants; the arbiter has no state at all, which is the strongest form
//! of "a function of state, not history".

use crate::StreamError;

/// The three consumers of the unified streaming budget.
///
/// The discriminants are the array indices of [`BudgetGrant::bytes`] and are
/// the fixed order the sub-byte remainder is handed out in, so they are part of
/// the determinism argument rather than a naming convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Consumer {
    /// `inf-vgeom`'s four suballocated meshlet pools.
    Geometry = 0,
    /// `inf-vt`'s physical page pool.
    Texture = 1,
    /// `inf-vsm`'s physical shadow-page atlas.
    Shadow = 2,
}

/// How many consumers the arbiter divides between.
pub const CONSUMERS: usize = 3;

impl Consumer {
    /// Every consumer, in the arbiter's own order.
    pub const ALL: [Consumer; CONSUMERS] =
        [Consumer::Geometry, Consumer::Texture, Consumer::Shadow];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The name a summary line uses.
    pub const fn label(self) -> &'static str {
        match self {
            Consumer::Geometry => "geometry",
            Consumer::Texture => "texture",
            Consumer::Shadow => "shadow",
        }
    }
}

/// What one consumer is asking for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BudgetRequest {
    /// Bytes residency may never fall below. Granted before any want.
    pub floor_bytes: u64,
    /// Bytes the consumer would use if the budget allowed. A request of `0`
    /// means "this consumer is not live in this frame" — a tier that turned the
    /// meshlet path off asks for nothing rather than for its default, so its
    /// share is available to the two that are still running.
    pub want_bytes: u64,
}

impl BudgetRequest {
    /// A request with no mandatory floor — the shape a consumer whose residency
    /// may legally reach zero uses (`inf-vsm`: a page nothing marked is a page
    /// nothing reads).
    pub const fn want(want_bytes: u64) -> Self {
        Self {
            floor_bytes: 0,
            want_bytes,
        }
    }

    /// A request whose want is at least its floor — the normalized form
    /// [`arbitrate`] works on, exposed so a caller can see what it really
    /// asked for.
    #[inline]
    pub const fn normalized(self) -> Self {
        Self {
            floor_bytes: self.floor_bytes,
            want_bytes: if self.want_bytes > self.floor_bytes {
                self.want_bytes
            } else {
                self.floor_bytes
            },
        }
    }
}

/// What each consumer may spend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BudgetGrant {
    pub bytes: [u64; CONSUMERS],
}

impl BudgetGrant {
    #[inline]
    pub fn get(&self, c: Consumer) -> u64 {
        self.bytes[c.index()]
    }

    /// The bytes actually handed out — **never** more than the unified ceiling.
    #[inline]
    pub fn total(&self) -> u64 {
        self.bytes.iter().sum()
    }

    /// A one-line human summary, in the shape `VtStats::summary` and
    /// `VgeomStreamStats::summary` already ship so the streamers read alike in
    /// one log.
    pub fn summary(&self, total_bytes: u64) -> String {
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        format!(
            "stream budget: {:.1} MiB of {:.1} MiB granted — geometry {:.1}, texture {:.1}, \
             shadow {:.1}",
            mib(self.total()),
            mib(total_bytes),
            mib(self.get(Consumer::Geometry)),
            mib(self.get(Consumer::Texture)),
            mib(self.get(Consumer::Shadow)),
        )
    }
}

/// **Divide `total_bytes` between the consumers.** See the module docs for the
/// rule and for why the split is even rather than proportional.
pub fn arbitrate(
    total_bytes: u64,
    requests: &[BudgetRequest; CONSUMERS],
) -> Result<BudgetGrant, StreamError> {
    let req: [BudgetRequest; CONSUMERS] = [
        requests[0].normalized(),
        requests[1].normalized(),
        requests[2].normalized(),
    ];
    let floor_bytes: u64 = req.iter().map(|r| r.floor_bytes).sum();
    if floor_bytes > total_bytes {
        return Err(StreamError::FloorExceedsBudget {
            floor_bytes,
            total_bytes,
        });
    }
    let mut grant = BudgetGrant {
        bytes: [req[0].floor_bytes, req[1].floor_bytes, req[2].floor_bytes],
    };
    let mut remaining = total_bytes - floor_bytes;
    // Water-fill: an even share to every consumer that can still use one,
    // repeated until the budget is spent or every consumer has its want. Each
    // round strictly reduces the unsaturated set or the remainder, so this
    // terminates in at most CONSUMERS + 1 rounds.
    loop {
        let unsaturated: Vec<usize> = (0..CONSUMERS)
            .filter(|&i| grant.bytes[i] < req[i].want_bytes)
            .collect();
        if remaining == 0 || unsaturated.is_empty() {
            break;
        }
        let share = remaining / unsaturated.len() as u64;
        if share == 0 {
            // Fewer bytes than consumers: hand them out one at a time in the
            // fixed `Consumer` order, so the last few bytes of a budget are
            // deterministic rather than "whoever the loop reached".
            for &i in &unsaturated {
                if remaining == 0 {
                    break;
                }
                grant.bytes[i] += 1;
                remaining -= 1;
            }
            break;
        }
        for &i in &unsaturated {
            let take = share.min(req[i].want_bytes - grant.bytes[i]);
            grant.bytes[i] += take;
            remaining -= take;
        }
    }
    Ok(grant)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;

    /// The three shipped defaults, as `inf-render`'s settings carry them.
    fn shipped() -> [BudgetRequest; CONSUMERS] {
        [
            BudgetRequest {
                floor_bytes: 4 * MIB,
                want_bytes: 256 * MIB,
            },
            BudgetRequest {
                floor_bytes: 2 * MIB,
                want_bytes: 24 * MIB,
            },
            BudgetRequest::want(64 * MIB),
        ]
    }

    /// **The identity property, and it is what keeps every committed golden and
    /// every P26/P27 gate byte-identical across this batch.**
    ///
    /// The shipped unified ceiling is the sum of the three requests, so the
    /// arbiter hands each consumer exactly the number it had before there was
    /// an arbiter. If this arm fails, the batch changed a world rather than an
    /// arbitration.
    #[test]
    fn the_shipped_default_is_an_identity() {
        let req = shipped();
        let total: u64 = req.iter().map(|r| r.want_bytes).sum();
        let g = arbitrate(total, &req).expect("the floors fit");
        for (i, r) in req.iter().enumerate() {
            assert_eq!(
                g.bytes[i], r.want_bytes,
                "consumer {i} did not get its want"
            );
        }
        assert_eq!(g.total(), total);
        // …and one byte more than the sum is still an identity, not a bonus.
        let g = arbitrate(total + 1, &req).expect("the floors fit");
        assert_eq!(g.total(), total, "the arbiter handed out an unwanted byte");
    }

    /// **Byte-budget conservation**, swept: the grant never exceeds the
    /// ceiling, and it is exactly `min(sum(want), total)` whenever the floors
    /// fit.
    ///
    /// The sweep matters more than any single case: an off-by-one in the
    /// water-fill's clamp shows up on exactly the totals where one consumer
    /// saturates mid-round, which a hand-picked fixture is unlikely to sit on.
    #[test]
    fn the_grant_conserves_the_budget_over_a_dense_sweep() {
        let req = [
            BudgetRequest {
                floor_bytes: 3,
                want_bytes: 100,
            },
            BudgetRequest {
                floor_bytes: 7,
                want_bytes: 40,
            },
            BudgetRequest {
                floor_bytes: 0,
                want_bytes: 13,
            },
        ];
        let floors: u64 = req.iter().map(|r| r.floor_bytes).sum();
        let wants: u64 = req.iter().map(|r| r.want_bytes).sum();
        let mut saturating = 0usize;
        for total in 0..=(wants + 20) {
            match arbitrate(total, &req) {
                Err(StreamError::FloorExceedsBudget { floor_bytes, .. }) => {
                    assert!(total < floors, "refused a total of {total} that fits");
                    assert_eq!(floor_bytes, floors);
                }
                Ok(g) => {
                    assert!(total >= floors);
                    assert!(g.total() <= total, "total {total}: granted {}", g.total());
                    assert_eq!(
                        g.total(),
                        total.min(wants),
                        "total {total}: the grant is not min(want, total)"
                    );
                    for (i, r) in req.iter().enumerate() {
                        assert!(
                            g.bytes[i] >= r.floor_bytes,
                            "total {total}: consumer {i} below its floor"
                        );
                        assert!(
                            g.bytes[i] <= r.want_bytes,
                            "total {total}: consumer {i} over its want"
                        );
                    }
                    saturating += usize::from(g.total() == wants);
                }
            }
        }
        // ANTI-VACUITY: the sweep really did cross both regimes.
        assert!(saturating > 0 && saturating < (wants + 21) as usize);
    }

    /// **The floor is a refusal, not a clamp**, and the refusal names both
    /// numbers.
    #[test]
    fn a_total_that_cannot_hold_the_floors_is_refused_by_name() {
        let req = [
            BudgetRequest {
                floor_bytes: 10 * MIB,
                want_bytes: 100 * MIB,
            },
            BudgetRequest {
                floor_bytes: 8 * MIB,
                want_bytes: 20 * MIB,
            },
            BudgetRequest::want(4 * MIB),
        ];
        let err = arbitrate(17 * MIB, &req).expect_err("18 MiB of floor into 17");
        assert_eq!(
            err,
            StreamError::FloorExceedsBudget {
                floor_bytes: 18 * MIB,
                total_bytes: 17 * MIB,
            }
        );
        assert!(err.to_string().contains("mandatory floors"), "{err}");
        // The boundary is granted, not refused: floors exactly filling the
        // budget is the specified answer, as `VtResidency` already rules.
        let g = arbitrate(18 * MIB, &req).expect("floor == budget, to the byte");
        assert_eq!(g.bytes, [10 * MIB, 8 * MIB, 0]);
        assert_eq!(g.total(), 18 * MIB);
    }

    /// **An even split does not starve the smallest consumer — measured
    /// against a proportional control written out here.**
    ///
    /// The control is the split the obvious implementation makes, so this is a
    /// comparison rather than an assertion that the chosen rule is nice.
    #[test]
    fn an_even_split_does_not_starve_the_smallest_consumer() {
        let req = shipped();
        let ceiling = 64 * MIB;
        let g = arbitrate(ceiling, &req).expect("the floors fit");

        // The proportional control: floors first, then the remainder shared in
        // proportion to each consumer's headroom.
        let floors: u64 = req.iter().map(|r| r.floor_bytes).sum();
        let head: Vec<u64> = req.iter().map(|r| r.want_bytes - r.floor_bytes).collect();
        let head_total: u64 = head.iter().sum();
        let rest = ceiling - floors;
        let control: Vec<u64> = (0..CONSUMERS)
            .map(|i| req[i].floor_bytes + rest * head[i] / head_total)
            .collect();

        assert_eq!(g.total(), ceiling, "the even split left bytes unspent");
        assert!(
            g.get(Consumer::Texture) > control[Consumer::Texture.index()],
            "even {} vs proportional {} for the texture pool",
            g.get(Consumer::Texture),
            control[Consumer::Texture.index()]
        );
        // The number the module docs quote: proportional puts the texture pool
        // BELOW its own Low-tier ceiling (6 MiB) on a machine asking for High.
        assert!(
            control[Consumer::Texture.index()] < 6 * MIB,
            "the control is not actually starving anything: {control:?}"
        );
        assert!(
            g.get(Consumer::Texture) >= 6 * MIB,
            "the even split starves it too: {:?}",
            g.bytes
        );
    }

    /// A consumer that asks for nothing gets nothing, and leaves its share to
    /// the others — the shape a tier that turned the meshlet path off makes.
    #[test]
    fn a_consumer_that_is_not_live_leaves_its_share_behind() {
        let req = [
            BudgetRequest::default(),
            BudgetRequest::want(12 * MIB),
            BudgetRequest::want(32 * MIB),
        ];
        let g = arbitrate(20 * MIB, &req).expect("no floors at all");
        assert_eq!(g.get(Consumer::Geometry), 0);
        assert_eq!(g.total(), 20 * MIB, "{:?}", g.bytes);
        // Even between the two that are live, clamped at the smaller want.
        assert_eq!(g.get(Consumer::Texture), 10 * MIB);
        assert_eq!(g.get(Consumer::Shadow), 10 * MIB);
    }

    /// The sub-byte remainder is handed out in the fixed [`Consumer`] order,
    /// so the last bytes of a budget are a function of the request and not of
    /// the loop.
    #[test]
    fn the_sub_byte_remainder_is_deterministic() {
        let req = [
            BudgetRequest::want(10),
            BudgetRequest::want(10),
            BudgetRequest::want(10),
        ];
        // 5 bytes between three consumers: 1 each, then 2 left over.
        let g = arbitrate(5, &req).expect("no floors");
        assert_eq!(g.bytes, [2, 2, 1]);
        assert_eq!(g.total(), 5);
        let again = arbitrate(5, &req).expect("no floors");
        assert_eq!(g, again, "the arbiter is not a pure function");
    }

    /// A want below its own floor is normalized up rather than granted down —
    /// otherwise a caller could ask for less than the floor and be told it got
    /// what it asked for while residency sits above it.
    #[test]
    fn a_want_below_its_floor_is_normalized_to_the_floor() {
        let req = [
            BudgetRequest {
                floor_bytes: 8,
                want_bytes: 2,
            },
            BudgetRequest::want(4),
            BudgetRequest::default(),
        ];
        let g = arbitrate(100, &req).expect("the floors fit");
        assert_eq!(g.bytes, [8, 4, 0]);
        assert_eq!(
            BudgetRequest {
                floor_bytes: 8,
                want_bytes: 2
            }
            .normalized()
            .want_bytes,
            8
        );
    }

    /// **A floor larger than an even share is granted whole, and its owner
    /// still takes an even share of what is left.**
    ///
    /// The edge the P28.3 audit found unpinned, and it is worth pinning because
    /// the answer is not the one "water-fill" suggests. A hydraulic water-fill
    /// equalizes *levels*: with a floor of 60 under a 90 ceiling it would leave
    /// 60 alone and pour the remaining 30 into the other two. This one grants
    /// every floor first and then splits the **remainder** evenly, so the same
    /// case comes out 70/10/10 rather than 60/15/15.
    ///
    /// That is what the module docs say ("the remainder is water-filled") and
    /// it is the right rule here — a floor is content the consumer may not drop,
    /// not a head start it should be taxed for — but it is a choice, and an
    /// unarmed choice is one a refactor may reverse without anything noticing.
    #[test]
    fn a_floor_larger_than_an_even_share_is_granted_whole() {
        let req = [
            BudgetRequest {
                floor_bytes: 60,
                want_bytes: 80,
            },
            BudgetRequest::want(100),
            BudgetRequest::want(100),
        ];
        let g = arbitrate(90, &req).expect("60 of floor fits a 90 ceiling");
        assert_eq!(g.bytes, [70, 10, 10], "floors first, then the remainder");
        assert_eq!(g.total(), 90);
        // The control: the level-equalizing reading, which this is NOT.
        assert_ne!(g.bytes, [60, 15, 15]);
        // A floor that is already the consumer's whole want takes nothing more,
        // so the others split the rest between them and the arm above is about
        // the floor rather than about the order.
        let req = [
            BudgetRequest {
                floor_bytes: 50,
                want_bytes: 50,
            },
            BudgetRequest::want(100),
            BudgetRequest::want(100),
        ];
        assert_eq!(
            arbitrate(100, &req).expect("the floor fits").bytes,
            [50, 25, 25]
        );
    }

    /// The summary names every number it carries.
    #[test]
    fn the_budget_line_says_what_it_granted() {
        let g = arbitrate(64 * MIB, &shipped()).expect("the floors fit");
        let s = g.summary(64 * MIB);
        assert!(s.contains("stream budget"), "{s}");
        assert!(
            s.contains("geometry") && s.contains("texture") && s.contains("shadow"),
            "{s}"
        );
        assert!(s.contains("64.0 MiB"), "{s}");
        for c in Consumer::ALL {
            assert!(s.contains(c.label()), "{s} is missing {c:?}");
        }
    }
}
