//! Collision filtering + material combine rules (P12.1), shared by [`crate::d2`]
//! and [`crate::d3`].
//!
//! These are dimension-agnostic value types (like [`ContactPhase`](crate::d2::ContactPhase)),
//! so they live once at the crate root and both facades map them onto rapier's
//! `InteractionGroups` / `CoefficientCombineRule` inside their world modules.

/// Bitmask collision layers for a collider.
///
/// * `memberships` — the set of layers this collider *belongs to*.
/// * `filter` — the set of layers this collider *may interact with*.
///
/// Two colliders interact only if **each** is a member of a layer the other
/// filters for (rapier's symmetric AND test). The default is "all bits set" in
/// both fields, i.e. interact with everything — so a collider that never touches
/// these fields behaves exactly as before layers existed.
///
/// The 32 bits are named per-project by the `.infinity/collision_layers.toml`
/// registry (see `inf-editor-core::collision_layers`); the raw `u32`s here carry
/// no names of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CollisionLayers {
    /// Layers this collider belongs to (bit `i` ⇒ member of layer `i`).
    pub memberships: u32,
    /// Layers this collider will interact with (bit `i` ⇒ collides with layer `i`).
    pub filter: u32,
}

impl Default for CollisionLayers {
    fn default() -> Self {
        Self {
            memberships: u32::MAX,
            filter: u32::MAX,
        }
    }
}

impl CollisionLayers {
    /// Explicit membership + filter masks.
    pub fn new(memberships: u32, filter: u32) -> Self {
        Self {
            memberships,
            filter,
        }
    }
}

/// How two colliders' friction (or restitution) coefficients combine into the
/// single effective value used for their contact. Mirrors rapier's
/// `CoefficientCombineRule` (which itself resolves a disagreement between the two
/// colliders by "stronger rule wins": `Max > Multiply > Min > Average`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CombineRule {
    /// `(a + b) / 2` — the default; balanced and intuitive.
    #[default]
    Average,
    /// `min(a, b)` — "slippery/soft wins" (ice on any surface stays icy).
    Min,
    /// `a * b` — both must be high for a high result.
    Multiply,
    /// `max(a, b)` — "sticky/bouncy wins" (rubber on any surface grips).
    Max,
}
