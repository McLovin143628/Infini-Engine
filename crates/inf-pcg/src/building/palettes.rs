//! The seven **building archetypes**, shipped as code-level constants.
//!
//! # What a palette is
//!
//! An archetype is two things bolted together:
//!
//! * a **grammar text** in the P19.4 DSL — the module palette (what a wall panel,
//!   a post, a slab, a stair step, a desk *are*) plus the wall rules that lay
//!   modules along a run. It is parsed by [`Grammar::parse`], the same parser the
//!   `grammar.rules` node uses, so an archetype is exactly as expressive as
//!   anything an author can type — and a test parses all seven.
//! * a **table of plan parameters** — storey range, room sizes, corridor width,
//!   opening dimensions, the weighted room-type table and the per-room-type
//!   furniture set.
//!
//! # No imported art, and no longer any cubes either
//!
//! Every module here declares **no mesh GUID in its text**, and it never will:
//! a building must need no imported art to exist, which is what "enterable"
//! requires and what an engine can ship without a licence question. The
//! `collider` attribute (P19.5's one DSL addition) is what makes a module
//! solid.
//!
//! What island wave I8b changed is what a module *draws*. [`grammar`] stamps
//! each module with the GUID of its **shape family**
//! ([`super::modules`]) — a framed panel, a glazed leaf, a fascia'd deck, a
//! legged table — and the assembler writes the module's own half-extents onto
//! the instance, so the drawn thing is the size of the solid thing. The text
//! stays free of GUIDs, an authored `mesh <guid>` still wins over the derived
//! one, and a palette that adds a module and forgets to classify it fails
//! `modules::tests::every_palette_module_has_a_shape` rather than silently
//! going back to a box.
//!
//! [`grammar`]: BuildingArchetype::grammar
//!
//! # Why constants and not assets
//!
//! v1 ships the seven as `&'static` data. An archetype has no identity a user
//! edits, no dependency edges, and no versioning story yet; making it an asset
//! kind would buy a `.inf_barch` format, a sidecar, a migration ladder and a
//! Content Drawer glyph before anybody has asked to author one. The seam is
//! ready for that move — [`BuildingArchetype`] is plain data and
//! [`archetype`] is the only lookup — and it is stated here rather than
//! discovered later.
//!
//! # Units
//!
//! Metres and m², everywhere, per architecture rule 6. A `FurnitureDef`'s
//! `half` is half-extents in metres; `per_10m2` is pieces per ten square metres
//! of room floor.

use super::RoomType;

/// Which of the seven palettes a plan is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArchetypeId {
    Office,
    Apartment,
    Industrial,
    House,
    Estate,
    Hotel,
    Shop,
}

impl ArchetypeId {
    /// Every archetype, in the canonical order the node's choice param and the
    /// gate both use.
    pub const ALL: [ArchetypeId; 7] = [
        ArchetypeId::Office,
        ArchetypeId::Apartment,
        ArchetypeId::Industrial,
        ArchetypeId::House,
        ArchetypeId::Estate,
        ArchetypeId::Hotel,
        ArchetypeId::Shop,
    ];

    /// The stable identifier used in the node param, diagnostics and traces.
    pub fn name(self) -> &'static str {
        match self {
            ArchetypeId::Office => "Office",
            ArchetypeId::Apartment => "Apartment",
            ArchetypeId::Industrial => "Industrial",
            ArchetypeId::House => "House",
            ArchetypeId::Estate => "Estate",
            ArchetypeId::Hotel => "Hotel",
            ArchetypeId::Shop => "Shop",
        }
    }

    /// Parse a [`name`](Self::name). Unknown text answers `None` so the lowerer
    /// can raise a node-anchored warning instead of silently picking one.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == s)
    }
}

/// One weighted entry of an archetype's room-type table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoomWeight {
    pub kind: RoomType,
    /// Relative weight for the hashed pick. Non-positive entries never occur.
    pub weight: f64,
}

/// One piece of furniture a room type may hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FurnitureDef {
    /// The palette module placed — must be declared by the archetype's grammar
    /// text, which a unit test checks for all seven.
    pub module: &'static str,
    /// Half-extents in metres `(x, y, z)`, in the **piece's own frame**: `+Z` is
    /// the direction it faces (away from the wall, for a wall-aligned piece).
    pub half: [f64; 3],
    /// Wall-aligned pieces are stationed along the room's inset perimeter;
    /// free-standing pieces go on a jittered grid in the room's interior.
    pub against_wall: bool,
    /// Pieces per ten square metres of room floor — the density knob, so a big
    /// room gets proportionally more without a per-room count being authored.
    pub per_10m2: f64,
    /// Minimum centre-to-centre distance to any other piece, in metres.
    pub clearance: f64,
}

/// A complete building palette: the grammar, the plan parameters, the room table
/// and the furniture sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildingArchetype {
    pub id: ArchetypeId,
    /// Human-readable name for diagnostics.
    pub display: &'static str,
    /// The module palette + wall rules, in the P19.4 grammar DSL.
    pub rules: &'static str,

    // ── the modules the assembler looks up by name ──────────────────────────
    /// The rule an **exterior** wall run expands from.
    pub exterior_axiom: &'static str,
    /// The rule an **interior** wall run expands from.
    pub interior_axiom: &'static str,
    /// **The glazed leaf hung in every window void** (island wave I8b).
    ///
    /// Declared by the palette and placed by the assembler exactly like the
    /// lintel and the parapet — and, unlike them, placed **without a collider**:
    /// a window you cannot see through is a wall, and
    /// [`BuildingPlan::opening_is_clear`](super::BuildingPlan::opening_is_clear)
    /// is an assertion about *solids*, which a pane is not. It is the reason
    /// the assembler's instance list is now one aligned prefix plus a decoration
    /// tail (see [`super::assemble`](mod@super::assemble)).
    pub pane: &'static str,
    /// The header placed over every opening.
    pub lintel: &'static str,
    /// The solid wall placed under a window (`sill` metres tall).
    pub parapet: &'static str,
    /// The floor slab under a room.
    pub slab: &'static str,
    /// One tread of a stair flight.
    pub step: &'static str,
    /// The roof deck over the top storey.
    pub roof: &'static str,

    // ── the stack ───────────────────────────────────────────────────────────
    /// Inclusive storey count range, drawn from the counter hash.
    pub floors: (u32, u32),
    /// Metres from one walking surface to the next.
    pub floor_height: f64,
    /// Slab thickness in metres (the walking surface sits on top of it).
    pub slab_thickness: f64,

    // ── the plan ────────────────────────────────────────────────────────────
    /// The smallest a **partitioned** room may be on either axis, in metres.
    ///
    /// Precisely: every leaf [`partition_floor`](super::partition_floor) emits
    /// clears this on both axes, and every slab the core carve hands it does
    /// too — asserted over a lot-size sweep by
    /// `plan::tests::no_planned_room_falls_below_min_room`.
    ///
    /// **The two structural rooms are exempt, and deliberately so.** The stair
    /// core is `stair_size` and the corridor is `corridor_width`, both of which
    /// a palette may legitimately set below `min_room` — a 2.2 m stairwell in a
    /// house is right, and forcing it to the room minimum would make a small
    /// house mostly stairwell. They are not products of a split, so the
    /// connectivity proof (which is about *split* leaves) does not rest on them;
    /// what it needs from them is only that the runs they share with their
    /// neighbours can host a door, which `connect`'s filter enforces directly.
    pub min_room: f64,
    /// A region above this area is split again, in m².
    pub max_room_area: f64,
    /// `true` when the core strip carries a circulation corridor (offices,
    /// apartments, hotels) rather than a plain hall.
    pub corridor: bool,
    /// Metres — the width of the corridor band inside the core strip.
    pub corridor_width: f64,
    /// Wall thickness in metres (exterior and interior alike, in v1).
    pub wall_thickness: f64,
    /// The stair core's plan size in metres `(along the strip, across it)`.
    pub stair_size: (f64, f64),

    // ── openings ────────────────────────────────────────────────────────────
    pub door_width: f64,
    pub door_height: f64,
    pub window_width: f64,
    pub window_sill: f64,
    pub window_head: f64,
    /// Metres of exterior wall per window — the façade rhythm.
    pub window_pitch: f64,

    // ── contents ────────────────────────────────────────────────────────────
    /// Room types the ground floor draws from.
    pub ground_rooms: &'static [RoomWeight],
    /// Room types the upper floors draw from.
    pub upper_rooms: &'static [RoomWeight],
    /// Per-room-type furniture sets. A room type absent from this table is
    /// simply unfurnished.
    pub furniture: &'static [(RoomType, &'static [FurnitureDef])],
}

impl BuildingArchetype {
    /// The parsed grammar. Every shipped palette parses — asserted for all seven
    /// by [`every_palette_parses`](self::tests::every_palette_parses) — so a
    /// caller may treat a failure here as a programming error in a *new*
    /// palette rather than as authored input.
    pub fn grammar(&self) -> Result<crate::grammar::Grammar, crate::grammar::GrammarError> {
        let mut g = crate::grammar::Grammar::parse(self.rules)?;
        // **The one place a palette becomes a grammar**, so it is the one place
        // module meshes are stamped (island wave I8b). The palette text declares
        // no GUIDs — see `super::modules` for why a derived id is right for
        // geometry that is a function of the module's own name — and every
        // consumer downstream reads `ModuleDef::mesh` rather than deriving it,
        // which is what keeps the assembler and `expand_span` from growing two
        // answers.
        g.stamp_module_meshes();
        Ok(g)
    }

    /// The furniture set for `kind`, or an empty slice.
    pub fn furniture_for(&self, kind: RoomType) -> &'static [FurnitureDef] {
        self.furniture
            .iter()
            .find(|(k, _)| *k == kind)
            .map(|(_, f)| *f)
            .unwrap_or(&[])
    }

    /// The room-type table `floor` draws from.
    pub fn room_table(&self, floor: u32) -> &'static [RoomWeight] {
        if floor == 0 {
            self.ground_rooms
        } else {
            self.upper_rooms
        }
    }
}

/// The archetype for `id`.
pub fn archetype(id: ArchetypeId) -> &'static BuildingArchetype {
    match id {
        ArchetypeId::Office => &OFFICE,
        ArchetypeId::Apartment => &APARTMENT,
        ArchetypeId::Industrial => &INDUSTRIAL,
        ArchetypeId::House => &HOUSE,
        ArchetypeId::Estate => &ESTATE,
        ArchetypeId::Hotel => &HOTEL,
        ArchetypeId::Shop => &SHOP,
    }
}

/// All seven archetypes, in [`ArchetypeId::ALL`] order.
pub fn archetypes() -> [&'static BuildingArchetype; 7] {
    ArchetypeId::ALL.map(archetype)
}

// ── the shared structural vocabulary ────────────────────────────────────────
//
// Every palette declares the same six *structural* module names — `Lintel`,
// `Parapet`, `Slab`, `Step`, `Roof` and at least one wall panel — because the
// assembler looks them up by name. Their dimensions differ per archetype (a
// factory's panel is not a house's), so the text is written out per palette
// rather than concatenated from fragments: a palette is meant to be readable as
// one thing, and a reader who wants to know how thick an office wall is should
// find the answer in `OFFICE.rules` and nowhere else.
//
// `size` is the metres a module consumes **along the wall run**; `offset` is in
// the slot frame (`+X` right of travel, `+Y` up, `+Z` along the run), anchored at
// the start of the module's interval; `collider` is half-extents in that same
// frame.

const OFFICE: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::Office,
    display: "Office block",
    rules: "\
# Office block — a curtain-wall façade on a 1.5 m module, light partitions.
module Mullion  = size 0.3 offset 0,1.75,0.15 collider 0.15,1.75,0.15
module Glazing  = size 1.5 offset 0,1.75,0.75 collider 0.06,1.75,0.75
module Spandrel = size 1.5 offset 0,1.75,0.75 collider 0.09,1.75,0.75
module Partition= size 1.2 offset 0,1.6,0.6   collider 0.06,1.6,0.6
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1   offset 0,0,0       collider 0.09,0.2,0.5
module Parapet  = size 1   offset 0,0,0       collider 0.09,0.5,0.5
module Slab     = size 1   offset 0,0,0       collider 1,0.1,1
module Step     = size 1   offset 0,0,0       collider 0.6,0.09,0.14
module Roof     = size 1   offset 0,0,0       collider 1,0.12,1
module Desk     = size 1   offset 0,0,0       collider 0.7,0.37,0.35
module Cabinet  = size 1   offset 0,0,0       collider 0.5,0.9,0.25
module Table    = size 1   offset 0,0,0       collider 1.2,0.37,0.6
module Plant    = size 1   offset 0,0,0       collider 0.3,0.6,0.3
module Counter  = size 1   offset 0,0,0       collider 1.2,0.55,0.35
module Locker   = size 1   offset 0,0,0       collider 0.4,0.9,0.3

Facade -> Mullion Bay* Mullion
Bay    -> Glazing | Spandrel@0.35
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (3, 8),
    floor_height: 3.6,
    slab_thickness: 0.25,
    min_room: 4.0,
    max_room_area: 90.0,
    corridor: true,
    corridor_width: 2.4,
    wall_thickness: 0.2,
    stair_size: (5.0, 4.0),
    door_width: 1.0,
    door_height: 2.1,
    window_width: 1.5,
    window_sill: 0.9,
    window_head: 2.7,
    window_pitch: 3.0,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Lobby,
            weight: 2.0,
        },
        RoomWeight {
            kind: RoomType::Meeting,
            weight: 1.0,
        },
        RoomWeight {
            kind: RoomType::Service,
            weight: 1.0,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Office,
            weight: 4.0,
        },
        RoomWeight {
            kind: RoomType::Meeting,
            weight: 2.0,
        },
        RoomWeight {
            kind: RoomType::Service,
            weight: 1.0,
        },
    ],
    furniture: &[
        (
            RoomType::Office,
            &[
                FurnitureDef {
                    module: "Desk",
                    half: [0.7, 0.37, 0.35],
                    against_wall: false,
                    per_10m2: 1.4,
                    clearance: 1.6,
                },
                FurnitureDef {
                    module: "Cabinet",
                    half: [0.5, 0.9, 0.25],
                    against_wall: true,
                    per_10m2: 0.6,
                    clearance: 1.2,
                },
            ],
        ),
        (
            RoomType::Meeting,
            &[
                FurnitureDef {
                    module: "Table",
                    half: [1.2, 0.37, 0.6],
                    against_wall: false,
                    per_10m2: 0.5,
                    clearance: 2.6,
                },
                FurnitureDef {
                    module: "Cabinet",
                    half: [0.5, 0.9, 0.25],
                    against_wall: true,
                    per_10m2: 0.4,
                    clearance: 1.4,
                },
            ],
        ),
        (
            RoomType::Lobby,
            &[
                FurnitureDef {
                    module: "Counter",
                    half: [1.2, 0.55, 0.35],
                    against_wall: true,
                    per_10m2: 0.3,
                    clearance: 3.0,
                },
                FurnitureDef {
                    module: "Plant",
                    half: [0.3, 0.6, 0.3],
                    against_wall: true,
                    per_10m2: 0.5,
                    clearance: 2.0,
                },
            ],
        ),
        (
            RoomType::Service,
            &[FurnitureDef {
                module: "Locker",
                half: [0.4, 0.9, 0.3],
                against_wall: true,
                per_10m2: 1.2,
                clearance: 0.9,
            }],
        ),
    ],
};

const APARTMENT: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::Apartment,
    display: "Apartment block",
    rules: "\
# Apartment block — punched-window masonry, double-loaded corridor.
module Pier     = size 0.4 offset 0,1.4,0.2  collider 0.14,1.4,0.2
module Wall     = size 1.2 offset 0,1.4,0.6  collider 0.14,1.4,0.6
module Balcony  = size 2.4 offset 0,1.4,1.2  collider 0.14,1.4,1.2
module Partition= size 1.0 offset 0,1.3,0.5  collider 0.06,1.3,0.5
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1   offset 0,0,0      collider 0.14,0.2,0.5
module Parapet  = size 1   offset 0,0,0      collider 0.14,0.5,0.5
module Slab     = size 1   offset 0,0,0      collider 1,0.11,1
module Step     = size 1   offset 0,0,0      collider 0.55,0.09,0.14
module Roof     = size 1   offset 0,0,0      collider 1,0.12,1
module Bed      = size 1   offset 0,0,0      collider 0.9,0.28,1.0
module Wardrobe = size 1   offset 0,0,0      collider 0.6,1.05,0.3
module Sofa     = size 1   offset 0,0,0      collider 0.95,0.4,0.45
module Table    = size 1   offset 0,0,0      collider 0.7,0.37,0.45
module Units    = size 1   offset 0,0,0      collider 0.6,0.45,0.3
module Basin    = size 1   offset 0,0,0      collider 0.3,0.42,0.25
module Shelf    = size 1   offset 0,0,0      collider 0.5,0.9,0.25

Facade -> Pier Bay* Pier
Bay    -> Wall | Balcony@0.2
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (3, 6),
    floor_height: 2.9,
    slab_thickness: 0.22,
    min_room: 2.8,
    max_room_area: 26.0,
    corridor: true,
    corridor_width: 1.6,
    wall_thickness: 0.18,
    stair_size: (4.4, 3.2),
    door_width: 0.9,
    door_height: 2.05,
    window_width: 1.2,
    window_sill: 0.85,
    window_head: 2.3,
    window_pitch: 3.2,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Lobby,
            weight: 1.5,
        },
        RoomWeight {
            kind: RoomType::Living,
            weight: 2.0,
        },
        RoomWeight {
            kind: RoomType::Storage,
            weight: 1.0,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Living,
            weight: 2.0,
        },
        RoomWeight {
            kind: RoomType::Bedroom,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Kitchen,
            weight: 1.5,
        },
        RoomWeight {
            kind: RoomType::Bath,
            weight: 1.5,
        },
    ],
    furniture: &[
        (
            RoomType::Bedroom,
            &[
                FurnitureDef {
                    module: "Bed",
                    half: [0.9, 0.28, 1.0],
                    against_wall: true,
                    per_10m2: 1.0,
                    clearance: 2.4,
                },
                FurnitureDef {
                    module: "Wardrobe",
                    half: [0.6, 1.05, 0.3],
                    against_wall: true,
                    per_10m2: 0.7,
                    clearance: 1.6,
                },
            ],
        ),
        (
            RoomType::Living,
            &[
                FurnitureDef {
                    module: "Sofa",
                    half: [0.95, 0.4, 0.45],
                    against_wall: true,
                    per_10m2: 0.7,
                    clearance: 2.2,
                },
                FurnitureDef {
                    module: "Table",
                    half: [0.7, 0.37, 0.45],
                    against_wall: false,
                    per_10m2: 0.5,
                    clearance: 1.8,
                },
            ],
        ),
        (
            RoomType::Kitchen,
            &[FurnitureDef {
                module: "Units",
                half: [0.6, 0.45, 0.3],
                against_wall: true,
                per_10m2: 1.8,
                clearance: 1.3,
            }],
        ),
        (
            RoomType::Bath,
            &[FurnitureDef {
                module: "Basin",
                half: [0.3, 0.42, 0.25],
                against_wall: true,
                per_10m2: 1.2,
                clearance: 1.0,
            }],
        ),
        (
            RoomType::Storage,
            &[FurnitureDef {
                module: "Shelf",
                half: [0.5, 0.9, 0.25],
                against_wall: true,
                per_10m2: 1.4,
                clearance: 1.1,
            }],
        ),
    ],
};

const INDUSTRIAL: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::Industrial,
    display: "Factory / warehouse",
    rules: "\
# Factory / warehouse — profiled cladding on a wide bay, few internal walls.
module Column   = size 0.6 offset 0,3.2,0.3  collider 0.3,3.2,0.3
module Cladding = size 3.0 offset 0,3.2,1.5  collider 0.12,3.2,1.5
module RollDoor = size 4.0 offset 0,3.2,2.0  collider 0.12,3.2,2.0
module Partition= size 2.0 offset 0,1.6,1.0  collider 0.1,1.6,1.0
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1   offset 0,0,0      collider 0.12,0.3,0.5
module Parapet  = size 1   offset 0,0,0      collider 0.12,0.5,0.5
module Slab     = size 1   offset 0,0,0      collider 1,0.2,1
module Step     = size 1   offset 0,0,0      collider 0.7,0.1,0.15
module Roof     = size 1   offset 0,0,0      collider 1,0.15,1
module Rack     = size 1   offset 0,0,0      collider 1.3,1.8,0.6
module Crate    = size 1   offset 0,0,0      collider 0.6,0.6,0.6
module Bench    = size 1   offset 0,0,0      collider 1.0,0.45,0.4
module Locker   = size 1   offset 0,0,0      collider 0.4,0.9,0.3

Facade -> Column Bay* Column
Bay    -> Cladding | RollDoor@0.12
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (1, 2),
    floor_height: 6.5,
    slab_thickness: 0.3,
    min_room: 8.0,
    max_room_area: 600.0,
    corridor: false,
    corridor_width: 3.0,
    wall_thickness: 0.25,
    stair_size: (5.0, 3.2),
    door_width: 1.2,
    door_height: 2.4,
    window_width: 2.0,
    window_sill: 3.0,
    window_head: 5.0,
    window_pitch: 9.0,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Workshop,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Storage,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Service,
            weight: 1.0,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Storage,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Office,
            weight: 1.0,
        },
    ],
    furniture: &[
        (
            RoomType::Storage,
            &[
                FurnitureDef {
                    module: "Rack",
                    half: [1.3, 1.8, 0.6],
                    against_wall: false,
                    per_10m2: 0.35,
                    clearance: 3.4,
                },
                FurnitureDef {
                    module: "Crate",
                    half: [0.6, 0.6, 0.6],
                    against_wall: true,
                    per_10m2: 0.4,
                    clearance: 1.6,
                },
            ],
        ),
        (
            RoomType::Workshop,
            &[
                FurnitureDef {
                    module: "Bench",
                    half: [1.0, 0.45, 0.4],
                    against_wall: true,
                    per_10m2: 0.5,
                    clearance: 2.6,
                },
                FurnitureDef {
                    module: "Crate",
                    half: [0.6, 0.6, 0.6],
                    against_wall: false,
                    per_10m2: 0.25,
                    clearance: 2.2,
                },
            ],
        ),
        (
            RoomType::Service,
            &[FurnitureDef {
                module: "Locker",
                half: [0.4, 0.9, 0.3],
                against_wall: true,
                per_10m2: 1.0,
                clearance: 1.0,
            }],
        ),
    ],
};

const HOUSE: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::House,
    display: "House",
    rules: "\
# House — small footprint, a hall through the middle, domestic rooms.
module Quoin    = size 0.35 offset 0,1.3,0.175 collider 0.125,1.3,0.175
module Brick    = size 1.0  offset 0,1.3,0.5   collider 0.125,1.3,0.5
module Partition= size 0.9  offset 0,1.25,0.45 collider 0.05,1.25,0.45
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1    offset 0,0,0       collider 0.125,0.2,0.5
module Parapet  = size 1    offset 0,0,0       collider 0.125,0.5,0.5
module Slab     = size 1    offset 0,0,0       collider 1,0.1,1
module Step     = size 1    offset 0,0,0       collider 0.45,0.09,0.13
module Roof     = size 1    offset 0,0,0       collider 1,0.12,1
module Bed      = size 1    offset 0,0,0       collider 0.75,0.28,1.0
module Wardrobe = size 1    offset 0,0,0       collider 0.55,1.0,0.3
module Sofa     = size 1    offset 0,0,0       collider 0.9,0.4,0.45
module Table    = size 1    offset 0,0,0       collider 0.75,0.37,0.5
module Units    = size 1    offset 0,0,0       collider 0.6,0.45,0.3
module Basin    = size 1    offset 0,0,0       collider 0.3,0.42,0.25
module Shelf    = size 1    offset 0,0,0       collider 0.45,0.9,0.22

Facade -> Quoin Brick+ Quoin
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (1, 3),
    floor_height: 2.7,
    slab_thickness: 0.2,
    min_room: 2.6,
    max_room_area: 22.0,
    corridor: false,
    corridor_width: 1.2,
    wall_thickness: 0.2,
    stair_size: (3.2, 2.2),
    door_width: 0.85,
    door_height: 2.0,
    window_width: 1.1,
    window_sill: 0.9,
    window_head: 2.1,
    window_pitch: 3.0,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Living,
            weight: 2.5,
        },
        RoomWeight {
            kind: RoomType::Kitchen,
            weight: 2.0,
        },
        RoomWeight {
            kind: RoomType::Bath,
            weight: 1.0,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Bedroom,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Bath,
            weight: 1.5,
        },
        RoomWeight {
            kind: RoomType::Storage,
            weight: 1.0,
        },
    ],
    furniture: &[
        (
            RoomType::Bedroom,
            &[
                FurnitureDef {
                    module: "Bed",
                    half: [0.75, 0.28, 1.0],
                    against_wall: true,
                    per_10m2: 1.0,
                    clearance: 2.2,
                },
                FurnitureDef {
                    module: "Wardrobe",
                    half: [0.55, 1.0, 0.3],
                    against_wall: true,
                    per_10m2: 0.7,
                    clearance: 1.5,
                },
            ],
        ),
        (
            RoomType::Living,
            &[
                FurnitureDef {
                    module: "Sofa",
                    half: [0.9, 0.4, 0.45],
                    against_wall: true,
                    per_10m2: 0.8,
                    clearance: 2.0,
                },
                FurnitureDef {
                    module: "Table",
                    half: [0.75, 0.37, 0.5],
                    against_wall: false,
                    per_10m2: 0.5,
                    clearance: 1.7,
                },
            ],
        ),
        (
            RoomType::Kitchen,
            &[FurnitureDef {
                module: "Units",
                half: [0.6, 0.45, 0.3],
                against_wall: true,
                per_10m2: 1.8,
                clearance: 1.3,
            }],
        ),
        (
            RoomType::Bath,
            &[FurnitureDef {
                module: "Basin",
                half: [0.3, 0.42, 0.25],
                against_wall: true,
                per_10m2: 1.2,
                clearance: 1.0,
            }],
        ),
        (
            RoomType::Storage,
            &[FurnitureDef {
                module: "Shelf",
                half: [0.45, 0.9, 0.22],
                against_wall: true,
                per_10m2: 1.2,
                clearance: 1.0,
            }],
        ),
    ],
};

const ESTATE: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::Estate,
    display: "Estate house",
    rules: "\
# Estate — a large house: stone piers, tall rooms, generous plan.
module Pier     = size 0.6 offset 0,2.0,0.3  collider 0.2,2.0,0.3
module Ashlar   = size 1.4 offset 0,2.0,0.7  collider 0.2,2.0,0.7
module Pilaster = size 0.5 offset 0,2.0,0.25 collider 0.24,2.0,0.25
module Partition= size 1.1 offset 0,1.9,0.55 collider 0.07,1.9,0.55
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1   offset 0,0,0      collider 0.2,0.25,0.5
module Parapet  = size 1   offset 0,0,0      collider 0.2,0.5,0.5
module Slab     = size 1   offset 0,0,0      collider 1,0.14,1
module Step     = size 1   offset 0,0,0      collider 0.7,0.09,0.15
module Roof     = size 1   offset 0,0,0      collider 1,0.16,1
module Bed      = size 1   offset 0,0,0      collider 0.95,0.3,1.05
module Wardrobe = size 1   offset 0,0,0      collider 0.7,1.1,0.32
module Sofa     = size 1   offset 0,0,0      collider 1.1,0.42,0.5
module Table    = size 1   offset 0,0,0      collider 1.4,0.38,0.7
module Units    = size 1   offset 0,0,0      collider 0.7,0.45,0.32
module Basin    = size 1   offset 0,0,0      collider 0.35,0.42,0.27
module Plant    = size 1   offset 0,0,0      collider 0.35,0.7,0.35
module Shelf    = size 1   offset 0,0,0      collider 0.6,1.1,0.28

Facade -> Pier Bay* Pier
Bay    -> Ashlar | Pilaster@0.3
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (2, 4),
    floor_height: 4.0,
    slab_thickness: 0.28,
    min_room: 4.5,
    max_room_area: 70.0,
    corridor: false,
    corridor_width: 2.0,
    wall_thickness: 0.28,
    stair_size: (5.5, 4.0),
    door_width: 1.1,
    door_height: 2.4,
    window_width: 1.4,
    window_sill: 0.85,
    window_head: 3.0,
    window_pitch: 4.0,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Lobby,
            weight: 1.5,
        },
        RoomWeight {
            kind: RoomType::Living,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Kitchen,
            weight: 1.5,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Bedroom,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Guest,
            weight: 2.0,
        },
        RoomWeight {
            kind: RoomType::Bath,
            weight: 1.5,
        },
    ],
    furniture: &[
        (
            RoomType::Bedroom,
            &[
                FurnitureDef {
                    module: "Bed",
                    half: [0.95, 0.3, 1.05],
                    against_wall: true,
                    per_10m2: 0.7,
                    clearance: 2.8,
                },
                FurnitureDef {
                    module: "Wardrobe",
                    half: [0.7, 1.1, 0.32],
                    against_wall: true,
                    per_10m2: 0.5,
                    clearance: 1.9,
                },
            ],
        ),
        (
            RoomType::Guest,
            &[
                FurnitureDef {
                    module: "Bed",
                    half: [0.95, 0.3, 1.05],
                    against_wall: true,
                    per_10m2: 0.6,
                    clearance: 2.8,
                },
                FurnitureDef {
                    module: "Shelf",
                    half: [0.6, 1.1, 0.28],
                    against_wall: true,
                    per_10m2: 0.5,
                    clearance: 1.7,
                },
            ],
        ),
        (
            RoomType::Living,
            &[
                FurnitureDef {
                    module: "Sofa",
                    half: [1.1, 0.42, 0.5],
                    against_wall: true,
                    per_10m2: 0.6,
                    clearance: 2.6,
                },
                FurnitureDef {
                    module: "Table",
                    half: [1.4, 0.38, 0.7],
                    against_wall: false,
                    per_10m2: 0.35,
                    clearance: 3.0,
                },
            ],
        ),
        (
            RoomType::Lobby,
            &[FurnitureDef {
                module: "Plant",
                half: [0.35, 0.7, 0.35],
                against_wall: true,
                per_10m2: 0.6,
                clearance: 2.2,
            }],
        ),
        (
            RoomType::Kitchen,
            &[FurnitureDef {
                module: "Units",
                half: [0.7, 0.45, 0.32],
                against_wall: true,
                per_10m2: 1.6,
                clearance: 1.5,
            }],
        ),
        (
            RoomType::Bath,
            &[FurnitureDef {
                module: "Basin",
                half: [0.35, 0.42, 0.27],
                against_wall: true,
                per_10m2: 1.0,
                clearance: 1.2,
            }],
        ),
    ],
};

const HOTEL: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::Hotel,
    display: "Hotel",
    rules: "\
# Hotel — a long double-loaded corridor of identical guest rooms.
module Pier     = size 0.35 offset 0,1.55,0.175 collider 0.125,1.55,0.175
module Wall     = size 1.3  offset 0,1.55,0.65  collider 0.125,1.55,0.65
module Panelled = size 1.3  offset 0,1.55,0.65  collider 0.14,1.55,0.65
module Partition= size 1.0  offset 0,1.5,0.5    collider 0.06,1.5,0.5
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1    offset 0,0,0        collider 0.125,0.2,0.5
module Parapet  = size 1    offset 0,0,0        collider 0.125,0.5,0.5
module Slab     = size 1    offset 0,0,0        collider 1,0.12,1
module Step     = size 1    offset 0,0,0        collider 0.6,0.09,0.14
module Roof     = size 1    offset 0,0,0        collider 1,0.13,1
module Bed      = size 1    offset 0,0,0        collider 0.85,0.28,1.0
module Wardrobe = size 1    offset 0,0,0        collider 0.55,1.0,0.3
module Desk     = size 1    offset 0,0,0        collider 0.6,0.37,0.3
module Basin    = size 1    offset 0,0,0        collider 0.3,0.42,0.25
module Counter  = size 1    offset 0,0,0        collider 1.4,0.55,0.35
module Sofa     = size 1    offset 0,0,0        collider 0.9,0.4,0.45
module Plant    = size 1    offset 0,0,0        collider 0.3,0.65,0.3

Facade -> Pier Bay* Pier
Bay    -> Wall | Panelled@0.25
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (4, 10),
    floor_height: 3.1,
    slab_thickness: 0.22,
    min_room: 3.4,
    max_room_area: 30.0,
    corridor: true,
    corridor_width: 1.8,
    wall_thickness: 0.18,
    stair_size: (4.8, 3.4),
    door_width: 0.95,
    door_height: 2.1,
    window_width: 1.3,
    window_sill: 0.8,
    window_head: 2.4,
    window_pitch: 3.4,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Lobby,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Service,
            weight: 1.0,
        },
        RoomWeight {
            kind: RoomType::Meeting,
            weight: 1.0,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Guest,
            weight: 6.0,
        },
        RoomWeight {
            kind: RoomType::Service,
            weight: 1.0,
        },
    ],
    furniture: &[
        (
            RoomType::Guest,
            &[
                FurnitureDef {
                    module: "Bed",
                    half: [0.85, 0.28, 1.0],
                    against_wall: true,
                    per_10m2: 1.0,
                    clearance: 2.4,
                },
                FurnitureDef {
                    module: "Wardrobe",
                    half: [0.55, 1.0, 0.3],
                    against_wall: true,
                    per_10m2: 0.6,
                    clearance: 1.5,
                },
                FurnitureDef {
                    module: "Desk",
                    half: [0.6, 0.37, 0.3],
                    against_wall: true,
                    per_10m2: 0.6,
                    clearance: 1.4,
                },
            ],
        ),
        (
            RoomType::Lobby,
            &[
                FurnitureDef {
                    module: "Counter",
                    half: [1.4, 0.55, 0.35],
                    against_wall: true,
                    per_10m2: 0.25,
                    clearance: 3.4,
                },
                FurnitureDef {
                    module: "Sofa",
                    half: [0.9, 0.4, 0.45],
                    against_wall: false,
                    per_10m2: 0.4,
                    clearance: 2.4,
                },
                FurnitureDef {
                    module: "Plant",
                    half: [0.3, 0.65, 0.3],
                    against_wall: true,
                    per_10m2: 0.5,
                    clearance: 2.0,
                },
            ],
        ),
        (
            RoomType::Service,
            &[FurnitureDef {
                module: "Basin",
                half: [0.3, 0.42, 0.25],
                against_wall: true,
                per_10m2: 1.0,
                clearance: 1.1,
            }],
        ),
    ],
};

const SHOP: BuildingArchetype = BuildingArchetype {
    id: ArchetypeId::Shop,
    display: "Shop",
    rules: "\
# Shop — a glazed shopfront over a deep retail floor with a back-of-house.
module Stall    = size 0.3 offset 0,1.6,0.15 collider 0.12,1.6,0.15
module Shopfront= size 1.8 offset 0,1.6,0.9  collider 0.05,1.6,0.9
module Solid    = size 1.2 offset 0,1.6,0.6  collider 0.12,1.6,0.6
module Partition= size 1.0 offset 0,1.5,0.5  collider 0.06,1.5,0.5
module Pane     = size 1   offset 0,0,0
module Lintel   = size 1   offset 0,0,0      collider 0.12,0.25,0.5
module Parapet  = size 1   offset 0,0,0      collider 0.12,0.5,0.5
module Slab     = size 1   offset 0,0,0      collider 1,0.12,1
module Step     = size 1   offset 0,0,0      collider 0.5,0.09,0.14
module Roof     = size 1   offset 0,0,0      collider 1,0.13,1
module Shelf    = size 1   offset 0,0,0      collider 1.0,1.0,0.35
module Counter  = size 1   offset 0,0,0      collider 1.1,0.55,0.35
module Crate    = size 1   offset 0,0,0      collider 0.5,0.5,0.5
module Rack     = size 1   offset 0,0,0      collider 0.9,1.5,0.45

Facade -> Stall Bay* Stall
Bay    -> Shopfront | Solid@0.4
Inner  -> Partition+
",
    exterior_axiom: "Facade",
    interior_axiom: "Inner",
    pane: "Pane",
    lintel: "Lintel",
    parapet: "Parapet",
    slab: "Slab",
    step: "Step",
    roof: "Roof",
    floors: (1, 2),
    floor_height: 3.4,
    slab_thickness: 0.2,
    min_room: 3.0,
    max_room_area: 60.0,
    corridor: false,
    corridor_width: 1.4,
    wall_thickness: 0.18,
    stair_size: (3.6, 2.6),
    door_width: 1.1,
    door_height: 2.2,
    window_width: 1.8,
    window_sill: 0.4,
    window_head: 2.8,
    window_pitch: 2.6,
    ground_rooms: &[
        RoomWeight {
            kind: RoomType::Retail,
            weight: 4.0,
        },
        RoomWeight {
            kind: RoomType::Storage,
            weight: 1.5,
        },
        RoomWeight {
            kind: RoomType::Service,
            weight: 1.0,
        },
    ],
    upper_rooms: &[
        RoomWeight {
            kind: RoomType::Storage,
            weight: 3.0,
        },
        RoomWeight {
            kind: RoomType::Office,
            weight: 1.5,
        },
    ],
    furniture: &[
        (
            RoomType::Retail,
            &[
                FurnitureDef {
                    module: "Shelf",
                    half: [1.0, 1.0, 0.35],
                    against_wall: false,
                    per_10m2: 0.6,
                    clearance: 2.6,
                },
                FurnitureDef {
                    module: "Counter",
                    half: [1.1, 0.55, 0.35],
                    against_wall: true,
                    per_10m2: 0.3,
                    clearance: 3.0,
                },
            ],
        ),
        (
            RoomType::Storage,
            &[
                FurnitureDef {
                    module: "Rack",
                    half: [0.9, 1.5, 0.45],
                    against_wall: true,
                    per_10m2: 0.8,
                    clearance: 2.0,
                },
                FurnitureDef {
                    module: "Crate",
                    half: [0.5, 0.5, 0.5],
                    against_wall: false,
                    per_10m2: 0.5,
                    clearance: 1.6,
                },
            ],
        ),
        (
            RoomType::Office,
            &[FurnitureDef {
                module: "Counter",
                half: [1.1, 0.55, 0.35],
                against_wall: true,
                per_10m2: 0.6,
                clearance: 1.8,
            }],
        ),
        (
            RoomType::Service,
            &[FurnitureDef {
                module: "Crate",
                half: [0.5, 0.5, 0.5],
                against_wall: true,
                per_10m2: 1.0,
                clearance: 1.2,
            }],
        ),
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    /// **The shipped-content gate.** Every palette must parse with the very
    /// parser an author's own rule text goes through — the palettes are not a
    /// privileged dialect.
    #[test]
    fn every_palette_parses() {
        for a in archetypes() {
            let g = a
                .grammar()
                .unwrap_or_else(|e| panic!("{} does not parse: {e}", a.display));
            assert!(!g.modules().is_empty(), "{} declares no modules", a.display);
            assert!(!g.is_empty(), "{} declares no rules", a.display);
        }
    }

    /// Every module the assembler and the furniture tables look up **by name**
    /// must exist in the palette, or the building would quietly lose a piece.
    #[test]
    fn every_named_module_and_axiom_resolves() {
        for a in archetypes() {
            let g = a.grammar().expect("parses");
            for (what, name) in [
                ("lintel", a.lintel),
                ("parapet", a.parapet),
                ("slab", a.slab),
                ("step", a.step),
                ("roof", a.roof),
            ] {
                assert!(
                    g.module_index(name).is_some(),
                    "{}: {what} module `{name}` is not declared",
                    a.display
                );
            }
            for (what, axiom) in [
                ("exterior", a.exterior_axiom),
                ("interior", a.interior_axiom),
            ] {
                assert!(
                    g.rule(axiom).is_some(),
                    "{}: {what} axiom `{axiom}` names no rule",
                    a.display
                );
            }
            for (kind, set) in a.furniture {
                for f in *set {
                    assert!(
                        g.module_index(f.module).is_some(),
                        "{}: {} furniture `{}` is not declared",
                        a.display,
                        kind.name(),
                        f.module
                    );
                }
            }
        }
    }

    /// The plan parameters have to be *consistent*, not merely present: a door
    /// must fit through the thinnest wall run a room can offer, and the stair
    /// core must fit inside the smallest footprint the archetype accepts. These
    /// are the preconditions the partition's connectivity proof rests on.
    #[test]
    fn palette_parameters_are_self_consistent() {
        for a in archetypes() {
            assert!(
                a.min_room >= 2.0 * a.door_width,
                "{}: min_room {} cannot host a {} m door on every shared wall",
                a.display,
                a.min_room,
                a.door_width
            );
            // The connectivity proof's *actual* precondition: the overlap it
            // guarantees (`min_room`) must clear a full-width door plus both
            // jambs. Implied by the 2x rule for every shipped `door_width`, but
            // asserted directly so a palette with a very narrow door cannot
            // satisfy the implication and break the thing it implies.
            let need = a.door_width + 2.0 * crate::building::partition::DOOR_JAMB;
            assert!(
                a.min_room >= need,
                "{}: min_room {} is under the {need} m a full-width door plus jambs needs",
                a.display,
                a.min_room
            );
            // …and the jamb has to out-reach the widest module a WALL RUN can
            // put at a corner, or a corner post ends up inside a doorway on the
            // perpendicular wall. Only rule-placed modules count: slabs, treads
            // and furniture are placed directly by the assembler at plan-derived
            // dimensions and never straddle a wall line.
            let g = a.grammar().expect("parses");
            for name in g.placed_modules() {
                let m = &g.modules()[g.module_index(name).expect("declared") as usize];
                if let Some(c) = m.collider {
                    assert!(
                        c.x <= crate::building::partition::DOOR_JAMB,
                        "{}: wall module `{}` is {} m across — wider than the {} m jamb, \
                         so it can reach into a doorway on a perpendicular wall",
                        a.display,
                        m.name,
                        c.x,
                        crate::building::partition::DOOR_JAMB
                    );
                }
            }
            assert!(
                a.floors.0 >= 1 && a.floors.0 <= a.floors.1,
                "{}: floor range {:?} is empty",
                a.display,
                a.floors
            );
            assert!(
                a.door_height > 1.9 && a.door_height < a.floor_height,
                "{}: a {} m door does not fit under a {} m storey",
                a.display,
                a.door_height,
                a.floor_height
            );
            // A **positive** sill, not merely one below the head: a window at
            // sill 0 is a door-height void, so `place_windows` would carve a
            // full-height hole with no parapet under it and the façade would be
            // a colonnade. The assembler's `if op.sill > 0.0` guard makes that
            // silent rather than wrong, which is exactly why it is pinned here.
            assert!(
                a.window_sill > 0.0,
                "{}: a zero sill makes a window a doorway",
                a.display
            );
            assert!(
                a.window_head <= a.floor_height && a.window_sill < a.window_head,
                "{}: window band {}..{} does not fit a {} m storey",
                a.display,
                a.window_sill,
                a.window_head,
                a.floor_height
            );
            assert!(
                a.max_room_area > a.min_room * a.min_room,
                "{}: max_room_area {} is below one minimum room",
                a.display,
                a.max_room_area
            );
            assert!(
                a.wall_thickness > 0.0 && a.wall_thickness < a.min_room * 0.5,
                "{}: wall thickness {} is not usable",
                a.display,
                a.wall_thickness
            );
            assert!(
                a.stair_size.0 > 0.0 && a.stair_size.1 > 0.0,
                "{}: stair core has no size",
                a.display
            );
            assert!(
                a.corridor_width >= a.door_width,
                "{}: a {} m corridor cannot take a {} m door",
                a.display,
                a.corridor_width,
                a.door_width
            );
        }
    }

    #[test]
    fn ids_round_trip_through_their_names() {
        for id in ArchetypeId::ALL {
            assert_eq!(ArchetypeId::parse(id.name()), Some(id));
            assert_eq!(archetype(id).id, id);
        }
        assert_eq!(ArchetypeId::parse("Castle"), None);
        assert_eq!(ArchetypeId::parse("office"), None, "matching is exact");
        // The lookup table covers every variant, in order.
        let ids: Vec<ArchetypeId> = archetypes().iter().map(|a| a.id).collect();
        assert_eq!(ids, ArchetypeId::ALL.to_vec());
    }

    #[test]
    fn room_tables_are_non_empty_and_positively_weighted() {
        for a in archetypes() {
            for floor in [0u32, 1] {
                let table = a.room_table(floor);
                assert!(
                    !table.is_empty(),
                    "{} floor {floor} has no rooms",
                    a.display
                );
                assert!(
                    table.iter().all(|w| w.weight > 0.0),
                    "{} floor {floor} has a non-positive weight",
                    a.display
                );
                // Neither table may offer Stair or Corridor — those are placed
                // structurally, and drawing one would create a second.
                assert!(
                    table
                        .iter()
                        .all(|w| w.kind != RoomType::Stair && w.kind != RoomType::Corridor),
                    "{} floor {floor} draws a structural room type",
                    a.display
                );
            }
        }
    }
}
