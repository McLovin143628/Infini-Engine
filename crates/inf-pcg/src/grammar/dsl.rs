//! The grammar text DSL: lexer, parser, rule table and module palette.
//!
//! # The grammar of the grammar
//!
//! One authored string holds two kinds of statement — **module declarations**
//! (the palette a terminal places) and **rules** (how symbols rewrite). A
//! statement ends at a `;` **or at a newline**, whichever comes first, so the
//! usual one-per-line layout needs no punctuation and a dense one-liner is still
//! legal. `#` and `//` start a comment that runs to the end of the line.
//!
//! ```text
//! grammar      ::= { statement }
//! statement    ::= module_decl | rule_decl
//!
//! module_decl  ::= 'module' IDENT '=' { module_attr } TERM
//! module_attr  ::= 'mesh'   GUID                       -- the .inf_mesh asset placed
//!                | 'offset' num ',' num ',' num        -- metres, in the slot frame (X right, Y up, Z along)
//!                | 'rot'    num ',' num ',' num        -- euler DEGREES (X, Y, Z), applied before the slot frame
//!                | 'scale'  num                        -- uniform
//!                | 'size'   num                        -- default metres consumed along the span
//!                | 'collider' num ',' num ',' num      -- half-extents (metres) in the slot frame; P19.5
//!
//! rule_decl    ::= IDENT '->' alternatives TERM
//! alternatives ::= alternative { '|' alternative }
//! alternative  ::= { element } [ '@' num ]             -- optional relative weight (default 1)
//! element      ::= primary [ postfix ]
//! primary      ::= IDENT [ size ] | '(' alternatives ')'
//! postfix      ::= '*' | '+' | '?' | '{' INT '}'
//! size         ::= '[' num [ '..' num ] [ 'm' ] ']'
//!
//! TERM         ::= ';' | end-of-line | end-of-input
//! ```
//!
//! A worked example — a fence with posts at both ends, panels filling the
//! middle, and one in three panels carrying a gate:
//!
//! ```text
//! module Post  = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ff size 0.2
//! module Panel = mesh 6f9619ff-8b86-d011-b42d-00c04fc964f0 size 2 offset 0,0,1
//! module Gate  = mesh 6f9619ff-8b86-d011-b42d-00c04fc964f1 size 2 offset 0,0,1
//!
//! Fence -> Post Bay* Post
//! Bay   -> Panel | Gate@0.5
//! ```
//!
//! # Terminals, non-terminals and gaps
//!
//! A symbol with a rule is a **non-terminal** and rewrites. A symbol without one
//! is a **terminal**: it consumes span length and places its module. A terminal
//! with **no module declaration places nothing but still consumes its size** —
//! that is the `Gap[0.5..1.5m]` idiom, and it is deliberately not an error, so a
//! spacer needs no palette entry. Because a typo would otherwise vanish silently,
//! [`Grammar::gaps`] lists every such terminal and the graph lowerer turns the
//! list into a node-anchored warning.
//!
//! # Sizes are mandatory on terminals
//!
//! A terminal's length along the span comes from its inline `[…]` if it has one,
//! otherwise from its module's `size`. A terminal with neither is a **parse
//! error** — a layout cannot be computed without it, and guessing a default
//! would place a wall of 1 m panels nobody asked for. A non-terminal may not
//! carry a size at all (its length is whatever it rewrites to), which is also an
//! error rather than a silent no-op.
//!
//! # Colliders are opt-in (P19.5)
//!
//! `collider hx,hy,hz` gives a module a **solid box**, as half-extents in the
//! slot frame, centred on the module's own `offset`. Without it a module is
//! geometry and nothing else — the P19.4 behaviour, kept as the default because
//! a fence panel that silently started blocking the road it follows would be a
//! regression, not a feature. With it, expansion emits a
//! [`PcgCollider`](crate::scatter::PcgCollider) beside every instance, which is
//! what makes P19.5's buildings *enterable* rather than merely drawn: a doorway
//! is a stretch of wall where no module — and therefore no box — was placed.
//!
//! # v1 rejects recursion, by construction
//!
//! The rule reference graph must be **acyclic**; a cycle (`A -> B`, `B -> A`) is
//! a parse error naming the loop. This is not a limitation the implementation
//! stumbled into — it is what makes expansion terminate without a depth fuse, so
//! a grammar that parses always produces a finite derivation. Self-similar
//! structure is expressed with repetition (`*`, `+`, `{n}`), which is bounded by
//! the span rather than by the rule.
//!
//! # Determinism
//!
//! Parsing is a pure function of the text. Nothing here draws a random number;
//! every choice and repetition count is resolved later, in
//! [`expand`](super::expand), from the counter hash.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use uuid::Uuid;

/// A parse or validation failure, anchored at a 1-based line and column of the
/// authored rule text.
///
/// The lowerer prefixes the node's own message with this, so a diagnostic reads
/// `Grammar Rules: line 4:12: …` and the canvas can select the offending node —
/// the same anchoring contract the WGSL emitter and the density walk already
/// have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarError {
    /// 1-based line.
    pub line: u32,
    /// 1-based column (in `char`s, not bytes).
    pub col: u32,
    pub message: String,
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for GrammarError {}

impl GrammarError {
    fn new(line: u32, col: u32, message: impl Into<String>) -> Self {
        Self {
            line,
            col,
            message: message.into(),
        }
    }
}

/// How much of the span one terminal consumes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeSpec {
    /// Exactly this many metres.
    Fixed(f64),
    /// Anywhere in `[min, max]` metres, resolved per slot from the counter hash
    /// and then apportioned so the span fills exactly.
    Flex { min: f64, max: f64 },
}

impl SizeSpec {
    /// The smallest length this spec can occupy.
    pub fn min(&self) -> f64 {
        match self {
            SizeSpec::Fixed(v) => *v,
            SizeSpec::Flex { min, .. } => *min,
        }
    }

    /// The largest length this spec can occupy.
    pub fn max(&self) -> f64 {
        match self {
            SizeSpec::Fixed(v) => *v,
            SizeSpec::Flex { max, .. } => *max,
        }
    }

    /// The length used when counting how many copies of a repeat body fit: the
    /// midpoint of a flexible range (what an author expects "on average"), the
    /// value itself for a fixed size.
    pub fn nominal(&self) -> f64 {
        match self {
            SizeSpec::Fixed(v) => *v,
            SizeSpec::Flex { min, max } => (min + max) * 0.5,
        }
    }
}

/// A postfix repetition operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repeat {
    /// No postfix: exactly one.
    One,
    /// `*` — as many as fit after reserving what follows (may be zero).
    Fill,
    /// `+` — like `*`, but never fewer than one (it may overflow the span, and
    /// the layout then truncates from the end).
    FillAtLeastOne,
    /// `?` — zero or one, chosen by an even hashed draw.
    Optional,
    /// `{n}` — exactly `n`.
    Exactly(u32),
}

/// One element of an alternative: a primary with its repetition.
#[derive(Debug, Clone, PartialEq)]
pub struct Element {
    pub primary: Primary,
    pub repeat: Repeat,
}

/// A symbol reference or a parenthesised group.
#[derive(Debug, Clone, PartialEq)]
pub enum Primary {
    /// A symbol (terminal or non-terminal) with an optional inline size.
    Symbol {
        name: String,
        size: Option<SizeSpec>,
    },
    /// `( … | … )` — an inline alternation.
    Group(Vec<Alternative>),
}

/// One right-hand side of a rule, with its relative selection weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Alternative {
    /// Relative weight for the hashed pick; defaults to `1`. Non-positive
    /// weights are rejected at parse time.
    pub weight: f64,
    /// May be empty — an epsilon production, i.e. "sometimes nothing".
    pub elements: Vec<Element>,
}

/// One rule: a symbol and the alternatives it rewrites to.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub symbol: String,
    pub alternatives: Vec<Alternative>,
}

/// One palette entry: what a terminal places, and how it sits in its slot.
///
/// `offset` and `rot` are expressed in the **slot frame** — `+Z` runs along the
/// span, `+Y` is world up, `+X` is `Y × Z` (to the right of travel). A module is
/// anchored at the **start** of its slot, so a 2 m panel modelled around its own
/// centre is centred with `offset 0,0,1`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDef {
    pub name: String,
    /// The `.inf_mesh` asset placed, or `None` for a bare marker (still a real
    /// instance — it just names no asset, exactly like `PcgKind::mesh == None`).
    pub mesh: Option<Uuid>,
    /// Metres, in the slot frame.
    pub offset: DVec3,
    /// Euler **degrees** (X, Y, Z), composed `YXZ` exactly like
    /// `inf_ecs::components::Transform`.
    pub rotation_deg: DVec3,
    /// Uniform scale (`ScatteredInstance::scale` is one f64 — see the module
    /// docs on `super::expand` for why flexible slots change *spacing*, not mesh
    /// size).
    pub scale: f64,
    /// Default metres consumed along the span when a terminal names this module
    /// without an inline size.
    pub size: Option<f64>,
    /// **P19.5.** Optional collision box, as half-extents in metres in the
    /// **slot frame** (`+X` right of travel, `+Y` up, `+Z` along the span),
    /// centred on the module's own `offset`.
    ///
    /// `None` means the module places geometry and nothing solid — the P19.4
    /// behaviour, and still the default, because a fence panel nobody has asked
    /// to be solid should not silently start blocking a road.
    ///
    /// The box is oriented by the slot's **yaw only**, not by the module's
    /// authored `rot`: half-extents are stated in the slot frame, so rotating
    /// them by an authored euler would mean the numbers no longer describe what
    /// the author typed. A module whose *mesh* is turned inside its slot keeps a
    /// slot-aligned collider, which is the predictable reading and the one a
    /// wall wants.
    pub collider: Option<DVec3>,
    /// **How brightly this module emits at night** (island wave I8b), as a
    /// multiplier on its own colour. `0.0` for everything that does not glow.
    ///
    /// **Derived, not parsed.** There is no `glow` keyword: the DSL describes
    /// where a module goes, and *what a module is* is the palette's business.
    /// [`Grammar::stamp_module_meshes`] sets it from the module's shape family,
    /// so a window is a window in every archetype by one rule rather than by
    /// seven authored numbers that can disagree.
    pub glow: f32,
    /// **What a module of this name is made of** (wave VEN1a) — emission,
    /// metal, roughness and tint.
    ///
    /// **Derived, not parsed**, on exactly the terms [`glow`](Self::glow) is
    /// and for exactly the reason: there is no `metallic` keyword and there
    /// will not be one. [`Grammar::stamp_module_meshes`] sets it from the
    /// module's shape family
    /// ([`ModuleShape::surface`](crate::building::modules::ModuleShape::surface)),
    /// so a chrome pole is chrome in every archetype by one rule.
    ///
    /// [`PcgSurface::DEFAULT`](crate::scatter::PcgSurface::DEFAULT) for every
    /// module of every palette that predates
    /// the venue families, which is what keeps every building the engine has
    /// ever drawn byte-identical.
    pub surface: crate::scatter::PcgSurface,
}

impl ModuleDef {
    fn new(name: String) -> Self {
        Self {
            name,
            mesh: None,
            offset: DVec3::ZERO,
            rotation_deg: DVec3::ZERO,
            scale: 1.0,
            size: None,
            collider: None,
            glow: 0.0,
            surface: crate::scatter::PcgSurface::DEFAULT,
        }
    }
}

/// A parsed, validated grammar: the rule table and the module palette.
///
/// Declaration order is preserved and load-bearing twice over: the **first rule**
/// is the default axiom, and a module's **index is its palette slot**, which
/// becomes the `kind_index` of every instance it places.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Grammar {
    rules: Vec<Rule>,
    modules: Vec<ModuleDef>,
    /// 1-based `(line, column)` of each rule's own first token, parallel to
    /// `rules` — the anchor every post-parse diagnostic about that rule uses,
    /// so an indented statement reports where it actually starts. For
    /// diagnostics only, and deliberately **excluded from round-trip
    /// comparisons**: a printer cannot preserve where a statement happened to
    /// sit, and pretending otherwise would make the round-trip property
    /// vacuous.
    rule_sites: Vec<(u32, u32)>,
    rule_index: BTreeMap<String, usize>,
    module_index: BTreeMap<String, usize>,
}

impl Grammar {
    /// Parse and validate `text`.
    pub fn parse(text: &str) -> Result<Self, GrammarError> {
        let tokens = lex(text)?;
        Parser {
            toks: &tokens,
            pos: 0,
            depth: 0,
        }
        .parse_grammar()
    }

    /// The rules, in declaration order.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// The module palette, in declaration order (index == `kind_index`).
    pub fn modules(&self) -> &[ModuleDef] {
        &self.modules
    }

    /// **Give every unassigned module its shape family's mesh and glow**
    /// (island wave I8b).
    ///
    /// A module that already names a `mesh` — an author who typed
    /// `module Panel = mesh <guid>` — is left alone: an authored asset always
    /// wins over a derived one, which is what makes this safe to run over any
    /// grammar rather than only over the seven palettes.
    ///
    /// Idempotent, and a pure function of the module NAMES, so two grammars that
    /// declare the same vocabulary stamp the same ids. Called by
    /// [`BuildingArchetype::grammar`](crate::building::palettes::BuildingArchetype::grammar)
    /// — the one place a palette becomes a grammar — so `place_module` and the
    /// building assembler read one field instead of each deriving an id.
    pub fn stamp_module_meshes(&mut self) {
        for m in &mut self.modules {
            let Some(shape) = crate::building::modules::shape_of(&m.name) else {
                continue;
            };
            if m.mesh.is_none() {
                m.mesh = Some(crate::building::modules::module_mesh_guid(shape));
            }
            if shape.is_glazing() {
                m.glow = crate::building::modules::GLAZING_GLOW;
            }
            // **Wave VEN1a**: the same stamp, one field wider. An authored
            // `mesh` still wins above; a surface has no authored spelling at
            // all, so this is unconditional -- a module of a family IS made of
            // what that family is made of, and there is nothing for an author
            // to have said differently.
            m.surface = shape.surface();
        }
    }

    /// **Stamp an archetype's surface set over the structural families** (wave
    /// ASSET0, clause 4).
    ///
    /// Runs after [`stamp_module_meshes`](Self::stamp_module_meshes) and only
    /// over families whose [`role`](crate::building::modules::ModuleShape::role)
    /// is not `Stated` — so the chrome pole, the glowing screen and the glazed
    /// leaf keep the material their FAMILY states, exactly as VEN1a ruled, and
    /// what moves is the wall, the floor and the furniture, which are the three
    /// things an archetype is entitled to an opinion about.
    ///
    /// # Why a tint and not a `.inf_mat`
    ///
    /// Because a scattered instance cannot wear one. Measured in wave ASSET0:
    /// `inf_render::ScatterBatch` carries `metallic`, `roughness`, `emissive`
    /// and a per-instance tint and **no virtual-texture set at all**, and
    /// `scatter_mesh.wgsl` names virtual pages only for shadows. Every building
    /// module in this engine is a scattered instance, so a `.inf_mat` on a wall
    /// would be a field with no path to a pixel. The tints below are what the
    /// channel that EXISTS can carry, and the gap is the wave's headline carried
    /// item.
    pub fn stamp_module_surfaces(&mut self, set: &crate::building::palettes::SurfaceSet) {
        use crate::building::modules::SurfaceRole;
        for m in &mut self.modules {
            let Some(shape) = crate::building::modules::shape_of(&m.name) else {
                continue;
            };
            m.surface = match shape.role() {
                SurfaceRole::Wall => set.wall,
                SurfaceRole::Floor => set.floor,
                SurfaceRole::Furniture => set.furniture,
                SurfaceRole::Stated => continue,
            };
        }
    }

    /// The rule for `symbol`, if it is a non-terminal.
    pub fn rule(&self, symbol: &str) -> Option<&Rule> {
        self.rule_index.get(symbol).map(|&i| &self.rules[i])
    }

    /// `symbol`'s index in [`rules`](Self::rules), if it is a non-terminal —
    /// the key into any per-rule table a consumer precomputes.
    pub fn rule_position(&self, symbol: &str) -> Option<usize> {
        self.rule_index.get(symbol).copied()
    }

    /// The palette index of `name`, if the grammar declares that module.
    pub fn module_index(&self, name: &str) -> Option<u32> {
        self.module_index.get(name).map(|&i| i as u32)
    }

    /// The default axiom: the first rule declared, or `None` for a grammar that
    /// declares only modules.
    pub fn default_axiom(&self) -> Option<&str> {
        self.rules.first().map(|r| r.symbol.as_str())
    }

    /// `true` when the grammar has no rules at all.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// The 1-based source line a rule was declared on (diagnostics).
    pub fn rule_line(&self, symbol: &str) -> Option<u32> {
        self.rule_site(symbol).map(|(line, _)| line)
    }

    /// The 1-based `(line, column)` a rule was declared at (diagnostics).
    pub fn rule_site(&self, symbol: &str) -> Option<(u32, u32)> {
        self.rule_index.get(symbol).map(|&i| self.rule_sites[i])
    }

    /// Terminals referenced by some rule that name **no module** — the gap
    /// symbols. Sorted and deduplicated, so a diagnostic built from this is
    /// deterministic.
    ///
    /// These are legal (`Gap[0.5..1.5m]` is the whole point) and place nothing;
    /// the list exists so a typo is *visible* rather than silently becoming
    /// empty space.
    pub fn gaps(&self) -> Vec<&str> {
        let mut out: BTreeSet<&str> = BTreeSet::new();
        for rule in &self.rules {
            visit_symbols(&rule.alternatives, &mut |name, _| {
                if !self.rule_index.contains_key(name) && !self.module_index.contains_key(name) {
                    out.insert(name);
                }
            });
        }
        out.into_iter().collect()
    }

    /// Every module a **rule actually places**, sorted and deduplicated — the
    /// exact complement of [`gaps`](Self::gaps), which lists the referenced
    /// symbols that resolve to *no* module.
    ///
    /// A palette may declare modules no rule references: P19.5's archetypes do
    /// exactly that for slabs, stair treads and furniture, which the building
    /// assembler places directly at plan-derived dimensions. Anything that needs
    /// to reason about what a *span expansion* can put on a wall — a jamb-width
    /// check, a future palette linter — wants this list and not
    /// [`modules`](Self::modules).
    pub fn placed_modules(&self) -> Vec<&str> {
        let mut out: BTreeSet<&str> = BTreeSet::new();
        for rule in &self.rules {
            visit_symbols(&rule.alternatives, &mut |name, _| {
                if let Some((n, _)) = self.module_index.get_key_value(name) {
                    out.insert(n.as_str());
                }
            });
        }
        out.into_iter().collect()
    }

    /// Every mesh GUID the palette references, sorted and deduplicated — the
    /// cook's `.inf_pcg` → module edge.
    pub fn mesh_refs(&self) -> Vec<Uuid> {
        let set: BTreeSet<Uuid> = self.modules.iter().filter_map(|m| m.mesh).collect();
        set.into_iter().collect()
    }

    /// The size a terminal reference resolves to: its inline size, else its
    /// module's declared `size`. Validated non-`None` at parse time for every
    /// terminal reachable from a rule, so expansion can rely on it.
    pub fn terminal_size(&self, name: &str, inline: Option<SizeSpec>) -> Option<SizeSpec> {
        inline.or_else(|| {
            self.module_index
                .get(name)
                .and_then(|&i| self.modules[i].size)
                .map(SizeSpec::Fixed)
        })
    }

    /// Render back to canonical DSL text. `parse(g.to_text())` reproduces `g`'s
    /// rules and modules (the round-trip property); line numbers are *not*
    /// preserved and are excluded from that comparison by design.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        for m in &self.modules {
            s.push_str("module ");
            s.push_str(&m.name);
            s.push_str(" =");
            if let Some(mesh) = m.mesh {
                s.push_str(&format!(" mesh {mesh}"));
            }
            if m.offset != DVec3::ZERO {
                s.push_str(&format!(
                    " offset {},{},{}",
                    m.offset.x, m.offset.y, m.offset.z
                ));
            }
            if m.rotation_deg != DVec3::ZERO {
                s.push_str(&format!(
                    " rot {},{},{}",
                    m.rotation_deg.x, m.rotation_deg.y, m.rotation_deg.z
                ));
            }
            if m.scale != 1.0 {
                s.push_str(&format!(" scale {}", m.scale));
            }
            if let Some(size) = m.size {
                s.push_str(&format!(" size {size}"));
            }
            if let Some(c) = m.collider {
                s.push_str(&format!(" collider {},{},{}", c.x, c.y, c.z));
            }
            s.push('\n');
        }
        for rule in &self.rules {
            s.push_str(&rule.symbol);
            s.push_str(" ->");
            print_alternatives(&rule.alternatives, &mut s);
            s.push('\n');
        }
        s
    }
}

fn print_alternatives(alts: &[Alternative], out: &mut String) {
    for (i, alt) in alts.iter().enumerate() {
        if i > 0 {
            out.push_str(" |");
        }
        for el in &alt.elements {
            out.push(' ');
            print_primary(&el.primary, out);
            match el.repeat {
                Repeat::One => {}
                Repeat::Fill => out.push('*'),
                Repeat::FillAtLeastOne => out.push('+'),
                Repeat::Optional => out.push('?'),
                Repeat::Exactly(n) => out.push_str(&format!("{{{n}}}")),
            }
        }
        if alt.weight != 1.0 {
            out.push_str(&format!(" @{}", alt.weight));
        }
    }
}

fn print_primary(p: &Primary, out: &mut String) {
    match p {
        Primary::Symbol { name, size } => {
            out.push_str(name);
            match size {
                Some(SizeSpec::Fixed(v)) => out.push_str(&format!("[{v}]")),
                Some(SizeSpec::Flex { min, max }) => out.push_str(&format!("[{min}..{max}]")),
                None => {}
            }
        }
        Primary::Group(alts) => {
            out.push('(');
            // `print_alternatives` leads each element with a space; trim the
            // first so `( A | B )` prints as `(A | B)`.
            let mut inner = String::new();
            print_alternatives(alts, &mut inner);
            out.push_str(inner.trim_start());
            out.push(')');
        }
    }
}

/// Walk every symbol reference in `alts` (recursing into groups), calling `f`
/// with the name and its inline size. The name borrows from `alts`, so a caller
/// can collect the references themselves rather than cloning every symbol.
fn visit_symbols<'a>(alts: &'a [Alternative], f: &mut impl FnMut(&'a str, Option<SizeSpec>)) {
    for alt in alts {
        for el in &alt.elements {
            match &el.primary {
                Primary::Symbol { name, size } => f(name, *size),
                Primary::Group(inner) => visit_symbols(inner, f),
            }
        }
    }
}

// ── lexer ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Number(f64),
    Guid(Uuid),
    Arrow,
    Pipe,
    At,
    Star,
    Plus,
    Question,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    LParen,
    RParen,
    Comma,
    Eq,
    DotDot,
    /// `;` or a newline — a statement terminator.
    Term,
    Eof,
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("`{s}`"),
            Tok::Number(n) => format!("number `{n}`"),
            Tok::Guid(g) => format!("GUID `{g}`"),
            Tok::Arrow => "`->`".into(),
            Tok::Pipe => "`|`".into(),
            Tok::At => "`@`".into(),
            Tok::Star => "`*`".into(),
            Tok::Plus => "`+`".into(),
            Tok::Question => "`?`".into(),
            Tok::LBracket => "`[`".into(),
            Tok::RBracket => "`]`".into(),
            Tok::LBrace => "`{`".into(),
            Tok::RBrace => "`}`".into(),
            Tok::LParen => "`(`".into(),
            Tok::RParen => "`)`".into(),
            Tok::Comma => "`,`".into(),
            Tok::Eq => "`=`".into(),
            Tok::DotDot => "`..`".into(),
            Tok::Term => "end of statement".into(),
            Tok::Eof => "end of input".into(),
        }
    }
}

#[derive(Debug, Clone)]
struct Spanned {
    tok: Tok,
    line: u32,
    col: u32,
}

/// Is this the start of a `8-4-4-4-12` hex GUID? Checked before ident/number
/// scanning, because a GUID contains both digits and `-` and would otherwise lex
/// as three different tokens.
fn uuid_at(chars: &[char], i: usize) -> Option<Uuid> {
    const LEN: usize = 36;
    if i + LEN > chars.len() {
        return None;
    }
    let s: String = chars[i..i + LEN].iter().collect();
    // Reject a longer word that merely *starts* with a GUID.
    if let Some(&next) = chars.get(i + LEN) {
        if next.is_ascii_alphanumeric() || next == '-' || next == '_' {
            return None;
        }
    }
    Uuid::parse_str(&s).ok().filter(|_| s.contains('-'))
}

fn lex(text: &str) -> Result<Vec<Spanned>, GrammarError> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Spanned> = Vec::new();
    let (mut line, mut col) = (1u32, 1u32);
    let mut i = 0usize;

    // Collapse runs of terminators so blank lines do not produce empty
    // statements the parser would have to skip twice.
    let push = |out: &mut Vec<Spanned>, tok: Tok, line: u32, col: u32| {
        if tok == Tok::Term && matches!(out.last().map(|s| &s.tok), None | Some(Tok::Term)) {
            return;
        }
        out.push(Spanned { tok, line, col });
    };

    while i < chars.len() {
        let c = chars[i];
        let (tl, tc) = (line, col);
        // Newline → terminator.
        if c == '\n' {
            push(&mut out, Tok::Term, tl, tc);
            i += 1;
            line += 1;
            col = 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            col += 1;
            continue;
        }
        // Comments run to the end of the line (the newline itself still
        // terminates the statement).
        if c == '#' || (c == '/' && chars.get(i + 1) == Some(&'/')) {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
                col += 1;
            }
            continue;
        }
        if let Some(guid) = uuid_at(&chars, i) {
            push(&mut out, Tok::Guid(guid), tl, tc);
            i += 36;
            col += 36;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
                col += 1;
            }
            push(
                &mut out,
                Tok::Ident(chars[start..i].iter().collect()),
                tl,
                tc,
            );
            continue;
        }
        if c.is_ascii_digit() || (c == '-' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let start = i;
            if c == '-' {
                i += 1;
                col += 1;
            }
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
                col += 1;
            }
            // A single `.` is a decimal point; `..` is the range operator, so a
            // lone dot followed by a dot ends the number.
            if chars.get(i) == Some(&'.') && chars.get(i + 1) != Some(&'.') {
                i += 1;
                col += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
            }
            if matches!(chars.get(i), Some('e') | Some('E')) {
                let save = (i, col);
                i += 1;
                col += 1;
                if matches!(chars.get(i), Some('+') | Some('-')) {
                    i += 1;
                    col += 1;
                }
                if chars.get(i).is_some_and(|d| d.is_ascii_digit()) {
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                        col += 1;
                    }
                } else {
                    // Not an exponent after all (`2e` followed by a symbol).
                    (i, col) = save;
                }
            }
            let s: String = chars[start..i].iter().collect();
            let n = s
                .parse::<f64>()
                .map_err(|_| GrammarError::new(tl, tc, format!("`{s}` is not a number")))?;
            push(&mut out, Tok::Number(n), tl, tc);
            continue;
        }
        let two: Option<Tok> = match (c, chars.get(i + 1)) {
            ('-', Some('>')) => Some(Tok::Arrow),
            ('.', Some('.')) => Some(Tok::DotDot),
            _ => None,
        };
        if let Some(tok) = two {
            push(&mut out, tok, tl, tc);
            i += 2;
            col += 2;
            continue;
        }
        let one = match c {
            ';' => Tok::Term,
            '|' => Tok::Pipe,
            '@' => Tok::At,
            '*' => Tok::Star,
            '+' => Tok::Plus,
            '?' => Tok::Question,
            '[' => Tok::LBracket,
            ']' => Tok::RBracket,
            '{' => Tok::LBrace,
            '}' => Tok::RBrace,
            '(' => Tok::LParen,
            ')' => Tok::RParen,
            ',' => Tok::Comma,
            '=' => Tok::Eq,
            _ => {
                return Err(GrammarError::new(
                    tl,
                    tc,
                    format!("unexpected character `{c}`"),
                ))
            }
        };
        push(&mut out, one, tl, tc);
        i += 1;
        col += 1;
    }
    out.push(Spanned {
        tok: Tok::Eof,
        line,
        col,
    });
    Ok(out)
}

// ── parser ──────────────────────────────────────────────────────────────────

struct Parser<'a> {
    toks: &'a [Spanned],
    pos: usize,
    /// Group-nesting depth, capped at [`MAX_NESTING`].
    ///
    /// **This is a safety property, not a style guard.** `parse_alternatives`
    /// -> `parse_alternative` -> `parse_element` -> (group) ->
    /// `parse_alternatives` is a native-stack recursion driven directly by
    /// authored text, and a stack overflow is an *uncatchable process abort* --
    /// it would take down the editor, the cook and the shipped player alike,
    /// since every one of them re-lowers a stored `graph_json`. The
    /// acyclic-rule check that makes v1 "terminate by construction" runs one
    /// layer DOWNSTREAM of this and cannot help: the text never reaches it.
    depth: u32,
}

impl Parser<'_> {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos.min(self.toks.len() - 1)].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn here(&self) -> (u32, u32) {
        let s = &self.toks[self.pos.min(self.toks.len() - 1)];
        (s.line, s.col)
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks[self.pos.min(self.toks.len() - 1)].tok.clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn err(&self, message: impl Into<String>) -> GrammarError {
        let (l, c) = self.here();
        GrammarError::new(l, c, message)
    }

    fn expect(&mut self, want: &Tok, what: &str) -> Result<(), GrammarError> {
        if self.peek() == want {
            self.bump();
            Ok(())
        } else {
            Err(self.err(format!("expected {what}, found {}", self.peek().describe())))
        }
    }

    fn ident(&mut self, what: &str) -> Result<String, GrammarError> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.bump();
                Ok(s)
            }
            other => Err(self.err(format!("expected {what}, found {}", other.describe()))),
        }
    }

    fn number(&mut self, what: &str) -> Result<f64, GrammarError> {
        match *self.peek() {
            Tok::Number(n) => {
                self.bump();
                Ok(n)
            }
            ref other => Err(self.err(format!("expected {what}, found {}", other.describe()))),
        }
    }

    fn at_end_of_statement(&self) -> bool {
        matches!(self.peek(), Tok::Term | Tok::Eof)
    }

    fn parse_grammar(&mut self) -> Result<Grammar, GrammarError> {
        let mut g = Grammar::default();
        loop {
            while matches!(self.peek(), Tok::Term) {
                self.bump();
            }
            if matches!(self.peek(), Tok::Eof) {
                break;
            }
            let site = self.here();
            let is_module = matches!(self.peek(), Tok::Ident(s) if s == "module")
                && matches!(self.peek_at(1), Tok::Ident(_));
            if is_module {
                let m = self.parse_module()?;
                if g.module_index.contains_key(&m.name) {
                    return Err(GrammarError::new(
                        site.0,
                        site.1,
                        format!("module `{}` is declared twice", m.name),
                    ));
                }
                g.module_index.insert(m.name.clone(), g.modules.len());
                g.modules.push(m);
            } else {
                let rule = self.parse_rule()?;
                if g.rule_index.contains_key(&rule.symbol) {
                    return Err(GrammarError::new(
                        site.0,
                        site.1,
                        format!("rule `{}` is declared twice", rule.symbol),
                    ));
                }
                g.rule_index.insert(rule.symbol.clone(), g.rules.len());
                g.rules.push(rule);
                g.rule_sites.push(site);
            }
        }
        validate(&g)?;
        Ok(g)
    }

    fn parse_module(&mut self) -> Result<ModuleDef, GrammarError> {
        self.bump(); // `module`
        let name = self.ident("a module name")?;
        self.expect(&Tok::Eq, "`=`")?;
        let mut m = ModuleDef::new(name);
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while !self.at_end_of_statement() {
            let (l, c) = self.here();
            let attr = self.ident("a module attribute (mesh/offset/rot/scale/size/collider)")?;
            if !seen.insert(attr.clone()) {
                return Err(GrammarError::new(
                    l,
                    c,
                    format!("duplicate module attribute `{attr}`"),
                ));
            }
            match attr.as_str() {
                "mesh" => match self.peek().clone() {
                    Tok::Guid(g) => {
                        self.bump();
                        m.mesh = Some(g);
                    }
                    other => {
                        return Err(self.err(format!(
                            "`mesh` expects a GUID (8-4-4-4-12 hex), found {}",
                            other.describe()
                        )))
                    }
                },
                "offset" => m.offset = self.parse_triple("offset")?,
                "rot" => m.rotation_deg = self.parse_triple("rot")?,
                // The bad-value anchors point at the VALUE, not at the keyword
                // that introduced it: `scale 0` is wrong at the `0`, and that is
                // where an editor should put the caret. (The position is taken
                // BEFORE consuming, because `number` advances past it.)
                "scale" => {
                    let at = self.here();
                    let v = self.number("a scale")?;
                    if !(v.is_finite() && v > 0.0) {
                        return Err(GrammarError::new(at.0, at.1, "`scale` must be positive"));
                    }
                    m.scale = v;
                }
                "size" => {
                    let at = self.here();
                    let v = self.number("a size in metres")?;
                    if !(v.is_finite() && v >= 0.0) {
                        return Err(GrammarError::new(
                            at.0,
                            at.1,
                            "`size` must be zero or positive metres",
                        ));
                    }
                    m.size = Some(v);
                }
                // P19.5. A **zero** half-extent on any axis is rejected rather
                // than accepted-and-ignored: `collider 0.1,1.5,0` reads as "a
                // collider" and would silently be a zero-volume shape no
                // character could ever touch, which is exactly the failure a
                // building must not be able to ship.
                "collider" => {
                    let at = self.here();
                    let v = self.parse_triple("collider")?;
                    if !(v.x > 0.0 && v.y > 0.0 && v.z > 0.0) {
                        return Err(GrammarError::new(
                            at.0,
                            at.1,
                            "`collider` half-extents must all be positive metres",
                        ));
                    }
                    m.collider = Some(v);
                }
                other => {
                    return Err(GrammarError::new(
                        l,
                        c,
                        format!(
                            "unknown module attribute `{other}` \
                             (expected mesh, offset, rot, scale, size or collider)"
                        ),
                    ))
                }
            }
        }
        Ok(m)
    }

    fn parse_triple(&mut self, what: &str) -> Result<DVec3, GrammarError> {
        let at = self.here();
        let x = self.number(&format!("{what} X"))?;
        self.expect(&Tok::Comma, "`,`")?;
        let y = self.number(&format!("{what} Y"))?;
        self.expect(&Tok::Comma, "`,`")?;
        let z = self.number(&format!("{what} Z"))?;
        let v = DVec3::new(x, y, z);
        if !v.is_finite() {
            return Err(GrammarError::new(
                at.0,
                at.1,
                format!("`{what}` must be finite"),
            ));
        }
        Ok(v)
    }

    fn parse_rule(&mut self) -> Result<Rule, GrammarError> {
        let symbol = self.ident("a rule name")?;
        self.expect(&Tok::Arrow, "`->`")?;
        let alternatives = self.parse_alternatives(false)?;
        if !self.at_end_of_statement() {
            return Err(self.err(format!(
                "expected end of rule, found {}",
                self.peek().describe()
            )));
        }
        Ok(Rule {
            symbol,
            alternatives,
        })
    }

    /// `alternative { '|' alternative }`. Inside a group the closing `)` also
    /// ends the list; at statement level a terminator does.
    ///
    /// **The one recursion point, and therefore the one place the depth is
    /// bounded.** Every path back into the parser's own cycle -- a group inside
    /// an element inside an alternative -- passes through here, so guarding
    /// this function guards the cycle. Exceeding [`MAX_NESTING`] is an ordinary
    /// anchored [`GrammarError`], never a native-stack abort.
    fn parse_alternatives(&mut self, in_group: bool) -> Result<Vec<Alternative>, GrammarError> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            self.depth -= 1;
            return Err(self.err(format!(
                "grammar nested deeper than {MAX_NESTING} levels of `( ... )` -- \
                 flatten it, or name the inner group as its own rule"
            )));
        }
        let out = self.parse_alternatives_inner(in_group);
        self.depth -= 1;
        out
    }

    /// The body of [`parse_alternatives`], split out so the depth counter is
    /// decremented on every exit including the `?` ones.
    fn parse_alternatives_inner(
        &mut self,
        in_group: bool,
    ) -> Result<Vec<Alternative>, GrammarError> {
        let mut alts = vec![self.parse_alternative(in_group)?];
        while matches!(self.peek(), Tok::Pipe) {
            self.bump();
            alts.push(self.parse_alternative(in_group)?);
        }
        Ok(alts)
    }

    fn parse_alternative(&mut self, in_group: bool) -> Result<Alternative, GrammarError> {
        let mut elements = Vec::new();
        while matches!(self.peek(), Tok::Ident(_) | Tok::LParen) {
            elements.push(self.parse_element(in_group)?);
        }
        let mut weight = 1.0;
        if matches!(self.peek(), Tok::At) {
            self.bump();
            let (l, c) = self.here();
            weight = self.number("an alternative weight")?;
            if !(weight.is_finite() && weight > 0.0) {
                return Err(GrammarError::new(
                    l,
                    c,
                    "an alternative weight must be positive",
                ));
            }
        }
        Ok(Alternative { weight, elements })
    }

    fn parse_element(&mut self, in_group: bool) -> Result<Element, GrammarError> {
        let primary = match self.peek().clone() {
            Tok::LParen => {
                self.bump();
                let alts = self.parse_alternatives(true)?;
                self.expect(&Tok::RParen, "`)`")?;
                Primary::Group(alts)
            }
            Tok::Ident(name) => {
                self.bump();
                let size = if matches!(self.peek(), Tok::LBracket) {
                    Some(self.parse_size()?)
                } else {
                    None
                };
                Primary::Symbol { name, size }
            }
            other => return Err(self.err(format!("expected a symbol, found {}", other.describe()))),
        };
        let _ = in_group;
        let repeat = match self.peek() {
            Tok::Star => {
                self.bump();
                Repeat::Fill
            }
            Tok::Plus => {
                self.bump();
                Repeat::FillAtLeastOne
            }
            Tok::Question => {
                self.bump();
                Repeat::Optional
            }
            Tok::LBrace => {
                self.bump();
                let (l, c) = self.here();
                let n = self.number("a repeat count")?;
                if !(n.is_finite() && n >= 0.0 && n <= MAX_EXACT_REPEAT as f64) {
                    return Err(GrammarError::new(
                        l,
                        c,
                        format!("a `{{n}}` repeat count must be between 0 and {MAX_EXACT_REPEAT}"),
                    ));
                }
                if n.fract() != 0.0 {
                    return Err(GrammarError::new(
                        l,
                        c,
                        "a `{n}` repeat count must be whole",
                    ));
                }
                self.expect(&Tok::RBrace, "`}`")?;
                Repeat::Exactly(n as u32)
            }
            _ => Repeat::One,
        };
        Ok(Element { primary, repeat })
    }

    fn parse_size(&mut self) -> Result<SizeSpec, GrammarError> {
        // A bad single value is wrong at the value; a bad RANGE is wrong as a
        // pair, so it anchors at the `[` that opens the whole specification.
        let (l, c) = self.here();
        self.expect(&Tok::LBracket, "`[`")?;
        let first = self.here();
        let a = self.number("a size in metres")?;
        let spec = if matches!(self.peek(), Tok::DotDot) {
            self.bump();
            let b = self.number("the upper bound of a size range")?;
            SizeSpec::Flex { min: a, max: b }
        } else {
            SizeSpec::Fixed(a)
        };
        // An optional bare `m` unit, purely decorative: 1 world unit is 1 metre
        // everywhere in this engine (architecture rule 6), so there is nothing
        // to convert.
        if matches!(self.peek(), Tok::Ident(u) if u == "m") {
            self.bump();
        }
        self.expect(&Tok::RBracket, "`]`")?;
        match spec {
            SizeSpec::Fixed(v) if !(v.is_finite() && v >= 0.0) => Err(GrammarError::new(
                first.0,
                first.1,
                "a size must be zero or positive metres",
            )),
            SizeSpec::Flex { min, max }
                if !(min.is_finite() && max.is_finite() && min >= 0.0 && max >= min) =>
            {
                Err(GrammarError::new(
                    l,
                    c,
                    "a size range must be `[min..max]` with `0 <= min <= max`",
                ))
            }
            ok => Ok(ok),
        }
    }
}

/// The upper bound on a literal `{n}` repetition. Not a resource limit (the
/// layout's own slot cap is that) — it is a typo guard, so `Panel{100000}`
/// fails at parse time instead of at expansion.
pub const MAX_EXACT_REPEAT: u32 = 4096;

/// The deepest `( ... )` nesting the parser will descend into.
///
/// Sixty-four is far beyond anything an author writes — three or four levels
/// is already unreadable — and it is chosen to leave the native stack
/// untouched rather than to constrain expression. See the guard on
/// [`Parser::parse_alternatives`] for why a *parser* limit is a safety property
/// here and not a style rule.
pub const MAX_NESTING: u32 = 64;

// ── validation ──────────────────────────────────────────────────────────────

fn validate(g: &Grammar) -> Result<(), GrammarError> {
    // Sizes: a terminal needs one, a non-terminal must not carry one.
    for (ri, rule) in g.rules.iter().enumerate() {
        let (line, col) = g.rule_sites[ri];
        let mut failure: Option<GrammarError> = None;
        visit_symbols(&rule.alternatives, &mut |name, size| {
            if failure.is_some() {
                return;
            }
            let is_rule = g.rule_index.contains_key(name);
            if is_rule {
                if size.is_some() {
                    failure = Some(GrammarError::new(
                        line,
                        col,
                        format!(
                            "`{name}` is a rule, so it cannot carry a size — \
                             a rule's length is whatever it rewrites to"
                        ),
                    ));
                }
            } else if g.terminal_size(name, size).is_none() {
                failure = Some(GrammarError::new(
                    line,
                    col,
                    format!(
                        "terminal `{name}` has no size — write `{name}[2m]` \
                         or declare `module {name} = … size 2`"
                    ),
                ));
            }
        });
        if let Some(e) = failure {
            return Err(e);
        }
    }

    // The rule reference graph must be acyclic (see the module docs).
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        White,
        Grey,
        Black,
    }
    let mut mark = vec![Mark::White; g.rules.len()];
    // Explicit stack DFS: a deep chain must not blow the native stack.
    for start in 0..g.rules.len() {
        if mark[start] != Mark::White {
            continue;
        }
        let mut stack: Vec<(usize, Vec<usize>, usize)> = vec![(start, refs_of(g, start), 0)];
        mark[start] = Mark::Grey;
        while let Some((node, kids, next)) = stack.last_mut() {
            if *next >= kids.len() {
                mark[*node] = Mark::Black;
                stack.pop();
                continue;
            }
            let kid = kids[*next];
            *next += 1;
            match mark[kid] {
                Mark::Grey => {
                    let cycle: Vec<&str> = stack
                        .iter()
                        .skip_while(|(n, _, _)| *n != kid)
                        .map(|(n, _, _)| g.rules[*n].symbol.as_str())
                        .collect();
                    let path = if cycle.is_empty() {
                        g.rules[kid].symbol.clone()
                    } else {
                        cycle.join(" -> ")
                    };
                    return Err(GrammarError::new(
                        g.rule_sites[kid].0,
                        g.rule_sites[kid].1,
                        format!(
                            "rule `{}` is recursive ({} -> {}); v1 grammars must be acyclic — \
                             use a repetition (`*`, `+`, `{{n}}`) instead",
                            g.rules[kid].symbol, path, g.rules[kid].symbol
                        ),
                    ));
                }
                Mark::White => {
                    mark[kid] = Mark::Grey;
                    stack.push((kid, refs_of(g, kid), 0));
                }
                Mark::Black => {}
            }
        }
    }
    Ok(())
}

/// The rule indices rule `i` references (deduplicated, ascending).
fn refs_of(g: &Grammar, i: usize) -> Vec<usize> {
    let mut out: BTreeSet<usize> = BTreeSet::new();
    visit_symbols(&g.rules[i].alternatives, &mut |name, _| {
        if let Some(&j) = g.rule_index.get(name) {
            out.insert(j);
        }
    });
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FENCE: &str = "\
# a fence
module Post  = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ff size 0.2
module Panel = mesh 6f9619ff-8b86-d011-b42d-00c04fc964f0 size 2 offset 0,0,1

Fence -> Post Bay* Post
Bay   -> Panel | Gate[2m]@0.5
";

    #[test]
    fn parses_the_worked_example() {
        let g = Grammar::parse(FENCE).unwrap();
        assert_eq!(g.modules().len(), 2);
        assert_eq!(g.modules()[0].name, "Post");
        assert_eq!(g.modules()[0].size, Some(0.2));
        assert_eq!(g.modules()[1].offset, DVec3::new(0.0, 0.0, 1.0));
        assert_eq!(g.module_index("Panel"), Some(1));
        assert_eq!(g.default_axiom(), Some("Fence"));
        assert_eq!(g.rules().len(), 2);
        let fence = g.rule("Fence").unwrap();
        assert_eq!(fence.alternatives.len(), 1);
        assert_eq!(fence.alternatives[0].elements.len(), 3);
        assert_eq!(fence.alternatives[0].elements[1].repeat, Repeat::Fill);
        let bay = g.rule("Bay").unwrap();
        assert_eq!(bay.alternatives.len(), 2);
        assert_eq!(bay.alternatives[0].weight, 1.0);
        assert_eq!(bay.alternatives[1].weight, 0.5);
        // `Gate` has an inline size but no module: a legal gap, and reported.
        assert_eq!(g.gaps(), vec!["Gate"]);
        assert_eq!(
            g.mesh_refs(),
            vec![
                Uuid::parse_str("6f9619ff-8b86-d011-b42d-00c04fc964ff").unwrap(),
                Uuid::parse_str("6f9619ff-8b86-d011-b42d-00c04fc964f0").unwrap(),
            ]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
        );
    }

    /// **P19.5's one DSL addition.** A `collider` is optional, keeps its
    /// half-extents verbatim, survives the canonical-text round trip, and is
    /// absent by default — a P19.4 grammar must not silently start being solid.
    #[test]
    fn a_collider_is_optional_and_round_trips() {
        let g = Grammar::parse(
            "module Panel = size 2 offset 0,1.5,1 collider 0.1,1.5,1\n\
             module Ghost = size 2\n\
             W -> Panel Ghost\n",
        )
        .unwrap();
        assert_eq!(g.modules()[0].collider, Some(DVec3::new(0.1, 1.5, 1.0)));
        assert_eq!(g.modules()[1].collider, None, "opt-in, not opt-out");
        // Every P19.4 grammar in this file is collider-free by construction.
        let fence = Grammar::parse(FENCE).unwrap();
        assert!(fence.modules().iter().all(|m| m.collider.is_none()));
        // Round trip: printing and re-parsing reproduces the palette exactly.
        let again = Grammar::parse(&g.to_text()).unwrap();
        assert_eq!(again.modules(), g.modules());
        assert!(g.to_text().contains("collider 0.1,1.5,1"));
    }

    /// A non-positive half-extent is a **parse error anchored at the value**,
    /// not a silently-ignored zero-volume box nothing could ever touch.
    #[test]
    fn a_degenerate_collider_is_rejected_at_its_value() {
        for text in [
            "module M = size 1 collider 0,1,1",
            "module M = size 1 collider 1,0,1",
            "module M = size 1 collider 1,1,0",
            "module M = size 1 collider -1,1,1",
        ] {
            let err = Grammar::parse(text).expect_err(text);
            assert!(
                err.message.contains("half-extents must all be positive"),
                "{text}: {}",
                err.message
            );
            assert_eq!((err.line, err.col), (1, 28), "{text}");
        }
        // The unknown-attribute message now names `collider` too, so a typo
        // points at the real list.
        let err = Grammar::parse("module M = collide 1,1,1").unwrap_err();
        assert!(err.message.contains("collider"), "{}", err.message);
        // And it is still a duplicate-checked attribute.
        let dup = Grammar::parse("module M = size 1 collider 1,1,1 collider 2,2,2").unwrap_err();
        assert!(dup.message.contains("duplicate"), "{}", dup.message);
    }

    #[test]
    fn statements_end_at_a_newline_or_a_semicolon() {
        let a = Grammar::parse("A -> B[1] C[1]\nB2 -> C[2]").unwrap();
        let b = Grammar::parse("A -> B[1] C[1]; B2 -> C[2];").unwrap();
        assert_eq!(a.rules(), b.rules());
        assert_eq!(a.rules().len(), 2);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let g =
            Grammar::parse("\n\n# lead comment\nA -> B[1] // trailing\n\n  # gap\nC -> D[3]\n\n")
                .unwrap();
        assert_eq!(g.rules().len(), 2);
        assert_eq!(g.rules()[1].symbol, "C");
    }

    #[test]
    fn sizes_parse_fixed_flexible_and_unit_suffixed() {
        let g = Grammar::parse("A -> F[2] G[0.5..1.5] H[3m] I[1..2m]").unwrap();
        let els = &g.rules()[0].alternatives[0].elements;
        let size = |i: usize| match &els[i].primary {
            Primary::Symbol { size, .. } => *size,
            _ => panic!("group"),
        };
        assert_eq!(size(0), Some(SizeSpec::Fixed(2.0)));
        assert_eq!(size(1), Some(SizeSpec::Flex { min: 0.5, max: 1.5 }));
        assert_eq!(size(2), Some(SizeSpec::Fixed(3.0)));
        assert_eq!(size(3), Some(SizeSpec::Flex { min: 1.0, max: 2.0 }));
    }

    #[test]
    fn repeats_and_groups_parse() {
        let g = Grammar::parse("A -> (B[1] | C[2])* D[1]{3} E[1]? F[1]+").unwrap();
        let els = &g.rules()[0].alternatives[0].elements;
        assert!(matches!(els[0].primary, Primary::Group(ref a) if a.len() == 2));
        assert_eq!(els[0].repeat, Repeat::Fill);
        assert_eq!(els[1].repeat, Repeat::Exactly(3));
        assert_eq!(els[2].repeat, Repeat::Optional);
        assert_eq!(els[3].repeat, Repeat::FillAtLeastOne);
    }

    #[test]
    fn an_empty_alternative_is_an_epsilon_production() {
        let g = Grammar::parse("A -> B[1] |").unwrap();
        assert_eq!(g.rules()[0].alternatives.len(), 2);
        assert!(g.rules()[0].alternatives[1].elements.is_empty());
    }

    /// **Round-trip.** Printing and re-parsing reproduces the rules and the
    /// palette. Line numbers are not preserved (a printer cannot know them) and
    /// are excluded — stated here rather than quietly asserted around.
    #[test]
    fn to_text_round_trips() {
        for src in [
            FENCE,
            "A -> B[1]",
            "A -> (B[1] | C[0.5..2])* D[1]{2} E[1]? F[1]+ @2.5 | G[1]",
            "module M = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ff offset 1,-2,3.5 rot 0,90,0 scale 2.25 size 4\nA -> M",
            "A -> |B[1]",
        ] {
            let g = Grammar::parse(src).unwrap();
            let printed = g.to_text();
            let back = Grammar::parse(&printed)
                .unwrap_or_else(|e| panic!("reparse of\n{printed}\nfailed: {e}"));
            assert_eq!(g.rules(), back.rules(), "rules differ after\n{printed}");
            assert_eq!(g.modules(), back.modules(), "modules differ");
        }
    }

    /// Every error case, with the message and **both** anchor coordinates
    /// asserted — the diagnostics are the product here, not a by-product.
    ///
    /// The columns are exact, not `>= 1`: an assertion a `u32` cannot fail is
    /// not a test, and the anchor is what an editor puts a caret on. Where a
    /// case is deliberately anchored somewhere other than the offending token
    /// (a bad *range* is wrong as a pair, so it points at the `[` that opens
    /// the specification) the comment says so.
    #[test]
    fn errors_are_anchored_and_say_what_to_do() {
        /// `(source, line, column, message fragment)`.
        type Case = (&'static str, u32, u32, &'static str);
        const GUID: &str = "6f9619ff-8b86-d011-b42d-00c04fc964ff";
        let dup_attr = format!("module M = mesh {GUID} mesh {GUID}");
        let cases: Vec<Case> = vec![
            // ── post-parse validation: anchored at the RULE'S OWN first token,
            // so an indented statement reports where it actually starts.
            ("A -> B", 1, 1, "has no size"),
            ("  A -> B", 1, 3, "has no size"),
            ("A -> B[1]\nC -> A[2]", 2, 1, "cannot carry a size"),
            ("A -> B[1]\n    C -> A[2]", 2, 5, "cannot carry a size"),
            ("A -> B\n", 1, 1, "has no size"),
            // A size on a rule reference is caught before the cycle check.
            ("A -> B[1] C[1]\nB -> C[1]", 1, 1, "cannot carry a size"),
            // The cycle is reported on the rule the walk re-enters, which is
            // where the loop closes.
            ("A -> C[1] B\nB -> A", 1, 1, "recursive"),
            ("A -> A", 1, 1, "recursive"),
            ("   A -> A", 1, 4, "recursive"),
            // ── duplicate declarations: the statement's own first token.
            ("A -> B[1]\nA -> C[1]", 2, 1, "declared twice"),
            ("A -> B[1]\n   A -> C[1]", 2, 4, "declared twice"),
            ("module M =\nmodule M =", 2, 1, "declared twice"),
            ("module M =\n  module M =", 2, 3, "declared twice"),
            // ── in-parse: the token that was actually found.
            ("A B[1]", 1, 3, "expected `->`"),
            ("A -> B[1] )", 1, 11, "expected end of rule"),
            ("A -> B[1..]", 1, 11, "expected the upper bound"),
            ("A -> B[1]@0", 1, 11, "weight must be positive"),
            ("A -> B[1]{2.5}", 1, 11, "must be whole"),
            ("A -> B[1]{99999}", 1, 11, "between 0 and"),
            ("module M = colour 1,2,3", 1, 12, "unknown module attribute"),
            ("module M = mesh nope", 1, 17, "expects a GUID"),
            ("module M = rot 1,2,\nA -> X[1]", 1, 20, "expected rot Z"),
            ("A -> B[1] $", 1, 11, "unexpected character"),
            // Unbalanced: the anchor is end-of-input, which is where the `)`
            // should have been.
            ("A -> (B[1]", 1, 11, "expected `)`"),
            // ── bad VALUES anchor on the value, not on the keyword that
            // introduced it: `scale 0` is wrong at the `0`.
            (
                "module M = scale 0\nA -> M[1]",
                1,
                18,
                "scale` must be positive",
            ),
            (
                "module M = size -2\nA -> M",
                1,
                17,
                "size` must be zero or positive",
            ),
            ("A -> B[-1]", 1, 8, "zero or positive"),
            // …but a bad RANGE is wrong as a PAIR, so it anchors at the `[`.
            ("A -> B[2..1]", 1, 7, "`[min..max]`"),
            // A duplicated attribute anchors on the SECOND occurrence.
            (
                "module M = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ff mesh \
                 6f9619ff-8b86-d011-b42d-00c04fc964ff",
                1,
                54,
                "duplicate module attribute",
            ),
        ];
        assert_eq!(
            dup_attr,
            cases
                .last()
                .unwrap()
                .0
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
            "the duplicate-attribute fixture drifted from its GUID constant"
        );
        for (src, line, col, needle) in cases {
            let err = Grammar::parse(src)
                .map(|_| String::from("<parsed>"))
                .unwrap_err_or_report(src);
            assert!(
                err.message.contains(needle),
                "`{src}`: message {:?} does not contain {needle:?}",
                err.message
            );
            assert_eq!((err.line, err.col), (line, col), "`{src}`: wrong anchor");
        }
    }

    /// **The parser's own recursion is bounded, and exceeding the bound is an
    /// ordinary error rather than a process abort.**
    ///
    /// `parse_alternatives` → `parse_alternative` → `parse_element` → group →
    /// `parse_alternatives` is a native-stack cycle driven by authored text. A
    /// stack overflow there cannot be caught, and the same text reaches the
    /// editor, the cook *and* the shipped player (every one of them re-lowers a
    /// stored `graph_json`) — so this is a safety property, not ergonomics. The
    /// acyclic-rule check that makes v1 terminate by construction sits one layer
    /// downstream and never sees the input at all.
    #[test]
    fn deeply_nested_text_errors_instead_of_overflowing_the_stack() {
        // Exactly at the cap parses; one past it is an anchored error.
        let nest = |n: usize, closed: bool| {
            let mut s = String::from("A -> ");
            s.push_str(&"(".repeat(n));
            s.push_str("B[1]");
            if closed {
                s.push_str(&")".repeat(n));
            }
            s
        };
        assert!(
            Grammar::parse(&nest(MAX_NESTING as usize - 1, true)).is_ok(),
            "the cap must not reject ordinary nesting"
        );
        let err = Grammar::parse(&nest(MAX_NESTING as usize + 1, true)).unwrap_err();
        assert!(err.message.contains("nested deeper than"), "{err}");
        assert!(err.line >= 1 && err.col >= 1);

        // The reported case: ~10 KB of nesting, and an unbalanced variant so the
        // guard is not quietly doing its work through the `)` matcher.
        for closed in [true, false] {
            let err = Grammar::parse(&nest(5000, closed)).unwrap_err();
            assert!(
                err.message.contains("nested deeper than"),
                "closed={closed}: {err}"
            );
        }
        // Nesting inside a MODULE-heavy document, and inside an alternative, so
        // the guard is not specific to one entry point.
        assert!(Grammar::parse(&format!(
            "module M = size 1\nA -> M | {}M{}",
            "(".repeat(4000),
            ")".repeat(4000)
        ))
        .unwrap_err()
        .message
        .contains("nested deeper than"));
        // A very long FLAT sequence is not nesting and must still parse: the cap
        // bounds depth, never length.
        let flat = format!("A -> {}", "B[1] ".repeat(20_000));
        let g = Grammar::parse(&flat).expect("a flat sequence is not deep");
        assert_eq!(g.rules()[0].alternatives[0].elements.len(), 20_000);
    }

    /// A helper so the error corpus reads as data. Panics with the source when a
    /// case unexpectedly parses.
    trait UnwrapErrOrReport {
        fn unwrap_err_or_report(self, src: &str) -> GrammarError;
    }
    impl UnwrapErrOrReport for Result<String, GrammarError> {
        fn unwrap_err_or_report(self, src: &str) -> GrammarError {
            match self {
                Ok(_) => panic!("`{src}` parsed but should not have"),
                Err(e) => e,
            }
        }
    }

    /// **Every SHAPE a cycle can hide in is rejected.**
    ///
    /// `refs_of` walks the whole alternative tree today, but nothing in its
    /// signature says it must — a future "fast path" that only looked at the
    /// first element of the first alternative would pass a direct-recursion
    /// test and quietly reintroduce non-termination, which in v1 is not a
    /// performance problem but a hang. So each structural hiding place gets its
    /// own case, with a non-recursive twin beside it so the test cannot pass by
    /// rejecting everything.
    #[test]
    fn a_cycle_is_rejected_in_every_shape_it_can_hide_in() {
        let cyclic: Vec<(&str, &str)> = vec![
            ("direct", "A -> A"),
            ("indirect (2)", "A -> B\nB -> A"),
            ("indirect (4)", "A -> B\nB -> C\nC -> D\nD -> A"),
            // Not the first element: a fast path that only inspected element 0
            // would miss this.
            ("late in the sequence", "A -> X[1] Y[2] B\nB -> A"),
            // Not the first alternative.
            ("second alternative", "A -> X[1] | B\nB -> A"),
            // Behind a weight.
            ("weighted alternative", "A -> X[1] | B @3\nB -> A"),
            // Inside a group, and inside a nested group.
            ("inside a group", "A -> (X[1] | B)\nB -> A"),
            ("inside a nested group", "A -> ((X[1] | (B)))\nB -> A"),
            // Behind every repetition operator.
            ("behind `*`", "A -> B*\nB -> A"),
            ("behind `+`", "A -> B+\nB -> A"),
            ("behind `?`", "A -> B?\nB -> A"),
            ("behind `{n}`", "A -> B{2}\nB -> A"),
            ("behind a repeated group", "A -> (B)*\nB -> A"),
            // The cycle does not involve the FIRST rule declared.
            ("not through the axiom", "Root -> A\nA -> B\nB -> A"),
            // A rule reachable from nothing still cannot be recursive.
            ("in an unreachable rule", "Root -> X[1]\nA -> B\nB -> A"),
        ];
        for (label, src) in cyclic {
            let err = Grammar::parse(src)
                .map(|_| String::from("<parsed>"))
                .unwrap_err_or_report(src);
            assert!(
                err.message.contains("recursive"),
                "{label}: expected a recursion error, got {:?}",
                err.message
            );
        }

        // The non-recursive twins: the same shapes, acyclic, must all parse —
        // otherwise the test above proves only that the validator says no.
        for (label, src) in [
            ("direct", "A -> X[1]"),
            ("indirect (4)", "A -> B\nB -> C\nC -> D\nD -> X[1]"),
            ("late in the sequence", "A -> X[1] Y[2] B\nB -> Z[1]"),
            ("second alternative", "A -> X[1] | B\nB -> Z[1]"),
            ("weighted alternative", "A -> X[1] | B @3\nB -> Z[1]"),
            ("inside a nested group", "A -> ((X[1] | (B)))\nB -> Z[1]"),
            ("behind `*`", "A -> B*\nB -> Z[1]"),
            ("behind a repeated group", "A -> (B)*\nB -> Z[1]"),
            // A DIAMOND is not a cycle: two rules may both reach a third.
            ("a diamond", "A -> B C\nB -> D\nC -> D\nD -> X[1]"),
            // …and one rule may reference another twice.
            ("a repeated reference", "A -> B B\nB -> X[1]"),
        ] {
            assert!(
                Grammar::parse(src).is_ok(),
                "{label}: an ACYCLIC grammar was rejected — `{src}`"
            );
        }
    }

    /// A long non-recursive chain must not blow the validator's stack: the cycle
    /// check is an explicit-stack DFS, not native recursion.
    #[test]
    fn a_deep_rule_chain_validates_without_recursing_natively() {
        let mut src = String::new();
        const N: usize = 2000;
        for i in 0..N {
            src.push_str(&format!("R{i} -> R{}\n", i + 1));
        }
        src.push_str(&format!("R{N} -> Leaf[1]\n"));
        let g = Grammar::parse(&src).unwrap();
        assert_eq!(g.rules().len(), N + 1);
        assert_eq!(g.default_axiom(), Some("R0"));
    }

    /// The GUID lexer must not eat a longer word that merely starts like one,
    /// and must survive a GUID at the very end of the text.
    #[test]
    fn guid_lexing_is_exact() {
        let g = Grammar::parse("module M = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ff").unwrap();
        assert!(g.modules()[0].mesh.is_some());
        assert!(Grammar::parse("module M = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ffx").is_err());
        assert!(Grammar::parse("module M = mesh 6f9619ff").is_err());
    }

    #[test]
    fn an_empty_grammar_is_legal_and_empty() {
        let g = Grammar::parse("   \n # nothing \n").unwrap();
        assert!(g.is_empty());
        assert_eq!(g.default_axiom(), None);
        assert!(g.gaps().is_empty());
        assert!(g.mesh_refs().is_empty());
    }

    #[test]
    fn negative_and_exponent_numbers_lex() {
        let g = Grammar::parse("module M = offset -1.5,2e2,-3e-1 size 1\nA -> M").unwrap();
        assert_eq!(g.modules()[0].offset, DVec3::new(-1.5, 200.0, -0.3));
    }

    #[test]
    fn terminal_size_prefers_the_inline_override() {
        let g = Grammar::parse("module M = size 2\nA -> M M[5]").unwrap();
        assert_eq!(g.terminal_size("M", None), Some(SizeSpec::Fixed(2.0)));
        assert_eq!(
            g.terminal_size("M", Some(SizeSpec::Fixed(5.0))),
            Some(SizeSpec::Fixed(5.0))
        );
        assert_eq!(g.terminal_size("Nope", None), None);
    }
}
