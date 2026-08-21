//! **The engine's in-game UI layer** (Ring 0, island wave I5).
//!
//! Before this crate the engine could draw a sprite in the world and nothing on
//! the screen: `inf-render-2d` is a *world* quad batcher going through the game
//! camera, and the shipped player had no HUD, no menu, no toast and no focus
//! model at all. A game with no settings dialog is a game a player cannot change
//! the resolution of, and one with no rebinding screen is a game a left-handed
//! player cannot play.
//!
//! # The shape
//!
//! Four pure layers, none of which owns a GPU, a window or a clock:
//!
//! * [`draw`] — screen-space rects and text, and the [`draw::UiDrawList`] they
//!   land in. Pixels, top-left origin, `+y` down.
//! * [`settings`] — [`settings::GameSettings`], the shipped game's user settings
//!   file, under the editor's own doctrine (absent-default, corrupt-refused,
//!   newer-refused, atomic, NaN-guarded).
//! * [`bindings`] — the rebinding table's rows over an `inf_input::InputMap`,
//!   the token a row carries, and the conflict a new one would cause.
//! * [`menu`] — the dialog's state, its keyboard reducer, and the projection
//!   that turns both into rows. [`view`] draws that same projection, and
//!   [`toast`] is the honest half of a control nothing consumes yet.
//!
//! Everything is a pure function of its inputs, so the whole layer is tested
//! without an adapter — which is what lets a settings dialog exist in a shipped
//! player at all in a repository whose CI has no GPU.
//!
//! # What draws it
//!
//! `inf_render::passes::ui`, after the tonemap. The node lives in `inf-render`
//! for exactly the reason the sprite node does — a node here would make
//! `inf-render -> inf-ui -> inf-render` — and it draws *after* the tonemap
//! because a menu that went through the scene's HDR target would be bloomed,
//! TAA'd and colour-graded along with the world behind it.

pub mod bindings;
pub mod draw;
// I6: the `I` key's surface - the settings dialog's state/reducer/projection
// triple again, over a snapshot of what a character is carrying.
pub mod inventory;
pub mod menu;
pub mod settings;
pub mod toast;
pub mod view;

pub use draw::{Align, Color, Rect, UiDrawList};
pub use inventory::{
    InventoryInput, InventoryOutcome, InventorySlot, InventoryState, InventoryVerb, InventoryView,
};
pub use menu::{Capture, MenuInput, MenuOutcome, MenuState, Page, Row, RowId};
pub use settings::GameSettings;
pub use toast::Toasts;
