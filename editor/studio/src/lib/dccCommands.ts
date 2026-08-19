/**
 * **The Model Editor's keyboard, through the ONE registry** (Wave D).
 *
 * Until now the Model Editor's only key was Escape, handled by a local
 * `onKeyDown` on the preview `<img>`, and `DEFAULT_KEYBINDINGS` contained zero
 * DCC chords. A modeller's hands live on 1/2/3 and G/R/S; a modelling tool
 * without them is a tool they will use for one session.
 *
 * # Why these are commands and not a `switch` in the panel
 *
 * Wave E built the settings surface: `DEFAULT_KEYBINDINGS` is the shipped table,
 * `editor-settings.toml` stores only a user's OVERRIDES, and
 * `applyKeybindingOverrides` composes the two. Registering here means every
 * chord below is **rebindable for free**, appears in the command palette, and is
 * one row in the conflict check rather than a private listener nobody can see.
 * It is also what retires the P23 remainder "sculpt `[`/`]` is still outside the
 * keybinding registry; conflict resolution is a toast".
 *
 * # How a command finds the mesh
 *
 * The dock records the last panel to take a pointer-down. A DCC command reads
 * it, checks the panel is a Model Editor, and takes its parameter — the asset
 * id. Two Model Editors are two `"model"` panels, so keying on the panel TYPE
 * alone would send the chord to whichever document the store happened to hold;
 * this is the same reasoning `UndoScope`'s `params` argument already carries.
 *
 * A chord pressed with no Model Editor focused does **nothing**, silently. That
 * is deliberate: `G` is not a global shortcut, and a refusal toast every time an
 * author types in the Outliner would be worse than no keyboard at all.
 */
import { useDockLayout } from "../panels/dock/dockLayoutStore";
import { panelTypeOf } from "../panels/panelRegistry";
import { useDccStore } from "../stores/dccStore";
import { registerCommands, type CommandDef } from "./commands";
import type { DccGizmoModeDto } from "../bindings/DccGizmoModeDto";
import type { DccModeDto } from "../bindings/DccModeDto";
import type { DccSelectDto } from "../bindings/DccSelectDto";

/**
 * The focused Model Editor's asset id, or `null`.
 *
 * Read from the dock store at call time rather than subscribed to: a keybinding
 * fires once and wants the state as it is now.
 */
export function focusedMeshAsset(): string | null {
  const layout = useDockLayout.getState();
  const focused = layout.focusedPanel;
  if (!focused || panelTypeOf(focused) !== "model") return null;
  return layout.layout.panels[focused]?.params ?? null;
}

function onMesh(fn: (assetId: string) => void | Promise<void>): () => void {
  return () => {
    const id = focusedMeshAsset();
    if (id) void fn(id);
  };
}

const mode = (m: DccModeDto): CommandDef["run"] =>
  onMesh((id) => useDccStore.getState().setMode(id, m));

const gizmo = (m: DccGizmoModeDto): CommandDef["run"] =>
  onMesh((id) => useDccStore.getState().setGizmo(id, m));

const selection = (action: DccSelectDto): CommandDef["run"] =>
  onMesh((id) => useDccStore.getState().select(id, action));

/**
 * Every DCC command, with the category the palette groups them under.
 *
 * The ids are `dcc.*` so a settings file's overrides sort together, and the
 * titles say what the chord does to a MESH rather than to a panel — "Vertex
 * mode", not "Set mode 1".
 */
export const DCC_COMMANDS: CommandDef[] = [
  { id: "dcc.mode.vert", title: "Vertex mode", category: "Model", shortcut: "1", run: mode("vert") },
  { id: "dcc.mode.edge", title: "Edge mode", category: "Model", shortcut: "2", run: mode("edge") },
  { id: "dcc.mode.face", title: "Face mode", category: "Model", shortcut: "3", run: mode("face") },
  {
    id: "dcc.gizmo.translate",
    title: "Move gizmo",
    category: "Model",
    shortcut: "G",
    run: gizmo("translate"),
  },
  {
    id: "dcc.gizmo.rotate",
    title: "Rotate gizmo",
    category: "Model",
    shortcut: "R",
    run: gizmo("rotate"),
  },
  {
    id: "dcc.gizmo.scale",
    title: "Scale gizmo",
    category: "Model",
    shortcut: "S",
    run: gizmo("scale"),
  },
  {
    id: "dcc.select.all",
    title: "Select all",
    category: "Model",
    shortcut: "A",
    run: selection({ action: "all" }),
  },
  {
    id: "dcc.select.none",
    title: "Deselect all",
    category: "Model",
    shortcut: "Alt+A",
    run: selection({ action: "none" }),
  },
  {
    id: "dcc.select.invert",
    title: "Invert selection",
    category: "Model",
    shortcut: "Ctrl+I",
    run: selection({ action: "invert" }),
  },
  {
    id: "dcc.select.grow",
    title: "Grow selection",
    category: "Model",
    shortcut: "Ctrl+NumpadAdd",
    run: selection({ action: "grow" }),
  },
  {
    id: "dcc.select.linked",
    title: "Select linked",
    category: "Model",
    shortcut: "L",
    run: selection({ action: "linked" }),
  },
  {
    id: "dcc.select.loop",
    title: "Select edge loop",
    category: "Model",
    shortcut: "Alt+L",
    run: selection({ action: "loop" }),
  },
  {
    id: "dcc.select.ring",
    title: "Select edge ring",
    category: "Model",
    shortcut: "Alt+R",
    run: selection({ action: "ring" }),
  },
  {
    id: "dcc.save",
    title: "Save mesh",
    category: "Model",
    shortcut: "Ctrl+Shift+M",
    run: onMesh((id) => useDccStore.getState().save(id)),
  },
  {
    id: "dcc.frame",
    title: "Frame the model",
    category: "Model",
    shortcut: "F",
    run: onMesh((id) => useDccStore.getState().frame(id)),
  },
];

/**
 * Register the commands. Called once, from the shell's command bootstrap.
 *
 * The **chords** live in `DEFAULT_KEYBINDINGS` beside every other one, not here:
 * that is the table `applyKeybindingOverrides` composes with the user's file and
 * `PreferencesDialog` checks for conflicts, and a second table nobody composes
 * is a second table with its own conflicts. (It is also the direction that does
 * not make an import cycle — this module needs `keybindings.ts` for its type.)
 */
export function registerDccCommands(): void {
  registerCommands(DCC_COMMANDS);
}
