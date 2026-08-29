// @vitest-environment jsdom
/**
 * **Open a `.infini`, edit it, Ctrl+S** (wave SCRIPT2b) — the whole loop the
 * arc's iteration story rests on, driven end to end on this side of the wire.
 *
 * The engine half is already gated and is not re-proved here: SCRIPT1b's
 * `hot_reload_scripts` compiles a saved script through `inf_script::source` and
 * swaps it into a running Simulate on the next fixed step, and its own arms
 * measure that. What had never been armed is the half a designer actually
 * touches — that a double-click in the Content Drawer reaches the editor, that
 * the tab carries the language and the linter, and that Ctrl+S writes the
 * bytes the watcher then picks up.
 *
 * The two ends meet at ONE fact, which is why this suite is worth writing: the
 * path the drawer emits is the path `file_write` receives, so the file the
 * author edited is the file the watcher recompiles.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  files: { read: vi.fn(), write: vi.fn(), list: vi.fn() },
  script: { check: vi.fn() },
}));
vi.mock("../../panels/dock/dockLayoutStore", () => ({
  useDockLayout: { getState: () => ({ openPanel: vi.fn() }) },
}));

import { files, script } from "../../lib/ipc";
import { contentAbsPath } from "../assetStore";
import { requestOpenFile } from "../../lib/openFile";
import { scriptCompartment, languageCompartment } from "../../lib/editor/setup";
import { useLspStore } from "../lspStore";
import { pathToUri } from "../../lib/editor/fileUri";
import {
  blueprintTextPath,
  initEditorSync,
  tabStateFor,
  useEditorStore,
} from "../editorStore";

const ROOT = "C:/proj/Content";
const REL = "Scripts/Door.infini";
const ABS = "C:/proj/Content/Scripts/Door.infini";
const SOURCE = 'actor "Door"\n\non tick(dt)\n  debug.print("tick")\nend\n';

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(files.read).mockResolvedValue(SOURCE);
  vi.mocked(files.write).mockResolvedValue(undefined);
  vi.mocked(script.check).mockResolvedValue([]);
  useEditorStore.setState({ tabs: [], activeId: null });
  useLspStore.getState().reset();
});

describe("a .infini goes from the drawer to the editor and back to disk", () => {
  it("joins the content root and the asset's relative path exactly once", () => {
    expect(contentAbsPath(ROOT, REL)).toBe(ABS);
    // Trailing and leading separators do not double up…
    expect(contentAbsPath("C:/proj/Content/", "/Scripts/Door.infini")).toBe(ABS);
    expect(contentAbsPath("C:\\proj\\Content\\", REL)).toBe("C:\\proj\\Content/Scripts/Door.infini");
    // …and an unknown root is NOT a path pointing at the filesystem's root.
    expect(contentAbsPath("", REL)).toBe("");
  });

  it("opens through the existing infinity:open-file event", async () => {
    const dispose = initEditorSync();
    // Exactly what the drawer's double-click does.
    requestOpenFile(contentAbsPath(ROOT, REL));
    await vi.waitFor(() => expect(useEditorStore.getState().tabs.length).toBe(1));

    const tab = useEditorStore.getState().tabs[0];
    expect(tab.path).toBe(ABS);
    expect(tab.name).toBe("Door.infini");
    expect(tab.dirty).toBe(false);
    expect(files.read).toHaveBeenCalledWith(ABS);
    dispose();
  });

  it("gives the tab a language AND a linter, from its path alone", async () => {
    await useEditorStore.getState().openFile(ABS);
    const state = tabStateFor(useEditorStore.getState().activeId!)!;
    expect(state.doc.toString()).toBe(SOURCE);
    expect(languageCompartment.get(state)).toBeTruthy();
    const linter = scriptCompartment.get(state);
    expect(Array.isArray(linter) && linter.length === 0).toBe(false);
  });

  it("Ctrl+S writes the buffer to the path the drawer named", async () => {
    await useEditorStore.getState().openFile(ABS);
    const id = useEditorStore.getState().activeId!;
    useEditorStore.getState().markDirty(id, true);

    await useEditorStore.getState().saveActive();

    // THE ARM: the bytes and the path the watcher will see.
    expect(files.write).toHaveBeenCalledWith(ABS, SOURCE);
    expect(useEditorStore.getState().tabs[0].dirty).toBe(false);
  });

  it("a failed save leaves the tab dirty rather than claiming it saved", async () => {
    await useEditorStore.getState().openFile(ABS);
    const id = useEditorStore.getState().activeId!;
    useEditorStore.getState().markDirty(id, true);
    vi.mocked(files.write).mockRejectedValueOnce(new Error("read-only"));
    vi.spyOn(console, "error").mockImplementation(() => {});

    await useEditorStore.getState().saveActive();

    expect(useEditorStore.getState().tabs[0].dirty).toBe(true);
  });

  it("closing the tab retires the script's Problems rows", async () => {
    await useEditorStore.getState().openFile(ABS);
    const uri = pathToUri(ABS);
    useLspStore.getState().setDiagnostics(uri, [
      {
        range: { start: { line: 3, character: 2 }, end: { line: 3, character: 7 } },
        severity: 1,
        message: "a refusal",
        source: "infiniscript",
      },
    ]);
    expect(useLspStore.getState().diagnostics[uri]).toHaveLength(1);

    useEditorStore.getState().closeTab(useEditorStore.getState().activeId!);

    expect(useLspStore.getState().diagnostics[uri]).toBeUndefined();
    expect(useEditorStore.getState().tabs).toHaveLength(0);
  });

  it("opens a blueprint as read-only InfiniScript text, with the language and the linter", () => {
    const path = blueprintTextPath("11112222-3333-4444-5555-666677778888", "Door");
    useEditorStore.getState().openText(path, SOURCE);

    const tab = useEditorStore.getState().tabs[0];
    expect(tab.readOnly).toBe(true);
    expect(tab.name).toBe("Door.infini");
    // The synthetic path is unmistakably not a filesystem location…
    expect(path.startsWith("infini://")).toBe(true);
    expect(files.read).not.toHaveBeenCalled();

    // …and yet it still gets the `.infini` mode AND the linter, so the emitted
    // text is checked by the same Ring-0 compiler a real file would be.
    const state = tabStateFor(tab.id)!;
    expect(state.doc.toString()).toBe(SOURCE);
    expect(languageCompartment.get(state)).toBeTruthy();
    const linter = scriptCompartment.get(state);
    expect(Array.isArray(linter) && linter.length === 0).toBe(false);
    expect(state.readOnly).toBe(true);
  });

  it("never writes a read-only tab, even when asked directly", async () => {
    useEditorStore.getState().openText(blueprintTextPath("id", "Door"), SOURCE);
    await useEditorStore.getState().saveActive();
    expect(files.write).not.toHaveBeenCalled();
  });

  it("re-opening a blueprint as text refreshes it rather than duplicating the tab", () => {
    const path = blueprintTextPath("id", "Door");
    useEditorStore.getState().openText(path, SOURCE);
    const first = useEditorStore.getState().tabs[0].id;

    // The class moved since; a read-only view has no edits to protect.
    useEditorStore.getState().openText(path, 'actor "Door"\n');

    expect(useEditorStore.getState().tabs).toHaveLength(1);
    expect(useEditorStore.getState().tabs[0].id).toBe(first);
    expect(tabStateFor(first)!.doc.toString()).toBe('actor "Door"\n');
  });

  it("leaves a Rust tab's rows alone — those are the server's to retract", async () => {
    await useEditorStore.getState().openFile("C:/proj/src/main.rs");
    const uri = pathToUri("C:/proj/src/main.rs");
    useLspStore.getState().setDiagnostics(uri, [
      {
        range: { start: { line: 0, character: 0 }, end: { line: 0, character: 1 } },
        severity: 1,
        message: "mismatched types",
        source: "rustc",
      },
    ]);
    useEditorStore.getState().closeTab(useEditorStore.getState().activeId!);
    expect(useLspStore.getState().diagnostics[uri]).toHaveLength(1);
  });
});
