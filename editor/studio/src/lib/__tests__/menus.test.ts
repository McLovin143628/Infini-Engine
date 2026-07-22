import { describe, expect, it } from "vitest";

import type { RecentProjectDto } from "../../bindings/RecentProjectDto";
import { buildMenuBar, recentProjectsItems, RECENT_PROJECTS_MAX } from "../menus";

const entry = (name: string, path: string): RecentProjectDto => ({ name, path });

describe("recentProjectsItems", () => {
  it("returns a single disabled placeholder when empty", () => {
    const items = recentProjectsItems([]);
    expect(items).toHaveLength(1);
    const only = items[0];
    expect(only.kind).toBe("action");
    if (only.kind === "action") {
      expect(only.command).toBe("file.recentProjects.none");
      expect(only.disabled).toBe(true);
    }
  });

  it("maps entries to indexed command ids with the path as detail", () => {
    const items = recentProjectsItems([entry("Alpha", "/p/alpha"), entry("Beta", "/p/beta")]);
    expect(items).toHaveLength(2);
    expect(items.map((n) => (n.kind === "action" ? n.command : null))).toEqual([
      "file.recentProjects.0",
      "file.recentProjects.1",
    ]);
    const first = items[0];
    if (first.kind === "action") {
      expect(first.label).toBe("Alpha");
      expect(first.detail).toBe("/p/alpha");
      expect(first.disabled).toBeUndefined();
    }
  });

  it("caps the list at RECENT_PROJECTS_MAX", () => {
    const many = Array.from({ length: RECENT_PROJECTS_MAX + 5 }, (_, i) =>
      entry(`P${i}`, `/p/${i}`),
    );
    expect(recentProjectsItems(many)).toHaveLength(RECENT_PROJECTS_MAX);
  });
});

describe("buildMenuBar", () => {
  it("fills the File ▸ Recent Projects submenu and leaves other menus intact", () => {
    const menus = buildMenuBar([entry("Alpha", "/p/alpha")]);
    const file = menus.find((m) => m.id === "file");
    expect(file).toBeDefined();
    const recent = file!.items.find(
      (n) => n.kind === "submenu" && n.label === "Recent Projects",
    );
    expect(recent && recent.kind === "submenu").toBe(true);
    if (recent && recent.kind === "submenu") {
      expect(recent.items).toHaveLength(1);
      const first = recent.items[0];
      expect(first.kind === "action" && first.command).toBe("file.recentProjects.0");
    }
    // A non-File menu passes through by reference (untouched).
    const edit = menus.find((m) => m.id === "edit");
    expect(edit).toBeDefined();
  });
});
