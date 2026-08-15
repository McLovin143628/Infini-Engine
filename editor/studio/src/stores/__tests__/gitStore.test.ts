// @vitest-environment jsdom
//
// **Every git mutation answers whether it happened** (round-2 finding R2.F10).
//
// `stage`/`unstage`/`discard`/`commit`/`init` each bare-`await`ed the IPC, the
// `error` slot was written only by `refresh`, and every call site is
// `void action(...)`. So a commit that failed — no `user.email` on a fresh
// machine, a pre-commit hook, an `index.lock` — was an unhandled promise
// rejection: nothing was shown, the `await refresh()` after the throw never ran
// so the file list did not even move, and `GitPanel` cleared the typed message
// unconditionally. **The user's evidence said it committed.**
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  git: {
    status: vi.fn(),
    stage: vi.fn(),
    unstage: vi.fn(),
    discard: vi.fn(),
    commit: vi.fn(),
    init: vi.fn(),
  },
}));

import { git } from "../../lib/ipc";
import type { GitStatusDto } from "../../bindings/GitStatusDto";
import { useGitStore } from "../gitStore";
import { useProjectStore } from "../projectStore";

const STATUS: GitStatusDto = {
  is_repo: true,
  branch: "main",
  ahead: 0,
  behind: 0,
  files: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  useProjectStore.setState({
    current: { root: "C:/proj", name: "Demo", manifest: "C:/proj/inf.toml" },
  } as never);
  useGitStore.setState({ status: null, loading: false, error: null });
  vi.mocked(git.status).mockResolvedValue(STATUS);
  vi.mocked(git.stage).mockResolvedValue(undefined);
  vi.mocked(git.unstage).mockResolvedValue(undefined);
  vi.mocked(git.discard).mockResolvedValue(undefined);
  vi.mocked(git.commit).mockResolvedValue("abc1234");
  vi.mocked(git.init).mockResolvedValue(undefined);
});

describe("a mutation that works", () => {
  it("answers true, refreshes, and leaves no error", async () => {
    const ok = await useGitStore.getState().commit("a message");
    expect(ok).toBe(true);
    expect(vi.mocked(git.commit)).toHaveBeenCalledWith("C:/proj", "a message");
    expect(vi.mocked(git.status)).toHaveBeenCalled();
    expect(useGitStore.getState().error).toBeNull();
  });
});

describe("a mutation that fails", () => {
  it("answers false and keeps the reason", async () => {
    // THE case: `git commit` refusing because the machine has no identity.
    const why = "Author identity unknown: please tell me who you are";
    vi.mocked(git.commit).mockRejectedValue(why);

    const ok = await useGitStore.getState().commit("a message");
    expect(ok, "a failed commit reported success").toBe(false);
    expect(useGitStore.getState().error).toContain("who you are");
  });

  it("refreshes anyway, and the refresh does not erase the reason", async () => {
    // Both halves matter. `git` can fail part-way and still move the working
    // tree (a partial add, a hook that wrote files) — the old shape's
    // `await refresh()` sat AFTER the throw, so a failure left the panel
    // showing a tree that no longer existed. And a successful `git status`
    // clears `error`, which would erase the only account of the failure.
    vi.mocked(git.stage).mockRejectedValue("permission denied");

    const ok = await useGitStore.getState().stage(["a.rs"]);
    expect(ok).toBe(false);
    expect(vi.mocked(git.status), "the tree was not re-read").toHaveBeenCalled();
    expect(useGitStore.getState().status).toEqual(STATUS);
    expect(useGitStore.getState().error).toContain("permission denied");
  });

  it("covers every mutation, not just the one that was noticed", async () => {
    // Five doors had the same shape; a fix applied to one of them is the
    // half-fix this campaign keeps finding.
    for (const run of [
      () => useGitStore.getState().unstage(["a.rs"]),
      () => useGitStore.getState().discard(["a.rs"]),
      () => useGitStore.getState().init(),
    ]) {
      vi.mocked(git.unstage).mockRejectedValue("nope");
      vi.mocked(git.discard).mockRejectedValue("nope");
      vi.mocked(git.init).mockRejectedValue("nope");
      useGitStore.setState({ error: null });
      expect(await run()).toBe(false);
      expect(useGitStore.getState().error).toContain("nope");
    }
  });
});

describe("an empty commit message", () => {
  it("is not a commit and not an error", async () => {
    // The button is disabled for it, so this is belt-and-braces — but it must
    // answer `false`, or the panel clears a message it never sent.
    expect(await useGitStore.getState().commit("   ")).toBe(false);
    expect(vi.mocked(git.commit)).not.toHaveBeenCalled();
    expect(useGitStore.getState().error).toBeNull();
  });
});

describe("no project open", () => {
  it("answers false rather than pretending", async () => {
    useProjectStore.setState({ current: null } as never);
    expect(await useGitStore.getState().commit("x")).toBe(false);
    expect(vi.mocked(git.commit)).not.toHaveBeenCalled();
  });
});
