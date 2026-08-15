// @vitest-environment jsdom
//
// `refCountedInit` — the StrictMode subscription lifetime (F-lens L7.M1/M2).
//
// Every case here is written against React 18 StrictMode's ACTUAL ordering:
// mount → cleanup → mount, all in one commit, all before anything the first
// mount awaited has resolved. That ordering is what broke the two shapes this
// helper replaced, so it is what the arms reproduce:
//
//  - a guard set AFTER the first await double-subscribes (M1);
//  - a synchronous guard that hands the second caller a no-op tears everything
//    down and never re-subscribes — LSP dead in `tauri dev` (M2).
import { describe, expect, it, vi } from "vitest";

import { refCountedInit } from "../events";

/**
 * The exact effect body every `App.tsx` call site uses, run the way StrictMode
 * runs it: mount, cleanup, mount — synchronously, in that order.
 */
function strictModeMount(acquire: () => Promise<() => void>) {
  const mount = () => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    const settled = acquire().then((fn) => (disposed ? fn() : (dispose = fn)));
    return {
      settled,
      cleanup: () => {
        disposed = true;
        dispose?.();
      },
    };
  };
  const first = mount();
  first.cleanup();
  const second = mount();
  return { first, second };
}

describe("refCountedInit", () => {
  it("subscribes ONCE across a StrictMode double mount", async () => {
    const teardown = vi.fn();
    const start = vi.fn(async () => teardown);
    const acquire = refCountedInit(start);

    const { first, second } = strictModeMount(acquire);
    await Promise.all([first.settled, second.settled]);

    // The whole point: two mounts, one subscription.
    expect(start).toHaveBeenCalledTimes(1);
    // …and the first cleanup released only its own hold.
    expect(teardown).not.toHaveBeenCalled();
  });

  it("leaves the subscription LIVE after the StrictMode remount (the LSP bug)", async () => {
    // The M2 defect in one assertion. With the old flag guard, mount #2 got a
    // no-op disposer and cleanup #1 tore the whole bridge down — so after this
    // sequence nothing was subscribed and nothing ever re-subscribed.
    const teardown = vi.fn();
    const acquire = refCountedInit(async () => teardown);

    const { first, second } = strictModeMount(acquire);
    await Promise.all([first.settled, second.settled]);

    expect(teardown).not.toHaveBeenCalled();

    // And the surviving holder can still release it.
    second.cleanup();
    expect(teardown).toHaveBeenCalledTimes(1);
  });

  it("tears down only when the LAST holder releases", async () => {
    const teardown = vi.fn();
    const acquire = refCountedInit(async () => teardown);

    const a = await acquire();
    const b = await acquire();
    a();
    expect(teardown).not.toHaveBeenCalled();
    b();
    expect(teardown).toHaveBeenCalledTimes(1);
  });

  it("re-subscribes after a genuine teardown", async () => {
    // A real unmount must not leave the module permanently "inited" — that was
    // the other half of the LSP bug.
    const teardown = vi.fn();
    const start = vi.fn(async () => teardown);
    const acquire = refCountedInit(start);

    (await acquire())();
    expect(start).toHaveBeenCalledTimes(1);
    expect(teardown).toHaveBeenCalledTimes(1);

    (await acquire())();
    expect(start).toHaveBeenCalledTimes(2);
    expect(teardown).toHaveBeenCalledTimes(2);
  });

  it("counts a new holder the moment it CALLS, not when its await resolves", async () => {
    // The synchronous-increment property, isolated — and it is load-bearing on
    // the re-entrant path, not the cold one. A holds the subscription; B calls
    // `acquire` (whose memo is already resolved, so B's own continuation is a
    // microtask away); A releases *synchronously* in that window. If the count
    // only moved on B's continuation, A's release would see zero holders, tear
    // the subscription down, and hand B a live-looking disposer for something
    // that is already dead.
    const teardown = vi.fn();
    const acquire = refCountedInit(async () => teardown);

    const a = await acquire();
    const pendingB = acquire(); // counted here…
    a(); // …so this must NOT be the last release.
    expect(teardown).not.toHaveBeenCalled();

    const b = await pendingB;
    expect(teardown).not.toHaveBeenCalled();
    b();
    expect(teardown).toHaveBeenCalledTimes(1);
  });

  it("does not re-run start for a holder that arrived while one was live", async () => {
    // The companion to the case above: B joined A's subscription, so releasing
    // both and acquiring again is the SECOND start, not the third.
    const start = vi.fn(async () => vi.fn());
    const acquire = refCountedInit(start);

    const a = await acquire();
    const b = await acquire();
    a();
    b();
    await acquire();
    expect(start).toHaveBeenCalledTimes(2);
  });

  it("ignores a disposer called twice", async () => {
    // Otherwise one careless caller drops the count below the truth and tears
    // down a subscription another holder is still using.
    const teardown = vi.fn();
    const acquire = refCountedInit(async () => teardown);

    const a = await acquire();
    const b = await acquire();
    a();
    a();
    a();
    expect(teardown).not.toHaveBeenCalled();
    b();
    expect(teardown).toHaveBeenCalledTimes(1);
  });

  /**
   * **The partial start** (round-2 finding R2-7).
   *
   * A `start` takes its disposers one `await` at a time. When a later one
   * rejects it never returns a teardown, so everything it had already
   * subscribed was orphaned: live for the life of the process, with no
   * reference anywhere that could release it. The failure that causes this is
   * exactly the kind that hits a whole batch (the IPC bridge going away), so it
   * stranded all of them at once.
   */
  it("releases what a failed start had already taken", async () => {
    const first = vi.fn();
    const second = vi.fn();
    const acquire = refCountedInit(async (sink) => {
      sink(first);
      sink(second);
      throw new Error("the third listen failed");
    });

    await expect(acquire()).rejects.toThrow("the third listen failed");
    expect(first, "a subscription outlived the start that took it").toHaveBeenCalledTimes(1);
    expect(second).toHaveBeenCalledTimes(1);
  });

  it("does not release a sink a start SUCCEEDED with", async () => {
    // The other direction. Draining on success would tear down the listeners
    // the caller is about to be handed — a correct-looking init that hears
    // nothing.
    const held = vi.fn();
    const teardown = vi.fn();
    const acquire = refCountedInit(async (sink) => {
      sink(held);
      return teardown;
    });

    const dispose = await acquire();
    expect(held).not.toHaveBeenCalled();
    dispose();
    expect(teardown).toHaveBeenCalledTimes(1);
  });

  it("keeps draining when one disposer throws", async () => {
    const bad = vi.fn(() => {
      throw new Error("already gone");
    });
    const good = vi.fn();
    const acquire = refCountedInit(async (sink) => {
      sink(bad);
      sink(good);
      throw new Error("start failed");
    });

    await expect(acquire()).rejects.toThrow("start failed");
    expect(good, "one bad disposer stranded the rest").toHaveBeenCalledTimes(1);
  });

  it("propagates a failed start and lets the next caller retry", async () => {
    let attempt = 0;
    const teardown = vi.fn();
    const acquire = refCountedInit(async () => {
      attempt += 1;
      if (attempt === 1) throw new Error("listen failed");
      return teardown;
    });

    await expect(acquire()).rejects.toThrow("listen failed");
    // A failed start must not leave the memo poisoned.
    const dispose = await acquire();
    expect(attempt).toBe(2);
    dispose();
    expect(teardown).toHaveBeenCalledTimes(1);
  });
});
