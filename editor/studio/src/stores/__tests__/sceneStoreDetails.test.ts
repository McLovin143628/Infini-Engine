import { beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import type { DetailsDto } from "../../bindings/DetailsDto";
import type { PropValueDto } from "../../bindings/PropValueDto";

// The scene IPC is mocked so we can control response ordering and prove the
// Details request token discards stale, out-of-order responses.
vi.mock("../../lib/ipc", () => ({
  scene: {
    details: vi.fn(),
    setProperty: vi.fn(),
    resetProperty: vi.fn(),
  },
}));

import { scene as sceneIpc } from "../../lib/ipc";
import { useSceneStore } from "../sceneStore";

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const dto = (tag: string) => ({ tag }) as unknown as DetailsDto;
const details = sceneIpc.details as unknown as Mock;
const setPropertyIpc = sceneIpc.setProperty as unknown as Mock;

beforeEach(() => {
  vi.clearAllMocks();
  useSceneStore.setState({ details: null, selection: [] });
});

describe("Details request token", () => {
  it("ignores a stale refreshDetails response resolving after a newer one", async () => {
    const first = deferred<DetailsDto>();
    const second = deferred<DetailsDto>();
    details.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    const store = useSceneStore.getState();
    const p1 = store.refreshDetails(); // token 1
    const p2 = store.refreshDetails(); // token 2 (newest)

    // Newest resolves first and is applied…
    second.resolve(dto("newest"));
    await p2;
    expect(useSceneStore.getState().details).toEqual(dto("newest"));

    // …then the older, slower response arrives and must NOT clobber it.
    first.resolve(dto("stale"));
    await p1;
    expect(useSceneStore.getState().details).toEqual(dto("newest"));
  });

  it("a stale setProperty response does not overwrite a newer selection's details", async () => {
    useSceneStore.setState({ selection: ["a"] });

    const slowSet = deferred<DetailsDto>();
    setPropertyIpc.mockReturnValueOnce(slowSet.promise);
    const freshRefresh = deferred<DetailsDto>();
    details.mockReturnValueOnce(freshRefresh.promise);

    const pSet = useSceneStore
      .getState()
      .setProperty("T", "f", {} as PropValueDto); // token N
    const pRefresh = useSceneStore.getState().refreshDetails(); // token N+1 (selection moved on)

    freshRefresh.resolve(dto("current-selection"));
    await pRefresh;
    slowSet.resolve(dto("old-edit"));
    await pSet;

    expect(useSceneStore.getState().details).toEqual(dto("current-selection"));
  });

  it("applies the setProperty result when it is the newest request", async () => {
    useSceneStore.setState({ selection: ["a"] });
    setPropertyIpc.mockResolvedValueOnce(dto("edited"));
    await useSceneStore.getState().setProperty("T", "f", {} as PropValueDto);
    expect(useSceneStore.getState().details).toEqual(dto("edited"));
  });
});
