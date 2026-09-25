// Stand the editor's 3D camera at an eye point looking at a target, over the
// Chrome DevTools Protocol -- wave VEH3g audit.
//
//   node tools/demo/lookat.mjs [port] --eye=x,y,z --at=x,y,z
//
// Exit 0 = the command was accepted; 2 = no page target; 3 = the command
// refused (or bad arguments).
//
// # Why this exists
//
// The editor opens on `EngineHost::player_start_pose` -- behind the pawn -- and
// on the island that is 510 m from the Harbour City airfield, facing away. The
// wave that built the airfield could not put it in an editor frame: no IPC
// door moved the camera. `viewport_look_at` is that door (a jump cut, the
// level-open latch's `set_pose`); the document is not touched.

const port = Number(process.argv[2] ?? 9222);
const vec = (k) => {
  const a = process.argv.find((s) => s.startsWith(`--${k}=`));
  if (!a) return null;
  const v = a.slice(k.length + 3).split(",").map(Number);
  return v.length === 3 && v.every(Number.isFinite) ? v : null;
};
const eye = vec("eye");
const at = vec("at");
if (!eye || !at) {
  console.log("usage: lookat.mjs [port] --eye=x,y,z --at=x,y,z");
  process.exit(3);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function pageTarget() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/json`);
      const list = await r.json();
      const page = list.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
      if (page) return page;
    } catch {
      /* the port is not open yet */
    }
    await sleep(1000);
  }
  return null;
}

const page = await pageTarget();
if (!page) {
  console.log(`NO PAGE TARGET on port ${port} after 60 s`);
  process.exit(2);
}
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res, rej) => {
  ws.onopen = res;
  ws.onerror = rej;
});
let id = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
};
const send = (method, params = {}) =>
  new Promise((res) => {
    const i = ++id;
    pending.set(i, res);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const r = await send("Runtime.evaluate", {
  expression: `window.__TAURI_INTERNALS__.invoke("viewport_look_at", ${JSON.stringify({
    eye,
    target: at,
  })}).then(() => "ok")`,
  returnByValue: true,
  awaitPromise: true,
});
const value = r.result?.result?.value;
if (r.result?.exceptionDetails || value !== "ok") {
  console.log("REFUSED:", JSON.stringify(r.result).slice(0, 400));
  process.exit(3);
}
console.log(`camera at ${eye.join(",")} looking at ${at.join(",")}`);
ws.close();
process.exit(0);
