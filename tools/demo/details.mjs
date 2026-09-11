// Select a named object in the open level and filter its Details grid, over the
// Chrome DevTools Protocol — VEH3a's audit.
//
//   node tools/demo/details.mjs [port] [name-substring] [filter]
//
// Exit 0 = selected and the grid carries the filter's first match; 2 = no page
// target; 3 = nothing matched the name; 4 = the grid did not populate.
//
// # Why this exists
//
// Wave VEH3a owed a frame of the hundred tunables on the Details grid and could
// not get one. Its account: *"clicking the car in the viewport selects a body
// PANEL (`lower`), not the chassis that carries the `VehicleClass`; the
// Outliner's `chassis` filter finds all eight rigs and highlights a row, but
// the Details panel kept reading 'Select an object to view details' through
// four attempts at the selection."*
//
// A highlight is not a selection — the Outliner's filter box matches rows, and
// matching a row is not clicking it — and a viewport pick answers the thing
// under the cursor, which on a rigged car is a panel child. Both are the editor
// behaving as designed, and neither is something a scripted loop should be
// re-discovering with the mouse. `scene_select` is the door the Outliner's own
// click calls, so this calls it.
//
// The filter is driven through the React value setter rather than by typing,
// because a controlled input ignores `el.value = x` and the demo loop has no
// business synthesising keystrokes into a panel it can address directly.

const port = Number(process.argv[2] ?? 9222);
const want = (process.argv[3] ?? "chassis").toLowerCase();
const filter = process.argv[4] ?? "";
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
const evalJs = async (expression) => {
  const r = await send("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (r.result?.exceptionDetails) {
    return "EXC: " + JSON.stringify(r.result.exceptionDetails).slice(0, 400);
  }
  return r.result?.result?.value;
};
const invoke = (cmd, args) =>
  evalJs(
    `window.__TAURI_INTERNALS__.invoke(${JSON.stringify(cmd)}, ${JSON.stringify(args)})`,
  );

const snap = await invoke("scene_snapshot", {});
const rows = Array.isArray(snap?.nodes) ? snap.nodes : [];
console.log(`snapshot: ${rows.length} entities`);
const hit = rows.find((e) => String(e.name ?? "").toLowerCase().includes(want));
if (!hit) {
  console.log(`NO ENTITY whose name contains "${want}"`);
  ws.close();
  process.exit(3);
}
console.log(`selecting: ${hit.name} ${hit.guid}`);
await invoke("scene_select", { guids: [hit.guid], additive: false });
await sleep(600);

// The grid, from the BACKEND rather than from the DOM: `scene_details` is what
// the panel renders, so a claim made about it is a claim about the same data.
const details = await invoke("scene_details", {});
const comps = Array.isArray(details?.components) ? details.components : [];
const fields = comps.flatMap((c) => (c.fields ?? []).map((f) => f.label));
console.log(`details: ${details?.name} — ${comps.length} components, ${fields.length} fields`);
if (fields.length === 0) {
  console.log("THE GRID IS EMPTY for that selection");
  ws.close();
  process.exit(4);
}
const named = fields.filter((f) => f.toLowerCase().includes(filter.toLowerCase()));
console.log(`fields matching "${filter}": ${named.length}${named.length ? " — " + named.slice(0, 8).join(", ") : ""}`);

if (filter) {
  // A CONTROLLED input ignores `el.value = x`; the native setter plus a bubbled
  // `input` event is what React listens for.
  const typed = await evalJs(`(() => {
    const el = document.querySelector('input[placeholder="Filter properties\\u2026"]');
    if (!el) return "NO FILTER INPUT";
    const set = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    set.call(el, ${JSON.stringify(filter)});
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return "ok";
  })()`);
  console.log(`filter box: ${typed}`);
  await sleep(400);
}

const shown = await evalJs(`(() => {
  const rows = Array.from(document.querySelectorAll("div,span,label"))
    .map((n) => (n.childElementCount === 0 ? (n.textContent || "").trim() : ""))
    .filter((t) => t && t.length < 48);
  return rows.filter((t) => t.toLowerCase().includes(${JSON.stringify(filter.toLowerCase())})).slice(0, 8).join(" | ");
})()`);
console.log(`on screen: ${shown}`);

ws.close();
process.exit(named.length > 0 ? 0 : 4);
