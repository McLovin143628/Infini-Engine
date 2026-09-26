// Select ONE named child of a named rig in the open level -- the Details grid
// then shows that node's components -- over the Chrome DevTools Protocol
// (`audit(VEH3f.2b)`).
//
//   node tools/demo/select.mjs [port] [parent-name-substring] [child-name-substring]
//
// Exit 0 = selected; 2 = no page target; 3 = no rig of that name carries a child
// of that name.
//
// # Why this exists
//
// `details.mjs` finds the FIRST node whose name matches and walks up to the
// component it was asked for -- which is the chassis's `VehicleClass`. The
// VEH3f.2b shells hang their body on a child called `shell_body` on every
// shell car on the island, so "the first `shell_body`" is whichever car the
// document lists first, not the saloon the camera is looking at. This picks
// the rig by ITS name first, then the child, and selects exactly that node --
// the door the Outliner's own click calls (`scene_select`).

const port = Number(process.argv[2] ?? 9222);
const parentWant = (process.argv[3] ?? "").toLowerCase();
const childWant = (process.argv[4] ?? "shell_body").toLowerCase();
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
const byGuid = new Map(rows.map((e) => [e.guid, e]));
console.log(`snapshot: ${rows.length} entities`);
let pick = null;
for (const rig of rows) {
  if (!String(rig.name ?? "").toLowerCase().includes(parentWant)) continue;
  for (const c of rig.children ?? []) {
    const child = byGuid.get(c);
    if (child && String(child.name ?? "").toLowerCase().includes(childWant)) {
      pick = { rig, child };
      break;
    }
  }
  if (pick) break;
}
if (!pick) {
  console.log(`NO RIG named like "${parentWant}" carries a child named like "${childWant}"`);
  ws.close();
  process.exit(3);
}
await invoke("scene_select", { guids: [pick.child.guid], additive: false });
await sleep(500);
const d = await invoke("scene_details", {});
const cs = Array.isArray(d?.components) ? d.components : [];
console.log(`selected: ${pick.rig.name} / ${pick.child.name} ${pick.child.guid}`);
for (const c of cs) {
  const fs = (c.fields ?? []).map((f) => `${f.label}=${JSON.stringify(f.value ?? "").slice(0, 60)}`);
  console.log(`  ${c.name ?? c.type ?? "?"}: ${fs.slice(0, 6).join("; ")}`);
}
ws.close();
process.exit(0);
