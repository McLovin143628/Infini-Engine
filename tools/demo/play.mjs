// Press the editor's Play button BY NAME, over the Chrome DevTools Protocol.
//
// Wave FIX1. `demo.ps1` launches the editor with `INF_WEBVIEW_DEBUG_PORT` set,
// which is the only way the embedder's WebView2 argument string can be reached
// for a window declared in `tauri.conf.json` — see
// `inf_studio_lib::debuggable_context` for why the environment variable WebView2
// documents does not work here.
//
//   node tools/demo/play.mjs [port] [waitSeconds] [embedded|window]
//
// Exit 0 = the button was found and clicked (the transport is then reported for
// `waitSeconds`); exit 2 = no page target; exit 3 = no button. `demo.ps1` falls
// back to a coordinate click on anything but 0.

const port = Number(process.argv[2] ?? 9222);
const waitS = Number(process.argv[3] ?? 20);
const mode = (process.argv[4] ?? "embedded").toLowerCase();
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
console.log(`target: ${page.title} ${page.url}`);

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
    return "EXC: " + JSON.stringify(r.result.exceptionDetails).slice(0, 300);
  }
  return r.result?.result?.value;
};

const CLUSTER = `(() => { const c = document.querySelector('[data-tour="play-cluster"]'); return c ? c.innerText.replace(/\\s+/g,' ').trim() : 'NO CLUSTER'; })()`;
console.log("title:", await evalJs("document.title"));
console.log("cluster before:", await evalJs(CLUSTER));

let clicked;
if (mode === "window") {
  // "Play in New Window" lives behind the cluster's own split-button dropdown,
  // so it takes two clicks and a paint between them. Found by its LABEL rather
  // than by position: the menu's contents differ between running and stopped.
  const opened = await evalJs(
    `(() => { const b = document.querySelector('[data-tour="play-cluster"] button[aria-label="Play options"]'); if (!b) return ''; b.click(); return 'opened'; })()`,
  );
  if (!opened) {
    console.log("NO PLAY OPTIONS button in the play cluster");
    ws.close();
    process.exit(3);
  }
  await sleep(400);
  clicked = await evalJs(
    `(() => { const items = Array.from(document.querySelectorAll('[data-tour="play-cluster"] button')); const b = items.find((e) => /New Window/i.test(e.innerText || '')); if (!b) return ''; b.click(); return b.innerText.trim(); })()`,
  );
} else {
  clicked = await evalJs(
    `(() => { const b = document.querySelector('[data-tour="play-cluster"] button'); if (!b) return ''; b.click(); return (b.getAttribute('aria-label') || b.title || b.innerText || 'button').trim(); })()`,
  );
}
if (!clicked) {
  console.log(`NO PLAY BUTTON for mode ${mode}`);
  ws.close();
  process.exit(3);
}
console.log(`clicked (${mode}): ${clicked}`);

// Report what the toolbar says while the player loads — a dialog here is the
// "this level has no player-controlled character" question, and a script that
// did not print it would look like a hang.
const t0 = Date.now();
let last = "";
while (Date.now() - t0 < waitS * 1000) {
  await sleep(2000);
  const now = await evalJs(CLUSTER);
  if (now !== last) {
    console.log(`[+${((Date.now() - t0) / 1000).toFixed(0)}s] cluster: ${now}`);
    last = now;
  }
  const dialog = await evalJs(
    `(() => { const d = document.querySelector('[role="dialog"]'); return d ? d.innerText.replace(/\\s+/g,' ').slice(0,300) : ''; })()`,
  );
  if (dialog) console.log("DIALOG:", dialog);
}
console.log("cluster final:", await evalJs(CLUSTER));
ws.close();
process.exit(0);
