// Frame a CAR — with the doors, bumpers and glass wave VEH3c gave it — in the
// editor viewport, over the Chrome DevTools Protocol.
//
//   node tools/demo/car.mjs [port] [--dist=9] [--lift=0.4] [--turn=62] [--open=<.inf_lvl>]
//
// Exit 0 = framed (the numbers are printed); 2 = no page target; 3 = the
// commands refused; 4 = the open document has no car with parts in it, which is
// itself the finding.
//
// # Why this exists, and why it moves the CAR rather than the camera
//
// `portrait.mjs`'s ruling, verbatim: there is no "put the camera here" IPC door
// and there should not be one just for this. `viewport_focus` frames the
// SELECTION off its render instances and was measured doing nothing at all for
// an object the projector had not seen. So the subject moves instead, onto the
// camera's own opening view ray — `EngineHost::player_start_pose`, eye at
// `p + 1.75·Y − flat·7` with a −0.12 rad pitch, which is arithmetic this script
// can do.
//
// The edit is to the open DOCUMENT only. Nothing is written to disk, exactly as
// `place.mjs`'s and `portrait.mjs`'s are not, and no caller of this presses
// Ctrl+S.
//
// # `--open`, and why it is not cheating
//
// The editor's boot ladder opens whatever SHOWCASE PROJECT it finds by walking
// up from the executable, and on a developer's machine that is a generated
// project whose copy of a level can be older than the tree — measured here: the
// local `island-build/project`'s island was **four days and one wave stale**, so
// its cars were the four-panel bodies and this script correctly exited 4.
// `--open` hands `scene_open` an explicit path so the frame is of the level the
// TREE generates rather than of whatever a developer's project happens to hold.
//
// # What it proves
//
// That the EDITOR draws the real parts. A door, a bumper and a pane of glass are
// `RigNode`s out of the one recipe both hosts walk, so they are in this frame or
// they are in neither host. The script finds its subject by looking for a part
// NAMED `door_fl` or `door_l` in the open document and framing its PARENT, so a
// level whose cars have no doors exits 4 rather than photographing a box.

const port = Number(process.argv[2] ?? 9222);
const arg = (k, d) =>
  Number(process.argv.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d);
const DIST_M = arg("dist", 7.0);
const LIFT_M = arg("lift", 0.1);
// **AND IT IS TURNED**, because a car photographed nose-away is a picture of a
// boot lid. Sixty-two degrees off the view direction is the three-quarter view
// every car ever photographed is in: a flank with its doors and its side glass,
// a nose with its bumper, and enough of the roof to read the greenhouse.
const TURN_DEG = arg("turn", 62.0);
const EYE_M = 1.75; // EngineHost::PLAYER_START_EYE_M
const BACK_M = 7.0; // EngineHost::PLAYER_START_BACK_M
const PITCH = -0.12; // EngineHost::PLAYER_START_PITCH, radians

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

const TRANSFORM = "inf_ecs::components::Transform";
const field = (details, name) =>
  details?.components
    ?.find((c) => c.type_path === TRANSFORM)
    ?.fields?.find((f) => f.name === name)?.value?.value;

// ── open an explicit level, if one was named ────────────────────────────────
const openPath = process.argv.find((a) => a.startsWith("--open="))?.slice(7);
if (openPath) {
  const opened = await invoke("scene_open", { path: openPath });
  if (!opened || typeof opened === "string") {
    console.log("OPEN REFUSED:", String(opened).slice(0, 300));
    process.exit(3);
  }
  console.log(`opened ${openPath}`);
  await sleep(4000);
}

// ── find a car that HAS doors ───────────────────────────────────────────────
const snap = await invoke("scene_snapshot", {});
if (!snap || typeof snap === "string") {
  console.log("NO SNAPSHOT:", snap);
  process.exit(3);
}
// `SceneSnapshot::nodes` is a FLAT list in no particular order, and every
// `SceneNode` carries its own `parent` — so there is nothing to walk.
const flat = snap.nodes ?? [];
console.log(`the open document has ${flat.length} entities`);
const door = flat.find((e) => e.name === "door_fl" || e.name === "door_l");
if (!door || !door.parent) {
  console.log(
    "NO CAR WITH DOORS in the open document — the level's vehicles are the " +
      "four-panel bodies that predate wave VEH3c, or nothing is open",
  );
  process.exit(4);
}
const chassis = door.parent;
const parts = flat.filter((e) => e.parent === chassis).map((e) => e.name);
console.log(`car ${chassis} has ${parts.length} drawn parts: ${parts.join(", ")}`);

// ── put it on the camera's own view ray ─────────────────────────────────────
const pawn = await invoke("scene_player_pawn", {});
let eye = null;
let fwd = null;
if (typeof pawn === "string" && pawn.length > 8 && !pawn.startsWith("EXC:")) {
  await invoke("scene_select", { guids: [pawn], additive: false });
  const d = await invoke("scene_details", {});
  const pos = field(d, "translation");
  const rot = field(d, "rotation");
  if (Array.isArray(pos) && Array.isArray(rot)) {
    const yaw = (rot[1] * Math.PI) / 180;
    const back = [Math.sin(yaw), 0, -Math.cos(yaw)];
    eye = [pos[0] - back[0] * BACK_M, pos[1] + EYE_M, pos[2] - back[2] * BACK_M];
    fwd = [
      Math.sin(yaw) * Math.cos(PITCH),
      Math.sin(PITCH),
      -Math.cos(yaw) * Math.cos(PITCH),
    ];
  }
}
if (!eye) {
  console.log("NO PLAYER START to take the camera's opening pose from");
  process.exit(3);
}
const at = [
  eye[0] + fwd[0] * DIST_M,
  eye[1] + fwd[1] * DIST_M + LIFT_M,
  eye[2] + fwd[2] * DIST_M,
];
console.log(
  "eye:",
  JSON.stringify(eye.map((v) => Number(v.toFixed(3)))),
  "car to:",
  JSON.stringify(at.map((v) => Number(v.toFixed(3)))),
);
const res = await invoke("scene_set_property", {
  guids: [chassis],
  typePath: TRANSFORM,
  field: "translation",
  value: { kind: "vec3", value: at },
});
if (typeof res === "string" && res.startsWith("EXC:")) {
  console.log("MOVE REFUSED:", res);
  process.exit(3);
}
const yawDeg = (Math.atan2(fwd[0], -fwd[2]) * 180) / Math.PI + TURN_DEG;
const spun = await invoke("scene_set_property", {
  guids: [chassis],
  typePath: TRANSFORM,
  field: "rotation",
  value: { kind: "vec3", value: [0, yawDeg, 0] },
});
if (typeof spun === "string" && spun.startsWith("EXC:")) {
  console.log("TURN REFUSED:", spun);
}
console.log(`turned to yaw ${yawDeg.toFixed(1)} deg (${TURN_DEG} off the view)`);
// Drop the selection: a selected car is drawn inside its own collision box, and
// this frame is a picture of the bodywork and not of a wire box.
await invoke("scene_select", { guids: [], additive: false });
await sleep(2500);
console.log(`CAR FRAMED at ${DIST_M} m with ${parts.length} parts`);
ws.close();
process.exit(0);
