# `tools/demo` — end a wave with something a person can judge

A wave that ends in a green battery has proved that the tests agree with the
code. It has not proved that the editor opens, that the Play button plays, or
that the character walks — and in wave FIX1 the author found all three false at
a head whose battery was green (`7 019 passed, 0 failed`).

So every wave from FIX1 onward ends here: the real editor, built the way it
ships, launched, driven through its own Play button, and photographed.

## Run it

```powershell
pwsh -NoProfile -File tools/demo/demo.ps1
```

That builds the editor (`npx tauri build --no-bundle`), launches
`target/release/inf-studio.exe` **from its own directory** — which is how the
boot ladder discovers the showcase island by walking up from the executable —
presses Play, waits for `inf-player.exe`, holds `W` on the running game, and
writes three PNGs plus a CSV of where the hero was.

Useful switches:

| | |
|---|---|
| `-SkipBuild` | use whatever is already in `target/release` |
| `-OutDir <path>` | where the PNGs and the CSV go (default: a timestamped folder under the system temp dir) |
| `-KeepOpen` | leave the editor running at the end |
| `-Port <n>` | the WebView2 debug port (default 9222) |
| `-PlayMode window` | drive "Play in New Window" instead of the embedded viewport. Needs the CDP path — a menu item has no coordinate to fall back to |

## What it produces

```
01-editor.png     the editor as it booted, on the showcase island
02-pie-a.png      the running game
03-pie-b.png      the same, two seconds later, with W held throughout
hero.csv          t,frame,x,y,z,mode,speed,camera_pull — four rows a second
demo.log          every step the driver took, with timings
```

and it prints the hero's first and last position and the distance between them.
**That distance is the point.** Two screenshots cannot tell a character that
walked from a camera that drifted; the CSV can.

`hero.csv` also carries the player's own `#` notes — its keyboard-focus report
and every focus handover, with the window and process that took it. That is not
decoration: the editor's Output Log is behind the game's window, so a scripted
session has nowhere else to read the player's stderr, and this file is how wave
FIX1 found out that its own synthetic click was what pushed the new-window
session out of the foreground.

## How it presses Play

By name, over the Chrome DevTools Protocol: `INF_WEBVIEW_DEBUG_PORT` makes the
editor's WebView2 listen, `play.mjs` finds `[data-tour="play-cluster"] button`
and clicks it. If node is not on the PATH or the port never opens, the driver
falls back to clicking the button's screen coordinate on a maximized 1080p
window and says so in the log.

Setting `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` yourself does **not** work and
the reason is in `inf_studio_lib::debuggable_context`: WebView2 reads that
variable only when the embedder passes no arguments of its own, and Tauri always
passes some.

## Before you run it

The driver refuses to build while an `inf-studio` or `inf-player` is running —
the island's pack is memory-mapped and a build that tries to replace a running
executable fails as a sharing violation, which surfaces as `LNK1104` and looks
like a disk problem. Close the editor first, or pass `-SkipBuild`.

## It is a gate, not a screenshot service (audit FIX1)

`demo.ps1` **exits non-zero (7)** when the hero moved less than `-MinMetres`
(default **5 m**), when the player wrote no positions, or when there is no
`hero.csv` at all. The first version printed `HERO MOVED` and exited 0 whatever
the number was — including the runs the wave later found had moved **0.000 m**,
which were caught by a person reading a log rather than by the gate whose whole
purpose is to catch them.

Five metres is the bound because a held `W` buys twelve in the seconds this
script allows, and because a settle, a slide or a camera drift is centimetres.

It also echoes the player's own `keyboard focus …` line beside the number, so a
session that moved is read next to the reason it could:

```
player: keyboard focus hwnd=0x250406 parent=0x0 fg=0x250406 focus 0x250406 ->
        0x250406 attached=false landed=true
```

`parent=0x0` there means the player's window was still **top-level** when it
asked — the editor had not reparented it yet — so the embedded child branch of
`win_host::take_keyboard_focus` did not run. In 28 recorded sessions across two
machines' worth of runs it has never run, and `attached` has never once been
`true`. Read that line before believing a story about which branch made a
session work.

## Exit codes

| code | meaning |
|---|---|
| 0 | the hero moved, the frames are in `-OutDir` |
| 2 | an `inf-studio` or `inf-player` was already running |
| 3 | a build failed, or there is no editor/player to run |
| 4 | the editor exited while we waited |
| 5 | no `inf-player` appeared within `-PieWaitS` |
| 6 | `-PlayMode window` without node on the PATH (a menu item has no coordinate) |
| 7 | **Play did not play**: the hero moved less than `-MinMetres` |
