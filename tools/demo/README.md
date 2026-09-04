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
