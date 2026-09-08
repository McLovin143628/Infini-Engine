# Weapons and ballistics

A shot in this engine is a **hybrid**. Close in it is an instant ray, which is what keeps a
trigger feeling immediate; past a per-weapon threshold it is a **body in flight** with gravity,
drag, travel time and bullet drop.

```
Muzzle ──► [ 0 .. hitscan_threshold_m : one instant ray ] ──► [ beyond : a round flies ]
```

The switch is `WeaponDef::hitscan_threshold_m` and it is per weapon, because it is per class: a
pistol's is 30 m — its whole effective envelope, so a pistol is "pure hitscan" as a *consequence*
rather than as a second kind — an SMG's is 15 m, an assault rifle's 25 m, a sniper's 10 m, and a
rocket launcher's is **0**, so a rocket is a body from the muzzle. A `ShotKind::Hitscan` ignores
the field entirely and behaves exactly as it always has.

### What a round does

Each fixed step is split into four sub-steps. Per sub-step the velocity takes gravity and a
quadratic drag, the position follows, and the move is resolved as a **segment cast** from the
previous position to the next — never a test of the endpoint. That is not a refinement: a 900 m/s
round covers **15 metres** in one 60 Hz step and a wall is 20 cm thick, so a point move steps clean
over most of the geometry in the game.

| knob | unit | what it is |
|---|---|---|
| `muzzle_speed_mps` | m/s | the speed the round leaves at |
| `drag_k` | 1/m | the whole of `½ρC_dA/m` as one coefficient — `drag_k · v²` is m/s² |
| `gravity_scale` | — | ×9.81 m/s²; `0.0` is a flat trajectory |
| `range_m` | m | where the round dies if it has hit nothing |

A round also dies on a hit, on leaving the streamed partition, and at eight seconds. The pool is
bounded — 64 rounds, which is 256 segment casts a step — and a spawn past the bound is **refused
with a value** that the report and the pool both count, never a round silently dropped.

### Damage over distance

One curve, asked by both halves of the hybrid:

```
distance ≤ effective_range_m          →  the full damage
distance ≥ max_range_m                →  damage × min_damage_frac
between                               →  the Hermite smoothstep between the two
a head hit                            →  × headshot_mult
```

Damage is in **joules**, because everything in this engine that can hurt something is. A body has
2 000 J (`DEFAULT_VITALITY_J`) and the weapon registry is authored against a 100 HP body, so
**1 HP = 20 J** — one conversion, applied once when the rows are written, and nothing in the
engine reads a hit point.

A weapon that names no ranges has a **flat** curve, which is what every level authored before this
existed still gets.

### Headshots

The hit point is tested against the target's **head**, which is the rig's own `head` socket when
the character is posed and the top of its capsule when it is not. It is a **band** and not a
sphere, and the reason is worth knowing: a character's collider is a capsule about 0.30 m in
radius, so a ray stops on that surface — a small sphere around the head *joint* is a target no shot
could ever reach. The test is the height the round arrived at (±12 cm of the head) inside the
body's own radius.

There is no second collider and no hitbox rig. A headshot costs one comparison.

### The registry

Eighty-five weapons ship in `crates/inf-ecs/src/weapons.toml` — 10 pistols, 20 SMGs, 20 assault
rifles, 10 marksman rifles, 10 snipers, 10 shotguns and 5 launchers. A level defines them with one
Blueprint call:

```rust
item.define(WEAPON_REGISTRY_TOML)   // or your own TOML, in the same shape
```

A row is an item with a `[weapon]` table and four optional sub-tables:

```toml
[m4a1]
label = "M4A1"
mass_kg = 3.6
[m4a1.weapon]
kind = "projectile"
automatic = true
damage_j = 600.0
rounds_per_minute = 800.0
magazine = 30
range_m = 500.0
muzzle_speed_mps = 910.0
ads_time_ms = 240.0
move_speed_mult = 0.90
[m4a1.weapon.ballistics]
hitscan_threshold_m = 25.0
drag_k = 0.00035
gravity_scale = 1.0
[m4a1.weapon.damage_curve]
headshot_mult = 1.5
effective_range_m = 35.0
max_range_m = 80.5
min_damage_frac = 0.60
[m4a1.weapon.recoil]
recoil_intensity = 3.8
[m4a1.weapon.audio]
report_max_m = 320.0
```

Every key inside every sub-table goes through the same by-name door
(`WeaponDef::set` / `WeaponDef::names`), so a UI can enumerate the knobs rather than restate them,
and a **misspelled sub-table is refused by name** rather than skipped — a `[ballisitcs]` typo that
quietly dropped a muzzle velocity would fire the default at you.

`move_speed_mult` is real: it scales the character's own walk, run, sprint and crouch speeds while
the weapon is equipped, so a sniper rifle costs you 24 % of your speed. `report_max_m` is how far
the shot is heard. `recoil_intensity` and `ads_time_ms` are authored for the feel wave and nothing
reads them yet.

### What is not there yet

**Shotgun pellets** — a shotgun fires one ray carrying the whole shell's energy, not eight.
**Launcher blast** — a rocket spends its direct damage on what it hits and has no radius.
**Burst fire** — the engine has automatic and semi-automatic, so the three-round-burst weapons are
authored as automatic at their listed rate. **Attachments**, **recoil springs and sway**, the
**four-layer gunshot** and **shell casings** are all later waves of the same arc. And a round
**cannot hurt a vehicle**: a shot's cast sees static and kinematic geometry only, which is why a
hitscan could not either.
