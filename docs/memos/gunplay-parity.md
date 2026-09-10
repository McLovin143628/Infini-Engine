# Gunplay parity — the research doc, item by item, with its arm and its number

**Wave WPN2e, 2026-09-10.** The user's research document
(`docs/Weapons-in-Rust-based-Game-Engine-GTA-Game.md`, outside this repository)
asks for six things in its architecture section and prints five tables of rows
underneath them. This memo walks all of it. Every row carries **the arm that
proves it** — a file and a test function you can run — and **the number that arm
measured**. A row with no arm is not in the table; it is in *What parity is NOT*,
at the bottom, with its reason.

The rule this memo is written under: **a claim without an arm is a paragraph.**
`wpn2e_gate::every_row_of_the_parity_memo_cites_an_arm_that_exists` reads this
file, extracts every `file::function` citation, and looks each one up in the
tree — so a row that cites something that has been renamed or deleted reds a
test rather than quietly becoming untrue.

---

## §1 Hybrid ballistics: hitscan meets projectile physics

| the doc asks for | arm | measured |
|---|---|---|
| a **distance switch**: raycast near, spawn a projectile past a threshold | `wpn2a_gate::a_shot_past_the_threshold_becomes_a_body_in_flight` | a 900 m/s rifle's round is minted at **z 25.00 m** with its odometer already reading 25 m, and moves **14.9579 m** in the next 60 Hz step |
| **gravity and drag** each tick | `wpn2a_gate::a_rifle_hits_later_and_lower_at_two_hundred_metres_than_at_ten` | the drop at 200 m is **0.1989 m** against a dragless parabola's **0.1855 m** over the same time of flight; **12 steps** to 200 m. The WPN2a audit decomposed it: **0.19617 m** of ballistic drop plus **0.0027250 m** of the shooter's own one-step fall — the model is exact and the caption is 1.4 % generous. Drag itself costs **6.05 %** of muzzle speed over 200 m at the assault-rifle row's `drag_k` of **0.00035 /m** |
| **sub-stepped raycasting**, never `position += velocity * dt` | `wpn2a_gate::a_thin_wall_at_a_hundred_metres_is_not_tunnelled` | four segment casts a fixed step; a **0.2 m** wall at 100 m stops the round at **z 99.9** and nothing reaches 100.2 |
| a bound on the whole thing | `wpn2a_gate::eight_shooters_at_nine_hundred_rpm_engage_the_ceiling_and_the_refusal_is_a_value` | `MAX_SHOT_RAYS_PER_STEP` **256**, `MAX_ROUNDS_IN_FLIGHT` **derived** (256 / 4 = 64); at eight shooters, **126 minted, 239 refused and counted**, peak **63 of 64** in flight, peak **252 of 256** casts |
| …and the round dies | `wpn2a_gate::a_round_dies_at_its_range_and_the_pool_empties` | the pool empties; a round that outlives its range **and** its band dies of age and is counted |
| the cost | `wpn2a_gate::a_full_pool_costs_what_it_costs` | the wave measured the pool's own share at **0.0305 ms** at 63 rounds (**0.48 µs a round**, dev); its audit re-timed it at **0.0243 ms** / **0.39 µs a round**. Against `WEAPON_STEP_BUDGET_MS` **1.5 ms**, which is roughly sixty times the whole phase |

The one place this engine is stricter than the doc: the flight's integrator is
`+ - * /` and a single `sqrt`, with **no trigonometry, `powf` or `cbrt`**,
because a round's position reaches a determinism trace and `libm` is not
bit-portable across targets (the P14 law). `portable_character` scans it.

## §2 The recoil and procedural animation system — three layers

| the doc asks for | arm | measured |
|---|---|---|
| **layer 1**, a viewmodel spring back to rest | `wpn2b_gate::a_shot_kicks_the_hold_point_and_it_comes_home_without_overshooting` | a shot drives the weapon **104.47 mm** back into the shoulder, overshoots by **0.0000 %** and returns to **0.0000 mm** |
| **layer 2**, camera pitch/yaw with centre-recovery | `wpn2b_gate::a_five_round_burst_raises_the_aim_and_it_returns` | one round moves the aim **+3.4254°**, five move it **+7.9517°**, and it comes home to **0.000000°** in **0.833 s**. Per row: pistol **2.8736°/1.0715°**, AR **3.4254°/1.3810°**, sniper **6.0147°/2.8335°**. In the shipped host, at the weapon's own rate: **52.809 → 122.435 → 118.713 mm** of hold point and **2.3403 → 5.6528 → 5.3424°** of aim |
| **layer 3**, spread variance | `wpn2b_gate::the_thirty_round_pattern_orders_ads_then_hip_then_moving` | thirty rounds at 25 m: ADS + crouched + still **0.157 m**, hip + still **0.349 m**, hip + sprinting **0.895 m** (the audit's re-priced decay: **0.1739 / 0.3863 / 0.9893 m**). Every round adds `recoil_intensity × 0.06°` and the decay takes it back |
| the profile is per weapon | `wpn2b_gate::the_profile_scales_with_the_registrys_own_recoil_stat` | the doc's 1–10 `recoil_intensity` mapping, verbatim: a 2.5 pistol kicks **−0.0875 m** back and **0.0400 m** up, a 3.8 AR **−0.1070 / 0.0504**, a 9.9 sniper **−0.1985 / 0.0992** |
| **sway** from movement and breathing | `wpn2b_gate::the_sway_is_zero_at_rest_and_grows_with_speed` | **0.000000 mm** with no animation clock, **6.0000 mm** breathing and still, **91.4362 mm** at a 5.85 m/s sprint |
| ADS timing | `wpn2b_gate::the_aim_arrives_in_the_weapons_own_ads_time` | rifle **250.0 ms** for 240 asked, pistol **133.3** for 140 — both inside one frame; the field 70 → 55.2°, the boom 3.180 → 2.311 m |
| the cost | `wpn2b_gate::eight_shooters_bursting_cost_what_they_cost` | **0.0133 ms** against **0.0046 ms** with nobody armed — **0.970 µs a shooter** |

**Two pieces of the doc's own arithmetic were wrong and are corrected here**, with
the mutation living permanently in the gate:

* the doc calls its spring *"critically damped"* and prints `k = 220, c = 18`.
  Critical damping of a unit mass is `2√k` = **29.6648**. At the doc's 18 the
  60 Hz recurrence overshoots its rest position by **6.328 %**;
  `wpn2b_gate::the_docs_own_damping_would_overshoot_this_arms_ceiling` is that
  measurement. The stiffness ships; the damping is derived from it;
* `v0 = peak·√k·e` is **42 % wrong** at 60 Hz — a 0.05 m peak measures 0.029104,
  because the first semi-implicit update spends half the impulse before the
  position moves. `discrete_peak_gain` measures the gain from the recurrence that
  will actually run, exact to **0.0000 %**.

**A CAMERA KICK IS REFUSED, BY RULING.** The doc's layer 2 asks for camera pitch;
this engine gives an **aim** impulse in the look integrator instead, and the
camera follows the aim. `wpn2b_gate::the_camera_has_no_per_shot_input` greps
`camera.rs` and `camera.toml` for one and finds none, and
`wpn2b_gate::the_reticle_stays_on_the_aim_line_through_a_burst` is why: a camera
kick that does not move the aim makes the reticle **lie** (measured at
**0.0384°** off at rest, **2.83°** at its worst mid-burst, against a 120°/s
mouse flick's **9.74°**), and one that does move the aim is a camera→sim write,
which `d3::camera` Ruling 4 forbids.

## §3 Shell casing ejection and particle effects

| the doc asks for | arm | measured |
|---|---|---|
| a casing at the ejection port with an impulse and a spin | `wpn2c_gate::a_casing_is_ejected_bounces_once_and_settles` | gravity, **one bounce at e = 0.3**, settle on the second contact, an eight-second life |
| **pooling**, no per-shell allocation | `wpn2c_gate::the_casing_pool_is_bounded_and_folds_nothing_when_it_is_empty` | a ring of **128**; one rifle at 600 rpm keeps **69** on the ground and cannot fill it, so the arm that proves the bound is four rifles — **688 ejected, 560 recycled, peak 128** |
| a **bounce sound**, pitch-randomised | `wpn2c_gate::every_clip_the_engine_names_is_a_file_that_decodes` | one metal one-shot per FIRST contact, pitch **1 ± 0.25** from the counter hash — 128 casings, 128 distinct notes. The port is `EJECT_RISE_DEG` **25°** with a per-weapon azimuth; measured, the case leaves **1.4339 m** to the right of the receiver |
| the casings are **drawn** | `wpn2c_gate::every_live_casing_is_an_instance_the_renderer_draws` | every live casing is an entity and every dead one is not |
| the cost | `wpn2c_gate::the_cost_of_eight_shooters_is_measured_and_printed` | **174.3 µs** for the whole gameplay phase, **66.3 casing rays a step**, peak 128 |

**One case per PULL, not per pellet** — a shotgun throws eight pellets out of one
shell (`wpn2d_gate::a_pull_throws_its_pellets_and_each_carries_its_share`).

## §4 Audio engineering and spatialization — four layers

| the doc asks for | arm | measured |
|---|---|---|
| **four layers**: transient, body, room tail, distant crack | `wpn2c_gate::every_loud_shot_is_four_plays_on_four_salted_keys_in_a_pinned_order` | **30 rounds → 120 layer commands**, in the pinned order, on four keys, naming four clips |
| the layers do not share a reach | `wpn2c_gate::the_distant_layer_is_audible_at_the_range_it_is_written_for` | volumes 0.85 / 0.90 / 0.70 (0.80 indoors) / 0.90; the transient carries **12 %** of the weapon's report range, the indoor tail **25 %** at a 3 500 Hz cutoff, the body and outdoor tail all of it, and the distant layer **four times** it at 700 Hz with its near field at **150 m**. Measured on the M4A1 (body 320 m, distant 1 280 m), the spatial gain at 300 / 400 / 1 000 m is body **0.0100 / 0.0000 / 0.0000** and distant **0.5000 / 0.3750 / 0.1500** — before the WPN2c audit's `DISTANT_REACH_MULT` the distant layer was **silent at the four hundred metres its own doc names** |
| **real-time occlusion / indoor-vs-outdoor** | `wpn2c_gate::the_probe_calls_a_room_a_room_and_a_street_a_street` | six rays from the muzzle, indoors at four hits inside 8 m: **1 hit** on open ground, **6** in a closed room, a street between two buildings **3** |
| …and it costs | `wpn2c_gate::the_enclosure_probe_pays_for_its_own_rays` | a loud shot costs **seven** casts (1 shot + 6 probe); at eight shooters the probe alone is **6.88 rays a step** — **3.1 %** of the ray traffic and **2.7 %** of the ceiling — and the whole bill is **222.2 rays a step against 256** |
| **the low-pass is audible** | `wpn2c_gate::the_report_carries_its_cutoffs_all_the_way_to_the_command` | the textbook RC discretisation is **−15.9 %** wrong at 1 500 Hz; the exact root of `a² + 2aC − 2C = 0` is **0.00 % at every cutoff, −3.010 dB at each**, by a DFT of the filter's own impulse response |
| **supersonic cracks** past 343 m/s | `wpn2c_gate::a_supersonic_round_cracks_within_four_metres_and_a_subsonic_one_never_does` | at the M4A1's own registry muzzle speed of **910 m/s** a 2 m pass cracks at **2.27 m**, measured at the closest point on the round's SEGMENT rather than at a position (at 900 m/s a round crosses 3.75 m in a sub-step and 15 m in a fixed step, so a point test never lands inside four metres of an ear); 10 m does not crack, and the AS VAL at **330 m/s** never does — by physics, not by a flag |
| a log a firefight cannot silently evict | `wpn2c_gate::the_audio_log_holds_a_hundred_and_twenty_seconds_of_eight_shooters` | on the shipped host, **8 232 rounds, 41 115 commands, 0 dropped** over two minutes against a ring of **65 536** — a **191 s** horizon, where the old 8 192 evicted after **17.8 s**. `size_of::<AudioCommand>()` is **160 B**, so the ring is exactly **10.0 MiB** |

**The clips are arithmetic, and that is a limitation as much as a feature.** All
**36** are generated by `inf_audio::synth` with no `sin`, `exp`, `powf` or even
`sqrt` on the sample path (a parabolic sine, a one-pole decay, counter-based
splitmix noise),
because they are committed bytes and a generator leaning on the host's libm
writes different files on two machines. They are **773 KiB** and they are
*synthetic*: envelopes measured off the samples (transient ≤ 5 ms, body
0.20–0.55 s with its fundamental inside the doc's 40–80 Hz band — 78.0 authored /
**77.8 measured** for a pistol, 40.0 / **40.2** for a launcher; peak exactly
**0.900** on every clip, read back with an independent RIFF parser). A recorded
pack is in *What parity is NOT*, and its shape is named there.

## §5 The class table, per class

The doc prints one row per class. Each is a real, distinct set of numbers in
`crates/inf-ecs/src/weapons.toml`, read through one by-name door.

| class | ballistics | arm | measured |
|---|---|---|---|
| **pistols** | pure hitscan under 30 m | `wpn2a_gate::the_registrys_class_census_is_read_off_the_rows` | the doc's *"pure hitscan (< 30 m)"* is the row's own threshold; the census is read off the rows and reports **0 class-rule mismatches** |
| **SMGs** | short threshold, fast decay | `wpn2a_gate::the_registrys_class_census_is_read_off_the_rows` | 20 rows, and the whole seven-column class-rule table is compared against the doc row by row |
| **assault rifles** | hybrid, 700–900 m/s past the threshold | `wpn2a_gate::a_shot_past_the_threshold_becomes_a_body_in_flight` | minted at **z 25.00** at the row's own muzzle speed |
| **DMRs / snipers** | projectile with drop | `wpn2a_gate::a_rifle_hits_later_and_lower_at_two_hundred_metres_than_at_ten` | **0.1989 m** of drop at 200 m |
| **shotguns** | **multi-raycast cone spread** | `wpn2d_gate::a_pull_throws_its_pellets_and_each_carries_its_share` | the Remington's **2 880 J a pull** over **eight pellets** is **360.0 J each = 18 HP**, eight hits, and exactly **one** of them `loud`. The cone is per load: **8 pellets 4.00°, 10 pellets 4.50°, 12 (the duckbill row) 5.50°** |
| …and the cone is measured | `wpn2d_gate::the_pattern_at_ten_metres_is_the_cone_the_row_says_it_is` | at 9.80 m the eight impacts sit inside **0.331 m** of their centroid (mean 0.205) where a 4.00° cone allows **0.342 m** |
| **rocket launchers** | a physical entity with **accelerating engine force** | `wpn2d_gate::a_rocket_accelerates_from_its_muzzle_speed_to_its_burnout` | a launcher's `hitscan_threshold_m` is **0**, so a rocket is a body from the muzzle: launched at **115.0 m/s**, **132.7** ten steps later, peaking at **289.1** against its row's 300 m/s burnout |
| …and its blast | `wpn2d_gate::the_blast_falls_off_by_the_closed_form_at_five_distances` | `(1 − d/r)²`: **1.000000, 0.562500, 0.250000, 0.062500, 0.000000**, all to **1e-12**, and the joules five bodies lost match to **1e-6** |
| …and it is shadowed | `wpn2d_gate::a_wall_shadows_a_body_from_a_blast` | two bodies at 5 m, one behind a 0.5 m slab: the exposed one loses **340.3 J**, the shadowed one **0.0** |
| **throwables** | a parabola with drag and bounce | `wpn2d_gate::a_grenade_follows_the_parabola_and_bounces` | the flight against a closed form, and the fuse goes off **where the grenade is** |
| **melee** | a short box/sphere cast, surface-specific impact audio | `wpn2d_gate::the_m9_is_a_melee_item_with_its_own_reach_and_its_blow_names_a_surface` + `wpn2d_gate::a_punch_cannot_reach_through_a_wall` | the M9 reaches **2.00 m over an 80° arc for 900 J** against a pair of fists' **1.20 m and 150 J**; the blow names a surface; a wall stops it at the cost of **one** box cast per swing that found a body and **zero** for one that did not |
| **lock-on** (the doc's launcher row, taken further) | `wpn2d_gate::a_lock_holds_for_lock_s_and_releases_outside_the_cone` | the Javelin holds **1.600 s** of its row's 1.6 inside an **8–10°** cone and **releases to nothing** the instant the target leaves; the Stinger's `lock_air_only` is ALTITUDE (`MIN_AIRBORNE_M` **6 m**), because this engine has no is-flying flag |

## §6 Attachments

| the doc asks for | arm | measured |
|---|---|---|
| the catalogue | `wpn2d_gate::the_attachment_catalogue_is_the_docs_own_lists` | **530 rows** over **10 rail slots** — 80 each for pistol / smg / ar / dmr / sniper / shotgun and 50 for launcher. The doc's header promises 600; its AR list names a `Lasers` slot it never fills, so **530 is the honest total** and the arm counts the doc rather than the header |
| one fold, applied through one door | `wpn2d_gate::the_fold_is_applied_through_the_one_weapon_def_door` | one `apply`, one `fold` — a source arm counts the spellings |
| a **suppressor** that is quieter | `wpn2d_gate::a_suppressor_is_quieter_in_the_command_and_not_in_a_table` | on the M4A1: report reach **320.0 → 112.0 m**, body volume **0.900 → 0.315**, report gain **1.0000 → 0.3500**, transient reach **38.4 → 13.4 m**, range **500 → 460 m**, ADS **240 → 280 ms**. The multiplier reaches the `AudioCommand`, which is where a table becomes an effect |
| an **extended magazine** | `wpn2d_gate::an_extended_magazine_changes_the_readout` | the readout goes **`30 / 120` → `45 / 120`** |
| a **scope** | `wpn2d_gate::a_scope_changes_the_ads_time_which_is_the_cameras_own_blend` | a 4× ACOG takes ADS **240 → 288 ms**, which is the camera rig's own `state_blend_speed` **14.6271 → 12.3337 /s** |
| a bench a player uses | `wpn2d_gate::the_bench_round_trips_through_the_panels_own_verb` | through the panel's own verb, not a test-only door |

A fitted part costs the trace **57 B bare → 77 B with a can on it**, and the
bytes are identical again when it is stripped.

**44 of the 530 rows (8.3 %) still do nothing** — every multiplier 1.0, no ADS
delta, no magazine override, no art — and they are pinned rather than fixed (the
WPN2d audit's carried item 264, and its 257: nothing pins the keyword rule the
numbers are derived by). That is in *What parity is NOT*.

## §7 Damage curves and the registry

| the doc asks for | arm | measured |
|---|---|---|
| a **multi-segment falloff** | `wpn2a_gate::the_damage_curve_is_the_docs_own_at_five_distances_in_the_world` | the doc §5's Hermite smoothstep, measured in the world against a closed form written out by hand: **35 m 600.000 J** (600), **85 m 562.904** (562.5), **135 m 480.540** (480), **185 m 397.892** (397.5), **235 m 360.001** (360). The sub-joule gaps are not slop: back-solving gives **84.700 / 134.700 / 184.710 m** — exactly **0.300 m** short each time, which is the ray stopping on the target capsule's near surface |
| a **headshot multiplier** | `wpn2a_gate::a_shot_at_the_head_multiplies_and_one_at_the_pelvis_does_not` | the head at **1.500 m** spends 600 J and the pelvis at **0.600 m** spends 300 |
| …and it works past the threshold too | `wpn2a_gate::a_headshot_past_the_threshold_multiplies_as_well` | the same door, asked by both halves of the hybrid — a round that flew 120 m and arrived at the head spent **600 J** |
| …and it is a BAND, because a character is a capsule | `wpn2a_gate::a_shorter_bodys_head_band_is_lower` | a shot aimed exactly at a head socket 12 m away arrives **0.300 m** from it — a capsule radius — so a 12 cm sphere about the head point was a target **no shot in this game could hit**. `HEAD_RADIUS_M` is 12 cm *plus the body's own collider radius*, and a 1.40 m body's band derives to **1.100 m** |
| **the doc's 85 rows** | `wpn2a_gate::the_registry_is_the_docs_own_eighty_five_rows` | **85 of the doc's own**: 10 pistols, 20 SMGs, 20 assault rifles, 10 DMRs, 10 snipers, 10 shotguns, 5 launchers, checked against the document over nine columns with **0 mismatches**. `weapons.toml` holds **88** today: wave WPN2d added `m9_knife` and `thrown_knife` (classed `pistol`) and `g67_grenade` (classed `launcher`), which the doc's tables do not have and its class table's *Throwables* and *Melee* rows do. A LEVEL's catalogue is **91** — those 88 plus the `phase30-gameplay` fixture's own rifle, pistol and bandage, which every arm written before this arc picks up and fires. |
| …and a typo is refused by name | `wpn2a_gate::a_misspelled_sub_table_is_refused_by_name` | the four sub-tables (`[ballistics]`, `[damage_curve]`, `[recoil]`, `[audio]`) go through the same by-name `set` door their parent does, and a `[ballisitcs]` typo that would have silently dropped a muzzle velocity is refused **by name** |
| the joule scale is forced, not chosen | `wpn2a_gate::the_damage_curve_is_the_docs_own_at_five_distances_in_the_world` | `DEFAULT_VITALITY_J` **2 000 J** is the doc's 100 HP body, so **1 HP = 20 J**, which reproduces island wave I6's own 600 J pistol from the doc's 30 HP Glock |

## §8 …and what the doc does not ask for: the police

The research doc is about the weapon in the player's hands. Wave WPN2e is the
other half of a firefight.

| what | arm | measured |
|---|---|---|
| the response **ladder as behaviour** | `wpn2e_gate::the_ladder_is_three_behaviours_on_one_street` | patrol **181 aimed / 0 triggers / 0 rounds**; multi-unit **0 triggers** until fired upon, then **33**; SWAT **33 triggers / 2 rounds** |
| **the police do not cheat** | `wpn2e_gate::a_unit_with_no_sight_and_no_trail_never_aims` | clear line **30 rays / 0 blocked / 30 aimed**; a wall between **30 / 30 / 0**; a trail 240 steps cold **0 rays / 0 aimed** |
| a crew is **issued** a weapon on arrival | `wpn2e_gate::an_arriving_police_crew_is_issued_a_weapon_and_a_cold_town_issues_none` | `Cold` none, `Patrol`/`MultiUnit` `glock_17`, `Swat` `m4a1`; a fire crew none; arriving twice does not refill a magazine |
| **friendly fire discipline** | `wpn2e_gate::a_unit_behind_another_unit_holds_its_fire` | near officer **2 rounds**, far officer **0**, **181 friendly holds** |
| **a civilian in the line** | `wpn2e_gate::a_civilian_in_the_line_stops_the_shot` | in the line **181 holds / 0 rounds**; six metres aside **0 holds / 2 rounds**; the pedestrian is never hit |
| a minted budget that is nobody else's | `wpn2e_gate::a_unit_that_is_not_engaged_runs_no_engage_rays` | `NPC_ENGAGE_RAYS_PER_STEP` **16**; **0** rays with nobody wanted, **4** with one file, **16 of 16** at twenty officers |
| **cover, and the cadence** | `wpn2e_gate::an_officer_under_fire_takes_cover_and_fires_from_it` | 1 152 steps: **242 in cover**, **133 leaned out** (55 % duty), **32 rounds** |
| **blind fire** | `wpn2e_gate::a_blind_round_leaves_along_the_covers_own_normal_and_clears_the_wall` | the round leaves **`BLIND_CLEAR_M` above the parapet's own measured top**, along the cover's outward normal, inside a cone widened by **9°**, and never hits its own cover |
| …by an NPC too | `wpn2e_gate::an_npc_fires_blind_from_cover_too` | **238 blind decisions, 55 blind rounds**, the authored one-shot reaching its full **0.700 s** |
| a **wounding is its own crime** | `wpn2e_gate::a_non_fatal_round_raises_an_act_of_its_own` | at a wall `[Shot]`, heat **2**; at a person `[Wounded, Shot]`, heat **4** |
| a killing still names the shooter | `wpn2e_gate::a_killing_is_filed_against_the_shooter_and_a_corpse_is_not_also_wounded` | 1 kill filed against the hero; **60** further rounds reached the body on the floor and raised **0** woundings |
| the escalation is real | `wpn2e_gate::a_hit_officer_staggers_a_civilian_flees_and_the_heat_rises` | heat **2 → 60**, stars **1 → 5**, 3 staggers, a civilian flees, the officer loses joules |
| the hero is shot back at | `wpn2e_gate::the_hero_under_fire_takes_the_lazy_health_path_and_the_hud_reacts` | the hero carries **no `Health` component at all** until the first round lands |
| **PIE == shipping == two cooks** | `wpn2e_gate::pie_equals_shipping_and_two_cooks_agree_over_the_shootout_course` | 420 traced steps, byte-identical on three hosts; hero **30** rounds, officers **14**, **331** engage rays, **268** refused shots |
| the cost | `wpn2e_gate::a_firefight_is_inside_the_weapon_budget` + `wpn2e_gate::the_firing_policy_is_inside_the_npc_budget_at_a_thousand_agents` | a nine-shooter firefight **126.9 µs** against 1.5 ms (the audit re-timed it at **101.6 µs**); the policy at **1 000 agents 84.0 µs** against 1.0 ms (audit: **70.3 µs**) |
| a gunshot is **HEARD**, not only seen | `wpn2e_gate::a_gunshot_nobody_saw_is_still_heard_and_opens_a_file` | a listener 200 m away behind a wall, outside the 120 m witness radius: **0 acts seen, 40 heard**, heat 39, and the file carries **0 description channels**. `weapon::audible_radius_m` is the report's own range (250 m) through the enclosure verdict; **0 extra rays** — 280 shot rays with the listener and 280 without |
| …and a shot fired INSIDE is muffled | `wpn2e_gate::a_shot_fired_indoors_is_heard_a_quarter_as_far` | 250.0 m outdoors against **62.5 m** indoors (`INDOOR_TAIL_REACH_FRACTION`), measured on a room built out of slabs the enclosure probe calls a room |
| hearing never moves `last_seen` | the same arm | the shooter walks 20 m and fires again and the place the police are searching **does not move** — hearing gets them into the street, sight is what keeps them on you |
| an officer **reloads** | `wpn2e_gate::an_officer_whose_magazine_empties_reloads_and_goes_on_firing` | **37 rounds out of a 17-round magazine** over 60 s, 2 reloads, reserve 68 → 34 |
| …and a cold town takes the weapons back | `wpn2e_gate::a_town_that_goes_cold_takes_the_weapons_back` | the officer carries `glock_17` while wanted and `None` once cold; the weapon entity is gone and the item is still in the inventory |
| the trigger comes down when the policy stops looking | `wpn2e_gate::an_officer_the_policy_stops_visiting_stops_firing` | caught with the trigger DOWN, then a wall goes up: **1 round then, 0 after, 1 release** |
| an officer is not fired upon by its own weapon | `wpn2e_gate::an_officer_that_fires_is_not_fired_upon_by_its_own_weapon` | 7 rounds fired, **0 steps under fire, 0 steps in cover** on a street where the only gunfire is its own |
| a rig that authors no sockets gets them at LOAD | `wpn2e_gate::the_islands_own_hero_carries_its_weapon_in_its_hand` | 104 island `.inf_skel` decoded, 92 with a `hand_r` joint, **92 publishing a `hand_r` socket**; the weapon entity **0.387 m from the character origin and 0.000 m from the hand** |
| a dispatched unit reaches the scene | `wpn2e_gate::a_dispatched_unit_reaches_a_warm_file_on_the_island` | on the shipped island, **12 m at 62.6 s** against an `ON_SCENE_M` of 12 — it was 170 m and never |

---

## Against the reference frames

`docs/reference_videos/frames/` holds two clips of the game this campaign is
measured against (local only — nothing from them is committed). Read against
this wave's own shootout frames:

**`steal-car/`, frames 0010–0035 — the wanted reaction chain.** The reference
shows: a crime, a HUD line, stars appearing, a siren, a cruiser arriving, an
officer getting out, the officer drawing. **Every link of that chain now exists
here and is measured**: the act is witnessed by whoever could see it
(`d3::gameplay::step_witness`), it opens a file keyed on a *description* rather
than on a name (`crime::report_act`), the stars are `crime::wanted_readout` and
they move (**1 → 5** in the escalation arm), the dispatcher sends the nearest free
unit by route cost, the crew gets out at the scene, and — new in this wave — it
is **issued a weapon and points it at you**. The link the reference has and this
engine does not is the HUD *text*: there is no "Crime reported" banner, only the
star readout.

What does not match is **art and effects, not behaviour**: the reference's
officer has a drawing animation and a holster; ours has a rifle that appears in
its hand on the step it arrives. The reference's shot has a muzzle flash and an
impact puff; ours has a sound and a hole. Both are named below.

**…and the SHIPPED island's chain stopped twice, and both stops are closed** (the
WPN2e audit). `wpn2e_gate::the_islands_own_chain_from_a_gunshot_to_an_engaged_officer`
measures the whole thing:

| link | wave WPN2e | after the audit |
|---|---|---|
| acts recorded at the spawn | 17 | 17 |
| …with an OBSERVER | **0** (nearest crowd agent 117 m) | 0 |
| …with a HEARER | — | **16** |
| files open | **0**, heat 0, rung `cold` | **1**, heat 12, rung `swat` |
| assignments over three minutes | 3 | **8** |
| units on scene | **0** | **2** |
| crews carrying a weapon | 0 | **1** |
| units **ENGAGED** | **0** | **1** |
| nearest responder | **93 m** | **2 m** |

Neither stop was a level-design fact. The first was a missing **channel** — you
do not have to see a gunshot — and the second was a cruiser wedged in the
geometry of the station block the PCG parked it in, at full throttle, doing
0.00 m/s (`inf_ecs::dispatch::ESCORT_LAG_M` carries the trace).

**`police-bike/` — the brawl.** The reference shows an officer closing to melee,
a struggle, and bystanders scattering. Here: the melee door exists and is
measured (reach, cone, an LOS box cast, a surface-named impact), the scatter
exists and is measured (`step_panic`, and the officers exempt from it), and the
**struggle does not** — there is no grapple, no takedown and no arrest animation.
An officer that reaches you stands at `SECURE_S` and secures the scene. The
arrest verb is EMS2's and it resolves an incident rather than putting handcuffs
on anybody.

---

## What parity is NOT yet, said plainly

Every one of these is a real gap, and each names the wave that owns it.

* **No particles. → PAR2.** No muzzle flash, no impact dust, no backblast, no
  smoke, and **no impact decal of any kind** — thirty rounds into a wall leave a
  wall. This engine has had no particle system since P22 recorded it as a
  remainder, and it is the single biggest visual difference between a firefight
  here and one in the reference clips. What ships instead is the first 2 % of the
  tracer's own `debug.line`, drawn brighter, and a 1.80 s generated `Blast`
  clip for a launcher. A muzzle-flash *light* is separately refused with a
  number: `MAX_LIGHTS` is **16**, the sun is pushed as `lights[0]` so the real
  budget is **15**, and the projection is first-N with no sort and no priority —
  so a flash on a lit street either vanishes or pushes an authored light out and
  nothing reports either. **PAR0**'s.
* **No recorded audio. → the user's pack.** All 36 clips are generated
  waveforms, **773 KiB** of them. The four-layer stack, the reaches, the tails,
  the occlusion filter and the crack are all real and all measured; what is
  playing through them is a parabolic sine and a splitmix noise, and what it
  cannot give is the transient. The door is `AudioAsset::from_encoded` against
  the same 36 GUIDs, so dropping a recorded library in is a content change and
  needs no code; the shape to buy is **five clips per class** (transient / body /
  indoor tail / outdoor tail / crack) plus one metallic casing drop. Two
  narrower gaps ride with it: a casing has **no per-calibre clip and no
  per-calibre mesh** (a 12-gauge hull and a 9 mm case make the same noise at a
  different pitch — carried 237 / 245), and there is **no reverb bus** — `Effect`
  has two variants and this arc implemented the one that had been declared and
  never applied.
* **Cars take no damage. → VEH3c.** A round stops in a chassis (measured, 600 J
  at the near face) and the chassis spends nothing, because it carries no
  `Health` and no `Destructible`.
  `wpn2e_gate::a_car_shot_at_stops_the_round_and_spends_nothing` asserts that
  *absence*, so the day VEH3c gives a chassis health this memo's row goes red.
* **Fourteen rows draw a primitive** — **9 shotguns and 5 launchers** — because
  no imported mesh fits them. The Lyra `SK_Shotgun`/`SK_Pistol` skeletal meshes
  are the route, and the WPN2d audit's judgement is that it is *"one export
  selector and one rebind"* rather than its own wave (carried 248). Two smaller
  ones ride with it: `SM_KA74U` is imported and **0 registry rows reach it** — 4
  MB of local content nothing draws (carried 249) — and a **magazine is not
  drawn** at all, because `step_accessories` has no `Magazine` seat in its
  offsets (carried 250).
* **44 of 530 attachment rows do nothing** (WPN2d audit, carried 264) and
  nothing pins the keyword rule the other 486 derive their numbers from (carried
  257). The five behaviours are measured above; the rest are catalogue entries.
* **No first-person arms.** The first-person seat is at the head and the weapon
  hangs at the hand socket, 63° below the horizon — outside any sane frustum.
  The fade rule is measured and correct; what is missing is an arms-and-weapon
  rig, which is what an FPS draws (WPN2d audit, carried 268).
* **No burst fire.** The engine has `automatic` and semi-automatic and no burst,
  so the doc's three-round-burst rows — FAMAS F1, Type 95, CZ 75 Auto, TEC-9 —
  are authored automatic at their listed RPM.
* **No cleave**, decided rather than deferred (WPN2d carried 252): a swing
  resolves against one body.
* **No squad.** Two officers at one scene each decide alone. The only thing they
  share is a cone test that stops one shooting the other.
* ~~**No reload in the firing policy.**~~ **CLOSED by the WPN2e audit** —
  `gameplay::npc_press_reload` and the policy's own `needs_a_reload`. Measured:
  37 rounds out of a 17-round magazine.
* ~~**A weapon issued at arrival is never taken back.**~~ **CLOSED by the WPN2e
  audit** — `gameplay::unequip_weapon`, called when the town goes cold.
* **No search pattern.** When a suspect breaks line of sight the officer's aim
  stays on the place the gunfire came from. `last_seen` is EMS3's and moving a
  unit along it is a wave of its own.
* **A unit with no line of sight and NO COVER still does nothing** (carried 278,
  judged and not closed by the WPN2e audit). Blind fire is available from cover
  and only from cover, which is right — a unit standing in the open with a wall
  between it and a suspect is not suppressing anything, it is shooting a wall —
  so what is missing is not a shot but a MOVE. `step_npc_cover` walks it to cover
  when it finds any, and COV1's carried 193 is the case where it will not.
* ~~**The island's own hero carries its weapon in its pelvis**~~ **CLOSED by the
  WPN2e audit, and not by the re-import the carry prescribed.** Wave WPN2d fixed
  the two IMPORTERS, which touches a rig somebody re-imports and leaves every
  `.inf_skel` already on disk exactly as broken — measured, **0 of the island's
  104 rigs published a socket table**. `SkeletonAsset::migrate` derives one at
  DECODE, in the one door both hosts read a rig through, and only for a rig that
  authors none.
