# Driving parity — the research doc, item by item, with its arm and its number

**Wave VEH3h, 2026-09-26 — the cert of the VEH3 arc (VEH3a..g, VEH3f.2a, VEH3f.2b).**
The user's research document (`docs/GTA-6-and-Forza-Style-Driving-in-a-rust-based-Game-Engine.md`,
outside this repository) closes on a five-item *Parity Checklist*; its body asks for more
than the checklist names (Pacejka, multi-contact, heat, differentials, weight transfer,
the boarding pipeline, granular audio, class profiles, a 150-row roster, air and sea).
This memo walks all of it. Every row carries **the arm that proves it** — a test file and
a function you can run — **what that arm reads**, and **the number it prints today**, re-run
by this wave on the tree it ships. A row with no arm is not a pass; it is in *What parity
is NOT yet* with the wave that closes it.

The rule this memo is written under: **a claim without an arm is a paragraph, and a cert
is a set of citations the auditor re-runs.**
`veh3h_gate::every_arm_the_driving_memo_cites_exists_and_is_not_ignored` reads this file
(CR-stripped), extracts every `` `file::function` `` citation and looks each one up in the
tree — a renamed, deleted or `#[ignore]`d arm reds a test. Three of the memo's numbers are
re-derived by the gate from their own arms and compared to the digit:
`veh3h_gate::three_classes_against_their_forza_inspirations` (the `FEEL-VS-FORZA` lines)
and `veh3h_gate::the_lap_logs_at_sixty_hertz_and_pie_equals_shipping` (the `LAP-FACTS`
line).

Verdicts: **✓** held, measured today, nothing carried under it; **PARTIAL** the arm holds
and the claim is narrower than the doc's; **✗** not built, or measured failing — each with
the wave that closes it by name. No row is ✓ on a carried item.

Re-run everything below with `cargo test -p inf-player --test veh3<x>_gate -- --nocapture`
(dev builds are optimised here; clocks are quoted from `--release` runs, off CI, min of
five, as the house conditioning asserts them).

---

## §0 The verdict, counted

| | rows |
|---|---|
| ✓ | **103** |
| PARTIAL | **12** |
| ✗ | **9** |

(The cert counted 99 / 13 / 10. The VEH3h audit CLOSED two ✗ — the parked vehicle beyond
the collider band and the Demon row — added two ✓ rows for the stall it diagnosed (a fixed
step beside every body kind; a crowd pile set aside), and moved the parked-car cost row from
PARTIAL to ✗: conditioned, its ceiling is not held.)

The ✗ rows and where they go: the island's frame (**PERF1**), the lit driving frame with
64 shells (**PERF1**), the parked-car cost ceiling (**PERF1**), headlight cones (**PAR1** on
**PAR0**), tyre smoke / exhaust / dust (**PAR2**), clearcoat paint (**PAR** arc, ~2.5 d),
the per-sample granular synth (VEH3-carried, ~1.5 d), the art-machine cab doors
(VEH3-carried, ~1 d a machine), the template rig's knee and elbow limits (VEH3-carried, a
rig + clip re-bless, ~1 d). Counted over the five-column checklist tables of §1-§6; the
tables after them are measurements, not rows.

---

## §1 The doc's checklist, item 1 — "Physics thread (300 Hz+): raycast/shapecast tyre physics using the Pacejka formula outside the rendering frame rate"

| the doc asks for | arm | what it READS | measured today | verdict |
|---|---|---|---|---|
| **sub-stepping at 300-400 Hz** | `veh3a_gate::the_substep_loop_runs_and_one_is_what_ships` | end positions of two rigs over 600 contact-steps at N = 1 and N = 4, and `SubstepAdvance::Shipped` against `Frozen` | **substep loop present, N = 1 shipped, 300 Hz not reached.** N = 1 and N = 4 end **19.796 m** apart; the between-substep chassis advance is worth **2.825 m**. VEH3a's price for N = 4: the sports row's sprint 3.98 → 6.77 s, i.e. the vehicle phase doubled. The fixed step is 60 Hz; the tyres are solved once in it | **PARTIAL** — N is PERF1's to buy (the frame the island does not yet hold, §11) |
| the Pacejka **magic formula** on slip ratio κ and slip angle α | `veh3a_gate::the_magic_formula_is_not_the_old_curve` | two pure curve functions over 400 samples (the arm that tells "the table held" from "the model was never wired") | **0.2127** of peak grip apart from the pre-VEH3a curve at 0.34 × peak slip, **0.0** at the peak | ✓ |
| …and it is the model the cars drive on | `vehicle_grade::every_catalogue_row_sprints_stops_and_tops_out_inside_its_own_spec` | sprint, stop and top speed off the rapier body for every island row, ±5 % bands | sports **3.70 s**, stops **28.8 m**, tops out **60.9 m/s**; sedan **7.53 s / 36.9 m / 33.6 m/s** | ✓ |
| κ and α **per wheel, every step** | `veh3h_gate::the_lap_logs_at_sixty_hertz_and_pie_equals_shipping` | `WheelState::slip_ratio`/`slip_lat` per corner at 60 Hz on the shipped host | logged in `slip_fl..rr` / `slip_lat_fl..rr` over **1999** rows of the lap (§10) | ✓ |
| **multi-contact sensing**, 4-8 casts a tyre | `veh3a_gate::four_casts_do_not_make_a_kerb_worse` | chassis vertical acceleration off `body_linvel` over a 12 cm kerb at three approach speeds, `Footprint::CENTRE` as the control | four casts a wheel; the four/one ratio spans **1.00×–1.02×** (11.8 / 11.6 m/s² at 14.6 km/h). Four casts, at the 60 Hz step — not 360-400 Hz | **PARTIAL** — the rate is the sub-step row's |
| the contact normal feeds the tyre | `veh3a_gate::a_garbage_contact_normal_changes_the_trace` | `tyre_force_with` through `TyreContext::camber_at` | level **0 N**, then **−3 028 / 2 463 / −2 463 N** as the plane tilts | ✓ |
| …and the heightfield's own normal | `vehicle_ground::a_wheel_ray_normal_snaps_at_a_heightfield_cell_diagonal` | a wheel ray across a cell diagonal | a levelled road: worst normal step **0.0643°**; 0.15 m of relief: **15.69°** — which is why the solve reads the cast, not the triangle's normal (VEH1a's disposition) | ✓ |
| **surface µ modifies D** | `veh3a_gate::the_ground_under_the_wheel_decides_the_stop` | `WheelState::surface` after the bridge classified, two real stops on two splats | from 72 km/h **50.0 m** on grass, **65.6 m** on sand | ✓ |
| wet asphalt scales µ down | `veh3a_gate::rain_lengthens_the_stop` | `WheelState::wetness` against the sky, two stops | **40.7 m** dry, **57.0 m** in rain | ✓ |
| a tyre compound per surface | `veh3a_gate::the_compound_row_changes_what_a_surface_is_worth` | two stops, the compound set through the tune door | on sand: road tyres **53.1 m**, off-road **36.9 m** | ✓ |
| mud reachable | `veh3a_gate::mud_is_reachable_through_the_collider_the_wheel_is_standing_on` | `WheelState::surface` on a 0.35-friction slab | **61** steps on mud, µ **0.25** | ✓ |
| **heat shifts B and D** | `veh3a_gate::a_burnout_heats_the_tyre_and_costs_it_grip` | `WheelState::temp_c` over a slipping burnout, a rolling control, two stops at one air temperature | **421** slipping steps took a tyre **20.0 → 25.4 °C**; at 20 °C the rig stops in **32.4 m** under an 85 °C optimum and **65.3 m** under a −30 °C one | ✓ |
| the air the tyre cools into | `veh3a_gate::the_air_temperature_is_the_weathers` | `weather_at` reading `SkyAtmosphere::weather_ambient_c` | **24.0 / −6.0 °C** off the sky's own field | ✓ |
| **torque curve and flywheel** | `veh3b_gate::a_launch_flares_the_crank` | `clutch_slip_rad_s` and `rpm` over 60 steps of a standing start, a rigid control in the same arm | the crank **1 048 rpm** ahead of its gearing, **35 of 60** steps slipping; the rigid control **0** | ✓ |
| …a downshift blips | `veh3b_gate::a_downshift_flares_the_crank` | a recorded `(rpm, gear)` trace | biggest sweep **929 rpm over 22 steps**; rigid **1 441 rpm in 1** | ✓ |
| a fuel cut at redline | `veh3b_gate::the_limiter_cuts_and_restores` | `fuel_cut`'s own edges | **27** cut edges, **13** saws, **153 rpm** mean amplitude over **5.8** steps, peak **6 652** (plateau 7 020) | ✓ |
| **differentials — open / LSD / locked** | `veh3b_gate::the_three_differentials_launch_differently` | driven-axle ω and distance off the body, one wheel on mud, 3 s | open **6.60 m / 307.3 rad/s** apart; LSD **8.04 m / 0.0**; locked **9.00 m / 28.4** | ✓ |
| the LSD law | `veh3b_gate::the_lsd_transfer_is_a_preload_and_a_ramp` | a pure function (the anti-vacuity beside the world arm) | preload **60 N·m** alone, **200 N·m** on a 0.5 ramp over a 400 N·m axle, bounded at 260 | ✓ |
| **weight transfer** `F_z = F_static − m·a·h/L` | `veh3b_gate::the_axle_loads_are_the_formulas_on_the_shipped_rows` | `WheelState::load_n` per axle on the SHIPPED springs | sedan braking **7.3 %** from the formula, launching **8.4 %**; truck **2.2 % / 3.0 %**; **0 of 150** braked steps pinned at travel | ✓ |
| …with a bump stop, not a clamp | `veh3b_gate::the_bump_stop_is_a_rate_that_rises_and_the_rows_are_sprung_for_it` | the strut's force past 85 % of travel; the nine rows' standing fractions | a millimetre at the stop costs **370 N** against **20 N** mid-travel (19×); all nine rows stand on **45 %** of their travel | ✓ |
| nose-dive and squat | `veh3b_gate::the_nose_dives_and_the_tail_squats` | front and rear compression | fixture **45.6 / 8.9 mm**; the shipped sedan **118.5 / 14.1 mm** | ✓ |
| the lap's own load transfer | `veh3h_gate::the_lap_logs_at_sixty_hertz_and_pie_equals_shipping` | four `load_n` per step on the shipped host | front axle **49.6 %** standing → **75.5 %** braking (§10) | ✓ |
| **turbo boost** | `veh3b_gate::the_boost_spools_and_blows_off` | `DrivetrainState::boost` over 240 steps, instant and N/A controls, the re-armed dead time | peak **0.70**, half after **202** steps, **0.70** dumped in six steps; dead time **9 steps** at 0.15 s, 0 without | ✓ |
| the tyre model's cost | `veh3a_gate::the_vehicle_phase_costs_what_it_prints` | the vehicle phase alone, 64 cars, min of five, a no-car control | **0.4036 ms** release (**6.31 µs a car**) against 0.5 ms | ✓ |
| a missed ray is AIR, not asphalt | `veh3b_gate::a_car_with_no_wheel_on_the_ground_says_air` | `surface_census` over `WheelState::contact` | 4 of 4 → asphalt µ 1.000; 0 of 4 → **air** µ 0.000 | ✓ |

## §2 Item 2 — "Modular joint hierarchy: doors, hoods and components as child entities connected by breakable physics joints"

| the doc asks for | arm | what it READS | measured today | verdict |
|---|---|---|---|---|
| parts as child entities | `veh3c_gate::every_authored_family_is_a_car_with_doors` | the spawned rig's part entities per family | sedan **24** parts (4 doors, 2 bumpers, 6 panes, 6 hinged), cruiser **25**, SUV **23**, sports **19**, truck **17**, van **12** — **120 parts over six rows** | ✓ |
| doors on **real revolutes** | `veh3c_gate::a_door_opens_on_its_hinge_and_shuts_again` | the door's rapier body and its joint, angle every 12 steps | opens to **65.7°** on its limit, the hinge carries **5.3 N·s**, shuts at **0.184°** | ✓ |
| …and the imported cars' doors on the PACK'S hinges | `veh3f2a_gate::the_calibration_doors_swing_on_the_packs_hinge` | the solved door's hinge point over the swing | **7.545 mm** off the pack pivot at worst, opened to **66.0°** (VEH3c's primitive door: 9.160 mm) | ✓ |
| **breakable** — over the threshold the joint goes | `veh3c_gate::a_crash_at_sixty_sheds_the_bumper_and_pops_the_bonnet` | parts attached before/after, the peak blow | 60 km/h: **18 945 N·s**, 24 → 23 attached, the front bumper shed and the bonnet live | ✓ |
| the crash table | `veh3c_gate::the_crash_table_is_monotone_in_speed` | `m·|dv|` per blow, parts, panes, hull joules | §5 | ✓ |
| a hinge torn off by what it meets | `veh3c_gate::an_open_door_is_torn_off_by_a_lamp_post` | `joint_impulse` (the WHOLE step's since VEH3g) against `HINGE_TEAR_MPS` | a post: **976 N·s**, 24 → 23, shed; open air **8 N·s**, 24 → 24 | ✓ |
| the joint's impulse is the whole step's | `joints3d::a_joints_impulse_is_the_whole_steps_not_one_substeps` | a hanging 1 000 kg box on three joint kinds | fixed, revolute and spherical each carry **163.500 N·s** against m·g·dt **163.500** (**1.0000**; it read 0.2500 before VEH3g) | ✓ |
| a shed part is a body on the road | `veh3c_gate::a_shed_bumper_in_the_road_is_run_over` | a cast over the panel; a car crossing it | 58.5 mm proud; a car rides **1.9 mm** higher over it and keeps going at **3.01 m/s** | ✓ |
| glass health | `veh3c_gate::a_round_through_a_window_takes_the_window` | the pane's state after 1 500 J | a 120 J pane: gone, not drawn, **6 shards**; at 1e9 J it holds | ✓ |
| **the dent** (a panel push-in on primitives) | `veh3c_gate::a_crash_dents_the_panel_it_reaches_and_not_the_one_it_does_not` | the drawn box of the bonnet vs the boot | 60 km/h: bonnet **40.5 mm**, boot **0.0 mm** | ✓ |
| the **mesh-space crumple** on shells and art | `veh3f2b_gate::a_shell_crumples_its_mesh_by_the_crash_table` | the damage resource and the committed shell's vertices | §5 — **0.0405 → 0.2200 m** over 15-90 km/h, 518 of 5 134 vertices | ✓ |
| …drawn by the shipped projector | `veh3f2b_gate::the_shipped_projector_draws_the_crumple` | vgeom instances, crumples built | whole **32** instances, after 60 km/h **30**, **1** crumple built and reused | ✓ |
| …on imported art | `veh3f2b_gate::the_calibration_sedans_art_crumples_through_the_same_door` | the art body state; LOCAL: the art vertices | art_body dent **0.2200 m**; locally **6 206 of 56 932** vertices | ✓ |
| engine health, fire | `veh3c_gate::a_car_with_no_hull_left_burns` | `fire_step` beside the hull | lights at **36 000 J** of a 36 000 J hull | ✓ |
| a hurt engine / a flat | `veh3c_gate::a_dead_engine_stalls_and_a_hurt_one_is_slower`, `veh3c_gate::a_flat_tyre_pulls` | 0-100 whole vs half; rpm after five seconds; drift and yaw | whole **3.70 s**, half **6.52 s**; dead **851 → 0 rpm**; a flat drifts **−8.542 m / 1.46°**, rides **17.8 mm** lower | ✓ |
| PIE == shipping on a crash | `veh3c_gate::pie_equals_shipping_on_a_crash_course`, `veh3f2b_gate::pie_equals_shipping_on_a_shell_crash` | both hosts' damage folds | equal; the shell drop's hull dent **0.1960 m** on both | ✓ |
| cheap when nothing happens | `veh3c_gate::a_thousand_parked_cars_with_parts_cost_what_they_cost_without_them` | 1 000 parked cars with and without parts, release, min of five INTERLEAVED rounds of twenty steps a side (the VEH3h audit's conditioning; it was the min of five single steps) | **×1.054 / ×1.068 / ×1.041 / ×1.072 / ×1.058 / ×1.044 / ×1.066** over seven audit runs, median **×1.058** against a ×1.05 ceiling — **NOT held** (the cert read ×1.020-×1.051 on the unconditioned clock). Printed, not re-minted; the arm asserts a ×1.25 tripwire | **✗** — **PERF1** (the latched parts' per-row work in `step_bodywork`, not attributed further) |
| **the art machines' cab doors** | — | — | not built: an art row keeps only its seats | **✗** — VEH3-carried, ~1 d a machine |
| the crumple in the editor's Simulate viewport | — | — | the builder is in `inf-player::vmesh`; `inf-viewport` cannot reach it without a lockfile move | **PARTIAL** — VEH3-carried, ~0.5 d |

## §3 Item 3 — "Kinematic entry: character entry linked to full-body IK targets on door handles, seats and steering wheels"

| the doc asks for | arm | what it READS | measured today | verdict |
|---|---|---|---|---|
| the state machine Locked → … → Driving | `veh3d_gate::the_machine_walks_its_phases_in_order_with_their_durations` | the phase trace, the root path, the yaw | the saloon from 2 m back: locked **0.067 s**, unlocking **1.333**, opening **0.750**, entering **0.550**, seated **0.917**, driving; the approach **2.239 m in 1.333 s = 1.680 m/s**, facing ≤ **0.0138°** off the flank normal | ✓ |
| hand IK to the outer handle, blended in 0.2 s | `veh3d_gate::the_hand_takes_the_outer_handle_within_two_centimetres` | the POSED hand joint vs `handle_world` | weight 0.083 → 1.000 over **0.200 s**; at weight 1, **7** steps, worst **0.00 mm** | ✓ |
| the door on its motor as the hand reaches | `veh3d_gate::the_door_opens_on_its_revolute_motor_as_the_hand_reaches` | the hinge angle off the joint | the motor turns the real revolute; the door phase ends at its limit | ✓ |
| **the seat INSIDE the car** | `veh3d_gate::the_seat_is_inside_the_cabin_on_every_family` | pelvis and head joints vs roof, per family | sedan pelvis **−0.900 m** from the roof, sports −0.842, SUV −1.248, van −1.742, truck −1.190, cruiser −0.958 | ✓ |
| hands on the rim through a full lock | `veh3d_gate::the_hands_follow_the_rim_through_a_full_lock` | posed hand joints vs `wheel_grips` at the rim the arm reads off the wheels | the rim swept ±**450°**, **880** hand-steps at weight 1, worst **0.00 mm** | ✓ |
| feet on the pedals | `veh3d_gate::the_feet_press_the_pedals_with_the_inputs` | posed foot joints vs pedal faces | throttle 1.00: the right foot pressed **70.0 mm**, **0.00 mm** off its pedal | ✓ |
| a passenger's grab bar and floor | `veh3d_gate::a_passenger_rides_its_own_seat_and_does_not_drive` | the passenger's joints; the car's travel | pelvis on its cushion, hands **0.00 mm** off the bar, feet **0.00 mm**; the car moved under the DRIVER'S stick | ✓ |
| **the posed joints on the SHIPPED host** | `veh3d_gate::the_shipped_host_measures_the_posed_joints_against_the_sockets` | `RuntimeSim::boarding_residuals` — the posed joint vs sockets recomputed from the live car | outer **9** rows **0.00 mm**; inner **21** rows **0.00 mm**; rim **589** rows **0.00 mm**; pedals **601** rows **0.00 mm** | ✓ |
| every roster family on the shipped host | `veh3d_gate::the_roster_families_board_at_their_sockets_on_the_shipped_host` | the same residuals over twelve rows | §6 — all rows 0.00 mm but the semi (outer **15.55 mm**, rim **2.42 mm**) | **PARTIAL** — the semi's cab step |
| the cab-step climb for tall cabs | `veh3f2b_gate::the_cab_step_climb_and_the_residuals_on_the_shipped_host` | the machine, the capsule trace, posed joints | bus step **0.388 m**, 6x6 **0.348**, semi **0.737**; the semi's outer handle **15.55 mm** at weight 1 | **PARTIAL** — the semi handle over 2 cm (VEH3-carried) |
| the inner latch fires at reach end | `veh3f2b_gate::the_inner_latch_waits_for_the_reach_and_the_hand_holds_to_the_shut` | `mark_s`, the hinge, the posed hand | latch at **0.317 s** at **65.8-66.0°** on all nine rows; the hand holds the pull to the shut on **5** | ✓ |
| the carjack | `veh3d_gate::the_carjack_plays_the_same_pipeline` | both bodies' phase traces | the hero 0.050 / 1.333 / 1.217 / 0.550 / 0.917 s; the victim `jacked` **2.133 s** → exiting **0.450 s**, lands **1.910 m** out, the car at **0.009 m/s** | ✓ |
| the exit and the roll | `veh3d_gate::the_exit_is_the_reverse_and_never_lands_in_geometry`, `veh3d_gate::a_moving_exit_is_a_roll` | the phases, the hinge, the clearance; the mode trace | exiting **0.883 s**, closing **0.600 s**; bail at **7.05 m/s** → a roll | ✓ |
| a press lands once whatever the frame rate | `veh3d_gate::every_host_sees_a_press_once_whatever_its_frame_rate` | both hosts × zero-step E, two-step release, the wheel | the hero `locked` after a zero-step press on both hosts; the release crouches once | ✓ |
| the island's MetaHuman boards | `veh3d_gate::the_islands_hero_boards_drives_and_rolls_out` | the MetaHuman's joints vs sockets (LOCAL) | seated pelvis **−1.072 m** from the roof; hands **0.00 mm** off the rim; feet **0.00 mm** | ✓ |
| the imported calibration car boards | `veh3f2a_gate::the_calibration_car_boards_at_its_sockets_on_the_shipped_host` | posed joints vs the pack's sockets | outer **0.00 mm** (17), rim **0.00** (589), pedals **0.00** (656) | ✓ |
| a drawn steering wheel turns with the rack | `veh3f2a_gate::the_drawn_steering_wheel_turns_with_the_rack` | the hub's roll, the grips on the drawn rim | rim angle **450.00°**; grips **1.44 mm** off the drawn rim — on the ART cars; the shells and primitives draw no wheel | **PARTIAL** — a drawn wheel on the shells is VEH3-carried |
| PIE == shipping on the boarding | `veh3d_gate::pie_equals_shipping_on_a_board_drive_exit_course` | both hosts' boarding folds | **900** steps, **306** folded bytes, every phase seen, equal | ✓ |
| the knee / elbow limits of the template rig | — | — | inverted on the committed template (`template.rs`), costed at a rig + clip re-bless | **✗** — VEH3-carried (CHAR1-class re-bless) |

## §4 Item 4 — "RPM/load audio blending: a granular player modulating cylinder firing, intake and gear whine off telemetry"

| the doc asks for | arm | what it READS | measured today | verdict |
|---|---|---|---|---|
| combustion grains at three loads | `veh3e_gate::the_grain_period_is_in_the_committed_bytes` | the committed grain bytes, the arm's own period estimator | 15 grains; worst **−0.058 %** off `(rpm/60)(cyl/2)`; a four against a V8 **2.0009** | ✓ |
| blended by RPM on the stream | `veh3e_gate::the_load_crossfade_moves_between_the_three_grains` | grain volumes vs triangles written in the arm | **782** running steps, worst |volume − level × weight| **0** | ✓ |
| heard at the firing rate | `veh3e_gate::the_stack_is_eight_loops_on_salted_keys_in_a_pinned_order` | the engine-start `Play`s: keys, clips, order | **8** loops on salted keys; **282** driven steps heard within **0.0577 %** of the firing rate | ✓ |
| gear whine ∝ v · r | `veh3e_gate::the_whine_steps_down_on_every_upshift` | the whine's pitch across shifts vs the ratios | 1→2: **0.6634** against 0.6591; 2→3: **0.6586** against 0.6552 | ✓ |
| turbo spool and a blow-off on the lift | `veh3e_gate::the_turbo_spools_and_blows_off_on_the_lift` | turbo pitch; blow-off `Play`s | **169** audible turbo steps; one blow-off at boost **0.651** | ✓ |
| **squeal by slip, not speed** | `veh3e_gate::the_squeal_follows_slip_and_not_speed` | the squeal key at two speeds; the course | slip 1.6 at **10 and 100 km/h** → identical `SetVolume 0.30321, SetPitch 0.95`; loud below 5 m/s on **172** steps, above 15 m/s on **45** | ✓ |
| surface impulses — kerbs, gravel | `veh3e_gate::the_kerb_thumps_front_then_rear`, `veh3e_gate::a_squeal_changes_its_clip_with_the_surface` | impulse `Play`s at the kerb; the squeal re-Play on gravel | front at step **586**, rear at **593**; the gravel clip on **665** | ✓ |
| the rolling road by speed and surface | `veh3e_gate::the_road_rolls_by_speed_on_its_surface` | the roll key against a curve written in the arm | **782** live steps, error **0**; loud > 15 m/s on **157**, silent standing on **263** | ✓ |
| doors: creak, slam, silent motor | `veh3e_gate::the_door_slams_on_the_shut_step_and_the_motor_rows_are_silent`, `veh3e_gate::a_door_held_open_through_the_close_does_not_slam` | the door keys vs `door_deg`/`handle_weight`; the joint angle | **46** motor-only rows silent; a door held open at **65.66°** does not slam | ✓ |
| no clipping | `veh3e_gate::the_course_render_does_not_clip` | the course WAV's bytes | without the limiter **185** clipped samples, shipped **0** (peak 0.882) | ✓ |
| near traffic sings | `veh3e_gate::a_traffic_car_sings_the_near_stack_on_its_own_cadence` | traffic `Play`s, the re-tell cadence | the NEAR stack (grain, squeal, roll); **64** traffic updates against 480 for the hero's car over 160 steps | ✓ |
| PIE == shipping on the stream | `veh3e_gate::pie_equals_shipping_on_the_audio_course` | both hosts' command slices, whole | **1 012** steps, **4 676** commands, **47** kinds, identical | ✓ |
| cost | `veh3e_gate::sixty_four_cars_cost_what_they_print` | the audio phase's clock, 64 driven cars, release, min of five | **0.0164 / 0.0233 / 0.1081 ms** at 0 / 1 / 64 voiced (886.6 commands a step at 64) against 1.0 ms | ✓ |
| **sound out loud** | `offline::the_device_opens_or_says_why` | the device path | Windows / macOS open a device; Linux keeps the null backend (no ALSA on the runner) | **PARTIAL** — Linux device, VEH3-carried |
| **a per-sample granular synthesizer** (the doc's `GranularEngineSynthesizer`) | — | — | the grains are LOOPS re-pitched per step, not a per-sample grain scheduler | **✗** — VEH3-carried, ~1.5 d |

## §5 Item 5 — "Per-surface data: physical materials mapped to friction, drag and acoustic samples"

| the doc asks for | arm | what it READS | measured today | verdict |
|---|---|---|---|---|
| the µ table | `veh3a_gate::the_surface_table_says_what_the_research_says` | six rows restated from the doc | asphalt 1.00 … sand; any edit reds it | ✓ |
| a map of the ground's surface | `veh3a_gate::a_road_laid_over_grass_is_a_road` | two real `SurfaceMap`s, one with a street | a 24 m street stamps **216 of 49 284** cells asphalt | ✓ |
| the stop reads it | `veh3a_gate::the_ground_under_the_wheel_decides_the_stop` | two stops on two splats | grass **50.0 m**, sand **65.6 m** from 72 km/h | ✓ |
| rain, compound, mud | see §1 | | 40.7 / 57.0 m; 53.1 / 36.9 m; µ 0.25 | ✓ |
| the acoustic sample per surface | `veh3e_gate::a_squeal_changes_its_clip_with_the_surface` | the re-Play clip on gravel | `Squeal_Gravel` on step 665 | ✓ |
| the HUD shows it | `veh3a_gate::the_shipped_host_draws_the_tyre_row` | `window.rs` by source fragment | the player's `drive_readout` draws the Ring-0 tyre row | ✓ |
| drag per surface | — | — | rolling resistance is per ROW, not per surface | **PARTIAL** — a surface drag term is VEH3-carried (~0.25 d) |

## §6 Beyond the checklist — the doc's other claims and the mandate's

| the claim | arm | what it READS | measured today | verdict |
|---|---|---|---|---|
| **class handling profiles** (CoM, suspension, friction by class) | `veh3f_gate::every_class_drives_inside_its_feel_band` | driven sprint / stop / lateral per row on the shipped host | 135 wheeled rows in their bands; class order coupe > sedan > SUV > jeep by lateral g | ✓ |
| a static fraction a road car keeps | `veh3f_gate::every_roster_row_settles_inside_its_static_fraction` | settled struts, 135 rows | 30-45 % | ✓ |
| the sports row under four seconds by gearing | `veh3f_gate::the_sports_row_is_under_four_seconds_by_gearing_with_its_spring_kept` | 0-100 and the settled fraction | **3.70 s**, static **0.448** of travel, first gear 3.8; stop **30.3 m at 1.30 g**, lateral **1.23 g** | ✓ |
| **the 150-row roster** | `veh3f_gate::every_roster_row_parses_round_trips_and_names_only_known_keys` | `merge_toml`, every key re-applied | **155 rows, 3 953 keys** (§7) | ✓ |
| multi-axle rigs | `veh3f_gate::a_multi_axle_rig_stands_on_all_its_wheels_and_they_carry_its_weight` | wheels in the world, strut loads vs weight | Hauler 6/6 grounded, loads **93 189 / 93 195 N**; APC 8/8, **133 412 / 133 416 N** | ✓ |
| trailers, tracks | `veh3f_gate::the_trailer_follows_through_a_slalom_and_does_not_separate`, `veh3f_gate::the_tracked_rig_turns_by_skid` | headings and the kingpin gap; yaw under a skid | peak hitch **9.8°**, the trailer's heading **0.93 s** behind, kingpin gap **0.000 m**; tracks yaw **0.670 rad/s** at full steer, **0.0000** with none | ✓ |
| the tandem brakes by axle load | `veh3f2b_gate::the_tandem_brakes_by_axle_load_and_no_rear_axle_locks_first` | per-axle loads and lock speeds | Hauler **0.697 g**, front **62.7 %** never locks, the rear at 5.4 m/s | ✓ |
| traffic draws the roster by class weight | `veh3f_gate::traffic_draws_the_roster_by_class_weight` | 20 000 identities, the parked and the circuit draws | parked: sedan **31.2 %** against a 31.0 % weight, SUV 21.6 / 21.4, coupe 9.6 / 9.5 … within a point on every class | ✓ |
| **body kinds on the street** | `veh3f2b_gate::the_island_traffic_makes_no_moving_contact_and_its_hero_classes_draw_shells` | the physics world's contact events; each resident traffic car's own children | 10 island minutes, release: **180 shell / 420 imported / 0 primitive** samples, **0** moving contacts in **3 186** solid contact events (§7) | ✓ |
| **the shells' silhouette** — the brief's 25 % side-outline bar | `veh3f2b_gate::the_committed_shells_are_closed_cut_and_not_the_box_family` | committed bytes, settled wheels, the outline rasters | pickup **27.6 % MET**; saloon **16.2 %**, coupe **13.5 %**, SUV **16.8 %**, cruiser **16.8 % NOT MET** (the VEH3f.2b audit's measured ruling: a glazed real car sits near the box family; imported Fab cars score 28-36 % mostly from window holes) | **PARTIAL** — NOT MET on four, pinned honestly |
| the arches fitted to the tyres | same arm | each settled wheel centre straight up to the body, less the radius | saloon **0.116 m**, cruiser **0.117 m** (≤ 0.125 asserted); coupe **0.272**, SUV **0.253**, pickup **0.361** | **PARTIAL** — three arch sets VEH3-carried (§13) |
| every shell on an island car | `veh3f2b_gate::every_shell_hangs_on_a_car_in_the_committed_island` | the committed level | coupe 1, cruiser 4, pickup 2, saloon 1, SUV 1 | ✓ |
| **the imported cars** | `veh3f2a_gate::every_art_row_draws_committed_geometry_without_the_art` | fallback GUIDs | **74** art rows, **1 789** drawn meshes, every one a committed fallback on CI | ✓ |
| …their wheels turn | `veh3f2a_gate::the_art_wheels_turn_with_wheel_speed` | the drawn spin vs the roll | **31.88 m** on r 0.351 m: 5 204° owed, **5 176.3°** drawn | ✓ |
| …the import is a pure function | `veh3f2a_gate::the_import_is_deterministic` | every file of two imports (LOCAL) | **920 and 920** files, **0** differ | ✓ |
| …PIE draws what the cook draws | `veh3f2a_gate::play_draws_every_texture_the_cook_draws` | the textures by path (LOCAL) | **103** textures, **103** by path, **0** missing | ✓ |
| **airplanes** — lift, stall, the runway | `veh3g_gate::every_aeroplane_lifts_off_inside_the_islands_runway`, `veh3g_gate::the_stall_collapses_the_lift_and_drops_the_nose` | brake release to 35 ft per row; α past `stall_deg` | §8 — Luxor **1 370 m** of 1 700; the stall at **16°**, deepest α **32.2°** | ✓ |
| **helicopters** — hover, the winch, the air lane | `veh3g_gate::five_helicopters_lift_off_hover_and_translate`, `veh3g_gate::the_cargobob_winch_lifts_a_car_and_lets_it_go`, `veh3g_gate::the_police_helicopter_flies_its_air_lane_and_never_the_road` | climb and hover; cable tension; the lane | §8 | ✓ |
| **boats** — planing, Archimedes, the sail | `veh3g_gate::a_planing_hull_rises_and_a_displacement_hull_does_not`, `veh3g_gate::the_big_hulls_float_where_archimedes_puts_them`, `veh3g_gate::the_sail_drives_across_the_wind_and_not_into_it` | draughts; the force log | §8 | ✓ |
| aircraft parked are chocked | `veh3f2b_gate::a_parked_aircraft_is_chocked` | a parked Dodo on a 3° slab | **0.040 m** in ten seconds (saloon control 0.150) | ✓ |
| **streaming at aircraft speed** | `veh3g_gate::streaming_holds_at_120_mps_over_the_real_island` | the player's loader, both streamers (LOCAL) | **0** blocking loads; **15** activations; closest arrival **252.0 m**; the worst step **288 ms** when two cells arrive together | **PARTIAL** — the 288 ms activation step is **PERF1**'s |
| a moving vehicle is never deleted by streaming | `cell_stream::a_mover_over_an_empty_cell_is_rehomed_not_despawned`, `veh3g_gate::the_islands_dodo_flies_off_the_apron_and_stays_in_the_world` | the mover's entity after its birth cell leaves | the Dodo moved **900 m** east and is still in the world | ✓ |
| **a parked vehicle beyond the collider band** | `veh3h_gate::a_parked_vehicle_beyond_the_collider_band_holds_its_pad` | the CI island's camp fire appliance's body over 15 s; `PhysicsBridge3D::parked_beyond_band` | **CLOSED by the VEH3h audit** (a parking-hold defect, not sim LOD): **0.00 m** and 0.00 m of fall with the hero 120 m away, held kinematic (was **36.63 m**, 3.65 m of fall); **1.51 m** with the hero 20 m away (inside the band, simulated: the in-band creep, §13) | ✓ |
| **a fixed step beside every body kind** (the cert's "Harbour City cruiser stall") | `veh3h_gate::a_fixed_step_beside_every_body_kind_is_a_fixed_step` (and its LOCAL, ignored companion in the same file, `the_harbour_city_pile_is_set_aside_on_the_real_island` — not a citation: the gate refuses an ignored one) | the whole fixed step and `queries` per step with the hero 2.5 m off each side of five shells, two imported rows and a primitive, against the hero 20 m off; LOCAL: the real island's Harbour City cruiser | CI island: the dearest spot **1.30×** its control, queries equal; the real island: **50.0 ms** a step beside the cruiser against **34.0** at 20 m (1.47×) — it was **266.0** (`character move` 232.8 ms; 358.7 on an earlier dwell). The cause was not the car (`shell_body` carries no collider) but a crowd PILE, the next row | ✓ |
| **a crowd pile does not stall the step** | `character_move_cost::a_pile_of_characters_is_set_aside_and_one_in_the_way_still_blocks` | 64 steered capsules inside a quarter metre vs a 2 m lattice; `pawns_set_aside`; a body 1.0 m ahead of a walker | the mover sets aside the NPCs it overlaps: **4 032** on the pile's first step, 0 on the lattice; release **2.99 ms** against **1.80** (1.67×; fought: 135.99 ms, 76×); a body in the way still blocks (the walker stops at **0.62 m**) | ✓ — the pile itself (the society's nearest-desk assignment) is carried, §13 |
| **traffic makes no moving contact** | `veh3f2b_gate::the_island_traffic_makes_no_moving_contact_and_its_hero_classes_draw_shells` | contact events | **0** moving pairs in 10 island minutes | ✓ |
| **the arrival snap** (VEH3f.2b carried, 4.47 m) | `traffic_3d::a_car_whose_leg_closes_short_of_its_slot_is_not_snapped_onto_it` | each Full commuter's record against its chassis; the per-step XZ jump across the hand-off | **CLOSED by this wave** (the arrival hold): the jump was the whole shortfall (**112.54 m** under the arm's accelerated clock); now within the half-lane bound | ✓ |
| the driving hand-off moves nothing | `traffic_3d::a_car_leaving_the_steered_tier_lands_where_its_body_already_was` | the body across the 64 m boundary | within the half-lane + two steps bound | ✓ |
| **PIE == shipping on the island** | `island_gate::pie_equals_shipping_on_an_island_drive` | both hosts' `state_bytes` over 900 steps | **900** steps, **900** distinct states of **11 358** bytes, equal; **337** posed characters on both hosts | ✓ |
| …over the whole lap | `veh3h_gate::the_lap_logs_at_sixty_hertz_and_pie_equals_shipping` | both hosts' 60 Hz rows and every step's state digest | **2 107** rows and every step digest equal (re-derived by the PAR1b.2 audit after `drive_intent`'s braking horizon; the VEH3h close read 1 999 / 2 231) | ✓ |
| …over the whole lap in an imported car | `veh3h_gate::the_imported_row_drives_the_same_lap_on_both_hosts` | the same, `obey_rocoto` (`dd_suv`) | equal and CLEAN since the PAR1b.2 audit (§10): peak one-step deceleration **1.07 g** where it struck a sign post at **37.82 g** | ✓ |
| **two cooks** | `veh3h_gate::the_lap_logs_at_sixty_hertz_and_pie_equals_shipping` | every file of two cooks of the CI island | byte-identical | ✓ |
| **GPU instancing, mesh LOD** for high-poly vehicles | `veh3f2b_gate::sixty_four_shell_cars_cost_what_they_cost` | the projector's wall time and instance count; the GPU frame | 64 shells: **0.463 ms** projected (the cert: 0.461), **2 048** instances, **17 876** LOD-0 triangles a car, GPU **0.326 ms** (min of 30; the cert quoted 2.721 ms, the VEH3f.2a audit 2.274 — the GPU column does not reproduce across sessions and is printed, never asserted) — the meshlet DAG decides the tiers; there are no pack LODs for a shell and no HLOD | **PARTIAL** — vehicle LOD/HLOD is **PERF1**'s |
| **the island's frame** at 1080p | the ignored `fps_instrument` island arm (`the_island_in_imported_traffic`, needs a local cook) | 1080p release at the Harbour City crossroads | §11 — SHIPPED p50 **69.2** / p95 **72.2 ms** (the audit's run; the cert's 68.5 / 71.0) against a 38 ms ceiling | **✗** — **PERF1** |
| **the driving frame**, 64 cars + an occupied car, the composed city | `fps_instrument::sixty_four_cars_drive_the_composed_city_at_shipping_resolution` | the frame, the GPU clock, the in-frame vehicle and audio phases, min of five sessions | §11 — SHIPPED p95 **25.8 ms** held; LIT p95 **39.2 ms** over 38 (the cert's 24.7 / 39.6) | **✗** (lit) — **PERF1** |
| **headlight cones** at night | `veh3h_gate::a_headlamp_is_an_emissive_lens_and_no_car_carries_a_light` | every catalogue and roster row spawned; lamp parts, their `Material`, any `Light` | **59** front lamps across **166** rows, all emissive; **0** `Light` components on any rig — there is no cone | **✗** — **PAR1** (vehicle lamps on **PAR0**'s many-lights substrate; `MAX_LIGHTS` is 16 a frame) |
| tyre smoke, exhaust, dust | — | — | no particle system | **✗** — **PAR2** |
| clearcoat / flake paint | — | — | one PBR layer | **✗** — the **PAR** arc, ~2.5 d |
| **the licence wall** | `veh3f2b_gate::the_cook_refuses_content_whose_licence_may_not_ship`, `veh3f_gate::nothing_from_unreal_is_committed` | the cook's result; `git ls-files` | refuses `UE5_Mannequins: Mannequin_Ref`; **3 063** tracked paths, **0** Unreal-shaped | ✓ |
| the v28 vehicle schema (the doc's `VehicleDataProfile`) | `veh3a_gate::every_v28_tunable_survives_the_wire` | real bytes through the editor's codec | **100** tunables authored, encoded, decoded, read back by name | ✓ |
| **the Demon row against Forza** | `veh3h_gate::three_classes_against_their_forza_inspirations` | 0-60 mph off the body on the shipped host | **CLOSED by the VEH3h audit**: retuned to the Demon's own 1 044 N·m and a drag-radial launch (`longitudinal_grip` 2.0): **2.48 s** against FH5's **2.259 s** (+10 %, WITHIN); it was 3.72 s (+65 % against the sheet as it reads today) | ✓ |

---

## §7 The roster census

**155 rows, 3 953 keys** (`veh3f_gate::every_roster_row_parses_round_trips_and_names_only_known_keys`):

| class | rows | the doc's count | | class | rows | the doc's count |
|---|---|---|---|---|---|---|
| coupe | 20 | 20 | | cargo | 5 | 5 |
| sedan | 20 | 20 | | bus | 5 | 5 |
| suv | 20 | 20 | | service | 5 | 5 |
| truck | 20 | 20 | | van | 5 | 5 |
| jeep | 5 | 5 | | airplane | 5 | 5 |
| hummer | 5 | 5 | | helicopter | 5 | 5 |
| emergency | 5 | 5 | | marine | 8 | 8 |
| military | 5 | 5 | | trailer | 2 | 0 (ours) |
| construction | 5 | 5 | | | | |
| utility | 5 | 5 | | | | |
| freight | 5 | 5 | | | | |

**By body kind on the street** — the island's traffic after **10 island minutes**, release
(`veh3f2b_gate::the_island_traffic_makes_no_moving_contact_and_its_hero_classes_draw_shells`,
600 samples):

| class | shell | imported | primitive |
|---|---|---|---|
| sedan | 60 | 60 | 0 |
| suv | 60 | 120 | 0 |
| truck | 60 | 0 | 0 |
| van | 0 | 240 | 0 |
| **total** | **180** | **420** | **0** |

The committed roster rows by the kind they draw: 74 art rows (imported, with a committed
fallback), the shell rows of five shells, and the rest the primitive box families
(`veh3f2a_gate::every_art_row_draws_committed_geometry_without_the_art`: 1 789 drawn meshes
over the art rows).

## §8 Air and sea

| row | measured today (`veh3g_gate`, dev = release: no clock) |
|---|---|
| take-off to 35 ft, flaps set | Duster **415 m**, Dodo **492 m**, Luxor **1 370 m** (81 % of the 1 700 m strip), Titan **1 237 m**, Jetliner **1 289 m** |
| the Dodo circuit | lift-off after **308.9 m** at 28.6 m/s; climb **2.38 m/s**; peak 42.1 m; touchdown **1.21 m/s**; roll-out **158.5 m** |
| the stall (`stall_deg` 16) | α past it **5.35 s** after the stick at 28.1 m/s; CL **1.380 → 0.593**; **89.4 m** lost; deepest α **32.2°**, **5.4 s** stalled |
| helicopters | five rows climb **21.7 m**, hover |vs| **0.000** |
| the winch | the car to **28.8 m**; tension **18 639 N** against its **18 639 N** weight (−0.00 %) |
| the air lane | **0 of 3 600** steps on a road route; lowest **43.3 m**; on scene **30.1 s**; the orbit swept **127°** |
| planing | jetski **0.105 → 0.051 m** at 16.2 m/s, top **32.7**; Jetmax **0.210 → 0.102**; Speeder, dinghy likewise |
| Archimedes | tug **1.9999 m**, superyacht **1.3020**, cruise **4.3128** — ±0.00 mm |
| the sail | beam reach **5.97 m/s**, heel 1.9°; dead run **3.91**; head to wind **−1.52** |
| the tug | the 900 t superyacht **396.3 m**, at **4.96 m/s** |
| PIE == shipping | **1 800** steps, 5 craft bit-identical |
| cost (release) | 18 craft **0.0403 ms** against 18 sedans **0.1063 ms** — **2.24 µs a craft** |

## §9 Streaming

| row | measured |
|---|---|
| 120 m/s over the real island (LOCAL) | **0** blocking loads, **15** activations, closest arrival **252.0 m**, the cell ahead missing on **10 of 1 280** steps; worst step **288 ms** (two cells on one step) — **PERF1** |
| a mover over an empty cell | re-homed, not despawned (`cell_stream::a_mover_over_an_empty_cell_is_rehomed_not_despawned`) |
| a parked vehicle outside the collider band | holds: **0.00 m** in 15 s (was 36.63 m) — CLOSED by the VEH3h audit's parking hold |

---

## §4 of the tables — the feel against Forza

The published figures are Forza Horizon 5's own acceleration test for the STOCK car, read
off the `forza.labsgg.com` FH5 car sheets' "simulation results" — RE-READ by the VEH3h audit on 2026-09-26: the cert's figures (2.293 / 5.615 / 2.500 s) were each 1-2.5 % off what the sheets print, and the gate now carries the page's own numbers. The band is **25 %**, justified
per class rather than chosen: a roster row authors its inspiration's mass and engine on
one gearbox shape per class, on a µ 0.9 slab against Forza's dry asphalt, with no launch
control and no drag-radial compound — the surface alone is worth ~10 % of a traction-limited
0-60 and the class gearbox about as much again. Three rows re-measured by this wave on the
shipped host (`veh3h_gate::three_classes_against_their_forza_inspirations`; the gate
re-derives each line and compares it to the digit):

| row (body) | inspiration | engine 0-60 mph | FH5 0-60 | engine 0-100 mph | FH5 0-100 | verdict |
|---|---|---|---|---|---|---|
| `bravado_gauntlet_hellfire` (shell) | 2018 Dodge Challenger SRT Demon | **2.48 s** | 2.259 s | **4.37 s** | 5.847 s | WITHIN (+10 %) — retuned by the audit (1 044 N·m, drag-radial `longitudinal_grip` 2.0); was 3.72 s OUTSIDE. Past 60 mph it out-runs FH5 by 25 %: the compound is no longer the limit there |
| `karin_asterope_gz` (imported) | 2023 Toyota Camry TRD | **6.40 s** | 5.555 s | **13.42 s** | 13.785 s | WITHIN (+15 %) |
| `pegassi_zentorno` (shell) | 2011 Lamborghini Sesto Elemento | **2.35 s** | 2.442 s | **4.57 s** | 5.088 s | WITHIN (−4 %) |

FEEL-VS-FORZA bravado_gauntlet_hellfire: 0-60 mph 2.48 s -- WITHIN the band (retuned by the VEH3h audit: the Demon's own 1 044 N.m and a drag-radial launch, longitudinal_grip 2.0; it was 3.72 s OUTSIDE at 652.5 N.m on road tyres).
FEEL-VS-FORZA karin_asterope_gz: 0-60 mph 6.40 s -- WITHIN the band.
FEEL-VS-FORZA pegassi_zentorno: 0-60 mph 2.35 s -- WITHIN the band.

**Three more rows, drawn at random by the audit** (the counter-hash: the smallest
`sha256("VEH3h-audit-2026-09-26:" + row)` per body kind among the road classes, the gate's
three excluded), sprinted by the same `sprint` on the shipped host, FH5 figures read off the
same sheets the same day:

| row (body) | inspiration | engine 0-60 mph | FH5 0-60 | engine 0-100 mph | FH5 0-100 | verdict |
|---|---|---|---|---|---|---|
| `gallivanter_baller_st_d` (shell) | 2015 Range Rover Sport SVR | **4.75 s** | 4.228 s | **11.62 s** | 10.266 s | WITHIN (+12 %) |
| `dinka_sugoi` (imported) | 2018 Honda Civic Type R | **6.22 s** | 4.872 s | **12.92 s** | 11.334 s | **OUTSIDE (+28 %)** — a front-drive launch the traction aid holds at ~0.44 g; 400 N·m (the car's own) reaches only 5.83 s. VEH3-carried (§13) |
| `karin_rebel` (primitive) | 2019 Toyota Tacoma TRD Pro | **7.03 s** | 7.617 s | **21.57 s** | 23.271 s | WITHIN (−8 %) |

The whole catalogue's bands (VEH3f's, re-derived by its audit on the chassis's own lateral
acceleration) hold today (`veh3f_gate::every_class_drives_inside_its_feel_band`): coupe
sprint 2.45-6.38 s, sedan 2.82-10.60, SUV 3.48-9.88, truck 3.17-14.82, buses 18.50-29.58
(and the island rows in `vehicle_grade::every_catalogue_row_sprints_stops_and_tops_out_inside_its_own_spec`: sports **3.70 s / 28.8 m**, sedan **7.53 s / 36.9 m**).

## §5 of the tables — the crash and the dent

The crash table (`veh3c_gate::the_crash_table_is_monotone_in_speed`, the saloon into a
wall, `J = m·|dv|`), today, on the saloon's VEH3f.2b ride:

| wall at | blow | parts | shed | open | panes | hull |
|---|---|---|---|---|---|---|
| 15 km/h | **30 N·s** | 24 → 24 | 0 | 0 | 0 | 0 J |
| 30 km/h | **4 219 N·s** | 24 → 24 | 0 | 0 | 0 | 753 J |
| 45 km/h | **9 287 N·s** | 24 → 23 | 1 | 0 | 0 | 4 213 J |
| 60 km/h | **17 757 N·s** | 24 → 23 | 1 | 1 | 0 | 15 469 J |
| 90 km/h | **27 948 N·s** | 24 → 21 | 2 | 2 | 1 | 38 704 J |

(The VEH3g audit's table read 30 / 4 056 / 9 308 / 17 730 / 28 606 N·s on the saloon's old
ride; VEH3f.2b seated it 8 cm deeper. The numbers moved with the ride, not the model.)

The mesh-space dent on the committed shell (`veh3f2b_gate::a_shell_crumples_its_mesh_by_the_crash_table`):

| wall at | hull dent | direction | vertices moved of 5 134 | deepest |
|---|---|---|---|---|
| control (nothing ahead) | 0 | — | 0 | 0 |
| 15 km/h | **0.0405 m** | (+0.046, −0.003, +0.999) | 501 | 0.0405 |
| 30 km/h | **0.0823 m** | (+0.028, +0.006, +1.000) | 518 | 0.0813 |
| 60 km/h | **0.1902 m** | (+0.031, +0.008, +0.999) | 518 | 0.1886 |
| 90 km/h | **0.2200 m** (cap) | (+0.033, −0.000, +0.999) | 518 | 0.2188 |

## §6 of the tables — boarding, posed joints

The phases (`veh3d_gate::the_machine_walks_its_phases_in_order_with_their_durations`,
the saloon from 2 m back): locked **0.067 s** · unlocking **1.333** · opening **0.750** ·
entering **0.550** · seated **0.917** · driving. (VEH3d measured 1.433 and 0.600: the
approach is 0.170 m shorter on the seat that moved into its cushion, and the inner latch
waits for the reach since VEH3f.2b.)

The residuals are POSED-JOINT numbers — the joint the GPU skins against a socket recomputed
from the live car (`RuntimeSim::boarding_residuals`), never the solver's own error
(`veh3d_gate::the_roster_families_board_at_their_sockets_on_the_shipped_host`):

| row | outer handle (rows) | inner | rim (rows) | pedals (rows) |
|---|---|---|---|---|
| sedan | 0.00 mm (9) | 0.00 (9) | 0.00 (589) | 0.00 (601) |
| sports | 0.00 (10) | 0.00 (3) | 0.00 (589) | 0.00 (601) |
| suv | 0.00 (13) | 0.00 (3) | 0.00 | 0.00 |
| truck | 0.00 (16) | 0.00 (3) | 0.00 | 0.00 |
| van | 0.00 (29) | — | 0.00 | 0.00 |
| cruiser | 0.00 (10) | 0.00 (7) | 0.00 | 0.00 |
| ambulance / swat | 0.00 (30) | — | 0.00 | 0.00 |
| brute_bus | 0.00 (31) | — | 0.00 | 0.00 |
| jobuilt_hauler (semi) | **15.55 (27)** | — | **2.42** | 0.00 |
| vapid_contender | 0.45 (21) | — | 0.00 | 0.00 |
| mtl_packer (art) | no door | — | 0.00 | 0.00 |

## §7 of the tables — the audio command stream

One hero car, the VEH3e course (`veh3e_gate::the_audio_log_holds_the_drive_and_the_count_is_stated`):

| state | steps | commands a step (mean) | min |
|---|---|---|---|
| parked / boarding | 232 | 0.04 | 0 |
| idle | 154 | 0.00 | 0 |
| launch (burnout) | 45 | 9.38 | 4 |
| full throttle | 195 | 11.09 | 9 |
| lift / coast | 30 | 8.57 | 8 |
| handbrake slide | 60 | 7.55 | 6 |
| sliding to a stop | 210 | 5.10 | 0 |
| engine off | 86 | 0.09 | 0 |

**1 012** steps, **4 676** commands (277.2 a second); the 131 072-entry ring holds **473 s**
of it. The squeal table: step 293 at **0.03 m/s**, |slip| 21.2 → volume **0.500**; step 564
at **24.61 m/s**, |slip| 2.03 → **0.480** — loud because it slides, not because it is fast.

---

## §10 THE LAP

The island's circuit, driven at 60 Hz on the SHIPPED host
(`veh3h_gate::the_lap_logs_at_sixty_hertz_and_pie_equals_shipping`): the CI island cooked
(twice — byte-identical) and booted as `run_headless` boots it; the first of the
settlement's circuits (largest block group first, nearest the centre) whose line stays on
GRADED ground — no half metre of it climbing more than 0.20 rise over run
(`clear_circuit`'s `steepest_grade`; the VEH3h audit moved the lap there, see below) —
the street centreline round a 2 × 1 block group, **304.2 m**, four corners, **5.3 m** of
climb across it; the crowd, the traffic and the level's two parked vehicles set aside (a
scripted driver has no eyes); a shell muscle coupe (`bravado_gauntlet_hellfire`, turbo,
rear drive) spawned on the start line through the rig door; the hero boarded through
VEH3d's pipeline (one interact press); a three-second burnout at the line with the traction
aid off; one lap on the engine's own lane driver (`traffic::drive_intent`, 22 m/s on the
straights, its bend rule in the corners); a five-second cool-down straight.

LAP-FACTS: 2107 rows at 60 Hz; the burnout took the rear tyres +4.11 C; the cool-down straight cooled them 2.22 C; boost 0.971 peak, 0.000 half a second after the lift; the front axle 49.6 % standing and 84.6 % braking; peak braking 1.40 g; the lap 27.12 s.

(Re-derived by the PAR1b.2 audit, 2026-10-08: `traffic::drive_intent` now brakes for a
corner beyond its one-second look — the braking horizon at `CORNER_BRAKE_MPS2` — so every
corner of the lap is entered slower and braked earlier and softer. The VEH3h close read
1999 rows, 2.34 C, 75.5 %, 1.82 g, 25.32 s. The plot paragraph below is the VEH3h close's
plot of that earlier lap; it was not regenerated.)

The columns (the CSV's SHAPE is the arm's): `t, phase, x, z, s_m, speed_mps, long_g, lat_g,
throttle, brake, rpm, gear, boost, slip_fl..rr, slip_lat_fl..rr, load_fl..rr, temp_fl..rr,
surface` — 30 columns, corners named off each wheel's own mount. The bytes live in the
session scratchpad (`AUDIT-VEH3h-FINAL\lap.csv`); the PLOT (`AUDIT-VEH3h-FINAL\lap.png`,
never committed) shows the rear tyres climbing **20.2 → 24.3 °C** through the burnout's
staircase of slip (the fronts flat), both axles heating at every braking zone and cooling on
every straight (the hottest tyre peaks at **37.4 °C**), the boost spooling to
**0.971** off the line (0.965 inside the burnout's three seconds) and to ~0.55-0.70 on every
straight, dumping to 0 on every lift, and
the front axle's share of the wheel loads jumping from ~49 % to **~65-76 %** in each braking
zone. The steepest one-step deceleration of the lap is its braking (**1.82 g** on that
lap; **1.40 g** on the PAR1b.2 audit's re-derivation) — asserted.

**The cert's lap struck a terrace bank.** Its circuit (the settlement's largest, 609.6 m)
read one step at **−36.75 g** on its west street, which the cert called "the fire hall's pad
edge". The audit mapped it with a ray grid over the impact point: every ray answers the
TERRAIN collider, which rises **1.5 m at ~39°** (0.804 rise over run) between z −262 and −264
on x 354 where two levelled block rows meet. Neither a kerb nor a collider seam — a line a
road car should not be driven on — so the LAP LINE moved (four of the nine candidate
circuits cross the bank at 0.804; the chosen one's steepest half metre is 0.109). The
Demon row was retuned in the same commit (§4 of the tables), so every lap number above is
new: the cert's lap read 3 161 rows / 44.68 s, +6.35 °C, 2.99 °C, boost 0.965, 48.4 → 70.3 %,
1.37 g.

PIE == shipping over the whole lap: the editor's loose-level host drives the same lap and
its **2 107** rows and every step digest of `state_bytes` are equal to the shipped host's,
bit for bit (the VEH3h close: 1 999 rows, 2 231 digests). The same lap in the IMPORTED
`obey_rocoto` (`veh3h_gate::the_imported_row_drives_the_same_lap_on_both_hosts`) is equal
on both hosts too: **2 371** rows, the lap **31.52 s**, the burnout **+1.57 °C** (all four
tyres share an all-wheel-drive launch), boost **0.945**, the front axle **47.9 % → 81.3 %**
braking (the PAR1b.2 audit's re-derivation; the VEH3h close read 2 445 rows, 32.75 s, 0.953,
81.9 %). **It is a clean lap since the PAR1b.2 audit**: the steepest one-step deceleration
is **1.07 g**. Before, the heavier SUV ran ~13 m wide out of the second corner and struck
something at (421.7, −287.1) at **37.82 g** — the PAR1b.2 audit NAMED it: the sign post of
the southbound approach at (423.78, −290.2), on the far side of the junction the car was
overshooting (it left the brake at ~9.4 m/s where the coupe reaches ~4.5, because the lane
driver's bend rule saw only one second of path). PAR1b.2 moved that post 1 m toward the kerb
and the car stopped dead against it; the fix is in the driver (`drive_intent`'s braking
horizon), not in the post.

## §11 THE FPS ROW

**1080p, release, RTX 4070 Ti, off CI** (`fps_instrument::sixty_four_cars_drive_the_composed_city_at_shipping_resolution`):
the composed city (100 grammar blocks, the streamed terrain, the phase-29 character) with
**64 island saloons** driving the full model down its middle street (four casts a wheel,
N = 1, the drivetrain and the tyre heat live) and a **65th, occupied** (a seated driver
through the traffic's `occupy` door) singing its whole voice stack; **min of five
sessions**, each a fresh fixture and three rounds of 120 frames after a discarded pass.

| configuration | p50 | p95 | worst | GPU frame | in-frame vehicle phase | audio |
|---|---|---|---|---|---|---|
| composed city, no cars, SHIPPED (`fps_instrument::the_frame_at_shipping_resolution`) — the audit's three runs, min by p50 (the first, straight after the driving frame, read p50 17.81) | 11.58 ms | 12.49 ms | 13.47 ms | 3.13 ms | — | — |
| composed city, no cars, LIT (the audit's first run) | 16.90 ms | 18.63 ms | — | 5.59 ms | — | — |
| **64 + 1 cars, SHIPPED** (the audit, min of five) | **23.51 ms** | **25.75 ms** | **26.66 ms** | 10.37 ms | 0.903 ms (13.89 µs a car) | 0.134 ms, 710.8 commands a frame |
| **64 + 1 cars, LIT** (the audit, min of five) | **37.36 ms** | **39.20 ms** | **41.12 ms** | 17.91 ms | — | — |
| the island, Harbour City crossroads, the art, SHIPPED (400 traffic records, 1 027 skinned) — the audit's re-run, ONE session (the cert: min of five, 68.47 / 70.96 / 72.78) | **69.21 ms** | **72.16 ms** | **79.98 ms** | 27.05 ms | — | — |
| the island, LIT — the audit's one session (the cert: 81.14 / 86.30 / 89.66) | 82.27 ms | 86.69 ms | 91.97 ms | — | — | — |

**`SHIPPING_FRAME_CEILING_MS` (38)** — UNCHANGED: held by the driving frame SHIPPED (p95
**25.75** on the audit's run, 24.67 on the cert's); **NOT held** by the driving frame LIT
(**+1.2 ms** on the audit's run, **+1.6 ms** on the cert's — over, PERF1, never less than
over) nor by the island (**+34.2 ms** SHIPPED p95 on the audit's run, +33.0 on the cert's).
The audit's re-run of the driving frame is the SECOND of two: the first read the vehicle
phase at 0.334 ms and 80.2 audio commands a frame because the audit's own parking hold had
frozen the harness's cars (their throttle carried no `occupied`) — fixed
(`VehicleControls::commands`) before the numbers above were taken.
The ceiling is not re-minted. The breakdown, routed to **PERF1** by name:

* the island (SHIPPED, GPU 27.0 ms on the audit's run): the crowd's skinned pass
  **11.4 ms**, the depth prepass **7.2 ms**, vgeom **3.3 ms** (record **7.5 ms** CPU),
  scatter 2.4 ms; the CPU: the fixed step **11.1 ms** (the cert: 11.7) (physics sync 3.4, solver 3.3, animation 1.1, gameplay 0.7,
  vehicle 0.6), projection 5.6 ms, render record 24.2 ms — the VEH3f.2a audit's breakdown,
  a little dearer;
* the lit driving frame (GPU **17.8 ms** against 6.0 without the cars): scatter
  **9.1 ms** (3.4 without), vgeom **2.3** (0.45), gi **1.9** (0.76), terrain **1.7** (0.73),
  vsm-raster **0.70** (0.03); the CPU: the fixed step **5.9 ms** (3.2 without — the vehicle
  phase 1.07 of it), render record **12.2 ms** (4.8). Measured as deltas, not attributed
  pass by pass to the shells — which is PERF1's job, by name (vehicle LOD/HLOD and the
  per-asset vgeom record).

**The tier ladder's price per car** (`veh3h_gate::the_tier_ladder_prices_a_car_per_rung`,
release, min of five rounds of sixty steps, 64 parked saloons on one rung against a
no-car control):

| rung | what a car is | per car |
|---|---|---|
| Full (≤ 64 m) | a rig: four wheels, four rays each, the drivetrain, a handbrake | **13.94 µs** (the cert: 14.13) |
| Near (64-128 m) | a `Body`: chassis and panels, kinematic, placed by its clock | **3.54 µs** (3.60) |
| Dormant (> 128 m) | a record, nothing built | **0.03 µs** |

A car has THREE rungs (the VEH3h audit, priority d'): the `Far` rung is the crowd's, and `TRAFFIC_RADII` sets near == far so no car can reach it — by design since VEH2b (`TRAFFIC_NEAR_M`'s doc: a car has no impostor path). The arm now asserts that no car reached it on any rung; there is no table row for a rung nothing reaches.

## §12 THE REFERENCE FRAMES

The engine's frames are the hands-on's, retaken by the VEH3h audit ON THE CRUISER
(`AUDIT-VEH3h-FINAL\130..136`): the real editor relaunched on the audit's tree, the island
open, the Harbour City cruiser's `shell_body` selected (`01f`), Play in New Window, the hero
placed 2.5 m off the cruiser's flank facing it, and `demo.ps1 -CertOnly` firing every frame
on a hero.csv column. Taken: the door phase (**130**: `BOARDING opening`, the `[KeyE] Enter
vehicle` prompt on the cruiser's door), at the wheel (**131**), the burnout (**132**: the rear
tyres **20.4 → 26.9 °C**), the drive with the telemetry HUD (**133**), the crash (**135**: hull
**94 %**, **2** parts shed, 21/26/26/21 °C on the HUD) and after it (**136**). The cert's
three attempts at the cruiser ran at 2.5 s a frame; this session boarded it in the time the
pipeline takes (the stall was the crowd pile, closed). **Not taken: the kerb** — `thumps`
stayed 0 (the leg's blind tap does not reach a kerb; the arm is
`veh3e_gate::the_kerb_thumps_front_then_rear`). **The cruiser is SILENT**: the island's EMS
fleet is spawned with `engine_voice: false` (EMS1's reason — idle loops at the kerb — is
retired since VEH3e silences an unoccupied car anyway), so the squeal column read **0.00**
over the session and the leg now reads an engine as RUNNING off its rpm (column 45) when it
has no voice. Carried, §13. The audio device line: `Speakers (Steam Streaming Speakers)
48000 Hz opened`. The reference is
`docs/reference_videos/frames/driving/0006-0030` and `steal-car/` (never committed). The
contact sheet is `VEH3h-FINAL\reference-contact-sheet.png`. The honest sentence on each:

| reference | what it shows | the engine | what matches, what does not, who closes it |
|---|---|---|---|
| driving/0006 | a ute doing a rolling burnout at a junction, tyre smoke off its rear wheels, sun glare | the burnout frame | the physics matches — a rear axle past its peak slip, a squeal, a tyre heating (§10) — and the SMOKE does not exist: **PAR2** |
| driving/0014 | a road-rage bark subtitle beside a box van and a ute in traffic | the drive frame | traffic with drivers and passengers and a Near tier that sings matches; an NPC voice line on a near miss does not exist — the bark is dialogue, outside the VEH3 arc (no wave in the mandate names it; recorded, not claimed) |
| driving/0022 | a kerb-side takedown on a verge by a highway, a helicopter in the air | the kerb frame; §8 | a car over a kerb with the thump and a helicopter that flies its lane match; the takedown is melee, not driving |
| driving/0030 | a hotel forecourt with a FULL parking lot | the drive frame | parked cars at every kerb slot match in kind (0.45 occupancy by design — the reference's lot is fuller); the lot's lighting and asphalt grime are the PAR arc |
| steal-car/0016, 0022, 0028 | night: headlight cones on the road, tail lamps glowing, underglow, street-light pools | the engine has no night frame this wave | the lenses glow (emissive) and **no car casts light** — `veh3h_gate::a_headlamp_is_an_emissive_lens_and_no_car_carries_a_light`: **PAR1** on **PAR0** |
| steal-car/0040 | exhaust FLAMES off a tuned car | — | **PAR2** |
| steal-car 0010-0035 | the carjack beats: approach, yank the door, pull the driver out, drive off | the boarding frames | the pipeline matches beat for beat (`veh3d_gate::the_carjack_plays_the_same_pipeline`); the driver's resistance / fight is not built |

## §13 The licence table

| content | where | provenance | licence | committed? |
|---|---|---|---|---|
| the roster (155 rows) | `crates/inf-ecs/src/roster.toml` | the doc's lore names; published specs | ours | yes |
| the five shells | `samples/vehicle-shells/` | `inf-dcc` lofts, byte-locked to their generator | ours | yes |
| the art fallback | `samples/vehicle-art/` | `inf_dcc::cube` + `cylinder` at measured extents | ours | yes |
| DrivableCars, VehicleVarietyPack Vol.1/Vol.2, MarketplaceBlockout (the 13 imported cars) | `island-build/project/Content/UE/` | Fab via UE 5.8's export → our importer | Fab Standard, commercial tier (the user, 2026-09-25) | **no** |
| ConstructionVehiclesPack1 (7 machines) | the same | the same | `licence_may_ship = false`; the cook refuses it without `--local-reference` | **no** |
| UE5 Mannequins | the same | the same | refused by the cook | **no** |

---

## What parity is NOT yet — CARRIED, with the wave that closes each

| gap | cause | closes it | price |
|---|---|---|---|
| the sub-step rate (N = 1 of a 300 Hz ask) | N = 4 doubles the vehicle phase and the island's frame is already over its ceiling | **PERF1** (the budget that would pay for N) | — |
| the island's frame (p95 72.2 ms / 38 on the audit's run) | crowd skinned 11.4 ms GPU, depth prepass 7.2, the fixed step 11.1, vgeom per-asset record | **PERF1** | ≥ 3 d |
| the lit driving frame with 64 shells (p95 39.2 / 38 on the audit's run, 39.6 on the cert's) | 64 shells' parts through vgeom + VSM casters | **PERF1** (vehicle LOD/HLOD) | — |
| the 288 ms two-cell activation step | activation is not amortised over steps | **PERF1** | ~1-2 d |
| headlight cones, underglow | no vehicle carries a `Light`; `MAX_LIGHTS` 16 a frame | **PAR1** on **PAR0** | — |
| tyre smoke, exhaust flames, dust | no particle system | **PAR2** | — |
| clearcoat / flake paint | one PBR layer | the **PAR** arc | ~2.5 d |
| the per-sample granular synth | the grains are pitched loops | VEH3-carried | ~1.5 d |
| the Linux audio device | no ALSA headers on the runner | VEH3-carried | a runner package + a `cargo deny` look |
| the art machines' cab doors | an art row keeps its seats only | VEH3-carried | ~1 d a machine |
| the crumple in the Simulate viewport | the builder lives in `inf-player` | VEH3-carried | ~0.5 d |
| the arch daylight on the coupe / SUV / pickup (0.272 / 0.253 / 0.361 m) | one body per shell serves rows whose settled tyres sit 13-25 cm apart | VEH3-carried: a second body per ride-height cluster, the island rows re-pointed, both levels re-blessed | ~1 d (re-priced from 0.5 d: three bodies, their table rows, six arms that name five shells) |
| the four shells under the 25 % side-outline bar | a measured ruling (VEH3f.2b audit): the bar does not measure car-ness | — (pinned, not carried as a defect) | — |
| the semi's outer handle 15.55 mm, rim 2.42 mm | the cab-step climb's hand target on a 0.737 m step | VEH3-carried | ~0.5 d |
| a drawn steering wheel on the shells and primitives | only the art cars draw a wheel | VEH3-carried | ~0.5 d |
| a surface drag term | rolling resistance is per row | VEH3-carried | ~0.25 d |
| the template rig's knee / elbow limits | inverted in `template.rs` | VEH3-carried (a rig + clip re-bless) | ~1 d |
| the parked-car cost ceiling at ×1.05 NOT held (median ×1.058 over seven conditioned release runs) | the latched parts' per-row work in `step_bodywork`, not attributed further | **PERF1** | — |
| **the crowd pile** the cert found as the "Harbour City cruiser stall" — its COST is closed (`character_move_cost::a_pile_of_characters_is_set_aside_and_one_in_the_way_still_blocks`); the PILE remains: 204 residents on four points of one building, 80 on one quarter metre | `society::plan_day` sends every resident to the NEAREST workplace and its errand, unclaimed (the night shift and the evening are claimed; the day is not) | VEH3-carried to the society's owner (a claimed or capacity-bounded desk, and the NPC1d/VEN1 counts re-blessed) | ~0.5 d |
| the in-band creep of the CI island's camp appliance (**1.51 m** in 15 s, ~0.11 m/s) | the generator parks it 6 m off its lane's centreline on a 7.8° grade between two buildings; it rolls 18.6° within 2 s with one wheel hanging (`island.rs`'s apron placement reads no ground by law) | VEH3-carried (the apron placement) | ~0.5 d |
| `dinka_sugoi` (imported, Civic Type R) against FH5 — **+28 %**, OUTSIDE the band (the audit's random row) | a front-drive launch the traction aid holds at ~0.44 g; the car's own 400 N·m reaches only 5.83 s | VEH3-carried (a front-drive launch model) | ~0.5 d |
| ~~**the imported lap strikes something**~~ **CLOSED by the PAR1b.2 audit** (§10) | the obstacle was the southbound approach's sign post at (423.78, −290.2); the cause the lane driver's one-second look (no braking for a corner beyond it); fixed by `drive_intent`'s braking horizon (`CORNER_BRAKE_MPS2` 6.0), unit arm `traffic::tests::a_corner_beyond_the_lookahead_is_braked_for` (mutation: horizon off -> 25 m/s against 17.03, red); the imported lap's peak one-step deceleration is now 1.07 g | closed | — |
| the island kerb frame | a blind tap does not reach a kerb on the island's streets (the audit's cruiser session: thumps 0 again) | VEH3-carried (VEH3e's placement door) | ~2 h |
| **the boarded fleet car is silent** (the Harbour City cruiser: squeal 0.00, no grain) | the island's EMS fleet spawns with `engine_voice: false` (EMS1); VEH3e already silences an unoccupied car | VEH3-carried: the fleet spawned with a voice, both committed island levels re-blessed with that cause, the island audio arms re-read | ~0.5 d |
| the fallback-body island frame | not re-measured this wave (the art cook only) | PERF1's baseline row | ~1 h |
| NPC road-rage barks, a carjack victim who fights | dialogue and melee, not driving | no wave in the mandate names them | — |
