# P27.5 — tiers, retirement, and the rulings the finale owed

*2026-08-13. The decisions Phase 27's last batch made, each with the measurement
it rests on. The gate itself is `runtime/inf-player/tests/phase27_gate.rs` and
its arms carry their own reasoning; this memo is for the rulings that are
choices rather than checks.*

---

## 1. The tier flip, and the golden trap under it

The ROADMAP's clause is *"High/Medium run VSM; Low keeps CSM (the clamp law);
CSM code stays as the fallback, demoted not deleted."* Three things have to be
true at once and only two of them are about code.

**The clamp landed at P27.1**, deliberately early, so the phase could not ship
something that forgot it: `RenderTier::apply` does
`settings.vsm.enabled &= !matches!(self, RenderTier::Low)` — an `&=` against a
constant, which can clear a flag and can never set one — plus three `min`s on
the numbers. `caps::a_tier_clamps_virtual_shadow_maps_down_and_never_up` is its
arm and `vsm_receiver::the_tier_clamp_runs_vsm_on_high_and_medium_and_keeps_the
_cascade_on_low` is the device half.

**The trap is that shadows have been OFF BY DEFAULT since P13.** So "High runs
VSM" is a claim about which *mechanism* serves shadows when a project enables
them, and not a claim that a High frame has shadows in it. The P27.4 batch
already had to correct its own first draft on exactly this point
(`ShadowSettings::enabled` is cleared at Low too, and has been since P13.4.2).

**Nothing about the shipped default changed in P27.5, and that is a product
decision recorded rather than a thing left undone.** Turning shadows on by
default is a change to what every existing project looks like and what every
committed golden records; it belongs to whoever owns the shipping profile, with
its own re-bless of fifty-four frames. The goldens stay frozen by keeping the
golden-generating configurations explicit — the four `vsm_*` frames pass
`vsm_golden_settings()` by hand — and by the two new tier knobs defaulting to
exactly the P27.1–P27.4 configuration.

**Verification that the default path did not move**: all 100 arms of
`crates/inf-render/tests/golden.rs` pass under `INF_GOLDEN_STRICT`, which
compares against the committed PNGs pixel for pixel. That is a stronger check
than the battery runs by default, and it is the one this batch's shader edits
needed.

## 2. The two tier knobs, and why they clamp in opposite directions

`VsmSettings` gained `mark_stride` and `pcf_radius`. Read together they look
like a bug in a diff and they are one law:

| knob | clamp | High | Medium/Low | why that direction |
|---|---|---|---|---|
| `mark_stride` | `max` | 1 | 2 | a *larger* stride dispatches *fewer* threads |
| `pcf_radius` | `min` | 1 | 0 | a *larger* radius loads *more* texels |

Measured, in the two units that decide: marking threads per pixel **1.0 → 0.25**
and kernel taps per shaded fragment **9 → 1**. `caps::the_marking_stride_and_the
_kernel_radius_clamp_cost_down_on_every_tier` asserts the *cost* falls rather
than that two numbers move, so a re-tune that inverts one of them fails.

**What a stride costs, stated in the phase's own direction**: a page only a
skipped pixel would have asked for is not marked, is not resident, and resolves
to `VSM_ENTRY_NONE` — which the receiver reads as **lit**. A coarser marking
grid therefore leaks light at a silhouette; it never punches a hole in one.
`a_coarser_marking_stride_marks_a_subset_of_what_every_pixel_marks` is the
containment arm, and `VsmStreamStats::mark_threads` is why it is not vacuous: a
page set is a containment claim, and a stride that reached nothing produces a
subset of itself.

**What the radius costs**: the slope bias is `(R + ½)·√2` texels, so it follows
the radius rather than staying at High's — a bias sized for a kernel that is not
running is peter-panning nobody asked for. The two travel in **one uniform**
(`counts.z` and `params.w`, both previously reserved) and are derived from each
other in `VsmReceiverParams::new`, which is what makes disagreeing impossible
rather than merely unlikely.

Two WGSL constants were retired for this and it is an exchange, not a loss: what
pinned them (`the_receivers_two_halves_spell_one_set_of_constants`) is replaced
by `the_kernel_and_its_bias_are_read_from_the_uniform_the_tier_writes`, which
names the loop header and the multiply — the expressions whose *absence* is the
defect (the P27.4-audit corollary about where a source pin must aim).

## 3. The two light ceilings: REFUSE, do not lift

The P27.4 audit found it and left it armed: `MAX_LIGHTS` is 16 (the lights
uniform's array, and the largest scene index any lit shader's loop reaches) while
`VSM_MAX_PROJECTIONS` is 64, and `VsmSystem::for_scene` registered a tree for
every shadow-casting light in scene order with no reference to the first number.
A point or spot light at index ≥ 16 could hold a tree that marks, rasterizes and
evicts pages **no shader can sample**.

**Ruled: refuse at registration, typed and counted.** The alternative — lifting
`MAX_LIGHTS` — is the wrong lever. That number is the **forward renderer's
analytic light loop**, a per-pixel shading cost P7.1 chose; moving it makes every
lit fragment in the engine pay for a virtual-shadow ceiling. What a shadow phase
may decide is whether to allocate pages for a light nothing shades.

`vsm_light_trees` returns `VsmTreeSet` now — the trees plus **two** counters, one
per ceiling, because the two refusals are different facts and a single number
would wear both names. `for_scene` logs each once. It is a **stop** rather than a
skip, and that is free rather than careful: `index ≥ MAX_LIGHTS` is a *suffix* of
the light list, so the handle invariant the projection cap protects is untouched
by construction.

**The sun is exempt from the slot mechanism and not from the ceiling.** It rides
`VsmReceiverParams::counts.x` rather than a `GpuLight` — which is exactly why it
was put there — but a directional light past index 16 has no direct term either,
and shadowing a light nothing shades is the same waste in a different place.

The invariant this buys, and it is now assertable: **every rasterized page is
sampleable.** A tree exists only for a light some lit shader can shade; a
point/spot slot is `GpuLight::params.w`, inside the array; the sun's is
`counts.x`.

## 4. Receiver completeness: closed for meshlets and foliage, refused for voxels

The P27.4 remainder was literal and verified at the time: *"`vgeom_mesh.wgsl` and
`scatter_mesh.wgsl` take the analytic term but not the directional one, because
neither has ever called `shadow_factor`"* — `git log -S` over both files' whole
history returned nothing. So the engine's flagship geometry path and its ground
cover received no sun shadow **from either mechanism**.

**Closed.** Both shaders already bind the shared environment group, so
`shadow_factor` was composed in and only the call site was missing. It is spelled
exactly as `mesh.wgsl` spells it — first directional only, guarded by
`sun_shadowing_enabled()` — so the off path is instruction-for-instruction
unchanged. Closing it through `shadow_factor` closes it for the **cascade** at
the same time, which the device arm measures rather than argues.

**`voxel.wgsl` is refused, with a sentence and an arm.** It is
`ShaderKind::Plain`: it binds no environment group and *cannot* call either
receiver. Giving it one to get a shadow would make it look integrated — AO'd,
fogged, shadow-receiving — while it still casts no shadow, feeds no GI and is
invisible to the depth prepass, which is the half-wired shape its own header has
refused since P21.1. The fix is not a call site; it is **P28.1's VisBuffer
resolve**, where meshlet and voxel surfaces are shaded through one material pass
and the env group is bound once for all of them.
`every_env_bound_lit_path_receives_the_suns_shadow_and_voxel_is_refused` reads
`passes::shader_kind` for the composition, so the day it changes the refusal is a
failing test rather than a stale comment. (`sprite.wgsl`'s `MAX_2D_LIGHTS` loop
is a separate radial-falloff system, not an oversight — the P27.4 audit's
amendment, unchanged.)

## 5. The GI seam, from the other side

P27.1's `the_gi_pass_never_reads_a_shadow_page` bans VSM from GI's own sources.
It could not see the meeting point **this batch created**: four lit shaders now
compute a virtual-shadow factor and a GI irradiance in one fragment, a few lines
apart, and multiplying the second by the first is a one-character edit that would
make the ambient term at a fixed world point a function of where the viewer
stands.

`a_shadow_page_never_reaches_the_ambient_term` is the second half: it scopes the
region from the ambient block to the end of each fragment shader, asserts the
region really contains the GI call, and bans every shadow term inside it — with
the head of the same shader required to *take* the sun shadow, so the ban is not
satisfied by a shader with no shadows in it.

## 6. The blend's resident partner: refused for P27, routed to P28.4

Measured, on a receding plane with four resident levels: **10 of 17** resident
pages (59 %) already have their coarser partner resident, so the level blend
acts on most of the band and is inert on the rest. Both halves are asserted, in
both directions, so the number is a measurement rather than a printout.

**Refused here.** The mechanism a second want needs already exists and already
has an owner: `VSM_PRIORITY_SPECULATIVE`, a rank with no producer, whose
producer the ROADMAP assigns to **P28.4**. Marking the partner unconditionally at
equal priority would make the want set a function of `level_blend` — a *quality*
knob — and would move the page-allocation trace that P27.3's caching clause and
`phase27_gate`'s arm (a) are both measured against. A speculative want at
strictly lower priority does neither, which is the whole reason that rank was
built with one producer missing.

## 7. The skinned cull sphere: one correction, and the fix measured

The carried sentence said the exact bound is *"the posed AABB the renderer is
still not handed"*, which implied `inf-anim`'s cooperation. **It does not.** A
skinned instance already carries its joint palette — the renderer has it, draws
with it, and folds it into the caster's content stamp — so a pose-following bound
is computable from data in hand: transform the bind sphere's centre by each joint
and union, scaling the radius by each matrix's largest axis length.

Measured on the pose the device arm already drives: the shipped bound is
**2.1213 m** at the bind centre, the palette union is **1.4142 m** at the posed
centre — **67 %** of the radius and **30 %** of the volume, and both contain the
posed geometry.

> **P27.5 audit, 2026-08-13 — what those two percentages are.** The fixture is a
> **single translating joint at unit scale**, so the union radius *is* the bind
> radius and the shipped one is `1.5 ×` it: 67 % and 30 % are
> `1/(1 + SKINNED_POSE_MARGIN)` and its cube — the margin's own reciprocal —
> and they would be the same two numbers for a pose that moved a hundred metres.
> The containment checks are real and the correction above stands: the palette is
> in hand and `inf-anim` is not needed. What is **not** established is that a
> palette union is tighter *in general* — with joints far apart the union can
> exceed the inflated bind sphere, and no arm tests that. P28.3 inherits the
> question rather than the number.

**Not landed here.** A tighter caster sphere changes which pages the cull keeps
and therefore which pages a mover invalidates, which is the exact quantity
`phase27_gate`'s arm (c) asserts. It is a change that needs its own arms and its
own re-measurement, in the batch that owns the caster pack — **P28.3**, where the
CPU caster cache already lives.
`the_palette_union_bound_is_tighter_than_the_shipped_pose_margin` is what makes
the day it moves a decision with a number in it rather than a rewrite of a
paragraph.

## 8. Observability

The P27.1 remainder — *"nothing logs `vsm_summary` in a host and no editor
surface exists"* — is closed at both halves.

`ViewMode::VsmPages` is the VT heat-map's twin, one virtual system over, and it
earns a mode of its own because of the state it shows that a lit frame cannot: a
page with **no resident ancestor** reads as *lit*, so a missing shadow and a
surface nothing shadows are the same picture. Blue gives it a colour, beside
green/yellow/orange/red for how many levels behind the served page is and grey
for no tree at all — which is also what §3's refusal looks like from outside.

It rides `flags.w`, which was reserved. `vsm_heat` is deliberately a **page**
view and not a tap view: it re-derives the address from the undisplaced world
position with no normal offset and no kernel, because what it is about is which
page a pixel lands in. It is not a second sampling door — it reads the same table
and the same projection and it never returns a shadow factor.

Both hosts log the line: the editor on entering the view (the ramp answers
*which* pixels are behind, the line answers *by how much, out of how big an
atlas, and did the budget defer*), and the player once a second beside the
terrain and cell streamers. A level with virtual shadows off produces **no line
at all** rather than a line of zeros — `vsm_summary` is `None` there, and an
empty atlas is a different state.

## 9. The ≥8k claim, measured

*Effective resolution* is the virtual extent a receiver can address at its finest
level: pages a side × `VSM_PAGE_SIZE`. Measured off the **live tree** on a
configuration produced by `RenderTier::High.apply`: **64 × 128 = 8 192** texels a
side, **7.81 mm** a texel over a ±32 m level 0. An equivalent single shadow map
is 8 192² × 4 B = **268 MB** against the atlas's **64 MiB** — the sentence the
whole phase exists to make true, as arithmetic.

Medium's clamp takes it to 4 096, which is the honest reading of "High/Medium run
VSM": both run the mechanism, and ≥8k is a **High** claim.
