# Starter character

**The engine's committed starter character** - the exact eight assets the
New Character wizard writes for its own default spec, on the 161-bone
mannequin (`BodyPlan::Biped`).

| file | what it is |
|---|---|
| `Starter.inf_skel` | the rig: 161 bones, role table, twist drivers, IK handles, hand cones and the grip catalogue |
| `Starter Body.inf_mesh` | the generated body, heat-weighted onto the rig |
| `Starter Skin.inf_mat` | a neutral matte dielectric, named as the body's material dependency |
| `Starter Idle/Walk/Run.inf_anim` | the generated, **derived** cycles |
| `Starter Locomotion.inf_sm` | the machine proposed from what the derivation measured, with the `Mask_AimOffset` upper-body profile on it |
| `Starter Locomotion.inf_sm.txt` | its reviewable text face |
| `Starter Controller.inf_act` | the Blueprint class the character binds |
| `camera.toml` / `input.toml` | the camera table and the bindings |

Two things ship it: `ProjectTemplate::starter_content` scaffolds it into
every new 3D project, and `samples/island*/island.toml` names it under
`[content]` so the island's hero is this character rather than a capsule.

Generated - do not hand-edit. Regenerate with:

```sh
INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples
```
