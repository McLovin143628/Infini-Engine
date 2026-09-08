# Starter character

**The engine's committed starter character** - the exact thirteen assets the
New Character wizard writes for its own default spec, on the 161-bone
mannequin (`BodyPlan::Biped`).

| file | what it is |
|---|---|
| `Starter.inf_skel` | the rig: 161 bones, role table, twist drivers, IK handles, hand cones and the grip catalogue |
| `Starter_Body.inf_mesh` | the generated body, heat-weighted onto the rig |
| `Starter_Skin.inf_mat` | a neutral matte dielectric, named as the body's material dependency |
| `Starter_Idle.inf_anim`, `Starter_Walk.inf_anim`, `Starter_Run.inf_anim` | the generated, **derived** cycles |
| `Starter_Locomotion.inf_sm` | the machine proposed from what the derivation measured, with the `Mask_AimOffset` upper-body profile on it |
| `Starter_Locomotion.inf_sm.txt` | its reviewable text face |
| `Starter_Controller.inf_act` | the Blueprint class the character binds |
| `Starter_Outfit.inf_mesh` | the clothes: a tee and a pair of trousers, shrink-wrapped off the body's own surface over the joints each covers, skinned by the body's own weights |
| `Starter_Outfit_Top.inf_mat`, `Starter_Outfit_Bottom.inf_mat` | the outfit's two slot materials |
| `Starter_Hair_Mesh.inf_mesh` | a hair cap: authored geometry fitted to the measured skull, rigidly bound to the `head` joint |
| `Starter_Hair.inf_mat` | the hair's material |
| `camera.toml` / `input.toml` | the camera table and the bindings |

Two things ship it: `ProjectTemplate::starter_content` scaffolds it into
every new 3D project, and `samples/island*/island.toml` names it under
`[content]` so the island's hero is this character rather than a capsule.

Generated - do not hand-edit. Regenerate with:

```sh
INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples
```
