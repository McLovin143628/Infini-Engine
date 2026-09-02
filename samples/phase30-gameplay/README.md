# samples/phase30-gameplay

The island's gameplay fixture (wave I6): one grammar-built house, two
hand-hung doors, a rifle on the floor, a destructible target and one hero.

Everything a level cannot carry in its scene is authored by the hero's own
Blueprint on `BeginPlay` - the item catalogue, the two doors, the pickup and
the hero's own health. That is the one authoring surface which reaches the
editor's Simulate, a PIE payload AND a cooked pack with no schema move; see
`inf_blueprint::nodekit`'s `gameplay_nodes` for the accounting.

Beside them, since wave WPN1, **one `.inf_audio`**: the gunshot every round
leaving a barrel names by GUID (`inf_ecs::weapon::WEAPON_REPORT_CLIP`). It is
engine content that happens to live here because this is the one committed
level in the tree that fires a weapon.

The gate over it is `runtime/inf-player/tests/phase30_gameplay_gate.rs`.

Generated - do not hand-edit. Regenerate with:

```sh
INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples
```
