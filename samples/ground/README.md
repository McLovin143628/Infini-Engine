# The engine's ground library (wave TER2a)

Five PBR ground sets — grass, rock, forest floor, sand and soil — and the
first `.inf_tex` files this repository has ever committed. Before TER2a the
whole virtual-texture stack had no content that reached it and the 51 km²
island's ground was one flat colour.

Each set is a `.inf_mat` naming three or four `.inf_tex` v2 tiled
containers: a 1 024² albedo, a 512² tangent-space normal, a 512² ORM, and —
for grass and rock — a 512² high-frequency detail normal.

| set | tiles every | albedo texel | detail tile |
|---|---|---|---|
| `Ground_Grass` | 2.0 m | 1.95 mm | 12.5 cm |
| `Ground_Rock` | 3.0 m | 2.93 mm | 15.0 cm |
| `Ground_ForestFloor` | 2.5 m | 2.44 mm | — |
| `Ground_Sand` | 1.5 m | 1.46 mm | — |
| `Ground_Soil` | 2.2 m | 2.15 mm | — |

**Every map is BC1**, including the normals. That is a measurement, not a
preference: `inf_render::build_vt_level` picks the atlas format from the
stored formats of the textures a level binds, and a MIXED set demotes the
whole pool to RGBA8 at eight times the page bytes. Wave T's `PageFormat::Bc5`
is the right normal-map format and cannot be used beside a BC1 albedo until
the atlas can hold two formats.

Nothing here is hand-painted or imported. `inf_material::ground` synthesises
every texel from an integer hash with no transcendental in the path, so the
bytes are identical on every platform and the lock test below compares them
on every CI leg. Regenerate with:

```
INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples
```
