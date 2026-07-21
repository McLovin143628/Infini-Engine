# Materials

Infinity Engine shades with a physically-based (PBR) model and lets you author materials either as
simple parameter sets or as full node graphs that compile to live WGSL.

## The PBR model

The renderer's mesh pass is a Cook-Torrance GGX metallic-roughness BRDF (Smith geometry, Schlick
Fresnel) lit by a scene-lights buffer — directional and point lights, with hemispheric ambient and
an ACES tonemap. Every actor's **Material** component exposes base color plus **metallic**,
**roughness**, and **emissive**, all editable in the Details panel via reflection. The quickest way
to skin an object is to drag a material asset from the Content Drawer onto it (or use "Apply to
Selection") — the apply is a single undo step.

## Material graphs

For anything beyond flat parameters, open a `.inf_mat` **Material Graph**. The material editor
reuses the same node canvas as Blueprints, with a node kit for texture samples, UVs, math, vector
ops, lerp/mask, procedural generators, and a required **output.surface** sink. As you edit, the
graph is compiled to a `material_surface` WGSL function: the codegen walks backward from the sink,
hoists any value used by two or more nodes into a single `let`, and coerces scalars and vectors so
the WGSL stays well-typed. The result is validated by **naga** at author time, and any validation
error maps straight back onto the offending node.

The editor's left column shows a **live preview** — a PBR-lit sphere rendered offscreen — that
updates as you change the graph, plus a diagnostics list and a drawer showing the generated WGSL.
Because the same generated shader drives both the preview and the asset thumbnailer, what you see
in the preview is what the material actually compiles to.

## Texture graphs and bake

A material graph can also be baked to a texture. The **Bake Texture** action wraps the generated
surface in a compute shader (`cs_bake`) that evaluates the graph per texel into a storage texture,
then writes the result as a new `.inf_tex` asset under `Content/baked`. You can then feed that
baked texture back in as a `tex.sample` input — mixing procedural authoring with baked detail.
Procedural generators (noise, gradients, radial and linear ramps, blends) are available directly in
the graph for both live and baked use.

## Material instances

To vary a material without duplicating its graph, create a **material instance** (`.inf_mati`): a
parent reference plus a sparse set of parameter overrides. Overrides are resolved along the parent
chain when the instance is applied, so you can maintain one base material and tune metallic,
roughness, color, or emissive per instance — the standard pattern for dressing many objects from a
shared look.
