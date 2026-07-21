/**
 * Material-editor-specific wire types (P7.2). The graph document, nodes, edits,
 * and node definitions reuse the generic `inf-graph` shapes in `blueprintTypes`
 * — only the compile result (WGSL + naga diagnostics + preview image) is
 * material-specific. Field casing matches the Rust `serde` derives (camelCase).
 */

/** A material-compile diagnostic, optionally anchored to a node id. */
export interface MatIssue {
  node?: number | null;
  severity: "error" | "warning";
  message: string;
}

/** The result of `material_compile`: generated WGSL, diagnostics, preview. */
export interface MaterialCompileResult {
  /** The complete naga-validated module ("View Generated WGSL"). */
  wgsl: string;
  ok: boolean;
  issues: MatIssue[];
  textureCount: number;
  /** The preview sphere as a PNG data-URL (null if compile failed / no GPU). */
  image?: string | null;
}
