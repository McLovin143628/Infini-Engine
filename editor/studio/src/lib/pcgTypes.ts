/** Hand-written TS mirrors of the PCG command results (camelCase to match serde).
 *  The doc/node/edit shapes reuse `blueprintTypes.ts` — only these results are
 *  PCG-specific. */

/** A lowering diagnostic, optionally anchored to a node id. */
export interface PcgIssue {
  node?: number | null;
  severity: "error" | "warning";
  message: string;
}

/** The result of `pcg_compile`: lowering diagnostics + a summary of the runtime
 *  document the graph lowers to. */
export interface PcgCompileResult {
  ok: boolean;
  issues: PcgIssue[];
  /** How many rules the lowered document holds (v1: 0 or 1). */
  ruleCount: number;
  /** A short human summary of the lowered document. */
  summary: string;
}

/** The result of `pcg_evaluate`: how many instances landed in the target volume. */
export interface PcgEvaluateResult {
  /** Entity GUID the scatter was written to. */
  entity: string;
  /** How many instances were placed. */
  placed: number;
  issues: PcgIssue[];
  ok: boolean;
}
