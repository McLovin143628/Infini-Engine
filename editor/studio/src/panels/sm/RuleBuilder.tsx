/**
 * **The condition rule builder** (P29.5, §13's P29.5 clause 1) — an editor for
 * `.inf_sm` v2's condition **trees**.
 *
 * v1's inspector edits a flat AND of float compares, and the overwhelming
 * majority of authored transitions still are exactly that; P29.1 shipped the
 * tree model with the flat view kept beside it so that opening a v2 machine and
 * pressing save could not destroy an `Or`, a `Not`, a trigger or a typed
 * compare the UI could not draw. This is the UI that draws them.
 *
 * ## Fully controlled, like every other canvas in this editor
 *
 * There is **no local mirror of the tree**. Every node is rendered from the
 * `cond` prop and every gesture computes a whole new tree with the pure helpers
 * in `lib/smTypes` and calls `onChange` once. That is the same rule
 * `BlueprintCanvas` follows and the same reason: a `useState` copy of a document
 * plus an effect that syncs it is exactly what `react-hooks/set-state-in-effect`
 * bans, and a stale copy of a condition tree is a transition that fires when it
 * should not.
 *
 * ## The flat view still wins where it applies
 *
 * The builder writes through `smStore.setCondition`, which clears `conditions`
 * as it sets `condition` — because `dto_to_transition` prefers the flat list
 * when it is present, so leaving a stale one beside a new tree would save the
 * list and discard the tree. Converting the other way (a flat list into the
 * builder) is the "Edit as tree" button below, and it is one-way on purpose:
 * a tree that happens to be flat is still saved as a tree, which is the same
 * machine.
 */
import {
  SM_COND_KINDS,
  SM_OPS,
  addTermAt,
  condAt,
  condChildren,
  defaultCond,
  removeCondAt,
  setCondAt,
  type SmCondDto,
  type SmCondKind,
  type SmOp,
  type SmParamDto,
} from "../../lib/smTypes";

interface Props {
  /** The whole tree. The builder never holds a copy of it. */
  cond: SmCondDto;
  /** The machine's declared parameters, for the name picker. Undeclared names
   *  are still legal (they read as a `Float` defaulting to 0 — v1's rule), so
   *  the picker offers a free-text row beside the list rather than restricting
   *  to it. */
  params: SmParamDto[];
  onChange: (next: SmCondDto) => void;
}

/** The id of the shared parameter-name `<datalist>`. One per builder, not one
 *  per node: ids are document-unique and a condition tree is recursive. */
const PARAM_LIST_ID = "sm-param-names";

/** A node and its subtree. Recursive, bounded by the engine's own
 *  `MAX_COND_DEPTH` — a tree past it is refused by `validate` at the save door,
 *  which is where the bound belongs. */
function CondNode({
  root,
  path,
  params,
  onChange,
}: {
  root: SmCondDto;
  path: number[];
  params: SmParamDto[];
  onChange: (next: SmCondDto) => void;
}) {
  const node = condAt(root, path);
  // A plain closure, not a `useCallback`: `path` is `[...path, i]` — a fresh
  // array the parent builds on every render — so the dependency list never
  // matches and the memo has never once been reused (P29.5 audit, nit). The
  // React-compiler lint says the same thing from the other end.
  const replace = (next: SmCondDto) => onChange(setCondAt(root, path, next));
  if (!node) return null;

  const removable = path.length > 0;
  const parent = removable ? condAt(root, path.slice(0, -1)) : null;
  const inGroup = parent?.kind === "and" || parent?.kind === "or";

  const header = (
    <div className="sm-rule__head">
      <select
        className="sm-rule__kind"
        value={node.kind}
        onChange={(e) => replace(defaultCond(e.target.value as SmCondKind))}
        title="What this node is"
      >
        {SM_COND_KINDS.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
      {(node.kind === "and" || node.kind === "or") && (
        <button
          className="bp-btn bp-btn--sm"
          onClick={() => onChange(addTermAt(root, path, defaultCond("compare")))}
          title="Add a term"
        >
          + term
        </button>
      )}
      {removable && inGroup && (
        <button
          className="bp-btn bp-btn--sm"
          onClick={() => onChange(removeCondAt(root, path))}
          title="Remove this node"
        >
          ×
        </button>
      )}
    </div>
  );

  // The `<datalist>` itself is rendered ONCE, by `RuleBuilder` below. A tree is
  // recursive, so emitting it here gave every node in a nested condition the
  // same element id — and `list=` resolves to whichever one the document happens
  // to hold first (P29.5 audit, nit).
  const paramPicker = (value: string, set: (v: string) => void) => (
    <input
      className="sm-rule__param"
      value={value}
      list={PARAM_LIST_ID}
      placeholder="parameter"
      onChange={(e) => set(e.target.value)}
    />
  );

  return (
    <div className={`sm-rule sm-rule--${node.kind}`}>
      {header}
      {node.kind === "compare" && (
        <div className="sm-rule__row">
          {paramPicker(node.param, (param) => replace({ ...node, param }))}
          <select
            value={node.op}
            onChange={(e) => replace({ ...node, op: e.target.value as SmOp })}
          >
            {SM_OPS.map((op) => (
              <option key={op} value={op}>
                {op}
              </option>
            ))}
          </select>
          <input
            className="sm-rule__val"
            type="number"
            step={0.1}
            value={node.value}
            onChange={(e) => replace({ ...node, value: Number(e.target.value) })}
          />
          <select
            value={node.valueKind}
            onChange={(e) =>
              replace({ ...node, valueKind: e.target.value as "bool" | "int" | "float" })
            }
            title="How the operand is read"
          >
            <option value="float">float</option>
            <option value="int">int</option>
            <option value="bool">bool</option>
          </select>
        </div>
      )}
      {node.kind === "trigger" && (
        <div className="sm-rule__row">
          {paramPicker(node.param, (param) => replace({ ...node, param }))}
          <span className="sm-insp__note">consumed by the transition that reads it</span>
        </div>
      )}
      {condChildren(node).length > 0 && (
        <div className="sm-rule__kids">
          {condChildren(node).map((_, i) => (
            <CondNode key={i} root={root} path={[...path, i]} params={params} onChange={onChange} />
          ))}
        </div>
      )}
      {(node.kind === "and" || node.kind === "or") && condChildren(node).length === 0 && (
        <div className="sm-insp__note">
          {node.kind === "and" ? "empty AND — always true" : "empty OR — never true"}
        </div>
      )}
    </div>
  );
}

/** The rule builder over a whole transition condition. */
export function RuleBuilder({ cond, params, onChange }: Props) {
  return (
    <>
      <datalist id={PARAM_LIST_ID}>
        {params.map((p) => (
          <option key={p.name} value={p.name}>
            {p.kind}
          </option>
        ))}
      </datalist>
      <CondNode root={cond} path={[]} params={params} onChange={onChange} />
    </>
  );
}
