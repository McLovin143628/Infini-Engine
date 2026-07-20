/**
 * Tiny fuzzy matcher for the command palette / instant-filter fields.
 * Case-insensitive subsequence match; score prefers earlier, denser, and
 * word-boundary matches. Not a general search engine — just enough for
 * "instant-filter everything" (ROADMAP §4).
 */

export interface FuzzyResult {
  /** Higher is better. */
  score: number;
  /** Matched character indices in the haystack (for highlighting). */
  indices: number[];
}

export function fuzzyMatch(needle: string, haystack: string): FuzzyResult | null {
  if (needle.length === 0) return { score: 0, indices: [] };
  const n = needle.toLowerCase();
  const h = haystack.toLowerCase();
  const indices: number[] = [];
  let hi = 0;
  let score = 0;
  let streak = 0;
  for (let ni = 0; ni < n.length; ni++) {
    const c = n[ni];
    const found = h.indexOf(c, hi);
    if (found === -1) return null;
    // Consecutive matches and word starts score higher; distance penalizes.
    streak = found === hi && indices.length > 0 ? streak + 1 : 0;
    const wordStart = found === 0 || h[found - 1] === " " || h[found - 1] === "." || h[found - 1] === "/";
    score += 10 + streak * 5 + (wordStart ? 15 : 0) - Math.min(10, found - hi);
    indices.push(found);
    hi = found + 1;
  }
  // Shorter haystacks win ties.
  score -= Math.floor(haystack.length / 8);
  return { score, indices };
}

/** Filter + rank `items` by a needle over `key(item)`. Stable for ties. */
export function fuzzyFilter<T>(
  needle: string,
  items: readonly T[],
  key: (item: T) => string,
): T[] {
  if (needle.trim().length === 0) return [...items];
  const scored: Array<{ item: T; score: number; i: number }> = [];
  items.forEach((item, i) => {
    const m = fuzzyMatch(needle.trim(), key(item));
    if (m) scored.push({ item, score: m.score, i });
  });
  scored.sort((a, b) => b.score - a.score || a.i - b.i);
  return scored.map((s) => s.item);
}
