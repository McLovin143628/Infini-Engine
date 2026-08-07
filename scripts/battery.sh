#!/usr/bin/env bash
# The workspace test battery, reported in a way that cannot claim green while red.
#
# # Why this exists
#
# P24.3 shipped three commits and three reports claiming "3598 passed, 0 failed"
# while `inf-dcc`'s `determinism_law` had been failing since `f917426`. The claim
# came from this pipeline:
#
#     cargo test --workspace -j2 --no-fail-fast 2>&1 \
#       | grep -E "^test result" \
#       | awk -F'[ ;]' '{p+=$4; f+=$6} END {print "PASSED:", p, "FAILED:", f}'
#
# Three independent signals were each suppressed, and the failure needed all
# three:
#
#  1. **The awk field index was structurally wrong.** `-F'[ ;]'` treats `; ` as
#     TWO separators, so between them sits an empty field: in
#     `test result: FAILED. 9 passed; 1 failed; …`, `$6` is `""` and the failure
#     count is `$7`. `f+=$6` therefore added zero on every line of every run —
#     "FAILED: 0" was a constant, never a measurement.
#  2. **`grep -E "^test result"` discarded the other evidence** — the `failures:`
#     blocks, the panic text, and cargo's own `error: test failed` line.
#  3. **The pipeline's exit status was awk's**, which is 0 regardless, so a
#     non-zero `cargo test` was invisible to the shell as well.
#
# The rule that falls out, and the reason this script prints what it prints: a
# green claim is evidence only when it quotes the summary lines VERBATIM. So this
# never re-derives a total — it shows every `test result:` line as cargo wrote it,
# then reports cargo's own exit status, and any aggregate is clearly marked as a
# convenience computed from those same lines.
set -o pipefail

cd "$(dirname "$0")/.." || exit 2
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

cargo test --workspace -j2 --no-fail-fast "$@" 2>&1 | tee "$OUT"
STATUS=${PIPESTATUS[0]}

echo
echo "──────────── VERBATIM SUMMARY LINES ────────────"
grep -E "^test result:" "$OUT"
echo "────────────────────────────────────────────────"
# Aggregated from the lines above, positionally: `test result: ok. P passed; F
# failed; I ignored; …` splits on whitespace into $4=P, $6=F, $8=I once the
# semicolons are stripped. Stripping them first is what the original got wrong.
tr -d ';' < "$OUT" | awk '
  /^test result:/ { p += $4; f += $6; i += $8; n += 1 }
  END { printf "AGGREGATE over %d binaries: %d passed, %d failed, %d ignored\n", n, p, f, i }'
echo "FAILING BINARIES:"
grep -E "^error: test failed" "$OUT" || echo "  (none)"
echo "cargo test exit status: $STATUS"
exit "$STATUS"
