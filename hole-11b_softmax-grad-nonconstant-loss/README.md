# Hole 11b: the SOFT_MAX gradient check compares a constant

**File:** `ours_hole11b_softmax-grad-nonconstant-loss.diff` (SHA256 `61f35767f3f72260cb393cac604284c40242ed8cb8c8b05c66e553d7a068b3e8`). It changes `tests/test-backend-ops.cpp` only.

## What breaks

Without sinks every row of the `SOFT_MAX` output sums to 1, so the summed loss does not depend on the input, its gradient is 0, and
the check compares rounding noise against 0. Hole 11 makes those 24 cases fail, as it should.

## The patch

It adds a `grad_loss` hook to `test_case`, whose default is the sum, so every other test builds the same loss as before.
`test_soft_max` overrides it for the `sinks=0` cases only: the loss is `sum(out * w)` with `w` alternating +1 and -1, and the step is
`0.1 / scale`, so the logits move by 0.1 at every scale. With sinks the rows do not sum to 1 and the plain sum is kept.

Five variants were measured in round 3 (2026-09-24). This is the only one with no failure anywhere. The first variant used random
weights. The patch's own comment gives the reason for weights of +1 and -1 instead: with random weights some elements sit near the
weighted mean of their row, their gradient is near 0, and the float32 finite difference cannot resolve it.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08). It is only needed with hole 11 in the tree.

## Test

```
build/bin/test-backend-ops grad -o SOFT_MAX -p inplace=0 -b CPU
```

BEFORE is the tree with hole 11 and without this patch. AFTER adds this patch.

## Proof (one-tree run, Linux, 2026-09-25)

| | BEFORE (hole 11 in, this patch out) | AFTER |
|---|---|---|
| CPU, 10 runs each | exit 1 in 10 of 10; in every run 24 of 24 `sinks=0` verdicts FAIL (`[SUM] MAA = 1.000000000 > 0.000100000 FAIL`), 0 OK; the 41 `sinks=1` verdicts OK; `1053/1077 tests passed` | 0 FAIL in 10 of 10 runs; 24 of 24 `sinks=0` and 41 of 41 `sinks=1` verdicts OK in each |
| card, 3 runs | | exit 0, 0 FAIL, `1077/1077` in 3 of 3 |

The worst error in round 3 was 6.6e-5. A flake seen in round 2 (a `sinks=0`, `ne=[4,1,1,1]` case, 2 of 20 runs before this patch) did not
appear in 13 CPU and 3 card runs of the one-tree run. That is a statement about those 16 runs only.

## Not verified

Linux only (CPU and one card).
