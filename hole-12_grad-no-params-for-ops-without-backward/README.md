# Hole 12: grad mode makes parameters for ops that have no backward

**File:** `ours_hole12_grad-no-params-for-ops-without-a-backward.diff` (SHA256 `adcdd4b5b70841634a45a8ac1fd96d735c8d89a54df85d9e38ce09802ba09920`). It changes `tests/test-backend-ops.cpp` only (15 hunks).

## What breaks

Test cases in 18 ops make a tensor a parameter although ggml has no backward for that op. A `test-backend-ops grad` run with no
filter therefore aborts at the first of them, with a message such as `unsupported glu op for backward pass: REGLU`. The 18 ops, with
their case strings: CEIL 2, CUMSUM 3, DIAG 1, FILL 1, FLOOR 2, GEGLU 2, GEGLU_ERF 2, GEGLU_QUICK 2, IM2COL_3D 2, POOL_1D 216, REGLU 2,
ROUND 2, SOLVE_TRI 4, SWIGLU_CLAMP 4, SWIGLU_OAI 8, TRI 4, TRUNC 2, XIELU 2: 261 in all. For example `test_glu_split` sets both inputs as
parameters for every GLU op, while ggml implements split SWIGLU only.

## The patch

The tests stop making parameters for those ops, with a one-line comment each saying which backward case is missing. `test_glu_split`
keeps its parameters for split SWIGLU. No library code changes. Where a backward is added later, the comment is the place to put the
parameter back.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

```
build/bin/test-backend-ops grad -o REGLU -b CPU          # and each of the other 17 ops
build/bin/test-backend-ops grad -b CPU                   # the whole grad mode, no filter
```

BEFORE is the tree without the patch, AFTER is with it.

## Proof

One-tree run, Linux CPU, 2026-09-25:

| | BEFORE | AFTER |
|---|---|---|
| each of the 18 ops, `grad -o X -b CPU` | exit 134 in 18 of 18, each output with exactly 1 line naming the backward pass (`ggml.c:7279: unsupported glu op for backward pass: REGLU`, `ggml.c:7288: ggml_compute_backward: unsupported ggml op for backward pass: CUMSUM`, `ggml_compute_backward: unsupported unary op for backward pass: CEIL`) | exit 0 in 18 of 18; 0 lines naming a backward pass; every line of the op is a case header or `not supported`; each `20372/20372 tests passed` |
| whole grad mode, no filter, once | | exit 1 after 101 s, no abort, `20193/20372 tests passed` |

The whole run's 179 failing cases (20372 minus 20193) are in 13 ops: POOL_2D 60, CPY 52, MUL_MAT 47, MUL_MAT_HADAMARD 4,
DIAG_MASK_INF 3, EXPM1 2, SILU 2, SOFTPLUS 2, SWIGLU 2, RELU 2, CROSS_ENTROPY_LOSS 1, SCALE 1, SUM 1. All 13 ops were on a list of
known failing families written before the run, and none is outside it. **The whole grad run does not pass. About 170 known cases
still fail**, and this patch makes the run reach them instead of aborting. That they are not caused by the patch is inferred: the
patch only removes parameters.

The same on the Windows CPU (round 4, 2026-09-25, base plus the earlier holes): BEFORE exit 127 in each of the 18 ops, naming the
backward pass; AFTER exit 0 in each, no line naming a backward pass; whole grad mode exit 1, `20203/20372`, 169 failing cases in the same
families, except that RELU did not fail there. The counts move between runs because the inputs are random on every run
(`std::random_device` in `init_tensor_uniform`).

## Not verified

CPU only. The card and the other backends were not run for this patch.
