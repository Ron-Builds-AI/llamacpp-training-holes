# Hole 11: `test-backend-ops grad` passes a gradient check whose error is NaN

**File:** `ours_hole11_grad-maa-nan-fails.diff` (SHA256 `5380bd644983140b08cf723b83ae5d65c70d9e57d2fe982526d04afa88551830`). It changes `tests/test-backend-ops.cpp` only.

## What breaks

The error metric `mean_abs_asymm` computes `(a - b) / (a + b)` for every pair. One pair that is exactly 0 in both makes the mean NaN,
and `err > max_maa_err()` is false for a NaN, so the check passes. In round 3 (2026-09-24, Linux CPU), 34.3 percent of the gradient
checks in a sweep (235 of 689 parameter checks) compared NaN and passed without comparing anything.

## The patch

It skips pairs that are 0 in both (they agree), counts the pairs it compared, fails a check that compared nothing (unless the test
declares the gradient to be exactly 0 everywhere), and writes the bound as `!(err <= max)` so that a NaN fails.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08). Hole 15 needs it.

## Test

The instrument has to be shown to see both outcomes. Hole 4's broken CUDA kernel is the specimen: the test of hole 4 with the old
kernel, with and without this patch (Linux, CUDA build):

```
build/bin/test-backend-ops grad -o RMS_NORM -p inplace=0 -b CUDA0
```

## Proof (one-tree run, Linux, RTX 2070 Max-Q, 2026-09-25)

| tree | exit | result |
|---|---|---|
| hole 4's test with the old kernel, without this patch | 0 | `1077/1077`, `Backend CUDA0: OK`, 30 of 30 verdict lines OK including the 15 `v=1` cases, 0 FAIL. Blind: the broken kernel passes. |
| the same, with this patch | 1 | `1062/1077`; 30 FAIL lines, all on the 15 `v=1` cases (2 lines each), for example `SUM(type=f32,ne=[64,5,4,3],v=1,eps=0.000000,inplace=0): [SUM] MAA = 0.142407890 > 0.000100000 FAIL`; 0 on `v=0`, whose 15 verdicts read OK |

With hole 4's kernel fix as well, the same command exited 0 in 3 of 3 runs in the one-tree run (round 3: 13 of 14 runs, the one failing run
failed one case at 1.18e-4 against the 1e-4 bound).

## The cost: this patch turns grad mode red on tests that were passing without checking anything

Round 3 (2026-09-24, Linux CPU, not repeated in the one-tree run): with this patch 117 more cases fail on the CPU (103 of them in 5 of
5 runs). By every measurement made then, none is a wrong backward. Each is a test that compares nothing (a constant or non-finite
loss) or a finite difference too coarse for the gradient, and the backward matched a double-precision closed form wherever one was
computed.

| op | new failures (in 5 of 5 runs) | what they are |
|---|---|---|
| SOFT_MAX sinks=0 | 24 (24) | a constant loss: each row sums to 1 (hole 11b fixes these) |
| POOL_2D max | 60 (60) | 28 sum a -inf loss (a window entirely in padding); 32 hit ties at the max, which a finite difference cannot handle |
| MUL_MAT f16 | 8 (8) | the f16 precision failure master already shows on every f16 check that compares anything |
| MUL_MAT f32 | 9 (0) | intermittent, the same family as master's own f32 failures |
| CPY | 3 (0) | intermittent, the same family as master's own |
| DIAG_MASK_INF | 3 (3) | a -inf loss |
| SILU, SOFTPLUS, SWIGLU | 2, 2, 2 | the finite difference is too coarse over inputs in [-150, 150] |
| RELU | 2 (0) | a finite difference straddling the kink |
| CROSS_ENTROPY_LOSS | 1 (1) | the finite difference is too coarse (step 1.0) |
| MUL_MAT_HADAMARD | 1 (1) | 99 percent of the true gradient is exactly 0 |

With holes 10, 11, 11b, 12 and 15 in the tree, a whole `grad -b CPU` run, no filter, reads `20193/20372` on the Linux machine (179
failing cases in 13 ops) and `20203/20372` on the Windows machine (169 failing cases). See `../hole-12_grad-no-params-for-ops-without-backward`.

## Not verified

One card for the CUDA rows. The reading that none of the 117 cases is a wrong backward rests on round 3's measurements and was not
re-run.
