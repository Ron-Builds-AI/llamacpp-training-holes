# Hole S: attention sinks get no gradient

**File:** `ours_holeS_softmax-sinks-backward.diff` (SHA256 `a9512933725fde6156eaa5f22c78d0081eb8e74f33c2c631c0ca960689408a75`)

## What breaks

The `SOFT_MAX` case of `ggml_compute_backward` builds the gradient for its input, asserts that the mask needs none, and ignores the
sinks (`src2`). With the sinks as a parameter and no preallocated accumulator, building the backward graph dereferences a null
gradient and crashes. Through `ggml_opt`, where the accumulators exist, the gradient of the sinks silently stays 0. So in
`llama-finetune`, an `attn_sinks` tensor (models such as gpt-oss) is marked trainable and never trained.

## The patch

The sink of a head is one more logit in the normalization of every row of that head, with no output of its own. Per row, with
`p_s = 1 - sum_i y_i`, the gradient is `dL/ds = -p_s * sum_i y_i * dy_i`, summed over every row of the head. The patch builds this
from existing ops only (`MUL`, `SUM_ROWS`, `SCALE` with bias, `PERMUTE`, `CONT`, `REPEAT_BACK`), all of which run on CPU and CUDA.

The test change: grad mode sums the output. With sinks, the gradient of the input is small and its float32 finite difference is too
coarse to check it, so the `sinks=1` cases now check the gradient of the sinks (where the input is small enough to be checked at
all), with a bound of 1e-3. The file already uses 1e-3 for rope, sin, cos and bin_bcast. Measured in round 2: 32 heads of 2 rows reach
2.57e-4.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

```
git apply --include='tests/*' ours_holeS_softmax-sinks-backward.diff   # BEFORE: the new checks, the old backward
build/bin/test-backend-ops grad -o SOFT_MAX -p inplace=0
# for AFTER, start from a fresh tree, apply the whole patch and build again
git apply ours_holeS_softmax-sinks-backward.diff
build/bin/test-backend-ops grad -o SOFT_MAX -p inplace=0
```

The standalone probe `../instruments/probe_sinks_grad.cpp` checks the gradient against a closed form computed in double. Build it
against the tree's own ggml (shared libraries) and run both modes:

```
g++ -O2 -std=c++17 -I ggml/include probe_sinks_grad.cpp -o probe -L build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build/bin
./probe direct     # sinks as parameters, no preallocated accumulators (the test-backend-ops path)
./probe acc        # with accumulators (the ggml_opt and llama-finetune path)
```

## Proof (one-tree run, Linux CPU, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| `grad -o SOFT_MAX -p inplace=0` | exit 139; the last header is the first `sinks=1` case, `SOFT_MAX(type=f32,ne=[16,16,1,1],mask=0,sinks=1,...)` | exit 0, 0 FAIL lines in 10 of 10 runs, `1077/1077` in each |
| probe `acc` | `0/12 cases passed`, exit 1 | `12/12 cases passed` |
| probe `direct` | died, exit 139, after `building backward (no accumulators)...` | `12/12 cases passed` |

Round 2 (2026-09-24, Linux): the probe's 6 shapes by 2 seeds, with mask, scale, ALiBi and 32 heads, agree with the closed form to at
most 2.84e-6 of the largest gradient, and the closed form agrees with a double-precision central difference to 7.0e-8 relative. 20
repeated runs of the grad check: 820 sink checks, the largest error 2.57e-4, none above 1e-3, none NaN. On the card: `grad -o SOFT_MAX
-p inplace=0 -b CUDA0` `1077/1077` and `test -o SOFT_MAX -b CUDA0` `214/214`. In the one-tree run the card read `1077/1077` in 3 of 3.

## Not verified

No end-to-end training of a model with real attention sinks is reported here. This is verified at the op level only, on CPU and CUDA.
