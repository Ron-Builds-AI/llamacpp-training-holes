# Hole 16: `MUL_MAT_ID` backward through f16 expert weights

**File:** `ours_hole16_mul-mat-id-backward-through-f16-experts.diff` (SHA256 `0946d8b2997fcfca0339ba5cad01be76cb7cc08a28b3af2ae7e87e44daec9178`)

## What breaks

With hole 8 in, the `MUL_MAT_ID` backward asserts that the expert weights are f32 when it builds the gradient of the activations
(`GGML_ASSERT(src0->type == GGML_TYPE_F32)`). The gradient of the activations passes back through the expert weights whether they are
trained or frozen, and f16 experts hit the assert.

## The patch

The assert allows f16 as well as f32, and says so in words (`backward pass for MUL_MAT_ID needs f32 or f16 expert weights`): the
transpose and the routed matmul it uses exist for f32 and f16 weights, and a quantized block cannot be transposed (see hole 17). The
test `test_mul_mat_id_grad` gets a `type_a` argument. With f16 experts only the activations are a parameter, and the inputs are values
on a grid of 1/8 in [-2, 2], which stay exact in f16 together with b + 1 and b - 1, so the finite difference with the step of 1 is
exact.

## Needs, and where it applies

- Hole 8 (`../hole-08_mul-mat-id-backward`). The patch does not apply to a tree without it.
- Applies after hole 8 on `8212c7802`. On master `71ad0590f4` it needs hole 8, which does not apply there as is.

## Test

```
build/bin/test-backend-ops grad -o MUL_MAT_ID -b CPU
build/bin/test-backend-ops grad -o MUL_MAT_ID -b CUDA0       # Linux CUDA build
```

BEFORE is the tree with hole 8 and only this patch's `tests/` part. AFTER is the tree with the whole patch.

## Proof

| | BEFORE | AFTER |
|---|---|---|
| Linux CPU, 3 runs (2026-09-25) | exit 134 in 3 of 3, `GGML_ASSERT(src0->type == GGML_TYPE_F32) failed`; the last line is the header `MUL_MAT_ID(type_a=f16,n_mats=4,...)`, and the 6 f32 cases before it read OK | exit 0 in 3 of 3, `20378/20378`, 12 of 12 grad cases OK (6 f32, 6 f16), 0 FAIL, 0 `nothing compared` |
| Linux card, RTX 2070 Max-Q, 3 runs | exit 134 in 3 of 3, the same assert | exit 0 in 3 of 3, `20378/20378`: the 6 f16 cases read OK on the card, the 6 f32 cases read `not supported [CUDA0]` (no CUDA kernel exists for the expert weights' own gradient, expected), 0 FAIL |
| Windows CPU, 3 runs (round 4, 2026-09-25) | exit 127 in 3 of 3, the same assert, at the first f16 header | exit 0 in 3 of 3, `20378/20378`, 12 `SUM(...): OK` lines, 0 FAIL |

On the card, the f16 experts' path runs entirely on the GPU.

## Not verified

- A planted wrong gradient that the f16 check must catch was not run. Only the f32 path's check was shown to catch one (hole 8's
  planted control).
- Quantized experts are still refused by an assert, by design of this patch. Hole 17 covers them.
- Vulkan, Metal, SYCL.
