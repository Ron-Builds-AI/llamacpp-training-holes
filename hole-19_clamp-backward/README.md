# Hole 19: `CLAMP` has no backward

**Files:**
- `ours_hole19_clamp-backward.diff` (SHA256 `b02daa854b222ebe8c0d378f790946134ef7e717e36a8652f874b5b6ddd1a317`), the `CLAMP` case in `ggml_compute_backward`
- `ours_hole19t_clamp-grad-test.diff` (SHA256 `d74d885062f8324357242f7216a2e2fb4bbe5169e7dbe40f5aef9719dcffe15a`), one line: `ggml_set_param(a)` in `test_clamp`

## What breaks

A graph that contains `ggml_clamp` aborts in the backward build with `unsupported ggml op for backward pass: CLAMP`. Upstream's graphs
use it in several places (read in `src/llama-graph.cpp` at `8212c7802`). One is `build_moe_ffn`: when the expert weights are
normalized, `weights_sum = ggml_clamp(ggml_sum_rows(weights), 6.1e-5, INFINITY)` guards the division that follows (hole 18's `DIV`).
Others are the `f_clamp_kqv` clamps on q, k and v.

## The patch

The gradient passes where the clamp left `x` as it was (`min <= x <= max`, bounds included) and is 0 where it clamped:
`grad * (1 - step(abs(clamped - x)))`, built from existing ops.

## Where it applies

- The fix alone applies to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).
- **The test patch applies to `8212c7802` but not as is to master:** 1 hunk is rejected, in `test_clamp`. Upstream #29517 (merged
  2026-10-06, CLAMP on non-contiguous views, a forward-only change) edited that struct. A rebase is needed and would have to be
  measured again.

## Test

```
build/bin/test-backend-ops grad -o CLAMP -b CPU
```

BEFORE is the tree with the test patch and without the fix. AFTER adds the fix.

## Proof (Windows CPU, Ryzen 9 9950X3D, MSVC 19.51, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| `grad -o CLAMP -b CPU` | exit -1073740791 (0xC0000409, MSVC `abort()`); stderr's only line: `ggml.c:7299: ggml_compute_backward: unsupported ggml op for backward pass: CLAMP`; the last stdout line is the header of the first f32 case, `CLAMP(type=f32,ne=[10,5,4,3],min=-0.500000,max=0.500000): ` | exit 0, 0 FAIL, `20378/20378 tests passed`, stderr empty |

The six `CLAMP` cases in the AFTER run:
- f32 `ne=[10,5,4,3]` and f32 `ne=[7,1,5,3]`: `SUM(...): OK`. They passed in 20 of 20 repeats of the same command.
- f32 `ne=[1024,1024,1,1]`: `skipping large tensors for speed`. The harness skips it for size, so it was never computed.
- the three f16 cases: `not supported [out->type != FP32]`.

So the gradient is checked on two small shapes only, and not at all on the large one. Values exactly at a bound are not checked either,
because the case's expected gradient set skips them.

Planted control (a clamp backward that passes the gradient everywhere; a plant, not in this repository): exit 1, `20376/20378`,
both computed f32 cases FAIL (`[SUM] MAA = 0.533112304 > 0.000100000` and `0.519608072 > 0.000100000`). So the check sees a wrong clamp
gradient, and the OK lines above carry weight.

The whole `grad -b CPU` run with holes 12, 15, 16, 18 and 19 in the tree exits 1 with `20205/20378`, 173 failing cases, in families that
failed before these patches (see hole 12), plus one `DIV` broadcast case that hole 18's `18t3` addresses.

## Not verified

- The Windows CPU only. The Linux machine and the card were not run for holes 18 and 19.
- The large case and the bound values, as said above.
- Vulkan, Metal, SYCL.
