# Hole 18: `DIV` and `SUB` backward do not reduce a broadcast `src1`

**Files:**
- `ours_hole18_div-and-sub-backward-reduce-a-broadcast-src1.diff` (SHA256 `bae722726cb57f9e587ec3ced03207fdad311b4f65146f612d360629961b7abe`), the fix in `ggml_compute_backward`
- `ours_hole18t_grad-tests-for-sub-mul-div-with-broadcast.diff` (SHA256 `dc8647de36f0330c4cd360f7a643907fbeaa72f25074ad46686163c02cddb0a7`), turns on the grad checks for a broadcast `src1`
- `ours_hole18t3_div-grad-check-one-sign-per-divisor-element.diff` (SHA256 `bc95b2a336301089c62ab47275df6b68312294fcd56cadbeb45ff10977eac980`), the inputs of the `DIV` grad check. It is a test change, not part of the fix.

## What breaks

When `src1` of a `SUB` or a `DIV` is broadcast, the backward builds a gradient for it with the shape of the output, not of `src1`, and
`ggml_build_backward_expand` aborts on `GGML_ASSERT(!src1_needs_grads || ggml_are_same_shape(src1, cgraph->grads[isrc1]))`. `ADD` and
`MUL` already reduce a broadcast `src1`. The test file said so (`The backward pass supports broadcasting only for GGML_ADD`) and only
checked `ADD`, with equal shapes. In upstream's graphs the case arises in `build_moe_ffn` (`src/llama-graph.cpp`, read at `8212c7802`):
when the expert weights are normalized, `weights` of shape `[n_expert_used, n_tokens]` is divided by their row sum of shape
`[1, n_tokens]`, a broadcast `src1` in a `DIV`.

## The patch

When the two sources differ in shape, the gradient for `src1` goes through `ggml_repeat_back`, as `MUL` does, in the `SUB` and the
`DIV` case. The test patch `18t` lets the grad checks run for `SUB`, `MUL` and `DIV` with a broadcast `src1`, and `18t3` changes the
numerators of the `DIV` grad check (see the finding below).

## Where it applies

Each of the three applies alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08). The order used
in the runs was `18t`, `18`, then `18t3`.

## Test

```
build/bin/test-backend-ops grad -o DIV -b CPU
build/bin/test-backend-ops grad -o SUB -b CPU
build/bin/test-backend-ops grad -o MUL -b CPU      # control
```

BEFORE is the tree with the test patch `18t` and without the fix. AFTER adds the fix.

## Proof (Windows CPU, Ryzen 9 9950X3D, MSVC 19.51, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| `grad -o DIV -b CPU` | exit -1073740791 (0xC0000409, MSVC `abort()`); stderr's only line: `ggml.c:7303: GGML_ASSERT(!src1_needs_grads \|\| ggml_are_same_shape(src1, cgraph->grads[isrc1])) failed`; the last stdout line is the header of the first f32 case with a broadcast, `DIV(type=f32,ne=[1,1,1,1],nr=[32,1,1,1],nf=1,perm1=0,src_overlap=0): ` | exit 0, 0 FAIL, `20378/20378 tests passed`; 7 broadcast f32 cases read OK: `nr=[32,1,1,1]` with `ne=[1,1,1,1]`, and `ne=[10,5,4,3]` with `nr` of `[2,1,1,1]`, `[1,2,1,1]`, `[1,1,2,1]`, `[1,1,1,2]`, `[1,1,2,2]`, `[1,2,2,2]` |
| `grad -o SUB -b CPU` | the same assert, the last line `SUB(type=f32,ne=[1,1,1,1],nr=[32,1,1,1],...): ` | exit 0, 0 FAIL, `20378/20378`, the same 7 shapes OK |
| `grad -o MUL -b CPU` (control) | exit 0, 7 broadcast cases OK | exit 0, 7 broadcast cases OK |

The control matters: because `MUL`'s broadcast checks pass on the tree without the fix, the aborts above come from the missing fix and
not from the harness. The first row's AFTER passed in that one planned run. See the finding below for what the same command did in repeats.

## A finding about the `DIV` check, and the test patch `18t3`

The same AFTER command, repeated 20 times on the same binary, failed in 14 of the 20 runs on one case, `DIV(type=f32,ne=[10,5,4,3],nr=[1,2,2,2],...)`,
with `MAA` from 0.00104 to 0.00168 against the bound of 0.001; the other 6 broadcast shapes passed in all 20. `SUB`, `MUL` and
`CLAMP` passed 20 of 20. The inputs are drawn fresh on every run. The reading is inferred, not measured with a double-precision finite
difference: the divisor's gradient is the textbook derivative reduced the way `MUL`'s is, and the failing case is the largest
broadcast `DIV` case the harness computes (4,800 outputs, inputs drawn from [0.9, 1.1], so the summed loss is near 4,800), where
float32 rounding in the finite difference is of the size seen. The backward itself is unchanged by the test patch.

A first attempt at the inputs, a random sign on every numerator element, cut the failing runs from 11 of 20 to 4 of 20 but did not clear
them, and the remaining failures were larger. It is not in this repository. `18t3` gives the numerator one random sign per divisor
element, shared by all of that element's broadcast copies, so that the loss still sums near zero and no divisor's summed gradient
can cancel. That the cancellation was the second failure mode is inferred, not isolated.

A second run on the Windows CPU, `grad -o DIV -b CPU`, the same session for all four rows:

| arm | measured |
|---|---|
| before: the old test inputs, 20 runs | the named case FAIL in 11 of 20 runs (exit 1), OK in 9 |
| after: `18t3`, 20 runs | exit 0 in 20 of 20, 0 FAIL lines, `20378/20378` in each, the named case OK in 20 of 20 |
| planted wrong divisor gradient (a plant, not in this repository), 3 runs | exit 1 in 3 of 3, 13 `DIV` cases failing in each, the named case among them |
| `SUB`, `MUL`, `CLAMP` on the `18t3` tree | exit 0, `20378/20378` each |

(The 11 of 20 here and the 14 of 20 above are two separate sessions of the same unfixed check.)

## Not verified

- The Windows CPU only. The Linux machine and the card were not run for holes 18 and 19.
- Whether the remaining `DIV` check noise is zero on other machines.
- Vulkan, Metal, SYCL.
