# Hole 4: CUDA `rms_norm_back` ignores the strides of a non-contiguous input

**File:** `ours_hole4_cuda-rms-norm-back-strides.diff` (SHA256 `e42b3225639c441b6067a2bac6a491169b8737e8b17794193ee1742c1d85a371`)

## What breaks

The CUDA backward kernel for RMS norm reads its input `b` as if it were contiguous. The forward kernel `rms_norm_f32` already takes
the row, channel and sample strides. A non-contiguous `b` gives a wrong gradient, and nothing reports it.

## The patch

`rms_norm_back_f32` gets the same 3D grid and the same three strides that `rms_norm_f32` takes. `supports_op` for `RMS_NORM_BACK`
now also requires `ggml_is_contiguous_rows(src[1])`, the forward's own condition. `test_rms_norm_back` gets the non-contiguous
`v` variant that `test_rms_norm` already has. The patch holds both the fix and the test.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

Linux, CUDA build (see the top-level README for the flags):

```
git apply --include='tests/*' ours_hole4_cuda-rms-norm-back-strides.diff    # BEFORE: the new cases, the old kernel
build/bin/test-backend-ops test -o RMS_NORM_BACK -b CUDA0
# for AFTER, start from a fresh tree, apply the whole patch and build again
git apply ours_hole4_cuda-rms-norm-back-strides.diff
build/bin/test-backend-ops test -o RMS_NORM_BACK -b CUDA0
```

## Proof (one-tree run, Linux, RTX 2070 Max-Q sm_75, 2026-09-25)

| | BEFORE | AFTER (3 runs) |
|---|---|---|
| exit | 1 | 0 in 3 of 3 |
| result | `10/20 tests passed`, `Backend CUDA0: FAIL` | `20/20 tests passed` in each |
| `v=0` cases (contiguous) | 10 of 10 OK | 10 of 10 OK |
| `v=1` cases (non-contiguous) | 10 of 10 FAIL | 10 of 10 OK |

An example BEFORE line: `[RMS_NORM_BACK] ERR = 0.046070045 > 0.000000100   RMS_NORM_BACK(type=f32,ne=[64,5,4,3],v=1,eps=0.000000): FAIL`.
On master the test has no non-contiguous cases, so it reads `10/10`.

The gradient check cannot see this bug on master: `test-backend-ops grad -o RMS_NORM -p inplace=0 -b CUDA0` passes with the broken
kernel, `1077/1077`, and all 30 of its verdict lines read OK. That is hole 11, and it is why hole 11 exists.

## Not verified

One card (sm_75, CUDA 12.0, g++-12 as the host compiler). Vulkan, Metal and SYCL were not built or run.
