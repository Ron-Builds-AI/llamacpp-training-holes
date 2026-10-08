# Hole 9: `ggml_get_rows_back` accepts 2D only

**File:** `ours_hole9_get-rows-back-nd.diff` (SHA256 `1141d5634505f750ae2294f1edf1abec117e6521f2df3ab37d712a15ce66e1cf`)

## What breaks

`ggml_get_rows_back` asserts that its input is a matrix and its index a vector, while `ggml_get_rows` accepts 3D and 4D inputs. Any
graph whose `GET_ROWS` has a 3D input aborts in the backward build. In a mixture-of-experts model that is the router: once the routed
matmul has a backward (hole 8), the router gradient reaches a `GET_ROWS` over `probs [1, n_expert, n_tokens]`.

## The patch

`ggml_get_rows_back` accepts what `ggml_get_rows` accepts. Both CPU kernels walk the index tensor through its strides and assert on an
out-of-range index. The Vulkan `supports_op` for `GET_ROWS_BACK` is restricted to 2D, because its dispatch is 2D (CUDA and SYCL
already restrict it). `test_get_rows` sets its parameter for the 3D and 4D cases too.

## Where it applies

- Alone to `8212c7802`.
- **Not as is to master `71ad0590f4`:** 2 hunks are rejected, both in `ggml/src/ggml-cpu/ops.cpp`. Upstream #29575 by **angt**
  (merged 2026-09-29, "check row bounds in get_rows_back") added `GGML_ASSERT(r >= 0 && r < dst->ne[1]);` to the same two CPU kernels,
  and this patch adds the identical line in both (read from the diff of #29575 on 2026-10-08). On master at that commit, `ggml_get_rows_back` still
  asserts matrix and vector and the kernels still walk the index flat, so the N-dimensional part is not upstream. The rest of the patch
  applies. A rebase is needed and would have to be measured again.

## Test

```
git apply --include='tests/*' ours_hole9_get-rows-back-nd.diff        # BEFORE: the new 3D and 4D cases, the old code
build/bin/test-backend-ops grad -o GET_ROWS -b CPU
# for AFTER, start from a fresh tree, apply the whole patch and build again
git apply ours_hole9_get-rows-back-nd.diff
build/bin/test-backend-ops grad -o GET_ROWS -b CPU
```

## Proof (one-tree run, Linux CPU, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| exit | 134 | 0 |
| result | `ggml.c:4020: GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32) failed`, at the first 3D case, `GET_ROWS(type=f32,n=256,m=5,r=4,be1=7,be2=1,v=0,vs0=0,offset_cols=0)` | the `be1=7` cases that grad mode checks (f32, `vs0=0`, `v=0` and `v=1`) read OK, 2 of 2; the other `be1=7` lines are `not supported` or skipped, and none is FAIL |

Rounds 1 and 3 (2026-09-24, Linux CPU): AFTER `20356/20356`, and the 3D cases pass including the strided index view. On master these
`GET_ROWS` gradient checks pass without comparing anything (hole 11); with hole 11's metric they compare every pair and still pass,
largest error 5.6e-7.

## Not verified

The Vulkan line was not compiled (no Vulkan build). Linux CPU only.
