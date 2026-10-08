# Hole 8: `MUL_MAT_ID` has no backward

**File:** `ours_hole8_mul-mat-id-backward.diff` (SHA256 `d986107e6ab491b8bd5e5ab02c6b6f8629fe85702205f015374b7becea23907d`)

## Credit first

**srossitto79 did this first.** Their pull request #22704 ("Feat/backward mul mat", opened 2026-05-05, closed unmerged on 2026-09-24)
adds the same op under the same name, `GGML_OP_OUT_PROD_ID`, with the same function, `ggml_out_prod_id(ctx, a, b, ids, n_expert)`, the
same result shape and the same meaning, plus a CPU kernel and a CUDA kernel. They built MoE training work on it in #22705 (closed
2026-09-24). Names, signature and meaning were read from the diff of #22704 on 2026-10-08; the kernels were not compared line by line.
The priority and the design are theirs. #22704 is also the earliest report of the accumulator defect (see the companion repository).

This patch is a smaller version: a CPU kernel only, no CUDA kernel, and a gradient test that makes both inputs parameters. Its op name
and signature match #22704's. Whether it is useful next to a larger pull request that was closed unmerged is for upstream to say.

## What breaks

Fine-tuning any mixture-of-experts model aborts at the routed matmul:
`ggml_compute_backward: unsupported ggml op for backward pass: MUL_MAT_ID`.

## The patch

It adds one op, `GGML_OP_OUT_PROD_ID`, appended last so that no existing op number moves. It is the gradient for the expert matrices:
a sum of outer products over the (slot, token) pairs that chose each expert, and exactly zero for an expert nobody chose. It adds the
`MUL_MAT_ID` case to `ggml_compute_backward` (expert weights through `OUT_PROD_ID`, activations through `ggml_mul_mat_id` of the
transposed experts), the CPU kernel, and `RPC_PROTO_PATCH_VERSION` 0 to 1 with the op-count assert 101 to 102. There is no CUDA
kernel; with the CUDA backend the op runs on the CPU, and `test-backend-ops` reports the expert-weight cases `not supported [CUDA0]`.

The new test `test_mul_mat_id_grad` makes both inputs parameters and picks ids so that every expert is used (an unused expert has a
zero gradient and the error check would give NaN). Its `grad_eps` is 1.0, because the output is linear in each parameter: at the
default 0.1 it failed 5 of 600 case runs on float rounding, at 1.0 it failed 0 of 600 (round 1, 2026-09-24).

## Where it applies

- Alone to `8212c7802`.
- **Not as is to master `71ad0590f4`:** 1 hunk is rejected, in `ggml/include/ggml-rpc.h` (the RPC protocol version there is now at
  major version 8). The other hunks apply. A rebase is needed and would have to be measured again.
- Holes 16 and 17 need this patch.

## Test

```
git apply --include='tests/*' ours_hole8_mul-mat-id-backward.diff     # BEFORE: the new cases, no backward
build/bin/test-backend-ops grad -o MUL_MAT_ID -b CPU
# for AFTER, start from a fresh tree, apply the whole patch and build again
git apply ours_hole8_mul-mat-id-backward.diff
build/bin/test-backend-ops grad -o MUL_MAT_ID -b CPU
```

## Proof (one-tree run, Linux CPU, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| exit | 134 | 0 |
| result | `ggml.c:7238: ggml_compute_backward: unsupported ggml op for backward pass: MUL_MAT_ID`; the last header printed is `MUL_MAT_ID(n_mats=4,n_used=2,b=0,m=6,n=8,k=8)` and no verdict line follows | the 6 verdict lines `SUM(n_mats=4,n_used=2,b=0,...)` to `SUM(n_mats=4,n_used=4,b=1,...)` all read OK |

Planted control (round 2, 2026-09-24, Windows CPU, an earlier base): with the kernel deliberately broken to drop one (slot, token)
pair per expert, all 6 cases failed (`MAA = 0.416 .. 1.295 > 0.0001`), so the check can see a wrong gradient. The run without the
plant read `20371/20371` on that base.

## Not verified

- The `ggml-rpc.h` hunk was never compiled (RPC was not enabled in any build).
- No CUDA kernel exists, so no CUDA run of the op itself was made.
- Verified at the op level with gradient checks. No end-to-end fine-tune of a mixture-of-experts model is reported here.
- Linux CPU for the proof run, plus the earlier Windows CPU run named above.
