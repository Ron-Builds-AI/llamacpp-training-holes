# Confirmations: defects that another author's pull request fixes

These are not claimed here. Each belongs to the author named, and the pull request is the fix. What this folder holds is a
measurement of the defect and of the fix, taken on 2026-09-25 on the Linux machine (CPU, and for holes 13 and 14 the card) against base
`8212c7802`, so that the author and the reviewers have a second data point. No patch of theirs is reproduced here. Links go to the pull requests, and the state of each was
read from GitHub on 2026-10-08.

In the one-tree runs, a BEFORE tree and its AFTER tree differ only in the change named, and both carry the rest of this repository's set
and the other pull requests of the run. "Tests only" means the pull request's `tests/` part alone, so that its new test runs against the
old code.

## Hole 1: CPU `soft_max_back` writes its output over an input it still reads

**#27096 by devYRPauli, "ggml-cpu : fix soft_max_back wrong output when dst aliases src1". Merged 2026-10-02** (`dd4c286f3`).
Linux CPU, `build/bin/test-soft-max-back`, the pull request's own test:

| tree | exit | result |
|---|---|---|
| BEFORE: the pull request's tests only | 1 | `soft_max_back ne=[4,2,1,1] max_abs_err=0.0612500012 FAILED`, `ne=[1024,4,2,1] max_abs_err=0.706823885 FAILED`, `2 cases: 0 passed, 2 failed` |
| AFTER: the whole pull request | 0 | `max_abs_err=0 PASSED`, `max_abs_err=1.45519152e-11 PASSED`, `2 cases: 2 passed, 0 failed` |

## Hole 3: K and V get no gradient through the KV cache

**#28520 by ngxson, "llama: properly handle KV on training". Merged 2026-09-30** (`81ff93ea1`).
Linux CPU. The same `llama-finetune` command in both trees (`-m qwen05_f32_tied.gguf -ngl 0 -t 10 -epochs 1 -lr 1e-5 -f tied_small.txt
-c 512 -b 512 -ub 512 -val-split 0`), then a tensor-by-tensor comparison of the output with the input
(`../instruments/gguf_diff_stdlib.py`). The BEFORE tree also carries the companion repository's `PATCH_node_budget.diff`, which
raises only the graph node budget so that base can run this model at all.

| tree | `attn_k.weight`, `attn_k.bias`, `attn_v.weight`, `attn_v.bias` | `attn_q.*`, `ffn_up.weight` |
|---|---|---|
| BEFORE: without the pull request, with the node budget | changed in 0 of 24 layers each | changed in 24 of 24 |
| AFTER: with the pull request | changed in 24 of 24 layers each | changed in 24 of 24 |

The instrument was shown to see both outcomes: the input compared with itself reads identical in 290 of 290 tensors.

## Hole 6, the first part: a training text shorter than one window

**#27690 by unravel1020, "common : fix unsigned underflow in common_opt_dataset_init (#27648)". Open.** Fixes issue #27648.
Linux CPU, `llama-finetune ... -f short.txt -c 512 -b 512 -ub 512 -val-split 0` with the 184-token text in `../data/short.txt`:

| tree | exit | result |
|---|---|---|
| BEFORE: without #27690 | 134 | `ggml_aligned_malloc: insufficient memory (attempted to allocate 17592186044415.99 MB)`, then `std::bad_alloc` |
| AFTER: with #27690 (without hole 6's follow-up) | 134 | `ggml-opt.cpp:95: GGML_ASSERT(ndata > 0) failed`; the huge allocation is gone |

The follow-up is `../hole-06_short-text-clean-error`.

## Hole 13: CUDA `SUM` on a tensor with gaps

**#28851 by rssndr, "ggml: support non-contiguous SUM (CPU, CUDA)". Open.** The first report is #25568 by edenfunf. Measured on 2026-09-25
with the diff as it stood then (SHA256 `837cb036f721b6c378e16db8447872e2e151b9163644eef2ffdcdc104b3941cd`); the pull request has been
updated since, so these numbers are for that snapshot. Linux, RTX 2070 Max-Q.

- `test-backend-ops grad -o CPY -b CUDA0`, a destination with gaps (`dst_alloc=[128,4,3,1]`): without the pull request, exit 134 on
  `ggml-cuda/sum.cu:30: GGML_ASSERT(ggml_is_contiguously_allocated(src0)) failed`. With it: no `sum.cu` line, the case reads
  `SUM(...): not supported [CUDA0]`, the summary prints (`20323/20378`; the other failures are `CPY`'s known finite-difference ones).
- `test-backend-ops test -o SUM -b CUDA0`: a change to the CUDA half alone (its `supports_op` condition, tried here, not published) stops
  the abort but turns the permuted case `SUM(type=f32,ne=[33,256,1,1],permute=[1,0,2,3])` red: exit 1, `7/8`, `ERR = 0.013217982`. With the
  whole pull request, CPU and CUDA halves: exit 0, `8/8`.
  Reading, inferred from that pairing: the CPU reference sum was the wrong side, which #25568 reported for Release builds. The complete
  fix is the pull request as a whole.

## Hole 14: CUDA `OUT_PROD` with a leading dimension of size 1

**#30142 by edenfunf, "CUDA: fix OUT_PROD leading dimensions for size-1 dimensions". Open, opened 2026-10-08, after the measurement
below.** Linux, RTX 2070 Max-Q, 3 runs each of `test-backend-ops grad -o MUL_MAT -b CUDA0`:

| tree | result |
|---|---|
| BEFORE: no leading-dimension change | exit 134 in 3 of 3: ` ** On entry to SGEMM  parameter number 10 had an illegal value`, then `CUDA error: an unsupported value or parameter was passed to the function` in `cublasSgemm_v2`; the last case printed is `MUL_MAT(type_a=f32,type_b=f32,m=1,n=7,k=2048,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0)` |
| AFTER: a leading-dimension clamp of the same kind as #30142's (tried here, not that pull request's code, not published) | 0 SGEMM lines and no abort in 3 of 3; that case reads OK in 3 of 3. Exit 1 in each, from 6, 7 and 8 other f32 `MUL_MAT` cases failing on MAA (0.000103 to 0.0239), a known finite-difference family |

`test-backend-ops test -o OUT_PROD -b CUDA0` with the clamp: exit 0, `71/71` OK, 0 FAIL (1,344 further cases are `not supported` on the card and
are not compared). #30142 was not run here.

## What is not claimed

Nothing here says anything about these pull requests beyond what the runs above printed. One machine and one card were used, and
only for these commands. The measurement of hole 13 is of an older state of #28851.
