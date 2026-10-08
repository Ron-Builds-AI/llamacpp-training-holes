# Hole 10: `test-backend-ops grad` aborts on in-place cases

**File:** `ours_hole10_grad-inplace-not-supported.diff` (SHA256 `4eb3df66afef571bd4381c830cd5d505fac7196534c57bac5a399a771562b1c8`). It changes `tests/test-backend-ops.cpp` only.

## What breaks

The backward pass does not support in-place ops, and `ggml_build_backward_expand` aborts on one that needs a gradient. Grad mode
builds such cases anyway, so a grad run that meets one dies instead of finishing.

## The patch

A check, using the same rule as the library's own guard, reports those cases as `not supported [in-place OP]`, the way grad mode
already reports non-F32 cases. The library is not changed. The check runs before the size skip.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

```
build/bin/test-backend-ops grad -o SOFT_MAX -p inplace=1
```

BEFORE is the tree without the patch, AFTER is with it. The runs used a CPU-only build.

## Proof (one-tree run, Linux CPU, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| exit | 134 | 0 |
| result | `ggml.c:7479: GGML_ASSERT(!node->view_src || node->op == GGML_OP_CPY || ... || node->op == GGML_OP_TRANSPOSE) failed`, after the header of an `inplace=1` case | 8 lines `not supported [in-place SOFT_MAX]`, `78/78 tests passed` |

Round 3 (2026-09-24, Linux CPU): across all ops there are 44 in-place cases in 6 ops (ROPE 30, SOFT_MAX 8, SET 3, OPT_STEP_SGD 1,
RMS_NORM 1, SCALE 1). 24 of them abort on master when run alone. The other 20 are large ROPE cases that master skips for speed
before they reach the backward build; the patch reports them `not supported` instead, because its check runs before the size skip.

## Not verified

Linux CPU only.
