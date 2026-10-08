# Hole 5: `ggml_opt_grad_acc` crashes when gradients accumulate over more than one step

**File:** `ours_hole5_grad-acc-dynamic-opt-period.diff` (SHA256 `b2a61e04a58e449869b750122426f0d9672bc63226ac1dc16dbe0130a321e2f2`)

## What breaks

On dynamic graphs `gb_opt` is not built on every step, and `ggml_opt_grad_acc` dereferences it. With `opt_period` above 1 the call
reads a null pointer.

## The patch

`ggml_opt_grad_acc` uses `gb_opt` if it exists, else `gb_grad` (both hold the same accumulator tensors), and asserts that one of
them exists. The patch also adds `test_grad_dynamic_opt_period` to `tests/test-opt.cpp`. It is the dynamic-graph test of the
accumulator reset fix with `opt_period = 2`, and it expects the accumulator to read 1, 2, 1, 2, 1, 2.

## Needs, and where it applies

- The test passes only with the accumulator reset in the tree: `PATCH_gradient-accumulator-reset.diff` and
  `PATCH_regression_test.diff` from the companion repository
  [llamacpp-gradient-accumulator-fix](https://github.com/Ron-Builds-AI/llamacpp-gradient-accumulator-fix) (SHA256
  `7d2f65e9ce936995ce136af3c1172994d4253de5acab6cec7b30e6990bc2add5` and
  `2dc682bc9de5f94689a549904b739a0f5a3279b822bce4bd63d8c8ba4f94962f`). Without it, the crash is gone but the test reads
  `1 2 3 4 5 6`, because master never clears the accumulator.
- Stacking it on PR #30170 (closed unmerged 2026-10-08) instead of the companion patch was not tested.
- The patch alone applies to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

```
git apply PATCH_gradient-accumulator-reset.diff PATCH_regression_test.diff        # companion repository
git apply --include='tests/*' ours_hole5_grad-acc-dynamic-opt-period.diff         # BEFORE: the new test, the old code
build/bin/test-opt
# for AFTER, start from a fresh tree with the companion patches, then apply the whole patch and build again
git apply ours_hole5_grad-acc-dynamic-opt-period.diff
build/bin/test-opt
```

## Proof (one-tree run, Linux CPU, 2026-09-25)

| | BEFORE (companion fix plus this patch's test only) | AFTER (companion fix plus the whole patch) |
|---|---|---|
| `test-opt` exit | 139 | 0 |
| output | stdout ends at `test_grad_dynamic(optimizer=adamw): OK`, then the core dump | `test_grad_dynamic_opt_period` OK for adamw and sgd, `120/120` (AdamW), `48/48` (SGD), `2/2 backend*optimizer passed` |

That the crash is inside `test_grad_dynamic_opt_period` is inferred from the output order and the source order. No backtrace was
taken: `gdb` was not installed on the machine, and nothing was installed.

Round 1 (2026-09-24, Windows CPU, an earlier base): the crash read as an access violation, `0xC0000005`, at the third step.

## Not verified

One CPU, one machine for the proof run. The crash location is inferred, as said above.

## Credit

srossitto79 reported the accumulator defect first, in #22704 (2026-05-05). The reset fix this patch depends on is the companion
repository's.
