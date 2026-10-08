# Hole 6: a training text shorter than one window, and the error after upstream's fix

This hole has two parts. The first is upstream's #27690 by **unravel1020**, which is theirs and is not reproduced here. The second is
a follow-up on top of it.

## What breaks

When the training text has fewer tokens than one data point needs, the window count in `common_opt_dataset_init` underflows.
#27690 (open, fix for issue #27648) fixes the arithmetic. With #27690 alone, `llama-finetune` still aborts, on an assert. The same
crash was reported in #22040, #15535, #15532 and #21037 (read 2026-09-24).

## The patches

- `ours_hole6_finetune-short-text-clean-error_on-pr27690.diff` (SHA256 `3dfd42d5bcd885fb90a0b90bba5bc8ddf5a8ff35e6e6f43ddeed77a5621f5445`)
  goes on top of #27690. `common_opt_dataset_init` logs one error naming the token counts and returns `nullptr`, and `finetune`
  exits with 1.
- `ours_hole6c_finetune-short-text-clean-error_on-pr27690-and-hole7.diff` (SHA256
  `f8025c6191aa026a2b30727025875e94cf22cd32960ad999cd943d33f0b49a8b`) is the same change on top of #27690 and hole 7. Hole 7 and
  the first patch both rewrite the lines around `common_opt_dataset_init` in `examples/training/finetune.cpp` and conflict in either
  order, so use this one when hole 7 is in. Use one of the two, not both.

## Needs, and where it applies

- #27690 itself. The diff used in the runs is SHA256 `6b4105fee9e82505c0302b8515e310c99c29497c4ce2c9d2f5defbcce591beb0`. Its
  current diff on GitHub, read on 2026-10-08, is byte-identical to it. Fetch it from the pull request page.
- Each patch applies on top of #27690 on `8212c7802` and on master `71ad0590f4`, and the second on top of #27690 and hole 7 (git
  apply on the touched files, 2026-10-08). Neither applies without #27690.

## Test

```
llama-finetune -m qwen05_f32_tied.gguf -ngl 0 -t 10 -epochs 1 -lr 1e-5 -f short.txt -c 512 -b 512 -ub 512 -val-split 0 -o out.gguf
```

`../data/short.txt` is the first 1,000 bytes of the companion repository's `REPRO_window_synthetic.txt` (184 tokens; SHA256
`8e4c2991985c1c0057a14d42338c29872832381b9aa40e52bb089c3b4766e894`). The model is the tied Qwen file described in `../hole-02_tied-output-layer`.
The count of 184 tokens is that model's tokenizer; another model prints another count in the same message (inferred, not run).

## Proof (one-tree run, Linux CPU, 2026-09-25)

| tree | exit | what the run printed |
|---|---|---|
| base without #27690 | 134 | `E ggml_aligned_malloc: insufficient memory (attempted to allocate 17592186044415.99 MB)`, `failed to allocate buffer of size 18446744073709543424`, `terminate called after throwing an instance of 'std::bad_alloc'` |
| plus #27690 | 134 | `ggml/src/ggml-opt.cpp:95: GGML_ASSERT(ndata > 0) failed`, no huge allocation |
| plus #27690 and the second patch (6c, in the tree with hole 7) | 1 | exactly one line: `E common_opt_dataset_init: the training text has 184 tokens, fewer than the 769 needed for one data point (context size 512 + stride 256 + 1), use a longer text or a smaller context size`. No `GGML_ASSERT`, no `bad_alloc`, no `insufficient memory`, no core dump. |

A normal text still trains: `-f tied_small.txt -c 512 -val-split 0.05` exits 0 with finite losses (2.24516, then 2.06443).
`test-opt-dataset` still passes. In the one-tree run only the combined patch (6c) was run on the short text. The first patch, on its
own, was run in round 1 (2026-09-24, Linux CPU) with the same result: exit 1 and the same single error line.

## Not verified

Linux CPU only.

## Credit

unravel1020, #27690 (fix for #27648). Hole 6 is the clean error after it, nothing more.
