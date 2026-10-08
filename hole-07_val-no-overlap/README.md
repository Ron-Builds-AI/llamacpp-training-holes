# Hole 7: training and validation windows share tokens

**File:** `ours_hole7_finetune-val-no-overlap.diff` (SHA256 `af0627b7fcd881391d7fd8c42938051d0bcee8a03bf3699003936dfb168a0437`)

## What breaks

`llama-finetune` builds its windows from the whole text, then gives the last fraction of the windows to validation. Windows overlap
by the stride (half a context), so the last training window and the first validation window share up to half a context of labels.
The model is scored on tokens it just trained on, and the validation loss reads too good.

## The patch

It splits the tokens first, builds the training windows from the first part and the validation windows from the rest, and calls
`llama_opt_epoch` once for each. A `val_split` outside [0, 1] asserts. No public API changes.

**This changes what users see.** `val_split` now means a fraction of the tokens, and validation needs about `n_ctx + stride + 1`
held-out tokens to exist at all. A small text that used to print a (contaminated) validation line will print a warning and no
validation line.

## Where it applies

Alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08). It conflicts with the first patch of
hole 6 in `finetune.cpp`; `../hole-06_short-text-clean-error` has the combined version.

## Test

```
llama-finetune -m qwen05_f32_tied.gguf -ngl 0 -t 10 -epochs 1 -lr 1e-5 -f tied_small.txt -c 256 -b 256 -ub 256 -val-split 0.4 -o /dev/null
llama-finetune -m qwen05_f32_tied.gguf -ngl 0 -t 10 -epochs 1 -lr 1e-5 -f tied_small.txt -c 256 -b 256 -ub 256 -val-split 0.05 -o /dev/null
```

The text is `../data/tied_small.txt` (1,090 tokens with the Qwen tokenizer, stride 128). BEFORE is the tree without the patch, AFTER with it.
The `±nan` after the first window is the spread of a single sample, printed as nan by the progress bar. That reading is inferred from
it appearing only on window 1 of each run.

## Proof (one-tree run, Linux CPU, 2026-09-25)

| `-c 256` | BEFORE | AFTER |
|---|---|---|
| `-val-split 0.4`, train line | `train: data=0000003/0000003 loss=2.58453±0.25929` | the same line |
| `-val-split 0.4`, val line | `val: data=0000003/0000003 loss=2.21904±0.07583` | `val: data=0000001/0000001 loss=2.32574±nan` |
| `-val-split 0.05` | `train: data=0000005/0000005`, and a val line: `val: data=0000001/0000001 loss=1.72206±nan` | `train: data=0000006/0000006 loss=2.25158±0.19422`, `W main: not enough validation tokens (55), skipping validation`, no `val:` line |

The BEFORE validation loss at 0.4 sits below the training loss, 2.21904 against 2.58453. By the arithmetic the first validation window
shares labels with the last training window (inferred from the window arithmetic, not measured per token). A regression run at
`-c 512 -val-split 0.05` exits 0 with finite losses and prints `not enough validation tokens (55)`.

## Not verified

Linux CPU only, one text, one model. Whether a given user's validation numbers were inflated depends on the size of the text; no
claim about any trained model is made.

## Why an issue first

Hole 7 changes behaviour that users may depend on (the meaning of `val_split`, and small texts that stop printing a validation line).
That is a reason to discuss it before sending it as a pull request.
