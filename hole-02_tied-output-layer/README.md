# Hole 2: a tied output layer is never trained

**File:** `ours_hole2_train-tied-output-when-same-tensor.diff` (SHA256 `6e99326d2dc037eb0e156cd51da58f2ccddf7dec1764a321f33d3822d5a19596`)

## What breaks

`llama_set_param` in `src/llama-context.cpp` returns early for any tensor named `token_embd.weight`, next to a commented-out
`//llama_set_param(model->tok_embd, ...) // FIXME`. Both came in with the training support itself (#10544, 2025-05-12, read from
`git blame`). The commit gives no reason, and the review thread was not read. `tok_embd` is never passed to `llama_set_param`, so
the name check can only ever catch the output layer when it is tied to the token embedding. The effect: on a tied model,
`llama-finetune` trains everything except the output layer, and prints nothing about it.

## The patch

It trains the tied tensor when the output layer is the token embedding tensor itself (`model->output == model->tok_embd`). When the
loader has made a separate copy of the embedding for a different buffer type, training the copy would untie the two and the saved
model would not be the trained one, so that case is refused with one warning line.

## Needs, and where it applies

- To train this model at all on base `8212c7802`, the tree needs upstream #28520 (K and V gradient through the KV cache; merged
  2026-09-30). Every arm of the one-tree run had it.
- Applies alone to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

The tied model: the companion repository's `qwen05_f32.gguf` (SHA256 `89ac62425c46039797e260ad4789adeda174320aa400e8b4b0d4a5e9aaa4ac25`,
see its README) with the tensor `output.weight` left out, which makes llama.cpp tie the output layer to `token_embd.weight`.
`../instruments/make_tied_copy.py` does that. The file the runs used was SHA256
`10b105fdc755ae8d91f4180c3bd429f276f82976f0c900687425e5e1dd70535a`, 290 tensors. The training text is `../data/tied_small.txt` (SHA256
`40174901bf3aa2653f60906175580e725c7c3fdb2dadeffee0a00435a8d55796`).

```
llama-finetune -m qwen05_f32_tied.gguf -ngl 0 -t 10 -epochs 1 -lr 1e-5 -f tied_small.txt -c 512 -b 512 -ub 512 -val-split 0 -o out.gguf
python3 ../instruments/gguf_diff_stdlib.py qwen05_f32_tied.gguf out.gguf
```

BEFORE is the tree without this patch, AFTER is the tree with it. The diff tool compares the raw bytes of every tensor by name. Run it
on the input against itself first: every line must say `bytes_identical=True`.

## Proof (one-tree run, Linux CPU, 2026-09-25)

| | BEFORE | AFTER |
|---|---|---|
| exit | 0, no warning | 0 |
| last train line | `loss=2.06887±0.17629` | `loss=2.06443±0.18074` |
| `token_embd.weight` | `bytes_identical=True` | `bytes_identical=False` |
| every other kind of tensor | changed in 24 of 24 layers | changed in 24 of 24 layers |
| tensors in the output file | 290 | 290 |
| non-finite values in the output | 0 | 0 of 494,032,768 |

The non-finite count was made by a script written during the run, which is not one of the tools fixed in the plan written before it.
It was checked on a file holding one NaN and one infinity first, and it counted 2. It is not in this repository.

Round 1 (2026-09-24, Linux CPU, not repeated in the one-tree run): window 1 read `loss=2.24516` in both arms (the same forward pass
on the same weights). Peak resident memory was 9,392,376 kB before and 11,521,104 kB after.
With the tied copy on the card (`-c 256 -b 256 -ub 256 -ngl 1`, which puts a copy of the embedding on the card) the refusal warning
printed, the losses matched the BEFORE run at all 6 windows, and `token_embd.weight` stayed byte-identical.

## Not verified

- The same-object case in a GPU build. `-ngl 0` on the CUDA build ran out of memory on the 8 GB card.
- One machine, one model, 2 optimizer steps. Nothing is claimed about the quality of a trained model.
