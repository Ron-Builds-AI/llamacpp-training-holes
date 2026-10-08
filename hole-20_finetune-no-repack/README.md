# Hole 20: `llama-finetune` loads quantized weights repacked, and training aborts in the scheduler

**File:** `ours_hole20_finetune-loads-the-weights-without-repacking.diff` (SHA256 `a8b1d422fe36bd85da2edeb6bc0ce5f0964c56881689d03b7421a262041b16da`), one hunk in `examples/training/finetune.cpp`

## What breaks

llama.cpp's CPU backend can load quantized weights in a repacked layout, through its extra buffer types, and does so unless
`no_extra_bufts` is set. The field is in `common_params` (`common/common.h:582` at `8212c7802`, "disable extra buffer types (used for
weight repacking)", default false), and `common/common.cpp:1693` turns it into `use_extra_bufts` for the loader. As I understand it the
layout is there to make the matrix kernels faster; that was not traced here. `llama-finetune` leaves the field at its default. On a
quantized file the training graph then aborts in the scheduler before the first step:

```
ggml-backend.cpp:1294: GGML_ASSERT(*cur_backend_id != -1) failed
```

The trace runs through `ggml_backend_sched_split_graph`, `ggml_backend_sched_alloc_graph`, `ggml_opt_alloc` and
`llama_context::opt_epoch_iter`, and the process ends on signal 6 (exit 134). The assert says the scheduler found no backend for some
node. I did not identify the node. That the repacked layout is the cause is inferred. What was measured is that two trees which differ
only in this patch (one file, four lines) split the outcome: one aborts, the other runs the same file to the end.

An F32 file has no repacked layout, so this hole should not touch it (inferred). That is why F32 is the control below.

## The patch

`finetune` sets `params.no_extra_bufts = true` before the model loads, and logs one line when it does:
`main: disabling weight repacking: the backward pass reads the weights in their plain layout`. The fix is that one setting and its log
line. With the guard and the closing brace the hunk adds four lines. It touches only `examples/training/finetune.cpp`.

The cost is speed: with the plain layout the quantized arms below took 25 to 27 seconds per step against 3 to 3.4 seconds for F32, about
8 times as long, at the same 10 threads. I did not run the quantized file with repacking on except in the run that aborted, so how much
of that factor is the plain layout is not known. The cause was not traced.

## Needs, and where it applies

- No other patch in this repository is needed for it to apply. To train this model at all on `8212c7802` the tree needs upstream #28520
  (see hole 2), so the runs below put this patch last, after every patch in "Applying them together" in the top README.
- On the run tree, `git apply` took it cleanly (the run's apply log, 2026-10-08).
- A second check, made for this folder: the old side of the hunk, seven lines, appears once and verbatim in `examples/training/finetune.cpp`
  at `8212c7802` (line 37; the hunk header says 62, which I infer is the line number in the tree it was made in). GNU `patch -p1 --dry-run` accepts
  it there with an offset of 25 lines, and also on the same file after every patch in the set. Those two dry runs used `patch`, not
  `git apply`, and `git apply --check` was not run for this patch by me.
- Master `71ad0590f4` (2026-10-08): the copy of `finetune.cpp` kept from the earlier apply checks of that date is byte-identical to the
  one at `8212c7802` (SHA256 `1bd5d4979ea4463f4bddfd962f31af658a4becb0a10cfd5e28fb43df91f29312`), so the same dry run holds there.
  Nothing was built or run against master.
- I did not search upstream for an existing report or pull request about this hole. If another author has one, this hole belongs under
  [Confirmations](../confirmations/README.md) as holes 1, 3, 13 and 14 do.

## What the tested tree was

- **Base:** upstream master `8212c7802455255460ab8e18fc34754560031b34` (2026-09-24).
- **Tree:** the base with the patches of "Applying them together" in that order (read from the run's build record, not diffed against that
  list again), and one more local change to `finetune.cpp` that is not in this repository and is not part of the fix: it limits training to
  49 F32 tensors, 43,904 of the model's 494,032,768 elements. Every other tensor, including every quantized one, is frozen. The same
  local change was in both the BEFORE and the AFTER tree.
- **BEFORE** is that tree without this patch. **AFTER** is that tree with it. `diff -rq` between the two prints one line, the one file.
- **Machine:** Linux, with the same CPU model, OS and compiler as the top README's Linux machine: Intel i7-10750H (6 cores, 12 threads,
  AVX2 and FMA, no AVX512), 15 GB RAM, Ubuntu 24.04, g++ 13.3.0, cmake 3.28.3. CPU only, `-ngl 0 -t 10`. `system_info` in every run printed
  `AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1`. The `REPACK = 1` is printed with the patch too, so it lists what
  the build can do and not what the load used (inferred). The patch's own log line is the check that it took effect.
- **Build:** the Linux CPU line in the top README, with `cmake --build build --target llama-finetune llama-quantize -j 10`.

## Test

Models and text, all as in the top README and `../data`:

- F32: the tied Qwen2.5-0.5B-Instruct file of hole 2 (`../instruments/make_tied_copy.py`), SHA256
  `10b105fdc755ae8d91f4180c3bd429f276f82976f0c900687425e5e1dd70535a`. That it is the unmodified public model is inferred from its header
  (`qwen2.5-0.5b-instruct`, 24 blocks, embedding 896, 290 tensors) and parameter count. It was not compared byte for byte with the
  Hugging Face original.
- Q4_K_M, made from that file by the AFTER tree's quantizer with no importance matrix: 397,807,424 bytes, SHA256
  `3a07a8f302da9d432ff04dc98616c57987ba08492fb130863b369bae5506f178`. Its census (measured): 290 tensors, 121 F32, 132 Q5_0, 13 Q8_0, 12 Q4_K,
  12 Q6_K. So this standard Q4_K_M of this model is mostly Q5_0, and only 12 of its tensors are Q4_K. It is not an all-4-bit file.
- Text: `../data/tied_small.txt`, 1,090 tokens, six windows per epoch at context 256.

The `..` paths are this repository's folders. Adjust them to where you keep it.

```
build/bin/llama-quantize qwen05_f32_tied.gguf qwen05_Q4_K_M.gguf Q4_K_M 10
build/bin/llama-finetune -m MODEL -f ../data/tied_small.txt -ngl 0 -t 10 -fa off -c 256 -b 256 -ub 256 -epochs N -lr LR -val-split 0 -o out.gguf
python3 ../instruments/gguf_diff_stdlib.py MODEL out.gguf
```

Optimizer left at the default (AdamW, weight decay 0). One process at a time. The five arms:

| arm | binary | MODEL | N | LR |
|---|---|---|---|---|
| B | BEFORE | Q4_K_M | 2 | 1e-4 |
| FS | AFTER | F32 | 2 | 1e-4 |
| QS | AFTER | Q4_K_M | 2 | 1e-4 |
| LF | AFTER | F32 | 8 | 1e-3 |
| LQ | AFTER | Q4_K_M | 8 | 1e-3 |

Every run above also carried the local change that limits training to 49 F32 tensors. The upstream default marks every tensor as
trainable. I did not run the quantized file with that default, so I do not know whether B aborts at the same line without the local
change. Anyone reproducing this needs their own way to train a small set of F32 tensors on a quantized file, and should say so with the
result.

## Why there are five arms

An earlier check of this fix on another model, whose files are not in this repository, ended with a loss that stayed flat over two
epochs. That left two readings open: the fix hides a second defect on the quantized path, or two epochs at that rate are too little to
show learning. This run was set up to tell them apart on a public model. B shows whether the abort reproduces here. FS and QS are the
small setting. LF is the control: 4 times the epochs and 10 times the rate on F32, to see whether this setup can show learning at all.
LQ is the same large setting on the quantized file.

The settings, the bars and the reading each outcome would give were written down and frozen before any arm ran. The frozen file's SHA256
is `fa62d3f4775e03174f04fec575f56e0886832689c30cca8e935426c51cb1e77b`. The runner recorded that hash at 16:22:11 on 2026-10-08 and the first
arm started at 16:22:16. The file is not in this repository: it carries local paths and refers to work that is not published. The bars:
the control is met if LF's loss falls at least 1 percent from its first epoch to its last, and the quantized file follows F32 if LQ's fall
is at least half of LF's.

## Proof (Linux CPU, 2026-10-08, one run per arm)

"Epoch-final loss" is the loss printed on the last step of an epoch, which I read as that epoch's mean over its six windows (from the
printout, not from the source). The fall is first epoch to last epoch.

| arm | exit | epoch-final loss, first to last | fall |
|---|---|---|---|
| B | 134, signal 6, no loss line printed | none | none |
| FS | 0 | 3.12859 to 3.11796 | 0.34 percent |
| QS | 0 | 3.12544 to 3.11450 | 0.35 percent |
| LF | 0 | 3.10378 to 2.67406 | 13.85 percent |
| LQ | 0 | 3.10117 to 2.67440 | 13.76 percent |

The two eight-epoch series, epoch by epoch:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| LF (F32) | 3.10378 | 3.01095 | 2.93766 | 2.87414 | 2.81736 | 2.76607 | 2.71883 | 2.67406 |
| LQ (Q4_K_M) | 3.10117 | 3.01188 | 2.93744 | 2.87577 | 2.81974 | 2.76356 | 2.71880 | 2.67440 |

At no epoch end do the two differ by more than 0.003 (largest 0.00261, at epoch 0). LQ's fall is 0.994 of LF's. The control is met
(13.85 percent against the 1 percent bar) and the quantized file follows F32 (0.994 against the 0.5 bar). The frozen file names that
outcome BOTH_FALL. The percentages and the differences were recomputed for this README from the run's loss lines, by a script that is
not in this repository.

More that was measured:

- B: 1.7 seconds from start to abort, peak resident memory 0.6 GB. Its stderr has the filter summary line of the local change and then the
  assert, and no step.
- The four AFTER arms: 0 `GGML_ASSERT` lines in stderr, every printed loss finite, the patch's log line printed once, six steps per epoch.
- In each AFTER arm, the output file against its input (`gguf_diff_stdlib.py`, tallied by a script not in this repository): all 49 trained
  tensors changed, and the other 241 tensors byte-identical. In the quantized arms that includes every quantized tensor.
- The first step of epoch 0 reads `3.09453` in both F32 arms and `3.07944` in both quantized arms, as it must for the same weights and the
  same forward pass.
- A one-epoch repeat of FS and QS before these runs (from the build record) gave identical loss lines and byte-identical output files, so
  on this machine the same command repeats exactly. The five runs above were not repeated.

Cost, from the same runs (`/usr/bin/time -v`, and the gap between printed step lines):

| arm | wall time | peak resident | mean seconds per step |
|---|---|---|---|
| FS | 38.6 s | 3.6 GB | 3.0 |
| QS | 4 min 59 s | 1.7 GB | 24.8 |
| LF | 2 min 47 s | 3.6 GB | 3.4 |
| LQ | 21 min 38 s | 1.7 GB | 27.0 |

## What it shows, and what it does not

It shows, on this model, text and CPU: the unpatched build aborts on the Q4_K_M file at that assert, the patched build trains the same
file to the end, and at the larger setting its loss falls as F32's does. So the flat loss of the earlier check is not shown to be a second
hole here. "Not shown at this size" is not "absent".

It does not show:

- That anything learned in a wider sense. There is no validation split (`-val-split 0`), the text is 1,090 synthetic tokens, and each
  epoch reads the same six windows again.
- That the quantized weights train. They are frozen here. The gradient passes through them to 49 small F32 tensors.
- Anything on another CPU (the repack kernels depend on the CPU's features, inferred; AVX512 and ARM were not run), another quantization
  (a pure Q4_0 file was not run), another model, or CUDA, Vulkan or Metal.
- That the abort needs, or does not need, the local change that selects the trainable tensors (see "Test").
- Anything about run-to-run spread. One run per arm.

## Not verified

- A `git apply --check` of the patch on `8212c7802` and on master, by me. Only `patch` dry runs and the run's own `git apply`.
- The cause of the abort beyond the assert text, and the cause of the slower quantized steps.

Drafted and fact-checked with an AI assistant from my own records, measurements and decisions. I read and edited every sentence and I am responsible for its contents.
