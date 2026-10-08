# Fixes for holes in llama.cpp's training path

Patches, tests and measurements for defects in the training path of llama.cpp: `ggml-opt`, the backward pass in `ggml.c`, the CUDA
backward kernels, `llama-finetune`, and the gradient check in `test-backend-ops`. Each fix has a folder with the patch, how to apply
and test it, and the numbers from a run without it and a run with it. The base is upstream master `8212c7802` (2026-09-24). Where a
fix belongs to another author, it is listed under [Confirmations](confirmations/README.md) and credited, and nothing of theirs is
reproduced.

It is the companion to
[llamacpp-gradient-accumulator-fix](https://github.com/Ron-Builds-AI/llamacpp-gradient-accumulator-fix), the fix for the gradient
accumulators that are never cleared between steps (pushed 2026-09-23). srossitto79 reported that defect first, in #22704 on
2026-05-05. Several of the defects below were met while testing that fix, and hole 5 depends on it.

Written 2026-10-08. The upstream states in the Status and Credit sections were read from GitHub on that date.

## Status upstream

- The accumulator fix was opened as #29364 on 2026-09-24 and closed the same day by the maintainer of the ggml training code, not
  merged. The stated reason was review capacity. The closure makes no finding on the patch or its test.
- #30170, a draft by shashb27 opened 2026-10-08, carries a different change for the same accumulator defect, with a new test. Its
  description credits srossitto79 for the first report and Ron-Builds-AI for the fix work. Issue #30083 (joelteply) is open.
- Nothing in the folders below has been proposed upstream as of this date: #29364 is the only pull request from this account in
  ggml-org/llama.cpp, and the account has opened no issues there. Other authors' pull requests that match holes 1, 3, 6, 13 and 14 are listed in
  [Confirmations](confirmations/README.md).

## Base, machines, method

- **Base:** upstream master `8212c7802455255460ab8e18fc34754560031b34` (2026-09-24), unless a row says otherwise.
- **Linux:** Intel i7-10750H, Ubuntu 24.04, gcc 13.3. Its card, an NVIDIA RTX 2070 Max-Q (sm_75, 8 GB), ran the CUDA rows with
  CUDA 12.0 and g++-12 as the host compiler.
- **Windows:** AMD Ryzen 9 9950X3D, Windows 11, Visual Studio 18 2026 (MSVC 19.51), CPU only.
- **One tree per run.** Each proof run applied the whole set to one tree and measured each defect by removing only its own change
  (or, for a test, applying only the `tests/` part, `git apply --include='tests/*'`). BEFORE and AFTER differ only in the change named.
  The proof runs were 2026-09-25: the Linux CPU and card, and the Windows CPU.
- **Builds.** Linux CPU: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF`. Linux CUDA: the same with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12`. Windows: `-DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON`.
- **Source trees were LF.** The runs applied the patches with `git -c core.autocrlf=false apply` to LF trees. These `.diff` files must
  keep LF too, which `.gitattributes` in this repository enforces.
- **Model and text for the `llama-finetune` rows:** Qwen2.5-0.5B-Instruct in F32 (the upcast the companion repository describes), with
  `output.weight` left out so that the output layer is tied to the token embedding (`instruments/make_tied_copy.py`; the file the runs
  used is 290 tensors, SHA256 `10b105fdc755ae8d91f4180c3bd429f276f82976f0c900687425e5e1dd70535a`). The text is in `data/`: `tied_small.txt`
  (the first 45 lines of the companion repository's `REPRO_window_synthetic.txt`, 1,090 tokens) and `short.txt` (its first 1,000 bytes,
  184 tokens). Synthetic, no third-party text.
- **Reading the counts.** `test-backend-ops` prints `N/M tests passed` over the whole case list, and it counts cases that an `-o` filter
  skipped as passed. `M` is the size of the list, not the number of gradient checks run. In grad mode, read the result line of a case
  (named after the loss node, usually `SUM(...)`), never its header; see hole 15.
- **Marks.** A statement marked inferred was not measured. A figure marked with a round (round 1, 2, 3 or 4) comes from an earlier per-hole
  run on 2026-09-24 or 2026-09-25 and was not repeated in the one-tree run.

Raw run logs are not in this repository. They carry local paths and machine addresses. The numbers here were copied from the run records.

## The list

First recorded: holes 1 to 15 and S on 2026-09-24, holes 16 to 19 on 2026-09-25. Several of 1 to 9 were met while testing the
accumulator fix, which was published on 2026-09-23. Every fix below was proven on 2026-09-25.

| # | What breaks on base | Folder | Tested on |
|---|---|---|---|
| 1 | CPU `soft_max_back` writes its output over an input it still reads | [confirmations](confirmations/README.md), #27096 (devYRPauli), merged 2026-10-02 | Linux CPU |
| 2 | a tied output layer is never trained, silently | [hole-02](hole-02_tied-output-layer) | Linux CPU |
| 3 | K and V get no gradient through the KV cache | [confirmations](confirmations/README.md), #28520 (ngxson), merged 2026-09-30 | Linux CPU |
| 4 | CUDA `rms_norm_back` ignores the strides of a non-contiguous input | [hole-04](hole-04_cuda-rms-norm-back-strides) | Linux card |
| 5 | `ggml_opt_grad_acc` crashes when gradients accumulate over more than one step | [hole-05](hole-05_grad-acc-dynamic-opt-period) | Linux CPU |
| 6 | `llama-finetune` on a text shorter than one window tries to allocate 16 EiB; after #27690 (unravel1020) it aborts on an assert | [hole-06](hole-06_short-text-clean-error), follow-up on #27690 | Linux CPU |
| 7 | training and validation windows share tokens, so the validation loss is too good | [hole-07](hole-07_val-no-overlap) | Linux CPU |
| 8 | `MUL_MAT_ID` has no backward, so fine-tuning a mixture-of-experts model aborts. srossitto79's #22704 did this first | [hole-08](hole-08_mul-mat-id-backward) | Linux CPU |
| 9 | `ggml_get_rows_back` accepts only 2D, and a mixture-of-experts router reaches it once hole 8 is in | [hole-09](hole-09_get-rows-back-nd) | Linux CPU |
| S | attention sinks get no gradient: a crash without accumulators, a silent zero with them | [hole-S](hole-S_attention-sinks-backward) | Linux CPU |
| 10 | `test-backend-ops grad` aborts on in-place cases | [hole-10](hole-10_grad-inplace-not-supported) | Linux CPU |
| 11 | `test-backend-ops grad` passes a gradient check whose error is NaN | [hole-11](hole-11_grad-nan-fails) | Linux card |
| 11b | the `SOFT_MAX` gradient check compares a constant | [hole-11b](hole-11b_softmax-grad-nonconstant-loss) | Linux CPU and card |
| 12 | grad mode makes parameters for 18 ops that have no backward, so a whole grad run aborts | [hole-12](hole-12_grad-no-params-for-ops-without-backward) | Linux CPU, Windows CPU |
| 13 | CUDA `SUM` asserts on a tensor with gaps | [confirmations](confirmations/README.md), #28851 (rssndr), open | Linux card |
| 14 | CUDA `OUT_PROD` with a size-1 leading dimension aborts in cuBLAS | [confirmations](confirmations/README.md), #30142 (edenfunf), open | Linux card |
| 15 | grad mode prints `OK` for a case before it checks it | [hole-15](hole-15_grad-case-header-no-status) | Linux CPU, Windows CPU |
| 16 | `MUL_MAT_ID` backward through f16 expert weights asserts | [hole-16](hole-16_mul-mat-id-backward-f16-experts) | Linux CPU and card, Windows CPU |
| 17 | `MUL_MAT_ID` backward through quantized expert weights asserts | [hole-17](hole-17_mul-mat-id-backward-quantized-experts) | Windows CPU |
| 18 | `DIV` and `SUB` backward do not reduce a broadcast `src1`, so the backward build aborts | [hole-18](hole-18_div-sub-backward-broadcast) | Windows CPU |
| 19 | `CLAMP` has no backward | [hole-19](hole-19_clamp-backward) | Windows CPU |

Each folder's own README has the exact commands and the printed numbers. Holes 8, 9 and 16 to 19 all sit on the path that a
mixture-of-experts graph takes through the backward pass. In upstream's `build_moe_ffn` (read at `8212c7802`), the experts go through
`MUL_MAT_ID` (holes 8, 16, 17), the router reads rows of a 3D tensor (hole 9), and the expert weights are divided by a clamped row sum
(holes 18 and 19).

## Where each patch applies

Checked 2026-10-08 with `git apply --check` on the files each patch touches, taken from `8212c7802` and from master
`71ad0590f4808b6202f9213d166913858c73b1bc` (2026-10-08 16:34 UTC). Nothing was built against master.

| | on `8212c7802` | on master `71ad0590f4` |
|---|---|---|
| 2, 4, 5, 7, S, 10, 11, 11b, 12, 18, 18t, 18t3, 19, 17t | applies alone | applies alone |
| 15 | needs 11 | needs 11 |
| 6 | needs #27690 | needs #27690, then applies |
| 6c | needs #27690 and 7 | needs #27690 and 7, then applies |
| 8 | applies alone | **does not apply**: 1 hunk rejected, in `ggml-rpc.h` (the protocol version there moved) |
| 9 | applies alone | **does not apply**: 2 hunks rejected, in `ggml-cpu/ops.cpp`, where #29575 (angt, merged 2026-09-29) added the same bounds assert |
| 16, 17 | need 8 (16 first, then 17) | need 8, which does not apply as is |
| 19t | applies alone | **does not apply**: 1 hunk rejected, in `test_clamp`, which #29517 (merged 2026-10-06) edited |

The patches are proven against `8212c7802` as written. Any rebase changes the bytes and would need its own measurement, which has not
been made. Of #27096, #28520 and #28851, the first two are now in master.

## Applying them together

In the order the proof runs used, every patch applies to `8212c7802` with line offsets only (checked 2026-10-08, same method as above):
the companion repository's two patches, #27096, #27690, #28520, then holes 2, 4, 5, 7, 6c, 8, 9, S, 10, 11, 11b, 12, 15, #28851 (hole
13), hole 14's change (a CUDA file, not in this repository; nothing depends on it), 16, then the tests 18t and 19t, then the fixes 18 and
19, then 18t3, then 17 and 17t.

Dependencies: hole 5's test needs the accumulator reset fix. Hole 6's patches go on top of #27690. Holes 6 and 7 conflict in
`finetune.cpp` in either order, so with 7 in the tree use 6c. Use one of 6 or 6c. Hole 15 needs 11. Holes 16 and 17 need 8.

## Found, not fixed

Round 1 (2026-09-24), not repeated in the one-tree run:

- **Graph size.** On base, `llama-finetune` aborts on this 24-layer model before its first step:
  `ggml.c:7291: GGML_ASSERT(cgraph->n_nodes < cgraph->size) failed`, on both machines. #28520 lifts it only when `n_ubatch == n_ctx`.
  `-c` is padded up to a multiple of 256 (`src/llama-context.cpp:289`), so `-c 128 -ub 128` runs with `n_ctx` 256 and aborts again.
- **Out of memory while training** aborts with `GGML_ASSERT(buf != NULL && "tensor buffer not set")`, not a clean error.
- **`SCALE` grad** fails on base CPU in 5 of 5 runs, on `SCALE(ne=[10,10,10,10],scale=2,bias=1)`, MAA 4.9e-4. It is one of the known cases
  in the whole grad run (hole 12).
- **The window count** (`(n_tokens - n_ctx - 1) / stride`, on base and in #27690) leaves out the last window that fits. By the arithmetic it
  should be `floor((n_tokens - 1 - n_ctx) / stride) + 1` (inferred, not run). Changing it would move every window count and the
  train and validation split, so it is in no patch.

## What is not claimed

- No claim is made about the quality of any trained model. These are fixes to what trains, what crashes, and what the tests see.
- Holes 8, 9, S, 16 and 17 are verified with gradient checks at the op level. No end-to-end fine-tune is reported in this repository.
  Hole 17's test shows agreement with the dequantized weights, which is not a check against the quantized model's own loss.
- Hole 4 was run on one card (sm_75). Vulkan, Metal and SYCL were not built or run for any patch. Holes 17, 18 and 19 were run on the
  Windows CPU only.
- Hole 8's RPC hunk, hole 9's Vulkan line and hole 8's missing CUDA kernel are the parts no build reached.
- Nothing is claimed about upstream acceptance, and nothing about other authors' pull requests beyond what a run printed.
- This is not legal advice and makes no statement about anyone's compliance with any licence or policy.

## Credit

| | author | what | state, read 2026-10-08 |
|---|---|---|---|
| #22704 | srossitto79 | the first report of the accumulator defect, and the first `MUL_MAT_ID` backward (hole 8) | closed unmerged 2026-09-24 |
| #22705 | srossitto79 | MoE and QLoRA training work built on #22704 | closed 2026-09-24 |
| #26794 | DFveloper | the same accumulator hunk as #22704 | closed unmerged 2026-08-11 |
| #27096 | devYRPauli | hole 1 | merged 2026-10-02 |
| #28520 | ngxson | hole 3 | merged 2026-09-30 |
| #27690 | unravel1020 | hole 6, the arithmetic; fixes issue #27648 | open |
| #28851 | rssndr | hole 13 (first report #25568, edenfunf) | open |
| #30142 | edenfunf | hole 14 | open, opened 2026-10-08 |
| #29575 | angt | bounds assert in `get_rows_back`, overlaps hole 9 | merged 2026-09-29 |
| #27199 | ngxson | stopped the crash that preceded hole 3 (as the companion repository records) | merged 2026-09-02 |
| #30170, #30083 | shashb27, joelteply | the accumulator defect, re-filed and re-reported | open (#30170 a draft) |
| #29364 | Ron-Builds-AI | the accumulator fix, in the companion repository; hole 5 needs it | closed unmerged 2026-09-24 |

Independent reports of the defects behind hole 6: #27648, #22040, #15535, #15532 and #21037 (read 2026-09-24).

## How it was made

The patches were written with an AI assistant at my direction and tested on my own machines. The pull requests and issues that may follow
this repository are written by me, in my own words.

## License

MIT, matching llama.cpp. `LICENSE` is the same text as the companion repository's.

---

Drafted and fact-checked with an AI assistant from my own records, measurements and decisions. I read and edited every sentence and I am responsible for its contents.
