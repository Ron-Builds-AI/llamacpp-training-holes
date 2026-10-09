# Hole 21: a model loaded with repacked weights cannot be saved (`llama_model_save_to_file` calls a NULL `get_tensor`)

| file | what | SHA256 |
|---|---|---|
| `ours_hole21_repack-buffer-reads-back.diff` | the fix, two files, `ggml/src/ggml-backend.cpp` and `ggml/src/ggml-cpu/repack.cpp` | `18ef302d06200736de849572c0a89374a3b03819a4f9b4a5415ad0a59b8059f9` |
| `test_repack_readback.cpp` | the regression test, public API only | `2c309b32f81aaa389808f77aa50775eebe3caf53fbbb5a1044a36d0d0d0f0f79` |
| `hole21_save.cpp` | the reproduction: load a GGUF on the CPU, save it again | `24f297d7b3331d4d3cd86bcd19a5c921956905c1253e3b78a6f3313a2ff1a9de` |
| `instrument_layout_report_NOT_PART_OF_THE_FIX.diff` | a test instrument that goes on top of the fix and prints one line per layout; never part of the fix | `f4c4697baa0b6e6d122e0f3e9c3cf09c1a1494f12c986831c578fd1f817a58c8` |

## What breaks

llama.cpp's CPU backend loads quantized weights into an extra buffer type, CPU_REPACK, unless `use_extra_bufts` is false (the
`--no-repack` flag). Its `set_tensor` rearranges each tensor into an interleaved layout for the matrix kernels. Its `get_tensor` is NULL
(`ggml/src/ggml-cpu/repack.cpp:5178` at master `de7fa0a3c6a2`), and `ggml_backend_tensor_get` (`ggml/src/ggml-backend.cpp:490` to `503`)
calls the buffer's `get_tensor` with no check. The GGUF writer reads every tensor through it (`ggml/src/gguf.cpp:1591` and `1635`), so
`llama_model_save_to_file` on a model loaded with default settings jumps to address 0:

```
master, repacking on:  rc 139, "Command terminated by signal 11", on all four test files below
```

It was reported upstream as issue #29701 by west-ice-g on 2026-09-30, with a reproduction on a Q2_K GGUF of Qwen2.5-0.5B-Instruct. The
issue was closed as not planned the same day, with no comments. The bug is on master as of 2026-10-08.

Who meets it: any program that loads a quantized model with the defaults and then saves it, or reads a weight back. `llama-finetune` saves
at the end of every run; with hole 20 in, it loads without repacking and never reaches this path. As I read the source, upstream's own
round-trip test of the saver loads through `llama_model_init_from_user`, which sets `use_extra_bufts = false`, so it never sees a repacked
weight (inferred, not run).

The same family upstream: CPU_KLEIDIAI (`kleidiai.cpp:1752`), AMX (`amx.cpp:113`) and CPU_RISCV64_SPACEMIT have no `get_tensor` either. An
AMX crash of the same kind (#11115, January 2025) was handled by keeping callers away from those buffers (#11120, #11124); #11124 also
added the two NULLs to the CPU buffer this file grew from, as "unsupported". In #9921 (November 2024) a maintainer wrote that the buffer
type is there to hide the conversion, and that a tensor read back with `get_tensor` should come back in its original format. OpenCL (#18278) and Hexagon already restore the plain layout in their `get_tensor`, each with a hand-written inverse per
type.

## The patch

Two parts, 219 added lines and 1 removed, in two files. No new files and no new dependencies.

1. **The guard** (`ggml-backend.cpp`). When a buffer has no `get_tensor`, `ggml_backend_tensor_get` aborts with
   `tensor <name> is in a <buffer> buffer, which cannot be read back` instead of calling NULL. This covers the three other write-only
   CPU buffers and any future one.
2. **The restore** (`repack.cpp`). CPU_REPACK gets a `get_tensor`. Every repack routine in that file moves each input bit to a fixed place,
   and some flip fixed bits (Q4_0 flips `0x88` in each byte), in the same way for every group of rows. So the way back is learned from the
   repack function itself, once per layout: repacking all-zero blocks shows the flipped bits, and repacking blocks with one bit set shows
   where that bit goes. Reading that map backwards restores the plain bytes. A layout whose repack does anything else, or a shape that does
   not divide into its units, is refused with a message naming `--no-repack` and `use_extra_bufts = false`. Every read is checked by
   repacking the restored bytes again and comparing them with the bytes in memory. A view reads through the tensor it views.

The direction was mine: one small piece of code that connects the levers instead of a fix per layout. The pieces it is built from are old,
and I did not invent them:

- probing a black box with every single-bit input to recover a permutation (US 6,292,762, filed 1998, expired, for FPGA testing; learning a
  linear map over GF(2) with one query per bit is textbook);
- building a program's inverse from the forward program (PINS, Srivastava, Gulwani, Chaudhuri and Foster, PLDI 2011; Hu and D'Antoni,
  PLDI 2017), symbolically, for encoders and compressors;
- inverting declared layout maps and refusing the ones that are not one to one (TVM, MLIR, Triton, CuTe);
- hand-written inverses per layout (OpenCL, Hexagon, CANN and others), and round-trip checks.

I did not find this combination, learning the inverse of a runtime weight repack from the repack function, with the refusal and a check on
every read, in the searches I ran (GitHub, the web, some patent and paper pages). That is not a claim that nobody has done it.

## Other ways, kept open

- A check in the saver before it opens the file, or writing to a temporary file and renaming it. A crashed or refused save leaves a
  half-written file behind (measured: 151,216,128 bytes for three of the files below, 188,133,376 for the fourth). Not in this patch.
- A hand-written inverse per layout, upstream's usual way. More code, one function per layout to keep in step with the repack.
- Training with repacking on (hole 20's path). The restore does not reach it: the backward pass builds `OUT_PROD`, which reads the weight's
  raw data, not through `get_tensor` (from reading the source, not traced). Hole 20 keeps repacking off in `llama-finetune`.
- KleidiAI, AMX and SpacemiT stay write-only; only the guard changes for them. SYCL's reordered weights are read back without an error and in
  the wrong layout (from a reading of the source, not run). That is a different kind of defect and this patch does not touch it.

## What the tested tree was

- **Source:** upstream master `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b` (2026-10-08), from GitHub's source archive (SHA256
  `91fb3afb1720e243117150d7ba815004c90e014d47cf8f9686a48ce8ef5d623e`). The two files the patch touches were checked by SHA256 against the
  copies the patch was made from before anything was applied. This hole is on master, not on the `8212c7802` base of the other folders.
- **Machine:** the Linux machine of the top README (Intel i7-10750H, 6 cores, 12 threads, AVX2, no AVX512), Ubuntu 24.04, g++ 13.3.0,
  cmake 3.28.3. CPU only. On this CPU the loader repacks Q4_0, Q4_K, IQ4_NL and MXFP4 tensors; Q2_K repacks only with AVX512.
- **Build:** `cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`, then `cmake --build build --target llama llama-quantize -j 6`.
  One tree, built in stages: master; master with the `ggml-backend.cpp` part of the fix only (the guard); the whole fix; the fix with the
  layout report on top. Each stage's libraries were copied out and each program was run against its own stage's copy.
  The fix built with 0 lines containing "warning".

## Test

Models, all made on that machine by master's `llama-quantize` from the F32 file of the top README (SHA256 `10b105fd...e70535a`), no
importance matrix. Census by ggml type, 290 tensors each:

| file | SHA256 | census | tensors the loader repacked |
|---|---|---|---|
| Q4_0 | `a9bd75783653a959ebc73dac332ddae029a99783378ce912e30384dc6d8371b0` | F32 121, Q4_0 168, Q8_0 1 | 168 |
| Q4_K_M | `3a07a8f302da9d432ff04dc98616c57987ba08492fb130863b369bae5506f178` | F32 121, Q5_0 132, Q8_0 13, Q4_K 12, Q6_K 12 | 12 |
| IQ4_NL | `ec7a8fe1f6d2650fd35f66cc9c4646743faeeef244135b0b4b1dd790b421d26d` | F32 121, Q5_1 24, Q5_K 3, Q8_0 1, IQ4_NL 141 | 141 |
| Q2_K | `3478b7a43992256baaf7ac15e58fc39c5b86d1dba462b8ace50b76a81e6d851b` | F32 121, Q4_0 120, Q5_0 24, Q3_K 24, Q8_0 1 | 120 |

The Q2_K file holds no Q2_K tensor: this model's rows of 896 do not divide into 256-wide K blocks, so the quantizer fell back. Its 120 Q4_0
tensors are the ones that repack, the same count as the 120 "repack:" lines in #29701 (the same cause is inferred). The Q4_K_M file is
byte for byte the one hole 20 used.

```
g++ -O1 -g -std=c++17 hole21_save.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include -L<libs> -lllama -Wl,-rpath,<libs> -o hole21_save
./hole21_save MODEL.gguf OUT.gguf repack       # llama.cpp's defaults
./hole21_save MODEL.gguf OUT.gguf norepack     # use_extra_bufts = false
python3 ../instruments/gguf_diff_stdlib.py A.gguf B.gguf

g++ -O1 -g -std=c++17 test_repack_readback.cpp -I<llama.cpp>/ggml/include -L<libs> -lggml -lggml-base -Wl,-rpath,<libs> -o test_repack_readback
./test_repack_readback
```

`hole21_save` prints how many tensors the loader logged as repacked, then saves. `test_repack_readback` asks the CPU device for its
CPU_REPACK buffer type and, for every type that has a layout on the running CPU, sets random blocks into a 16-row tensor and reads them back
whole, by offset (rows 3 to 7), and through two views (rows 8 to 15, and rows 4 to 11, which crosses two row groups). Each read must equal
what was set. Types with no layout on the CPU are listed as not tested.

## Proof (Linux CPU, 2026-10-08, one run per arm)

Each run's settings, bars and predicted reading were written down and frozen (SHA256 recorded by the runner) before it ran. The frozen
files are not in this repository: they carry local paths.

| | Q4_0 | Q4_K_M | IQ4_NL | Q2_K |
|---|---|---|---|---|
| master, repacking off | rc 0, 0.67 s | rc 0, 0.64 s | rc 0, 0.66 s | rc 0, 0.64 s |
| master, repacking on | SIGSEGV | SIGSEGV | SIGSEGV | SIGSEGV |
| the fix, repacking on | rc 0, 1.23 s | rc 0, 0.94 s | rc 0, 1.20 s | rc 0, 1.02 s |
| the fix, repacking on, against master's save with repacking off | 0 of 290 tensors changed | 0 of 290 | 0 of 290 | 0 of 290 |
| the fix, repacking off, against the same | 0 of 290 changed | 0 of 290 | 0 of 290 | 0 of 290 |

In every comparison no tensor is in one file only. Master's save with repacking off is itself 0 of 290 changed against its input file.
The fixed save took 0.3 to 0.6 seconds longer than master's save with repacking off.

The guard alone (master with the `ggml-backend.cpp` part only), Q4_0, repacking on: rc 134 and
`ggml_backend_tensor_get: tensor blk.0.attn_q.weight is in a CPU_REPACK buffer, which cannot be read back`. That run used the first
version of the guard hunk; this version prints the same text and only wraps the source line.

The regression test:

| build | result |
|---|---|
| master | SIGSEGV (rc 139) at the first read |
| an earlier version of this fix, which decoded a view with the view's own layout | whole and offset reads ok; view of rows 8 to 15 ok; view of rows 4 to 11 **wrong bytes** on Q4_0, Q4_K, IQ4_NL and MXFP4; rc 1 |
| this fix | every read ok on Q4_0, Q4_K, IQ4_NL and MXFP4; PASS; rc 0 (Q2_K, Q5_K, Q6_K, Q8_0, Q1_0: no layout on this CPU, not tested) |

The earlier version's defect was found by a code review before publishing. The view of rows 8 to 15 sits exactly on a row group, which is
why it read correctly there.

The layout report (the instrument diff on top of the fix, `GGML_REPACK_PROBE_REPORT=1`, during a Q4_0 save that itself exited 0): every
one of the 18 layouts this file builds on x86 learned its way back, and a random round trip through a tensor of 3 by 3 units came back
exact. Layouts: q1_0 4x4 and 4x8; q4_0 4x4, 4x8 and 8x8; q4_K 8x4 and 8x8; q5_K 8x4 and 8x8; q6_K 8x4 and 8x8; q2_K 8x8; iq4_nl 4x4 and
8x8; mxfp4 4x4 and 8x8; q8_0 4x4 and 4x8. In 13 of them every byte moves whole. In q4_K and q5_K, 80 bytes of each unit (the 6-bit scales)
move bit by bit, and in q1_0 4x4, 64 of 72 bytes do. Those restored exactly too. Only 4 of the 18 are used by a model on this CPU; the
other 14 were exercised on random blocks only.

## What it shows, and what it does not

It shows, on this CPU and these four files: the save crashes on master with repacking on; with the fix it writes the same tensor bytes as a
save with repacking off; the regression test fails before the fix and passes after it; the guard turns a NULL call into a message.

It does not show:

- Anything on another CPU. AVX512 (where Q2_K repacks), ARM and RISC-V layouts were exercised only on random blocks in the layout report,
  never through a model. MSVC and big-endian builds were not made.
- A mixture-of-experts model (3D expert weights) through a save. The code handles them by the same row count as the repack (from reading).
- That a reloaded model gives the same logits. The comparison is of bytes, which is the stronger check for a save, but logits were not run.
- Memory under a 4 GB limit. A read holds the tensor's plain bytes and one more copy for the check, so about twice the largest repacked
  tensor (from reading).
- Concurrent reads, or any other reader of weights than the saver and the test.
- Anything about upstream acceptance. Nothing here has been proposed upstream.

## Not verified

- A build on Windows, ARM, or a machine with AVX512.
- Which existing upstream test file the regression test belongs in.
- That the 120 repacked tensors in #29701 are the same Q4_0 fallback tensors as here.

Drafted and fact-checked with an AI assistant from my own records, measurements and decisions. I read and edited every sentence and I am responsible for its contents.
