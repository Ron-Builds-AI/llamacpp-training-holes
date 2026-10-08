# Instruments

Small tools the runs used. None is part of a patch.

| file | what | SHA256 |
|---|---|---|
| `gguf_diff_stdlib.py` | compares the tensors of two GGUF files by name, standard library only. Prints whether the raw bytes of each tensor are identical, and for F32 `attn_q`, `attn_k` and `attn_v` tensors the number of changed elements and the largest change. Shown to see both outcomes: run on a file against itself, every line reads `bytes_identical=True`. | `45b6bb314e4cd3a018a2dfd0af09dc9516930f39ccb2b006693e04546679286e` |
| `probe_sinks_grad.cpp` | the standalone probe for hole S: the gradient of the softmax sinks against a closed form in double, in two modes, `direct` and `acc`. Build line in `../hole-S_attention-sinks-backward/README.md`. | `30c2203245927e0575c23eba19b2c845a53b8cf67514e81a625e413b42fb8ce9` |
| `make_tied_copy.py` | writes a copy of a GGUF without `output.weight`, so llama.cpp ties the output layer to `token_embd.weight`. Needs `gguf-py` from a llama.cpp checkout on `PYTHONPATH`. | see below |

`make_tied_copy.py` was edited for publication: the one line that put a local `gguf-py` path on `sys.path` was removed and a usage note
added. The unedited script was SHA256 `ce73d5cfcad1cc173e1922f25a644e8fc6635c0d60fe30d9385827045a94a869`. The edited script was not re-run.
The model in the proof runs was written by the unedited one.

The first two files are byte-identical to the ones the runs used.
