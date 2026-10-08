# Hole 15: grad mode prints `OK` for a case before it checks it

**File:** `ours_hole15_grad-case-header-carries-no-status.diff` (SHA256 `5fd839a32649b96a8cd8dc18ea1d60cf7720758551f820b0494f1bad4014ddea`). It changes `tests/test-backend-ops.cpp` only.

## What breaks

In grad mode, each case prints a header line of the form `OP(params): OK` before the backward graph is built and before anything is
checked. The case's own verdict follows later, as a line named after the loss node (`SUM(params): OK` or `... FAIL`). A case that
aborts leaves only its header, and its header says `OK`. A reader who counts `OK` lines calls the aborting case a pass.

## The patch

The header only names the case: it ends with `: ` and no status. The result follows on a line of its own. Nothing is checked
differently.

## Where it applies

It needs hole 11 in the tree, because its context includes hole 11's `is_nothing_compared` member; it does not apply to a tree
without it. With hole 11 first, it applies to `8212c7802` and to master `71ad0590f4` (git apply on the touched files, 2026-10-08).

## Test

```
build/bin/test-backend-ops grad -o SOFT_MAX -p inplace=0 -b CPU
build/bin/test-backend-ops grad -o REGLU -b CPU        # aborts by design while hole 12 is out of the tree
```

## Proof

One-tree run, Linux CPU, 2026-09-25:

| | BEFORE | AFTER |
|---|---|---|
| `grad -o SOFT_MAX -p inplace=0 -b CPU` | 206 lines of the form `SOFT_MAX(...): OK` (the headers); exit 0, `1077/1077` | 0 such lines. 206 bare headers `SOFT_MAX(...): `, each followed by exactly one result line with the same parameters: 65 `SUM(...): OK`, 125 `skipping large tensors for speed`, 16 `SUM(...): not supported [...]`. Exit 0, `1077/1077`, 0 FAIL: the same results as BEFORE. |
| `grad -o REGLU -b CPU`, hole 12 out | exit 134 on `ggml.c:7279: unsupported glu op for backward pass: REGLU`; the last stdout line is `REGLU(type=f32,ne_a=[128,2,2,2],v=0,split): OK` | the same abort; the last stdout line is `REGLU(type=f32,ne_a=[128,2,2,2],v=0,split): `, named, with no OK |

The name of the result line depends on the op: `SOFT_MAX` results read `SUM(...)`, and `MUL_MAT` results read `MUL_MAT(...)`. A reader or a
script should pair each bare header with the next line that carries the same parameters.

The same was run on the Windows CPU (round 4): 206 headers, each followed by exactly one result (65 results, 125 size skips, 16 not
supported), and the same REGLU lines, exit 127 on both sides there.

## Not verified

CPU only.
