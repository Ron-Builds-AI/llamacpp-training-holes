# Hole 17: `MUL_MAT_ID` backward through frozen quantized experts

**Files:**
- `ours_hole17_mul-mat-id-backward-through-frozen-quantized-experts.diff` (SHA256 `25c7f899b28d0970d1aa030263d1d0071f7d629d0e18f1f7dfc46422081b3afb`), the change to `ggml_compute_backward`
- `ours_hole17t_twin-test-quantized-against-dequantized-experts.diff` (SHA256 `cc114bfdbf5259832d61c380512508bc8d68d551310a5ef87d4e3332c2e9eea6`), a new test program `tests/test-mul-mat-id-quantized-grad.cpp` and its line in `tests/CMakeLists.txt`

The first line of the test patch's git header named a local folder in the record's copy. In this repository it reads
`a/tests/test-mul-mat-id-quantized-grad.cpp`, and nothing else changed: applied to the same tree, the record's copy and this copy write byte-identical
files (the new test file is SHA256 `604e15cf3fb3b5fc2467a1cccb88768076f3ae43746513779833cc444c868ee0`). The record's copy was SHA256
`ae21fc61a31e6703119372f68aab195da0f7ef725a9a99508ab87915ff228bbc`.

## What breaks

With holes 8 and 16 in, the `MUL_MAT_ID` backward refuses quantized expert weights with an assert, because a quantized block cannot
be transposed and the gradient of the activations needs the transposed experts.

## The patch

For quantized expert weights, the backward dequantizes them to f32 with `ggml_cast` first, and then transposes and routes as it does
for f32. The f32 copy exists only in the backward graph. The assert now names the three accepted cases in words (`needs f32, f16 or
quantized expert weights`).

## What the test shows, at its real size

The test compares two gradients of `sum(mul_mat_id(experts, b, ids))` with respect to `b`: one through the experts as quantized
(Q4_0, Q8_0, Q4_K, Q6_K), and one through the same experts dequantized to f32 beforehand. It does not compare against a finite
difference of the quantized forward pass. On the CPU, the forward pass rounds the activations to the quantized dot type, so a finite
difference of it is not the gradient of the dequantized weights; the test's header comment says so. The measured result is that the
two backward passes agree to a relative difference of exactly 0.000e+00 for all four types. That is expected, because the backward
dequantizes with the same function the test calls on the host (inferred from the code, not traced). **This is not a check that the gradient is the derivative
of the quantized model's loss, and no end-to-end training with this change is reported here.**

## Needs, and where it applies

- Holes 8 and 16, in that order. Not applicable to a tree without them.
- The test patch alone applies to `8212c7802` and to master `71ad0590f4`, but the test reaches the quantized assert (its BEFORE) only with
  holes 8 and 16 in the tree.
- On master `71ad0590f4`, hole 17 needs hole 8, which does not apply there as is.

## Test

```
cmake -B build ... -DLLAMA_BUILD_TESTS=ON
cmake --build build -j --target test-mul-mat-id-quantized-grad
build/bin/test-mul-mat-id-quantized-grad
```

BEFORE is the tree with the test patch and without the fix. AFTER adds the fix.

## Proof (Windows CPU, Ryzen 9 9950X3D, MSVC 19.51, 2026-09-25)

| | BEFORE | AFTER (3 runs) |
|---|---|---|
| exit | -1073740791 (0xC0000409, MSVC `abort()`) | 0 in 3 of 3, empty stderr |
| result | stderr: `ggml.c:7008: GGML_ASSERT((src0->type == GGML_TYPE_F32 \|\| src0->type == GGML_TYPE_F16) && "backward pass for MUL_MAT_ID needs f32 or f16 expert weights") failed`; stdout empty (buffered, not flushed by `abort()`) | `q4_0: ... = 0.000e+00, \|grad\| = 1.470167e+02: OK`, `q8_0 ... 1.481606e+02: OK`, `q4_K ... 1.484883e+02: OK`, `q6_K ... 1.481881e+02: OK`, `PASS`; the three outputs are byte-identical |
| planted control | | the fix with the dequantized weights scaled by 2 (a plant, not in this repository): exit 1, all four types print `= 1.000e+00 ... FAIL`, so the test sees a wrong gradient at full size |
| regression | | `grad -o MUL_MAT_ID -b CPU`: exit 0, 12 grad cases OK (6 f32, 6 f16), `20378/20378`; `test-opt`: `120/120` (AdamW), `48/48` (SGD), `2/2 backend*optimizer passed` |

That the BEFORE failure is at the first type (Q4_0) is inferred from the test's source, not from the empty stdout. The relative
difference of exactly 0 means the 1e-5 tolerance in the test was not exercised.

## Not verified

- Linux and the card were not run for this patch. CPU only, one machine.
- No end-to-end training with quantized experts is reported here. The gradient was compared only with its dequantized twin.
- Vulkan, Metal, SYCL.
