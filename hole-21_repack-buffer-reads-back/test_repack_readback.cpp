// test_repack_readback.cpp - hole 21's regression test: weights the CPU backend repacked can be read back.
// For every type that has a repack layout on this CPU, random blocks are set into a CPU_REPACK tensor of 16 rows, then read back
// whole, by offset (rows 3 to 7), and through two views of 8 rows: rows 8 to 15 (on a row-group boundary) and rows 4 to 11
// (across two groups). Each read must equal what was set. On llama.cpp master before the fix the first read calls the buffer's
// NULL get_tensor.
// build: g++ -O1 -std=c++17 test_repack_readback.cpp -I<llama.cpp>/ggml/include -L<libs> -lggml -lggml-base
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == nullptr) {
        printf("SKIP: no CPU device\n");
        return 0;
    }
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = nullptr;
    for (ggml_backend_buffer_type_t * p = get_extra ? get_extra(dev) : nullptr; p && *p; ++p) {
        if (strcmp(ggml_backend_buft_name(*p), "CPU_REPACK") == 0) {
            repack = *p;
        }
    }
    if (repack == nullptr) {
        printf("SKIP: this build has no CPU_REPACK buffer type\n");
        return 0;
    }

    const ggml_type types[] = { GGML_TYPE_Q4_0,   GGML_TYPE_Q4_K,  GGML_TYPE_Q2_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q8_0, GGML_TYPE_Q1_0 };
    int      tested = 0;
    int      failed = 0;
    uint64_t seed   = 42;
    for (ggml_type type : types) {
        ggml_init_params ip  = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_tensor *    w   = ggml_new_tensor_2d(ctx, type, 512, 16);
        ggml_set_name(w, "w");
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, repack);
        if (buf == nullptr || w->extra == nullptr) {
            printf("%-7s no repack layout on this CPU, not tested\n", ggml_type_name(type));
            if (buf != nullptr) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            continue;
        }
        const size_t         n = ggml_nbytes(w);
        std::vector<uint8_t> in(n), back(n);
        for (uint8_t & b : in) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            b    = (uint8_t) (seed >> 56);
        }
        ggml_backend_tensor_set(w, in.data(), 0, n);

        ggml_backend_tensor_get(w, back.data(), 0, n);
        const bool whole_ok = memcmp(back.data(), in.data(), n) == 0;

        const size_t         off = 3 * w->nb[1];
        const size_t         len = 5 * w->nb[1];
        std::vector<uint8_t> part(len);
        ggml_backend_tensor_get(w, part.data(), off, len);
        const bool part_ok = memcmp(part.data(), in.data() + off, len) == 0;
        printf("%-7s whole %s, rows 3 to 7 %s\n", ggml_type_name(type), whole_ok ? "ok" : "DIFFERS", part_ok ? "ok" : "DIFFERS");

        bool view_ok = true;
        for (int64_t first : { (int64_t) 8, (int64_t) 4 }) {
            ggml_tensor * v  = ggml_view_2d(ctx, w, w->ne[0], 8, w->nb[1], first * w->nb[1]);
            bool          ok = ggml_backend_view_init(v) == GGML_STATUS_SUCCESS;
            if (ok) {
                std::vector<uint8_t> vb(ggml_nbytes(v));
                ggml_backend_tensor_get(v, vb.data(), 0, vb.size());
                ok = memcmp(vb.data(), in.data() + first * w->nb[1], vb.size()) == 0;
            }
            printf("%-7s view of rows %lld to %lld %s\n", ggml_type_name(type), (long long) first, (long long) first + 7,
                   ok ? "ok" : "DIFFERS");
            view_ok = view_ok && ok;
        }

        tested++;
        failed += (whole_ok && part_ok && view_ok) ? 0 : 1;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    const bool pass = tested > 0 && failed == 0;
    printf("%s: %d types tested, %d failed\n", pass ? "PASS" : "FAIL", tested, failed);
    return pass ? 0 : 1;
}
