// probe_sinks_grad.cpp, round 2 (2026-09-24), B308: the gradient of the softmax sink logits.
// Not part of any patch. Links the tree's ggml (CPU only). No allocator: every tensor has its own memory.
//
//   probe_sinks_grad direct   sinks (and a) are parameters, no preallocated accumulators:
//                             ggml_build_backward_expand(ctx, gb, NULL), the test-backend-ops path.
//   probe_sinks_grad acc      the same parameters WITH preallocated accumulators, the ggml_opt / llama-finetune path.
//
// Forward (CPU kernel semantics): per row, x'_i = scale*a_i + slope*mask_i, s = sinks[i2] (not scaled),
//   y_i = exp(x'_i) / (sum_j exp(x'_j) + exp(s)),   p_s = exp(s) / (same denominator) = 1 - sum_i y_i.
// Derivation: the sink is one more logit of the softmax whose output is dropped (its dy is 0), so with
//   dot = sum_i y_i*dy_i:   dL/dx'_i = y_i*(dy_i - dot),   dL/ds = p_s*(0 - dot) = -p_s*dot,
//   summed over every row that uses sinks[i2] (all i1, all i3). dL/da_i = scale*dL/dx'_i.
// Loss here = sum(y*w) with w a fixed random tensor, so the loss is not a near constant (sum(y) alone is n_rows - sum p_s).
//
// Three checks per case:
//   (1) ggml backward vs the closed form in double                     (the subject)
//   (2) closed form vs a double central difference of the double forward  (checks the derivation itself)
//   (3) ggml forward central difference (float32, eps 1e-2) vs the closed form (checks ggml's forward is the closed form's)
// Exit 0 only if every check of every case is inside its bar.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct scase {
    int64_t ne0, ne1, ne2, ne3;
    bool    mask;
    float   scale;
    float   max_bias;
};

struct fwd_out {
    double loss;
    std::vector<double> ds;  // [ne2]
    std::vector<double> da;  // [ne0*ne1*ne2*ne3]
};

static float alibi_slope(uint32_t h, uint32_t n_head, float max_bias) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    return h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1);
}

// closed form, double: loss = sum(y*w), and the exact gradients
static fwd_out closed_form(const scase & c, const std::vector<float> & a, const std::vector<float> & m,
                           const std::vector<double> & s, const std::vector<float> & w) {
    fwd_out o;
    o.loss = 0.0;
    o.ds.assign(c.ne2, 0.0);
    o.da.assign(a.size(), 0.0);
    std::vector<double> x(c.ne0), y(c.ne0);
    for (int64_t i3 = 0; i3 < c.ne3; i3++) {
        for (int64_t i2 = 0; i2 < c.ne2; i2++) {
            const double slope = alibi_slope((uint32_t) i2, (uint32_t) c.ne2, c.max_bias);
            for (int64_t i1 = 0; i1 < c.ne1; i1++) {
                const int64_t row = ((i3*c.ne2 + i2)*c.ne1 + i1)*c.ne0;
                const int64_t mrow = (i2*c.ne1 + i1)*c.ne0; // mask [ne0, ne1, ne2, 1], broadcast over i3
                double mx = s[i2];
                for (int64_t i0 = 0; i0 < c.ne0; i0++) {
                    x[i0] = (double) c.scale*a[row + i0] + (c.mask ? slope*(double) m[mrow + i0] : 0.0);
                    mx = std::max(mx, x[i0]);
                }
                double z = exp(s[i2] - mx);
                for (int64_t i0 = 0; i0 < c.ne0; i0++) {
                    z += exp(x[i0] - mx);
                }
                const double ps = exp(s[i2] - mx)/z;
                double dot = 0.0;
                for (int64_t i0 = 0; i0 < c.ne0; i0++) {
                    y[i0] = exp(x[i0] - mx)/z;
                    dot += y[i0]*w[row + i0];
                }
                o.loss += dot;
                o.ds[i2] += -ps*dot;
                for (int64_t i0 = 0; i0 < c.ne0; i0++) {
                    o.da[row + i0] = (double) c.scale*y[i0]*(w[row + i0] - dot);
                }
            }
        }
    }
    return o;
}

static double rel(double a, double b) {
    return fabs(a - b)/std::max(1e-12, std::max(fabs(a), fabs(b)));
}

static int run_case(const scase & c, bool use_acc, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    const int64_t n = c.ne0*c.ne1*c.ne2*c.ne3;
    std::vector<float> ha(n), hm(c.ne0*c.ne1*c.ne2), hs(c.ne2), hw(n);
    for (auto & v : ha) v = u(rng);
    for (auto & v : hm) v = u(rng);
    for (auto & v : hs) v = u(rng);
    for (auto & v : hw) v = u(rng);

    ggml_init_params ip = { /*mem_size*/ (size_t) 256*1024*1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne0, c.ne1, c.ne2, c.ne3);
    ggml_set_name(a, "a");
    // the CPU softmax backward asserts max_bias == 0 (ALiBi has no backward for a): with max_bias only the sinks are parameters
    const bool param_a = c.max_bias == 0.0f;
    if (param_a) {
        ggml_set_param(a);
    }
    ggml_tensor * mask = nullptr;
    if (c.mask) {
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne0, c.ne1, c.ne2, 1);
        ggml_set_name(mask, "mask");
        memcpy(mask->data, hm.data(), hm.size()*sizeof(float));
    }
    ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.ne2);
    ggml_set_name(sinks, "sinks");
    ggml_set_param(sinks);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne0, c.ne1, c.ne2, c.ne3);
    ggml_set_name(w, "w");
    memcpy(a->data, ha.data(), n*sizeof(float));
    memcpy(sinks->data, hs.data(), c.ne2*sizeof(float));
    memcpy(w->data, hw.data(), n*sizeof(float));

    ggml_tensor * y = ggml_soft_max_ext(ctx, a, mask, c.scale, c.max_bias);
    ggml_soft_max_add_sinks(y, sinks);
    ggml_set_name(y, "y");
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, y, w));
    ggml_set_name(loss, "loss");
    ggml_set_loss(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, loss);
    ggml_cgraph * gb = ggml_new_graph_custom(ctx, 1024, true);
    ggml_graph_cpy(gf, gb);

    std::vector<ggml_tensor *> accs;
    ggml_tensor * acc_s = nullptr;
    ggml_tensor * acc_a = nullptr;
    if (use_acc) {
        const int nn = ggml_graph_n_nodes(gf);
        accs.assign(nn, nullptr);
        for (int i = 0; i < nn; i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->flags & (GGML_TENSOR_FLAG_PARAM | GGML_TENSOR_FLAG_LOSS)) {
                accs[i] = ggml_dup_tensor(ctx, node);
                if (node == sinks) acc_s = accs[i];
                if (node == a)     acc_a = accs[i];
            }
        }
    }

    printf("  building backward (%s)...\n", use_acc ? "preallocated accumulators" : "no accumulators");
    fflush(stdout);
    ggml_build_backward_expand(ctx, gb, use_acc ? accs.data() : nullptr);
    printf("  backward built: %d nodes\n", ggml_graph_n_nodes(gb));
    fflush(stdout);

    ggml_graph_reset(gb);
    ggml_graph_compute_with_ctx(ctx, gb, 1);

    ggml_tensor * gs = use_acc ? acc_s : ggml_graph_get_grad(gb, sinks);
    ggml_tensor * ga = use_acc ? acc_a : ggml_graph_get_grad(gb, a);

    std::vector<double> sd(hs.begin(), hs.end());
    const fwd_out cf = closed_form(c, ha, hm, sd, hw);

    // (1) ggml backward vs closed form
    // bars are relative to the largest element of the vector (a single element can be near zero)
    double max_ds = 0.0;
    for (int64_t h = 0; h < c.ne2; h++) {
        max_ds = std::max(max_ds, fabs(cf.ds[h]));
    }
    double max_rel_s = 0.0, max_abs_s = 0.0, max_abs_a = 0.0, max_abs_a_ref = 0.0;
    std::string s_line;
    for (int64_t h = 0; h < c.ne2; h++) {
        const double gv = gs ? (double) ((float *) gs->data)[h] : 0.0;
        max_rel_s = std::max(max_rel_s, rel(gv, cf.ds[h]));
        max_abs_s = std::max(max_abs_s, fabs(gv - cf.ds[h]));
        if (h < 3) {
            char buf[128];
            snprintf(buf, sizeof(buf), " ds[%lld] ggml %+.9f closed %+.9f;", (long long) h, gv, cf.ds[h]);
            s_line += buf;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        const double gv = ga ? (double) ((float *) ga->data)[i] : 0.0;
        max_abs_a = std::max(max_abs_a, fabs(gv - cf.da[i]));
        max_abs_a_ref = std::max(max_abs_a_ref, fabs(cf.da[i]));
    }

    // (2) closed form vs double central difference of the double forward (the derivation)
    double max_rel_deriv = 0.0;
    for (int64_t h = 0; h < c.ne2; h++) {
        const double e = 1e-5;
        std::vector<double> sp = sd, sm = sd;
        sp[h] += e; sm[h] -= e;
        const double fdv = (closed_form(c, ha, hm, sp, hw).loss - closed_form(c, ha, hm, sm, hw).loss)/(2*e);
        max_rel_deriv = std::max(max_rel_deriv, rel(fdv, cf.ds[h]));
    }

    // (3) ggml forward central difference (float32) vs closed form
    double max_abs_fwd = 0.0, max_fd_ref = 0.0;
    for (int64_t h = 0; h < c.ne2; h++) {
        const float e = 1e-2f;
        float * sv = (float *) sinks->data;
        const float s0 = sv[h];
        sv[h] = s0 + e; ggml_graph_compute_with_ctx(ctx, gf, 1); const double lp = ((float *) loss->data)[0];
        sv[h] = s0 - e; ggml_graph_compute_with_ctx(ctx, gf, 1); const double lm = ((float *) loss->data)[0];
        sv[h] = s0;
        // the central difference has an O(e^2) truncation term; compare to the double central difference at the same e
        std::vector<double> sp = sd, sm = sd;
        sp[h] += e; sm[h] -= e;
        const double fd_ref = (closed_form(c, ha, hm, sp, hw).loss - closed_form(c, ha, hm, sm, hw).loss)/(2*(double) e);
        max_abs_fwd = std::max(max_abs_fwd, fabs((lp - lm)/(2*(double) e) - fd_ref));
        max_fd_ref  = std::max(max_fd_ref, fabs(fd_ref));
    }
    const double max_rel_fwd = max_abs_fwd/std::max(1e-30, max_fd_ref);
    const double norm_rel_s  = max_abs_s/std::max(1e-30, max_ds);

    const bool ok1 = gs != nullptr && norm_rel_s < 1e-5;
    const bool ok1a = !param_a || (ga != nullptr && max_abs_a < 1e-5*std::max(1.0, max_abs_a_ref));
    const bool ok2 = max_rel_deriv < 1e-6;
    const bool ok3 = max_rel_fwd < 5e-3;
    printf("  case ne=[%lld,%lld,%lld,%lld] mask=%d scale=%g max_bias=%g seed=%u:%s\n",
        (long long) c.ne0, (long long) c.ne1, (long long) c.ne2, (long long) c.ne3, (int) c.mask, c.scale, c.max_bias, seed, s_line.c_str());
    if (param_a) {
        printf("    (1) sinks: ggml vs closed max abs %.3e = %.3e of the largest |ds| (bar 1e-5), per-element max rel %.3e %s | a: max abs %.3e (largest |da| %.3e) %s\n",
            max_abs_s, norm_rel_s, max_rel_s, ok1 ? "OK" : "FAIL", max_abs_a, max_abs_a_ref, ok1a ? "OK" : "FAIL");
    } else {
        printf("    (1) sinks: ggml vs closed max abs %.3e = %.3e of the largest |ds| (bar 1e-5), per-element max rel %.3e %s | a: not a parameter (max_bias > 0)\n",
            max_abs_s, norm_rel_s, max_rel_s, ok1 ? "OK" : "FAIL");
    }
    printf("    (2) derivation: closed vs double central difference max rel %.3e %s\n", max_rel_deriv, ok2 ? "OK" : "FAIL");
    printf("    (3) ggml float32 forward difference vs double difference, max abs over the largest (bar 5e-3) %.3e %s\n", max_rel_fwd, ok3 ? "OK" : "FAIL");
    fflush(stdout);

    ggml_free(ctx);
    return (ok1 && ok1a && ok2 && ok3) ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc != 2 || (std::string(argv[1]) != "direct" && std::string(argv[1]) != "acc")) {
        fprintf(stderr, "usage: %s direct|acc\n", argv[0]);
        return 2;
    }
    const bool use_acc = std::string(argv[1]) == "acc";
    const scase cases[] = {
        {  4,  1, 1, 1, false, 1.0f, 0.0f },
        { 16, 16, 1, 1, false, 1.0f, 0.0f },
        { 15, 15, 2, 3, true,  1.0f, 0.0f },
        { 16,  2, 32, 1, true, 0.1f, 0.0f },
        { 16, 16, 4, 2, true,  0.1f, 8.0f },
        { 64,  8, 6, 2, true,  0.125f, 0.0f },
    };
    int fails = 0, n = 0;
    for (const scase & c : cases) {
        for (unsigned seed : {1u, 2u}) {
            fails += run_case(c, use_acc, seed);
            n++;
        }
    }
    printf("%s: %d/%d cases passed\n", argv[1], n - fails, n);
    return fails == 0 ? 0 : 1;
}
