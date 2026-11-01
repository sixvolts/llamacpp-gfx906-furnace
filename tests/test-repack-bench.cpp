// Single-GPU microbenchmark for the GCN-repacked MMQ kernels (repack-gcn.cu).
//
// Weights are placed in the "<dev>_Repacked" extra buffer type, so the op runs
// mmq_gemm_*_repacked exactly as in the model. test-backend-ops cannot do this:
// it allocates weights in the default buffer type.
//
// Needs a GCN device. Run with HIP_VISIBLE_DEVICES=0 so exactly one GPU is used.
// Shapes whose type is not repacked are skipped: Q8_0 and Q5_1 are opt-in
// (GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1), and GGML_CUDA_REPACK_<type>=0
// turns a default-on type off.
//
// usage: test-repack-bench [--check] [--filter s] [--routing wiki|bench|uniform]
//        [--sigma f] [--dead f] [--ids-file f] [--dump f] [-r n] [-w n] [-t n] [--seed n]
//        [--x3 n] [--ids-view]
// --x3 n: dense src1 is [K, n, T] (several columns per slice, e.g. MTP eh_proj over hc streams)
// --ids-view: MoE ids is a row-strided view of a wider tensor, as the model passes argsort output

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// repack-gcn.cu: the same per-type switch and shape check the model loader uses
bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct params {
    bool        check    = false;
    std::string filter;
    std::string routing  = "all"; // all = wiki + bench for the Q4_K MoE, wiki elsewhere
    float       sigma    = -1.0f; // override for the routing preset
    float       dead     = -1.0f;
    std::string ids_file;
    std::string dump;
    int         reps     = 50;
    int         warmup   = 5;
    int         tokens   = 2048;
    uint64_t    seed     = 1234;
    int         x3       = 0;
    bool        ids_view = false;
};

// ---------------------------------------------------------------------
// deterministic RNG (splitmix64 + Box-Muller), independent of libstdc++

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    double uniform() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); } // (0,1)
    float normal() {
        const double u1 = uniform(), u2 = uniform();
        return (float) (std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
    double gumbel() { return -std::log(-std::log(uniform())); }
};

static void fill_normal(float * dst, size_t n, float scale, uint64_t seed) {
    rng r(seed);
    for (size_t i = 0; i < n; i++) {
        dst[i] = scale * r.normal();
    }
}

// quantize [nrow, K] rows of N(0,1)*0.02 into q, parallel over row chunks
static std::vector<uint8_t> make_weight(ggml_type type, int64_t K, int64_t nrow, uint64_t seed) {
    const size_t row_size = ggml_row_size(type, K);
    std::vector<uint8_t> q(row_size * nrow);
    ggml_quantize_init(type);
    const int64_t chunk = 256;
    const int64_t nchunk = (nrow + chunk - 1) / chunk;
    int nth = (int) std::min<unsigned>(std::max(1u, std::thread::hardware_concurrency()), 16u);
    std::vector<std::thread> th;
    for (int t = 0; t < nth; t++) {
        th.emplace_back([&, t]() {
            std::vector<float> f(chunk * K);
            for (int64_t c = t; c < nchunk; c += nth) {
                const int64_t r0 = c * chunk;
                const int64_t nr = std::min(chunk, nrow - r0);
                fill_normal(f.data(), nr * K, 0.02f, seed * 1000003ull + c);
                ggml_quantize_chunk(type, f.data(), q.data() + r0 * row_size, 0, nr, K, nullptr);
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
    return q;
}

// ---------------------------------------------------------------------
// routing

// ids [T][n_used]: per call, log p_e = sigma * z_e (a fraction `dead` of experts
// gets p = 0), then each token takes n_used distinct experts sampled without
// replacement from p (Gumbel top-k).
static std::vector<int32_t> make_ids(int n_expert, int n_used, int T, float sigma, float dead, uint64_t seed) {
    rng r(seed);
    std::vector<double> lp(n_expert);
    for (int e = 0; e < n_expert; e++) {
        lp[e] = sigma * r.normal();
    }
    std::vector<int> perm(n_expert);
    for (int e = 0; e < n_expert; e++) {
        perm[e] = e;
    }
    for (int e = n_expert - 1; e > 0; e--) {
        std::swap(perm[e], perm[r.next() % (e + 1)]);
    }
    const int n_dead = std::min(n_expert - n_used, (int) std::lround(dead * n_expert));
    for (int i = 0; i < n_dead; i++) {
        lp[perm[i]] = -1e30;
    }
    std::vector<int32_t> ids((size_t) T * n_used);
    std::vector<std::pair<double, int>> key(n_expert);
    for (int t = 0; t < T; t++) {
        for (int e = 0; e < n_expert; e++) {
            key[e] = { lp[e] + r.gumbel(), e };
        }
        std::partial_sort(key.begin(), key.begin() + n_used, key.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });
        for (int k = 0; k < n_used; k++) {
            ids[(size_t) t * n_used + k] = key[k].second;
        }
    }
    return ids;
}

static void print_routing(const std::vector<int32_t> & ids, int n_expert) {
    std::vector<int> cnt(n_expert, 0);
    for (int32_t e : ids) {
        cnt[e]++;
    }
    const int edges[] = { 0, 8, 16, 32, 64, 128, 256 };
    int hist[8] = {};
    int64_t tiles = 0;
    std::vector<int> nz;
    for (int c : cnt) {
        int b = 7;
        for (int i = 0; i < 7; i++) {
            if (c <= edges[i]) {
                b = i;
                break;
            }
        }
        hist[b]++;
        tiles += (c + 15) / 16;
        if (c > 0) {
            nz.push_back(c);
        }
    }
    std::sort(nz.begin(), nz.end());
    const int med = nz.empty() ? 0 : nz[nz.size() / 2];
    printf("  routing: tok/expert 0:%d 1-8:%d 9-16:%d 17-32:%d 33-64:%d 65-128:%d 129-256:%d >256:%d |"
           " hit %zu median %d max %d | 16-tok tiles %lld, slot eff %.1f%%\n",
        hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7],
        nz.size(), med, nz.empty() ? 0 : nz.back(), (long long) tiles,
        100.0 * ids.size() / (16.0 * std::max<int64_t>(tiles, 1)));
}

// ---------------------------------------------------------------------

struct shape {
    std::string name;
    ggml_type   type;
    int64_t     K, N;
    int         n_expert; // 0 = dense MUL_MAT
    int         n_used;
    bool        bcast;    // MoE src1 is [K, 1, T] (gate/up) instead of [K, n_used, T] (down)
    std::string routing;  // wiki / bench / uniform / file
};

static ggml_backend_buffer_type_t find_repack_buft(ggml_backend_dev_t dev) {
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    if (!get_extra) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * p = get_extra(dev); p && *p; p++) {
        const std::string n = ggml_backend_buft_name(*p);
        const std::string suf = "_Repacked";
        if (n.size() >= suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0) {
            return *p;
        }
    }
    return nullptr;
}

static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int warmup, int reps) {
    for (int i = 0; i < warmup; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) {
        ggml_backend_graph_compute_async(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

// whether a [K, N, ne2] weight of `type` would be repacked (set_tensor asserts otherwise)
static bool repack_on(ggml_type type, int64_t K, int64_t N, int64_t ne2) {
    ggml_init_params ip = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    const bool on = ggml_cuda_repack_tensor_supported(ggml_new_tensor_3d(ctx, type, K, N, ne2));
    ggml_free(ctx);
    return on;
}

static const char * repack_switch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q3_K:   return "GGML_CUDA_REPACK_Q3_K";
        case GGML_TYPE_Q4_K:   return "GGML_CUDA_REPACK_Q4_K";
        case GGML_TYPE_Q5_K:   return "GGML_CUDA_REPACK_Q5_K";
        case GGML_TYPE_Q6_K:   return "GGML_CUDA_REPACK_Q6_K";
        case GGML_TYPE_Q8_0:   return "GGML_CUDA_REPACK_Q8_0";
        case GGML_TYPE_Q5_1:   return "GGML_CUDA_REPACK_Q5_1";
        case GGML_TYPE_Q4_0:   return "GGML_CUDA_REPACK_Q4_0";
        default:               return "GGML_CUDA_REPACK_IQ";
    }
}

static bool run_shape(const params & p, ggml_backend_t backend, ggml_backend_buffer_type_t repack_buft,
        const shape & s, FILE * fdump) {
    const bool    moe = s.n_expert > 0;
    const int64_t T   = p.tokens;
    const int64_t ne2 = moe ? s.n_expert : 1;

    // routing first, so its histogram prints before the timing line
    std::vector<int32_t> ids_host;
    if (moe) {
        if (!p.ids_file.empty()) {
            ids_host.resize((size_t) T * s.n_used);
            FILE * f = fopen(p.ids_file.c_str(), "rb");
            if (!f || fread(ids_host.data(), sizeof(int32_t), ids_host.size(), f) != ids_host.size()) {
                fprintf(stderr, "cannot read %lld x %d int32 ids from %s\n", (long long) T, s.n_used, p.ids_file.c_str());
                exit(1);
            }
            fclose(f);
            for (int32_t e : ids_host) {
                GGML_ASSERT(e >= 0 && e < s.n_expert);
            }
        } else {
            float sigma = 0.0f, dead = 0.0f;
            if (s.routing == "wiki") {
                // fitted to route-stats-wiki-ub2048 (wikitext, ub 2048): ~20% experts
                // empty, median ~18, max ~1000, ~1500 tiles, ~85% slot efficiency
                sigma = 1.5f; dead = 0.17f;
            } else if (s.routing == "bench") {
                // fitted to route-stats-ub2048 (llama-bench random tokens): ~245 experts hit
                sigma = 3.3f; dead = 0.0f;
            }
            if (p.sigma >= 0.0f) { sigma = p.sigma; }
            if (p.dead  >= 0.0f) { dead  = p.dead;  }
            ids_host = make_ids(s.n_expert, s.n_used, (int) T, sigma, dead, p.seed);
        }
        print_routing(ids_host, s.n_expert);
    }

    ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead() * 2, nullptr, true };
    ggml_context * ctx_w   = ggml_init(ip);
    ggml_context * ctx_ref = ggml_init(ip);
    ggml_context * ctx_a   = ggml_init(ip);

    ggml_tensor * w     = ggml_new_tensor_3d(ctx_w,   s.type, s.K, s.N, ne2);
    ggml_tensor * w_ref = ggml_new_tensor_3d(ctx_ref, s.type, s.K, s.N, ne2);

    ggml_tensor * x;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out, * out_ref;
    if (moe) {
        x   = ggml_new_tensor_3d(ctx_a, GGML_TYPE_F32, s.K, s.bcast ? 1 : s.n_used, T);
        if (p.ids_view) {
            ggml_tensor * ids_full = ggml_new_tensor_2d(ctx_a, GGML_TYPE_I32, 4 * s.n_used, T);
            ids = ggml_view_2d(ctx_a, ids_full, s.n_used, T, ids_full->nb[1], 0);
        } else {
            ids = ggml_new_tensor_2d(ctx_a, GGML_TYPE_I32, s.n_used, T);
        }
        out     = ggml_mul_mat_id(ctx_a, w,     x, ids);
        out_ref = ggml_mul_mat_id(ctx_a, w_ref, x, ids);
    } else {
        x       = p.x3 > 0 ? ggml_new_tensor_3d(ctx_a, GGML_TYPE_F32, s.K, p.x3, T)
                           : ggml_new_tensor_2d(ctx_a, GGML_TYPE_F32, s.K, T);
        out     = ggml_mul_mat(ctx_a, w,     x);
        out_ref = ggml_mul_mat(ctx_a, w_ref, x);
    }

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, repack_buft);
    GGML_ASSERT(buf_w);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx_a, backend);
    GGML_ASSERT(buf_a);

    {
        std::vector<uint8_t> q = make_weight(s.type, s.K, s.N * ne2, p.seed + 17);
        GGML_ASSERT(q.size() == ggml_nbytes(w));
        ggml_backend_tensor_set(w, q.data(), 0, q.size()); // repacks on the host
        if (p.check) {
            ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors_from_buft(
                ctx_ref, ggml_backend_get_default_buffer_type(backend));
            GGML_ASSERT(buf_ref);
            ggml_backend_buffer_set_usage(buf_ref, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(w_ref, q.data(), 0, q.size());
        }
    }
    {
        std::vector<float> xf(ggml_nelements(x));
        fill_normal(xf.data(), xf.size(), 1.0f, p.seed + 29);
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        if (moe) {
            if (p.ids_view) {
                for (int64_t t = 0; t < T; t++) {
                    ggml_backend_tensor_set(ids, ids_host.data() + t * s.n_used, t * ids->nb[1], s.n_used * sizeof(int32_t));
                }
            } else {
                ggml_backend_tensor_set(ids, ids_host.data(), 0, ggml_nbytes(ids));
            }
        }
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_a, 4, false);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_supports_op(backend, out));

    const double us   = time_graph(backend, gf, p.warmup, p.reps);
    const double macs = (double) s.K * s.N * T * (moe ? s.n_used : std::max(1, p.x3));

    char tname[16];
    snprintf(tname, sizeof(tname), "%s", ggml_type_name(s.type));
    printf("%-36s %-5s %10.1f us/op %7.2f TMAC/s", s.name.c_str(), tname, us, macs / us * 1e-6);

    bool ok = true;
    std::vector<float> y(ggml_nelements(out));
    if (p.check || fdump) {
        ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    }
    if (p.check) {
        ggml_cgraph * gr = ggml_new_graph_custom(ctx_a, 4, false);
        ggml_build_forward_expand(gr, out_ref);
        const double us_ref = time_graph(backend, gr, p.warmup, p.reps);
        std::vector<float> yr(ggml_nelements(out_ref));
        ggml_backend_tensor_get(out_ref, yr.data(), 0, ggml_nbytes(out_ref));
        double err = 0.0, ref = 0.0;
        size_t nonfinite = 0;
        for (size_t i = 0; i < y.size(); i++) {
            if (!std::isfinite(y[i])) {
                nonfinite++;
                continue;
            }
            const double d = (double) y[i] - yr[i];
            err += d * d;
            ref += (double) yr[i] * yr[i];
        }
        const double nmse = err / std::max(ref, 1e-30);
        ok = nonfinite == 0 && nmse <= 5e-4;
        printf("  nmse %.3e  (canonical %10.1f us/op %7.2f TMAC/s)%s", nmse, us_ref, macs / us_ref * 1e-6,
            ok ? "" : (nonfinite ? "  FAIL non-finite" : "  FAIL"));
    }
    printf("\n");
    fflush(stdout);
    if (fdump) {
        fwrite(y.data(), sizeof(float), y.size(), fdump);
    }

    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_w);
    if (w_ref->buffer) {
        ggml_backend_buffer_free(w_ref->buffer);
    }
    ggml_free(ctx_a);
    ggml_free(ctx_ref);
    ggml_free(ctx_w);
    return ok;
}

// One MoE layer: gate, up (Q4_K, shared src1) and down (Q5_1) with shared ids,
// once as a single 3-node graph and once as 3 separate graphs. The backend
// caches routing and quantized activations within one graph; outputs must
// be bitwise identical.
static bool run_moe_layer(const params & p, ggml_backend_t backend, ggml_backend_buffer_type_t repack_buft,
        const std::string & routing) {
    const int64_t T = p.tokens, K = 2560, N = 640;
    const int n_expert = 512, n_used = 10;
    float sigma = routing == "bench" ? 3.3f : 1.5f, dead = routing == "bench" ? 0.0f : 0.17f;
    if (p.sigma >= 0.0f) { sigma = p.sigma; }
    if (p.dead  >= 0.0f) { dead  = p.dead;  }
    std::vector<int32_t> ids_host = make_ids(n_expert, n_used, (int) T, sigma, dead, p.seed);
    print_routing(ids_host, n_expert);

    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead() * 8, nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);
    ggml_context * ctx_a = ggml_init(ip);
    ggml_tensor * wg  = ggml_new_tensor_3d(ctx_w, GGML_TYPE_Q4_K, K, N, n_expert);
    ggml_tensor * wu  = ggml_new_tensor_3d(ctx_w, GGML_TYPE_Q4_K, K, N, n_expert);
    ggml_tensor * wd  = ggml_new_tensor_3d(ctx_w, GGML_TYPE_Q5_1, N, K, n_expert);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx_a, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * xd  = ggml_new_tensor_3d(ctx_a, GGML_TYPE_F32, N, n_used, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx_a, GGML_TYPE_I32, n_used, T);
    ggml_tensor * og  = ggml_mul_mat_id(ctx_a, wg, x,  ids);
    ggml_tensor * ou  = ggml_mul_mat_id(ctx_a, wu, x,  ids);
    ggml_tensor * od  = ggml_mul_mat_id(ctx_a, wd, xd, ids);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, repack_buft);
    GGML_ASSERT(buf_w);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_t buf_a = ggml_backend_alloc_ctx_tensors(ctx_a, backend);
    GGML_ASSERT(buf_a);
    {
        std::vector<uint8_t> q = make_weight(GGML_TYPE_Q4_K, K, N * n_expert, p.seed + 17);
        ggml_backend_tensor_set(wg, q.data(), 0, q.size());
        q = make_weight(GGML_TYPE_Q4_K, K, N * n_expert, p.seed + 18);
        ggml_backend_tensor_set(wu, q.data(), 0, q.size());
        q = make_weight(GGML_TYPE_Q5_1, N, K * n_expert, p.seed + 19);
        ggml_backend_tensor_set(wd, q.data(), 0, q.size());
        std::vector<float> xf(ggml_nelements(x));
        fill_normal(xf.data(), xf.size(), 1.0f, p.seed + 29);
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        xf.resize(ggml_nelements(xd));
        fill_normal(xf.data(), xf.size(), 1.0f, p.seed + 31);
        ggml_backend_tensor_set(xd, xf.data(), 0, ggml_nbytes(xd));
        ggml_backend_tensor_set(ids, ids_host.data(), 0, ggml_nbytes(ids));
    }
    auto get_all = [&]() {
        std::vector<float> y(ggml_nelements(og) + ggml_nelements(ou) + ggml_nelements(od));
        ggml_backend_tensor_get(og, y.data(), 0, ggml_nbytes(og));
        ggml_backend_tensor_get(ou, y.data() + ggml_nelements(og), 0, ggml_nbytes(ou));
        ggml_backend_tensor_get(od, y.data() + ggml_nelements(og) + ggml_nelements(ou), 0, ggml_nbytes(od));
        return y;
    };

    ggml_cgraph * g1[3];
    ggml_tensor * outs[3] = { og, ou, od };
    for (int i = 0; i < 3; i++) {
        g1[i] = ggml_new_graph_custom(ctx_a, 4, false);
        ggml_build_forward_expand(g1[i], outs[i]);
    }
    ggml_cgraph * g3 = ggml_new_graph_custom(ctx_a, 8, false);
    for (int i = 0; i < 3; i++) {
        ggml_build_forward_expand(g3, outs[i]);
    }

    double us_sep = 0.0;
    for (int i = 0; i < 3; i++) {
        us_sep += time_graph(backend, g1[i], p.warmup, p.reps);
    }
    const std::vector<float> y_sep = get_all();
    const double us_one = time_graph(backend, g3, p.warmup, p.reps);
    const std::vector<float> y_one = get_all();
    const bool same = memcmp(y_sep.data(), y_one.data(), y_sep.size() * sizeof(float)) == 0;
    printf("%-36s 3 graphs %10.1f us | 1 graph %10.1f us | %+.1f%% | %s\n",
        ("moe_layer [" + routing + "]").c_str(), us_sep, us_one, 100.0 * (us_one / us_sep - 1.0),
        same ? "bitwise identical" : "MISMATCH");
    fflush(stdout);

    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_a);
    ggml_free(ctx_w);
    return same;
}

int main(int argc, char ** argv) {
    params p;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "--check")    { p.check = true; }
        else if (a == "--filter")   { p.filter = next(); }
        else if (a == "--routing")  { p.routing = next(); }
        else if (a == "--sigma")    { p.sigma = atof(next()); }
        else if (a == "--dead")     { p.dead = atof(next()); }
        else if (a == "--ids-file") { p.ids_file = next(); }
        else if (a == "--dump")     { p.dump = next(); }
        else if (a == "-r")         { p.reps = atoi(next()); }
        else if (a == "-w")         { p.warmup = atoi(next()); }
        else if (a == "-t")         { p.tokens = atoi(next()); }
        else if (a == "--seed")     { p.seed = strtoull(next(), nullptr, 10); }
        else if (a == "--x3")       { p.x3 = atoi(next()); }
        else if (a == "--ids-view") { p.ids_view = true; }
        else {
            fprintf(stderr, "usage: %s [--check] [--filter s] [--routing wiki|bench|uniform] [--sigma f] [--dead f]\n"
                            "       [--ids-file f] [--dump f] [-r reps] [-w warmup] [-t tokens] [--seed n] [--x3 n] [--ids-view]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        fprintf(stderr, "no GPU device\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t repack_buft = find_repack_buft(dev);
    if (!repack_buft) {
        fprintf(stderr, "no *_Repacked buffer type (GGML_CUDA_REPACK=0 or not a GCN device)\n");
        return 1;
    }
    printf("device %s, buft %s, T=%d, reps %d\n", ggml_backend_dev_description(dev),
        ggml_backend_buft_name(repack_buft), p.tokens, p.reps);

    std::vector<shape> shapes;
    const bool file_ids = !p.ids_file.empty();
    auto moe_routings = [&](bool primary) {
        std::vector<std::string> r;
        if (file_ids) {
            r.push_back("file");
        } else if (p.routing != "all") {
            r.push_back(p.routing);
        } else {
            r.push_back("wiki");
            if (primary) {
                r.push_back("bench");
            }
        }
        return r;
    };
    for (const auto & r : moe_routings(true)) {
        shapes.push_back({ "moe_gate_up 2560x640x512 [" + r + "]", GGML_TYPE_Q4_K, 2560, 640, 512, 10, true, r });
    }
    for (const auto & r : moe_routings(false)) {
        shapes.push_back({ "moe_down 640x2560x512 [" + r + "]", GGML_TYPE_Q5_1, 640, 2560, 512, 10, false, r });
        // Qwen 3.6-35B-A3B class experts: 128 experts, 8 used, K = 768 (24 sub-blocks)
        shapes.push_back({ "moe_down 768x2048x128 [" + r + "]", GGML_TYPE_Q5_K, 768, 2048, 128, 8, false, r });
        shapes.push_back({ "moe_down 768x2048x128 [" + r + "]", GGML_TYPE_Q6_K, 768, 2048, 128, 8, false, r });
        shapes.push_back({ "moe_down 768x2048x128 [" + r + "]", GGML_TYPE_Q4_K, 768, 2048, 128, 8, false, r });
    }
    const int64_t dense[][2] = {
        { 2560, 10240 }, { 6144, 2560 }, { 2560, 6144 }, { 320, 10240 }, { 10240, 320 },
        { 2560, 12288 }, { 2560, 640 }, { 640, 2560 }, { 2560, 512 },
        { 2816, 4096 }, { 2816, 2048 }, { 4096, 2816 }, { 2816, 2112 }, { 2112, 2816 }, // Gemma 26B-A4B dense
        { 4096, 2048 }, { 2048, 4096 }, // Qwen 3.6-35B-A3B ssm_out / attn_output and in_proj class
        { 2560, 248320 }, // Flash-Next LM head
    };
    for (const auto & d : dense) {
        shapes.push_back({ "dense " + std::to_string(d[0]) + "x" + std::to_string(d[1]),
            GGML_TYPE_Q8_0, d[0], d[1], 0, 0, false, "" });
    }
    // dense K-quants (27B / Gemma 31B shapes) for the R10 loads-first matvecs
    const int64_t kq_dense[][2] = { { 5120, 17408 }, { 17408, 5120 }, { 5120, 6144 }, { 2560, 6144 } };
    for (const auto & d : kq_dense) {
        for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
            shapes.push_back({ "dense " + std::to_string(d[0]) + "x" + std::to_string(d[1]),
                t, d[0], d[1], 0, 0, false, "" });
        }
    }
    // nibble-plane formats (Q4_0 now; the IQ4 family relabels onto the same kernels): the 27B/31B
    // dense shapes plus a Gemma-31B-class FFN
    const int64_t nib_dense[][2] = { { 5120, 17408 }, { 17408, 5120 }, { 5120, 6144 }, { 5376, 21504 }, { 2560, 6144 } };
    for (const auto & d : nib_dense) {
        for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_Q3_K }) {
            if (d[0] % 256 != 0 && t != GGML_TYPE_Q4_0 && t != GGML_TYPE_IQ4_NL) {
                continue;
            }
            shapes.push_back({ "dense " + std::to_string(d[0]) + "x" + std::to_string(d[1]),
                t, d[0], d[1], 0, 0, false, "" });
        }
    }

    FILE * fdump = nullptr;
    if (!p.dump.empty()) {
        fdump = fopen(p.dump.c_str(), "wb");
        GGML_ASSERT(fdump);
    }

    bool all_ok = true;
    for (const auto & s : shapes) {
        if (!p.filter.empty() && s.name.find(p.filter) == std::string::npos &&
                std::string(ggml_type_name(s.type)).find(p.filter) == std::string::npos) {
            continue;
        }
        // a type whose repack is off (Q8_0 / Q5_1 are opt-in) would make set_tensor assert
        if (!repack_on(s.type, s.K, s.N, s.n_expert > 0 ? s.n_expert : 1)) {
            printf("%-36s %-5s skipped (repack off: %s)\n", s.name.c_str(), ggml_type_name(s.type), repack_switch(s.type));
            continue;
        }
        all_ok &= run_shape(p, backend, repack_buft, s, fdump);
    }

    // opt-in only (--filter moe_layer), so the default dump layout is unchanged
    if (p.filter.find("moe_layer") != std::string::npos) {
        // Q4_K gate/up [2560, 640, 512] and Q5_1 down [640, 2560, 512], as in run_moe_layer
        if (!repack_on(GGML_TYPE_Q4_K, 2560, 640, 512) || !repack_on(GGML_TYPE_Q5_1, 640, 2560, 512)) {
            printf("moe_layer skipped (repack off: GGML_CUDA_REPACK_Q4_K or GGML_CUDA_REPACK_Q5_1)\n");
        } else {
            for (const auto & r : moe_routings(true)) {
                all_ok &= run_moe_layer(p, backend, repack_buft, r);
            }
        }
    }

    if (fdump) {
        fclose(fdump);
    }
    ggml_backend_free(backend);
    if (p.check) {
        printf(all_ok ? "check: OK\n" : "check: FAILED\n");
    }
    return all_ok ? 0 : 1;
}
