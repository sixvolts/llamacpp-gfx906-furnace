#include "qsa-mask.cuh"
#include "top-k.cuh"

// GGML_OP_QSA_MASK: the attention mask of a qwen4exp block-sparse (QSA) layer from the indexer's block scores.
// For query row i, a cell j the KQ mask keeps is worth score[cell_blk[j], i]; the `width` most valuable cells (ties in
// ascending cell index) keep their mask value and every other cell becomes -inf. This is the mask the unfused graph
// builds by expanding the scores to every cell ([n_kv, n_q] f32), adding the mask, a radix top-k over the cells and
// a fill / set_rows / add to turn the selection back into a mask - here without the per-cell intermediates:
//   1. k_qsa_mask_weight: each block's number of visible cells (one pass over the cells)
//   2. k_qsa_mask_select: a weighted radix select over the blocks finds the threshold key T and how many cells at T
//      are taken (one workgroup per row, passes over the blocks only)
//   3. k_qsa_mask_eq:     cells at T per chunk of cells, for their rank in ascending index (only rows that cut at T)
//   4. k_qsa_mask_write:  the mask
// The cells' value is score + mask with the mask 0 where it keeps a cell, as the qwen4exp causal mask is; a mask with
// other finite values would make a cell's value differ from its block's, which the per-block select cannot see.

#define QSA_MASK_NT    256
#define QSA_MASK_CHUNK 4096   // cells per workgroup in the cell passes: 16 rounds of 256

struct qsa_mask_state {
    int      all;    // the row keeps every visible cell
    uint32_t T;      // threshold key
    int      need;   // cells at T to take, in ascending index
    int      all_T;  // every cell at T is taken (no ranking needed)
};

static __device__ __forceinline__ uint32_t qsa_mask_key(const float x) {
    const uint32_t u = __float_as_uint(x + 0.0f);   // -0 -> +0, as score + a 0 mask is
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}

template <typename T>
static __device__ __forceinline__ bool qsa_mask_keeps(const T m) {
    return (float) m != -INFINITY;
}

// exclusive scan over the workgroup; *total gets the sum
static __device__ int qsa_mask_scan(const int v, int * sh, int * total) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int off = 1; off < QSA_MASK_NT; off *= 2) {
        const int t = tid >= off ? sh[tid - off] : 0;
        __syncthreads();
        sh[tid] += t;
        __syncthreads();
    }
    const int incl = sh[tid];
    *total = sh[QSA_MASK_NT - 1];
    __syncthreads();
    return incl - v;
}

struct qsa_mask_args {
    const float   * score;    // row i: score + (i % n_tps)*s_t + (i / n_tps)*s_s
    const int32_t * cell_blk; // stream s: cell_blk + s*cb_s
    const char    * mask;     // row i: mask + (i % n_tps)*m_t + (i / n_tps)*m_s (bytes)
    char          * dst;      // as mask, dst_t / dst_s
    int64_t s_t, s_s, cb_s, m_t, m_s, d_t, d_s;
    int     n_kv;
    int     n_blocks;
    int     n_tps;
    int     width;
    int     row0;             // first row of this launch
};

template <typename T>
static __device__ __forceinline__ const T * qsa_mask_row(const qsa_mask_args & a, const int i) {
    return (const T *) (a.mask + (int64_t) (i % a.n_tps)*a.m_t + (int64_t) (i / a.n_tps)*a.m_s);
}

// 1. weights: w[row][b] = cells of block b the mask keeps
template <typename T>
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_weight(const qsa_mask_args a, uint32_t * __restrict__ w) {
    const int ir = blockIdx.y;
    const int i  = a.row0 + ir;
    const T       * m  = qsa_mask_row<T>(a, i);
    const int32_t * cb = a.cell_blk + (int64_t) (i / a.n_tps)*a.cb_s;
    uint32_t      * wr = w + (int64_t) ir*a.n_blocks;

    const int j0 = blockIdx.x*QSA_MASK_CHUNK;
    const int j1 = min(a.n_kv, j0 + QSA_MASK_CHUNK);
    for (int j = j0 + threadIdx.x; j < j1; j += QSA_MASK_NT) {
        if (qsa_mask_keeps(m[j])) {
            atomicAdd(&wr[cb[j]], 1u);
        }
    }
}

// one radix pass over the digit [shift, shift + BITS) of the keys matching the resolved prefix: the weighted histogram,
// then the bin holding the need-th heaviest cell (searching from the top). Resolves BITS more bits of T
template <int BITS>
static __device__ void qsa_mask_pass(const float * __restrict__ srow, const uint32_t * __restrict__ wr, const int n_blocks,
        const int shift, uint32_t * hist, int * sh, uint32_t * prefix, uint32_t * pmask, int * need, int * w_bin) {
    constexpr int NBINS = 1 << BITS;
    constexpr int PER   = NBINS / QSA_MASK_NT;
    static_assert(NBINS % QSA_MASK_NT == 0, "bins must split evenly over the threads");
    const int tid = threadIdx.x;

    for (int k = tid; k < NBINS; k += QSA_MASK_NT) {
        hist[k] = 0;
    }
    __syncthreads();
    const uint32_t pf = *prefix;
    const uint32_t pm = *pmask;
    for (int b = tid; b < n_blocks; b += QSA_MASK_NT) {
        const uint32_t wb = wr[b];
        if (wb == 0) {
            continue;
        }
        const uint32_t key = qsa_mask_key(srow[b]);
        if ((key & pm) == pf) {
            atomicAdd(&hist[(key >> shift) & (NBINS - 1)], wb);
        }
    }
    __syncthreads();

    // thread t owns bins [NBINS - PER*(t+1), NBINS - PER*t), scanned from the top
    const int top = NBINS - PER*tid - 1;
    int own = 0;
    for (int k = 0; k < PER; ++k) {
        own += (int) hist[top - k];
    }
    int tot;
    const int above = qsa_mask_scan(own, sh, &tot);
    const int nd = *need;
    if (above < nd && above + own >= nd) {
        int acc = above;
        int bin = top - PER + 1;
        for (int k = 0; k < PER; ++k) {
            if (acc + (int) hist[top - k] >= nd) {
                bin = top - k;
                break;
            }
            acc += (int) hist[top - k];
        }
        *prefix = pf | ((uint32_t) bin << shift);
        *pmask  = pm | ((uint32_t) (NBINS - 1) << shift);
        *need   = nd - acc;
        *w_bin  = (int) hist[bin];
    }
    __syncthreads();
}

// 2. per row: the threshold key and the cells to take at it
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_select(const qsa_mask_args a, const uint32_t * __restrict__ w,
        qsa_mask_state * __restrict__ st) {
    const int ir  = blockIdx.x;
    const int i   = a.row0 + ir;
    const int tid = threadIdx.x;
    const float    * srow = a.score + (int64_t) (i % a.n_tps)*a.s_t + (int64_t) (i / a.n_tps)*a.s_s;
    const uint32_t * wr   = w + (int64_t) ir*a.n_blocks;

    __shared__ uint32_t hist[4096];
    __shared__ int      sh[QSA_MASK_NT];
    __shared__ uint32_t prefix, pmask;
    __shared__ int      need, w_bin;

    int part = 0;
    for (int b = tid; b < a.n_blocks; b += QSA_MASK_NT) {
        part += (int) wr[b];
    }
    int total;
    qsa_mask_scan(part, sh, &total);

    if (total <= a.width) {
        if (tid == 0) {
            st[ir] = { 1, 0u, 0, 1 };
        }
        return;
    }

    if (tid == 0) {
        prefix = 0;
        pmask  = 0;
        need   = a.width;
        w_bin  = 0;
    }
    __syncthreads();

    qsa_mask_pass<12>(srow, wr, a.n_blocks, 20, hist, sh, &prefix, &pmask, &need, &w_bin);
    qsa_mask_pass<12>(srow, wr, a.n_blocks,  8, hist, sh, &prefix, &pmask, &need, &w_bin);
    qsa_mask_pass< 8>(srow, wr, a.n_blocks,  0, hist, sh, &prefix, &pmask, &need, &w_bin);

    if (tid == 0) {
        st[ir] = { 0, prefix, need, need >= w_bin ? 1 : 0 };
    }
}

// the visible cells at key T in the current round of 256 cells: this lane's flag, the round's lanes before it, the
// round's total (a wave-ordered prefix, so ranks follow the cell index)
static __device__ __forceinline__ int qsa_mask_round_prefix(const bool f, int * wsum, int * total) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    constexpr int NW = QSA_MASK_NT / WS;
    const int lane   = threadIdx.x % WS;
    const int wid    = threadIdx.x / WS;
#ifdef GGML_USE_HIP
    const uint64_t bal  = __ballot(f);
    const int      in_w = __popcll(bal & ((uint64_t(1) << lane) - 1));
    const int      n_w  = __popcll(bal);
#else
    const uint32_t bal  = __ballot_sync(0xffffffff, f);
    const int      in_w = __popc(bal & ((1u << lane) - 1));
    const int      n_w  = __popc(bal);
#endif
    if (lane == 0) {
        wsum[wid] = n_w;
    }
    __syncthreads();
    int before = 0;
    int all    = 0;
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        before += k < wid ? wsum[k] : 0;
        all    += wsum[k];
    }
    __syncthreads();
    *total = all;
    return before + in_w;
}

// 3. cells at T per chunk of cells (rows that take only part of them)
template <typename T>
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_eq(const qsa_mask_args a, const qsa_mask_state * __restrict__ st,
        int * __restrict__ counts) {
    const int ir = blockIdx.y;
    const int i  = a.row0 + ir;
    const qsa_mask_state s = st[ir];
    if (s.all || s.all_T) {
        return;
    }
    const T       * m    = qsa_mask_row<T>(a, i);
    const int32_t * cb   = a.cell_blk + (int64_t) (i / a.n_tps)*a.cb_s;
    const float   * srow = a.score + (int64_t) (i % a.n_tps)*a.s_t + (int64_t) (i / a.n_tps)*a.s_s;

    __shared__ int sh[QSA_MASK_NT];
    const int j0 = blockIdx.x*QSA_MASK_CHUNK;
    const int j1 = min(a.n_kv, j0 + QSA_MASK_CHUNK);
    int n = 0;
    for (int j = j0 + threadIdx.x; j < j1; j += QSA_MASK_NT) {
        n += qsa_mask_keeps(m[j]) && qsa_mask_key(srow[cb[j]]) == s.T;
    }
    int total;
    qsa_mask_scan(n, sh, &total);
    if (threadIdx.x == 0) {
        counts[(int64_t) ir*gridDim.x + blockIdx.x] = total;
    }
}

// 4. the mask
template <typename T>
static __global__ void __launch_bounds__(QSA_MASK_NT) k_qsa_mask_write(const qsa_mask_args a, const qsa_mask_state * __restrict__ st,
        const int * __restrict__ counts) {
    const int ir = blockIdx.y;
    const int i  = a.row0 + ir;
    const qsa_mask_state s = st[ir];
    const T       * m    = qsa_mask_row<T>(a, i);
    const int32_t * cb   = a.cell_blk + (int64_t) (i / a.n_tps)*a.cb_s;
    const float   * srow = a.score + (int64_t) (i % a.n_tps)*a.s_t + (int64_t) (i / a.n_tps)*a.s_s;
    T             * d    = (T *) (a.dst + (int64_t) (i % a.n_tps)*a.d_t + (int64_t) (i / a.n_tps)*a.d_s);

    const int j0 = blockIdx.x*QSA_MASK_CHUNK;
    const int j1 = min(a.n_kv, j0 + QSA_MASK_CHUNK);
    const T drop = (T) -INFINITY;

    if (s.all) {
        for (int j = j0 + threadIdx.x; j < j1; j += QSA_MASK_NT) {
            d[j] = m[j];
        }
        return;
    }

    const bool rank = !s.all_T;
    int taken = 0;   // cells at T in earlier chunks and rounds
    if (rank) {
        for (int k = 0; k < (int) blockIdx.x; ++k) {
            taken += counts[(int64_t) ir*gridDim.x + k];
        }
    }

    __shared__ int wsum[QSA_MASK_NT / ggml_cuda_get_physical_warp_size()];
    // every thread runs the same number of rounds, so the round prefix's barriers line up
    for (int base = j0; base < j1; base += QSA_MASK_NT) {
        const int  j   = base + threadIdx.x;
        const bool in  = j < j1;
        const T    mj  = in ? m[j] : drop;
        const bool vis = in && qsa_mask_keeps(mj);
        const uint32_t key = vis ? qsa_mask_key(srow[cb[j]]) : 0u;
        const bool eq  = vis && key == s.T;
        bool keep = vis && key > s.T;
        if (rank) {
            int n_round;
            const int r = qsa_mask_round_prefix(eq, wsum, &n_round);
            keep = keep || (eq && taken + r < s.need);
            taken += n_round;
        } else {
            keep = keep || eq;
        }
        if (in) {
            d[j] = keep ? mj : drop;
        }
    }
}

template <typename T>
static void qsa_mask_launch(ggml_backend_cuda_context & ctx, qsa_mask_args a, const int n_rows) {
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    const int n_cc = (a.n_kv + QSA_MASK_CHUNK - 1)/QSA_MASK_CHUNK;

    // rows per launch: the block weights of a launch stay within 32 MiB
    const int64_t per_row = (int64_t) a.n_blocks*sizeof(uint32_t);
    const int rc = (int) std::max<int64_t>(1, std::min<int64_t>(n_rows, (int64_t(32) << 20)/std::max<int64_t>(1, per_row)));

    ggml_cuda_pool_alloc<uint32_t>       w     (pool, (size_t) rc*a.n_blocks);
    ggml_cuda_pool_alloc<qsa_mask_state> st    (pool, (size_t) rc);
    ggml_cuda_pool_alloc<int>            counts(pool, (size_t) rc*n_cc);

    for (int r0 = 0; r0 < n_rows; r0 += rc) {
        const int nr = std::min(rc, n_rows - r0);
        a.row0 = r0;
        CUDA_CHECK(cudaMemsetAsync(w.get(), 0, (size_t) nr*a.n_blocks*sizeof(uint32_t), stream));
        k_qsa_mask_weight<T><<<dim3(n_cc, nr, 1), QSA_MASK_NT, 0, stream>>>(a, w.get());
        k_qsa_mask_select   <<<nr,                QSA_MASK_NT, 0, stream>>>(a, w.get(), st.get());
        k_qsa_mask_eq<T>    <<<dim3(n_cc, nr, 1), QSA_MASK_NT, 0, stream>>>(a, st.get(), counts.get());
        k_qsa_mask_write<T> <<<dim3(n_cc, nr, 1), QSA_MASK_NT, 0, stream>>>(a, st.get(), counts.get());
    }
    CUDA_CHECK(cudaGetLastError());
}

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
// GGML_CUDA_QSA_MASK_REF=1: the unfused computation with its own kernels (expand, radix top-k, mask rebuild), to tell
// a selection difference from anything else
template <typename T>
static __global__ void k_qsa_mask_ref_expand(const qsa_mask_args a, float * __restrict__ ex) {
    const int i = blockIdx.y;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= a.n_kv) {
        return;
    }
    const T       * m    = qsa_mask_row<T>(a, i);
    const int32_t * cb   = a.cell_blk + (int64_t) (i / a.n_tps)*a.cb_s;
    const float   * srow = a.score + (int64_t) (i % a.n_tps)*a.s_t + (int64_t) (i / a.n_tps)*a.s_s;
    ex[(int64_t) i*a.n_kv + j] = srow[cb[j]] + (float) m[j];
}

template <typename T>
static __global__ void k_qsa_mask_ref_fill(const qsa_mask_args a) {
    const int i = blockIdx.y;
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= a.n_kv) {
        return;
    }
    T * d = (T *) (a.dst + (int64_t) (i % a.n_tps)*a.d_t + (int64_t) (i / a.n_tps)*a.d_s);
    d[j] = (T) -INFINITY;
}

template <typename T>
static __global__ void k_qsa_mask_ref_set(const qsa_mask_args a, const int * __restrict__ idx, const int width) {
    const int i = blockIdx.y;
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= width) {
        return;
    }
    const T * m = qsa_mask_row<T>(a, i);
    T       * d = (T *) (a.dst + (int64_t) (i % a.n_tps)*a.d_t + (int64_t) (i / a.n_tps)*a.d_s);
    const int j = idx[(int64_t) i*width + k];
    d[j] = (T) (0.0f + (float) m[j]);
}

template <typename T>
static void qsa_mask_ref(ggml_backend_cuda_context & ctx, const qsa_mask_args & a, const int n_rows) {
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> ex (ctx.pool(), (size_t) n_rows*a.n_kv);
    ggml_cuda_pool_alloc<int>   idx(ctx.pool(), (size_t) n_rows*a.width);
    const dim3 g((a.n_kv + 255)/256, n_rows, 1);
    k_qsa_mask_ref_expand<T><<<g, 256, 0, stream>>>(a, ex.get());
    ggml_cuda_top_k_rows_f32(ctx.pool(), ex.get(), idx.get(), a.n_kv, n_rows, a.width, stream);
    k_qsa_mask_ref_fill<T><<<g, 256, 0, stream>>>(a);
    k_qsa_mask_ref_set<T><<<dim3((a.width + 255)/256, n_rows, 1), 256, 0, stream>>>(a, idx.get(), a.width);
    CUDA_CHECK(cudaGetLastError());
}
#endif

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * mask     = dst->src[2];

    GGML_ASSERT(score->type == GGML_TYPE_F32 && cell_blk->type == GGML_TYPE_I32 && mask->type == dst->type);
    GGML_ASSERT(score->nb[0] == sizeof(float) && cell_blk->nb[0] == sizeof(int32_t) && mask->nb[0] == ggml_type_size(mask->type));

    qsa_mask_args a;
    a.score    = (const float *) score->data;
    a.cell_blk = (const int32_t *) cell_blk->data;
    a.mask     = (const char *) mask->data;
    a.dst      = (char *) dst->data;
    a.s_t      = score->nb[1]/sizeof(float);
    a.s_s      = score->nb[2]/sizeof(float);
    a.cb_s     = cell_blk->nb[1]/sizeof(int32_t);
    a.m_t      = mask->nb[1];
    a.m_s      = mask->nb[3];
    a.d_t      = dst->nb[1];
    a.d_s      = dst->nb[3];
    a.n_kv     = (int) mask->ne[0];
    a.n_blocks = (int) score->ne[0];
    a.n_tps    = (int) mask->ne[1];
    a.width    = ggml_get_op_params_i32(dst, 0);
    a.row0     = 0;

    const int n_rows = (int) (mask->ne[1]*mask->ne[3]);


#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
    static const bool ref = getenv("GGML_CUDA_QSA_MASK_REF") != nullptr;
    if (ref) {
        if (dst->type == GGML_TYPE_F16) {
            qsa_mask_ref<half>(ctx, a, n_rows);
        } else {
            qsa_mask_ref<float>(ctx, a, n_rows);
        }
        return;
    }
#endif

    if (dst->type == GGML_TYPE_F16) {
        qsa_mask_launch<half>(ctx, a, n_rows);
    } else {
        qsa_mask_launch<float>(ctx, a, n_rows);
    }
}
