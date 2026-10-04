#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_rows,
                                     int64_t       state_row_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // state_rows: read s0 straight from the recurrent cache rows (the gather is elided)
    const int64_t state_in_offset      = (state_rows ? (int64_t) state_rows[sequence] * state_row_stride
                                                     : (int64_t) sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

#if defined(GGML_USE_HIP)
// Prefill variant for wave64 GPUs (S_v = 128, scalar gate). One 64-thread workgroup owns a 16-column
// slab of one head's state and keeps it in LDS across all tokens; 4 lane groups of 16 split the k
// dimension, reduced with two shuffles. The next token's q/k/v/g/beta are prefetched into registers
// while the current one computes. All H * 8 workgroups are co-resident, so the serial token loop runs
// once; one wave per column (the kernel above) leaves most of its waves queued behind the loop.
// The slab is XOR-swizzled (element (c, kk) at c*128 + (kk ^ 2c)) so the 16 columns hit 16 banks.
// Adapted from reinstinct's gdn_recurrent_batched_v2.
#define GDN_LDS_HD 128

// The workgroup is exactly one wave64 and a wave's LDS operations complete in order, so ordering the
// compiler is enough. __syncthreads() would also wait for all outstanding global loads (its fence
// drains vmcnt), stalling every token on the prefetch it is meant to hide.
static __device__ __forceinline__ void gdn_wave_sync() {
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

template <int COLS, bool keep_rs_t>
__global__ void __launch_bounds__(64) gated_delta_net_lds_wave64(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
        const int64_t H, const int64_t n_tokens,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic, const float scale,
        const int64_t state_slot_stride, const int K,
        const int32_t * __restrict__ state_rows, const int64_t state_row_stride) {
    constexpr int HD    = GDN_LDS_HD;
    constexpr int NG    = 64 / COLS;   // lane groups splitting the k dimension
    constexpr int PER_G = HD / NG;     // k elements per thread
    constexpr int PER_T = HD / 64;     // q/k elements each thread stages
    static_assert(COLS * NG == 64 && HD % NG == 0, "bad slab width");

    __shared__ float st[COLS * HD];
    __shared__ float q_lds[HD];
    __shared__ float k_lds[HD];

    const uint32_t h         = blockIdx.x;
    const uint32_t sequence  = blockIdx.y;
    const int      tile_base = blockIdx.z * COLS;
    const int      tid       = threadIdx.x;
    const int      grp       = tid / COLS;
    const int      lvv       = tid % COLS;
    const int      col       = tile_base + lvv;
    const int      swz       = 2 * lvv;

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    // state is [col][kk] (kk contiguous) per (seq, head)
    const float * s_in  = curr_state + (state_rows ? (int64_t) state_rows[sequence] * state_row_stride
                                                   : (int64_t) sequence * H * HD * HD) + (int64_t) h * HD * HD;
    float *       s_out = state + ((int64_t) sequence * H + h) * HD * HD;

    for (int i = tid; i < COLS * HD; i += 64) {
        const int c = i / HD, kk = i % HD;
        st[c * HD + (kk ^ (2 * c))] = s_in[(int64_t) (tile_base + c) * HD + kk];
    }

    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h * sv1 + col;
    const int64_t gb     = sequence * sb3 + h * sb1;
    float *       out    = dst + ((int64_t) sequence * n_tokens * H + h) * HD + col;

    float pq[PER_T], pk[PER_T];
#pragma unroll
    for (int i = 0; i < PER_T; i++) {
        pq[i] = q_base[tid + i * 64];
        pk[i] = k_base[tid + i * 64];
    }
    float pv = v_base[0];
    float pg = g[gb];
    float pb = beta[gb];

    float * st_col = st + lvv * HD;

    for (int64_t t = 0; t < n_tokens; t++) {
        gdn_wave_sync(); // token t-1 is done reading q/k
#pragma unroll
        for (int i = 0; i < PER_T; i++) {
            q_lds[tid + i * 64] = pq[i];
            k_lds[tid + i * 64] = pk[i];
        }
        const float v_t = pv, g_t = pg, b_t = pb;
        {
            const int64_t tn = t + 1 < n_tokens ? t + 1 : t;
#pragma unroll
            for (int i = 0; i < PER_T; i++) {
                pq[i] = q_base[tn * sq2 + tid + i * 64];
                pk[i] = k_base[tn * sq2 + tid + i * 64];
            }
            pv = v_base[tn * sv2];
            pg = g[gb + tn * sb2];
            pb = beta[gb + tn * sb2];
        }
        gdn_wave_sync();

        const float decay = expf(g_t);

        // kv = sum_kk (decay * S[kk][col]) * k[kk]
        float s_arr[PER_G];
        float pkv = 0.0f;
#pragma unroll
        for (int l = 0; l < PER_G; l++) {
            const int kk = l * NG + grp;
            const float sv = st_col[kk ^ swz] * decay;
            s_arr[l] = sv;
            pkv += sv * k_lds[kk];
        }
#pragma unroll
        for (int off = COLS; off < 64; off <<= 1) {
            pkv += __shfl_xor_sync(0xffffffffffffffffULL, pkv, off, 64);
        }
        const float delta = (v_t - pkv) * b_t;

        // S[kk][col] = decay * S[kk][col] + k[kk] * delta; out = sum_kk S[kk][col] * q[kk]
        float pout = 0.0f;
#pragma unroll
        for (int l = 0; l < PER_G; l++) {
            const int kk = l * NG + grp;
            const float sv = s_arr[l] + k_lds[kk] * delta;
            st_col[kk ^ swz] = sv;
            pout += sv * q_lds[kk];
        }
#pragma unroll
        for (int off = COLS; off < 64; off <<= 1) {
            pout += __shfl_xor_sync(0xffffffffffffffffULL, pout, off, 64);
        }
        if (grp == 0) {
            out[t * HD * H] = pout * scale;
        }

        if constexpr (keep_rs_t) {
            // slot 0 = most recent state, slot s = s tokens back (see the kernel above)
            const int target_slot = (int) (n_tokens - 1 - t);
            if (target_slot >= 0 && target_slot < K) {
                gdn_wave_sync();
                float * dst_state = s_out + target_slot * state_slot_stride;
                for (int i = tid; i < COLS * HD; i += 64) {
                    const int c = i / HD, kk = i % HD;
                    dst_state[(int64_t) (tile_base + c) * HD + kk] = st[c * HD + (kk ^ (2 * c))];
                }
            }
        }
    }

    gdn_wave_sync();
    if constexpr (!keep_rs_t) {
        for (int i = tid; i < COLS * HD; i += 64) {
            const int c = i / HD, kk = i % HD;
            s_out[(int64_t) (tile_base + c) * HD + kk] = st[c * HD + (kk ^ (2 * c))];
        }
    }
}
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// Decode variant (!KDA, S_v = 128, wave64): each wave updates CPW state columns instead of
// one, so a wave keeps CPW*2 independent state loads in flight and q/k are loaded once
// per wave. Per-column arithmetic and reduction order match gated_delta_net_cuda.
template <int CPW, bool keep_rs_t>
__global__ void __launch_bounds__(256, 2)
gated_delta_net_cpw(const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * curr_state, float * dst, float * state, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic, float scale,
        int64_t state_slot_stride, int K, const int32_t * state_rows, int64_t state_row_stride) {
    constexpr int S_v = 128;
    constexpr int warp_size = 64;
    constexpr int rows_per_lane = S_v / warp_size;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * blockDim.y + threadIdx.y) * CPW;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
    const int64_t state_in_offset  = (state_rows ? (int64_t) state_rows[sequence] * state_row_stride
                                                 : (int64_t) sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_out_offset;
    curr_state += state_in_offset;

    float s_shard[CPW][rows_per_lane];
#pragma unroll
    for (int c = 0; c < CPW; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[c][r] = curr_state[(col0 + c) * S_v + r * warp_size + lane];
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = beta[gb_offset];
        const float g_val    = expf(g[gb_offset]);

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            k_reg[r] = k_t[r * warp_size + lane];
            q_reg[r] = q_t[r * warp_size + lane];
        }

#pragma unroll
        for (int c = 0; c < CPW; c++) {
            const int col = col0 + c;
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[c][r] * k_reg[r];
            }
            const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_t[col] - g_val * kv_col) * beta_val;
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[c][r]  = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                attn_partial += s_shard[c][r] * q_reg[r];
            }
            const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }
        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * cs = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < CPW; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        cs[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < CPW; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                state[(col0 + c) * S_v + r * warp_size + lane] = s_shard[c][r];
            }
        }
    }
}
#endif // defined(GGML_USE_HIP)

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * state_rows, int64_t state_row_stride,
        cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

#if defined(GGML_USE_HIP)
    // short runs (MTP verify, 2..4 tokens) stay on the register-resident decode kernel below: the LDS
    // kernel costs ~2.3x a decode step at 3 tokens. GGML_CUDA_GDN_CPW_MAXT sets the longest such run
    static const int cpw_max_t = getenv("GGML_CUDA_GDN_CPW_MAXT") ? atoi(getenv("GGML_CUDA_GDN_CPW_MAXT")) : 4;
    if constexpr (!KDA) {
        if (S_v == GDN_LDS_HD && n_tokens > std::max(cpw_max_t, 1) && warp_size == 64) {
            constexpr int cols = 16; // 8: 1.3x slower (7.1 vs 5.6 ms at 2048 tokens, 48 heads)
            const dim3 grid(H, n_seqs, GDN_LDS_HD / cols);
            gated_delta_net_lds_wave64<cols, keep_rs_t><<<grid, 64, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows, state_row_stride);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
    if constexpr (!KDA) {
        // 4 columns per wave scales better with the sequence count (48 heads x 128, 6 seqs: 122 vs 173 us),
        // one sequence is slightly faster with 2; GGML_CUDA_GDN_CPW=2|4 forces one
        static const int cpw_env = getenv("GGML_CUDA_GDN_CPW") ? atoi(getenv("GGML_CUDA_GDN_CPW")) : 0;
        const int cpw = cpw_env ? cpw_env : (n_seqs >= 2 ? 4 : 2);
        if (S_v == 128 && warp_size == 64 && n_tokens <= std::max(cpw_max_t, 1) && (cpw == 2 || cpw == 4)) {
            const dim3 grid(H, n_seqs, S_v / (num_warps * cpw));
            const dim3 block(64, num_warps, 1);
            if (cpw == 4) {
                gated_delta_net_cpw<4, keep_rs_t><<<grid, block, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
                    state_slot_stride, K, state_rows, state_row_stride);
            } else {
                gated_delta_net_cpw<2, keep_rs_t><<<grid, block, 0, stream>>>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
                    state_slot_stride, K, state_rows, state_row_stride);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows, state_row_stride);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows, state_row_stride);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows, state_row_stride);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows, state_row_stride);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// The s0 input of a gated_delta_net is usually get_rows(ssm_states_all, s_copy): a copy of
// the whole recurrent state per layer and step. When that gather feeds nothing else, the
// graph loop skips it and the kernel reads the cache rows through the index instead.
static const ggml_tensor * gdn_state_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

bool ggml_cuda_gdn_state_gather_elidable(const ggml_cgraph * cgraph, const ggml_tensor * gr) {
    static const bool disabled = getenv("GGML_CUDA_NO_GDN_GATHER_FUSION") != nullptr;
    if (disabled || gr->op != GGML_OP_GET_ROWS || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            gr->type != GGML_TYPE_F32 || gr->src[0]->type != GGML_TYPE_F32 || gr->src[1]->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(gr) || !ggml_is_contiguous(gr->src[1]) || gr->src[0]->nb[0] != sizeof(float) ||
            gr->ne[2] != 1 || gr->ne[3] != 1 || gr->src[1]->ne[0] != gr->ne[1] || gr->src[0]->ne[0] != gr->ne[0]) {
        return false;
    }
    const ggml_tensor * user = nullptr;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n == gr || n->view_src != nullptr) {
            continue; // the gather itself and views of it
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (n->src[j] && gdn_state_root(n->src[j]) == gr) {
                if (user != nullptr || n->op != GGML_OP_GATED_DELTA_NET || j != 5) {
                    return false;
                }
                user = n;
            }
        }
    }
    if (user == nullptr || ggml_nelements(user->src[5]) != ggml_nelements(gr)) {
        return false;
    }
    // The GDN launch reads the gather's row indices, but the allocator keeps them alive only up to the
    // gather itself: a node scheduled between the two, or the GDN output, may reuse their memory (the
    // qwen35 graph computes gate/beta there; a single-token first ubatch then read garbage rows and
    // faulted). Keep the elision only when nothing from the gather to the GDN overlaps the indices.
    {
        const ggml_tensor * ids = gr->src[1];
        const char * i0 = (const char *) ids->data;
        const char * i1 = i0 + ggml_nbytes(ids);
        bool after_gr = false;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (n == gr) {
                after_gr = true;
                continue;
            }
            if (!after_gr) {
                continue;
            }
            if (n->view_src == nullptr && n->data != nullptr) {
                const char * a0 = (const char *) n->data;
                const char * a1 = a0 + ggml_nbytes(n);
                if (a0 < i1 && i0 < a1) {
                    return false;
                }
            }
            if (n == user) {
                break;
            }
        }
    }
    // The fused cache write-back stores sequence a's state into row kv_head + a in the same launch that
    // reads sequence b's state from row state_rows[b]. Fresh sequences all read the zero row rs_z, which
    // is one of their destination rows, so a prompt ubatch with several sequences read a state another
    // block was overwriting (multi-sequence batches were not reproducible). Decode ubatches read each
    // sequence's own row (or a rollback snapshot), so they keep the elision.
    const int64_t n_seq_tokens = user->src[2]->ne[2];
    const int64_t n_seqs       = user->src[2]->ne[3];
    return n_seqs == 1 || n_seq_tokens == 1;
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_cgraph * cgraph) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    const int32_t * state_rows       = nullptr;
    int64_t         state_row_stride = 0;
    {
        const ggml_tensor * gr = gdn_state_root(src_state);
        if (cgraph != nullptr && gr != src_state && ggml_cuda_gdn_state_gather_elidable(cgraph, gr)) {
            s_d              = (const float *) gr->src[0]->data;
            state_rows       = (const int32_t *) gr->src[1]->data;
            state_row_stride = gr->src[0]->nb[1] / sizeof(float);
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cgraph * cgraph) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr, cgraph);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache,
        const ggml_cgraph * cgraph) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache, cgraph);
}
