#include "repack-gcn.cuh"
#include "convert.cuh"
#include "quantize.cuh"

#include "ggml-backend-impl.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// layout helpers
// ---------------------------------------------------------------------

static __host__ __device__ inline int64_t repack_q4k_nsp(const int64_t ne0) {
    const int64_t n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

static inline size_t repack_q4k_nbytes(const int64_t ne0, const int64_t ne1) {
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const int64_t n_blocks = ne0 / 256;
    return (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2 + (size_t) ne1 * n_blocks * 4;
}

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    return t->type == GGML_TYPE_Q4_K && ggml_n_dims(t) == 2 && t->ne[0] % 256 == 0 &&
           ggml_is_contiguous(t);
}

// ---------------------------------------------------------------------
// host-side repack (one-shot at weight upload)
// ---------------------------------------------------------------------

// ggml-quants.c's get_scale_min_k4: unpack sub-block j's 6-bit (sc, m)
// from the 12-byte packed scales array.
static inline void repack_get_scale_min_k4(const int j, const uint8_t * q, uint8_t * sc, uint8_t * m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >>   4) | ((q[j    ] >> 6) << 4);
    }
}

static void repack_q4k_host(const block_q4_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    // The padding sub-block (when nsp != ne0/32) must read as zero
    // weights with zero scales so the kernel can include it harmlessly.
    memset(dst, 0, nib_len + sm_len + (size_t) ne1 * n_blocks * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q4_K * b = &blocks[row * n_blocks + blk];

            // superblock plane: raw fp16 d, dmin per 256 weights
            // (block_q4_K starts with d at byte 0, dmin at byte 2 —
            // guaranteed by the ggml-common.h size/layout asserts)
            uint8_t * dd = dst + nib_len + sm_len + (size_t)(row * n_blocks + blk) * 4;
            memcpy(dd, b, 4);

            for (int s = 0; s < 8; s++) {
                const int64_t gsb = blk * 8 + s; // sub-block index within the row

                // this sub-block's 32 nibble weights: qs bytes (s/2)*32..+32,
                // even sub-blocks take low nibbles, odd take high
                const uint8_t * qs = b->qs + (s >> 1) * 32;
                uint8_t w[32];
                if ((s & 1) == 0) {
                    for (int k = 0; k < 32; k++) { w[k] = qs[k] & 0x0F; }
                } else {
                    for (int k = 0; k < 32; k++) { w[k] = qs[k] >> 4; }
                }

                // nibble plane: byte 4j+b = w[4j+b] | (w[16+4j+b] << 4),
                // so uint32 j feeds dp4a with weights 4j..4j+3 / 16+4j..+3
                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                    }
                }

                // scale plane: 6-bit sc then m as two u8
                uint8_t sc, m;
                repack_get_scale_min_k4(s, b->scales, &sc, &m);
                uint8_t * sm = dst + nib_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = sc;
                sm[1] = m;
            }
        }
    }
}

// ---------------------------------------------------------------------
// kernels (GCN only — guarded so non-HIP / non-GCN builds still compile)
// ---------------------------------------------------------------------

// Repacked Q4_K matvec. Block = 256 threads = 4 wave64s; each wave
// computes ROWS=2 output rows; lane l streams sub-block l, l+64, ... —
// consecutive lanes read consecutive 16-byte chunks, a fully-coalesced
// sweep of the nibble plane.
//
//   dot = sum_sub [ (d*sc) * dx * <nibbles . q8> - (dmin*m) * sx ]
//
// with (dx, sx) = block_q8_1.ds — sx is dx * sum(q8), which is exactly
// the dequantized sub-block sum the min-term needs (same contract as
// vec_dot_q4_K_q8_1).
static __global__ void mul_mat_vec_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
    const uint32_t n_super = n_sub >> 3;

    float acc[ROWS] = {0.0f, 0.0f};

    for (uint32_t sb = lane; sb < n_sub; sb += 64) {
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }

            const uint4    q  = nib[(size_t) row * nsp + sb];
            const uint16_t sm = smp[(size_t) row * nsp + sb];
            const uint32_t dd = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                               * (float)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - deff * sx;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// int8 MMQ tile GEMM straight from the repacked planes (prefill path).
// Y[tok, row] = Xq8[tok, :] . W[row, :] without dequantizing W.
//
// A workgroup (256 threads as a 16x16 grid) computes a BM x BN output
// tile (BM = 64 weight rows, BN = 64 tokens), walking the contraction
// in BK = 4 sub-block chunks staged through LDS. Thread (tx,ty) owns a
// strided 4x4 register micro-tile (rows ty, ty+16, ..., tokens tx,
// tx+16, ...) so a wavefront's 16 token reads land on 16 distinct LDS
// banks (block_q8_1 stride is 36 B = 9 words; gcd(9,32)=1).
//
// Tile shape carried from the production kernel in reinstinct, where a
// sweep (BK in {4,8}, TM/TN in {4,8}, occupancy 1/2) found 4x4 at
// occupancy 2 flat-optimal on gfx906.
#define MMQ_RP_BK 4
#define MMQ_RP_TM 4
#define MMQ_RP_TN 4
#define MMQ_RP_BM (16 * MMQ_RP_TM)
#define MMQ_RP_BN (16 * MMQ_RP_TN)

static __global__ void __launch_bounds__(256, 2) mmq_gemm_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    const uint32_t tok0 = blockIdx.y * MMQ_RP_BN;

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW [MMQ_RP_BM][MMQ_RP_BK];     // packed nibbles
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK];     // (dsc, deff)
    __shared__ block_q8_1 sX [MMQ_RP_BN][MMQ_RP_BK + 1]; // int8 activations

    float acc[MMQ_RP_TM][MMQ_RP_TN] = {};

    constexpr int LDW = MMQ_RP_BM * MMQ_RP_BK / 256; // tile elems per thread
    constexpr int LDX = MMQ_RP_BN * MMQ_RP_BK / 256;

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
#pragma unroll
        for (int i = 0; i < LDW; i++) {
            const int e  = t + i * 256;
            const int lr = e / MMQ_RP_BK, lk = e % MMQ_RP_BK;
            const uint32_t wrow = row0 + lr;
            const uint32_t sb   = sb0 + lk;
            if (wrow < ne1 && sb < n_sub) {
                sW[lr][lk] = nib[(size_t) wrow * nsp + sb];
                const uint16_t sm = smp[(size_t) wrow * nsp + sb];
                const uint32_t dd = ddp[(size_t) wrow * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                sWs[lr][lk] = make_float2(
                    __half2float(*reinterpret_cast<const __half *>(&d_bits))
                        * (float)(sm & 0xFFu),
                    __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                        * (float)(sm >> 8));
            } else {
                sWs[lr][lk] = make_float2(0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int i = 0; i < LDX; i++) {
            const int e  = t + i * 256;
            const int lr = e / MMQ_RP_BK, lk = e % MMQ_RP_BK;
            const uint32_t xtok = tok0 + lr;
            const uint32_t sb   = sb0 + lk;
            if (xtok < n_tok && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) xtok * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wq[MMQ_RP_TM];
            float dsc[MMQ_RP_TM], deff[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq[r] = sW[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < MMQ_RP_TN; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot - deff[r] * sx;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < MMQ_RP_TN; n++) {
            const uint32_t tok = tok0 + tx + n * 16;
            if (tok < n_tok) {
                y[(size_t) tok * ne1 + row] = acc[r][n];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Repacked Q4_K -> fp16 row-major [ne1][ne0], for the GEMM path.
// grid = ne1 * (ne0/32) sub-blocks, block = 32 (one thread per weight).
static __global__ void dequant_q4k_repacked_f16(
        const uint8_t * __restrict__ wbase, half * __restrict__ out,
        const uint32_t ne0, const uint32_t ne1) {
    const uint32_t gidx = blockIdx.x;
    const uint32_t k    = threadIdx.x;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint32_t row = gidx / n_sub;
    const uint32_t sb  = gidx % n_sub;
    if (row >= ne1) {
        return;
    }

    const uint32_t n_super = n_sub >> 3;
    const uint8_t  * nib = wbase + (size_t)(row * nsp + sb) * 16;
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t)(row * nsp + sb) * 2);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2
              + (size_t)(row * n_super + (sb >> 3)) * 4);

    const uint16_t sm = *smp;
    const uint32_t dd = *ddp;
    const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
    const uint16_t dmin_bits = (uint16_t)(dd >> 16);
    const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))
                       * (float)(sm & 0xFFu);
    const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits))
                       * (float)(sm >> 8);

    const uint8_t nibble = (k < 16) ? (nib[k] & 0x0F) : (nib[k - 16] >> 4);
    out[(size_t) gidx * 32 + k] = __float2half(dsc * (float) nibble - deff);
}

// ---------------------------------------------------------------------
// MUL_MAT dispatch
// ---------------------------------------------------------------------

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // M
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1]; // N
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();
    const uint8_t * w  = (const uint8_t *) src0->data;
    float * dst_d      = (float *) dst->data;

    if (ne11 == 1) {
        // decode: quantize the activation row to q8_1, run the dp4a matvec
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
            ne10_padded * sizeof(block_q8_1) / QK8_1);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, ne10, ne10, ne10, ne10_padded, 1, 1, 1, stream);

        const dim3 grid((ne01 + 7) / 8, 1, 1);
        mul_mat_vec_q4k_repacked<<<grid, 256, 0, stream>>>(
            w, (const block_q8_1 *) src1_q8_1.get(), dst_d,
            (uint32_t) ne00, (uint32_t) ne01);
        return;
    }

    // prefill: int8 MMQ tile GEMM straight from the repacked planes.
    static const bool no_mmq = [] {
        const char * e = getenv("GGML_CUDA_REPACK_NO_MMQ");
        return e != nullptr && e[0] != '0';
    }();
    if (!no_mmq) {
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        const int64_t x_stride    = ne10_padded / QK8_1; // q8_1 blocks per token
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
            ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1);
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
                src0->type, ne10, s11, s11 * ne11, s11 * ne11, ne10_padded, ne11, 1, 1, stream);
        }

        const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM,
                        (ne11 + MMQ_RP_BN - 1) / MMQ_RP_BN, 1);
        mmq_gemm_q4k_repacked<<<grid, 256, 0, stream>>>(
            w, (const block_q8_1 *) src1_q8_1.get(), dst_d,
            (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride);
        return;
    }

    // debug fallback (GGML_CUDA_REPACK_NO_MMQ=1): dequantize the
    // repacked weight to fp16 and GEMM
    ggml_cuda_pool_alloc<half> w_f16(ctx.pool(), (size_t) ne00 * ne01);
    {
        const int64_t n_sub = ne00 / 32;
        const dim3 grid((unsigned) (ne01 * n_sub), 1, 1);
        dequant_q4k_repacked_f16<<<grid, 32, 0, stream>>>(
            w, w_f16.get(), (uint32_t) ne00, (uint32_t) ne01);
    }

    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(), (size_t) ne10 * ne11);
    {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda((const float *) src1->data, src1_f16.get(), ne10 * ne11, stream);
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    CUBLAS_CHECK(
        cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                &alpha, w_f16.get(),    CUDA_R_16F, ne00,
                        src1_f16.get(), CUDA_R_16F, ne10,
                &beta,  dst_d,          CUDA_R_32F, ne01,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------

struct ggml_backend_cuda_repack_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name;
}

static void ggml_backend_cuda_repack_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));
    GGML_ASSERT(ggml_cuda_repack_tensor_supported(tensor));

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];

    std::vector<uint8_t> staged(repack_q4k_nbytes(ne0, ne1));
    repack_q4k_host((const block_q4_K *) data, staged.data(), ne0, ne1);

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, staged.data(), staged.size(),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_repack_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    GGML_ABORT("repacked tensors cannot be read back (GGML_CUDA_REPACK)");
    GGML_UNUSED_VARS(buffer, tensor, data, offset, size);
}

static ggml_backend_buffer_t ggml_backend_cuda_repack_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;

    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.set_tensor  = ggml_backend_cuda_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = ggml_backend_cuda_repack_buffer_get_tensor;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (ggml_cuda_repack_tensor_supported(tensor)) {
        return repack_q4k_nbytes(tensor->ne[0], tensor->ne[1]);
    }
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_repack_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_cuda_repack_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_cuda_repack_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_cuda_repack_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ ggml_backend_cuda_repack_buffer_type_get_alloc_size,
    /* .is_host        = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    const char * env = getenv("GGML_CUDA_REPACK");
    if (env == nullptr || env[0] == '0') {
        return nullptr;
    }
    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    if (!GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[device].cc)) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft_storage[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    if (!initialized[device]) {
        buft_storage[device] = {
            /* .iface   = */ ggml_backend_cuda_repack_buffer_type_interface,
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            /* .context = */ new ggml_backend_cuda_repack_buffer_type_context{
                                 device, GGML_CUDA_NAME + std::to_string(device) + "_Repacked"},
        };
        initialized[device] = true;
    }
    return &buft_storage[device];
}
