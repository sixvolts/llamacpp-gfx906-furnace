#include "repack-gcn.cuh"
#include "convert.cuh"
#include "quantize.cuh"

#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#include "ggml-quants.h"
#include <vector>
#include <cmath>

#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------
// layout helpers
// ---------------------------------------------------------------------

// Sub-blocks (32 weights) per repacked row, padded by one when the
// natural count is a power of two (a power-of-two row stride aliases
// every row onto the same HBM channel: ~3x matvec penalty). Shared by
// all repacked types.
static __host__ __device__ inline int64_t repack_q4k_nsp(const int64_t ne0) {
    const int64_t n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

// Plane bytes per type:
//   Q3_K: 8 lo2 + 4 hi1 + 2 signed-scale-pair per sub-block, 2 (d) per superblock
//   Q4_K: 16 nib + 2 sc|m per sub-block, 4 (d|dmin fp16) per superblock
//   Q5_K: 16 nib + 4 qh + 2 sc|m per sub-block, 4 per superblock
//   Q6_K: 16 nib + 8 h2 + 2 signed-scale-pair per sub-block, 2 (d) per superblock
//   Q8_0: 32 qs + 2 (d fp16) per sub-block
//   Q5_1: 16 nib + 4 qh + 4 (d, m fp16) per sub-block
//   Q4_0: 16 nib + 2 (d fp16) per sub-block (crossport R11; the IQ4 family shares this layout)
// Q3_K is stored and computed as Q6_K planes (crossport R12, exact: q6 = q3 + 28 with the same
// signed per-16 scales), so it runs the Q6_K kernels; its own 3-bit kernels stay behind
// GGML_CUDA_Q3K_RELABEL=0. Every dispatch switch and the size table go through this.
static inline ggml_type repack_eff_type(const ggml_type type) {
    static const bool relabel = [] {
        const char * e = getenv("GGML_CUDA_Q3K_RELABEL");
        return e == nullptr || e[0] != '0';
    }();
    return (type == GGML_TYPE_Q3_K && relabel) ? GGML_TYPE_Q6_K : type;
}

static inline size_t repack_gcn_nbytes(const ggml_type type_in, const int64_t ne0, const int64_t ne1) {
    const ggml_type type    = repack_eff_type(type_in);
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const int64_t n_blocks = ne0 / 256;
    switch (type) {
        case GGML_TYPE_Q3_K: return (size_t) ne1 * (nsp * 14 + n_blocks * 2);
        case GGML_TYPE_Q4_K: return (size_t) ne1 * (nsp * 18 + n_blocks * 4);
        case GGML_TYPE_Q5_K: return (size_t) ne1 * (nsp * 22 + n_blocks * 4);
        case GGML_TYPE_Q6_K: return (size_t) ne1 * (nsp * 26 + n_blocks * 2);
        case GGML_TYPE_Q8_0: return (size_t) ne1 * nsp * 34;
        case GGML_TYPE_Q5_1: return (size_t) ne1 * nsp * 24;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:  return (size_t) ne1 * nsp * 18;
        default:             GGML_ABORT("unsupported repack type");
    }
}

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // 2D weights (MUL_MAT) or 3D per-expert stacks (MUL_MAT_ID)
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q3_K: {
            static const bool enabled = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q3_K");
                return e == nullptr || e[0] != '0';
            }();
            return enabled && t->ne[0] % 256 == 0;
        }
        case GGML_TYPE_Q4_K: {
            static const bool enabled = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q4_K");
                return e == nullptr || e[0] != '0';
            }();
            return enabled && t->ne[0] % 256 == 0;
        }
        case GGML_TYPE_Q5_K: {
            static const bool enabled = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q5_K");
                return e == nullptr || e[0] != '0';
            }();
            return enabled && t->ne[0] % 256 == 0;
        }
        case GGML_TYPE_Q6_K: {
            static const bool enabled = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q6_K");
                return e == nullptr || e[0] != '0';
            }();
            return enabled && t->ne[0] % 256 == 0;
        }
        case GGML_TYPE_Q8_0: {
            // Q8_0 repack is its own opt-in: the repacked MMQ wins
            // prefill big (+43% on a pure-Q8_0 0.8B) but the repacked
            // matvec loses ~6% decode to the canonical mmvq (it was
            // tuned on MoE-expert shapes in reinstinct; on-disk Q8_0 is
            // already nearly contiguous so repack buys less). Re-tune
            // before considering it for default-on.
            static const bool q8 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q8_0");
                return e != nullptr && e[0] != '0';
            }();
            return q8 && t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_Q5_1: {
            // opt-in while being validated; K only needs to be a multiple of 32 (qwen4exp ffn_down_exps has K=640)
            static const bool q51 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q5_1");
                return e != nullptr && e[0] != '0';
            }();
            return q51 && t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_Q4_0: {
            // 2D weights only (no expert kernels yet); GGML_CUDA_REPACK_Q4_0=0 opts out
            static const bool q40 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q4_0");
                return e == nullptr || e[0] != '0';
            }();
            return q40 && ggml_n_dims(t) == 2 && t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S: {
            // the IQ family relabels onto the Q4_0 planes (IQ4_NL exactly; IQ4_XS / IQ3_S with the
            // sub-block scale folded to fp16, and IQ3_S at 4.5 instead of 3.44 bpw on device);
            // 2D weights only. GGML_CUDA_REPACK_IQ=0 opts out.
            static const bool iq = [] {
                const char * e = getenv("GGML_CUDA_REPACK_IQ");
                return e == nullptr || e[0] != '0';
            }();
            return iq && ggml_n_dims(t) == 2 && t->ne[0] % (t->type == GGML_TYPE_IQ4_NL ? 32 : 256) == 0;
        }
        default:             return false;
    }
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

// Q5_K: like Q4_K plus a qh plane — per sub-block one u32 whose bit
// 4g+b is the 5th bit of weight b of dp4a group g.
static void repack_q5k_host(const block_q5_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  qh_len   = (size_t) ne1 * nsp * 4;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, nib_len + qh_len + sm_len + (size_t) ne1 * n_blocks * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q5_K * b = &blocks[row * n_blocks + blk];

            uint8_t * dd = dst + nib_len + qh_len + sm_len + (size_t)(row * n_blocks + blk) * 4;
            memcpy(dd, b, 4); // fp16 d, dmin lead the block

            for (int s = 0; s < 8; s++) {
                const int64_t gsb = blk * 8 + s;
                const uint8_t * qs = b->qs + (s >> 1) * 32;

                uint8_t w[32], hb[32];
                for (int k = 0; k < 32; k++) {
                    w[k]  = ((s & 1) == 0) ? (qs[k] & 0x0F) : (qs[k] >> 4);
                    hb[k] = (b->qh[k] >> s) & 1;
                }

                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint32_t qh_packed = 0;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = w[4 * j + bb] | (w[16 + 4 * j + bb] << 4);
                        // dp4a group 2j holds weights 4j+bb, group 2j+1 holds 16+4j+bb
                        qh_packed |= (uint32_t) hb[4 * j + bb]      << (4 * (2 * j)     + bb);
                        qh_packed |= (uint32_t) hb[16 + 4 * j + bb] << (4 * (2 * j + 1) + bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 4, &qh_packed, 4);

                uint8_t sc, m;
                repack_get_scale_min_k4(s, b->scales, &sc, &m);
                uint8_t * sm = dst + nib_len + qh_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = sc;
                sm[1] = m;
            }
        }
    }
}

// Q6_K: nibble plane + 8-byte h2 plane (the 6-bit quant's high pair per
// weight, 2 bits at position 2b of byte g) + signed per-16-weight scale
// pairs + d-only superblock plane.
static void repack_q6k_host(const block_q6_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  h2_len   = (size_t) ne1 * nsp * 8;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, nib_len + h2_len + sm_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q6_K * b = &blocks[row * n_blocks + blk];

            memcpy(dst + nib_len + h2_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);

            for (int s = 0; s < 8; s++) {
                const int64_t gsb  = blk * 8 + s;
                const int     chunk = s / 4;
                const int     quad  = s % 4;
                const int     ql_off = chunk * 64;
                const int     qh_off = chunk * 32;

                uint8_t lo[32], hi[32];
                for (int k = 0; k < 32; k++) {
                    const uint8_t qh = b->qh[qh_off + k];
                    switch (quad) {
                        case 0:  lo[k] = b->ql[ql_off + k]      & 0x0F; hi[k] =  qh       & 3; break;
                        case 1:  lo[k] = b->ql[ql_off + k + 32] & 0x0F; hi[k] = (qh >> 2) & 3; break;
                        case 2:  lo[k] = b->ql[ql_off + k]      >> 4;   hi[k] = (qh >> 4) & 3; break;
                        default: lo[k] = b->ql[ql_off + k + 32] >> 4;   hi[k] = (qh >> 6) & 3; break;
                    }
                }

                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint8_t h2p[8] = {};
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = lo[4 * j + bb] | (lo[16 + 4 * j + bb] << 4);
                        h2p[2 * j]     |= hi[4 * j + bb]      << (2 * bb);
                        h2p[2 * j + 1] |= hi[16 + 4 * j + bb] << (2 * bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 8, h2p, 8);

                uint8_t * sm = dst + nib_len + h2_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) b->scales[chunk * 8 + quad * 2];
                sm[1] = (uint8_t) b->scales[chunk * 8 + quad * 2 + 1];
            }
        }
    }
}

// Q3_K: the 3-bit quant stays sub-nibble to fit VRAM. A lo2 plane (low 2
// bits, packed like Q6_K's h2) and a hi1 plane (high bit, packed like
// Q5_K's qh) reconstruct q3 = lo2 | (hbit << 2) at compute. Symmetric
// like Q6_K (bias 4) with a signed per-16-weight scale pair (unpacked
// 6-bit scale minus 32) and a d-only superblock plane.
static void repack_q3k_host(const block_q3_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  lo2_len  = (size_t) ne1 * nsp * 8;
    const size_t  hi1_len  = (size_t) ne1 * nsp * 4;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;

    memset(dst, 0, lo2_len + hi1_len + sm_len + (size_t) ne1 * n_blocks * 2);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q3_K * b = &blocks[row * n_blocks + blk];

            memcpy(dst + lo2_len + hi1_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);

            // ggml-quants.c dequantize_row_q3_K: unpack the 16 6-bit scales
            uint32_t aux[4];
            memcpy(aux, b->scales, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const uint8_t * sc6 = (const uint8_t *) aux;

            for (int s = 0; s < 8; s++) {
                const int64_t gsb   = blk * 8 + s;
                const int     n     = s >> 2;
                const int     shift = 2 * (s & 3);
                const uint8_t * qs  = b->qs + n * 32;

                uint8_t  lo2[32], hb[32];
                for (int k = 0; k < 32; k++) {
                    lo2[k] = (qs[k] >> shift) & 3;
                    hb[k]  = (b->hmask[k] >> s) & 1;
                }

                // lo2 plane: byte 2j holds the low-half group j (weights
                // 4j..), byte 2j+1 the high-half (weights 16+4j..), weight
                // b at bits 2b..2b+1 -- the Q6_K h2 packing.
                uint8_t lo2p[8] = {};
                uint32_t hi1p = 0;
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        lo2p[2 * j]     |= lo2[4 * j + bb]      << (2 * bb);
                        lo2p[2 * j + 1] |= lo2[16 + 4 * j + bb] << (2 * bb);
                        hi1p |= (uint32_t) hb[4 * j + bb]      << (8 * j + bb);
                        hi1p |= (uint32_t) hb[16 + 4 * j + bb] << (8 * j + 4 + bb);
                    }
                }
                memcpy(dst + (size_t)(row * nsp + gsb) * 8, lo2p, 8);
                memcpy(dst + lo2_len + (size_t)(row * nsp + gsb) * 4, &hi1p, 4);

                uint8_t * sm = dst + lo2_len + hi1_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) (int8_t) ((int) sc6[2 * s]     - 32);
                sm[1] = (uint8_t) (int8_t) ((int) sc6[2 * s + 1] - 32);
            }
        }
    }
}

// Q3_K relabelled to Q6_K planes (exact): w = d*(sc-32)*(q3-4) = d*(sc-32)*((q3+28)-32), so q6 = q3+28
// and the signed per-16-weight scales carry over unchanged. Same plane layout as repack_q6k_host.
static void repack_q3k_as_q6k_host(const block_q3_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  h2_len   = (size_t) ne1 * nsp * 8;
    const size_t  sm_len   = (size_t) ne1 * nsp * 2;
    memset(dst, 0, nib_len + h2_len + sm_len + (size_t) ne1 * n_blocks * 2);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q3_K * b = &blocks[row * n_blocks + blk];
            memcpy(dst + nib_len + h2_len + sm_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
            uint32_t aux[4];
            memcpy(aux, b->scales, 12);
            const uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = ( aux[0]       & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = ( aux[1]       & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const uint8_t * sc6 = (const uint8_t *) aux;
            for (int s = 0; s < 8; s++) {
                const int64_t gsb   = blk * 8 + s;
                const int     n     = s >> 2;
                const int     shift = 2 * (s & 3);
                const uint8_t * qs  = b->qs + n * 32;
                uint8_t lo[32], hi[32];
                for (int k = 0; k < 32; k++) {
                    const int q6 = (((qs[k] >> shift) & 3) | (((b->hmask[k] >> s) & 1) << 2)) + 28;
                    lo[k] = (uint8_t) (q6 & 0x0F);
                    hi[k] = (uint8_t) (q6 >> 4);
                }
                uint8_t * nib = dst + (size_t)(row * nsp + gsb) * 16;
                uint8_t h2p[8] = {};
                for (int j = 0; j < 4; j++) {
                    for (int bb = 0; bb < 4; bb++) {
                        nib[j * 4 + bb] = lo[4 * j + bb] | (lo[16 + 4 * j + bb] << 4);
                        h2p[2 * j]     |= hi[4 * j + bb]      << (2 * bb);
                        h2p[2 * j + 1] |= hi[16 + 4 * j + bb] << (2 * bb);
                    }
                }
                memcpy(dst + nib_len + (size_t)(row * nsp + gsb) * 8, h2p, 8);
                uint8_t * sm = dst + nib_len + h2_len + (size_t)(row * nsp + gsb) * 2;
                sm[0] = (uint8_t) (int8_t) ((int) sc6[2 * s]     - 32);
                sm[1] = (uint8_t) (int8_t) ((int) sc6[2 * s + 1] - 32);
            }
        }
    }
}

// ggml_half is a device half type in this TU: read the superblock scale by its bits
static inline float repack_half_bits_to_f32(const void * p) {
    uint16_t b;
    memcpy(&b, p, 2);
    return ggml_fp16_to_fp32(b);
}

// IQ4_NL onto the Q4_0 planes: same 32-weight blocks (fp16 d, 16 nibble bytes), exact.
static void repack_iq4_nl_host(const block_iq4_nl * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    memset(dst, 0, nib_len + (size_t) ne1 * nsp * 2);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_iq4_nl * b = &blocks[row * n_blocks + blk];
            const size_t idx = (size_t)(row * nsp + blk);
            memcpy(dst + idx * 16, b->qs, 16);
            memcpy(dst + nib_len + idx * 2, &b->d, 2);
        }
    }
}

// IQ4_XS onto the Q4_0 planes: the 8 sub-blocks' nibbles copied, d*(ls-32) folded to one fp16 per
// sub-block (<= 2^-11 relative rounding; not bit-exact against ggml's dequant).
static void repack_iq4_xs_host(const block_iq4_xs * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    memset(dst, 0, nib_len + (size_t) ne1 * nsp * 2);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_iq4_xs * b = &blocks[row * n_blocks + blk];
            const float d = repack_half_bits_to_f32(&b->d);
            for (int ib = 0; ib < 8; ib++) {
                const size_t idx = (size_t)(row * nsp + blk * 8 + ib);
                memcpy(dst + idx * 16, b->qs + 16 * ib, 16);
                const int ls = ((b->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((b->scales_h >> 2 * ib) & 3) << 4);
                const ggml_fp16_t dl = ggml_fp32_to_fp16(d * (float) (ls - 32));
                memcpy(dst + nib_len + idx * 2, &dl, 2);
            }
        }
    }
}

// IQ3_S onto the nibble planes: ggml's own dequantizer recovers each weight as db * v with v an odd
// integer in -15..15, stored as the nibble (v + 15) / 2 (the device codebook is 2n - 15), and
// db = d * (1 + 2*s) folded to fp16 per sub-block. 4.5 bpw on device for 3.44 on disk.
static void repack_iq3_s_host(const block_iq3_s * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 256;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    memset(dst, 0, nib_len + (size_t) ne1 * nsp * 2);
    std::vector<float> yf(256);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_iq3_s * b = &blocks[row * n_blocks + blk];
            dequantize_row_iq3_s(b, yf.data(), 256);
            const float d = repack_half_bits_to_f32(&b->d);
            for (int ib = 0; ib < 8; ib++) {
                const size_t idx = (size_t)(row * nsp + blk * 8 + ib);
                const float db = d * (float) (1 + 2 * ((b->scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
                uint8_t * nib = dst + idx * 16;
                for (int k = 0; k < 16; k++) {
                    const int v0 = (int) lroundf(yf[32 * ib + k]      / db); // odd, -15..15
                    const int v1 = (int) lroundf(yf[32 * ib + k + 16] / db);
                    nib[k] = (uint8_t) (((v0 + 15) >> 1) | (((v1 + 15) >> 1) << 4));
                }
                const ggml_fp16_t dl = ggml_fp32_to_fp16(db);
                memcpy(dst + nib_len + idx * 2, &dl, 2);
            }
        }
    }
}

// Q8_0: two planes — 32 aligned qs bytes per sub-block, then the fp16
// d-scales as their own stream. Same bytes as on-disk modulo padding;
// the win is alignment (one i32 load per sdot4 instead of two
// uint16 loads OR-shifted around the on-disk 2-byte offset).
static void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  qs_len   = (size_t) ne1 * nsp * 32;

    memset(dst, 0, qs_len + (size_t) ne1 * nsp * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t)(row * nsp + blk) * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * nsp + blk) * 2, &b->d, 2);
        }
    }
}

// Q5_1: nibble plane (qs as-is), qh plane in the Q5_K dp4a-group bit order, (d, m) plane.
static void repack_q5_1_host(const block_q5_1 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    const size_t  qh_len   = (size_t) ne1 * nsp * 4;

    memset(dst, 0, nib_len + qh_len + (size_t) ne1 * nsp * 4);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q5_1 * b = &blocks[row * n_blocks + blk];
            const size_t idx = (size_t)(row * nsp + blk);

            memcpy(dst + idx * 16, b->qs, 16);

            uint32_t qh_raw;
            memcpy(&qh_raw, b->qh, 4);
            uint32_t qh_packed = 0;
            for (int j = 0; j < 4; j++) {
                for (int bb = 0; bb < 4; bb++) {
                    // dp4a group 2j holds weights 4j+bb, group 2j+1 holds 16+4j+bb
                    qh_packed |= ((qh_raw >> (4 * j + bb))      & 1u) << (4 * (2 * j)     + bb);
                    qh_packed |= ((qh_raw >> (16 + 4 * j + bb)) & 1u) << (4 * (2 * j + 1) + bb);
                }
            }
            memcpy(dst + nib_len + idx * 4, &qh_packed, 4);
            memcpy(dst + nib_len + qh_len + idx * 4, &b->dm, 4);
        }
    }
}

// Q4_0: nibble plane (qs as-is: byte k = weight k low / k+16 high, the dp4a group order) and an
// fp16 d plane. The pad sub-block reads as zero weights with a zero scale.
static void repack_q4_0_host(const block_q4_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  nib_len  = (size_t) ne1 * nsp * 16;
    memset(dst, 0, nib_len + (size_t) ne1 * nsp * 2);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q4_0 * b = &blocks[row * n_blocks + blk];
            const size_t idx = (size_t)(row * nsp + blk);
            memcpy(dst + idx * 16, b->qs, 16);
            memcpy(dst + nib_len + idx * 2, &b->d, 2);
        }
    }
}

// ---------------------------------------------------------------------
// kernels (GCN only — guarded so non-HIP / non-GCN builds still compile)
// ---------------------------------------------------------------------

// --- MUL_MAT_ID support -----------------------------------------------
// Expert routing comes compacted from ggml_cuda_launch_mm_ids_helper:
// assignment index a in [0, n_assign) is expert-sorted; expert_bounds
// gives each expert's [start, end) range; ids_src1[a] is the flat
// column index into the naturally-ordered activation buffer; ids_dst[a]
// is the flat destination column. Weights for expert e live at
// wbase + e * expert_stride (per-expert repacked slabs, identical
// layout to the 2D case).

// tile_off[e] = exclusive prefix sum of per-expert token-tile counts
// (BN-sized tiles), tile_off[n_expert] = total; tile_expert[tile] = the
// expert owning that tile. One 1024-thread block, Hillis-Steele scan
// over chunks of 1024 experts with a running carry.
template <int BN>
static __global__ void __launch_bounds__(1024, 1) repack_tile_map(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        int32_t * __restrict__ tile_expert, const int n_expert) {
    __shared__ int s[1024];
    const int t = threadIdx.x;
    int carry = 0;
    for (int e0 = 0; e0 < n_expert; e0 += 1024) {
        const int e   = e0 + t;
        const int cnt = e < n_expert ? (expert_bounds[e + 1] - expert_bounds[e] + BN - 1) / BN : 0;
        s[t] = cnt;
        __syncthreads();
        for (int off = 1; off < 1024; off <<= 1) {
            const int v = t >= off ? s[t - off] : 0;
            __syncthreads();
            s[t] += v;
            __syncthreads();
        }
        const int excl = carry + s[t] - cnt;
        if (e < n_expert) {
            tile_off[e] = excl;
            for (int k = 0; k < cnt; k++) {
                tile_expert[excl + k] = e;
            }
        }
        carry += s[1023];
        __syncthreads();
    }
    if (t == 0) {
        tile_off[n_expert] = carry;
    }
}

// GGML_CUDA_NO_KQ_HOIST=1: dense Q4_K/Q5_K/Q6_K matvecs and the dense Q4_K GLU run the pre-R10a
// guarded per-row loop (A/B switch for the clamp + hoisted-load change).
static bool kq_hoist() {
    static const bool h = getenv("GGML_CUDA_NO_KQ_HOIST") == nullptr;
    return h;
}

// GGML_CUDA_NO_Q4K_FENCE=1: drop the scheduling barrier after the hoisted loads in the dense Q4_K
// matvec / GLU (crossport P1b: without it the scheduler splits each trip into load / wait / dot;
// 78 -> 70 us on 17408x5120). Only Q4_K: it did nothing on Q8_0 and hurt Q6_K / Q4_0.
static bool q4k_fence() {
    static const bool f = getenv("GGML_CUDA_NO_Q4K_FENCE") == nullptr;
    return f;
}

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
template <bool HAS_IDS, bool FENCE = false, bool HOIST = true>
static __global__ void __launch_bounds__(256) mul_mat_vec_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
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

    // ID path (experts) uses 16-weight half-sub-block units: expert
    // tensors are small-K (down-proj K=768 -> 24 sub-blocks for 64
    // lanes) and full units leave most of the wave idle. The -deff*sx
    // min term is applied by the even half only.
    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub;
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        if constexpr (HAS_IDS || !HOIST) {
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
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
        } else {
        // rows clamped (not branched) so both rows' planes are in flight before the first dot;
        // the store below is masked (crossport R10)
        uint4    q[ROWS];
        uint16_t sm[ROWS];
        uint32_t dd[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint32_t row = (uint32_t) min(row0 + r, (int) ne1 - 1);
            q[r]  = nib[(size_t) row * nsp + sb];
            sm[r] = smp[(size_t) row * nsp + sb];
            dd[r] = ddp[(size_t) row * n_super + (sb >> 3)];
        }
        if constexpr (FENCE) {
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint16_t d_bits    = (uint16_t)(dd[r] & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd[r] >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm[r] & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm[r] >> 8);
            const uint32_t qa[4] = { q[r].x, q[r].y, q[r].z, q[r].w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
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
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3 — positions the Q5_K
// high bit above the dp4a nibble lanes.
static __device__ __forceinline__ uint32_t repack_spread4(const uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

// Spread four 2-bit fields (weight b at bits 2b..2b+1) to bits 4..5 of
// bytes 0..3 — the Q6_K quant's high pair.
static __device__ __forceinline__ uint32_t repack_spread2(const uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

// Bytewise v - 32 on four 6-bit values with no borrow between bytes: setting bit 7
// first keeps every byte >= 128 > 32; clearing it after leaves (v - 32) mod 256, the
// two's-complement int8 that dp4a expects (from reinstinct's Q6_K tile).
static __device__ __forceinline__ uint32_t repack_sub32(const uint32_t v) {
    return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}

// Q3_K reconstruction: four 2-bit fields to bits 0..1 of bytes 0..3 (the
// low pair) and four 1-bit fields to bit 2 of bytes 0..3 (the high bit).
static __device__ __forceinline__ uint32_t repack_spread2_lo(const uint32_t h) {
    return (h & 0x03u) | ((h & 0x0Cu) << 6)
         | ((h & 0x30u) << 12) | ((h & 0xC0u) << 18);
}
static __device__ __forceinline__ uint32_t repack_spread1_hi(const uint32_t h) {
    return ((h & 1u) << 2) | ((h & 2u) << 9) | ((h & 4u) << 16) | ((h & 8u) << 23);
}

// Q5_K repacked matvec — Q4_K's shape plus the qh plane OR-ed onto the
// nibbles before each dp4a.
template <bool HAS_IDS, bool HOIST = true>
static __global__ void __launch_bounds__(256) mul_mat_vec_q5k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        if constexpr (HAS_IDS || !HOIST) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const uint16_t sm  = smp[idx];
            const uint32_t dd  = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
        } else {
        uint4    q[ROWS];
        uint32_t qh[ROWS];
        uint16_t sm[ROWS];
        uint32_t dd[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) { // rows clamped, store masked (crossport R10)
            const uint32_t row = (uint32_t) min(row0 + r, (int) ne1 - 1);
            const size_t   idx = (size_t) row * nsp + sb;
            q[r]  = nib[idx];
            qh[r] = qhp[idx];
            sm[r] = smp[idx];
            dd[r] = ddp[(size_t) row * n_super + (sb >> 3)];
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint16_t d_bits    = (uint16_t)(dd[r] & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd[r] >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm[r] & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm[r] >> 8);
            const uint32_t qa[4] = { q[r].x, q[r].y, q[r].z, q[r].w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh[r] >> (8 * j))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh[r] >> (8 * j + 4)) & 0xFu);
                idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
        }
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
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// --- nibble + fp16 scale plane formats (Q4_0 now; IQ4_NL / IQ4_XS / IQ3_S relabel onto the same
// planes with a 16-entry codebook instead of q - 8, crossport R11/R12) ------------------------------
enum { REPACK_NIB_Q4_0 = 0, REPACK_NIB_IQ4NL = 1, REPACK_NIB_IQ3S = 2 };

// bytewise v - 8 on four nibbles without inter-byte borrow: the signed int8 Q4_0 weight dp4a wants
static __device__ __forceinline__ uint32_t repack_sub8(const uint32_t v) {
    return ((v | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

// IQ3_S values are the odd numbers -15..15: nibble n <-> 2n - 15
static const __device__ int8_t repack_kvalues_iq3s[16] = { -15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15 };

// one nibble word (weights 4j..4j+3 low, 16+4j..16+4j+3 high) to two signed int8 dp4a words
template <int MODE>
static __device__ __forceinline__ void repack_nib_decode(const uint32_t q, int & lo, int & hi) {
    if constexpr (MODE == REPACK_NIB_Q4_0) {
        lo = (int) repack_sub8( q       & 0x0F0F0F0Fu);
        hi = (int) repack_sub8((q >> 4) & 0x0F0F0F0Fu);
    } else {
        const int2 v = get_int_from_table_16((int) q, MODE == REPACK_NIB_IQ4NL ? kvalues_iq4nl : repack_kvalues_iq3s);
        lo = v.x;
        hi = v.y;
    }
}

// Decode matvec on the nibble + scale planes, ROWS=2 per wave (reinstinct's matvec_q4_0_repacked).
// Q4_0 folds the constant -8 into the integer domain against the quantized activation sum
// (xqsum = sum dp4a(1, xq)), one hoisted per sub-block; dw*dx*(idot - 8*xqsum) is then the exact
// integer dot the sub8 form computes, so this, the nc kernel and the MMQ tile agree exactly. The
// codebook modes decode the nibbles (two perm lookups per word) and dot directly.
template <int MODE>
static __global__ void __launch_bounds__(256) mul_mat_vec_nib_repacked(
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
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);

    float acc[ROWS] = {0.0f, 0.0f};
    for (uint32_t sb = lane; sb < n_sub; sb += 64) {
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);
        int xqsum = 0;
        if constexpr (MODE == REPACK_NIB_Q4_0) {
#pragma unroll
            for (int g = 0; g < 8; g++) {
                xqsum = ggml_cuda_dp4a(0x01010101, xq32[g], xqsum);
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint16_t db  = dp[idx];
            const float    dw  = __half2float(*reinterpret_cast<const __half *>(&db));
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if constexpr (MODE == REPACK_NIB_Q4_0) {
                    idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                    idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                } else {
                    int lo, hi;
                    repack_nib_decode<MODE>(qa[j], lo, hi);
                    idot = ggml_cuda_dp4a(lo, xq32[j],     idot);
                    idot = ggml_cuda_dp4a(hi, xq32[j + 4], idot);
                }
            }
            acc[r] += dw * dx * (float) (idot - 8 * xqsum);
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

// Dense gate+up GLU on the nibble planes (both slabs per sub-block, one xqsum), ROWS=2 per wave.
template <int MODE>
static __global__ void __launch_bounds__(256) mul_mat_vec_nib_repacked_glu(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op) {
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint8_t * wb[2] = { wup, wgate };

    float acc[2][ROWS] = {};
    for (uint32_t sb = lane; sb < n_sub; sb += 64) {
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);
        int xqsum = 0;
        if constexpr (MODE == REPACK_NIB_Q4_0) {
#pragma unroll
            for (int g = 0; g < 8; g++) {
                xqsum = ggml_cuda_dp4a(0x01010101, xq32[g], xqsum);
            }
        }
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * dp  = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 16);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= (int) ne1) {
                    continue;
                }
                const size_t   idx = (size_t) row * nsp + sb;
                const uint4    q   = nib[idx];
                const uint16_t db  = dp[idx];
                const float    dw  = __half2float(*reinterpret_cast<const __half *>(&db));
                const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
                int idot = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if constexpr (MODE == REPACK_NIB_Q4_0) {
                        idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                    } else {
                        int lo, hi;
                        repack_nib_decode<MODE>(qa[j], lo, hi);
                        idot = ggml_cuda_dp4a(lo, xq32[j],     idot);
                        idot = ggml_cuda_dp4a(hi, xq32[j + 4], idot);
                    }
                }
                acc[w2][r] += dw * dx * (float) (idot - 8 * xqsum);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float up_v   = warp_reduce_sum<64>(acc[0][r]);
        const float gate_v = warp_reduce_sum<64>(acc[1][r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            const float g = glu_op == (int) GGML_GLU_OP_SWIGLU ? ggml_cuda_op_silu_single(gate_v) : ggml_cuda_op_gelu_single(gate_v);
            y[row0 + r] = g * up_v;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_1 repacked matvec: Q5_K's shape with a per-sub-block (d, m); the min term adds (x = d*q + m).
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q5_1_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // half-sub-block units for the per-slot case, see Q4_K
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
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
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const float2   dm  = __half22float2(dmp[idx]);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
            int idot = 0;
#pragma unroll
            for (int j = j0; j < j1; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
            }
            acc[r] += dm.x * dx * (float) idot + (half == 0 ? dm.y * sx : 0.0f);
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
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Few-token MoE (speculative verify, a few decoding sequences): the per-slot kernels run one block row
// per (token, slot) pair, so an expert that several tokens picked is read once per token. Every such
// block reads the whole routing (one wave-wide load and a ballot); the first pair that picks an expert
// owns it and applies each weight unit to every later token routed there, the others exit. No sort,
// scratch memory or extra kernel, and each token's arithmetic is the per-slot kernel's in the same order.
// (Scheme from the GLM build on rune.) Needs n_used*n_tok <= 64. GGML_CUDA_NO_MOE_DEDUP=1 disables it.
// Used for the Q4_K gate/up GLU only: the K = 640 Q5_1 down projection measured slower deduplicated
// (96 vs 71 us per call at 3 tokens), its per-slot blocks are short and parallel enough already.
template <int NTMAX>
static __device__ __forceinline__ bool moe_dedup_route(
        const int32_t * __restrict__ ids, const int n_used, const int n_tok, int & e, int (&sl)[NTMAX]) {
    const int lane = threadIdx.x % 64;
    const int a    = blockIdx.y;
    const int tok  = a / n_used;
    const int idv  = lane < n_used*n_tok ? ids[lane] : -1;
    e = __builtin_amdgcn_readlane(idv, a);
    const uint64_t m = __ballot(idv == e);
    if (m & ((uint64_t{1} << (tok*n_used)) - 1)) {
        return false; // an earlier token owns e
    }
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        const uint64_t mt = t < n_tok ? (m >> (t*n_used)) & ((uint64_t{1} << n_used) - 1) : 0;
        sl[t] = (t >= tok && mt) ? __builtin_ctzll(mt) : -1;
    }
    return true;
}

static bool moe_dedup_enabled() {
    static const bool disabled = getenv("GGML_CUDA_NO_MOE_DEDUP") != nullptr;
    return !disabled;
}

// Q5_1 repacked matvec for short rows (ne0 <= 1024, e.g. qwen4exp ffn_down_exps with K = 640): one lane
// per sub-block, SEG lanes per row. The row-per-wave kernel above uses half-sub-block units that each
// load the whole 16-byte nibble chunk (twice the weight traffic) and leaves lanes idle at this K.
template <int SEG, bool HAS_IDS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q5_1_repacked_seg(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS_PER_BLOCK = 256 / SEG;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / SEG;
    const uint32_t sb  = threadIdx.x % SEG;

    float acc = 0.0f;
    if (row < ne1 && sb < n_sub) {
        const size_t   idx = (size_t) row * nsp + sb;
        const uint4    q   = nib[idx];
        const uint32_t qh  = qhp[idx];
        const float2   dm  = __half22float2(dmp[idx]);
        const block_q8_1 * xb = xq + sb;
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);
        const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
        int idot = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu);
            const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu);
            idot = ggml_cuda_dp4a((int) lo, xq32[j],     idot);
            idot = ggml_cuda_dp4a((int) hi, xq32[j + 4], idot);
        }
        acc = dm.x * __low2float(xb->ds) * (float) idot + dm.y * __high2float(xb->ds);
    }
    acc = warp_reduce_sum<SEG>(acc);
    if (sb == 0 && row < ne1) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_1 short rows, thread-packed (from reinstinct's MoE down kernel, crossport R4): thread -> (row,
// sub-block) with 256 / n_sub rows per group so no lane idles at K = 640 (the SEG kernel above keeps
// 12 of 32 lanes idle there), DOWN_R row groups per block with every load issued before the first
// dot, and the per-row partials summed through LDS. One activation load serves DOWN_R rows.
template <int DOWN_R, bool HAS_IDS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q5_1_repacked_pack(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    __shared__ float red[DOWN_R][256];
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t rpb   = 256u / n_sub; // rows per group
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    const uint32_t tid = threadIdx.x;
    const uint32_t r   = tid / n_sub;
    const uint32_t sb  = tid - r * n_sub;
    const bool active  = r < rpb;
    const uint32_t row_base = blockIdx.x * rpb * DOWN_R + r;

    // all loads first (rows clamped, not branched)
    uint4    q[DOWN_R];
    uint32_t qh[DOWN_R];
    half2    dm[DOWN_R];
#pragma unroll
    for (int g = 0; g < DOWN_R; ++g) {
        const uint32_t row = min(row_base + g * rpb, ne1 - 1);
        const size_t   idx = (size_t) row * nsp + (active ? sb : 0);
        q[g]  = nib[idx];
        qh[g] = qhp[idx];
        dm[g] = dmp[idx];
    }
    const block_q8_1 * xb = xq + (active ? sb : 0);
    const int * xq32 = reinterpret_cast<const int *>(xb->qs);
    int x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] = xq32[j];
    }
    const float dx = __low2float(xb->ds);
    const float sx = __high2float(xb->ds);

#pragma unroll
    for (int g = 0; g < DOWN_R; ++g) {
        const uint32_t qa[4] = { q[g].x, q[g].y, q[g].z, q[g].w };
        int idot = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh[g] >> (8 * j))     & 0xFu);
            const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh[g] >> (8 * j + 4)) & 0xFu);
            idot = ggml_cuda_dp4a((int) lo, x[j],     idot);
            idot = ggml_cuda_dp4a((int) hi, x[j + 4], idot);
        }
        const float2 d = __half22float2(dm[g]);
        red[g][tid] = active ? d.x * dx * (float) idot + d.y * sx : 0.0f;
    }
    __syncthreads();
    // one thread per (group, row): sum its n_sub partials
    const uint32_t n_rows_blk = rpb * DOWN_R;
    if (tid < n_rows_blk) {
        const uint32_t g  = tid / rpb;
        const uint32_t rr = tid - g * rpb;
        const uint32_t row = blockIdx.x * rpb * DOWN_R + g * rpb + rr;
        if (row < ne1) {
            float acc = 0.0f;
            const float * src = &red[g][rr * n_sub];
            for (uint32_t s = 0; s < n_sub; ++s) {
                acc += src[s];
            }
            y[row] = acc;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// MoE down projection with the router-weighted expert sum folded in (one token): a block owns R
// output rows; thread -> (expert slot, sub-block) so the n_used*n_sub dot pieces of a row run in one
// pass, each scaled by its slot's weight, and the LDS reduction produces the final mixed row. No
// [ne1][n_used] intermediate and no moe_weighted_reduction launch. Summation order differs from the
// two-kernel path (weights applied per piece), so results are reorder-close, not bit-identical.
template <int R>
static __global__ void __launch_bounds__(256) mul_mat_vec_q5_1_repacked_down_reduce(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        const int32_t * __restrict__ ids, const float * __restrict__ weights,
        const float * __restrict__ scale, float * __restrict__ dst,
        const uint32_t ne0, const uint32_t ne1, const size_t expert_stride,
        const uint32_t x_stride, const int n_used) {
#if defined(GGML_USE_HIP) && defined(GCN)
    __shared__ float red[R][256];
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t tid = threadIdx.x;
    const uint32_t e   = tid / n_sub;
    const uint32_t sb  = tid - e * n_sub;
    const bool active  = e < (uint32_t) n_used;
    const uint32_t ea  = active ? e : 0u;
    const uint32_t sba = active ? sb : 0u;
    const uint8_t * wb = wbase + (size_t) ids[ea] * expert_stride;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wb);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wb + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wb + (size_t) ne1 * nsp * 20);
    const float w_e = active ? weights[ea] * (scale != nullptr ? scale[ea] : 1.0f) : 0.0f;

    const uint32_t row0 = blockIdx.x * R;
    uint4    q[R];
    uint32_t qh[R];
    half2    dm[R];
#pragma unroll
    for (int g = 0; g < R; ++g) {
        const uint32_t row = min(row0 + g, ne1 - 1);
        const size_t   idx = (size_t) row * nsp + sba;
        q[g]  = nib[idx];
        qh[g] = qhp[idx];
        dm[g] = dmp[idx];
    }
    const block_q8_1 * xb = xq + (size_t) ea * x_stride + sba;
    const int * xq32 = reinterpret_cast<const int *>(xb->qs);
    int x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] = xq32[j];
    }
    const float dx = __low2float(xb->ds);
    const float sx = __high2float(xb->ds);

#pragma unroll
    for (int g = 0; g < R; ++g) {
        const uint32_t qa[4] = { q[g].x, q[g].y, q[g].z, q[g].w };
        int idot = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh[g] >> (8 * j))     & 0xFu);
            const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh[g] >> (8 * j + 4)) & 0xFu);
            idot = ggml_cuda_dp4a((int) lo, x[j],     idot);
            idot = ggml_cuda_dp4a((int) hi, x[j + 4], idot);
        }
        const float2 d = __half22float2(dm[g]);
        red[g][tid] = w_e * (d.x * dx * (float) idot + d.y * sx);
    }
    __syncthreads();
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    for (uint32_t g = wave; g < (uint32_t) R; g += 4) {
        float acc = red[g][lane] + red[g][lane + 64] + red[g][lane + 128] + red[g][lane + 192];
        acc = warp_reduce_sum<64>(acc);
        if (lane == 0 && row0 + g < ne1) {
            dst[row0 + g] = acc;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, ids, weights, scale, dst, ne0, ne1, expert_stride, x_stride, n_used);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// K-quant short rows, thread-packed, per expert slot (crossport P3/R4, the Q5_1 pack kernel's mapping
// for the 35B-A3B class Q5_K/Q6_K down projections, K = 768 -> 24 sub-blocks): thread -> (row,
// sub-block), 256 / n_sub rows per group, DOWN_R groups per block with every plane loaded before the
// first dot, per-row partials summed through LDS. One activation load serves DOWN_R rows.
template <ggml_type type, int DOWN_R>
static __global__ void __launch_bounds__(256) mul_mat_vec_kq_repacked_pack(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K, "unsupported type");
    __shared__ float red[DOWN_R][256];
    const uint32_t a = blockIdx.y;
    const uint32_t e = (uint32_t) ids_src1[a];
    wbase += e * expert_stride;
    xq    += (size_t) a * xs_id;
    y     += (size_t) a * dst_s1;

    const uint32_t n_sub   = ne0 >> 5;
    const uint32_t nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint32_t rpb     = 256u / n_sub;
    const size_t   plane   = (size_t) ne1 * nsp;
    const uint4   * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint8_t * p1  = wbase + plane * 16;
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        type == GGML_TYPE_Q4_K ? p1 : type == GGML_TYPE_Q5_K ? p1 + plane * 4 : p1 + plane * 8);
    const uint8_t * pdd = reinterpret_cast<const uint8_t *>(smp) + plane * 2;

    const uint32_t tid = threadIdx.x;
    const uint32_t r   = tid / n_sub;
    const uint32_t sb  = tid - r * n_sub;
    const bool active  = r < rpb;
    const uint32_t sba = active ? sb : 0u;
    const uint32_t row_base = blockIdx.x * rpb * DOWN_R + r;

    // all plane loads first (rows clamped, not branched)
    uint4    q[DOWN_R];
    uint16_t sm[DOWN_R];
    uint32_t qh[DOWN_R];
    uint2    h2[DOWN_R];
    uint32_t dd[DOWN_R];
#pragma unroll
    for (int g = 0; g < DOWN_R; ++g) {
        const uint32_t row = min(row_base + g * rpb, ne1 - 1);
        const size_t   idx = (size_t) row * nsp + sba;
        q[g]  = nib[idx];
        sm[g] = smp[idx];
        if constexpr (type == GGML_TYPE_Q5_K) {
            qh[g] = reinterpret_cast<const uint32_t *>(p1)[idx];
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            h2[g] = reinterpret_cast<const uint2 *>(p1)[idx];
        }
        if constexpr (type == GGML_TYPE_Q6_K) {
            dd[g] = reinterpret_cast<const uint16_t *>(pdd)[(size_t) row * n_super + (sba >> 3)];
        } else {
            dd[g] = reinterpret_cast<const uint32_t *>(pdd)[(size_t) row * n_super + (sba >> 3)];
        }
    }
    const block_q8_1 * xb = xq + sba;
    const int * xq32 = reinterpret_cast<const int *>(xb->qs);
    int x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] = xq32[j];
    }
    const float dx = __low2float(xb->ds);
    const float sx = __high2float(xb->ds);
    int xis0 = 0, xis1 = 0;
    if constexpr (type == GGML_TYPE_Q6_K) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            xis0 = ggml_cuda_dp4a(x[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(x[j + 4], 0x01010101, xis1);
        }
    }

#pragma unroll
    for (int g = 0; g < DOWN_R; ++g) {
        const uint32_t qa[4] = { q[g].x, q[g].y, q[g].z, q[g].w };
        float part;
        if constexpr (type == GGML_TYPE_Q6_K) {
            const uint16_t d_bits = (uint16_t) dd[g];
            const float d  = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float s0 = d * (float)(int)(int8_t)(sm[g] & 0xFFu);
            const float s1 = d * (float)(int)(int8_t)(sm[g] >> 8);
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t ge = 2 * j, go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2[g].x : h2[g].y) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2[g].x : h2[g].y) >> (8 * (go & 3))) & 0xFFu;
                idot0 = ggml_cuda_dp4a((int)(( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he)), x[j],     idot0);
                idot1 = ggml_cuda_dp4a((int)(((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho)), x[j + 4], idot1);
            }
            part = s0 * dx * (float)(idot0 - 32 * xis0) + s1 * dx * (float)(idot1 - 32 * xis1);
        } else {
            const uint16_t d_bits    = (uint16_t)(dd[g] & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd[g] >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm[g] & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm[g] >> 8);
            int idot = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu);
                uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu);
                if constexpr (type == GGML_TYPE_Q5_K) {
                    lo |= repack_spread4((qh[g] >> (8 * j))     & 0xFu);
                    hi |= repack_spread4((qh[g] >> (8 * j + 4)) & 0xFu);
                }
                idot = ggml_cuda_dp4a((int) lo, x[j],     idot);
                idot = ggml_cuda_dp4a((int) hi, x[j + 4], idot);
            }
            part = dsc * dx * (float) idot - deff * sx;
        }
        red[g][tid] = active ? part : 0.0f;
    }
    __syncthreads();
    const uint32_t n_rows_blk = rpb * DOWN_R;
    if (tid < n_rows_blk) {
        const uint32_t g   = tid / rpb;
        const uint32_t rr  = tid - g * rpb;
        const uint32_t row = blockIdx.x * rpb * DOWN_R + g * rpb + rr;
        if (row < ne1) {
            float acc = 0.0f;
            const float * src = &red[g][rr * n_sub];
            for (uint32_t k = 0; k < n_sub; ++k) {
                acc += src[k];
            }
            y[row] = acc;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// per-slot K-quant expert matvec with short rows: the thread-packed kernel (GGML_CUDA_KQ_DOWN_R rows
// groups per block, default 4; 0 = the per-slot 2-row kernels). false: not applicable.
template <ggml_type type>
static bool launch_mul_mat_vec_kq_repacked_pack(const uint8_t * w, const block_q8_1 * xq, float * y,
        const int64_t ne00, const int64_t ne01, const int64_t n_slots, const int32_t * ids,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1, cudaStream_t stream) {
    static const int down_r = getenv("GGML_CUDA_KQ_DOWN_R") ? atoi(getenv("GGML_CUDA_KQ_DOWN_R")) : 4;
    const int64_t n_sub = ne00 / 32;
    if (down_r <= 0 || n_sub < 4 || n_sub > 64) {
        return false;
    }
    const int64_t rpb = 256 / n_sub;
    auto launch = [&](auto r_c) {
        constexpr int R = decltype(r_c)::value;
        const dim3 grid((ne01 + rpb * R - 1) / (rpb * R), n_slots, 1);
        mul_mat_vec_kq_repacked_pack<type, R><<<grid, 256, 0, stream>>>(
            w, xq, y, (uint32_t) ne00, (uint32_t) ne01, ids, expert_stride, xs_id, dst_s1);
    };
    switch (down_r) {
        case 1:  launch(std::integral_constant<int, 1>{}); break;
        case 2:  launch(std::integral_constant<int, 2>{}); break;
        case 8:  launch(std::integral_constant<int, 8>{}); break;
        default: launch(std::integral_constant<int, 4>{}); break;
    }
    return true;
}

template <bool HAS_IDS>
static void launch_mul_mat_vec_q5_1_repacked_seg(
        const uint8_t * w, const block_q8_1 * xq, float * y, const int64_t ne00, const int64_t ne01,
        const int64_t n_slots, const int32_t * ids, const size_t expert_stride, const uint32_t xs_id,
        const uint32_t dst_s1, cudaStream_t stream) {
    const int64_t n_sub = ne00 / 32;
    // GGML_CUDA_Q5_1_DOWN_R: row groups per block for the thread-packed kernel (default 4; 0 = SEG kernel)
    static const int down_r = getenv("GGML_CUDA_Q5_1_DOWN_R") ? atoi(getenv("GGML_CUDA_Q5_1_DOWN_R")) : 4;
    if (down_r > 0 && n_sub >= 4 && n_sub <= 64) {
        const int64_t rpb = 256 / n_sub;
        auto launchp = [&](auto r_c) {
            constexpr int R = decltype(r_c)::value;
            const dim3 grid((ne01 + rpb * R - 1) / (rpb * R), n_slots, 1);
            mul_mat_vec_q5_1_repacked_pack<R, HAS_IDS><<<grid, 256, 0, stream>>>(
                w, xq, y, (uint32_t) ne00, (uint32_t) ne01, ids, expert_stride, xs_id, dst_s1);
        };
        switch (down_r) {
            case 1:  launchp(std::integral_constant<int, 1>{}); break;
            case 2:  launchp(std::integral_constant<int, 2>{}); break;
            case 8:  launchp(std::integral_constant<int, 8>{}); break;
            default: launchp(std::integral_constant<int, 4>{}); break;
        }
        return;
    }
    auto launch = [&](auto seg_c) {
        constexpr int SEG = decltype(seg_c)::value;
        const dim3 grid((ne01 + 256 / SEG - 1) / (256 / SEG), n_slots, 1);
        mul_mat_vec_q5_1_repacked_seg<SEG, HAS_IDS><<<grid, 256, 0, stream>>>(
            w, xq, y, (uint32_t) ne00, (uint32_t) ne01, ids, expert_stride, xs_id, dst_s1);
    };
    if (n_sub <= 8) {
        launch(std::integral_constant<int, 8>{});
    } else if (n_sub <= 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 32>{});
    }
}

// Q6_K repacked matvec. Symmetric quant (value = q-32); the offset is
// folded out via activation half-sums: sum (q-32)x = sum qx - 32 sum x.
// Two signed scales per sub-block, one per 16 weights.
template <bool HAS_IDS, bool HOIST = true>
static __global__ void __launch_bounds__(256) mul_mat_vec_q6k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * h2p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const int hj0 = HAS_IDS ? (int)(half * 2) : 0;
        const int hj1 = HAS_IDS ? hj0 + 2         : 4;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        // per-half activation sums: the -32 fold splits with them
        int xis0 = 0, xis1 = 0;
#pragma unroll
        for (int j = hj0; j < hj1; j++) {
            xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
        }

        if constexpr (HAS_IDS || !HOIST) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx  = (size_t) row * nsp + sb;
            const uint4    q    = nib[idx];
            const uint32_t h2lo = h2p[idx * 2];
            const uint32_t h2hi = h2p[idx * 2 + 1];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t) row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he);
                const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho);
                idot0 = ggml_cuda_dp4a((int) q6lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q6hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 32 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 32 * xis1);
        }
        } else {
        uint4    q[ROWS];
        uint2    h2[ROWS];
        uint16_t sm[ROWS];
        uint16_t db[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) { // rows clamped, store masked (crossport R10)
            const uint32_t row = (uint32_t) min(row0 + r, (int) ne1 - 1);
            const size_t   idx = (size_t) row * nsp + sb;
            q[r]  = nib[idx];
            h2[r] = reinterpret_cast<const uint2 *>(h2p)[idx];
            sm[r] = smp[idx];
            db[r] = ddp[(size_t) row * n_super + (sb >> 3)];
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint32_t h2lo = h2[r].x;
            const uint32_t h2hi = h2[r].y;
            const float d = __half2float(*reinterpret_cast<const __half *>(&db[r]));
            const float dsc_lo = d * (float)(int)(int8_t)(sm[r] & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm[r] >> 8);

            const uint32_t qa[4] = { q[r].x, q[r].y, q[r].z, q[r].w };
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he);
                const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho);
                idot0 = ggml_cuda_dp4a((int) q6lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q6hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 32 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 32 * xis1);
        }
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
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q3_K repacked matvec. Like Q6_K but the quant is 2-bit lo + 1-bit hi
// (no 4-bit nibble plane); reconstruct q3 = lo2 | (hbit << 2) per group.
// Symmetric with bias 4.
template <bool HAS_IDS>
static __global__ void mul_mat_vec_q3k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint2    * lo2p = reinterpret_cast<const uint2 *>(wbase);
    const uint32_t * hi1p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 8);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    float acc[ROWS] = {0.0f, 0.0f};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub; // see Q4_K note
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const int hj0 = HAS_IDS ? (int)(half * 2) : 0;
        const int hj1 = HAS_IDS ? hj0 + 2         : 4;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        int xis0 = 0, xis1 = 0;
#pragma unroll
        for (int j = hj0; j < hj1; j++) {
            xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
            xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
        }

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const size_t   idx    = (size_t) row * nsp + sb;
            const uint2    lo2v   = lo2p[idx];
            const uint32_t qh     = hi1p[idx];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t) row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            const uint32_t lo2lo = lo2v.x, lo2hi = lo2v.y;
            int idot0 = 0, idot1 = 0;
#pragma unroll
            for (int j = hj0; j < hj1; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t lb = ((ge < 4 ? lo2lo : lo2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t hb = ((go < 4 ? lo2lo : lo2hi) >> (8 * (go & 3))) & 0xFFu;
                const uint32_t q3lo = repack_spread2_lo(lb) | repack_spread1_hi((qh >> (8 * j))     & 0xFu);
                const uint32_t q3hi = repack_spread2_lo(hb) | repack_spread1_hi((qh >> (8 * j + 4)) & 0xFu);
                idot0 = ggml_cuda_dp4a((int) q3lo, xq32[j],     idot0);
                idot1 = ggml_cuda_dp4a((int) q3hi, xq32[j + 4], idot1);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 4 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 4 * xis1);
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
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q8_0 repacked matvec — NWAVES wave64s per block, ROWS output rows
// per wave. The original reinstinct tuning used single-wave blocks
// (NWAVES=1) for its MoE-expert shapes; small dense models want the
// 4-wave shape the K-quant matvecs use (NWAVES=4). ROWS=1 doubles the
// wavefront count and wins at out_dim >= 4096 where ROWS=2 leaves too
// few wavefront generations in flight to sustain HBM bandwidth.
// Repacked Q8_0 matvec for short rows (ne0 <= 1024): one lane per 32-weight sub-block and SEG lanes
// per row, so a wave covers 64/SEG rows. The one-row-per-wave kernel below leaves most lanes idle
// when a row has few sub-blocks (qwen4exp hc up-projections, K = 320: 10 of 64 lanes busy, ~90 GB/s).
template <int SEG, bool HAS_IDS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_seg(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, expert_stride, xs_id, dst_s1);
    }
    constexpr int ROWS_PER_BLOCK = 256 / SEG;
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const int4     * qs4     = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);

    const uint32_t row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / SEG;
    const uint32_t sb  = threadIdx.x % SEG;

    float acc = 0.0f;
    if (row < ne1 && sb < n_blocks) {
        const block_q8_1 * xb = xq + sb;
        const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
        const int4 w0 = qs4[((size_t) row * nsp + sb) * 2 + 0];
        const int4 w1 = qs4[((size_t) row * nsp + sb) * 2 + 1];
        const int4 a0 = x4[0];
        const int4 a1 = x4[1];
        int idot = 0;
        idot = ggml_cuda_dp4a(w0.x, a0.x, idot);
        idot = ggml_cuda_dp4a(w0.y, a0.y, idot);
        idot = ggml_cuda_dp4a(w0.z, a0.z, idot);
        idot = ggml_cuda_dp4a(w0.w, a0.w, idot);
        idot = ggml_cuda_dp4a(w1.x, a1.x, idot);
        idot = ggml_cuda_dp4a(w1.y, a1.y, idot);
        idot = ggml_cuda_dp4a(w1.z, a1.z, idot);
        idot = ggml_cuda_dp4a(w1.w, a1.w, idot);
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        acc = __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
    }
    acc = warp_reduce_sum<SEG>(acc);
    if (sb == 0 && row < ne1) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Repacked Q8_0 matvec for few long rows (non-ID): one 256-thread workgroup per row splits K over
// all four waves and reduces through LDS. The row-per-wave kernels launch too few waves to load the
// GPU when there are only a few hundred rows (qwen4exp hc down-projections, 320 rows x K = 10240).
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_splitk(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);

    const uint32_t row = blockIdx.x;
    float acc = 0.0f;
    // work unit: a 16-weight half sub-block, as in the row-per-wave kernel
    for (uint32_t hb = threadIdx.x; hb < n_blocks * 2; hb += 256) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const int * xq32  = reinterpret_cast<const int *>(xb->qs) + half * 4;
        const int * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
        int idot = 0;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
        }
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        acc += __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
    }
    acc = warp_reduce_sum<64>(acc);
    __shared__ float part[4];
    if ((threadIdx.x & 63) == 0) {
        part[threadIdx.x >> 6] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        y[row] = part[0] + part[1] + part[2] + part[3];
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Short rows (NB sub-blocks, NB not a power of two so rows are contiguous): thread t of a
// 256-thread workgroup takes sub-block t % NB of row t / NB, so every lane loads useful bytes;
// rows are summed through LDS. At K = 320 (qwen4exp hc up) the segmented kernel leaves 6 of
// 16 lanes idle: 7.7 -> 7.2 us per call.
template <int NB>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_flat(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    static_assert((NB & (NB - 1)) != 0, "power-of-two NB has padded rows");
    constexpr int R = 256 / NB;
    const int4     * qs4     = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * NB * 32);
    __shared__ float part[R * NB];
    const int t   = threadIdx.x;
    const uint32_t row0 = blockIdx.x * R;
    if (t < R * NB) {
        const uint32_t row = row0 + t / NB;
        const int      sb  = t % NB;
        float acc = 0.0f;
        if (row < ne1) {
            const block_q8_1 * xb = xq + sb;
            const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
            const size_t u = (size_t) row * NB + sb;
            const int4 w0 = qs4[u * 2 + 0];
            const int4 w1 = qs4[u * 2 + 1];
            const int4 a0 = x4[0];
            const int4 a1 = x4[1];
            int idot = 0;
            idot = ggml_cuda_dp4a(w0.x, a0.x, idot); idot = ggml_cuda_dp4a(w0.y, a0.y, idot);
            idot = ggml_cuda_dp4a(w0.z, a0.z, idot); idot = ggml_cuda_dp4a(w0.w, a0.w, idot);
            idot = ggml_cuda_dp4a(w1.x, a1.x, idot); idot = ggml_cuda_dp4a(w1.y, a1.y, idot);
            idot = ggml_cuda_dp4a(w1.z, a1.z, idot); idot = ggml_cuda_dp4a(w1.w, a1.w, idot);
            const uint16_t db = d_plane[u];
            acc = __half2float(*reinterpret_cast<const __half *>(&db)) * __low2float(xb->ds) * (float) idot;
        }
        part[t] = acc;
    }
    __syncthreads();
    if (t < R && row0 + t < ne1) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < NB; j++) {
            sum += part[t * NB + j];
        }
        y[row0 + t] = sum;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Dense Q8_0 decode with a compile-time trip count: LANES lanes per row (64/LANES rows per
// wave), each lane owns ITERS half-sub-block units (n_blocks*2 == ITERS*LANES) and issues all of
// their weight/scale/activation loads before the dot products. The runtime-bounded loop of
// mul_mat_vec_q8_0_repacked keeps only one unit in flight and mid-size tensors (17-33 MB)
// stream at 565-670 GB/s against ~840 for the 675 MB output head.
template <int ITERS, int LANES>
static __global__ void __launch_bounds__(64) mul_mat_vec_q8_0_repacked_rowu(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);
    const int lane = threadIdx.x % LANES;
    const uint32_t row = blockIdx.x * (64 / LANES) + threadIdx.x / LANES;
    const bool valid = row < ne1;
    const uint32_t rr = valid ? row : 0;

    int4     w[ITERS];
    int4     xv[ITERS];
    uint16_t db[ITERS];
    float    dx[ITERS];
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t hb = lane + j * LANES;
        const uint32_t sb = hb >> 1, half = hb & 1;
        w[j]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) rr * nsp + sb) * 8 + half * 4);
        db[j] = d_plane[(size_t) rr * nsp + sb];
        const block_q8_1 * xb = xq + sb;
        xv[j] = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
        dx[j] = __low2float(xb->ds);
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        int idot = 0;
        idot = ggml_cuda_dp4a(w[j].x, xv[j].x, idot);
        idot = ggml_cuda_dp4a(w[j].y, xv[j].y, idot);
        idot = ggml_cuda_dp4a(w[j].z, xv[j].z, idot);
        idot = ggml_cuda_dp4a(w[j].w, xv[j].w, idot);
        acc += __half2float(*reinterpret_cast<const __half *>(&db[j])) * dx[j] * (float) idot;
    }
    acc = warp_reduce_sum<LANES>(acc);
    if (lane == 0 && valid) {
        y[row] = acc;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Up to three dense Q8_0 matvecs that read the same activation (GDN qkv + z, attention
// q/k/v) in one launch over their concatenated rows; per row identical to
// mul_mat_vec_q8_0_repacked<1, 1, false>. Saves a launch and its ramp/tail per extra matrix.
struct q8_multi_args {
    const uint8_t * w[3];
    float *         y[3];
    uint32_t        ne1[3];
    uint32_t        start[3];
};

static __global__ void __launch_bounds__(64) mul_mat_vec_q8_0_repacked_multi(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t = blockIdx.x >= args.start[2] ? 2 : (blockIdx.x >= args.start[1] ? 1 : 0);
    const uint8_t * wbase = args.w[t];
    const uint32_t  ne1   = args.ne1[t];
    const uint32_t  row   = blockIdx.x - args.start[t];

    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);
    const int lane = threadIdx.x;

    float acc = 0.0f;
    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;
        const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
        const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
        const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));
        int idot = 0;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
        }
        acc += dw * dx * (float) idot;
    }
    acc = warp_reduce_sum<64>(acc);
    if (lane == 0) {
        args.y[t][row] = acc;
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// The multi kernel with every load issued before the first dot (crossport R10: the runtime-bounded
// loop above keeps ~one half-sub-block of weights in flight per lane). ITERS half sub-blocks per lane,
// lanes past the row's end re-read its first half with a zero activation scale.
template <int ITERS>
static __global__ void __launch_bounds__(64) mul_mat_vec_q8_0_repacked_multi_u(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t = blockIdx.x >= args.start[2] ? 2 : (blockIdx.x >= args.start[1] ? 1 : 0);
    const uint8_t * wbase = args.w[t];
    const uint32_t  ne1   = args.ne1[t];
    const uint32_t  row   = blockIdx.x - args.start[t];

    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);
    const uint32_t lane   = threadIdx.x;
    const uint32_t n_half = n_blocks * 2;

    int4     w[ITERS];
    int4     xv[ITERS];
    uint16_t db[ITERS];
    float    dx[ITERS];
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t hb  = lane + j * 64;
        const bool     ok  = hb < n_half;
        const uint32_t hbc = ok ? hb : 0u;
        const uint32_t sb = hbc >> 1, half = hbc & 1;
        w[j]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
        db[j] = d_plane[(size_t) row * nsp + sb];
        const block_q8_1 * xb = xq + sb;
        xv[j] = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
        dx[j] = ok ? __low2float(xb->ds) : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        int idot = 0;
        idot = ggml_cuda_dp4a(w[j].x, xv[j].x, idot);
        idot = ggml_cuda_dp4a(w[j].y, xv[j].y, idot);
        idot = ggml_cuda_dp4a(w[j].z, xv[j].z, idot);
        idot = ggml_cuda_dp4a(w[j].w, xv[j].w, idot);
        acc += __half2float(*reinterpret_cast<const __half *>(&db[j])) * dx[j] * (float) idot;
    }
    acc = warp_reduce_sum<64>(acc);
    if (lane == 0) {
        args.y[t][row] = acc;
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <bool HAS_IDS>
static void launch_mul_mat_vec_q8_0_repacked_seg(
        const uint8_t * w, const block_q8_1 * xq, float * y, const int64_t ne00, const int64_t ne01,
        const int64_t n_slots, const int32_t * ids, const size_t expert_stride, const uint32_t xs_id,
        const uint32_t dst_s1, cudaStream_t stream) {
    const int64_t n_blocks = ne00 / 32;
    auto launch = [&](auto seg_c) {
        constexpr int SEG = decltype(seg_c)::value;
        const dim3 grid((ne01 + 256 / SEG - 1) / (256 / SEG), n_slots, 1);
        mul_mat_vec_q8_0_repacked_seg<SEG, HAS_IDS><<<grid, 256, 0, stream>>>(
            w, xq, y, (uint32_t) ne00, (uint32_t) ne01, ids, expert_stride, xs_id, dst_s1);
    };
    if (n_blocks <= 8) {
        launch(std::integral_constant<int, 8>{});
    } else if (n_blocks <= 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 32>{});
    }
}

template <int ROWS, int NWAVES, bool HAS_IDS, bool HOIST = true>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        // decode (one token): slot a maps directly — expert ids_raw[a],
        // activation column a, dst column a. No compaction kernel needed
        // (ids_src1 carries the RAW ids tensor here).
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * NWAVES) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[ROWS] = {};

    // Work unit: a 16-weight half sub-block (4 sdot4s). At small ne0 a
    // full-sub-block unit leaves lanes idle (ne0=1024 -> 32 sub-blocks
    // for 64 lanes); halves keep the wave full down to ne0=1024.
    // (8-weight quarters were tried and regress: 4x the d-plane traffic
    // outweighs the extra balance.)
    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

        if constexpr (HAS_IDS || !HOIST) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
            const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
            const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));
            int idot = 0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
            }
            acc[r] += dw * dx * (float) idot;
        }
        } else {
        int4     w[ROWS];
        uint16_t db[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) { // rows clamped, store masked (crossport R10)
            const uint32_t row = (uint32_t) min(row0 + r, (int) ne1 - 1);
            w[r]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
            db[r] = d_plane[(size_t) row * nsp + sb];
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float dw = __half2float(*reinterpret_cast<const __half *>(&db[r]));
            int idot = 0;
            idot = ggml_cuda_dp4a(w[r].x, xq32[0], idot);
            idot = ggml_cuda_dp4a(w[r].y, xq32[1], idot);
            idot = ggml_cuda_dp4a(w[r].z, xq32[2], idot);
            idot = ggml_cuda_dp4a(w[r].w, xq32[3], idot);
            acc[r] += dw * dx * (float) idot;
        }
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
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}


// Fused gate+up Q4_K matvec with GLU epilogue. Walks BOTH weight slabs
// in one sub-block loop (one activation read, one launch) and writes
// y[row] = glu(gate_dot) * up_dot — replacing two matvec launches plus
// an elementwise GLU op. This is what canonical mmvq fuses too; without
// it the repacked MoE decode pays ~2x the launches (measured -7% on
// 35B-A3B). Used for both dense MUL_MAT (ids == nullptr) and
// MUL_MAT_ID decode. ID path uses half-sub-block units (small-K expert
// tensors; min term on the even half).
template <bool HAS_IDS, int ROWS = 2, int MIN_BLOCKS = 1, bool FENCE = false, bool HOIST = true>
static __global__ void __launch_bounds__(256, MIN_BLOCKS) mul_mat_vec_q4k_repacked_glu(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y; // slot index; see direct-map note above
        const uint32_t e = (uint32_t) ids_src1[a];
        wup   += e * expert_stride;
        wgate += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    // ROWS: template parameter
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;

    const uint8_t * wb[2] = { wup, wgate };
    float acc[2][ROWS] = {};

    const uint32_t n_unit = HAS_IDS ? n_sub * 2 : n_sub;
    for (uint32_t u = lane; u < n_unit; u += 64) {
        const uint32_t sb   = HAS_IDS ? (u >> 1) : u;
        const uint32_t half = HAS_IDS ? (u & 1)  : 0;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs);

        if constexpr (HAS_IDS || !HOIST) {
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
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
                const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu);
                const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
                const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
                const int j0 = HAS_IDS ? (int)(half * 2) : 0;
            const int j1 = HAS_IDS ? j0 + 2          : 4;
                int idot = 0;
#pragma unroll
                for (int j = j0; j < j1; j++) {
                    idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                    idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                }
                acc[w2][r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
            }
        }
        } else {
        // both slabs' rows loaded before the first dot; rows clamped, store masked (crossport R10)
        uint4    q[2][ROWS];
        uint16_t sm[2][ROWS];
        uint32_t dd[2][ROWS];
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const uint32_t row = (uint32_t) min(row0 + r, (int) ne1 - 1);
                q[w2][r]  = nib[(size_t) row * nsp + sb];
                sm[w2][r] = smp[(size_t) row * nsp + sb];
                dd[w2][r] = ddp[(size_t) row * n_super + (sb >> 3)];
            }
        }
        if constexpr (FENCE) {
            __builtin_amdgcn_sched_barrier(0);
        }
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const uint16_t d_bits    = (uint16_t)(dd[w2][r] & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd[w2][r] >> 16);
                const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm[w2][r] & 0xFFu);
                const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm[w2][r] >> 8);
                const uint32_t qa[4] = { q[w2][r].x, q[w2][r].y, q[w2][r].z, q[w2][r].w };
                const int j0 = HAS_IDS ? (int)(half * 2) : 0;
                const int j1 = HAS_IDS ? j0 + 2          : 4;
                int idot = 0;
#pragma unroll
                for (int j = j0; j < j1; j++) {
                    idot = ggml_cuda_dp4a((int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot);
                    idot = ggml_cuda_dp4a((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
                }
                acc[w2][r] += dsc * dx * (float) idot - (half == 0 ? deff * sx : 0.0f);
            }
        }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float up_v   = warp_reduce_sum<64>(acc[0][r]);
        const float gate_v = warp_reduce_sum<64>(acc[1][r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            const float g = glu_op == (int) GGML_GLU_OP_SWIGLU
                ? ggml_cuda_op_silu_single(gate_v)
                : ggml_cuda_op_gelu_single(gate_v);
            y[row0 + r] = g * up_v;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op, ids_src1, ids_dst,
                     expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Dense Q8_0 gate+up matvec with GLU epilogue (the shared-expert FFN):
// mul_mat_vec_q8_0_repacked<ROWS, 4> walking both weight slabs per unit.
template <int ROWS>
static __global__ void mul_mat_vec_q8_0_repacked_glu(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const uint8_t * wb[2] = { wup, wgate };

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[2][ROWS] = {};

    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const int      * qs_int  = reinterpret_cast<const int *>(wb[w2]);
            const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 32);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = row0 + r;
                if (row >= (int) ne1) {
                    continue;
                }
                const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
                const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
                const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));

                int idot = 0;
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
                }
                acc[w2][r] += dw * dx * (float) idot;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float up_v   = warp_reduce_sum<64>(acc[0][r]);
        const float gate_v = warp_reduce_sum<64>(acc[1][r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            const float g = glu_op == (int) GGML_GLU_OP_SWIGLU
                ? ggml_cuda_op_silu_single(gate_v)
                : ggml_cuda_op_gelu_single(gate_v);
            y[row0 + r] = g * up_v;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// The shared-expert Q8_0 gate+up GLU with one row per wave and both slabs' ITERS half
// sub-blocks per lane loaded before the first dot (R10 pattern). Rows past ne1 are clamped
// and their store masked.
template <int ITERS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_glu_u(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const int glu_op) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const uint8_t * wb[2] = { wup, wgate };
    const uint32_t wave = threadIdx.x >> 6;
    const uint32_t lane = threadIdx.x & 63;
    const uint32_t row  = blockIdx.x * 4 + wave;
    const bool valid = row < ne1;
    const uint32_t rr = valid ? row : 0u;
    const uint32_t n_half = n_blocks * 2;

    int4     w[2][ITERS];
    uint16_t db[2][ITERS];
    int4     xv[ITERS];
    float    dx[ITERS];
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t hb  = lane + j * 64;
        const bool     ok  = hb < n_half;
        const uint32_t hbc = ok ? hb : 0u;
        const uint32_t sb = hbc >> 1, half = hbc & 1;
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const int      * qs_int  = reinterpret_cast<const int *>(wb[w2]);
            const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 32);
            w[w2][j]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) rr * nsp + sb) * 8 + half * 4);
            db[w2][j] = d_plane[(size_t) rr * nsp + sb];
        }
        const block_q8_1 * xb = xq + sb;
        xv[j] = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
        dx[j] = ok ? __low2float(xb->ds) : 0.0f;
    }
    float acc[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            int idot = 0;
            idot = ggml_cuda_dp4a(w[w2][j].x, xv[j].x, idot);
            idot = ggml_cuda_dp4a(w[w2][j].y, xv[j].y, idot);
            idot = ggml_cuda_dp4a(w[w2][j].z, xv[j].z, idot);
            idot = ggml_cuda_dp4a(w[w2][j].w, xv[j].w, idot);
            acc[w2] += __half2float(*reinterpret_cast<const __half *>(&db[w2][j])) * dx[j] * (float) idot;
        }
    }
    const float up_v   = warp_reduce_sum<64>(acc[0]);
    const float gate_v = warp_reduce_sum<64>(acc[1]);
    if (lane == 0 && valid) {
        const float g = glu_op == (int) GGML_GLU_OP_SWIGLU ? ggml_cuda_op_silu_single(gate_v) : ggml_cuda_op_gelu_single(gate_v);
        y[row] = g * up_v;
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne0, ne1, glu_op);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// MoE decode Q4_K gate+up GLU for K = 16*ITERS sub-blocks: 16 lanes per row (4 rows per wave,
// 16 per workgroup), each lane takes ITERS whole sub-blocks, so every loaded byte is used, the
// work divides evenly and the row reduction is 16-wide. The half-unit kernel above splits
// K = 2560 into 160 halves over 64 lanes (3 vs 2 iterations) and loads each nibble word twice.
// (measured in qwen4exp decode: 16 lanes 34.1 us, 8 lanes 36.8, 4 lanes 43.4; capping registers
// for 6 or 8 waves/SIMD: 37.1 / 120 us)
template <int ITERS, int LANES = 16>
static __global__ void __launch_bounds__(256) mul_mat_vec_q4k_repacked_glu16(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne1, const int glu_op, const int32_t * __restrict__ ids,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr uint32_t n_sub   = LANES * ITERS;
    constexpr uint32_t nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    constexpr uint32_t n_super = n_sub / 8;
    const uint32_t a = blockIdx.y;
    const uint32_t e = (uint32_t) ids[a];
    xq += (size_t) a * xs_id;
    y  += (size_t) a * dst_s1;
    const uint8_t * wb[2] = { wup + e * expert_stride, wgate + e * expert_stride };

    const int l16 = threadIdx.x % LANES;
    const uint32_t row = blockIdx.x * (256 / LANES) + threadIdx.x / LANES;
    const bool valid = row < ne1;
    const uint32_t rr = valid ? row : 0;

    float acc[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t sb = l16 + LANES * j;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int4 xa = *reinterpret_cast<const int4 *>(xb->qs);
        const int4 xc = *reinterpret_cast<const int4 *>(xb->qs + 16);
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
            const uint4    q  = nib[(size_t) rr * nsp + sb];
            const uint16_t sm = smp[(size_t) rr * nsp + sb];
            const uint32_t dd = ddp[(size_t) rr * n_super + (sb >> 3)];
            const uint16_t d_bits = (uint16_t)(dd & 0xFFFF), dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits)) * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
            int idot = 0;
            idot = ggml_cuda_dp4a((int)( q.x       & 0x0F0F0F0Fu), xa.x, idot);
            idot = ggml_cuda_dp4a((int)( q.y       & 0x0F0F0F0Fu), xa.y, idot);
            idot = ggml_cuda_dp4a((int)( q.z       & 0x0F0F0F0Fu), xa.z, idot);
            idot = ggml_cuda_dp4a((int)( q.w       & 0x0F0F0F0Fu), xa.w, idot);
            idot = ggml_cuda_dp4a((int)((q.x >> 4) & 0x0F0F0F0Fu), xc.x, idot);
            idot = ggml_cuda_dp4a((int)((q.y >> 4) & 0x0F0F0F0Fu), xc.y, idot);
            idot = ggml_cuda_dp4a((int)((q.z >> 4) & 0x0F0F0F0Fu), xc.z, idot);
            idot = ggml_cuda_dp4a((int)((q.w >> 4) & 0x0F0F0F0Fu), xc.w, idot);
            acc[w2] += dsc * dx * (float) idot - deff * sx;
        }
    }
    const float up_v   = warp_reduce_sum<LANES>(acc[0]);
    const float gate_v = warp_reduce_sum<LANES>(acc[1]);
    if (l16 == 0 && valid) {
        const float g = glu_op == (int) GGML_GLU_OP_SWIGLU ? ggml_cuda_op_silu_single(gate_v) : ggml_cuda_op_gelu_single(gate_v);
        y[row] = g * up_v;
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne1, glu_op, ids, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// glu16 with the down projection's activation quantize folded in: 512 threads = 32 rows = one q8_1
// block of outputs per launch block, written in the [slot][pad(ne1)/32] layout the expert kernels read
// (quantize_q8_1's arithmetic, so the blocks are bit-identical to the separate launch). Also writes y.
template <int ITERS, int LANES = 16>
static __global__ void __launch_bounds__(512) mul_mat_vec_q4k_repacked_glu16_q8(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne1, const int glu_op, const int32_t * __restrict__ ids,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1,
        block_q8_1 * __restrict__ yq, const uint32_t yq_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr uint32_t n_sub   = LANES * ITERS;
    constexpr uint32_t nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    constexpr uint32_t n_super = n_sub / 8;
    constexpr int ROWS = 512 / LANES;
    static_assert(ROWS == QK8_1, "one q8_1 block of outputs per launch block");
    __shared__ float s_y[ROWS];
    const uint32_t a = blockIdx.y;
    const uint32_t e = (uint32_t) ids[a];
    xq += (size_t) a * xs_id;
    y  += (size_t) a * dst_s1;
    yq += (size_t) a * yq_stride;
    const uint8_t * wb[2] = { wup + e * expert_stride, wgate + e * expert_stride };

    const int l16 = threadIdx.x % LANES;
    const uint32_t row = blockIdx.x * ROWS + threadIdx.x / LANES; // ne1 % 32 == 0: always valid

    float acc[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t sb = l16 + LANES * j;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const float sx = __high2float(xb->ds);
        const int4 xa = *reinterpret_cast<const int4 *>(xb->qs);
        const int4 xc = *reinterpret_cast<const int4 *>(xb->qs + 16);
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
            const uint4    q  = nib[(size_t) row * nsp + sb];
            const uint16_t sm = smp[(size_t) row * nsp + sb];
            const uint32_t dd = ddp[(size_t) row * n_super + (sb >> 3)];
            const uint16_t d_bits = (uint16_t)(dd & 0xFFFF), dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half *>(&d_bits)) * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
            int idot = 0;
            idot = ggml_cuda_dp4a((int)( q.x       & 0x0F0F0F0Fu), xa.x, idot);
            idot = ggml_cuda_dp4a((int)( q.y       & 0x0F0F0F0Fu), xa.y, idot);
            idot = ggml_cuda_dp4a((int)( q.z       & 0x0F0F0F0Fu), xa.z, idot);
            idot = ggml_cuda_dp4a((int)( q.w       & 0x0F0F0F0Fu), xa.w, idot);
            idot = ggml_cuda_dp4a((int)((q.x >> 4) & 0x0F0F0F0Fu), xc.x, idot);
            idot = ggml_cuda_dp4a((int)((q.y >> 4) & 0x0F0F0F0Fu), xc.y, idot);
            idot = ggml_cuda_dp4a((int)((q.z >> 4) & 0x0F0F0F0Fu), xc.z, idot);
            idot = ggml_cuda_dp4a((int)((q.w >> 4) & 0x0F0F0F0Fu), xc.w, idot);
            acc[w2] += dsc * dx * (float) idot - deff * sx;
        }
    }
    const float up_v   = warp_reduce_sum<LANES>(acc[0]);
    const float gate_v = warp_reduce_sum<LANES>(acc[1]);
    if (l16 == 0) {
        const float g = glu_op == (int) GGML_GLU_OP_SWIGLU ? ggml_cuda_op_silu_single(gate_v) : ggml_cuda_op_gelu_single(gate_v);
        const float v = g * up_v;
        y[row] = v;
        s_y[threadIdx.x / LANES] = v;
    }
    __syncthreads();
    const int t = threadIdx.x;
    if (t < QK8_1) {
        const float xi = s_y[t];
        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        block_q8_1 * b = yq + blockIdx.x;
        b->qs[t] = q;
        if (t == 0) {
            b->ds = make_half2(d, sum);
        }
    }
    if (blockIdx.x == 0) {
        // the row padding blocks: quantize_q8_1 writes them as zeros
        const uint32_t n_blk = ne1 / QK8_1;
        uint32_t * pz = reinterpret_cast<uint32_t *>(yq + n_blk);
        for (uint32_t i = t; i < (yq_stride - n_blk) * (uint32_t) (sizeof(block_q8_1) / 4); i += 512) {
            pz[i] = 0;
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne1, glu_op, ids, expert_stride, xs_id, dst_s1, yq, yq_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q4_K gate+up GLU for a few tokens with the expert dedup (see moe_dedup_route): the glu16 lane
// mapping, each sub-block's weights and scales loaded once and applied to every routed token.
// The activation is per token (xq + t*xs), shared by all of its slots.
template <int ITERS, int NTMAX, int LANES = 16>
static __global__ void __launch_bounds__(256) mul_mat_vec_q4k_repacked_glu16_dedup(
        const uint8_t * __restrict__ wup, const uint8_t * __restrict__ wgate,
        const block_q8_1 * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne1, const int glu_op, const int32_t * __restrict__ ids,
        const size_t expert_stride, const uint32_t xs, const uint32_t dst_s1, const int n_used, const int n_tok) {
#if defined(GGML_USE_HIP) && defined(GCN)
    int e, sl[NTMAX];
    if (!moe_dedup_route<NTMAX>(ids, n_used, n_tok, e, sl)) {
        return;
    }
    constexpr uint32_t n_sub   = LANES * ITERS;
    constexpr uint32_t nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    constexpr uint32_t n_super = n_sub / 8;
    const uint8_t * wb[2] = { wup + (size_t) e * expert_stride, wgate + (size_t) e * expert_stride };

    const int l16 = threadIdx.x % LANES;
    const uint32_t row = blockIdx.x * (256 / LANES) + threadIdx.x / LANES;
    const bool valid = row < ne1;
    const uint32_t rr = valid ? row : 0;

    float acc[NTMAX][2];
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        acc[t][0] = 0.0f;
        acc[t][1] = 0.0f;
    }
#pragma unroll
    for (int j = 0; j < ITERS; j++) {
        const uint32_t sb = l16 + LANES * j;
        uint4 q[2];
        float dsc[2], deff[2];
#pragma unroll
        for (int w2 = 0; w2 < 2; w2++) {
            const uint4    * nib = reinterpret_cast<const uint4 *>(wb[w2]);
            const uint16_t * smp = reinterpret_cast<const uint16_t *>(wb[w2] + (size_t) ne1 * nsp * 16);
            const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wb[w2] + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);
            q[w2] = nib[(size_t) rr * nsp + sb];
            const uint16_t sm = smp[(size_t) rr * nsp + sb];
            const uint32_t dd = ddp[(size_t) rr * n_super + (sb >> 3)];
            const uint16_t d_bits = (uint16_t)(dd & 0xFFFF), dmin_bits = (uint16_t)(dd >> 16);
            dsc[w2]  = __half2float(*reinterpret_cast<const __half *>(&d_bits)) * (float)(sm & 0xFFu);
            deff[w2] = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
        }
#pragma unroll
        for (int t = 0; t < NTMAX; ++t) {
            if (sl[t] < 0) {
                continue;
            }
            const block_q8_1 * xb = xq + (size_t) t * xs + sb;
            const float dx = __low2float(xb->ds);
            const float sx = __high2float(xb->ds);
            const int4 xa = *reinterpret_cast<const int4 *>(xb->qs);
            const int4 xc = *reinterpret_cast<const int4 *>(xb->qs + 16);
#pragma unroll
            for (int w2 = 0; w2 < 2; w2++) {
                int idot = 0;
                idot = ggml_cuda_dp4a((int)( q[w2].x       & 0x0F0F0F0Fu), xa.x, idot);
                idot = ggml_cuda_dp4a((int)( q[w2].y       & 0x0F0F0F0Fu), xa.y, idot);
                idot = ggml_cuda_dp4a((int)( q[w2].z       & 0x0F0F0F0Fu), xa.z, idot);
                idot = ggml_cuda_dp4a((int)( q[w2].w       & 0x0F0F0F0Fu), xa.w, idot);
                idot = ggml_cuda_dp4a((int)((q[w2].x >> 4) & 0x0F0F0F0Fu), xc.x, idot);
                idot = ggml_cuda_dp4a((int)((q[w2].y >> 4) & 0x0F0F0F0Fu), xc.y, idot);
                idot = ggml_cuda_dp4a((int)((q[w2].z >> 4) & 0x0F0F0F0Fu), xc.z, idot);
                idot = ggml_cuda_dp4a((int)((q[w2].w >> 4) & 0x0F0F0F0Fu), xc.w, idot);
                acc[t][w2] += dsc[w2] * dx * (float) idot - deff[w2] * sx;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
        acc[t][0] = warp_reduce_sum<LANES>(acc[t][0]);
        acc[t][1] = warp_reduce_sum<LANES>(acc[t][1]);
    }
    if (l16 == 0 && valid) {
#pragma unroll
        for (int t = 0; t < NTMAX; ++t) {
            if (sl[t] >= 0) {
                const float g = glu_op == (int) GGML_GLU_OP_SWIGLU ? ggml_cuda_op_silu_single(acc[t][1]) : ggml_cuda_op_gelu_single(acc[t][1]);
                y[(size_t) (t*n_used + sl[t]) * dst_s1 + row] = g * acc[t][0];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wup, wgate, xq, y, ne1, glu_op, ids, expert_stride, xs, dst_s1, n_used, n_tok);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// MMQ activation block as staged in LDS: 40 B with qs 8 B-aligned (crossport finding: the 36 B
// block_q8_1 leaves qs 4 B-aligned, so every activation word read split into b32 pairs; with 5
// blocks per token row the 200 B row stride also keeps b64 reads conflict-free) and the scales
// already widened, one conversion per block at staging instead of per inner-loop iteration.
struct __align__(8) mmq_x8 {
    float  d;
    float  s;
    int8_t qs[32];
};
static_assert(sizeof(mmq_x8) == 40, "mmq_x8 is 40 bytes");
static __device__ __forceinline__ void mmq_x8_stage(mmq_x8 & dst, const block_q8_1 & src) {
    dst.d = __low2float(src.ds);
    dst.s = __high2float(src.ds);
    const int * q = reinterpret_cast<const int *>(src.qs);
    int * o = reinterpret_cast<int *>(dst.qs);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        o[j] = q[j];
    }
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
// MUL_MAT_ID instantiations (TN=1) are light enough for 4 waves/SIMD (64 VGPR cap)
#define MMQ_RP_OCC_ID 4

template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? MMQ_RP_OCC_ID : 2) mmq_gemm_q4k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sW [MMQ_RP_BM][MMQ_RP_BK + 1];     // packed nibbles
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK + 1];     // (dsc, deff)
    __shared__ mmq_x8 sX [(16 * TN_)][MMQ_RP_BK + 1]; // int8 activations

    float acc[MMQ_RP_TM][TN_] = {};

    constexpr int LDW = MMQ_RP_BM * MMQ_RP_BK / 256; // tile elems per thread

    // activation row for this thread's sX slot, resolved once
    static_assert((16 * TN_) * MMQ_RP_BK <= 256, "sX staging assumes one slot per thread");
    const int xlr = t / MMQ_RP_BK, xlk = t % MMQ_RP_BK;
    const bool xstage = t < (16 * TN_) * MMQ_RP_BK;
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + xlr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

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
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + xlr < n_tok;
            }
            if (xval && sb0 + xlk < n_sub) {
                if constexpr (HAS_IDS) {
                    mmq_x8_stage(sX[xlr][xlk], xq[xoff + sb0 + xlk]);
                } else {
                    mmq_x8_stage(sX[xlr][xlk], xq[(size_t) (tok0 + xlr) * x_stride + sb0 + xlk]);
                }
            } else {
                sX[xlr][xlk].d = 0.0f; sX[xlr][xlk].s = 0.0f;
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
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
                const float sx = xb->s;
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
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// MUL_MAT_ID variant of the Q4_K tile GEMM: one wave per workgroup,
// 64 rows x 16 assignments per tile. Lane l owns rows (l&15)+16i and
// assignments (l>>4)+4j, a 4x4 register tile, so each unpacked weight
// sub-block feeds 4 tokens and each token read feeds 4 rows. The next
// K chunk is prefetched into registers while the current one runs.
// Nibbles are split to int8 once at LDS staging instead of per dp4a.
template <int BK>
static __global__ void __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(2, 2))) mmq_gemm_q4k_repacked_id_w1(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    static_assert(BK == 1 || BK == 2 || BK == 4, "BK must be a power of two <= 4");
    const int l  = threadIdx.x;
    const int rg = l & 15;
    const int tg = l >> 4;
    const uint32_t row0 = blockIdx.x * 64;
    if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
        return;
    }
    const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
    const uint32_t a_base = (uint32_t) expert_bounds[e] + (blockIdx.y - (uint32_t) tile_off[e]) * 16;
    const uint32_t a_end  = (uint32_t) expert_bounds[e + 1];
    wbase += e * expert_stride;

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 2);

    constexpr int NW = BK; // weight (row, kk) items per lane
    __shared__ uint4  sW [64][2 * BK + 1]; // unpacked nibbles (lo, hi) per kk, padded rows
    __shared__ float2 sWs[64][BK];         // (dsc, deff)
    __shared__ uint4  sXq[16][2 * BK + 1]; // int8 activations, padded rows
    __shared__ float2 sXd[16][BK];         // (dx, sx)

    // token staging: lane l < 16*BK owns slot (l/BK, l%BK), row resolved once
    // (upper lanes and out-of-range slots alias a valid row: no extra traffic)
    const int xl = (l / BK) & 15;
    const int xk = l % BK;
    const uint32_t xa = min(a_base + xl, a_end - 1);
    const block_q8_1 * xrow = xq + (size_t) (uint32_t) ids_src1[xa] * x_stride;

    float acc[4][4] = {};

    // Prefetch loads are unconditional (indices clamped) so the compiler
    // keeps them in flight across the compute; out-of-range weight slots
    // get zero scales at staging. Out-of-range assignments stage a valid
    // row's (finite) data and are never written back.
    uint4 pw[NW]; uint16_t psm[NW]; uint32_t pdd[NW]; int px[9];
    auto gload = [&](uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < NW; i++) {
            const int it = l + 64 * i;
            const uint32_t wrow = min(row0 + it / BK, ne1 - 1);
            const uint32_t sb   = min(sb0 + it % BK, n_sub - 1);
            pw[i]  = nib[(size_t) wrow * nsp + sb];
            psm[i] = smp[(size_t) wrow * nsp + sb];
            pdd[i] = ddp[(size_t) wrow * n_super + (sb >> 3)];
        }
        const int * src = reinterpret_cast<const int *>(xrow + min(sb0 + xk, n_sub - 1));
#pragma unroll
        for (int j = 0; j < 9; j++) {
            px[j] = src[j];
        }
    };
    auto lstore = [&](uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < NW; i++) {
            const int it = l + 64 * i;
            const int lr = it / BK, lk = it % BK;
            sW[lr][2 * lk]     = make_uint4( pw[i].x       & 0x0F0F0F0Fu,  pw[i].y       & 0x0F0F0F0Fu,
                                              pw[i].z       & 0x0F0F0F0Fu,  pw[i].w       & 0x0F0F0F0Fu);
            sW[lr][2 * lk + 1] = make_uint4((pw[i].x >> 4) & 0x0F0F0F0Fu, (pw[i].y >> 4) & 0x0F0F0F0Fu,
                                             (pw[i].z >> 4) & 0x0F0F0F0Fu, (pw[i].w >> 4) & 0x0F0F0F0Fu);
            const uint16_t d_bits = (uint16_t)(pdd[i] & 0xFFFF), m_bits = (uint16_t)(pdd[i] >> 16);
            const bool ok = row0 + lr < ne1 && sb0 + lk < n_sub;
            sWs[lr][lk] = ok ? make_float2(
                __half2float(*reinterpret_cast<const __half *>(&d_bits)) * (float)(psm[i] & 0xFFu),
                __half2float(*reinterpret_cast<const __half *>(&m_bits)) * (float)(psm[i] >> 8))
                : make_float2(0.0f, 0.0f);
        }
        if (l < 16 * BK) {
            sXq[xl][2 * xk]     = make_uint4(px[1], px[2], px[3], px[4]);
            sXq[xl][2 * xk + 1] = make_uint4(px[5], px[6], px[7], px[8]);
            sXd[xl][xk] = __half22float2(*reinterpret_cast<const half2 *>(&px[0]));
        }
    };

    gload(0);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += BK) {
        __syncthreads();
        lstore(sb0);
        __syncthreads();
        gload(sb0 + BK); // clamped on the last step
#pragma unroll 1
        for (int kk = 0; kk < BK; kk++) {
            int   xv[4][8];
            float dx[4], sx[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint4 a = sXq[tg + 4 * j][2 * kk], b = sXq[tg + 4 * j][2 * kk + 1];
                xv[j][0] = a.x; xv[j][1] = a.y; xv[j][2] = a.z; xv[j][3] = a.w;
                xv[j][4] = b.x; xv[j][5] = b.y; xv[j][6] = b.z; xv[j][7] = b.w;
                const float2 d = sXd[tg + 4 * j][kk];
                dx[j] = d.x; sx[j] = d.y;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int lr = rg + 16 * i;
                const uint4  qa = sW[lr][2 * kk], qb = sW[lr][2 * kk + 1];
                const float2 s  = sWs[lr][kk];
                const int w8[8] = { (int) qa.x, (int) qa.y, (int) qa.z, (int) qa.w,
                                    (int) qb.x, (int) qb.y, (int) qb.z, (int) qb.w };
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int idot = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        idot = ggml_cuda_dp4a(w8[k],     xv[j][k],     idot);
                        idot = ggml_cuda_dp4a(w8[k + 4], xv[j][k + 4], idot);
                    }
                    acc[i][j] += s.x * dx[j] * (float) idot - s.y * sx[j];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t row = row0 + rg + 16 * i;
        if (row >= ne1) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t a = a_base + tg + 4 * j;
            if (a < a_end) {
                y[(size_t) ids_dst[a] * dst_s1 + row] = acc[i][j];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_K MMQ - Q4_K's tiles; the qh plane is folded into int8 at LDS staging, as in the Q5_1 kernel.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? MMQ_RP_OCC_ID : 2) mmq_gemm_q5k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4);
    const uint32_t * ddp = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    // int8 weights per (row, sub-block), 5th bit folded in at staging (see the Q5_1 kernel)
    __shared__ __align__(16) int sW8[MMQ_RP_BM][MMQ_RP_BK * 8 + 4]; // 144 B rows: 2-way instead of 16-way bank conflicts (scale rows stay 16 B-aligned, unpadded)
    __shared__ float2     sWs [MMQ_RP_BM][MMQ_RP_BK + 1];
    __shared__ mmq_x8 sX  [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const uint4    q  = nib[(size_t) wrow * nsp + sb];
            const uint32_t qh = qhp[(size_t) wrow * nsp + sb];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                sW8[lr][lk * 8 + j]     = (int) (( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu));
                sW8[lr][lk * 8 + 4 + j] = (int) (((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu));
            }
            const uint16_t sm = smp[(size_t) wrow * nsp + sb];
            const uint32_t dd = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            sWs[lr][lk] = make_float2(
                __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu),
                __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                sW8[lr][lk * 8 + j] = 0;
            }
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    mmq_x8_stage(sX[lr][lk], xq[xoff + sb]);
                } else {
                    mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
                }
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
                const float sx = xb->s;
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = &sW8[ty + r * 16][kk * 8];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    }
                    const float2 s = sWs[ty + r * 16][kk];
                    acc[r][n] += s.x * dx * (float) idot - s.y * sx;
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
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q5_1 MMQ. The 5th bit is folded into int8 once while staging the weight tile in LDS: doing it
// per dp4a in the inner loop repeats the unpack for every token column and made the kernel
// ALU-bound, 1.5x slower than the canonical MMQ. x = d*q + m, so the min adds.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? MMQ_RP_OCC_ID : 2) mmq_gemm_q5_1_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint32_t * qhp = reinterpret_cast<const uint32_t *>(wbase + (size_t) ne1 * nsp * 16);
    const half2    * dmp = reinterpret_cast<const half2 *>(wbase + (size_t) ne1 * nsp * 20);

    // int8 weights per (row, sub-block): [j] = weights 4j..4j+3, [4+j] = weights 16+4j..16+4j+3
    __shared__ __align__(16) int sW8[MMQ_RP_BM][MMQ_RP_BK * 8 + 4]; // 144 B rows: 2-way instead of 16-way bank conflicts
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK + 1];
    __shared__ mmq_x8 sX [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const size_t   idx = (size_t) wrow * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qh  = qhp[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                sW8[lr][lk * 8 + j]     = (int) (( qa[j]       & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j))     & 0xFu));
                sW8[lr][lk * 8 + 4 + j] = (int) (((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread4((qh >> (8 * j + 4)) & 0xFu));
            }
            sWs[lr][lk] = __half22float2(dmp[idx]);
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                sW8[lr][lk * 8 + j] = 0;
            }
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    mmq_x8_stage(sX[lr][lk], xq[xoff + sb]);
                } else {
                    mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
                }
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
                const float sx = xb->s;
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = &sW8[ty + r * 16][kk * 8];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    }
                    const float2 dm = sWs[ty + r * 16][kk];
                    acc[r][n] += dm.x * dx * (float) idot + dm.y * sx;
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
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// MMQ on the nibble + fp16 scale planes (Q4_0 and the IQ4 family): the Q5_1 tile with the weights
// decoded to signed int8 once at LDS staging (sub8 or the codebook) and one float scale per
// (row, sub-block). Dense only.
template <int MODE, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_nib_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    const uint32_t tok0 = blockIdx.y * (16 * TN_);

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);

    __shared__ __align__(16) int sW8[MMQ_RP_BM][MMQ_RP_BK * 8 + 4]; // 144 B rows: 2-way instead of 16-way bank conflicts
    __shared__ float      sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ mmq_x8 sX [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};
    const int lr = t >> 2;
    const int lk = t & 3;
    const bool xstage = lr < (16 * TN_);

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const size_t   idx = (size_t) wrow * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                repack_nib_decode<MODE>(qa[j], sW8[lr][lk * 8 + j], sW8[lr][lk * 8 + 4 + j]);
            }
            const uint16_t db = dp[idx];
            sWs[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&db));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                sW8[lr][lk * 8 + j] = 0;
            }
            sWs[lr][lk] = 0.0f;
        }
        if (xstage) {
            if (tok0 + lr < n_tok && sb < n_sub) {
                mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = &sW8[ty + r * 16][kk * 8];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) {
                        idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    }
                    acc[r][n] += sWs[ty + r * 16][kk] * dx * (float) idot;
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
        for (int n = 0; n < TN_; n++) {
            const uint32_t tok = tok0 + tx + n * 16;
            if (tok < n_tok) {
                y[(size_t) tok * dst_s1 + row] = acc[r][n];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q6_K MMQ — weights expanded to int8 (q6 - 32) at LDS staging (two uint4 per
// sub-block, in the activation's dp4a group order), so the inner loop is a plain
// int8 dot like Q8_0's. Unpacking the 6-bit planes there instead cost
// TM*TN*8 unpacks per tile step per thread plus the activation half-sums for
// the -32 fold (2x slower than reinstinct's staged form on Gemma E4B prefill).
// The integer dots equal the old (dot - 32*sum), so results are unchanged.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q6k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint2    * h2p = reinterpret_cast<const uint2 *>(
        wbase + (size_t) ne1 * nsp * 16);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 16 + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 2);

    __shared__ uint4      sWlo[MMQ_RP_BM][MMQ_RP_BK + 1]; // int8 (q6 - 32), weights 0-15
    __shared__ uint4      sWhi[MMQ_RP_BM][MMQ_RP_BK + 1]; // int8 (q6 - 32), weights 16-31
    __shared__ float2     sWs [MMQ_RP_BM][MMQ_RP_BK + 1];
    __shared__ mmq_x8 sX  [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const uint4 q  = nib[(size_t) wrow * nsp + sb];
            const uint2 h2 = h2p[(size_t) wrow * nsp + sb];
            // nibble word j: low nibbles = dp4a group 2j (activation int j), high = group 2j+1
            // (activation int j+4); h2 byte g holds group g's four 2-bit high fields
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            uint32_t lo[4], hi[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2.x : h2.y) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2.x : h2.y) >> (8 * (go & 3))) & 0xFFu;
                lo[j] = repack_sub32(( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he));
                hi[j] = repack_sub32(((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho));
            }
            sWlo[lr][lk] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
            sWhi[lr][lk] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
            const uint16_t sm     = smp[(size_t) wrow * nsp + sb];
            const uint16_t d_bits = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                      d * (float)(int)(int8_t)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    mmq_x8_stage(sX[lr][lk], xq[xoff + sb]);
                } else {
                    mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
                }
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint4 wlo[MMQ_RP_TM], whi[MMQ_RP_TM];
            float dlo[MMQ_RP_TM], dhi[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wlo[r] = sWlo[ty + r * 16][kk];
                whi[r] = sWhi[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int lo[4] = { (int) wlo[r].x, (int) wlo[r].y, (int) wlo[r].z, (int) wlo[r].w };
                    const int hi[4] = { (int) whi[r].x, (int) whi[r].y, (int) whi[r].z, (int) whi[r].w };
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot0 = ggml_cuda_dp4a(lo[j], xq32[j],     idot0);
                        idot1 = ggml_cuda_dp4a(hi[j], xq32[j + 4], idot1);
                    }
                    acc[r][n] += dlo[r] * dx * (float) idot0
                               + dhi[r] * dx * (float) idot1;
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
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q3_K MMQ — Q6_K's tiles but the quant is 2-bit lo + 1-bit hi (no 4-bit
// nibble plane); reconstruct q3 = lo2 | (hbit << 2). Symmetric, bias 4.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_q3k_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const uint2    * lo2p = reinterpret_cast<const uint2 *>(wbase);
    const uint32_t * hi1p = reinterpret_cast<const uint32_t *>(
        wbase + (size_t) ne1 * nsp * 8);
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4);
    const uint16_t * ddp = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 8 + (size_t) ne1 * nsp * 4 + (size_t) ne1 * nsp * 2);

    __shared__ uint2      sWl[MMQ_RP_BM][MMQ_RP_BK + 1];
    __shared__ uint32_t   sWh[MMQ_RP_BM][MMQ_RP_BK + 1];
    __shared__ float2     sWs[MMQ_RP_BM][MMQ_RP_BK + 1];
    __shared__ mmq_x8 sX [(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // activation row for this thread's sX slot, resolved once
    const bool xstage = lr < (16 * TN_);
    bool xval;
    uint32_t xoff; // block offset of the row in xq (32-bit: one VGPR)
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        xval = xstage && a < a_end;
        xoff = (xval ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        xval = xstage;
        xoff = 0; // dense: resolved per iteration (hoisting it costs VGPRs)
    }

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sWl[lr][lk] = lo2p[(size_t) wrow * nsp + sb];
            sWh[lr][lk] = hi1p[(size_t) wrow * nsp + sb];
            const uint16_t sm     = smp[(size_t) wrow * nsp + sb];
            const uint16_t d_bits = ddp[(size_t) wrow * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                      d * (float)(int)(int8_t)(sm >> 8));
        } else {
            sWs[lr][lk] = make_float2(0.0f, 0.0f);
        }
        if (xstage) {
            if constexpr (!HAS_IDS) {
                xval = tok0 + lr < n_tok;
            }
            if (xval && sb < n_sub) {
                if constexpr (HAS_IDS) {
                    mmq_x8_stage(sX[lr][lk], xq[xoff + sb]);
                } else {
                    mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
                }
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            uint2 wl[MMQ_RP_TM]; uint32_t wh[MMQ_RP_TM];
            float dlo[MMQ_RP_TM], dhi[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wl[r] = sWl[ty + r * 16][kk];
                wh[r] = sWh[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
                int xis0 = 0, xis1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = ggml_cuda_dp4a(xq32[j],     0x01010101, xis0);
                    xis1 = ggml_cuda_dp4a(xq32[j + 4], 0x01010101, xis1);
                }
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo2lo = wl[r].x, lo2hi = wl[r].y, qh = wh[r];
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t ge = 2 * j;
                        const uint32_t go = 2 * j + 1;
                        const uint32_t lb = ((ge < 4 ? lo2lo : lo2hi) >> (8 * (ge & 3))) & 0xFFu;
                        const uint32_t hb = ((go < 4 ? lo2lo : lo2hi) >> (8 * (go & 3))) & 0xFFu;
                        const uint32_t q3lo = repack_spread2_lo(lb) | repack_spread1_hi((qh >> (8 * j))     & 0xFu);
                        const uint32_t q3hi = repack_spread2_lo(hb) | repack_spread1_hi((qh >> (8 * j + 4)) & 0xFu);
                        idot0 = ggml_cuda_dp4a((int) q3lo, xq32[j],     idot0);
                        idot1 = ggml_cuda_dp4a((int) q3hi, xq32[j + 4], idot1);
                    }
                    acc[r][n] += dlo[r] * dx * (float)(idot0 - 4 * xis0)
                               + dhi[r] * dx * (float)(idot1 - 4 * xis1);
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
        for (int n = 0; n < TN_; n++) {
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + tx + n * 16;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Q8_0 MMQ — 32 qs bytes per sub-block staged as two uint4s; no offset
// term, so the accumulate is just dsc * dx * idot.
// The dense instantiation runs 3 blocks/CU (84 VGPR cap, 3 x 20 KB LDS):
// the next W step is prefetched into registers during the compute, and
// the kk step is split into 16-byte halves to cut live operands.
template <bool HAS_IDS, int TN_>
static __global__ void __launch_bounds__(256, HAS_IDS ? 2 : 3) mmq_gemm_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const int32_t * __restrict__ tile_expert,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    uint32_t tok0 = blockIdx.y * (16 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = (uint32_t) tile_expert[blockIdx.y];
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * qsp = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    // One raw LDS buffer with typed views. Rows of the qs planes are
    // 2*BK+1 uint4 (144 B) so b128 reads with tx-distinct rows are
    // conflict-free; slot 2*kk is qs[0..15], 2*kk+1 is qs[16..31].
    // After the K loop the buffer is reused as the output tile.
    constexpr int QS_LD  = 2 * MMQ_RP_BK + 1;
    constexpr int XR     = 16 * TN_;
    constexpr int Y_LD   = MMQ_RP_BM + 2; // 2tx+ty banks: conflict-free per half-wave
    constexpr int OFF_XQ = MMQ_RP_BM * QS_LD * 16;
    constexpr int OFF_WD = OFF_XQ + XR * QS_LD * 16;
    constexpr int OFF_XD = OFF_WD + MMQ_RP_BM * MMQ_RP_BK * 4;
    constexpr int SZ_K   = OFF_XD + XR * MMQ_RP_BK * 4;
    constexpr int SZ_Y   = HAS_IDS ? 0 : XR * Y_LD * 4;
    constexpr int SZ     = SZ_K > SZ_Y ? SZ_K : SZ_Y;
    __shared__ uint4 smem[SZ / 16];
    uint4 (*sW )[QS_LD]        = reinterpret_cast<uint4 (*)[QS_LD]>(smem);
    uint4 (*sXq)[QS_LD]        = reinterpret_cast<uint4 (*)[QS_LD]>((char *) smem + OFF_XQ);
    float (*sWd)[MMQ_RP_BK]    = reinterpret_cast<float (*)[MMQ_RP_BK]>((char *) smem + OFF_WD);
    float (*sXd)[MMQ_RP_BK]    = reinterpret_cast<float (*)[MMQ_RP_BK]>((char *) smem + OFF_XD);

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    // Staging slot of this thread: weight row lr and activation row lr,
    // sub-block lk of each BK step. Out-of-range rows and sub-blocks are
    // clamped to valid addresses so the loads issue unmasked; no zeroing
    // is needed: a clamped row only feeds outputs that are never stored,
    // and a clamped sub-block is skipped by the K-tail check. Selecting
    // on the loaded values would force an early vmcnt wait.
    const bool     xstage = XR >= 64 || lr < XR; // lr < 64 always
    // 32-bit element offsets against uniform bases keep one VGPR each
    const uint32_t w_off = min(row0 + lr, ne1 - 1) * nsp;
    uint32_t x_off;
    if constexpr (HAS_IDS) {
        const uint32_t a = a_base + lr;
        x_off = (xstage && a < a_end ? (uint32_t) ids_src1[a] : 0u) * x_stride;
    } else {
        x_off = min(tok0 + lr, n_tok - 1) * x_stride;
    }

    // W of the next BK step is prefetched into registers; X is loaded at
    // the top of its own step (prefetching it too spills at 84 VGPRs).
    // Native vectors: HIP uint4 locals here would stay in scratch.
    typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
    u32x4    pw_lo, pw_hi, px_a, px_b;
    uint32_t pd, px_c;
    auto gload_w = [&](const uint32_t sb0) {
        const uint32_t wi = w_off + min(sb0 + lk, n_sub - 1);
        pw_lo = reinterpret_cast<const u32x4 *>(qsp)[wi * 2];
        pw_hi = reinterpret_cast<const u32x4 *>(qsp)[wi * 2 + 1];
        pd    = dp[wi];
    };
    // X kept as loaded (d, qs[0..31] = 9 dwords), shuffled at the store
    auto gload_x = [&](const uint32_t sb0) {
        if (xstage) {
            const uint32_t * xi = reinterpret_cast<const uint32_t *>(xq + (x_off + min(sb0 + lk, n_sub - 1)));
            px_a = u32x4{xi[0], xi[1], xi[2], xi[3]};
            px_b = u32x4{xi[4], xi[5], xi[6], xi[7]};
            px_c = xi[8];
        }
    };
    auto lstore = [&]() {
        reinterpret_cast<u32x4 *>(sW[lr])[2 * lk]     = pw_lo;
        reinterpret_cast<u32x4 *>(sW[lr])[2 * lk + 1] = pw_hi;
        const uint16_t d_bits = (uint16_t) pd;
        sWd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&d_bits));
        if (xstage) {
            reinterpret_cast<u32x4 *>(sXq[lr])[2 * lk]     = u32x4{px_a.y, px_a.z, px_a.w, px_b.x};
            reinterpret_cast<u32x4 *>(sXq[lr])[2 * lk + 1] = u32x4{px_b.y, px_b.z, px_b.w, px_c};
            const uint16_t xd_bits = (uint16_t) px_a.x;
            sXd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&xd_bits));
        }
    };

    gload_w(0);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        gload_x(sb0);
        __syncthreads();
        lstore();
        __syncthreads();
        // unconditional: the last step reloads a clamped block, but a
        // branch here makes the compiler wait on the loads at the join
        gload_w(sb0 + MMQ_RP_BK);

        // K tail: skipped terms would add exactly +0.0f (scale 0)
        const int kk_end = min((int) MMQ_RP_BK, (int) (n_sub - sb0));
        if constexpr (!HAS_IDS) {
            // lo then hi 16-byte half; integer sums are order-free. The
            // sched barriers stop the halves' LDS reads from being hoisted
            // together, which spills at the 84 VGPR cap.
#pragma unroll 1
            for (int kk = 0; kk < kk_end; kk++) {
                int idot[MMQ_RP_TM][TN_];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    u32x4 wq[MMQ_RP_TM];
#pragma unroll
                    for (int r = 0; r < MMQ_RP_TM; r++) {
                        wq[r] = reinterpret_cast<const u32x4 *>(sW[ty + r * 16])[2 * kk + h];
                    }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int n = 0; n < TN_; n++) {
                        const u32x4 xv = reinterpret_cast<const u32x4 *>(sXq[tx + n * 16])[2 * kk + h];
#pragma unroll
                        for (int r = 0; r < MMQ_RP_TM; r++) {
                            int d = h ? idot[r][n] : 0;
                            d = ggml_cuda_dp4a((int) wq[r].x, (int) xv.x, d);
                            d = ggml_cuda_dp4a((int) wq[r].y, (int) xv.y, d);
                            d = ggml_cuda_dp4a((int) wq[r].z, (int) xv.z, d);
                            d = ggml_cuda_dp4a((int) wq[r].w, (int) xv.w, d);
                            idot[r][n] = d;
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
#pragma unroll
                for (int n = 0; n < TN_; n++) {
                    const float dx = sXd[tx + n * 16][kk];
#pragma unroll
                    for (int r = 0; r < MMQ_RP_TM; r++) {
                        acc[r][n] += sWd[ty + r * 16][kk] * dx * (float) idot[r][n];
                    }
                }
            }
            continue;
        }
#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            if (kk >= kk_end) {
                break;
            }
            uint4 wq_lo[MMQ_RP_TM], wq_hi[MMQ_RP_TM];
            float dsc[MMQ_RP_TM];
#pragma unroll
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq_lo[r] = sW [ty + r * 16][2 * kk];
                wq_hi[r] = sW [ty + r * 16][2 * kk + 1];
                dsc[r]   = sWd[ty + r * 16][kk];
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const uint4 xa = sXq[tx + n * 16][2 * kk];
                const uint4 xb = sXq[tx + n * 16][2 * kk + 1];
                const float dx = sXd[tx + n * 16][kk];
                const int xq32[8] = { (int) xa.x, (int) xa.y, (int) xa.z, (int) xa.w,
                                      (int) xb.x, (int) xb.y, (int) xb.z, (int) xb.w };
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo[4] = { wq_lo[r].x, wq_lo[r].y, wq_lo[r].z, wq_lo[r].w };
                    const uint32_t hi[4] = { wq_hi[r].x, wq_hi[r].y, wq_hi[r].z, wq_hi[r].w };
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int) lo[j], xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int) hi[j], xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float) idot;
                }
            }
        }
    }
    __syncthreads();

    if constexpr (HAS_IDS) {
#pragma unroll
        for (int r = 0; r < MMQ_RP_TM; r++) {
            const uint32_t row = row0 + ty + r * 16;
            if (row >= ne1) {
                continue;
            }
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const uint32_t a = a_base + tx + n * 16;
                if (a < a_end) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            }
        }
    } else {
        // transpose through LDS so each token row is stored contiguously
        float (*tileY)[Y_LD] = reinterpret_cast<float (*)[Y_LD]>(smem);
#pragma unroll
        for (int r = 0; r < MMQ_RP_TM; r++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                tileY[tx + n * 16][ty + r * 16] = acc[r][n];
            }
        }
        __syncthreads();
#pragma unroll
        for (int idx = t; idx < MMQ_RP_BM * XR; idx += 256) {
            const int tok = idx / MMQ_RP_BM;
            const int row = idx % MMQ_RP_BM;
            if (tok0 + tok < n_tok && row0 + row < ne1) {
                y[(size_t) (tok0 + tok) * dst_s1 + row0 + row] = tileY[tok][row];
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_expert, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// ---------------------------------------------------------------------
// MUL_MAT dispatch
// ---------------------------------------------------------------------

// Few long rows (qwen4exp hc down, 320 x 10240) with NC columns: a 256-thread workgroup per row, K split
// over the threads as in mul_mat_vec_q8_0_repacked_splitk
template <int NC>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_splitk_nc(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);

    const uint32_t row = blockIdx.x;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
    for (uint32_t hb = threadIdx.x; hb < n_blocks * 2; hb += 256) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const int4 w = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        const float    dw = __half2float(*reinterpret_cast<const __half *>(&db));
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
            const int4 a = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
            int idot = 0;
            idot = ggml_cuda_dp4a(w.x, a.x, idot); idot = ggml_cuda_dp4a(w.y, a.y, idot);
            idot = ggml_cuda_dp4a(w.z, a.z, idot); idot = ggml_cuda_dp4a(w.w, a.w, idot);
            acc[c] += dw * __low2float(xb->ds) * (float) idot;
        }
    }
    __shared__ float part[NC][4];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_reduce_sum<64>(acc[c]);
        if ((threadIdx.x & 63) == 0) {
            part[c][threadIdx.x >> 6] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < NC) {
        const int c = threadIdx.x;
        y[(size_t) c * ne1 + row] = part[c][0] + part[c][1] + part[c][2] + part[c][3];
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Short rows (up to 32 sub-blocks: hc up K = 320, shared-expert down K = 640) with NC columns: thread
// t of a workgroup takes whole sub-block t % nb of row t / nb, so no lane idles on a short row; the
// per-row sums go through LDS
template <int NC>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_flat_nc(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t nb  = ne0 >> 5;
    const uint32_t nsp = ((nb & (nb - 1u)) == 0u) ? (nb + 1u) : nb;
    const int4     * qs4     = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);
    const uint32_t R    = 256 / nb;
    const uint32_t t    = threadIdx.x;
    const uint32_t row0 = blockIdx.x * R;
    __shared__ float part[NC][256];

    if (t < R * nb) {
        const uint32_t row = row0 + t / nb;
        const uint32_t sb  = t % nb;
        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] = 0.0f;
        }
        if (row < ne1) {
            const size_t u = (size_t) row * nsp + sb;
            const int4 w0 = qs4[u * 2 + 0];
            const int4 w1 = qs4[u * 2 + 1];
            const uint16_t db = d_plane[u];
            const float    dw = __half2float(*reinterpret_cast<const __half *>(&db));
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
                const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
                const int4 a0 = x4[0];
                const int4 a1 = x4[1];
                int idot = 0;
                idot = ggml_cuda_dp4a(w0.x, a0.x, idot); idot = ggml_cuda_dp4a(w0.y, a0.y, idot);
                idot = ggml_cuda_dp4a(w0.z, a0.z, idot); idot = ggml_cuda_dp4a(w0.w, a0.w, idot);
                idot = ggml_cuda_dp4a(w1.x, a1.x, idot); idot = ggml_cuda_dp4a(w1.y, a1.y, idot);
                idot = ggml_cuda_dp4a(w1.z, a1.z, idot); idot = ggml_cuda_dp4a(w1.w, a1.w, idot);
                acc[c] = dw * __low2float(xb->ds) * (float) idot;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            part[c][t] = acc[c];
        }
    }
    __syncthreads();
    for (uint32_t i = t; i < R * NC; i += blockDim.x) {
        const uint32_t r = i % R;
        const uint32_t c = i / R;
        if (row0 + r < ne1) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < nb; j++) {
                sum += part[c][r * nb + j];
            }
            y[(size_t) c * ne1 + row0 + r] = sum;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Dense Q8_0 with a few activation columns (speculative verify, 2-8 tokens): one wave per row, each
// weight half-sub-block is loaded once and dotted with every column. The tiled MMQ GEMM costs
// ~135 us per call at N = 2 against ~30 us here.
// ITERS > 0: compile-time trip count (n_half <= ITERS*64); every weight load of a lane is
// issued before the dot products, as in mul_mat_vec_q8_0_repacked_rowu
template <int NC, int ITERS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_nc(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);
    const int lane = threadIdx.x % 64;
    const uint32_t row = blockIdx.x * 4 + threadIdx.x / 64;
    if (row >= ne1) {
        return;
    }

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
    const uint32_t n_half = n_blocks * 2;
    if constexpr (ITERS > 0) {
        int4     w[ITERS];
        uint16_t db[ITERS];
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            const uint32_t hb = lane + j * 64;
            const uint32_t sb = (hb < n_half ? hb : 0) >> 1, half = hb & 1;
            w[j]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
            db[j] = d_plane[(size_t) row * nsp + sb];
        }
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            const uint32_t hb = lane + j * 64;
            if (hb >= n_half) {
                break;
            }
            const uint32_t sb = hb >> 1, half = hb & 1;
            const float    dw = __half2float(*reinterpret_cast<const __half *>(&db[j]));
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
                const int4 a = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
                int idot = 0;
                idot = ggml_cuda_dp4a(w[j].x, a.x, idot); idot = ggml_cuda_dp4a(w[j].y, a.y, idot);
                idot = ggml_cuda_dp4a(w[j].z, a.z, idot); idot = ggml_cuda_dp4a(w[j].w, a.w, idot);
                acc[c] += dw * __low2float(xb->ds) * (float) idot;
            }
        }
    } else
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const int4 w = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        const float    dw = __half2float(*reinterpret_cast<const __half *>(&db));
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
            const int4 a = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
            int idot = 0;
            idot = ggml_cuda_dp4a(w.x, a.x, idot); idot = ggml_cuda_dp4a(w.y, a.y, idot);
            idot = ggml_cuda_dp4a(w.z, a.z, idot); idot = ggml_cuda_dp4a(w.w, a.w, idot);
            acc[c] += dw * __low2float(xb->ds) * (float) idot;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_reduce_sum<64>(acc[c]);
        if (lane == 0) {
            y[(size_t) c * ne1 + row] = v;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// The grouped launch (q8_multi_args) with several activation columns: mul_mat_vec_q8_0_repacked_nc per row,
// the matrix chosen per wave
template <int NC, int ITERS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_multi_nc(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    // the rows of the grouped matrices sit back to back; each wave resolves its own matrix
    const uint32_t grow = blockIdx.x * 4 + threadIdx.x / 64;
    const int      t    = grow >= args.start[2] ? 2 : (grow >= args.start[1] ? 1 : 0);
    const uint32_t row  = grow - args.start[t];
    const uint32_t ne1  = args.ne1[t];
    if (row >= ne1) {
        return;
    }
    const uint8_t * __restrict__ wbase = args.w[t];
    float * __restrict__ y = args.y[t];
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 32);
    const int lane = threadIdx.x % 64;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = 0.0f;
    }
    const uint32_t n_half = n_blocks * 2;
    if constexpr (ITERS > 0) {
        int4     w[ITERS];
        uint16_t db[ITERS];
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            const uint32_t hb = lane + j * 64;
            const uint32_t sb = (hb < n_half ? hb : 0) >> 1, half = hb & 1;
            w[j]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
            db[j] = d_plane[(size_t) row * nsp + sb];
        }
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            const uint32_t hb = lane + j * 64;
            if (hb >= n_half) {
                break;
            }
            const uint32_t sb = hb >> 1, half = hb & 1;
            const float    dw = __half2float(*reinterpret_cast<const __half *>(&db[j]));
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
                const int4 a = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
                int idot = 0;
                idot = ggml_cuda_dp4a(w[j].x, a.x, idot); idot = ggml_cuda_dp4a(w[j].y, a.y, idot);
                idot = ggml_cuda_dp4a(w[j].z, a.z, idot); idot = ggml_cuda_dp4a(w[j].w, a.w, idot);
                acc[c] += dw * __low2float(xb->ds) * (float) idot;
            }
        }
    } else
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const int4 w = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
        const uint16_t db = d_plane[(size_t) row * nsp + sb];
        const float    dw = __half2float(*reinterpret_cast<const __half *>(&db));
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
            const int4 a = *reinterpret_cast<const int4 *>(reinterpret_cast<const int *>(xb->qs) + half * 4);
            int idot = 0;
            idot = ggml_cuda_dp4a(w.x, a.x, idot); idot = ggml_cuda_dp4a(w.y, a.y, idot);
            idot = ggml_cuda_dp4a(w.z, a.z, idot); idot = ggml_cuda_dp4a(w.w, a.w, idot);
            acc[c] += dw * __low2float(xb->ds) * (float) idot;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_reduce_sum<64>(acc[c]);
        if (lane == 0) {
            y[(size_t) c * ne1 + row] = v;
        }
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// Dense Q8_0 with NC = 2..8 activation columns, LDS-staged (from reinstinct's small-batch matvec,
// crossport R2): the activation chunk of 32 sub-blocks is staged in LDS once per 256-thread block and
// each lane holds ROWS rows' weight halves, so an activation half is read from L2 once per 4*ROWS
// rows instead of once per row (the _nc kernels above re-read it NC times per weight half through
// L1). Rows of the grouped matrices sit back to back (q8_multi_args); a wave's ROWS rows never
// straddle a matrix (the launcher requires ne1 % ROWS == 0). Per (row, column) the lane sums run in
// the same order as mul_mat_vec_q8_0_repacked_nc, so results are identical.
template <int NC, int ROWS>
static __global__ void __launch_bounds__(256) mul_mat_vec_q8_0_repacked_lds_nc(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    __shared__ int4  sxq[NC][32][2];
    __shared__ float sxd[NC][32];
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const uint32_t grow0 = blockIdx.x * (4 * ROWS) + wave * ROWS;
    const int      t     = grow0 >= args.start[2] ? 2 : (grow0 >= args.start[1] ? 1 : 0);
    const uint32_t row0  = grow0 - args.start[t];
    const uint32_t ne1   = args.ne1[t];
    const bool wave_ok   = row0 < ne1; // an idle wave still joins the barriers
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      * qs_int  = reinterpret_cast<const int *>(args.w[t]);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(args.w[t] + (size_t) ne1 * nsp * 32);
    const uint32_t n_half = n_blocks * 2;

    float acc[ROWS][NC];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    for (uint32_t h0 = 0; h0 < n_half; h0 += 64) {
        const uint32_t sb0 = h0 >> 1;
        __syncthreads();
        for (int e = threadIdx.x; e < NC * 32; e += 256) {
            const int c = e / 32, l = e % 32;
            const uint32_t sb = sb0 + l;
            if (sb < n_blocks) {
                const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
                const int4 * q = reinterpret_cast<const int4 *>(xb->qs);
                sxq[c][l][0] = q[0];
                sxq[c][l][1] = q[1];
                sxd[c][l]    = __low2float(xb->ds);
            }
        }
        __syncthreads();
        const uint32_t hb = h0 + lane;
        if (wave_ok && hb < n_half) {
            const uint32_t sb = hb >> 1, half = hb & 1;
            const int l = lane >> 1;
            int4  w[ROWS];
            float dw[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint32_t row = row0 + r;
                w[r] = *reinterpret_cast<const int4 *>(qs_int + ((size_t) row * nsp + sb) * 8 + half * 4);
                const uint16_t db = d_plane[(size_t) row * nsp + sb];
                dw[r] = __half2float(*reinterpret_cast<const __half *>(&db));
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int4  a  = sxq[c][l][half];
                const float dx = sxd[c][l];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    int idot = 0;
                    idot = ggml_cuda_dp4a(w[r].x, a.x, idot); idot = ggml_cuda_dp4a(w[r].y, a.y, idot);
                    idot = ggml_cuda_dp4a(w[r].z, a.z, idot); idot = ggml_cuda_dp4a(w[r].w, a.w, idot);
                    acc[r][c] += dw[r] * dx * (float) idot;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float v = warp_reduce_sum<64>(acc[r][c]);
            if (wave_ok && lane == 0) {
                args.y[t][(size_t) c * ne1 + row0 + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <int NC>
static void launch_q8_lds_nc(const q8_multi_args & args, const uint32_t rows, const block_q8_1 * xq,
        const uint32_t ne0, const uint32_t x_stride, cudaStream_t stream) {
    const dim3 grid((rows + 15) / 16);
    mul_mat_vec_q8_0_repacked_lds_nc<NC, 4><<<grid, 256, 0, stream>>>(args, xq, ne0, x_stride);
}

static bool q8_lds_nc_dispatch(const q8_multi_args & args, const uint32_t rows, const block_q8_1 * xq,
        const int64_t ne11, const uint32_t ne0, const uint32_t x_stride, cudaStream_t stream) {
    static const bool disabled = getenv("GGML_CUDA_NO_Q8_LDS_NC") != nullptr;
    // at 2-3 columns the per-row _nc kernels are already near weight-once (and have 4x the blocks);
    // the staged kernel wins from 4 columns up (Flash-Next pp6: 13.5 -> 10.7 ms of Q8_0 per step)
    if (disabled || ne11 < 4 || ne11 > 8) {
        return false;
    }
    for (int i = 0; i < 3; i++) {
        if (args.start[i] != UINT32_MAX && args.ne1[i] % 4 != 0) {
            return false;
        }
    }
    switch (ne11) {
        case 2: launch_q8_lds_nc<2>(args, rows, xq, ne0, x_stride, stream); break;
        case 3: launch_q8_lds_nc<3>(args, rows, xq, ne0, x_stride, stream); break;
        case 4: launch_q8_lds_nc<4>(args, rows, xq, ne0, x_stride, stream); break;
        case 5: launch_q8_lds_nc<5>(args, rows, xq, ne0, x_stride, stream); break;
        case 6: launch_q8_lds_nc<6>(args, rows, xq, ne0, x_stride, stream); break;
        case 7: launch_q8_lds_nc<7>(args, rows, xq, ne0, x_stride, stream); break;
        default: launch_q8_lds_nc<8>(args, rows, xq, ne0, x_stride, stream); break;
    }
    return true;
}

// Dense Q8_0 with NC = 2..4 columns (MTP verify): the whole activation sits in LDS, loaded once per block,
// and the blocks loop over the rows. A wave issues all weight loads of its row before the dots; PF also
// keeps the next row's loads in flight during the dots. The _nc kernels read the activation through L1,
// where the weight stream evicts it, so each column cost one more L2 read of it per row. Per (row, column)
// the sums run in the order of mul_mat_vec_q8_0_repacked_nc, so the results are the same.
template <int NC, int ITERS, int WAVES, bool PF>
static __global__ void __launch_bounds__(64 * WAVES) mul_mat_vec_q8_0_repacked_lds_rows(
        const q8_multi_args args, const block_q8_1 * __restrict__ xq, const uint32_t ne0, const uint32_t x_stride,
        const uint32_t rows) {
#if defined(GGML_USE_HIP) && defined(GCN)
    extern __shared__ int4 lds_rows_smem[];
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t n_half   = n_blocks * 2;
    int4  * sx = lds_rows_smem;                                     // [NC][n_half]
    float * sd = reinterpret_cast<float *>(lds_rows_smem + NC * n_half); // [NC][n_blocks]
    for (uint32_t e = threadIdx.x; e < NC * n_half; e += 64 * WAVES) {
        const uint32_t c = e / n_half, hb = e % n_half;
        const block_q8_1 * xb = xq + (size_t) c * x_stride + (hb >> 1);
        sx[e] = reinterpret_cast<const int4 *>(xb->qs)[hb & 1];
        if ((hb & 1) == 0) {
            sd[c * n_blocks + (hb >> 1)] = __low2float(xb->ds);
        }
    }
    __syncthreads();
    const uint32_t nsp  = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;
    const int      lane = threadIdx.x % 64;
    const uint32_t step = gridDim.x * WAVES;

    int4     w[ITERS];
    uint16_t db[ITERS];
    auto load = [&](const uint32_t grow) {
        const uint32_t g = grow < rows ? grow : 0;
        const int      t = g >= args.start[2] ? 2 : (g >= args.start[1] ? 1 : 0);
        const uint32_t r = g - args.start[t];
        const int      * qs_int  = reinterpret_cast<const int *>(args.w[t]);
        const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(args.w[t] + (size_t) args.ne1[t] * nsp * 32);
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            const uint32_t hb = lane + j * 64;
            const uint32_t sb = (hb < n_half ? hb : 0) >> 1, half = hb & 1;
            w[j]  = *reinterpret_cast<const int4 *>(qs_int + ((size_t) r * nsp + sb) * 8 + half * 4);
            db[j] = d_plane[(size_t) r * nsp + sb];
        }
    };
    uint32_t grow = blockIdx.x * WAVES + threadIdx.x / 64;
    if constexpr (PF) {
        load(grow);
    }
    for (; grow < rows; grow += step) {
        if constexpr (!PF) {
            load(grow);
        }
        int4     wc[ITERS];
        uint16_t dc[ITERS];
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            wc[j] = w[j];
            dc[j] = db[j];
        }
        if constexpr (PF) {
            if (grow + step < rows) {
                load(grow + step);
            }
        }
        float acc[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < ITERS; ++j) {
            const uint32_t hb = lane + j * 64;
            if (hb >= n_half) {
                break;
            }
            const uint32_t sb = hb >> 1;
            const float    dw = __half2float(*reinterpret_cast<const __half *>(&dc[j]));
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const int4 a = sx[c * n_half + hb];
                int idot = 0;
                idot = ggml_cuda_dp4a(wc[j].x, a.x, idot); idot = ggml_cuda_dp4a(wc[j].y, a.y, idot);
                idot = ggml_cuda_dp4a(wc[j].z, a.z, idot); idot = ggml_cuda_dp4a(wc[j].w, a.w, idot);
                acc[c] += dw * sd[c * n_blocks + sb] * (float) idot;
            }
        }
        const int      t = grow >= args.start[2] ? 2 : (grow >= args.start[1] ? 1 : 0);
        const uint32_t r = grow - args.start[t];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float v = warp_reduce_sum<64>(acc[c]);
            if (lane == 0) {
                args.y[t][(size_t) c * args.ne1[t] + r] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(args, xq, ne0, x_stride, rows);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <int NC, int ITERS>
static void launch_q8_lds_rows(const q8_multi_args & args, const uint32_t rows, const block_q8_1 * xq,
        const uint32_t ne0, const uint32_t x_stride, const size_t lds, const int nsm, cudaStream_t stream) {
    // tuned on 4x MI50 (standalone A/B): 4 waves per block up to K = 4096, 8 above (LDS per CU); prefetch
    // at 2..3 columns on 4-wave blocks and at 4 columns on 8-wave blocks (K = 6144: 48 vs 54 us, although
    // it leaves 2 waves per SIMD)
    constexpr int  WAVES = ITERS > 4 ? 8 : 4;
    constexpr bool PF    = WAVES == 4 ? NC <= 3 : NC >= 4;
    auto kernel = mul_mat_vec_q8_0_repacked_lds_rows<NC, ITERS, WAVES, PF>;
    // whole rounds of resident blocks (registers and LDS limit residency to 1..6 blocks per CU): one round, two from
    // 64K rows (2560 x 10240 at 2-4 columns: 91 -> 72 us with one round; the LM head, 248320 rows: 1140 vs 1351 us
    // with two). GGML_CUDA_Q8_LDS_ROWS_ROUNDS=n forces n
    static const int rounds_env = getenv("GGML_CUDA_Q8_LDS_ROWS_ROUNDS") ? atoi(getenv("GGML_CUDA_Q8_LDS_ROWS_ROUNDS")) : 0;
    const int rounds = rounds_env > 0 ? rounds_env : (rows >= 65536 ? 2 : 1);
    static int per_cu[GGML_CUDA_MAX_DEVICES][64] = {};
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    const int lds_kb = (int) std::min<size_t>(63, lds / 1024);
    if (per_cu[device][lds_kb] == 0) {
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, 64 * WAVES, (lds_kb + 1) * 1024));
        per_cu[device][lds_kb] = std::max(n, 1);
    }
    const uint32_t grid = std::min<uint32_t>((rows + WAVES - 1) / WAVES, (uint32_t) (nsm * per_cu[device][lds_kb] * rounds));
    kernel<<<grid, 64 * WAVES, lds, stream>>>(args, xq, ne0, x_stride, rows);
}

static bool q8_lds_rows_dispatch(const q8_multi_args & args, const uint32_t rows, const block_q8_1 * xq,
        const int64_t ne11, const uint32_t ne0, const uint32_t x_stride, cudaStream_t stream) {
    static const bool disabled = getenv("GGML_CUDA_NO_Q8_LDS_ROWS") != nullptr;
    const uint32_t n_blocks = ne0 / 32;
    const int      iters    = (int) ((2 * n_blocks + 63) / 64);
    const size_t   lds      = (size_t) ne11 * n_blocks * (2 * sizeof(int4) + sizeof(float));
    if (disabled || ne11 < 2 || ne11 > 4 || iters < 2 || iters > 6 || lds > 32768 || rows < 2048) {
        return false;
    }
    // 3 columns on 4-wave blocks: the _nc kernel is level or faster below ~6K rows (K = 2048..4096 at 2-4K rows)
    if (ne11 == 3 && iters <= 4 && rows < 6144) {
        return false;
    }
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    const int nsm = ggml_cuda_info().devices[device].nsm;
    auto launch = [&](auto nc) {
        constexpr int NC = decltype(nc)::value;
        switch (iters) {
            case 2:  launch_q8_lds_rows<NC, 2>(args, rows, xq, ne0, x_stride, lds, nsm, stream); break;
            case 3:  launch_q8_lds_rows<NC, 3>(args, rows, xq, ne0, x_stride, lds, nsm, stream); break;
            case 4:  launch_q8_lds_rows<NC, 4>(args, rows, xq, ne0, x_stride, lds, nsm, stream); break;
            case 5:  launch_q8_lds_rows<NC, 5>(args, rows, xq, ne0, x_stride, lds, nsm, stream); break;
            default: launch_q8_lds_rows<NC, 6>(args, rows, xq, ne0, x_stride, lds, nsm, stream); break;
        }
    };
    switch (ne11) {
        case 2:  launch(std::integral_constant<int, 2>{}); break;
        case 3:  launch(std::integral_constant<int, 3>{}); break;
        default: launch(std::integral_constant<int, 4>{}); break;
    }
    return true;
}

// Dense repacked K-quant matvec for NC = 2..8 activation columns (spec-decode verify,
// short prompt chunks). Before this, ne11 >= 2 went to the int8 MMQ tile GEMM, whose
// 64-wide N tile made a 2-token batch cost about as much as a 64-token one (~6x a decode
// step on dense Q4_K/Q5_K/Q6_K models). Each wave decodes its ROWS weight sub-blocks once
// and dots them against every column; per (row, column) the lane sums run in the same
// order as the ne11 == 1 kernels, so each column matches a single-column launch exactly.
template <ggml_type type, int NC>
static __global__ void __launch_bounds__(256) mul_mat_vec_kq_repacked_nc(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1, const uint32_t x_stride) {
#if defined(GGML_USE_HIP) && defined(GCN)
    static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
                  type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S,
                  "unsupported type");
    constexpr bool NIB  = type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S;
    constexpr int  MODE = type == GGML_TYPE_Q4_0 ? REPACK_NIB_Q4_0 : type == GGML_TYPE_IQ3_S ? REPACK_NIB_IQ3S : REPACK_NIB_IQ4NL;
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub   = ne0 >> 5;
    const uint32_t nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint32_t n_super = n_sub >> 3;
    const size_t   plane   = (size_t) ne1 * nsp;

    // plane layout (see the per-type repack): nibbles, then
    //   Q4_K: sm u16, dd u32/super      Q5_K: qh u32, sm u16, dd u32/super
    //   Q6_K: h2 2xu32, sm u16, d u16/super
    const uint4 * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint8_t * p1 = wbase + plane * 16;
    const uint16_t * smp = reinterpret_cast<const uint16_t *>(
        type == GGML_TYPE_Q4_K ? p1 : type == GGML_TYPE_Q5_K ? p1 + plane * 4 : p1 + plane * 8);
    const uint8_t * pdd = reinterpret_cast<const uint8_t *>(smp) + plane * 2;

    float acc[ROWS][NC];
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            acc[r][c] = 0.0f;
        }
    }

    for (uint32_t sb = lane; sb < n_sub; sb += 64) {
        int   wl[ROWS][4], wh[ROWS][4];
        float s0[ROWS], s1[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const uint32_t row = (uint32_t) min(row0 + r, (int) ne1 - 1); // clamped; not stored
            const size_t   idx = (size_t) row * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            if constexpr (NIB) {
                // nibble + fp16 scale planes: decode to signed int8 once per row, no min term
                const uint16_t db = reinterpret_cast<const uint16_t *>(p1)[idx];
                s0[r] = __half2float(*reinterpret_cast<const __half *>(&db));
                s1[r] = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    repack_nib_decode<MODE>(qa[j], wl[r][j], wh[r][j]);
                }
                continue;
            }
            const uint16_t sm  = NIB ? 0 : smp[idx];
            if constexpr (type == GGML_TYPE_Q6_K) {
                const uint32_t * h2p = reinterpret_cast<const uint32_t *>(p1);
                const uint32_t h2lo = h2p[idx * 2];
                const uint32_t h2hi = h2p[idx * 2 + 1];
                const uint16_t d_bits = reinterpret_cast<const uint16_t *>(pdd)[(size_t) row * n_super + (sb >> 3)];
                const float d = __half2float(*reinterpret_cast<const __half *>(&d_bits));
                s0[r] = d * (float)(int)(int8_t)(sm & 0xFFu);
                s1[r] = d * (float)(int)(int8_t)(sm >> 8);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t ge = 2 * j, go = 2 * j + 1;
                    const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                    const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                    wl[r][j] = (int)(( qa[j]       & 0x0F0F0F0Fu) | repack_spread2(he));
                    wh[r][j] = (int)(((qa[j] >> 4) & 0x0F0F0F0Fu) | repack_spread2(ho));
                }
            } else {
                const uint32_t dd = reinterpret_cast<const uint32_t *>(pdd)[(size_t) row * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                s0[r] = __half2float(*reinterpret_cast<const __half *>(&d_bits))    * (float)(sm & 0xFFu);
                s1[r] = __half2float(*reinterpret_cast<const __half *>(&dmin_bits)) * (float)(sm >> 8);
                uint32_t qh = 0;
                if constexpr (type == GGML_TYPE_Q5_K) {
                    qh = reinterpret_cast<const uint32_t *>(p1)[idx];
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu);
                    uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu);
                    if constexpr (type == GGML_TYPE_Q5_K) {
                        lo |= repack_spread4((qh >> (8 * j))     & 0xFu);
                        hi |= repack_spread4((qh >> (8 * j + 4)) & 0xFu);
                    }
                    wl[r][j] = (int) lo;
                    wh[r][j] = (int) hi;
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const block_q8_1 * xb = xq + (size_t) c * x_stride + sb;
            const float dx = __low2float(xb->ds);
            const int * xq32 = reinterpret_cast<const int *>(xb->qs);
            int x[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                x[j] = xq32[j];
            }
            if constexpr (type == GGML_TYPE_Q6_K) {
                int xis0 = 0, xis1 = 0;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = ggml_cuda_dp4a(x[j],     0x01010101, xis0);
                    xis1 = ggml_cuda_dp4a(x[j + 4], 0x01010101, xis1);
                }
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    int idot0 = 0, idot1 = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot0 = ggml_cuda_dp4a(wl[r][j], x[j],     idot0);
                        idot1 = ggml_cuda_dp4a(wh[r][j], x[j + 4], idot1);
                    }
                    acc[r][c] += s0[r] * dx * (float)(idot0 - 32 * xis0)
                               + s1[r] * dx * (float)(idot1 - 32 * xis1);
                }
            } else {
                const float sx = __high2float(xb->ds);
#pragma unroll
                for (int r = 0; r < ROWS; r++) {
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a(wl[r][j], x[j],     idot);
                        idot = ggml_cuda_dp4a(wh[r][j], x[j + 4], idot);
                    }
                    acc[r][c] += s0[r] * dx * (float) idot - s1[r] * sx;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float a = warp_reduce_sum<64>(acc[r][c]);
            if (lane == 0 && (row0 + r) < (int) ne1) {
                y[(size_t) c * ne1 + row0 + r] = a;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, x_stride);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

template <ggml_type type>
static void launch_mul_mat_vec_kq_repacked_nc(const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11, const int64_t x_stride, cudaStream_t stream) {
    const dim3 grid((ne01 + 7) / 8, 1, 1);
    // columns in chunks of up to 8: the weights are read once per chunk
    for (int64_t c0 = 0; c0 < ne11; c0 += 8) {
        const int nc = (int) std::min<int64_t>(8, ne11 - c0);
        const block_q8_1 * x = xq + c0 * x_stride;
        float * y = dst_d + c0 * ne01;
        switch (nc) {
            case 1: mul_mat_vec_kq_repacked_nc<type, 1><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            case 2: mul_mat_vec_kq_repacked_nc<type, 2><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            case 3: mul_mat_vec_kq_repacked_nc<type, 3><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            case 4: mul_mat_vec_kq_repacked_nc<type, 4><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            case 5: mul_mat_vec_kq_repacked_nc<type, 5><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            case 6: mul_mat_vec_kq_repacked_nc<type, 6><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            case 7: mul_mat_vec_kq_repacked_nc<type, 7><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
            default: mul_mat_vec_kq_repacked_nc<type, 8><<<grid, 256, 0, stream>>>(w, x, y, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
        }
    }
}

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        int64_t x_stride, cudaStream_t stream);

static const block_q8_1 * repack_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        int64_t ne10_padded, ggml_cuda_pool_alloc<char> & fallback, cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float)); // rows may be strided; dim0 must be dense
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // M
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1]; // N
    const int64_t ne12 = src1->ne[2]; // broadcast slices (2D weight repeats)
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;

    static const bool trace = getenv("GGML_CUDA_REPACK_TRACE") != nullptr;
    if (trace) {
        fprintf(stderr, "repack-mm %s %s ne00=%ld ne01=%ld ne11=%ld ne12=%ld ne13=%ld\n", src0->name,
            ggml_type_name(src0->type), (long) ne00, (long) ne01, (long) ne11, (long) ne12, (long) ne13);
    }

    // Quantize the whole (possibly 3D/4D) activation once; blocks land
    // contiguously as [ne13][ne12][ne11][ne10_padded/QK8_1].
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1; // q8_1 blocks per column
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const block_q8_1 * xq_all = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);

    // the weight is shared by every slice, and xq_all holds the slices as adjacent columns: when dst
    // is contiguous too, run all of them as one matmul (GDN output with several sequences
    // [K, 1, n_seq]; MTP eh_proj [K, n_hc, n_tokens] would otherwise launch once per token)
    static const bool no_flat = getenv("GGML_CUDA_NO_REPACK_FLATTEN") != nullptr;
    if (!no_flat && ne12 * ne13 > 1 &&
        dst->nb[2] == (size_t) ne11 * dst->nb[1] && dst->nb[3] == (size_t) ne12 * dst->nb[2]) {
        ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq_all, (float *) dst->data,
            ne00, ne01, ne11 * ne12 * ne13, x_stride, stream);
        return;
    }

    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = xq_all + (i3 * ne12 + i2) * ne11 * x_stride;
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
            ne00, ne01, ne11, x_stride, stream);
    }
    }
}

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const int64_t x_stride, cudaStream_t stream) {
    static const bool no_nc = getenv("GGML_CUDA_NO_Q8_NC") != nullptr;
    // 9..32 columns on a small matrix (MTP eh_proj flattened over n_hc x tokens: 16-24 columns on
    // 13 MB): the staged kernel over column chunks of <= 8 re-reads the weight per chunk but stays
    // far below the tile GEMM (417 us) or the generic 16-column kernel (303 us) at this size
    static const bool no_chunk = getenv("GGML_CUDA_NO_Q8_NC_CHUNK") != nullptr;
    if (!no_chunk && !no_nc && src0->type == GGML_TYPE_Q8_0 && ne11 > 8 && ne11 <= 32 && ne00 * ne01 <= (int64_t) 64 << 20 &&
            ne01 % 4 == 0 && ne00 / 32 > 32) {
        for (int64_t c0 = 0; c0 < ne11; c0 += 8) {
            const int64_t nc = std::min<int64_t>(8, ne11 - c0);
            if (nc >= 4) {
                q8_multi_args one = {};
                one.w[0] = w; one.y[0] = dst_d + c0 * ne01; one.ne1[0] = (uint32_t) ne01;
                one.start[0] = 0; one.start[1] = UINT32_MAX; one.start[2] = UINT32_MAX;
                if (q8_lds_nc_dispatch(one, (uint32_t) ne01, xq + c0 * x_stride, nc, (uint32_t) ne00, (uint32_t) x_stride, stream)) {
                    continue;
                }
            }
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq + c0 * x_stride, dst_d + c0 * ne01, ne00, ne01, nc, x_stride, stream);
        }
        return;
    }
    if (!no_nc && src0->type == GGML_TYPE_Q8_0 && ne11 >= 2 && ne11 <= 16) {
        const int64_t nb = ne00 / 32;
        auto launch = [&](auto nc) {
            constexpr int NC = decltype(nc)::value;
            if (ne01 <= 1024 && ne00 >= 4096) {
                mul_mat_vec_q8_0_repacked_splitk_nc<NC><<<ne01, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride);
            } else if (nb <= 32) {
                const int64_t R = 256 / nb;
                mul_mat_vec_q8_0_repacked_flat_nc<NC><<<(ne01 + R - 1) / R, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride);
            } else {
                {
                    q8_multi_args one = {};
                    one.w[0] = w; one.y[0] = dst_d; one.ne1[0] = (uint32_t) ne01;
                    one.start[0] = 0; one.start[1] = UINT32_MAX; one.start[2] = UINT32_MAX;
                    if (q8_lds_rows_dispatch(one, (uint32_t) ne01, xq, ne11, (uint32_t) ne00, (uint32_t) x_stride, stream)) {
                        return;
                    }
                    if (q8_lds_nc_dispatch(one, (uint32_t) ne01, xq, ne11, (uint32_t) ne00, (uint32_t) x_stride, stream)) {
                        return;
                    }
                }
                static const bool no_ncu = getenv("GGML_CUDA_NO_Q8_NCU") != nullptr;
                const int64_t n_it = no_ncu ? 0 : (2*nb + 63) / 64;
                const dim3 grid((ne01 + 3) / 4);
                switch (n_it) {
                    case 3:  mul_mat_vec_q8_0_repacked_nc<NC, 3><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
                    case 4:  mul_mat_vec_q8_0_repacked_nc<NC, 4><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
                    case 6:  mul_mat_vec_q8_0_repacked_nc<NC, 6><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
                    default: mul_mat_vec_q8_0_repacked_nc<NC, 0><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride); break;
                }
            }
        };
        switch (ne11) {
            case 2:  launch(std::integral_constant<int, 2>{}); break;
            case 3:  launch(std::integral_constant<int, 3>{}); break;
            case 4:  launch(std::integral_constant<int, 4>{}); break;
            case 5:  launch(std::integral_constant<int, 5>{}); break;
            case 6:  launch(std::integral_constant<int, 6>{}); break;
            case 7:  launch(std::integral_constant<int, 7>{}); break;
            case 8:  launch(std::integral_constant<int, 8>{}); break;
            case 9:  launch(std::integral_constant<int, 9>{}); break;
            case 10: launch(std::integral_constant<int, 10>{}); break;
            case 11: launch(std::integral_constant<int, 11>{}); break;
            case 12: launch(std::integral_constant<int, 12>{}); break;
            case 13: launch(std::integral_constant<int, 13>{}); break;
            case 14: launch(std::integral_constant<int, 14>{}); break;
            case 15: launch(std::integral_constant<int, 15>{}); break;
            default: launch(std::integral_constant<int, 16>{}); break;
        }
        return;
    }
    // dense K-quants, 2..16 columns: weight-once matvec instead of the 64-wide MMQ tile
    static const bool no_kq_nc = getenv("GGML_CUDA_NO_KQ_NC") != nullptr;
    const ggml_type eff = repack_eff_type(src0->type);
    if (!no_kq_nc && ne11 >= 2 && ne11 <= 16 &&
        (eff == GGML_TYPE_Q4_K || eff == GGML_TYPE_Q5_K || eff == GGML_TYPE_Q6_K || eff == GGML_TYPE_Q4_0 ||
         eff == GGML_TYPE_IQ4_NL || eff == GGML_TYPE_IQ4_XS || eff == GGML_TYPE_IQ3_S)) {
        switch (eff) {
            case GGML_TYPE_Q4_K:   launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_Q4_K>  (w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
            case GGML_TYPE_Q5_K:   launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_Q5_K>  (w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
            case GGML_TYPE_Q4_0:   launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_Q4_0>  (w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
            case GGML_TYPE_IQ4_NL: launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_IQ4_NL>(w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
            case GGML_TYPE_IQ4_XS: launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_IQ4_XS>(w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
            case GGML_TYPE_IQ3_S:  launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_IQ3_S> (w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
            default:               launch_mul_mat_vec_kq_repacked_nc<GGML_TYPE_Q6_K>  (w, xq, dst_d, ne00, ne01, ne11, x_stride, stream); break;
        }
        return;
    }
    if (ne11 == 1) {
        // decode: dp4a matvec straight from the planes
        switch (repack_eff_type(src0->type)) {
            case GGML_TYPE_Q3_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q3k_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                if (!kq_hoist()) {
                    mul_mat_vec_q4k_repacked<false, false, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else if (q4k_fence()) {
                    mul_mat_vec_q4k_repacked<false, true><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                    mul_mat_vec_q4k_repacked<false, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                if (kq_hoist()) {
                    mul_mat_vec_q5k_repacked<false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                    mul_mat_vec_q5k_repacked<false, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                if (kq_hoist()) {
                    mul_mat_vec_q6k_repacked<false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                    mul_mat_vec_q6k_repacked<false, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            case GGML_TYPE_Q5_1: {
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q5_1_repacked_seg<false>(w, xq, dst_d, ne00, ne01, 1, nullptr, 0, 0, 0, stream);
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_q5_1_repacked<false><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_0: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_nib_repacked<REPACK_NIB_Q4_0><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
            } break;
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_nib_repacked<REPACK_NIB_IQ4NL><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
            } break;
            case GGML_TYPE_IQ3_S: {
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_nib_repacked<REPACK_NIB_IQ3S><<<grid, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
            } break;
            case GGML_TYPE_Q8_0: {
                // short rows: several rows per wave (see mul_mat_vec_q8_0_repacked_seg)
                if (ne00 == 320) {
                    mul_mat_vec_q8_0_repacked_flat<10><<<(ne01 + 24) / 25, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne01);
                    break;
                }
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q8_0_repacked_seg<false>(w, xq, dst_d, ne00, ne01, 1, nullptr, 0, 0, 0, stream);
                    break;
                }
                // few long rows: split K across a whole workgroup per row
                if (ne01 <= 1024 && ne00 >= 4096) {
                    mul_mat_vec_q8_0_repacked_splitk<<<ne01, 256, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01);
                    break;
                }
                // ne01 >= 512: single-wave ROWS=1 blocks maximize the
                // wavefront count (measured on gfx906: ROWS=2 at
                // ne01=4096 stalls ~184 GB/s, ROWS=1 ~2x it; at 512-2560
                // rows ROWS=1 is also faster in qwen4exp decode, tg +1.5%).
                // Small ne01: 4-wave ROWS=2 blocks (the K-quant matvec shape).
                if (ne01 >= 512 && ne00 % 2048 == 0 && ne00 <= 8192) {
                    // K = 6144 (Flash-Next ssm_out / attn_output) takes the row-unrolled kernel (every half
                    // sub-block of the row in flight: 29.6 -> 28.2 us). For the 35B-A3B / 26B K = 4096
                    // shapes the one-wave loop is latency-bound (48.6 us per call vs 11.4 for reinstinct's
                    // 2-rows-per-wave kernel), but rowu<4> measured worse at 2048 rows (52 vs 38 us), so
                    // GGML_CUDA_Q8_ROWU selects the mapping for the A/B: "all" = rowu for every K here,
                    // "r2" = 2 rows per 64-thread block, "old" = the one-wave loop; default: rowu at 6144 only.
                    static const char q8_rowu_mode = getenv("GGML_CUDA_Q8_ROWU") ? getenv("GGML_CUDA_Q8_ROWU")[0] : 'd';
                    // measured (1x MI50, us per call): K=4096 at 2048 rows one-wave 67 / rowu<4> 53 / 2-rows-per-wave
                    // 29.7 (canonical 30); at 2816 rows 41 / 42 / 35. K=6144 at 2560 rows rowu<6> 43.7 vs 2-rows 46.7.
                    if (q8_rowu_mode == 'r' || (q8_rowu_mode == 'd' && ne00 == 4096)) {
                        mul_mat_vec_q8_0_repacked<2, 1, false, false><<<(ne01 + 1) / 2, 64, 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                        break;
                    }
                    if (q8_rowu_mode == 'a' || (q8_rowu_mode == 'd' && ne00 == 6144)) {
                        switch (ne00 / 1024) {
                            case 2:  mul_mat_vec_q8_0_repacked_rowu<2, 64><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            case 4:  mul_mat_vec_q8_0_repacked_rowu<4, 64><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            case 6:  mul_mat_vec_q8_0_repacked_rowu<6, 64><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                            default: mul_mat_vec_q8_0_repacked_rowu<8, 64><<<ne01, 64, 0, stream>>>(w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01); break;
                        }
                        break;
                    }
                }
                if (ne01 >= 512) {
                    const dim3 grid(ne01, 1, 1);
                    // opt-in (GGML_CUDA_Q8_HOIST=1): standalone the hoisted loop wins at K <= 3072 (2560x10240:
                    // 74 -> 57 us) and loses at K = 4096, but in-model it cost Gemma 26B-A4B 0.7 ms/token
                    // (97 -> 90.5 tok/s, graphs on) with no measured win on Flash-Next, Qwen3.5-4B Q5_K_M or
                    // UD-Q4_K_XL, so the pre-R10 loop is the default
                    static const bool q8_hoist = getenv("GGML_CUDA_Q8_HOIST") != nullptr;
                    if (!q8_hoist || ne00 > 3072) {
                        mul_mat_vec_q8_0_repacked<1, 1, false, false><<<grid, 64, 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                        break;
                    }
                    mul_mat_vec_q8_0_repacked<1, 1, false><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                    // 4-wave ROWS=2 with half-sub-block work units is the
                    // best of the swept variants (gfx906, 0.8B-Q8_0 tg128:
                    // 231.0 vs 222.7 single-wave, 222.2 full-block units,
                    // 219.9 ROWS=4, 214.2 quarter units; canonical mmvq
                    // is 238.0 — the residual ~3% is why Q8_0 stays
                    // behind its own env gate)
                    const dim3 grid((ne01 + 7) / 8, 1, 1);
                    static const bool q8_hoist2 = getenv("GGML_CUDA_Q8_HOIST") != nullptr;
                    if (!q8_hoist2 || ne00 > 3072) {
                        mul_mat_vec_q8_0_repacked<2, 4, false, false><<<grid, 256, 0, stream>>>(
                            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, nullptr, nullptr, nullptr, 0, 0, 0, 0);
                        break;
                    }
                    mul_mat_vec_q8_0_repacked<2, 4, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // prefill: int8 MMQ tile GEMM straight from the repacked planes. The token tile is 16*TN wide;
    // 17..48 columns (short prompts, MTP eh_proj) take the 32/48-wide instantiations instead of
    // paying for a 64-wide tile (Q4_0 5120x17408 at 24 columns: 786 -> ~400 us).
    auto launch_tn = [&](auto tn_c) {
        constexpr int TN = decltype(tn_c)::value;
        const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM, (ne11 + 16 * TN - 1) / (16 * TN), 1);
        switch (repack_eff_type(src0->type)) {
            case GGML_TYPE_Q3_K:
                mmq_gemm_q3k_repacked<false, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_K:
                mmq_gemm_q4k_repacked<false, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_K:
                mmq_gemm_q5k_repacked<false, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q6_K:
                mmq_gemm_q6k_repacked<false, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_1:
                mmq_gemm_q5_1_repacked<false, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q8_0:
                mmq_gemm_q8_0_repacked<false, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_0:
                mmq_gemm_nib_repacked<REPACK_NIB_Q4_0, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride, (uint32_t) ne01);
                break;
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
                mmq_gemm_nib_repacked<REPACK_NIB_IQ4NL, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride, (uint32_t) ne01);
                break;
            case GGML_TYPE_IQ3_S:
                mmq_gemm_nib_repacked<REPACK_NIB_IQ3S, TN><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride, (uint32_t) ne01);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
    };
    static const bool no_narrow = getenv("GGML_CUDA_NO_MMQ_NARROW") != nullptr;
    if (no_narrow || ne11 > 48) {
        launch_tn(std::integral_constant<int, MMQ_RP_TN>{});
    } else if (ne11 <= 16) {
        launch_tn(std::integral_constant<int, 1>{});
    } else if (ne11 <= 32) {
        launch_tn(std::integral_constant<int, 2>{});
    } else {
        launch_tn(std::integral_constant<int, 3>{});
    }
    GGML_UNUSED(ctx);
}

// Routing cache use: bypassed on side streams and with
// GGML_CUDA_REPACK_NO_ROUTE_CACHE set. Under graph capture the hit/miss
// sequence is baked into the graph, which is fine: each replay reruns the
// misses in the same stream order, and cache buffers are never freed while
// the context lives. Growing (cudaMalloc) is not done while capturing;
// *capturing tells the caller to fall back to pool buffers instead.
static bool repack_route_cache_usable(ggml_backend_cuda_context & ctx, cudaStream_t stream, bool * capturing) {
    static const bool disabled = getenv("GGML_CUDA_REPACK_NO_ROUTE_CACHE") != nullptr;
    if (disabled || ctx.curr_stream_no != 0) {
        return false;
    }
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus st;
    CUDA_CHECK(hipStreamIsCapturing(stream, &st));
    *capturing = st != hipStreamCaptureStatusNone;
#else
    cudaStreamCaptureStatus st;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &st));
    *capturing = st != cudaStreamCaptureStatusNone;
#endif
    return true;
}

// grow-only cache buffer; false if it would have to grow during capture
static bool repack_rc_reserve(ggml_cuda_repack_route_cache & rc, void ** buf, size_t * cap, size_t need, bool capturing) {
    if (need <= *cap) {
        return true;
    }
    if (capturing) {
        return false;
    }
    if (*buf != nullptr) {
        rc.retired.push_back(*buf);
    }
    CUDA_CHECK(cudaMalloc(buf, need));
    *cap = need;
    return true;
}

void ggml_cuda_repack_xq_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * node, bool force,
        const ggml_tensor * keep) {
    if (node->data == nullptr) {
        return;
    }
    const char * lo = (const char *) node->data;
    const char * hi = lo + ggml_nbytes(node);
    for (auto & e : ctx.repack_rc.xqc) {
        if (e.gen == ctx.graph_gen && (force || (e.producer != node && (keep == nullptr || e.producer != keep))) &&
                lo < e.hi && e.lo < hi) {
            e.gen = 0;
        }
    }
}

static const ggml_tensor * repack_view_root(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

void * ggml_cuda_repack_xq_emit_target(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const ggml_tensor * t,
        int64_t * id_blocks_per_row) {
    static const bool disabled = getenv("GGML_CUDA_NO_Q8_EMIT") != nullptr;
    if (id_blocks_per_row != nullptr) {
        *id_blocks_per_row = 0;
    }
    if (disabled || cgraph == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            ggml_nelements(t) % QK8_1 != 0) {
        return nullptr;
    }
    bool consumer = false;
    const ggml_tensor * id_x = nullptr; // MUL_MAT_ID consumer: entry in its padded [ne2][ne1][pad(ne0)/32] layout
    for (int i = 0; i < cgraph->n_nodes && !consumer; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if ((n->op != GGML_OP_MUL_MAT && n->op != GGML_OP_MUL_MAT_ID) || !n->src[0]->buffer ||
                !ggml_backend_buft_is_cuda_repack(n->src[0]->buffer->buft)) {
            continue;
        }
        const ggml_tensor * x = n->src[1];
        if (n->op == GGML_OP_MUL_MAT_ID) {
            // the expert kernels read src1 as padded rows; only when the producer can write that layout
            if (id_blocks_per_row != nullptr && x == t && ggml_is_contiguous(x) && x->type == GGML_TYPE_F32 &&
                    x->ne[3] == 1 && x->ne[0] % QK8_1 == 0) {
                consumer = true;
                id_x     = x;
            }
            continue;
        }
        // several columns read the flat blocks as [ne1][ne0/32], which is the consumer's layout only when
        // ne0 needs no row padding
        consumer = repack_view_root(x) == repack_view_root(t) && x->data == t->data && ggml_is_contiguous(x) &&
            x->type == GGML_TYPE_F32 && x->ne[2] == 1 && x->ne[3] == 1 &&
            (x->ne[1] == 1 || x->ne[0] % MATRIX_ROW_PADDING == 0) && x->ne[0]*x->ne[1] <= ggml_nelements(t);
    }
    bool capturing = false;
    if (!consumer || !repack_route_cache_usable(ctx, ctx.stream(), &capturing)) {
        return nullptr;
    }
    ggml_cuda_repack_route_cache & rc = ctx.repack_rc;
    auto & e = rc.xqc[rc.xqc_next];
    rc.xqc_next = (rc.xqc_next + 1) % ggml_cuda_repack_route_cache::N_XQ;
    e.gen = 0;
    const int64_t id_stride = id_x != nullptr ? GGML_PAD(id_x->ne[0], MATRIX_ROW_PADDING) / QK8_1 : 0;
    const size_t bytes = id_x != nullptr ? (size_t) id_x->ne[2] * id_x->ne[1] * id_stride * sizeof(block_q8_1)
                                         : ggml_nelements(t) / QK8_1 * sizeof(block_q8_1);
    if (!repack_rc_reserve(rc, (void **) &e.buf, &e.cap, bytes, capturing)) {
        return nullptr;
    }
    e.gen      = ctx.graph_gen;
    e.data     = t->data;
    if (id_x != nullptr) {
        // served through the exact-shape match in repack_quantize_x, like a quantized entry
        memcpy(e.ne, id_x->ne, sizeof(e.ne));
        memcpy(e.nb, id_x->nb, sizeof(e.nb));
        *id_blocks_per_row = id_stride;
    } else {
        memset(e.ne, 0, sizeof(e.ne));
        memset(e.nb, 0, sizeof(e.nb));
    }
    e.lo       = (const char *) t->data;
    e.hi       = e.lo + ggml_nbytes(t);
    e.producer = t;
    e.flat_n   = id_x != nullptr ? 0 : ggml_nelements(t);
    return e.buf;
}

// Quantize src1 (F32, dim0 dense) to q8_1 as [ne3][ne2][ne1][ne10_padded/QK8_1]
// blocks. On the main stream the result is cached per graph: gate/up and the
// shared expert (or q/k/v) quantize the same activation. Entries die when any
// node writes their range (the graph loop calls ggml_cuda_repack_xq_invalidate),
// so reused allocator memory never serves stale data.
static const block_q8_1 * repack_quantize_x(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
        const int64_t ne10_padded, ggml_cuda_pool_alloc<char> & fallback, cudaStream_t stream) {
    const size_t bytes = src1->ne[3] * src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1;
    auto quantize = [&](char * out) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, out, GGML_TYPE_Q8_0, src1->ne[0],
            src1->nb[1] / sizeof(float), src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float),
            ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3], stream);
    };
    bool capturing = false;
    if (!repack_route_cache_usable(ctx, stream, &capturing)) {
        char * p = fallback.alloc(bytes);
        quantize(p);
        return (const block_q8_1 *) p;
    }
    ggml_cuda_repack_route_cache & rc = ctx.repack_rc;
    // an emitted (flat) entry serves one column, or several when ne0 needs no row padding (then the flat
    // blocks are exactly [ne1][ne10_padded/32])
    const bool flat_ok = src1->ne[2] == 1 && src1->ne[3] == 1 && ggml_is_contiguous(src1) &&
        (src1->ne[1] == 1 || ne10_padded == src1->ne[0]);
    for (auto & e : rc.xqc) {
        if (e.gen != ctx.graph_gen || e.data != src1->data) {
            continue;
        }
        if (e.flat_n > 0 ? (flat_ok && src1->ne[0]*src1->ne[1] <= e.flat_n) :
                (memcmp(e.ne, src1->ne, sizeof(e.ne)) == 0 && memcmp(e.nb, src1->nb, sizeof(e.nb)) == 0)) {
            return (const block_q8_1 *) e.buf;
        }
    }
    auto & e = rc.xqc[rc.xqc_next];
    rc.xqc_next = (rc.xqc_next + 1) % ggml_cuda_repack_route_cache::N_XQ;
    e.gen = 0;
    if (!repack_rc_reserve(rc, (void **) &e.buf, &e.cap, bytes, capturing)) {
        char * p = fallback.alloc(bytes);
        quantize(p);
        return (const block_q8_1 *) p;
    }
    quantize(e.buf);
    e.gen  = ctx.graph_gen;
    e.data = src1->data;
    memcpy(e.ne, src1->ne, sizeof(e.ne));
    memcpy(e.nb, src1->nb, sizeof(e.nb));
    e.lo = (const char *) src1->data;
    e.hi = e.lo + ggml_nbytes(src1);
    e.producer = nullptr;
    e.flat_n   = 0;
    return (const block_q8_1 *) e.buf;
}

// MUL_MAT_ID with src0 in the repack buffer type. The mm_ids_helper
// compacts routing into expert-sorted assignment order; activations are
// quantized once in natural column order and gathered per assignment
// via ids_src1 inside the kernels; outputs scatter via ids_dst.
// small-batch MoE: give every (token, expert) slot its own copy of the token's q8_1 activation column
static __global__ void repack_expand_x_slots(const block_q8_1 * __restrict__ xq, block_q8_1 * __restrict__ xe,
        const int x_stride, const int n_used) {
    const int a = blockIdx.x;
    const int * src = (const int *) (xq + (int64_t) (a / n_used) * x_stride);
    int       * dst = (int       *) (xe + (int64_t) a * x_stride);
    const int n = x_stride * (int) (sizeof(block_q8_1) / sizeof(int));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    // column-contiguity: ids_src1/ids_dst are flat column indices
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // rows per expert
    const int64_t ne02 = src0->ne[2]; // experts
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const size_t  xq_bytes    = src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1;
    const int     si1         = ids->nb[1] / sizeof(int32_t);
    const int     sis1        = src1->nb[2] / src1->nb[1];

    // a few tokens (speculative verify): run the per-slot decode kernels over all n_tokens*n_used slots
    // instead of the routed tile GEMM, whose fixed cost dominates at this size
    static const bool no_small = getenv("GGML_CUDA_NO_MOE_SMALL") != nullptr;
    const bool small = !no_small && n_tokens > 1 && n_tokens <= 16 &&
        (src1->ne[1] == 1 || (src1->ne[1] == n_expert_used && sis1 == n_expert_used));

    // batch: grouped tile GEMM, thin 16-token tiles (MoE routing spreads
    // tokens across experts; a 64-wide tile would be mostly empty)
    constexpr int TN_ID = 1;
    // over-launch upper bound: every expert can add one partial tile
    const int64_t max_tiles = n_assign / (16 * TN_ID) + ne02;
    GGML_ASSERT(ne02 <= 4096);

    int32_t * p_ids_src1    = nullptr;
    int32_t * p_ids_dst     = nullptr;
    int32_t * p_bounds      = nullptr;
    int32_t * p_tile_off    = nullptr;
    int32_t * p_tile_expert = nullptr;
    char    * p_xq          = nullptr;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> tile_off(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> tile_expert(ctx.pool());
    ggml_cuda_pool_alloc<char>    src1_q8_1(ctx.pool());

    auto quantize_x = [&](char * xq_out) {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, xq_out,
            src0->type, ne10, s11, s12, s12 * src1->ne[2], ne10_padded,
            src1->ne[1], src1->ne[2], 1, stream);
    };
    auto route = [&](int32_t * s1, int32_t * d, int32_t * b, int32_t * to, int32_t * te) {
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, s1, d, b,
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
        CUDA_CHECK(cudaGetLastError());
        repack_tile_map<16 * TN_ID><<<1, 1024, 0, stream>>>(b, to, te, ne02);
    };

    bool capturing = false;
    const bool use_rc = n_tokens > 1 && !small && repack_route_cache_usable(ctx, stream, &capturing);
    ggml_cuda_repack_route_cache & rc = ctx.repack_rc;

    if (n_tokens > 1 && use_rc) {
        // gate/up/down of one layer share ids: reuse the routing
        const bool route_hit = rc.gen == ctx.graph_gen && rc.ids == ids && rc.ids_data == ids->data &&
            rc.n_tokens == n_tokens && rc.n_used == n_expert_used && rc.ne02 == ne02 && rc.si1 == si1;
        // ids_src1[a] = it*sis1 + slot % ne11 (contiguous src1: sis1 == ne11),
        // ids_dst[a] = it*n_used + slot, so ne11 == n_used reuses ids_dst
        const bool src1_is_dst = src1->ne[1] == n_expert_used && sis1 == n_expert_used;
        if (route_hit && (rc.ne11 == src1->ne[1] || src1_is_dst)) {
            p_ids_src1    = rc.ne11 == src1->ne[1] ? rc.ids_src1 : rc.ids_dst;
            p_ids_dst     = rc.ids_dst;
            p_bounds      = rc.bounds;
            p_tile_off    = rc.tile_off;
            p_tile_expert = rc.tile_expert;
        } else {
            rc.gen = 0; // invalid until rewritten below
            const size_t need = (2 * n_assign + 2 * (ne02 + 1) + max_tiles) * sizeof(int32_t);
            if (repack_rc_reserve(rc, &rc.route_buf, &rc.cap, need, capturing)) {
                rc.ids_src1    = (int32_t *) rc.route_buf;
                rc.ids_dst     = rc.ids_src1 + n_assign;
                rc.bounds      = rc.ids_dst + n_assign;
                rc.tile_off    = rc.bounds + ne02 + 1;
                rc.tile_expert = rc.tile_off + ne02 + 1;
                rc.gen      = ctx.graph_gen;
                rc.ids      = ids;
                rc.ids_data = ids->data;
                rc.n_tokens = n_tokens;
                rc.n_used   = n_expert_used;
                rc.ne02     = ne02;
                rc.si1      = si1;
                rc.ne11     = src1->ne[1];
                p_ids_src1    = rc.ids_src1;
                p_ids_dst     = rc.ids_dst;
                p_bounds      = rc.bounds;
                p_tile_off    = rc.tile_off;
                p_tile_expert = rc.tile_expert;
            } else {
                p_ids_src1    = ids_src1.alloc(n_assign);
                p_ids_dst     = ids_dst.alloc(n_assign);
                p_bounds      = expert_bounds.alloc(ne02 + 1);
                p_tile_off    = tile_off.alloc(ne02 + 1);
                p_tile_expert = tile_expert.alloc(max_tiles);
            }
            route(p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert);
        }

        // gate/up share src1: reuse the quantized activations
        const bool x_hit = rc.x_gen == ctx.graph_gen && rc.x == src1 && rc.x_data == src1->data &&
            rc.x_ne0 == src1->ne[0] && rc.x_ne1 == src1->ne[1] && rc.x_ne2 == src1->ne[2];
        if (x_hit) {
            p_xq = rc.xq;
        } else {
            rc.x_gen = 0;
            if (repack_rc_reserve(rc, (void **) &rc.xq, &rc.xq_cap, xq_bytes, capturing)) {
                p_xq      = rc.xq;
                rc.x_gen  = ctx.graph_gen;
                rc.x      = src1;
                rc.x_data = src1->data;
                rc.x_ne0  = src1->ne[0];
                rc.x_ne1  = src1->ne[1];
                rc.x_ne2  = src1->ne[2];
            } else {
                p_xq = src1_q8_1.alloc(xq_bytes);
            }
            quantize_x(p_xq);
        }
    } else {
        if (n_tokens > 1 && !small) {
            p_ids_src1    = ids_src1.alloc(n_assign);
            p_ids_dst     = ids_dst.alloc(n_assign);
            p_bounds      = expert_bounds.alloc(ne02 + 1);
            p_tile_off    = tile_off.alloc(ne02 + 1);
            p_tile_expert = tile_expert.alloc(max_tiles);
            route(p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert);
        }
        // quantize all activation columns once, natural order
        p_xq = (char *) repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
    }
    // the slot kernels read ids as one flat row of n_tokens*n_used; ids is usually a view of the
    // full per-token expert ranking, so compact it
    const int32_t * ids_d = (const int32_t *) ids->data;
    ggml_cuda_pool_alloc<int32_t> ids_flat(ctx.pool());
    if (small && si1 != n_expert_used) {
        ids_flat.alloc(n_assign);
        CUDA_CHECK(cudaMemcpy2DAsync(ids_flat.get(), n_expert_used * sizeof(int32_t), ids->data, ids->nb[1],
            n_expert_used * sizeof(int32_t), n_tokens, cudaMemcpyDeviceToDevice, stream));
        ids_d = ids_flat.get();
    }
    ggml_cuda_pool_alloc<block_q8_1> x_slots(ctx.pool());
    if (small && src1->ne[1] == 1) {
        x_slots.alloc(n_assign * x_stride);
        repack_expand_x_slots<<<n_assign, 256, 0, stream>>>((const block_q8_1 *) p_xq, x_slots.get(), (int) x_stride, (int) n_expert_used);
        p_xq = (char *) x_slots.get();
    }
    const block_q8_1 * xq = (const block_q8_1 *) p_xq;

    if (n_tokens == 1 || small) {
        // decode: one matvec per slot; experts read directly from the
        // raw ids tensor in-kernel (no compaction kernels — launch
        // parity with canonical mmvq-id). Broadcast src1 (ne[1]==1, one
        // shared activation column for all slots) uses x-stride 0; a few
        // tokens have one expanded column per slot.
        const uint32_t xs_eff = src1->ne[1] == 1 && !small ? 0u : (uint32_t) x_stride;
        switch (repack_eff_type(src0->type)) {
            case GGML_TYPE_Q3_K: {
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q3k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_d, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q4_K: {
                if (ne00 <= 2048 && launch_mul_mat_vec_kq_repacked_pack<GGML_TYPE_Q4_K>(w, xq, dst_d, ne00, ne01, n_assign,
                        ids_d, expert_stride, xs_eff, dst_s1, stream)) {
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q4k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_d, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q5_K: {
                if (ne00 <= 2048 && launch_mul_mat_vec_kq_repacked_pack<GGML_TYPE_Q5_K>(w, xq, dst_d, ne00, ne01, n_assign,
                        ids_d, expert_stride, xs_eff, dst_s1, stream)) {
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q5k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_d, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q6_K: {
                if (ne00 <= 2048 && launch_mul_mat_vec_kq_repacked_pack<GGML_TYPE_Q6_K>(w, xq, dst_d, ne00, ne01, n_assign,
                        ids_d, expert_stride, xs_eff, dst_s1, stream)) {
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q6k_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_d, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q5_1: {
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q5_1_repacked_seg<true>(w, xq, dst_d, ne00, ne01, n_assign,
                        ids_d, expert_stride, xs_eff, dst_s1, stream);
                    break;
                }
                const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                mul_mat_vec_q5_1_repacked<true><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    ids_d, nullptr, nullptr,
                    (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
            } break;
            case GGML_TYPE_Q8_0: {
                if (ne00 <= 1024) {
                    launch_mul_mat_vec_q8_0_repacked_seg<true>(w, xq, dst_d, ne00, ne01, n_assign,
                        ids_d, expert_stride, xs_eff, dst_s1, stream);
                    break;
                }
                if (ne01 >= 4096) {
                    const dim3 grid(ne01, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, true><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        ids_d, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                } else {
                    const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, true><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        ids_d, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    const dim3 grid((ne01 + MMQ_RP_BM - 1) / MMQ_RP_BM, max_tiles, 1);

    switch (repack_eff_type(src0->type)) {
        case GGML_TYPE_Q3_K:
            mmq_gemm_q3k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q4_K:
            static_assert(TN_ID == 1, "w1 kernel tiles 16 assignments");
            mmq_gemm_q4k_repacked_id_w1<2><<<dim3((ne01 + 63) / 64, max_tiles, 1), 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q5_K:
            mmq_gemm_q5k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q6_K:
            mmq_gemm_q6k_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q5_1:
            mmq_gemm_q5_1_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<true, TN_ID><<<grid, 256, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                p_ids_src1, p_ids_dst, p_bounds, p_tile_off, p_tile_expert,
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
}

// Eligibility for the fused gate+up GLU path: both weights in the
// repack buffer type, Q4_K, identical shape; decode only (one output
// column per expert slot); SWIGLU or GEGLU.
bool ggml_cuda_repack_should_fuse_glu(const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * glu) {
    const ggml_tensor * wu = up->src[0];
    const ggml_tensor * wg = gate->src[0];
    if (wu->buffer == nullptr || wg->buffer == nullptr ||
        !ggml_backend_buft_is_cuda_repack(wu->buffer->buft) ||
        !ggml_backend_buft_is_cuda_repack(wg->buffer->buft)) {
        return false;
    }
    if (wu->type != wg->type || !ggml_are_same_shape(wu, wg)) {
        return false;
    }
    // Q8_0 / Q4_0 / IQ family: dense decode only
    const bool dense_only = wu->type == GGML_TYPE_Q8_0 || wu->type == GGML_TYPE_Q4_0 || wu->type == GGML_TYPE_IQ4_NL ||
                            wu->type == GGML_TYPE_IQ4_XS || wu->type == GGML_TYPE_IQ3_S;
    if (wu->type != GGML_TYPE_Q4_K && !(dense_only && up->src[2] == nullptr)) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (up->src[2] != nullptr) { // MUL_MAT_ID: one token, or a few with one shared column per token
        static const bool no_small = getenv("GGML_CUDA_NO_MOE_SMALL") != nullptr;
        const int64_t n_tokens = up->src[1]->ne[2];
        return glu->ne[2] == n_tokens && (n_tokens == 1 || (!no_small && n_tokens <= 16 && up->src[1]->ne[1] == 1));
    }
    return up->src[1]->ne[1] == 1; // dense: one column
}

bool ggml_cuda_mul_mat_id_repacked_down_reduce(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * weights, const ggml_tensor * expert_scale, ggml_tensor * dst, const bool copy_ids) {
    // GGML_CUDA_DOWN_REDUCE_R: rows per block for the fused down+reduce kernel (default 4; 0 = off)
    static const int down_r = getenv("GGML_CUDA_DOWN_REDUCE_R") ? atoi(getenv("GGML_CUDA_DOWN_REDUCE_R")) : 4;
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_sub = ne00 / 32;
    if (down_r <= 0 || src0->type != GGML_TYPE_Q5_1 || ne00 % 32 != 0 || n_sub > 64 || n_expert_used * n_sub > 256 ||
            ids->ne[1] != 1 || ids->nb[0] != sizeof(int32_t) || src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 ||
            src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || (src1->ne[1] != 1 && src1->ne[1] != n_expert_used) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || ggml_nelements(dst) != ne01 ||
            weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(weights) || ggml_nelements(weights) != n_expert_used ||
            (expert_scale != nullptr && (expert_scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(expert_scale) ||
                ggml_nelements(expert_scale) != n_expert_used))) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const block_q8_1 * xq = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t xs = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
    const int32_t * ids_d = (const int32_t *) ids->data;
    ggml_cuda_pool_alloc<int32_t> ids_copy(ctx.pool());
    if (copy_ids) {
        ids_copy.alloc(n_expert_used);
        CUDA_CHECK(cudaMemcpyAsync(ids_copy.get(), ids->data, n_expert_used * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        ids_d = ids_copy.get();
    }
    auto launch = [&](auto r_c) {
        constexpr int R = decltype(r_c)::value;
        const dim3 grid((ne01 + R - 1) / R, 1, 1);
        mul_mat_vec_q5_1_repacked_down_reduce<R><<<grid, 256, 0, stream>>>(
            (const uint8_t *) src0->data, xq, ids_d, (const float *) weights->data,
            expert_scale != nullptr ? (const float *) expert_scale->data : nullptr, (float *) dst->data,
            (uint32_t) ne00, (uint32_t) ne01, expert_stride, xs, (int) n_expert_used);
    };
    switch (down_r) {
        case 1:  launch(std::integral_constant<int, 1>{}); break;
        case 2:  launch(std::integral_constant<int, 2>{}); break;
        case 8:  launch(std::integral_constant<int, 8>{}); break;
        default: launch(std::integral_constant<int, 4>{}); break;
    }
    return true;
}

void ggml_cuda_mul_mat_repacked_fused_glu(ggml_backend_cuda_context & ctx,
        const ggml_tensor * up_w, const ggml_tensor * gate_w,
        const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const int glu_op) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    const int64_t ne00 = up_w->ne[0];
    const int64_t ne01 = up_w->ne[1];
    cudaStream_t stream = ctx.stream();
    const uint8_t * wu = (const uint8_t *) up_w->data;
    const uint8_t * wg = (const uint8_t *) gate_w->data;
    float * dst_d = (float *) dst->data;

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    if (ids == nullptr) {
        // dense decode column
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        const block_q8_1 * xq = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
        if (up_w->type == GGML_TYPE_Q4_0 || up_w->type == GGML_TYPE_IQ4_NL || up_w->type == GGML_TYPE_IQ4_XS ||
                up_w->type == GGML_TYPE_IQ3_S) {
            const dim3 grid((ne01 + 7) / 8, 1, 1);
            switch (up_w->type) {
                case GGML_TYPE_Q4_0:  mul_mat_vec_nib_repacked_glu<REPACK_NIB_Q4_0> <<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
                case GGML_TYPE_IQ3_S: mul_mat_vec_nib_repacked_glu<REPACK_NIB_IQ3S> <<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
                default:              mul_mat_vec_nib_repacked_glu<REPACK_NIB_IQ4NL><<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
            }
            return;
        }
        if (up_w->type == GGML_TYPE_Q8_0) {
            const dim3 grid((ne01 + 3) / 4, 1, 1);
            static const bool no_glu_unroll = getenv("GGML_CUDA_NO_Q8_GLU_UNROLL") != nullptr;
            const int64_t glu_iters = (ne00 / 16 + 63) / 64;
            if (!no_glu_unroll && glu_iters <= 4) {
                switch (glu_iters) {
                    case 1:  mul_mat_vec_q8_0_repacked_glu_u<1><<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
                    case 2:  mul_mat_vec_q8_0_repacked_glu_u<2><<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
                    case 3:  mul_mat_vec_q8_0_repacked_glu_u<3><<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
                    default: mul_mat_vec_q8_0_repacked_glu_u<4><<<grid, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op); break;
                }
                return;
            }
            mul_mat_vec_q8_0_repacked_glu<1><<<grid, 256, 0, stream>>>(
                wu, wg, xq, dst_d,
                (uint32_t) ne00, (uint32_t) ne01, glu_op);
            return;
        }
        const dim3 grid((ne01 + 7) / 8, 1, 1);
        if (!kq_hoist()) {
            mul_mat_vec_q4k_repacked_glu<false, 2, 1, false, false><<<grid, 256, 0, stream>>>(
                wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op, nullptr, nullptr, nullptr, 0, 0, 0, 0);
        } else if (q4k_fence()) {
            mul_mat_vec_q4k_repacked_glu<false, 2, 1, true><<<grid, 256, 0, stream>>>(
                wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op, nullptr, nullptr, nullptr, 0, 0, 0, 0);
        } else {
            mul_mat_vec_q4k_repacked_glu<false><<<grid, 256, 0, stream>>>(
                wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op, nullptr, nullptr, nullptr, 0, 0, 0, 0);
        }
        return;
    }

    // MoE decode: same routing machinery as the unfused ID path
    const int64_t ne02 = up_w->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    const int64_t n_assign = n_expert_used * n_tokens;
    const size_t expert_stride = repack_gcn_nbytes(up_w->type, ne00, ne01);
    GGML_ASSERT(dst->nb[1] == (size_t) dst->ne[0] * sizeof(float));
    GGML_ASSERT(dst->nb[2] == dst->nb[1] * dst->ne[1]);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    GGML_ASSERT(src1->ne[2] == n_tokens && src1->ne[3] == 1);
    GGML_ASSERT(n_tokens == 1 || src1->ne[1] == 1);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const block_q8_1 * xq = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
    uint32_t xs_eff = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
    const int32_t * ids_d = (const int32_t *) ids->data;
    ggml_cuda_pool_alloc<int32_t>    ids_flat(ctx.pool());
    ggml_cuda_pool_alloc<block_q8_1> x_slots(ctx.pool());
    if (n_tokens > 1 && n_tokens <= 6 && n_assign <= 64 && up_w->type == GGML_TYPE_Q4_K && ne00 == 2560 && moe_dedup_enabled()) {
        // a few tokens: each expert read once for all tokens routed to it, per-token activations
        if (ids->nb[1] != n_expert_used * sizeof(int32_t)) {
            ids_flat.alloc(n_assign);
            CUDA_CHECK(cudaMemcpy2DAsync(ids_flat.get(), n_expert_used * sizeof(int32_t), ids->data, ids->nb[1],
                n_expert_used * sizeof(int32_t), n_tokens, cudaMemcpyDeviceToDevice, stream));
            ids_d = ids_flat.get();
        }
        const dim3 grid16((ne01 + 15) / 16, n_assign, 1);
        if (n_tokens <= 4) {
            mul_mat_vec_q4k_repacked_glu16_dedup<5, 4><<<grid16, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne01, glu_op,
                ids_d, expert_stride, (uint32_t) x_stride, dst_s1, (int) n_expert_used, (int) n_tokens);
        } else {
            mul_mat_vec_q4k_repacked_glu16_dedup<5, 6><<<grid16, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne01, glu_op,
                ids_d, expert_stride, (uint32_t) x_stride, dst_s1, (int) n_expert_used, (int) n_tokens);
        }
        return;
    }
    if (n_tokens > 1) {
        // a few tokens: flat ids, one activation column per (token, expert) slot
        if (ids->nb[1] != n_expert_used * sizeof(int32_t)) {
            ids_flat.alloc(n_assign);
            CUDA_CHECK(cudaMemcpy2DAsync(ids_flat.get(), n_expert_used * sizeof(int32_t), ids->data, ids->nb[1],
                n_expert_used * sizeof(int32_t), n_tokens, cudaMemcpyDeviceToDevice, stream));
            ids_d = ids_flat.get();
        }
        x_slots.alloc(n_assign * x_stride);
        repack_expand_x_slots<<<n_assign, 256, 0, stream>>>(xq, x_slots.get(), (int) x_stride, (int) n_expert_used);
        xq = x_slots.get();
        xs_eff = (uint32_t) x_stride;
    }
    static const bool glu16 = getenv("GGML_CUDA_NO_Q4K_GLU16") == nullptr;
    if (glu16 && ne00 == 2560) {
        static const bool no_glu16_q8 = getenv("GGML_CUDA_NO_GLU16_Q8") != nullptr;
        if (!no_glu16_q8 && n_tokens == 1 && ne01 % QK8_1 == 0) {
            // the down projection reads this output: emit its q8_1 blocks here, no quantize launch
            int64_t yq_stride = 0;
            block_q8_1 * yq = (block_q8_1 *) ggml_cuda_repack_xq_emit_target(ctx, ctx.cur_cgraph, dst, &yq_stride);
            if (yq != nullptr && yq_stride >= ne01 / QK8_1) {
                const dim3 grid32(ne01 / QK8_1, n_assign, 1);
                mul_mat_vec_q4k_repacked_glu16_q8<5><<<grid32, 512, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne01, glu_op,
                    ids_d, expert_stride, xs_eff, dst_s1, yq, (uint32_t) yq_stride);
                return;
            }
        }
        const dim3 grid16((ne01 + 15) / 16, n_assign, 1);
        mul_mat_vec_q4k_repacked_glu16<5><<<grid16, 256, 0, stream>>>(wu, wg, xq, dst_d, (uint32_t) ne01, glu_op,
            ids_d, expert_stride, xs_eff, dst_s1);
        return;
    }
    // one row per wave: the 2-row kernel needs 51 VGPRs (4 waves/SIMD) and is
    // latency-bound at ~380 GB/s; ROWS=1 runs 41 vs 49 us per call in decode
    const dim3 grid((ne01 + 3) / 4, n_assign, 1);
    mul_mat_vec_q4k_repacked_glu<true, 1><<<grid, 256, 0, stream>>>(
        wu, wg, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, glu_op,
        ids_d, nullptr, nullptr, (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
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
    const int64_t ne2 = tensor->ne[2]; // experts (1 for plain 2D weights)

    const size_t src_stride = ggml_nbytes(tensor) / ne2;
    const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
    std::vector<uint8_t> staged(dst_stride * ne2);
    for (int64_t e = 0; e < ne2; e++) {
        const uint8_t * src_e = (const uint8_t *) data + e * src_stride;
        uint8_t       * dst_e = staged.data() + e * dst_stride;
        switch (tensor->type) {
            case GGML_TYPE_Q3_K:
                if (repack_eff_type(GGML_TYPE_Q3_K) == GGML_TYPE_Q6_K) {
                    repack_q3k_as_q6k_host((const block_q3_K *) src_e, dst_e, ne0, ne1);
                } else {
                    repack_q3k_host((const block_q3_K *) src_e, dst_e, ne0, ne1);
                }
                break;
            case GGML_TYPE_IQ4_NL: repack_iq4_nl_host((const block_iq4_nl *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_IQ4_XS: repack_iq4_xs_host((const block_iq4_xs *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_IQ3_S:  repack_iq3_s_host ((const block_iq3_s  *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q4_K: repack_q4k_host ((const block_q4_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q5_K: repack_q5k_host ((const block_q5_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q6_K: repack_q6k_host ((const block_q6_K *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q8_0: repack_q8_0_host((const block_q8_0 *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q5_1: repack_q5_1_host((const block_q5_1 *) src_e, dst_e, ne0, ne1); break;
            case GGML_TYPE_Q4_0: repack_q4_0_host((const block_q4_0 *) src_e, dst_e, ne0, ne1); break;
            default:             GGML_ABORT("unsupported repack type");
        }
    }

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, staged.data(), staged.size(),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// host-side entry points for tests/test-repack-host.cpp (layout checks without a device)
size_t ggml_cuda_repack_nbytes_for_test(ggml_type type, int64_t ne0, int64_t ne1) {
    return repack_gcn_nbytes(type, ne0, ne1);
}
int ggml_cuda_repack_eff_type_for_test(ggml_type type) {
    return (int) repack_eff_type(type);
}
void ggml_cuda_repack_host_for_test(ggml_type type, const void * src, void * dst, int64_t ne0, int64_t ne1) {
    uint8_t * d = (uint8_t *) dst;
    switch (type) {
        case GGML_TYPE_Q3_K:
            if (repack_eff_type(GGML_TYPE_Q3_K) == GGML_TYPE_Q6_K) {
                repack_q3k_as_q6k_host((const block_q3_K *) src, d, ne0, ne1);
            } else {
                repack_q3k_host((const block_q3_K *) src, d, ne0, ne1);
            }
            break;
        case GGML_TYPE_Q4_K:   repack_q4k_host   ((const block_q4_K   *) src, d, ne0, ne1); break;
        case GGML_TYPE_Q5_K:   repack_q5k_host   ((const block_q5_K   *) src, d, ne0, ne1); break;
        case GGML_TYPE_Q6_K:   repack_q6k_host   ((const block_q6_K   *) src, d, ne0, ne1); break;
        case GGML_TYPE_Q8_0:   repack_q8_0_host  ((const block_q8_0   *) src, d, ne0, ne1); break;
        case GGML_TYPE_Q5_1:   repack_q5_1_host  ((const block_q5_1   *) src, d, ne0, ne1); break;
        case GGML_TYPE_Q4_0:   repack_q4_0_host  ((const block_q4_0   *) src, d, ne0, ne1); break;
        case GGML_TYPE_IQ4_NL: repack_iq4_nl_host((const block_iq4_nl *) src, d, ne0, ne1); break;
        case GGML_TYPE_IQ4_XS: repack_iq4_xs_host((const block_iq4_xs *) src, d, ne0, ne1); break;
        case GGML_TYPE_IQ3_S:  repack_iq3_s_host ((const block_iq3_s  *) src, d, ne0, ne1); break;
        default: GGML_ABORT("unsupported repack type");
    }
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
        return repack_gcn_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]) * tensor->ne[2];
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

    // Default-on for GCN; GGML_CUDA_REPACK=0 opts out. (Repacked
    // weights cannot be read back: llama-quantize/save from a loaded
    // model needs the opt-out.)
    const char * env = getenv("GGML_CUDA_REPACK");
    if (env != nullptr && env[0] == '0') {
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

// structural part only: graph_optimize groups on it, so the node order does not depend on the
// token count (a count-dependent order differs between ubatches and forces a re-reserve)
bool ggml_cuda_repack_q8_multi_group(const ggml_tensor * mm) {
    static const bool disabled = getenv("GGML_CUDA_NO_Q8_MULTI") != nullptr;
    if (disabled || mm->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    return w->buffer && ggml_backend_buft_is_cuda_repack(w->buffer->buft) && w->type == GGML_TYPE_Q8_0 &&
        w->ne[0] == 2560 && w->ne[1] >= 512 && w->ne[2] == 1 && w->ne[3] == 1 &&
        mm->src[1]->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32;
}

bool ggml_cuda_repack_q8_multi_ok(const ggml_tensor * mm) {
    const ggml_tensor * x = mm->src[1];
    return ggml_cuda_repack_q8_multi_group(mm) &&
        x->ne[1] >= 1 && x->ne[1] <= 16 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) &&
        (x->ne[1] == 1 || ggml_is_contiguous(x)) && ggml_is_contiguous(mm);
}

void ggml_cuda_mul_mat_repacked_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mms, int n) {
    GGML_ASSERT(n >= 2 && n <= 3);
    const ggml_tensor * src1 = mms[0]->src[1];
    const int64_t ne00 = mms[0]->src[0]->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const block_q8_1 * xq = repack_quantize_x(ctx, src1, ne10_padded, src1_q8_1, stream);
    q8_multi_args args = {};
    uint32_t rows = 0;
    for (int i = 0; i < 3; i++) {
        args.start[i] = rows;
        if (i < n) {
            args.w[i]   = (const uint8_t *) mms[i]->src[0]->data;
            args.y[i]   = (float *) mms[i]->data;
            args.ne1[i] = (uint32_t) mms[i]->src[0]->ne[1];
            rows += args.ne1[i];
        } else {
            args.start[i] = UINT32_MAX;
        }
    }
    const int64_t ne11 = src1->ne[1];
    if (ne11 == 1) {
        static const bool no_unroll = getenv("GGML_CUDA_NO_Q8_MULTI_UNROLL") != nullptr;
        const int64_t iters = (ne00 / 16 + 63) / 64; // half sub-blocks per lane
        if (!no_unroll && iters >= 1 && iters <= 4) {
            switch (iters) {
                case 1:  mul_mat_vec_q8_0_repacked_multi_u<1><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
                case 2:  mul_mat_vec_q8_0_repacked_multi_u<2><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
                case 3:  mul_mat_vec_q8_0_repacked_multi_u<3><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
                default: mul_mat_vec_q8_0_repacked_multi_u<4><<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00); break;
            }
        } else {
            mul_mat_vec_q8_0_repacked_multi<<<rows, 64, 0, stream>>>(args, xq, (uint32_t) ne00);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // several columns (speculative verify, several sequences): the nc kernel over the grouped rows
    const int64_t x_stride = ne10_padded / QK8_1;
    if (q8_lds_rows_dispatch(args, rows, xq, ne11, (uint32_t) ne00, (uint32_t) x_stride, stream)) {
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (q8_lds_nc_dispatch(args, rows, xq, ne11, (uint32_t) ne00, (uint32_t) x_stride, stream)) {
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int64_t n_it     = (2*(ne00/32) + 63) / 64;
    const dim3 grid((rows + 3) / 4);
    auto launch = [&](auto nc) {
        constexpr int NC = decltype(nc)::value;
        if (n_it == 3) {
            mul_mat_vec_q8_0_repacked_multi_nc<NC, 3><<<grid, 256, 0, stream>>>(args, xq, (uint32_t) ne00, (uint32_t) x_stride);
        } else {
            mul_mat_vec_q8_0_repacked_multi_nc<NC, 0><<<grid, 256, 0, stream>>>(args, xq, (uint32_t) ne00, (uint32_t) x_stride);
        }
    };
    switch (ne11) {
        case 2:  launch(std::integral_constant<int, 2>{}); break;
        case 3:  launch(std::integral_constant<int, 3>{}); break;
        case 4:  launch(std::integral_constant<int, 4>{}); break;
        case 5:  launch(std::integral_constant<int, 5>{}); break;
        case 6:  launch(std::integral_constant<int, 6>{}); break;
        case 7:  launch(std::integral_constant<int, 7>{}); break;
        case 8:  launch(std::integral_constant<int, 8>{}); break;
        case 9:  launch(std::integral_constant<int, 9>{}); break;
        case 10: launch(std::integral_constant<int, 10>{}); break;
        case 11: launch(std::integral_constant<int, 11>{}); break;
        case 12: launch(std::integral_constant<int, 12>{}); break;
        case 13: launch(std::integral_constant<int, 13>{}); break;
        case 14: launch(std::integral_constant<int, 14>{}); break;
        case 15: launch(std::integral_constant<int, 15>{}); break;
        default: launch(std::integral_constant<int, 16>{}); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// qwen4exp hyper-connection mix, after the down projection (one token):
//   SCALE -> SILU -> MUL_MAT(hc up, repacked Q8_0, K = 320) -> DSV4_HC_PRE(gated, 4 streams)
// A block quantizes silu(s*lo + b) itself (scale_silu_f32's arithmetic), computes the 4 stream rows of
// 32 output dims (mul_mat_vec_q8_0_repacked_flat<10>'s per-sub-block dots and sum order) and mixes them
// as dsv4_hc_pre_f32 does, so the result is bitwise that of the three launches it replaces.
static constexpr int HC_UP_K  = 320;
static constexpr int HC_UP_NB = HC_UP_K/32;
static constexpr int HC_UP_HC = 4;
static constexpr int HC_UP_D  = 32; // output dims per block

static __global__ void __launch_bounds__(256) hc_up_pre_f32(
        const float * __restrict__ lo, const float scale, const float bias,
        const uint8_t * __restrict__ wbase, const uint32_t ne1,
        const float * __restrict__ xn, const int64_t sx1,
        const float pre_scale, float * __restrict__ dst, block_q8_1 * __restrict__ yq, const int64_t n_embd,
        const int64_t sx2) {
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr int ROWS = HC_UP_HC*HC_UP_D;
    __shared__ block_q8_1 aq[HC_UP_NB];
    __shared__ float      part[ROWS*HC_UP_NB];
    __shared__ float      gate[ROWS];

    const int t = threadIdx.x;
    // blockIdx.y is the token: its own lo row, xn slab, dst row and q8 row
    {
        const int it = blockIdx.y;
        lo  += (int64_t) it*HC_UP_K;
        xn  += (int64_t) it*sx2;
        dst += (int64_t) it*n_embd;
        yq   = yq ? yq + (int64_t) it*(n_embd/QK8_1) : yq;
    }

    const int4     * qs4     = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * HC_UP_NB * 32);
    const int64_t d0 = (int64_t) blockIdx.x * HC_UP_D;

    // 1280 (row, sub-block) items over 256 threads: all five weight loads of a thread go out
    // together, ahead of the activation quantize
    static_assert(ROWS*HC_UP_NB == 5*256, "five items per thread");
    int4     w0[5];
    int4     w1[5];
    uint16_t db[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int item = t + 256*k;
        const int r    = item / HC_UP_NB;
        const int sb   = item % HC_UP_NB;
        const size_t row = (size_t) (r / HC_UP_D) * n_embd + d0 + r % HC_UP_D;
        const size_t u   = row * HC_UP_NB + sb;
        w0[k] = qs4[u * 2 + 0];
        w1[k] = qs4[u * 2 + 1];
        db[k] = d_plane[u];
    }

    for (int i = t; i < HC_UP_K; i += blockDim.x) {
        const float v = ggml_cuda_op_silu_single(scale * lo[i] + bias);
        float amax = fabsf(v);
        float sum  = v;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
        aq[i / QK8_1].qs[i % QK8_1] = q;
        if (i % QK8_1 == 0) {
            aq[i / QK8_1].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int item = t + 256*k;
        const int sb   = item % HC_UP_NB;
        const block_q8_1 * xb = aq + sb;
        const int4 * x4 = reinterpret_cast<const int4 *>(xb->qs);
        const int4 a0 = x4[0];
        const int4 a1 = x4[1];
        int idot = 0;
        idot = ggml_cuda_dp4a(w0[k].x, a0.x, idot); idot = ggml_cuda_dp4a(w0[k].y, a0.y, idot);
        idot = ggml_cuda_dp4a(w0[k].z, a0.z, idot); idot = ggml_cuda_dp4a(w0[k].w, a0.w, idot);
        idot = ggml_cuda_dp4a(w1[k].x, a1.x, idot); idot = ggml_cuda_dp4a(w1[k].y, a1.y, idot);
        idot = ggml_cuda_dp4a(w1[k].z, a1.z, idot); idot = ggml_cuda_dp4a(w1[k].w, a1.w, idot);
        part[item] = __half2float(*reinterpret_cast<const __half *>(&db[k])) * __low2float(xb->ds) * (float) idot;
    }
    __syncthreads();

    if (t < ROWS) {
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < HC_UP_NB; j++) {
            sum += part[t * HC_UP_NB + j];
        }
        gate[t] = sum;
    }
    __syncthreads();

    if (t < HC_UP_D) {
        const int64_t i0 = d0 + t;
        float sum = 0.0f;
        for (int ih = 0; ih < HC_UP_HC; ++ih) {
            const float xv = xn[i0 + ih*sx1];
            const float wv = 1.0f / (1.0f + expf(-gate[ih*HC_UP_D + t]));
            sum += xv * wv;
        }
        const float v = pre_scale * sum;
        dst[i0] = v;
        if (yq != nullptr) {
            float amax = fabsf(v);
            float bsum = v;
            amax = warp_reduce_max<QK8_1>(amax);
            bsum = warp_reduce_sum<QK8_1>(bsum);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
            yq[i0 / QK8_1].qs[i0 % QK8_1] = q;
            if (i0 % QK8_1 == 0) {
                yq[i0 / QK8_1].ds = make_half2(d, bsum);
            }
        }
    }
#else
    GGML_UNUSED_VARS(lo, scale, bias, wbase, ne1, xn, sx1, pre_scale, dst, yq, n_embd, sx2);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

bool ggml_cuda_hc_up_pre_ok(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * up, const ggml_tensor * pre) {
    static const bool disabled = getenv("GGML_CUDA_NO_HC_UP_PRE") != nullptr;
    const ggml_tensor * w  = up->src[0];
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * xn = pre->src[0];
    if (disabled || !w->buffer || !ggml_backend_buft_is_cuda_repack(w->buffer->buft) || w->type != GGML_TYPE_Q8_0 ||
            w->ne[0] != HC_UP_K || w->ne[2] != 1 || w->ne[3] != 1 || up->src[1] != silu) {
        return false;
    }
    const bool gated = ggml_get_op_params_i32(pre, 1) != 0;
    const int64_t n_embd = pre->ne[0];
    const int64_t n_tok = pre->ne[1]; // one block row per token (speculative verify, several sequences)
    return gated && n_tok >= 1 && n_tok <= 16 && lo->type == GGML_TYPE_F32 && ggml_is_contiguous(lo) && ggml_nelements(lo) == HC_UP_K*n_tok &&
        up->type == GGML_TYPE_F32 && ggml_nelements(up) == w->ne[1]*n_tok && xn->type == GGML_TYPE_F32 &&
        xn->ne[0] == n_embd && xn->ne[1] == HC_UP_HC && xn->ne[2] == n_tok && xn->ne[3] == 1 && xn->nb[0] == sizeof(float) &&
        w->ne[1] == HC_UP_HC*n_embd && n_embd % HC_UP_D == 0 && pre->ne[2] == 1 && pre->ne[3] == 1 &&
        pre->type == GGML_TYPE_F32 && ggml_is_contiguous(pre) && pre->src[1]->ne[1] == HC_UP_HC;
}

void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * up,
        ggml_tensor * pre, void * yq) {
    float s;
    float b;
    memcpy(&s, (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) scale->op_params + 1, sizeof(float));
    const float pre_scale = ggml_get_op_params_f32(pre, 0);

    const ggml_tensor * xn = pre->src[0];
    const int64_t n_embd = pre->ne[0];
    hc_up_pre_f32<<<dim3(n_embd / HC_UP_D, pre->ne[1], 1), 256, 0, ctx.stream()>>>(
        (const float *) scale->src[0]->data, s, b, (const uint8_t *) up->src[0]->data, (uint32_t) up->src[0]->ne[1],
        (const float *) xn->data, xn->nb[1] / sizeof(float), pre_scale, (float *) pre->data, (block_q8_1 *) yq, n_embd,
        xn->nb[2] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}
