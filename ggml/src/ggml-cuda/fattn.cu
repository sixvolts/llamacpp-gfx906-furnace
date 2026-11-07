#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int ncols1, const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / WARP_SIZE;
    const int lane     = tid % WARP_SIZE;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        uint32_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            bool selected = false;
            for (int q = 0; q < q1 - q0 && !selected; ++q) {
                selected = i < ne30 && isfinite(__half2float(mask[q*s31 + i]));
            }
            selected_warp[item] = __ballot_sync(0xFFFFFFFF, selected);
            warp_count += __popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/WARP_SIZE; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const uint32_t lane_mask = lane == 0 ? 0 : (1u << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*WARP_SIZE + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + __popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (uint32_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += __popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, ncols1, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    const int64_t n_gather = (ncols1 == 1 ? Q->ne[1] : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

#if defined(GGML_USE_HIP)
// sparse decode without an index-aware kernel: list the finite mask columns, gather those K/V rows
// and mask values into dense buffers, then attend over n_sel keys instead of the whole cache
// list the finite mask columns in ascending order: blocks of 256 threads x 8 columns count theirs, then
// each block writes its columns after the counts of the blocks before it
static constexpr int FATTN_GATHER_CHUNK = 256*8;

static __device__ __forceinline__ int fattn_gather_flags(const half * mask, const int n_kv, const int c0) {
    int flags = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int i = c0 + k;
        flags |= (i < n_kv && !isinf(__half2float(mask[i]))) << k;
    }
    return flags;
}

// exclusive prefix of v over the block, and the block total
static __device__ __forceinline__ int fattn_gather_scan(const int v, int & total, int * wsum) {
    const int lane = threadIdx.x % 64;
    const int w    = threadIdx.x / 64;
    int incl = v;
#pragma unroll
    for (int off = 1; off < 64; off *= 2) {
        const int t = __shfl_up(incl, off, 64);
        incl += lane >= off ? t : 0;
    }
    if (lane == 63) {
        wsum[w] = incl;
    }
    __syncthreads();
    int base = 0;
    total = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        base  += k < w ? wsum[k] : 0;
        total += wsum[k];
    }
    return base + incl - v;
}

// blockIdx.y selects the mask row (a query); counts hold gridDim.x entries per row and idx n_sel per row
// blockIdx.y = stream*nrs + row (nrs rows per KV stream, mask_s3 halves between the streams' masks)
static __global__ void __launch_bounds__(256) fattn_gather_count(const half * __restrict__ mask, const int n_kv, int * __restrict__ counts,
        const int64_t mask_row, const int nrs, const int64_t mask_s3) {
    __shared__ int wsum[4];
    mask   += (blockIdx.y/nrs)*mask_s3 + (blockIdx.y%nrs)*mask_row;
    counts += blockIdx.y*gridDim.x;
    const int flags = fattn_gather_flags(mask, n_kv, blockIdx.x*FATTN_GATHER_CHUNK + threadIdx.x*8);
    int total;
    fattn_gather_scan(__popc(flags), total, wsum);
    if (threadIdx.x == 0) {
        counts[blockIdx.x] = total;
    }
}

static __global__ void __launch_bounds__(256) fattn_gather_compact(
        const half * __restrict__ mask, const int n_kv, const int * __restrict__ counts, int32_t * __restrict__ idx, const int n_sel,
        const int64_t mask_row, const int nrs, const int64_t mask_s3) {
    __shared__ int wsum[4];
    mask   += (blockIdx.y/nrs)*mask_s3 + (blockIdx.y%nrs)*mask_row;
    counts += blockIdx.y*gridDim.x;
    idx    += blockIdx.y*n_sel;
    int before = 0;
    int all    = 0;
    for (int k = 0; k < (int) gridDim.x; ++k) {
        const int n = counts[k];
        before += k < (int) blockIdx.x ? n : 0;
        all    += n;
    }

    const int c0    = blockIdx.x*FATTN_GATHER_CHUNK + threadIdx.x*8;
    const int flags = fattn_gather_flags(mask, n_kv, c0);
    int total;
    int dst = before + fattn_gather_scan(__popc(flags), total, wsum);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if ((flags >> k) & 1) {
            if (dst < n_sel) {
                idx[dst] = c0 + k;
            }
            dst++;
        }
    }

    for (int j = min(all, n_sel) + blockIdx.x*256 + threadIdx.x; j < n_sel; j += gridDim.x*256) {
        idx[j] = -1;
    }
}

// one wave per selected column: the K and V rows of every head, plus the mask value of every mask row
static __global__ void fattn_gather_rows(
        const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask, const int32_t * __restrict__ idx,
        char * __restrict__ Kg, char * __restrict__ Vg, half * __restrict__ maskg,
        const int n_head, const int row_k, const int row_v,
        const int64_t nb11, const int64_t nb12, const int64_t nb21, const int64_t nb22,
        const int64_t s31, const int n_mask_rows, const int n_sel) {
    const int lane = threadIdx.x % 64;
    const int j    = blockIdx.x*(blockDim.x/64) + threadIdx.x/64;
    if (j >= n_sel) {
        return;
    }
    const int c = idx[j];

    const int nk = row_k/16;
    const int nv = row_v/16;
    for (int t = lane; t < n_head*(nk + nv); t += 64) {
        const int  h    = t / (nk + nv);
        const int  o    = t % (nk + nv);
        const bool is_k = o < nk;
        const int  q    = is_k ? o : o - nk;

        int4 * d = (int4 *) (is_k ? Kg + ((int64_t) j*n_head + h)*row_k : Vg + ((int64_t) j*n_head + h)*row_v);
        const int4 * sp = (const int4 *) (is_k ? K + (int64_t) c*nb11 + h*nb12 : V + (int64_t) c*nb21 + h*nb22);
        d[q] = c >= 0 ? sp[q] : make_int4(0, 0, 0, 0);
    }
    for (int q = lane; q < n_mask_rows; q += 64) {
        maskg[(int64_t) q*n_sel + j] = c >= 0 ? mask[q*s31 + c] : __float2half(-INFINITY);
    }
}

static bool ggml_cuda_flash_attn_ext_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Up to FATTN_DEC_MAX_Q queries (one per mask row), f16 K/V, for the sparse-attention ops (n_kv_max hint): a block scores FATTN_DEC_C keys
// against every query head of one KV head (lanes = keys, waves = slices of the heads), turns the scores
// into chunk-local softmax weights and accumulates the V rows (threads = dims). fattn_dec_combine then
// merges the chunks. Keys are read through idx (the finite mask columns) or are the cells themselves.
static constexpr int FATTN_DEC_C     = 64;
static constexpr int FATTN_DEC_MAX_C = 128; // chunks the combine kernel takes
static constexpr int FATTN_DEC_MAX_Q = 16;  // query rows per launch (verify batches, decoding sequences)
static constexpr int FATTN_DEC_PF_ROWS = 64; // prompt rows per launch on the sparse prompt path (bounds the partials)

template <int D, int G>
static __global__ void __launch_bounds__(256) fattn_dec_chunk(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        const int32_t * __restrict__ idx, const int n_keys, const float scale,
        const int64_t q_sh, const int64_t nb11, const int64_t nb12, const int64_t nb21, const int64_t nb22,
        float * __restrict__ part_o, float2 * __restrict__ part_ml,
        const int64_t q_row, const int64_t mask_row, const int n_sel,
        const int nrs, const int64_t q_s3, const int64_t k_s3, const int64_t v_s3, const int64_t mask_s3) {
    constexpr int C  = FATTN_DEC_C; // keys per block, one per lane
    constexpr int DW = D/4;         // dims per wave in the scores

    __shared__ float4 qs[G][D/4];
    __shared__ float  sp[4][G][C];  // per-wave partial scores
    __shared__ float4 ps[G][C/4];
    __shared__ int    cells[C];

    const int chunk = blockIdx.x;
    const int g     = blockIdx.y;   // KV head
    const int n_kvh = gridDim.y;
    // blockIdx.z = stream*nrs + query: the stream's K/V, the query's own Q row, mask row, index list and partial buffers
    {
        const int iz = blockIdx.z;
        const int is = iz / nrs;
        const int iq = iz % nrs;
        Q    += is*q_s3 + iq*q_row;
        K    += is*k_s3;
        V    += is*v_s3;
        mask  = mask ? mask + is*mask_s3 + iq*mask_row : mask;
        idx   = idx  ? idx  + (int64_t) iz*n_sel : idx;
        part_o  += (int64_t) iz*gridDim.x*n_kvh*G*D;
        part_ml += (int64_t) iz*gridDim.x*n_kvh*G;
    }
    const int tid   = threadIdx.x;
    const int lane  = tid % 64;
    const int wave  = tid / 64;

    for (int i = tid; i < G*D/4; i += blockDim.x) {
        const int h  = i / (D/4);
        const int d4 = i % (D/4);
        const float4 q = ((const float4 *) (Q + (int64_t) (g*G + h)*q_sh))[d4];
        qs[h][d4] = make_float4(q.x*scale, q.y*scale, q.z*scale, q.w*scale);
    }
    if (tid < C) {
        const int j = chunk*C + tid;
        cells[tid] = j < n_keys ? (idx ? idx[j] : j) : -1;
    }
    __syncthreads();

    // partial scores: lane = key, wave = a quarter of the dims, all heads
    {
        const int cell = cells[lane];
        float s[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            s[h] = 0.0f;
        }
        const int4 * kr = (const int4 *) (K + (int64_t) max(cell, 0)*nb11 + g*nb12) + wave*(DW/8);
        int4 raw[DW/8];
#pragma unroll
        for (int c = 0; c < DW/8; ++c) {
            raw[c] = kr[c];
        }
#pragma unroll
        for (int c = 0; c < DW/8; ++c) {
            const half2 * k2 = (const half2 *) &raw[c];
            const float k0 = __low2float(k2[0]), k1 = __high2float(k2[0]);
            const float k2f = __low2float(k2[1]), k3 = __high2float(k2[1]);
            const float k4 = __low2float(k2[2]), k5 = __high2float(k2[2]);
            const float k6 = __low2float(k2[3]), k7 = __high2float(k2[3]);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = qs[h][wave*(DW/4) + 2*c + 0];
                const float4 qb = qs[h][wave*(DW/4) + 2*c + 1];
                s[h] += qa.x*k0 + qa.y*k1 + qa.z*k2f + qa.w*k3 + qb.x*k4 + qb.y*k5 + qb.z*k6 + qb.w*k7;
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            sp[wave][h][lane] = s[h];
        }
    }
    __syncthreads();

    // chunk-local softmax: wave w takes heads w, w + 4, ..., lanes over the keys
    for (int h = wave; h < G; h += 4) {
        const int cell = cells[lane];
        const float mk = cell >= 0 ? (mask ? __half2float(mask[cell]) : 0.0f) : -INFINITY;
        const float v  = cell >= 0 ? ((sp[0][h][lane] + sp[1][h][lane]) + (sp[2][h][lane] + sp[3][h][lane])) + mk : -INFINITY;
        const float m  = warp_reduce_max<64>(v);
        const float p  = v == -INFINITY ? 0.0f : expf(v - m);
        const float l  = warp_reduce_sum<64>(p);
        ((float *) ps[h])[lane] = p;
        if (lane == 0) {
            part_ml[((int64_t) chunk*n_kvh + g)*G + h] = make_float2(m, l);
        }
    }
    __syncthreads();

    // values: thread = dim, 16 V loads in flight (a masked key reads row 0 and has weight 0)
    for (int d = tid; d < D; d += blockDim.x) {
        float o[G];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            o[h] = 0.0f;
        }
        for (int j0 = 0; j0 < C; j0 += 16) {
            float v[16];
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                const int cell = max(cells[j0 + jj], 0);
                v[jj] = __half2float(((const half *) (V + (int64_t) cell*nb21 + g*nb22))[d]);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
#pragma unroll
                for (int q4 = 0; q4 < 4; ++q4) {
                    const float4 w = ps[h][j0/4 + q4];
                    o[h] += w.x*v[4*q4 + 0] + w.y*v[4*q4 + 1] + w.z*v[4*q4 + 2] + w.w*v[4*q4 + 3];
                }
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            part_o[(((int64_t) chunk*n_kvh + g)*G + h)*D + d] = o[h];
        }
    }
}

// merge the chunks of one query head; chunks with nothing visible (l == 0) drop out
template <int D>
static __global__ void __launch_bounds__(256) fattn_dec_combine(
        const float * __restrict__ part_o, const float2 * __restrict__ part_ml, float * __restrict__ dst,
        const int n_chunks, const int G) {
    __shared__ float fac[FATTN_DEC_MAX_C];
    __shared__ float red[4];

    const int hq    = blockIdx.x;   // query head
    const int n_kvh = gridDim.x / G;
    const int g     = hq / G;
    const int h     = hq % G;
    const int tid   = threadIdx.x;
    // blockIdx.y is the query
    part_o  += (int64_t) blockIdx.y*n_chunks*n_kvh*G*D;
    part_ml += (int64_t) blockIdx.y*n_chunks*n_kvh*G;
    dst     += (int64_t) blockIdx.y*gridDim.x*D;

    float2 ml = make_float2(-INFINITY, 0.0f);
    if (tid < n_chunks) {
        ml = part_ml[((int64_t) tid*n_kvh + g)*G + h];
    }
    float m = ml.y > 0.0f ? ml.x : -INFINITY;
    m = warp_reduce_max<64>(m);
    if (tid % 64 == 0) {
        red[tid / 64] = m;
    }
    __syncthreads();
    const float m_all = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();

    const float f = ml.y > 0.0f ? expf(ml.x - m_all) : 0.0f;
    if (tid < n_chunks) {
        fac[tid] = f;
    }
    float l = warp_reduce_sum<64>(ml.y * f);
    if (tid % 64 == 0) {
        red[tid / 64] = l;
    }
    __syncthreads();
    const float l_all = (red[0] + red[1]) + (red[2] + red[3]);
    const float inv_l = l_all > 0.0f ? 1.0f / l_all : 0.0f;

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            acc += part_o[(((int64_t) c*n_kvh + g)*G + h)*D + d] * fac[c];
        }
        dst[(int64_t) hq*D + d] = acc * inv_l;
    }
}

static bool ggml_cuda_flash_attn_ext_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_NO_FA_DEC") != nullptr || getenv("GGML_CUDA_NO_FA_GATHER") != nullptr;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (disabled || mask == nullptr || n_kv_max <= 0 || dst->src[4] != nullptr) {
        return false;
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t D = K->ne[0];
    const int     G = K->ne[2] > 0 ? (int) (Q->ne[2] / K->ne[2]) : 0;

    // up to FATTN_DEC_MAX_Q queries (speculative verify, several decoding sequences): each is a mask row.
    // A prompt ubatch takes the same path once the context is several times the selection: every query
    // row attends to its own n_kv_max keys, so the cost stays flat with depth, while the tile kernel runs
    // over all n_kv keys with the top-k mask (QSA prefill at 32K: ~90 ms per layer per 1024 tokens).
    // GGML_CUDA_NO_FA_SPARSE_PREFILL=1 disables it, GGML_CUDA_FA_SPARSE_PREFILL_MIN sets the n_kv/n_sel ratio (3).
    static const bool pf_disabled = getenv("GGML_CUDA_NO_FA_SPARSE_PREFILL") != nullptr;
    static const int  pf_min      = getenv("GGML_CUDA_FA_SPARSE_PREFILL_MIN") ? atoi(getenv("GGML_CUDA_FA_SPARSE_PREFILL_MIN")) : 3;
    const int n_q = (int) Q->ne[1];
    const bool prompt = n_q > FATTN_DEC_MAX_Q;
    if (prompt && (pf_disabled || K->ne[1] < (int64_t) pf_min*GGML_PAD((int64_t) n_kv_max, FATTN_KQ_STRIDE))) {
        return false;
    }
    // ne[3] are the KV streams (several sequences without a unified KV cache): one launch set per stream
    const int64_t n_stream = Q->ne[3];
    if (n_q < 1 || n_stream < 1 || K->ne[3] != n_stream || V->ne[3] != n_stream || mask->ne[2] != 1 ||
            (mask->ne[3] != 1 && mask->ne[3] != n_stream) || mask->ne[1] < n_q || Q->nb[1] % 16 != 0 || Q->nb[3] % 16 != 0 ||
            Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            max_bias != 0.0f || logit_softcap != 0.0f || K->ne[2] != V->ne[2] || V->ne[0] != D || Q->ne[0] != D ||
            (D != 128 && D != 256) || G*K->ne[2] != Q->ne[2] || (G != 8 && G != 12 && G != 16) ||
            Q->nb[0] != sizeof(float) || Q->nb[2] % 16 != 0 || ((uintptr_t) Q->data) % 16 != 0 || K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || mask->nb[0] != sizeof(half) ||
            K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || K->nb[3] % 16 != 0 || V->nb[3] % 2 != 0 || mask->ne[0] < K->ne[1]) {
        return false;
    }

    const int64_t n_sel  = GGML_PAD((int64_t) n_kv_max, FATTN_KQ_STRIDE);
    const bool    sparse = K->ne[1] >= 2*n_sel;
    const int64_t n_keys = sparse ? n_sel : K->ne[1];
    const int     n_ch   = (int) ((n_keys + FATTN_DEC_C - 1)/FATTN_DEC_C);
    // several KV streams (no unified cache) only once the context is several times the selection: below that the older
    // kernels stay faster (2 and 6 streams at 1K: -1...8 %, 2 streams at 4K: about -5 %).
    // GGML_CUDA_NO_FA_DEC_STREAMS=1 disables it, GGML_CUDA_FA_DEC_STREAMS_MIN sets the n_kv/n_sel ratio (4).
    static const bool st_disabled = getenv("GGML_CUDA_NO_FA_DEC_STREAMS") != nullptr;
    static const int  st_min      = getenv("GGML_CUDA_FA_DEC_STREAMS_MIN") ? atoi(getenv("GGML_CUDA_FA_DEC_STREAMS_MIN")) : 4;
    if (n_ch > FATTN_DEC_MAX_C || (n_stream > 1 && (st_disabled || K->ne[1] < (int64_t) st_min*n_sel))) {
        return false;
    }

    const int n_kvh = (int) K->ne[2];

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const int64_t mask_row = mask->nb[1]/sizeof(half);
    const int64_t q_row    = Q->nb[1]/sizeof(float);

    // a prompt ubatch goes through in blocks of rows (per stream) so the partial buffers stay small;
    // decoding queries of all streams go in one launch set (blockIdx = stream*n_q + query, dst rows are contiguous)
    const int n_rows = prompt ? FATTN_DEC_PF_ROWS : (int) n_stream*n_q;

    const int n_cnt = (int) ((K->ne[1] + FATTN_GATHER_CHUNK - 1)/FATTN_GATHER_CHUNK);
    ggml_cuda_pool_alloc<int32_t> idx(pool, sparse ? (size_t) n_rows*(n_sel + n_cnt) : 1);

    ggml_cuda_pool_alloc<float>  part_o (pool, (size_t) n_rows*n_ch*n_kvh*G*D);
    ggml_cuda_pool_alloc<float2> part_ml(pool, (size_t) n_rows*n_ch*n_kvh*G);

    auto launch = [&](auto d_c, auto g_c) {
        constexpr int DT = decltype(d_c)::value;
        constexpr int GT = decltype(g_c)::value;
        auto run = [&](const char * Q_s, const char * K_s, const char * V_s, const half * mask_r, float * dst_r, const int nz, const int nrs,
                       const int64_t q_s3, const int64_t k_s3, const int64_t v_s3, const int64_t mask_s3) {
            if (sparse) {
                int32_t * counts = idx.get() + (size_t) nz*n_sel;
                fattn_gather_count<<<dim3(n_cnt, nz, 1), 256, 0, stream>>>(mask_r, (int) K->ne[1], counts, mask_row, nrs, mask_s3);
                fattn_gather_compact<<<dim3(n_cnt, nz, 1), 256, 0, stream>>>(mask_r, (int) K->ne[1], counts, idx.get(), (int) n_sel, mask_row, nrs, mask_s3);
            }
            fattn_dec_chunk<DT, GT><<<dim3(n_ch, n_kvh, nz), 256, 0, stream>>>(
                (const float *) Q_s, K_s, V_s, mask_r,
                sparse ? idx.get() : nullptr, (int) n_keys, scale,
                Q->nb[2]/sizeof(float), K->nb[1], K->nb[2], V->nb[1], V->nb[2], part_o.get(), part_ml.get(),
                q_row, mask_row, (int) n_sel, nrs, q_s3, k_s3, v_s3, mask_s3);
            fattn_dec_combine<DT><<<dim3(n_kvh*GT, nz, 1), 256, 0, stream>>>(part_o.get(), part_ml.get(), dst_r, n_ch, GT);
        };
        if (!prompt) {
            run((const char *) Q->data, (const char *) K->data, (const char *) V->data, (const half *) mask->data, (float *) dst->data,
                (int) n_stream*n_q, n_q, Q->nb[3]/sizeof(float), K->nb[3], V->nb[3], mask->ne[3] == 1 ? 0 : mask->nb[3]/sizeof(half));
            return;
        }
        for (int64_t is = 0; is < n_stream; ++is)
        for (int r0 = 0; r0 < n_q; r0 += n_rows) {
            const int nr = std::min(n_rows, n_q - r0);
            const char * Q_s    = (const char *) Q->data + is*Q->nb[3] + (int64_t) r0*Q->nb[1];
            const char * K_s    = (const char *) K->data + is*K->nb[3];
            const char * V_s    = (const char *) V->data + is*V->nb[3];
            float      * dst_s  = (float *) ((char *) dst->data + is*dst->nb[3]) + (int64_t) r0*n_kvh*GT*DT;
            const half * mask_r = (const half *) ((const char *) mask->data + (is % mask->ne[3])*mask->nb[3]) + (int64_t) r0*mask_row;
            run(Q_s, K_s, V_s, mask_r, dst_s, nr, nr, 0, 0, 0, 0);
        }
    };
    auto launch_g = [&](auto d_c) {
        switch (G) {
            case  8: launch(d_c, std::integral_constant<int,  8>{}); break;
            case 12: launch(d_c, std::integral_constant<int, 12>{}); break;
            default: launch(d_c, std::integral_constant<int, 16>{}); break;
        }
    };
    if (D == 128) {
        launch_g(std::integral_constant<int, 128>{});
    } else {
        launch_g(std::integral_constant<int, 256>{});
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif // defined(GGML_USE_HIP)

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
#if defined(GGML_USE_HIP)
    if (ggml_cuda_flash_attn_ext_decode(ctx, dst) || ggml_cuda_flash_attn_ext_gather(ctx, dst)) {
        return;
    }
#endif // defined(GGML_USE_HIP)
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

#if defined(GGML_USE_HIP)
static bool ggml_cuda_flash_attn_ext_gather(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_NO_FA_GATHER") != nullptr;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (disabled || mask == nullptr || n_kv_max <= 0) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t n_sel = GGML_PAD((int64_t) n_kv_max, FATTN_KQ_STRIDE);

    // one query per sequence: its visible columns are exactly the finite mask entries of row 0
    if (Q->ne[1] != 1 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
            K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
            max_bias != 0.0f || logit_softcap != 0.0f || K->ne[2] != V->ne[2] ||
            K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half) || mask->nb[0] != sizeof(half) ||
            (K->ne[0]*sizeof(half)) % 16 != 0 || (V->ne[0]*sizeof(half)) % 16 != 0 ||
            K->nb[1] % 16 != 0 || K->nb[2] % 16 != 0 || V->nb[1] % 16 != 0 || V->nb[2] % 16 != 0 ||
            K->ne[1] < 2*n_sel || mask->ne[0] < K->ne[1]) {
        return false;
    }

    const int n_head = (int) K->ne[2];
    const int row_k  = (int) (K->ne[0]*sizeof(half));
    const int row_v  = (int) (V->ne[0]*sizeof(half));
    const int n_mrow = (int) mask->ne[1];

    ggml_cuda_pool & pool = ctx.pool();
    const int n_chunks = (int) ((K->ne[1] + FATTN_GATHER_CHUNK - 1)/FATTN_GATHER_CHUNK);

    ggml_cuda_pool_alloc<int32_t> idx(pool, n_sel + n_chunks);
    ggml_cuda_pool_alloc<char>    Kg(pool, n_sel*n_head*row_k);
    ggml_cuda_pool_alloc<char>    Vg(pool, n_sel*n_head*row_v);
    ggml_cuda_pool_alloc<half>    maskg(pool, n_sel*n_mrow);

    cudaStream_t stream = ctx.stream();

    int32_t * counts = idx.get() + n_sel;
    fattn_gather_count<<<n_chunks, 256, 0, stream>>>((const half *) mask->data, (int) K->ne[1], counts, 0, 1, 0);
    fattn_gather_compact<<<n_chunks, 256, 0, stream>>>((const half *) mask->data, (int) K->ne[1], counts, idx.get(), (int) n_sel, 0, 1, 0);
    fattn_gather_rows<<<(n_sel + 3)/4, 256, 0, stream>>>((const char *) K->data, (const char *) V->data, (const half *) mask->data, idx.get(),
        Kg.get(), Vg.get(), maskg.get(), n_head, row_k, row_v, K->nb[1], K->nb[2], V->nb[1], V->nb[2],
        mask->nb[1]/sizeof(half), n_mrow, (int) n_sel);
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor Kt = *K;
    ggml_tensor Vt = *V;
    ggml_tensor Mt = *mask;

    Kt.data  = Kg.get();
    Kt.ne[1] = n_sel;
    Kt.nb[1] = (size_t) n_head*row_k;
    Kt.nb[2] = row_k;
    Kt.nb[3] = Kt.nb[1]*n_sel;

    Vt.data  = Vg.get();
    Vt.ne[1] = n_sel;
    Vt.nb[1] = (size_t) n_head*row_v;
    Vt.nb[2] = row_v;
    Vt.nb[3] = Vt.nb[1]*n_sel;

    Mt.data  = maskg.get();
    Mt.ne[0] = n_sel;
    Mt.nb[1] = n_sel*sizeof(half);
    Mt.nb[2] = Mt.nb[1]*n_mrow;
    Mt.nb[3] = Mt.nb[2];

    ggml_tensor d = *dst;
    d.src[1] = &Kt;
    d.src[2] = &Vt;
    d.src[3] = &Mt;
    ggml_set_op_params_i32(&d, 4, 0);

    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), &d)) {
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, &d);
            return true;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, &d);
            return true;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, &d);
            return true;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }
    GGML_ABORT("fatal error");
}
#endif // defined(GGML_USE_HIP)
