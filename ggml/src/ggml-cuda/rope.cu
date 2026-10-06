#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T *            x,
                                  T *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const bool           inplace) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T *            x,
                            T *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, inplace);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_multi_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_multi_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                     s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                     corr_dims, freq_factors, sections, is_imrope, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// the f32 result rounded to f32 before the store converts it: an f16 store would otherwise let the compiler fuse
// the last multiply-add into a single rounding to f16 (v_fma_mixlo_f16), unlike ROPE followed by SET_ROWS
template <typename D>
static __device__ __forceinline__ D rope_store_cast(float v) {
#if defined(GGML_USE_HIP)
    if constexpr (!std::is_same_v<D, float>) {
        asm volatile("" : "+v"(v));
    }
#endif // defined(GGML_USE_HIP)
    return ggml_cuda_cast<D>(v);
}

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const int mrope, const mrope_sections sections) {
    ggml_cuda_pdl_lc();
    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    x += sample*s03 + channel*s02 + row*s01;

    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp/ncols + eps);

    int64_t idst = sample*s3 + channel*s2 + row*s1;
    if (set_rows_stride != 0) {
        idst = row*s1 + row_indices[channel]*set_rows_stride;
    }
    dst += idst;

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (is_neox && i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];

        if (i0 >= n_dims) {
            dst[ix0] = rope_store_cast<D>(x0);
            dst[ix1] = rope_store_cast<D>(x1);
            continue;
        }

        // mrope: 1 = M-RoPE, 2 = interleaved M-RoPE (the section choice of rope_multi; NEOX pairing)
        float theta_base;
        if (mrope == 0) {
            theta_base = pos[channel]*powf(theta_scale, i0/2.0f);
        } else {
            const int nch       = gridDim.y;
            const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
            const int sec_w     = sections.v[1] + sections.v[0];
            const int sector    = (i0 / 2) % sect_dims;
            int p = 3;
            if (mrope == 2) {
                if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
                    p = 1;
                } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
                    p = 2;
                } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
                    p = 0;
                }
            } else {
                p = sector < sections.v[0] ? 0 : sector < sec_w ? 1 : sector < sec_w + sections.v[2] ? 2 : 3;
            }
            theta_base = pos[channel + nch * p] * powf(theta_scale, i0 / 2.0f);
        }
        const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

        dst[ix0] = rope_store_cast<D>(x0*cos_theta - x1*sin_theta);
        dst[ix1] = rope_store_cast<D>(x0*sin_theta + x1*cos_theta);
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const int mrope, const mrope_sections sections, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, nchannels, nsamples);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, mrope, sections);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, mrope, sections);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, mrope, sections);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, mrope, sections);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

    const bool is_mrope = (mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION;
    const bool is_neox  = (mode & GGML_ROPE_TYPE_NEOX) || is_mrope; // M-RoPE rotates NEOX pairs
    const int  mrope    = !is_mrope ? 0 : mode == GGML_ROPE_TYPE_IMROPE ? 2 : 1;
    mrope_sections sections;
    memcpy(&sections.v, (const int32_t *) rope->op_params + 11, sizeof(int)*4);

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, mrope, sections, stream);
    } else if (dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, mrope, sections, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}

// QSA indexer key pooling: gather r member rows per block, mean, rms_norm * w, mrope
// half a wave per pooled row (two rows per wave), D = 128 channels, 4 per lane. the arithmetic matches
// get_rows -> add -> scale -> rms_norm_f32<256> -> mul -> rope_multi exactly: rms_norm_f32 sums each half
// of the row with a 64-lane butterfly, which is a balanced tree in channel order, rebuilt here from the
// in-lane pairs up
template <typename T> struct qsa_quad;
template <> struct qsa_quad<half>  { using type = uint2;  };
template <> struct qsa_quad<float> { using type = float4; };

static __device__ __forceinline__ void qsa_quad_load(const uint2 q, float v[4]) {
    const half2 lo = *reinterpret_cast<const half2 *>(&q.x);
    const half2 hi = *reinterpret_cast<const half2 *>(&q.y);
    v[0] = __low2float(lo); v[1] = __high2float(lo);
    v[2] = __low2float(hi); v[3] = __high2float(hi);
}

static __device__ __forceinline__ void qsa_quad_load(const float4 q, float v[4]) {
    v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
}

// R > 0 fixes the member count at compile time so all member rows load at once; R == 0 reads r
template <typename T, int R>
static __global__ void __launch_bounds__(256) qsa_pool_norm_rope_f32(
        const T * __restrict__ src, const int32_t * __restrict__ idx, const float * __restrict__ w,
        const int32_t * __restrict__ pos, float * __restrict__ dst,
        const int r, const int n_blocks, const int n_rows, const int64_t nb1, const int64_t nb2, const int64_t s_idx,
        const float scale, const float bias, const float eps,
        const int n_dims, const float theta_scale, const float freq_scale, const float ext_factor,
        const float attn_factor, const rope_corr_dims corr_dims, const mrope_sections sections, const bool is_imrope) {
    constexpr int D = 128;
    using quad_t = typename qsa_quad<T>::type;

    __shared__ float xs[8][D];
    __shared__ float cs[8][D/2];
    __shared__ float sn[8][D/2];

    const int slot  = threadIdx.x / 32;      // row slot in the block
    const int l     = threadIdx.x % 32;      // lane in the half wave
    const int hbase = threadIdx.x % 64 - l;  // first lane of this half in the wave

    const int half_dims = n_dims/2;
    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w     = sections.v[1] + sections.v[0];

    // rotated pair c = l: its frequency and which position row it reads do not depend on the row
    const int   c      = l;
    const float pw     = powf(theta_scale, (2*c) / 2.0f);
    const int   sector = c % sect_dims;
    int psel = 3;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
            psel = 1;
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
            psel = 2;
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
            psel = 0;
        }
    } else {
        if (sector < sections.v[0]) {
            psel = 0;
        } else if (sector < sec_w) {
            psel = 1;
        } else if (sector < sec_w + sections.v[2]) {
            psel = 2;
        }
    }

    float wl[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        wl[k] = w[4*l + k];
    }

    for (int base = blockIdx.x*8; base < n_rows; base += gridDim.x*8) {
        const int  row = base + slot;
        const bool ok  = row < n_rows;
        const int  rc  = ok ? row : n_rows - 1; // past the end, load a valid row and discard it

        // no branch around the loads, so the position and member index loads overlap
        const int s  = rc / n_blocks;
        const int bk = rc % n_blocks;
        const int pos_v   = pos[rc + n_rows*psel];
        const int my_cell = idx[s*s_idx + (int64_t) bk*r + min(l, r - 1)];

        float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        {
            const char * sbase = (const char *) src + s*nb2;

            if constexpr (R > 0) {
                quad_t q[R];
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    const int cell = __shfl(my_cell, hbase + i, 64);
                    q[i] = ((const quad_t *) (sbase + (int64_t) cell*nb1))[l];
                }
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    float m[4];
                    qsa_quad_load(q[i], m);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        v[k] = i == 0 ? m[k] : v[k] + m[k];
                    }
                }
            } else {
                for (int i = 0; i < r; ++i) {
                    const int cell = __shfl(my_cell, hbase + i, 64);
                    float m[4];
                    qsa_quad_load(((const quad_t *) (sbase + (int64_t) cell*nb1))[l], m);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        v[k] = i == 0 ? m[k] : v[k] + m[k];
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                v[k] = scale*v[k] + bias;
            }
        }

        // butterfly levels 1-2 inside the lane, 3-6 across the 16 lanes of each half row.
        // rms_norm rounds each square before adding, so keep these from contracting into fma
        float t = (__fmul_rn(v[0], v[0]) + __fmul_rn(v[1], v[1])) + (__fmul_rn(v[2], v[2]) + __fmul_rn(v[3], v[3]));
        t += __shfl_xor(t, 1, 64);
        t += __shfl_xor(t, 2, 64);
        t += __shfl_xor(t, 4, 64);
        t += __shfl_xor(t, 8, 64);
        const float sum = __shfl(t, hbase, 64) + __shfl(t, hbase + 16, 64);
        const float rs  = rsqrtf(sum/D + eps);

#pragma unroll
        for (int k = 0; k < 4; ++k) {
            xs[slot][4*l + k] = rs*v[k]*wl[k];
        }

        if (ok && c < half_dims) {
            const float theta_base = pos_v * pw;

            float cos_theta;
            float sin_theta;
            rope_yarn<true>(theta_base, freq_scale, corr_dims, 2*c, ext_factor, attn_factor, cos_theta, sin_theta);

            cs[slot][c] = cos_theta;
            sn[slot][c] = sin_theta;
        }
        __syncthreads();

        if (ok) {
            float out[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e = 4*l + k;
                if (e >= n_dims) {
                    out[k] = xs[slot][e];
                } else {
                    const int   ce = e % half_dims;
                    const float x0 = xs[slot][ce];
                    const float x1 = xs[slot][ce + half_dims];
                    out[k] = e < half_dims ? x0*cs[slot][ce] - x1*sn[slot][ce] : x0*sn[slot][ce] + x1*cs[slot][ce];
                }
            }
            *(float4 *) (dst + (int64_t) row*D + 4*l) = make_float4(out[0], out[1], out[2], out[3]);
        }
        __syncthreads();
    }
}

bool ggml_cuda_qsa_pool_ok(const ggml_tensor * get_rows, const ggml_tensor * scale, const ggml_tensor * rms_norm,
        const ggml_tensor * mul, const ggml_tensor * rope) {
    const ggml_tensor * k   = get_rows->src[0];
    const ggml_tensor * idx = get_rows->src[1];
    const ggml_tensor * w   = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    const int mode   = ((const int32_t *) rope->op_params)[2];
    const int n_offs = ((const int32_t *) rope->op_params)[15];

    const size_t pair = 4*ggml_type_size(k->type);

    return (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) && k->ne[0] == 128 && k->nb[0] == ggml_type_size(k->type) &&
        k->nb[1] % pair == 0 && k->nb[2] % pair == 0 && ((uintptr_t) k->data) % pair == 0 &&
        k->ne[3] == 1 && idx->type == GGML_TYPE_I32 && ggml_is_contiguous(idx) && idx->ne[2] == 1 && idx->ne[3] == 1 &&
        idx->ne[1] == k->ne[2] && w->type == GGML_TYPE_F32 && ggml_nelements(w) == 128 && ggml_is_contiguous(w) &&
        rope->type == GGML_TYPE_F32 && ggml_is_contiguous(rope) && rope->ne[0] == 128 && rope->ne[1] == 1 &&
        rope->src[2] == nullptr && (mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION &&
        n_offs == 0 && n_dims > 0 && n_dims <= 128 && n_dims % 2 == 0 &&
        ggml_nelements(scale) == ggml_nelements(rope) && ggml_nelements(rms_norm) == ggml_nelements(rope);
}

void ggml_cuda_op_qsa_pool_norm_rope(ggml_backend_cuda_context & ctx, const ggml_tensor * get_rows, int r,
        const ggml_tensor * scale, const ggml_tensor * rms_norm, const ggml_tensor * mul, ggml_tensor * rope) {
    const ggml_tensor * k   = get_rows->src[0];
    const ggml_tensor * idx = get_rows->src[1];
    const ggml_tensor * w   = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float sc;
    float sb;
    float eps;
    memcpy(&sc,  (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&sb,  (const float *) scale->op_params + 1, sizeof(float));
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    mrope_sections sections;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (const int32_t *) rope->op_params + 11, sizeof(int)*4);

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    const int n_stream = (int) k->ne[2];
    const int n_blocks = (int) (idx->ne[0] / r);
    const int n_rows   = n_blocks*n_stream;
    GGML_ASSERT(rope->ne[2] == n_rows);

    const int64_t s_idx = idx->nb[1] / sizeof(int32_t);

    const dim3 grid(std::min((n_rows + 7)/8, 4096), 1, 1);
    const dim3 block(256, 1, 1);
    cudaStream_t stream = ctx.stream();

    const bool imrope = mode == GGML_ROPE_TYPE_IMROPE;
    auto launch = [&](auto type_tag, auto r_c) {
        using T = decltype(type_tag);
        constexpr int R = decltype(r_c)::value;
        qsa_pool_norm_rope_f32<T, R><<<grid, block, 0, stream>>>((const T *) k->data, (const int32_t *) idx->data,
            (const float *) w->data, (const int32_t *) rope->src[1]->data, (float *) rope->data,
            r, n_blocks, n_rows, k->nb[1], k->nb[2], s_idx, sc, sb, eps,
            n_dims, theta_scale, freq_scale, ext_factor, attn_factor, corr_dims, sections, imrope);
    };
    auto launch_r = [&](auto type_tag) {
        switch (r) {
            case 2:  launch(type_tag, std::integral_constant<int, 2>{}); break;
            case 4:  launch(type_tag, std::integral_constant<int, 4>{}); break;
            case 8:  launch(type_tag, std::integral_constant<int, 8>{}); break;
            default: launch(type_tag, std::integral_constant<int, 0>{}); break;
        }
    };
    if (k->type == GGML_TYPE_F16) {
        launch_r(half());
    } else {
        launch_r(float());
    }
    CUDA_CHECK(cudaGetLastError());
}

// QSA indexer block scores for one query: relu(k_b . q_h) summed over heads, plus the block bias
// one wave per block; D = 128 channels, 2 per lane
template <int H>
static __global__ void qsa_score_blocks_f32(
        const float * __restrict__ keys, const float * __restrict__ q, const float * __restrict__ bias,
        float * __restrict__ score, const int n_blocks, const int64_t s_key, const int64_t s_q) {
    const int lane = threadIdx.x % 64;
    const int b    = blockIdx.x*(blockDim.x/64) + threadIdx.x/64;
    if (b >= n_blocks) {
        return;
    }

    // same pairing and order as mul_mat_vec_f: adjacent channels per lane, then a wave sum
    const float2 k = ((const float2 *) (keys + b*s_key))[lane];

    float total = 0.0f;
#pragma unroll
    for (int h = 0; h < H; ++h) {
        const float2 qh = ((const float2 *) (q + h*s_q))[lane];
        float d = 0.0f;
        ggml_cuda_mad(d, k.x, qh.x);
        ggml_cuda_mad(d, k.y, qh.y);
        d = fmaxf(warp_reduce_sum<64>(d), 0);
        total = h == 0 ? d : total + d;
    }
    if (lane == 0) {
        score[b] = total + bias[b];
    }
}

template <typename mask_t>
static __global__ void qsa_score_expand_f32(
        const float * __restrict__ score, const int32_t * __restrict__ cell_blk, const mask_t * __restrict__ mask,
        float * __restrict__ dst, const int n_kv) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_kv) {
        return;
    }
    dst[c] = score[cell_blk[c]] + ggml_cuda_cast<float>(mask[c]);
}

void ggml_cuda_op_qsa_score(ggml_backend_cuda_context & ctx, const ggml_tensor * keys, const ggml_tensor * q, int n_head,
        const ggml_tensor * bias, const ggml_tensor * cell_blk, const ggml_tensor * mask, ggml_tensor * dst) {
    const int n_blocks = (int) keys->ne[1];
    const int n_kv     = (int) dst->ne[0];

    // no cells: dst takes the block scores themselves
    const bool blocks_only = cell_blk == nullptr;

    ggml_cuda_pool_alloc<float> score_tmp(ctx.pool(), blocks_only ? 1 : n_blocks);
    float * score_ptr = blocks_only ? (float *) dst->data : score_tmp.get();
    struct { float * p; float * get() const { return p; } } score = { score_ptr };
    cudaStream_t stream = ctx.stream();

    const int64_t s_key = keys->nb[1]/sizeof(float);
    const int64_t s_q   = q->nb[1]/sizeof(float);

    const dim3 grid_b((n_blocks + 3)/4, 1, 1);
    switch (n_head) {
        case 1: qsa_score_blocks_f32<1><<<grid_b, 256, 0, stream>>>((const float *) keys->data, (const float *) q->data, (const float *) bias->data, score.get(), n_blocks, s_key, s_q); break;
        case 2: qsa_score_blocks_f32<2><<<grid_b, 256, 0, stream>>>((const float *) keys->data, (const float *) q->data, (const float *) bias->data, score.get(), n_blocks, s_key, s_q); break;
        case 4: qsa_score_blocks_f32<4><<<grid_b, 256, 0, stream>>>((const float *) keys->data, (const float *) q->data, (const float *) bias->data, score.get(), n_blocks, s_key, s_q); break;
        case 8: qsa_score_blocks_f32<8><<<grid_b, 256, 0, stream>>>((const float *) keys->data, (const float *) q->data, (const float *) bias->data, score.get(), n_blocks, s_key, s_q); break;
        default: GGML_ABORT("qsa_score: unsupported head count");
    }

    if (blocks_only) {
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 grid_c((n_kv + 255)/256, 1, 1);
    if (mask->type == GGML_TYPE_F16) {
        qsa_score_expand_f32<half><<<grid_c, 256, 0, stream>>>(score.get(), (const int32_t *) cell_blk->data, (const half *) mask->data, (float *) dst->data, n_kv);
    } else {
        qsa_score_expand_f32<float><<<grid_c, 256, 0, stream>>>(score.get(), (const int32_t *) cell_blk->data, (const float *) mask->data, (float *) dst->data, n_kv);
    }
    CUDA_CHECK(cudaGetLastError());
}
