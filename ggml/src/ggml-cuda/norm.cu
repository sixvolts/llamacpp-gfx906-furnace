#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0)) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else {
            dst[col] = scale * x[col];
        }
    }
}

// q8_1 of one value per lane, 32 consecutive lanes = one block: the arithmetic of quantize_q8_1
static __device__ __forceinline__ void rms_emit_q8_1(block_q8_1 * yq, const int64_t idx, const float v) {
    float amax = fabsf(v);
    float sum  = v;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
    const int64_t ib  = idx / QK8_1;
    const int     iqs = idx % QK8_1;
    yq[ib].qs[iqs] = q;
    if (iqs == 0) {
        yq[ib].ds = make_half2(d, sum);
    }
}

// RMS_NORM -> MUL that also writes the result as q8_1 for a repacked matvec consumer
// (see ggml_cuda_repack_xq_emit_target); ncols % 32 == 0, contiguous dst
template <int block_size>
static __global__ void rms_norm_mul_q8_f32(const float * x, float * dst, const int ncols, const int64_t stride_row,
        const int64_t stride_channel, const int64_t stride_sample, const float eps, const float * mul,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed,
        const uint3 mul_nsamples_packed, block_q8_1 * yq) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    const int64_t dst_off = ((int64_t)(sample*nchannels + channel)*nrows + row)*ncols;
    dst += dst_off;

    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const int mul_col = fastmodulo(col, mul_ncols_packed);
        const float v = scale * x[col] * mul[mul_col];
        dst[col] = v;
        rms_emit_q8_1(yq, dst_off + col, v);
    }
}

// GDN output gate: RMS_NORM -> MUL(w) -> MUL(sigmoid(z)) in one kernel, plus q8_1 blocks when yq != nullptr.
// Same operation order as rms_norm_f32<256, true> and unary_gated_q8_kernel<op_sigmoid>, so results are identical.
static __global__ void __launch_bounds__(256) rms_norm_mul_sigmoid_gate_f32(
        const float * x, const int64_t sx1, const int64_t sx2, const int64_t sx3, const float * __restrict__ mul, const int64_t smul1,
        const float * z, const int64_t sz1, float * dst, const int ncols, const float eps, block_q8_1 * __restrict__ yq) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;
    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x += sample*sx3 + channel*sx2 + row*sx1;
    const int64_t lrow = ((int64_t) sample*nchannels + channel)*nrows + row;
    const float * zr = z + lrow*sz1;
    const float * mr = mul + row*smul1;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += 256) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += 256) {
        const float nv = scale * x[col] * mr[col];
        const float v  = (1.0f / (1.0f + expf(-zr[col]))) * nv;
        dst[lrow*ncols + col] = v;
        if (yq != nullptr) {
            rms_emit_q8_1(yq, lrow*ncols + col, v);
        }
    }
}

// b + a * sigmoid(g) exactly as sigmoid_mul_add_f32 computes it: no contraction anywhere, including the expf
// expansion (contract(off) changes its lowering too)
static __device__ __forceinline__ float hc_sma_combine(const float b, const float a, const float g) {
#pragma clang fp contract(off)
    const float s = 1.0f / (1.0f + expf(-g));
    return b + a * s;
}

// qwen4exp hc combine + the next hc mix norm, one block per (stream, token): w = s_out * sigmoid(s_in * inject),
// post = x * w + residual (dsv4_hc_post_f32), then RMS_NORM -> MUL -> q8_1 (rms_norm_mul_q8_f32<1024>). SMA: x is the
// shared-expert combine b + a * sigmoid(g[token]) (sigmoid_mul_add_f32). Each step keeps the operation order of the
// kernel it replaces, so the results are identical.
template <int HC, bool SMA>
static __global__ void __launch_bounds__(1024) hc_post_rms_norm_mul_q8_f32(
        const float * inject, const int64_t sinj, const float * x, const int64_t sx1, const float * sma_a, const float * sma_g,
        const float * residual, const int64_t sr1, const int64_t sr2, float * post, const float s_in, const float s_out,
        const float * __restrict__ mul, float * dst, const int ncols, const float eps, block_q8_1 * __restrict__ yq) {
    const int s   = blockIdx.x;
    const int it  = blockIdx.y;
    const int tid = threadIdx.x;

    const float   w  = s_out / (1.0f + expf(-s_in * inject[s + it * sinj]));
    const float * xr = x + it * sx1;
    const float * rr = residual + s * sr1 + it * sr2;
    const int64_t off = ((int64_t) it * HC + s) * ncols;
    float * pr = post + off;
    // post first (dsv4_hc_post_f32), each thread writes and re-reads only its own columns
    for (int col = tid; col < ncols; col += 1024) {
        const float xv = SMA ? hc_sma_combine(xr[col], sma_a[it * sx1 + col], sma_g[it]) : xr[col];
        float sum = xv * w;
        sum += rr[col];
        pr[col] = sum;
    }
    // then rms_norm_mul_q8_f32<1024> on the post row, in its source form
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += 1024) {
        const float xi = pr[col];
        tmp += xi * xi;
    }

    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, 1024>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    const float * mr = mul + (int64_t) s * ncols;
    for (int col = tid; col < ncols; col += 1024) {
        const float v = scale * pr[col] * mr[col];
        dst[off + col] = v;
        if (yq != nullptr) {
            rms_emit_q8_1(yq, off + col, v);
        }
    }
}

// residual ADD -> RMS_NORM -> MUL in one kernel: writes the sum (the residual stream, still read later)
// and the normed activation, plus q8_1 blocks when yq != nullptr. Each thread reads a, b at a column
// before writing sum there, and the block reduction separates the two passes, so sum may alias a or b
// and dst may alias a or b exactly. Same operation order as the three ops, so results are identical.
template <int block_size>
static __global__ void add_rms_norm_mul_f32(const float * a, const float * b, float * sum, float * dst,
        const int ncols, const float eps, const float * mul, block_q8_1 * yq) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t off = (int64_t) row * ncols;
    a += off; b += off; sum += off; dst += off;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a[col] + b[col];
        sum[col] = xi;
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float v = scale * sum[col] * mul[col];
        dst[col] = v;
        if (yq != nullptr) {
            rms_emit_q8_1(yq, off + col, v);
        }
    }
}

// RMS_NORM -> SCALE (the GDN l2 norm): same rounding as the two ops
template <int block_size>
static __global__ void rms_norm_scale_f32(const float * x, float * dst, const int ncols, const int64_t stride_row,
        const int64_t stride_channel, const int64_t stride_sample, const float eps, const float s, const float b) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = s * (scale * x[col]) + b;
    }
}

// two independent RMS_NORM -> SCALE pairs of the same shape in one launch (the GDN q and k l2 norms):
// blockIdx.z selects the pair (z / nsamples); the body is rms_norm_scale_f32's
template <int block_size>
static __global__ void rms_norm_scale2_f32(const float * x0, float * dst0, const float * x1, float * dst1, const int ncols,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const int nsamples,
        const float eps, const float s, const float b) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int pair      = blockIdx.z / nsamples;
    const int sample    = blockIdx.z % nsamples;
    const int tid       = threadIdx.x;

    const float * x   = pair == 0 ? x0 : x1;
    float *       dst = pair == 0 ? dst0 : dst1;
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = s * (scale * x[col]) + b;
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0));
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

bool ggml_cuda_op_rms_norm_fused_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * yq) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * mul_src = mul_tensor->src[0] == dst ? mul_tensor->src[1] : mul_tensor->src[0];
    const int64_t ne00 = src->ne[0];
    if (yq == nullptr || ne00 % QK8_1 != 0 || ne00 < 1024 || !ggml_is_contiguous(mul_tensor) ||
            !ggml_are_same_shape(mul_tensor, dst) || src->nb[0] != sizeof(float) || mul_src->nb[0] != sizeof(float) ||
            mul_src->type != GGML_TYPE_F32) {
        return false;
    }
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    const dim3 blocks_num(src->ne[1], src->ne[2], src->ne[3]);
    rms_norm_mul_q8_f32<1024><<<blocks_num, 1024, 32 * sizeof(float), ctx.stream()>>>(
        (const float *) src->data, (float *) mul_tensor->data, (int) ne00,
        src->nb[1] / sizeof(float), src->nb[2] / sizeof(float), src->nb[3] / sizeof(float), eps,
        (const float *) mul_src->data, mul_src->nb[1] / sizeof(float), mul_src->nb[2] / sizeof(float), mul_src->nb[3] / sizeof(float),
        init_fastdiv_values(mul_src->ne[0]), init_fastdiv_values(mul_src->ne[1]),
        init_fastdiv_values(mul_src->ne[2]), init_fastdiv_values(mul_src->ne[3]), (block_q8_1 *) yq);
    return true;
}

bool ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * norm,
        ggml_tensor * mul_tensor, void * yq) {
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * w = mul_tensor->src[0] == norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const int64_t ncols = add->ne[0];
    const int64_t nrows = ggml_nrows(add);
    if (add->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
            w->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || mul_tensor->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, add) || !ggml_are_same_shape(add, mul_tensor) ||
            !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(add) ||
            !ggml_is_contiguous(mul_tensor) || !ggml_is_contiguous(w) || ggml_nelements(w) != ncols ||
            norm->src[0] != add || ncols < 1024 || (yq != nullptr && ncols % QK8_1 != 0)) {
        return false;
    }
    // outputs may equal an input exactly (in place) but must not partially overlap one
    auto partial = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data; const char * x1 = x0 + ggml_nbytes(x);
        const char * y0 = (const char *) y->data; const char * y1 = y0 + ggml_nbytes(y);
        return x0 < y1 && y0 < x1 && !(x0 == y0 && x1 == y1);
    };
    const ggml_tensor * outs[2] = { add, mul_tensor };
    for (const ggml_tensor * o : outs) {
        if (partial(o, a) || partial(o, b) || partial(o, w)) {
            return false;
        }
    }
    const char * s0 = (const char *) add->data;        const char * s1 = s0 + ggml_nbytes(add);
    const char * d0 = (const char *) mul_tensor->data; const char * d1 = d0 + ggml_nbytes(mul_tensor);
    if (s0 < d1 && d0 < s1) {
        return false; // sum and dst both live after the kernel
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    add_rms_norm_mul_f32<1024><<<(int) nrows, 1024, 32 * sizeof(float), ctx.stream()>>>(
        (const float *) a->data, (const float *) b->data, (float *) add->data, (float *) mul_tensor->data,
        (int) ncols, eps, (const float *) w->data, (block_q8_1 *) yq);
    return true;
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

bool ggml_cuda_op_rms_norm_scale2(ggml_backend_cuda_context & ctx, ggml_tensor * n0, ggml_tensor * sc0, ggml_tensor * n1, ggml_tensor * sc1) {
    const ggml_tensor * x0 = n0->src[0];
    const ggml_tensor * x1 = n1->src[0];
    if (x0->type != GGML_TYPE_F32 || x1->type != GGML_TYPE_F32 || sc0->type != GGML_TYPE_F32 || sc1->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(sc0) || !ggml_is_contiguous(sc1) || !ggml_are_same_shape(x0, x1) ||
            !ggml_are_same_shape(sc0, sc1) || x0->nb[0] != sizeof(float) ||
            x0->nb[1] != x1->nb[1] || x0->nb[2] != x1->nb[2] || x0->nb[3] != x1->nb[3] || x0->ne[0] >= 1024 ||
            memcmp(n0->op_params, n1->op_params, sizeof(float)) != 0 || memcmp(sc0->op_params, sc1->op_params, 2 * sizeof(float)) != 0) {
        return false;
    }
    // the second pair must not read the first pair's output, and the outputs must not overlap
    const char * a0 = (const char *) sc0->data; const char * a1 = a0 + ggml_nbytes(sc0);
    const char * b0 = (const char *) sc1->data; const char * b1 = b0 + ggml_nbytes(sc1);
    const char * r0 = (const char *) x1->data;  const char * r1 = r0 + ggml_nbytes(x1);
    if ((a0 < b1 && b0 < a1) || (a0 < r1 && r0 < a1)) {
        return false;
    }
    float eps, sv, bv;
    memcpy(&eps, n0->op_params, sizeof(float));
    memcpy(&sv, (const float *) sc0->op_params + 0, sizeof(float));
    memcpy(&bv, (const float *) sc0->op_params + 1, sizeof(float));
    const int64_t ts = sizeof(float);
    const dim3 blocks_num(x0->ne[1], x0->ne[2], x0->ne[3] * 2);
    rms_norm_scale2_f32<256><<<blocks_num, 256, 32 * sizeof(float), ctx.stream()>>>(
        (const float *) x0->data, (float *) sc0->data, (const float *) x1->data, (float *) sc1->data, (int) x0->ne[0],
        x0->nb[1] / ts, x0->nb[2] / ts, x0->nb[3] / ts, (int) x0->ne[3], eps, sv, bv);
    return true;
}

void ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_node) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_node->type == GGML_TYPE_F32 && ggml_is_contiguous(scale_node));

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    float s, b;
    memcpy(&s, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) scale_node->op_params + 1, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    const dim3 blocks_num(ne01, ne02, ne03);
    const float * x = (const float *) src0->data;
    float * y = (float *) scale_node->data;
    if (ne00 < 1024) {
        rms_norm_scale_f32<256><<<blocks_num, 256, 32 * sizeof(float), stream>>>(x, y, ne00, s01, s02, s03, eps, s, b);
    } else {
        rms_norm_scale_f32<1024><<<blocks_num, 1024, 32 * sizeof(float), stream>>>(x, y, ne00, s01, s02, s03, eps, s, b);
    }
}

bool ggml_cuda_op_hc_post_rms_norm_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * inject, ggml_tensor * post,
        const float s_in, const float s_out, const ggml_tensor * norm, ggml_tensor * mul_tensor, void * yq,
        const ggml_tensor * sma_g, const ggml_tensor * sma_a, const ggml_tensor * sma_b, const bool dry) {
    const ggml_tensor * x   = sma_b != nullptr ? sma_b : post->src[0];
    const ggml_tensor * res = post->src[1];
    const ggml_tensor * w   = mul_tensor->src[0] == norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const int64_t n_embd = post->ne[0];
    const int64_t hc     = post->ne[1];
    const int64_t nt     = post->ne[2];
    if (post->src[3] != nullptr || norm->src[0] != post || hc != 4 || n_embd != 2560 || post->ne[3] != 1 ||
            inject->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || res->type != GGML_TYPE_F32 ||
            w->type != GGML_TYPE_F32 || post->type != GGML_TYPE_F32 || mul_tensor->type != GGML_TYPE_F32 ||
            inject->ne[0] != hc || inject->ne[1] != nt || inject->nb[0] != sizeof(float) ||
            x->ne[0] != n_embd || x->ne[1] != nt || x->nb[0] != sizeof(float) ||
            res->ne[0] != n_embd || res->ne[1] != hc || res->ne[2] != nt || res->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(post) || !ggml_is_contiguous(mul_tensor) || !ggml_are_same_shape(post, mul_tensor) ||
            !ggml_is_contiguous(w) || w->ne[0] != n_embd || w->ne[1] != hc || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (sma_b != nullptr && (sma_a->type != GGML_TYPE_F32 || sma_g->type != GGML_TYPE_F32 || !ggml_are_same_shape(sma_a, sma_b) ||
            sma_a->nb[0] != sizeof(float) || sma_a->nb[1] != sma_b->nb[1] || sma_g->ne[0] != 1 || sma_g->ne[1] != nt ||
            !ggml_is_contiguous(sma_g))) {
        return false;
    }
    auto lo  = [](const ggml_tensor * t) { return (const char *) t->data; };
    auto hi  = [](const ggml_tensor * t) { return (const char *) t->data + ggml_nbytes(t); };
    auto ovl = [&](const ggml_tensor * a, const ggml_tensor * b) { return lo(a) < hi(b) && lo(b) < hi(a); };
    // stream s of token t is one block: an output may sit exactly on the residual (the same threads read and write
    // each element) but not on x, inject or the shared-expert inputs, which every stream of the token reads
    const bool res_same = res->nb[1] == (size_t) n_embd * sizeof(float) && res->nb[2] == (size_t) hc * n_embd * sizeof(float);
    const ggml_tensor * shared_in[4] = { x, inject, sma_a, sma_g };
    const int n_shared = sma_b != nullptr ? 4 : 2;
    if (ovl(post, mul_tensor)) {
        return false;
    }
    // with the shared-expert combine the caller runs the inject matvec before this kernel, ahead of its place in the
    // graph: its output may then sit on a, b or g (dead by then in the graph order) and must not
    if (sma_b != nullptr && (ovl(inject, sma_a) || ovl(inject, sma_b) || ovl(inject, sma_g))) {
        return false;
    }
    for (const ggml_tensor * o : { (const ggml_tensor *) post, (const ggml_tensor *) mul_tensor }) {
        if (ovl(o, res) && !(lo(o) == lo(res) && res_same)) {
            return false;
        }
        for (int k = 0; k < n_shared; k++) {
            if (ovl(o, shared_in[k])) {
                return false;
            }
        }
    }
    if (dry) {
        return true;
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const bool sma = sma_b != nullptr;
    auto kernel = sma ? hc_post_rms_norm_mul_q8_f32<4, true> : hc_post_rms_norm_mul_q8_f32<4, false>;
    kernel<<<dim3(hc, nt), 1024, 0, ctx.stream()>>>(
        (const float *) inject->data, inject->nb[1] / sizeof(float), (const float *) x->data, x->nb[1] / sizeof(float),
        sma ? (const float *) sma_a->data : nullptr, sma ? (const float *) sma_g->data : nullptr,
        (const float *) res->data, res->nb[1] / sizeof(float), res->nb[2] / sizeof(float),
        (float *) post->data, s_in, s_out, (const float *) w->data, (float *) mul_tensor->data, (int) n_embd, eps, (block_q8_1 *) yq);
    return true;
}

bool ggml_cuda_op_rms_norm_mul_sigmoid_gate(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul_tensor,
        ggml_tensor * sig, ggml_tensor * gate_mul, void * yq) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul_tensor->src[0] == norm ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * z = sig->src[0];
    const int64_t ncols = x->ne[0];
    const bool w_bcast = ggml_nelements(w) == ncols;
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || z->type != GGML_TYPE_F32 || gate_mul->type != GGML_TYPE_F32 ||
            x->nb[0] != sizeof(float) || z->nb[0] != sizeof(float) || !ggml_is_contiguous(w) || ncols > 256 || ncols % QK8_1 != 0 ||
            !(w_bcast || (w->ne[0] == ncols && w->ne[1] == x->ne[1] && ggml_nelements(w) == ncols * x->ne[1])) ||
            !ggml_is_contiguous(gate_mul) || ggml_nelements(z) != ggml_nelements(gate_mul) || z->ne[0] != ncols ||
            z->nb[2] != z->ne[1] * z->nb[1] || z->nb[3] != z->ne[2] * z->nb[2] ||
            !ggml_are_same_shape(mul_tensor, x) || !ggml_are_same_shape(gate_mul, mul_tensor)) {
        return false;
    }
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    const dim3 blocks_num(x->ne[1], x->ne[2], x->ne[3]);
    rms_norm_mul_sigmoid_gate_f32<<<blocks_num, 256, 32 * sizeof(float), ctx.stream()>>>(
        (const float *) x->data, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), x->nb[3] / sizeof(float),
        (const float *) w->data, w_bcast ? 0 : ncols, (const float *) z->data, z->nb[1] / sizeof(float),
        (float *) gate_mul->data, (int) ncols, eps, (block_q8_1 *) yq);
    return true;
}
