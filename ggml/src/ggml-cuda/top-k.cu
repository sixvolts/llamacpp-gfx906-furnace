#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    // independent partial sums keep several histogram loads in flight
    const int * hrow = block_histograms + (size_t) row * blocks_per_row * NBINS + tid;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int row_block = 0;
    for (; row_block + 4 <= blocks_per_row; row_block += 4) {
        c0 += hrow[(row_block + 0) * NBINS];
        c1 += hrow[(row_block + 1) * NBINS];
        c2 += hrow[(row_block + 2) * NBINS];
        c3 += hrow[(row_block + 3) * NBINS];
    }
    for (; row_block < blocks_per_row; ++row_block) {
        c0 += hrow[row_block * NBINS];
    }
    const int count = (c0 + c1) + (c2 + c3);
    histogram[tid] = count;
    __syncthreads();

    // suffix sums over the bins, so the selected bin is found in parallel instead of by a serial scan:
    // it is the highest bin whose suffix reaches the rank (bin 0 if none does)
    static_assert(NBINS == BLOCK_SIZE, "one thread per bin");
    const top_k_radix_state state = states[row];
    for (int off = 1; off < NBINS; off *= 2) {
        const int add = tid + off < NBINS ? histogram[tid + off] : 0;
        __syncthreads();
        histogram[tid] += add;
        __syncthreads();
    }

    const int  above   = tid + 1 < NBINS ? histogram[tid + 1] : 0;
    const bool reach   = histogram[tid] >= state.rank;
    const bool reach_1 = above >= state.rank;
    if ((reach && !reach_1) || (tid == 0 && !reach)) {
        top_k_radix_state st = state;
        st.rank        -= above;
        st.prefix      |= (uint32_t) tid << shift;
        st.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = st;
    }
}

// The selected entries, placed deterministically (after rune's 904105df8 / e4aa7799f): an atomic slot counter wrote
// them in arrival order, so the ORDER of the list (and, under a tied k-th key, the SET) changed from run to run and
// every consumer that accumulates over it (the QSA sparse attention) rounded differently each run. Each block of a
// row owns one contiguous column chunk; pass 1 counts the chunk's entries above and equal to the k-th key, pass 2
// writes them at the sum of the earlier chunks' counts, in column order (warp ballots). The ties kept are the first
// `rank` in column order, as on the CPU.
static __device__ __forceinline__ int top_k_radix_chunk(int ncols, int blocks_per_row, int block_size) {
    return ((ncols + blocks_per_row - 1) / blocks_per_row + block_size - 1) / block_size * block_size;
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_count(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int2 * __restrict__ block_counts,
        int ncols,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const uint32_t prefix = states[row].prefix;
    const int chunk = top_k_radix_chunk(ncols, blocks_per_row, BLOCK_SIZE);
    const int col_end = min(ncols, (row_block + 1) * chunk);
    __shared__ int n_greater;
    __shared__ int n_equal;

    if (tid == 0) {
        n_greater = 0;
        n_equal = 0;
    }
    __syncthreads();

    int greater = 0;
    int equal = 0;
    for (int col = row_block * chunk + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        greater += key > prefix;
        equal += key == prefix;
    }
    atomicAdd(&n_greater, greater);   // integer: order-independent
    atomicAdd(&n_equal, equal);
    __syncthreads();

    if (tid == 0) {
        block_counts[blockIdx.x] = make_int2(n_greater, n_equal);
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int2 * __restrict__ block_counts,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp = tid / warpSize;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const uint32_t prefix = states[row].prefix;
    const int rank = states[row].rank;
    const int chunk = top_k_radix_chunk(ncols, blocks_per_row, BLOCK_SIZE);
    const int col_end = min(ncols, (row_block + 1) * chunk);
    __shared__ int warp_greater[32];
    __shared__ int warp_equal[32];

    int n_greater = 0;   // entries before this point of the row, block-uniform
    int n_equal = 0;
    for (int b = 0; b < row_block; ++b) {
        const int2 c = block_counts[row * blocks_per_row + b];
        n_greater += c.x;
        n_equal += c.y;
    }

    const unsigned long long lane_mask = (1ULL << lane) - 1;
    for (int base = row_block * chunk; base < col_end; base += BLOCK_SIZE) {
        const int col = base + tid;
        const uint32_t key = col < col_end ? top_k_float_to_ordered(row_src[col]) : 0u;
        const bool greater = col < col_end && key > prefix;
        const bool equal = col < col_end && key == prefix;
        const unsigned long long mask_g = __ballot(greater);
        const unsigned long long mask_e = __ballot(equal);
        if (lane == 0) {
            warp_greater[warp] = __popcll(mask_g);
            warp_equal[warp] = __popcll(mask_e);
        }
        __syncthreads();
        int before_g = n_greater;
        int before_e = n_equal;
        for (int w = 0; w < BLOCK_SIZE / warpSize; ++w) {
            if (w < warp) {
                before_g += warp_greater[w];
                before_e += warp_equal[w];
            }
            n_greater += warp_greater[w];
            n_equal += warp_equal[w];
        }
        if (greater) {
            row_dst[before_g + __popcll(mask_g & lane_mask)] = col;
        }
        if (equal) {
            const int pos = before_e + __popcll(mask_e & lane_mask);
            if (pos < rank) {
                row_dst[k - rank + pos] = col;
            }
        }
        __syncthreads();
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    ggml_cuda_pool_alloc<int2> counts_alloc(pool, (size_t) nrows * blocks_per_row);
    top_k_radix_count<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, counts_alloc.get(), ncols, blocks_per_row);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, counts_alloc.get(), ncols, k, blocks_per_row);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
// the HIP top-k of ggml_cuda_op_top_k for contiguous rows (GGML_OP_QSA_MASK's reference mode compares against it)
void ggml_cuda_top_k_rows_f32(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src, dst, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> tmp(pool, (size_t) ncols * nrows);
        argsort_f32_i32_cuda_bitonic(src, tmp.get(), ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp.get(), ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
}
#endif

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
