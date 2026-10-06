#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
void ggml_cuda_top_k_rows_f32(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream);
#endif
