#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Recurrent conv step (decode / verify): GET_ROWS(conv cache) -> CONCAT(state, x^T) -> tail CPYs, one launch.
// Returns the number of nodes after i that the launch covered, or -1 when the CONCAT at i does not qualify.
int  ggml_cuda_try_conv_step_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// true: this conv-state gather is read in place by its fused conv step and must not be computed
bool ggml_cuda_conv_state_gather_elidable(const ggml_cgraph * cgraph, const ggml_tensor * gr);
// ggml_cuda_is_view_or_noop, exported for the fusion matchers
bool ggml_cuda_is_view_or_noop_public(const ggml_tensor * t);

// graph_optimize: move the conv step's SSM_CONV -> SILU behind its CONCAT so the step fusion covers them
void ggml_cuda_conv_step_graph_optimize(ggml_cgraph * cgraph);
