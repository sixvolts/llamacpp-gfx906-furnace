#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// RMS_NORM (dst) followed by SCALE on its result, written to scale_node
void ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_node);

// RMS_NORM -> MUL that also writes q8_1 blocks of the result into yq; false: not applicable, nothing launched
bool ggml_cuda_op_rms_norm_fused_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, void * yq);

// residual ADD -> RMS_NORM -> MUL (yq: optional q8_1 copy of the MUL output); false = not applicable
bool ggml_cuda_op_add_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * norm,
        ggml_tensor * mul_tensor, void * yq);

// two independent same-shape RMS_NORM -> SCALE pairs in one launch; false = not applicable
bool ggml_cuda_op_rms_norm_scale2(ggml_backend_cuda_context & ctx, ggml_tensor * n0, ggml_tensor * sc0, ggml_tensor * n1, ggml_tensor * sc1);

// qwen4exp hc combine fused with the next hc mix norm: the SCALE -> SIGMOID -> SCALE weight of inject (s_in, s_out),
// DSV4_HC_POST (post) and RMS_NORM -> MUL (+q8_1 into yq when non-null) in one launch; sma_g/a/b (optional): the block
// output is the shared-expert combine b + a * sigmoid(g), folded in. false = not applicable, nothing launched
bool ggml_cuda_op_hc_post_rms_norm_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * inject, ggml_tensor * post,
        float s_in, float s_out, const ggml_tensor * norm, ggml_tensor * mul_tensor, void * yq,
        const ggml_tensor * sma_g = nullptr, const ggml_tensor * sma_a = nullptr, const ggml_tensor * sma_b = nullptr,
        bool dry = false);

// RMS_NORM -> MUL(w) -> MUL(SIGMOID(z)) (GDN output gate) in one launch, q8_1 copy into yq when non-null;
// false = not applicable, nothing launched
bool ggml_cuda_op_rms_norm_mul_sigmoid_gate(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul_tensor,
        ggml_tensor * sig, ggml_tensor * gate_mul, void * yq);
