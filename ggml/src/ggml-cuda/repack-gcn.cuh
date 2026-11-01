#pragma once

#include "common.cuh"

// GCN (gfx906) weight-repacking buffer type.
//
// Q4_K weights are transformed at upload into a three-plane layout that
// the repacked matvec kernel streams fully coalesced — the on-disk
// superblock layout only sustains ~58% of HBM bandwidth on wave64
// because nibbles, scales and superblock constants interleave every 144
// bytes. Repacked, the same matvec sustains ~89% (measured 470 -> 740
// GB/s on a 21504x5376 FFN shape, gfx906).
//
// Layout (per 2D tensor [ne0 = K, ne1 = M], row-major rows of K):
//   nibble plane:     M * nsp * 16 bytes — one 16-byte chunk per
//                     32-weight sub-block; uint32 j of a chunk holds
//                     weights 4j..4j+3 in its low nibbles and weights
//                     16+4j..16+4j+3 in its high nibbles (dp4a-ready).
//   scale plane:      M * nsp * 2 bytes — per sub-block the 6-bit
//                     scale then min as two u8 (get_scale_min_k4 output).
//   superblock plane: M * (K/256) * 4 bytes — fp16 d then dmin.
// nsp = K/32, plus one padding sub-block when K/32 is a power of two
// (a power-of-two row stride aliases every row onto the same HBM
// channel: ~3x matvec slowdown).
//
// On by default on GCN devices only (GGML_CUDA_REPACK=0 disables it; per-type
// switches are listed at ggml_cuda_repack_tensor_supported). Weights placed
// in this buffer type cannot be read back (get_tensor aborts).

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

// Returns the repack buffer type for `device`, or nullptr when disabled
// (env GGML_CUDA_REPACK=0) or the device is not GCN.
ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device);

// True if `t` can live in the repack buffer type: a 2D weight (K-quants,
// Q8_0 and Q5_1 also 3D expert stacks) of a repackable type whose
// GGML_CUDA_REPACK_<type> switch is on, with the row length that type needs
// (K-quants: ne0 % 256 == 0).
bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

// MUL_MAT with src0 in the repack buffer type. ne11 == 1 runs the
// repacked dp4a matvec; larger ne11 runs the repacked int8 MMQ GEMM.
// 3D/4D src1 broadcasts the 2D weight per slice.
void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MUL_MAT_ID with 3D src0 (per-expert repacked slabs) in the repack
// buffer type. Decode (one token) runs one matvec per assignment;
// batches run a grouped tile GEMM over expert-sorted assignments.
void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);

// MUL_MAT_ID (repacked Q5_1 experts, one token) with the following router-weighted expert sum in one
// kernel, writing the reduction's dst. false: not applicable, run the two ops separately.
bool ggml_cuda_mul_mat_id_repacked_down_reduce(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        const ggml_tensor * weights, const ggml_tensor * expert_scale, ggml_tensor * dst, bool copy_ids);

// Fused gate+up GLU decode path (Q4_K, dense and MoE).
// q8_1 buffer for a producing kernel to fill alongside tensor t (flat 32-value blocks), when t
// feeds a single-column repacked matvec; the consumer then skips its quantize. nullptr: don't emit
// q8_1 buffer for a producer kernel to write t's quantized blocks into (nullptr: no repacked consumer).
// With id_blocks_per_row, a MUL_MAT_ID consumer is served too; it then holds the padded per-row stride
// the producer must write (0 = flat layout for a MUL_MAT consumer).
void * ggml_cuda_repack_xq_emit_target(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const ggml_tensor * t,
        int64_t * id_blocks_per_row = nullptr);

// drop cached q8_1 activations whose source range node writes
// (force: also entries this node produced itself)
// keep: also spare entries produced by this node (the last node of a fused group, whose q8_1 copy the
// fused kernel wrote from final values; the group's elided intermediates may share its memory)
void ggml_cuda_repack_xq_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * node, bool force = false,
        const ggml_tensor * keep = nullptr);

bool ggml_cuda_repack_should_fuse_glu(const ggml_tensor * up, const ggml_tensor * gate,
    const ggml_tensor * glu);
void ggml_cuda_mul_mat_repacked_fused_glu(ggml_backend_cuda_context & ctx,
    const ggml_tensor * up_w, const ggml_tensor * gate_w,
    const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, int glu_op);

// dense repacked Q8_0 single-column matvec eligible for ggml_cuda_mul_mat_repacked_multi
bool ggml_cuda_repack_q8_multi_ok(const ggml_tensor * mm);
// token-count independent part of it (graph_optimize grouping)
bool ggml_cuda_repack_q8_multi_group(const ggml_tensor * mm);
// 2-3 such matvecs sharing src1, one launch
void ggml_cuda_mul_mat_repacked_multi(ggml_backend_cuda_context & ctx, ggml_tensor * const * mms, int n);

// qwen4exp hc mix tail at decode: SCALE -> SILU -> MUL_MAT(repacked Q8_0 hc up, K = 320) -> DSV4_HC_PRE
bool ggml_cuda_hc_up_pre_ok(const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * up, const ggml_tensor * pre);
void ggml_cuda_hc_up_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * up,
        ggml_tensor * pre, void * yq);

// host-only hooks for tests/test-repack-host.cpp
size_t ggml_cuda_repack_nbytes_for_test(ggml_type type, int64_t ne0, int64_t ne1);
int    ggml_cuda_repack_eff_type_for_test(ggml_type type);
void   ggml_cuda_repack_host_for_test(ggml_type type, const void * src, void * dst, int64_t ne0, int64_t ne1);
