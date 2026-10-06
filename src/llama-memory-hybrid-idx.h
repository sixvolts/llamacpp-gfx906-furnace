#pragma once

#include "llama-memory-hybrid.h"
#include "llama-kv-cells.h"

#include "ggml-cpp.h"

#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // cell j of all of these is cell lo + j of the cache, the start of the KV window (llama_kv_cache::get_kv_window)
    // bias may be nullptr (and the tensors host scratch without a buffer): only the layout is brought up to date
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, uint32_t lo) const;

    // QSA block-key cache: per QSA layer, the finished (pooled, normed, roped) indexer key of every numbered block of
    // the cached layout, F32 [idx_dim, kv_size/ratio + 1] (the last row takes padding writes). A decode step then
    // re-pools only the blocks that are new or whose cells were rewritten, and scores against the cache.
    // Unified caches only; LLAMA_QSA_BLK_CACHE=0 disables it. nullptr when off
    ggml_tensor * get_qsa_blk_k(int32_t il) const;

    struct qsa_blk_plan {
        int     mode = 0;            // 0: pool every block, no cache; 1: pool every block and refresh the cache;
                                     // 2: pool `bids` only (padded to n_re) and score against the cache
        int32_t n_re = 0;            // re-pooled block slots of the graph (fixed by the ubatch shape)
        int     kind = 0;            // the layout the block ids belong to (see qsa_last_kind)
        uint64_t gen = 0;
        std::vector<int32_t> bids;   // mode 2: the blocks to re-pool
    };

    // bring the layout of each QSA ratio up to date for this ubatch and decide how its block keys are computed
    // called when the ubatch is applied, before the graph is built or reused (the graph's shape follows the plan)
    void qsa_prepare(const llama_ubatch & ubatch, uint32_t lo, int64_t n_kv, std::map<uint32_t, qsa_blk_plan> & plans) const;

    // mode 2 inputs: re_cells I32 [ratio*n_re] members, re_pos I32 [4*n_re] rope rows, re_ids I64 [n_re] cache rows
    void set_input_qsa_re(ggml_tensor * re_cells, ggml_tensor * re_pos, ggml_tensor * re_ids, uint32_t ratio,
                          const qsa_blk_plan & plan) const;

    // the set_input_qsa layout of one stream holding one sequence, kept between calls: decoding only fills
    // empty cells, so the next call updates the few cells that changed instead of regrouping every cell
    struct qsa_layout {
        bool                  valid    = false;
        uint64_t              gen      = 0;    // bumped by every full rebuild: block ids are only stable within one
        std::vector<int32_t>  dirty;           // numbered blocks whose cells were rewritten in place (seq_rm + re-add)
        const void          * cells    = nullptr;
        uint32_t              lo       = 0;
        llama_kv_cells::seq_set_t vis;   // sequences of the ubatch: the cells of others read as empty
        int64_t               n_kv     = 0;
        int64_t               n_blocks = 0;
        int64_t               ratio    = 0;
        std::vector<int32_t>  pos;       // [n_kv] cell positions the layout describes, -1 for empty
        std::vector<int32_t>  cell_blk;  // [n_kv]
        std::vector<int32_t>  blk_cells; // [ratio*n_blocks]
        std::vector<int32_t>  blk_pos;   // [4*n_blocks]
        std::vector<int32_t>  blk_of;    // [n_kv] block of each cell, -1 when it is in no full block
        std::vector<uint64_t> grp_slots; // [n_blocks] filled slots of each position bucket
        std::vector<int32_t>  grp_first; // [n_blocks] first cell seen in each bucket
        std::vector<int32_t>  bid_idx;   // first position of each numbered block
        std::vector<int32_t>  bid_cell;  // first cell of each numbered block
        std::vector<int32_t>  unpooled;  // every cell with blk_of < 0
        std::vector<int32_t>  added;     // scratch
    };

    // the same for a cell array holding several sequences (unified cache): a block is keyed on
    // (position bucket, sequence set) and numbered when it fills. numbering order differs from the
    // full rebuild (append instead of position order), which does not change any result: block scores
    // are per block and the top-k runs per cell
    struct qsa_layout_ms {
        bool                  valid    = false;
        uint64_t              gen      = 0;    // bumped by every full rebuild: block ids are only stable within one
        std::vector<int32_t>  dirty;           // numbered blocks whose cells were rewritten in place (seq_rm + re-add)
        const void          * cells    = nullptr;
        uint32_t              lo       = 0;
        llama_kv_cells::seq_set_t vis;
        int64_t               n_kv     = 0;
        int64_t               n_blocks = 0;
        int64_t               ratio    = 0;
        std::vector<int32_t>  pos;       // [n_kv] cell positions, -1 for empty
        std::vector<int32_t>  cell_grp;  // [n_kv] group of each cell, -1
        std::vector<int32_t>  cell_blk;  // [n_kv]
        std::vector<int32_t>  blk_of;    // [n_kv]
        std::vector<int32_t>  blk_cells; // [ratio*n_blocks]
        std::vector<int32_t>  blk_pos;   // [4*n_blocks]
        std::vector<int32_t>  bid_idx;   // first position of each numbered block
        std::vector<int32_t>  bid_cell;  // a cell of each numbered block
        std::vector<llama_kv_cells::seq_set_t> grp_seq; // sequence set of each group
        std::vector<uint64_t> grp_slots;
        std::vector<int32_t>  grp_first;
        std::vector<int32_t>  grp_slot0;
        std::vector<int32_t>  grp_bid;
        std::vector<int32_t>  grp_next;  // chain of the groups in one bucket
        std::vector<int32_t>  grp_head;  // [n_blocks]
        std::vector<int32_t>  unpooled;  // every cell in no full block
        std::vector<int32_t>  added;     // scratch
    };

private:
    mutable std::map<uint32_t, qsa_layout>    qsa_layouts;
    mutable std::map<uint32_t, qsa_layout_ms> qsa_layouts_ms;

    // drop every cached layout (block ids restart, so the block keys are recomputed)
    void qsa_layouts_drop();

    mutable uint64_t qsa_gen_next = 1;

    // cells seq_rm emptied since the layouts were last brought up to date (absolute indices of the indexer cache).
    // a cell filled again at the same position keeps its block, which then needs its key re-pooled
    mutable std::vector<uint32_t> qsa_rm_cells;

    // block-key cache
    std::map<int32_t, ggml_tensor *> qsa_blk_k;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> qsa_blk_bufs;
    std::vector<uint32_t> qsa_ratios;   // distinct ratios of the QSA layers

    // the layout set_input_qsa last used for a ratio (one stream): 1 qsa_layouts, 2 qsa_layouts_ms, 0 none it can keep
    mutable std::map<uint32_t, int> qsa_last_kind;

    // qsa_prepare brought this ratio's layout up to date for the current ubatch: the ubatch's own set_input_qsa takes
    // it as it is instead of scanning the cells again
    mutable std::map<uint32_t, bool> qsa_fresh;

    struct qsa_key_state {
        bool     valid   = false;
        uint64_t gen     = 0;   // layout generation the keys belong to
        int32_t  n_keyed = 0;   // blocks [0, n_keyed) hold current keys
    };
    mutable std::map<uint32_t, qsa_key_state> qsa_keys;

    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

    const llama_memory_hybrid_idx * get_mem() const { return mem; }

    // the block-key plan of the current ubatch for a ratio (see llama_memory_hybrid_idx::qsa_prepare), nullptr if none
    const llama_memory_hybrid_idx::qsa_blk_plan * get_qsa_plan(uint32_t ratio) const;

    void set_input_qsa_re(ggml_tensor * re_cells, ggml_tensor * re_pos, ggml_tensor * re_ids, uint32_t ratio) const;

private:
    std::map<uint32_t, llama_memory_hybrid_idx::qsa_blk_plan> qsa_plans;

    // made from a batch (has ubatches): only then does apply() plan the block keys
    bool is_batch = false;

    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
