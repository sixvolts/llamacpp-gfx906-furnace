#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // several sequences in one unified cache: attend only the span of the cells of the ubatch's sequences
    // the indexer mirrors the attention cells, so both caches find the same window
    get_mem_attn()->set_window(true);
    if (mem_idx) {
        mem_idx->set_window(true);
        GGML_ASSERT(mem_idx->get_window() == get_mem_attn()->get_window());
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    qsa_layouts.clear();
    qsa_layouts_ms.clear();

    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

// the cells [lo, lo + n_kv) a ubatch attends, numbered from lo. the cells of sequences outside the ubatch
// read as empty: every token masks them anyway, and inside a window their positions need not fit it
struct qsa_cells {
    const llama_kv_cells &          cells;
    const uint32_t                  lo;
    const llama_kv_cells::seq_set_t vis;

    bool is_empty(int64_t j) const {
        return cells.is_empty(lo + j) || (cells.seq_get_all(lo + j) & vis).none();
    }

    llama_pos pos_get(int64_t j) const {
        return cells.pos_get(lo + j);
    }

    const llama_kv_cell_ext & ext_get(int64_t j) const {
        return cells.ext_get(lo + j);
    }

    const llama_kv_cells::seq_set_t & seq_get_all(int64_t j) const {
        return cells.seq_get_all(lo + j);
    }

    bool seq_has(int64_t j, llama_seq_id seq_id) const {
        return cells.seq_has(lo + j, seq_id);
    }

    // over the whole cache, not the window
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        return cells.seq_pos_min(seq_id);
    }

    template <typename L>
    bool same(const L & lay) const {
        return lay.cells == (const void *) &cells && lay.lo == lo && lay.vis == vis;
    }

    template <typename L>
    void stamp(L & lay) const {
        lay.cells = (const void *) &cells;
        lay.lo    = lo;
        lay.vis   = vis;
    }
};

// bring a layout up to date when the only change since it was built is newly filled cells, and every
// block they complete lies past the numbered ones. anything else (moved or dropped cells, a repeated
// position, a block completing out of order) returns false and leaves the layout for a full rebuild
static bool qsa_layout_advance(llama_memory_hybrid_idx::qsa_layout & L, const qsa_cells & cells,
        int64_t n_kv, int64_t n_blocks, int64_t r) {
    if (!L.valid || !cells.same(L) || L.n_kv != n_kv || L.n_blocks != n_blocks || L.ratio != r) {
        return false;
    }

    L.valid = false;

    L.added.clear();
    for (int64_t j = 0; j < n_kv; ++j) {
        const int32_t p = cells.is_empty(j) ? -1 : cells.pos_get(j);
        if (p != L.pos[j]) {
            if (L.pos[j] != -1) {
                return false;
            }
            L.added.push_back((int32_t) j);
        }
    }

    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    for (const int32_t j : L.added) {
        const int32_t p  = cells.pos_get(j);
        const int64_t pb = p/r;
        const uint64_t bit = uint64_t(1) << (p%r);

        if (pb >= n_blocks || (L.grp_slots[pb] & bit) != 0) {
            return false;
        }

        if (L.grp_first[pb] < 0) {
            L.grp_first[pb] = j;
        }
        L.grp_slots[pb] |= bit;
        L.pos[j] = p;

        if (L.grp_slots[pb] != slots_full) {
            continue;
        }

        // blocks are numbered in position order, so a new one can only go after the last
        const int32_t bid = (int32_t) L.bid_idx.size();
        if ((bid > 0 && L.bid_idx.back() >= pb*r) || bid >= n_blocks) {
            return false;
        }

        L.bid_idx .push_back((int32_t) (pb*r));
        L.bid_cell.push_back(L.grp_first[pb]);

        for (int64_t sec = 0; sec < 4; ++sec) {
            L.blk_pos[sec*n_blocks + bid] = (int32_t) (pb*r);
        }

        size_t w = 0;
        for (const int32_t c : L.unpooled) {
            if (L.pos[c] >= 0 && L.pos[c]/r == pb) {
                L.blk_of[c]   = bid;
                L.cell_blk[c] = bid;
                L.blk_cells[bid*r + L.pos[c]%r] = c;
            } else {
                L.unpooled[w++] = c;
            }
        }
        L.unpooled.resize(w);
    }

    // the spare block follows the numbered ones
    const int32_t n_bid    = (int32_t) L.bid_idx.size();
    const int32_t dead_bid = n_bid < n_blocks ? n_bid : (int32_t) n_blocks - 1;
    for (const int32_t c : L.unpooled) {
        L.cell_blk[c] = dead_bid;
    }

    L.valid = true;
    return true;
}

// LLAMA_QSA_LAYOUT_CHECK=1: after a cached multi-sequence layout, run the full rebuild too and compare
static bool qsa_layout_check() {
    static const bool v = getenv("LLAMA_QSA_LAYOUT_CHECK") != nullptr;
    return v;
}

// LLAMA_QSA_NO_MS_CACHE=1: disable the multi-sequence layout cache (full rebuild every call)
static bool qsa_layout_ms_disabled() {
    static const bool v = getenv("LLAMA_QSA_NO_MS_CACHE") != nullptr;
    return v;
}

// bring a multi-sequence layout up to date with the cells: new cells join or open (bucket, sequence set)
// groups and a group that fills becomes the next numbered block. returns false when the change is not
// an append (a cell emptied, moved, doubled a slot, or the window changed), so the caller rebuilds
static bool qsa_layout_ms_advance(llama_memory_hybrid_idx::qsa_layout_ms & L, const qsa_cells & cells,
        int64_t n_kv, int64_t n_blocks, int64_t r) {
    if (!L.valid || !cells.same(L) || L.n_kv != n_kv || L.n_blocks != n_blocks || L.ratio != r) {
        return false;
    }

    L.valid = false;
    L.added.clear();

    for (int64_t j = 0; j < n_kv; ++j) {
        const int32_t p = cells.is_empty(j) ? -1 : cells.pos_get(j);
        if (p != L.pos[j]) {
            if (L.pos[j] != -1) {
                return false;
            }
            L.added.push_back((int32_t) j);
        }
    }

    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    for (const int32_t j : L.added) {
        const int32_t p  = cells.pos_get(j);
        const int64_t pb = p/r;
        if (pb >= n_blocks) {
            return false;
        }

        const auto & S = cells.seq_get_all(j);

        int32_t g = -1;
        for (int32_t c = L.grp_head[pb]; c >= 0; c = L.grp_next[c]) {
            if (L.grp_seq[c] == S) {
                g = c;
                break;
            }
        }
        if (g < 0) {
            g = (int32_t) L.grp_seq.size();
            L.grp_seq  .push_back(S);
            L.grp_slots.push_back(0);
            L.grp_first.push_back(j);
            L.grp_slot0.push_back(-1);
            L.grp_bid  .push_back(-1);
            L.grp_next .push_back(L.grp_head[pb]);
            L.grp_head[pb] = g;
        }

        const uint64_t bit = uint64_t(1) << (p%r);
        if ((L.grp_slots[g] & bit) != 0 || L.grp_bid[g] >= 0) {
            return false;
        }

        L.pos[j]      = p;
        L.cell_grp[j] = g;
        L.grp_slots[g] |= bit;
        L.grp_first[g]  = std::min(L.grp_first[g], j);
        if (p%r == 0) {
            L.grp_slot0[g] = j;
        }

        if (L.grp_slots[g] != slots_full) {
            L.blk_of[j] = -1;
            L.unpooled.push_back(j);
            continue;
        }

        const int32_t bid = (int32_t) L.bid_idx.size();
        if (bid >= n_blocks) {
            return false;
        }
        L.grp_bid[g] = bid;
        L.bid_idx .push_back((int32_t) (pb*r));
        L.bid_cell.push_back(L.grp_first[g]);
        for (int64_t sec = 0; sec < 4; ++sec) {
            L.blk_pos[sec*n_blocks + bid] = (int32_t) (pb*r);
        }

        // the other members wait in unpooled; the new cell is not there yet
        size_t w = 0;
        for (const int32_t c : L.unpooled) {
            if (L.cell_grp[c] == g) {
                L.blk_of[c]   = bid;
                L.cell_blk[c] = bid;
                L.blk_cells[bid*r + L.pos[c]%r] = c;
            } else {
                L.unpooled[w++] = c;
            }
        }
        L.unpooled.resize(w);

        L.blk_of[j]   = bid;
        L.cell_blk[j] = bid;
        L.blk_cells[bid*r + p%r] = j;
    }

    // the spare block follows the numbered ones
    const int32_t n_bid    = (int32_t) L.bid_idx.size();
    const int32_t dead_bid = n_bid < n_blocks ? n_bid : (int32_t) n_blocks - 1;
    for (const int32_t c : L.unpooled) {
        L.cell_blk[c] = dead_bid;
    }

    L.valid = true;
    return true;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        uint32_t lo) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // the full path is O(n_kv) per stream (about 1 ms at 33k context on a slow host core); decoding one
    // sequence takes the qsa_layout update instead, which only scans the cell positions
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    // a window covers one stream
    GGML_ASSERT(lo == 0 || n_ns == 1);

    llama_kv_cells::seq_set_t vis;
    for (uint32_t i = 0; i < ubatch->n_seqs_unq; ++i) {
        vis.set(ubatch->seq_id_unq[i]);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const qsa_cells cells = { get_mem_idx()->get_cells(seq_of_stream), lo, vis };

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        qsa_layout    * lay    = n_ns == 1 &&  one_seq ? &qsa_layouts[ratio] : nullptr;
        qsa_layout_ms * lay_ms = n_ns == 1 && !one_seq && !qsa_layout_ms_disabled() ? &qsa_layouts_ms[ratio] : nullptr;

        bool cached = lay != nullptr && qsa_layout_advance(*lay, cells, n_kv, n_blocks, r);

        const bool cached_ms = lay_ms != nullptr && qsa_layout_ms_advance(*lay_ms, cells, n_kv, n_blocks, r);

        // check mode: keep the cached layout aside, rebuild, then compare the two
        std::vector<int32_t> chk_cell_blk;
        std::vector<int32_t> chk_blk_cells;
        std::vector<int32_t> chk_blk_pos;
        int32_t              chk_n_bid = 0;

        if (cached_ms && qsa_layout_check()) {
            chk_cell_blk  = lay_ms->cell_blk;
            chk_blk_cells = lay_ms->blk_cells;
            chk_blk_pos   = lay_ms->blk_pos;
            chk_n_bid     = (int32_t) lay_ms->bid_idx.size();
        } else {
            cached = cached || cached_ms;
        }

        int32_t n_bid     = 0;
        bool    have_dead = false;
        int32_t dead_bid  = 0;

        bool ranked = false;

        if (cached) {
            const auto & c_cell_blk  = cached_ms ? lay_ms->cell_blk  : lay->cell_blk;
            const auto & c_blk_cells = cached_ms ? lay_ms->blk_cells : lay->blk_cells;
            const auto & c_blk_pos   = cached_ms ? lay_ms->blk_pos   : lay->blk_pos;

            memcpy(cur_cell_blk,  c_cell_blk .data(), n_kv*sizeof(int32_t));
            memcpy(cur_blk_cells, c_blk_cells.data(), r*n_blocks*sizeof(int32_t));
            memcpy(dst_blk_pos,   c_blk_pos  .data(), 4*n_blocks*sizeof(int32_t));

            bid_idx  = cached_ms ? lay_ms->bid_idx  : lay->bid_idx;
            bid_cell = cached_ms ? lay_ms->bid_cell : lay->bid_cell;
            if (!blk_bias) {
                blk_of = cached_ms ? lay_ms->blk_of : lay->blk_of;
            }

            n_bid     = (int32_t) bid_idx.size();
            have_dead = n_bid < n_blocks;
            dead_bid  = have_dead ? n_bid : (int32_t) n_blocks - 1;
        } else {
            // a cell no block covers needs its own -inf, which a per-block bias cannot carry
            // every cache path keeps the position below the cell window, so this stays false
            bool oor = false;

            bool dup = false;

            auto group_cells = [&]() {
                // -1 means no usable block: an incomplete or short group cannot be pooled
                std::fill(blk_of.begin(),   blk_of.end(),   -1);
                std::fill(cell_grp.begin(), cell_grp.end(), -1);
                std::fill(grp_head.begin(), grp_head.end(), -1);

                grp_next .clear();
                grp_first.clear();
                grp_slot0.clear();
                grp_slots.clear();
                grp_bid  .clear();

                oor = false;
                dup = false;

                // one sequence: a bucket holds at most one group, so the bucket index is the group id
                // and the chains below are not needed. same groups, same first cells, no per-group push_back
                if (one_seq) {
                    grp_next .assign(n_blocks, -1);
                    grp_first.assign(n_blocks, -1);
                    grp_slot0.assign(n_blocks, -1);
                    grp_slots.assign(n_blocks,  0);
                    grp_bid  .assign(n_blocks, -1);

                    for (int64_t j = 0; j < n_kv; ++j) {
                        if (cells.is_empty(j)) {
                            continue;
                        }

                        const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                        const int64_t pb  = idx/r;

                        if (pb >= n_blocks) {
                            oor = true;
                            continue;
                        }

                        if (grp_head[pb] < 0) {
                            grp_head [pb] = (int32_t) pb;
                            grp_first[pb] = (int32_t) j;
                        }

                        const uint64_t bit = uint64_t(1) << (idx%r);

                        dup |= (grp_slots[pb] & bit) != 0;

                        cell_grp[j]    = (int32_t) pb;
                        grp_slots[pb] |= bit;

                        if (idx%r == 0) {
                            grp_slot0[pb] = (int32_t) j;
                        }
                    }

                    return;
                }

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j)) {
                        continue;
                    }

                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                    const int64_t pb  = idx/r;

                    if (pb >= n_blocks) {
                        oor = true;
                        continue;
                    }

                    int32_t g = -1;

                    for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                        if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                            g = c;
                            break;
                        }
                    }

                    if (g < 0) {
                        g = (int32_t) grp_first.size();

                        grp_next .push_back(grp_head[pb]);
                        grp_first.push_back((int32_t) j);
                        grp_slot0.push_back(-1);
                        grp_slots.push_back(0);
                        grp_bid  .push_back(-1);

                        grp_head[pb] = g;
                    }

                    const uint64_t bit = uint64_t(1) << (idx%r);

                    dup |= (grp_slots[g] & bit) != 0;

                    cell_grp[j]   = g;
                    grp_slots[g] |= bit;

                    if (idx%r == 0) {
                        grp_slot0[g] = (int32_t) j;
                    }
                }
            };

            std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

            group_cells();

            // mrope repeats one position across an image, so rank cells instead of using the position
            if (dup && ubatch->is_pos_2d() && one_seq) {
                order.clear();
                order.reserve(n_kv);

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (!cells.is_empty(j)) {
                        order.push_back((int32_t) j);
                    }
                }

                // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
                std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                    const llama_pos pa = cells.pos_get(a);
                    const llama_pos pb = cells.pos_get(b);

                    if (pa != pb) {
                        return pa < pb;
                    }

                    const auto & ea = cells.ext_get(a);

                    return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
                });

                rank.assign(n_kv, -1);

                for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                    rank[order[k]] = (int32_t) k;
                }

                ranked = true;

                group_cells();
            }

            GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

            for (int64_t pb = 0; pb < n_blocks; ++pb) {
                for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                    if (grp_slots[g] != slots_full) {
                        continue;
                    }

                    grp_bid[g] = n_bid++;

                    bid_idx  .push_back((int32_t) (pb*r));
                    bid_cell .push_back(grp_first[g]);
                    bid_slot0.push_back(grp_slot0[g]);
                }
            }

            GGML_ASSERT(n_bid <= n_blocks);

            for (int32_t b = 0; b < n_bid; ++b) {
                int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

                if (ranked) {
                    const int32_t   c = bid_slot0[b];
                    const llama_pos p = cells.pos_get(c);
                    const auto &    e = cells.ext_get(c);

                    sec_pos[0] = p;
                    sec_pos[1] = e.y;
                    sec_pos[2] = e.x;
                    sec_pos[3] = p;
                }

                for (int64_t sec = 0; sec < 4; ++sec) {
                    dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
                }
            }

            // unpooled cells all point at one spare block. a spare block exists only when some
            // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
            have_dead = n_bid < n_blocks;
            dead_bid  = have_dead ? n_bid : n_blocks - 1;

            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t g = cell_grp[j];

                blk_of[j] = g < 0 ? -1 : grp_bid[g];

                if (blk_of[j] >= 0) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
                }

                cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
            }

            // one sequence and plain positions: keep the layout so the next call can update it in place
            if (lay != nullptr) {
                lay->valid = !ranked && !oor;
                if (lay->valid) {
                    cells.stamp(*lay);
                    lay->n_kv     = n_kv;
                    lay->n_blocks = n_blocks;
                    lay->ratio    = r;

                    lay->pos.resize(n_kv);
                    lay->unpooled.clear();
                    for (int64_t j = 0; j < n_kv; ++j) {
                        lay->pos[j] = cells.is_empty(j) ? -1 : cells.pos_get(j);
                        if (blk_of[j] < 0) {
                            lay->unpooled.push_back((int32_t) j);
                        }
                    }

                    lay->cell_blk .assign(cur_cell_blk,  cur_cell_blk  + n_kv);
                    lay->blk_cells.assign(cur_blk_cells, cur_blk_cells + r*n_blocks);
                    lay->blk_pos  .assign(dst_blk_pos,   dst_blk_pos   + 4*n_blocks);
                    lay->blk_of    = blk_of;
                    lay->grp_slots = grp_slots;
                    lay->grp_first = grp_first;
                    lay->bid_idx   = bid_idx;
                    lay->bid_cell  = bid_cell;
                }
            }

            // check mode: the cached layout must describe the same blocks as the rebuild. block numbers
            // may differ, so compare per cell: same block position and the same member cells
            if (!chk_cell_blk.empty()) {
                const int32_t chk_dead = chk_n_bid < n_blocks ? chk_n_bid : (int32_t) n_blocks - 1;
                bool ok = chk_n_bid == n_bid;
                for (int64_t j = 0; ok && j < n_kv; ++j) {
                    // an empty cell points anywhere in range (the mask hides it); the cache leaves it stale
                    if (cells.is_empty(j)) {
                        continue;
                    }
                    const int32_t b1 = chk_cell_blk[j];
                    const int32_t b2 = cur_cell_blk[j];
                    const bool d1 = b1 == chk_dead && chk_n_bid < n_blocks; // in the spare block
                    const bool d2 = blk_of[j] < 0;
                    if (d1 || d2) {
                        ok = d1 && d2;
                        if (!ok) {
                            LLAMA_LOG_ERROR("%s: qsa multi-sequence layout mismatch at cell %" PRId64 ": pooled state differs (cached %d, rebuilt %d)\n",
                                    __func__, j, b1, b2);
                        }
                        continue;
                    }
                    ok = memcmp(chk_blk_cells.data() + (size_t) b1*r, cur_blk_cells + (size_t) b2*r, r*sizeof(int32_t)) == 0 &&
                         chk_blk_pos[b1] == dst_blk_pos[b2];
                    if (!ok) {
                        LLAMA_LOG_ERROR("%s: qsa multi-sequence layout mismatch at cell %" PRId64 " (cached block %d, rebuilt block %d)\n",
                                __func__, j, b1, b2);
                    }
                }
                if (!ok) {
                    LLAMA_LOG_ERROR("%s: qsa multi-sequence layout check FAILED (n_bid cached %d, rebuilt %d)\n", __func__, chk_n_bid, n_bid);
                    GGML_ABORT("qsa layout check");
                }
            }

            // several sequences and plain positions: keep the layout for in-place updates
            if (lay_ms != nullptr) {
                auto & L = *lay_ms;
                L.valid = !ranked && !oor && !dup;
                if (L.valid) {
                    cells.stamp(L);
                    L.n_kv     = n_kv;
                    L.n_blocks = n_blocks;
                    L.ratio    = r;

                    L.pos.resize(n_kv);
                    L.unpooled.clear();
                    for (int64_t j = 0; j < n_kv; ++j) {
                        L.pos[j] = cells.is_empty(j) ? -1 : cells.pos_get(j);
                        if (L.pos[j] >= 0 && blk_of[j] < 0) {
                            L.unpooled.push_back((int32_t) j);
                        }
                    }

                    L.cell_grp = cell_grp;
                    L.cell_blk .assign(cur_cell_blk,  cur_cell_blk  + n_kv);
                    L.blk_cells.assign(cur_blk_cells, cur_blk_cells + r*n_blocks);
                    L.blk_pos  .assign(dst_blk_pos,   dst_blk_pos   + 4*n_blocks);
                    L.blk_of   = blk_of;
                    L.bid_idx  = bid_idx;
                    L.bid_cell = bid_cell;

                    const size_t n_grp = grp_first.size();
                    L.grp_seq.resize(n_grp);
                    for (size_t g = 0; g < n_grp; ++g) {
                        L.grp_seq[g] = cells.seq_get_all((uint32_t) grp_first[g]);
                    }
                    L.grp_slots = grp_slots;
                    L.grp_first = grp_first;
                    L.grp_slot0 = grp_slot0;
                    L.grp_bid   = grp_bid;
                    L.grp_next  = grp_next;
                    L.grp_head  = grp_head;
                }
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, get_idx()->get_kv_lo());
}
