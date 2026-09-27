import ttfs
import enum 
import dataclasses 

class KLSymbols:
    # qk symbols
    q = ttfs.TensorDescCreator(
        "S_qo->S,H_kv_qk,GQA_qk,D_total_qk->D_qk",
        block_sizes=dict(S_qo="BLOCK_M", H_kv_qk="1", GQA_qk="GQA_qk", D_qk="D_qk"),
    )
    q_remain = q.with_block_sizes(S_qo="BLOCK_M", H_kv_qk="1", GQA_qk="GQA_qk", D_qk="D_qk_remain")

    q_sparse = q.with_block_sizes(H_kv_qk="H_kv_qk", GQA_qk="GQA_qk", D_qk="D_qk")
    q_sparse_remain = q.with_block_sizes(H_kv_qk="H_kv_qk", GQA_qk="GQA_qk", D_qk="D_qk_remain")

    k = ttfs.TensorDescCreator(
        "S_kv->S,H_kv_mul_D_qk",
        block_sizes=dict(S_kv="BLOCK_N", H_kv_mul_D_qk="D_qk"),
        block_strides=dict(H_kv_mul_D_qk="D_total_qk"),
    )
    k_remain = k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D_qk="D_qk_remain")
    # sparse + tma gather (blackwell only)
    k_tma_gather = k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D_qk="D_qk").with_gather_dim(0)
    k_remain_tma_gather = k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D_qk="D_qk_remain").with_gather_dim(0)

    # iqik symbols
    index_q = ttfs.TensorDescCreator(
        "S_qo->S,H_kv_index,GQA_index,D_total_index->D_index",
        block_sizes=dict(S="BLOCK_M", GQA_index="GQA_index", D_index="D_index"),
    )
    index_q_loop_m = index_q.with_block_sizes(S="1", GQA_index="GQA_index", D_index="D_index")
    dindex_q = index_q.copy()
    dindex_q_loop_m = index_q_loop_m.copy()

    # mqa: H_kv == 1
    index_q_loop_gqa = index_q.with_block_sizes(S="BLOCK_M", GQA_index="GQA_qk", D_index="D_index")
    index_q_remain = index_q.with_block_sizes(S="BLOCK_M", GQA_index="GQA_index", D_index="D_index_remain")
    index_q_remain_loop_gqa = index_q.with_block_sizes(S="BLOCK_M", GQA_index="GQA_qk", D_index="D_index_remain")
    dindex_q_remain = index_q_remain.copy()
    index_k = ttfs.TensorDescCreator(
        "S_kv->S,H_kv_mul_D_index",
        block_sizes=dict(S_kv="BLOCK_N", H_kv_mul_D_index="D_index"),
        block_strides=dict(H_kv_mul_D_index="D_total_index"),
    )
    index_k_tma_gather = index_k.with_block_sizes(S_kv="1", H_kv_mul_D_index="D_index")
    index_k_remain = index_k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D_index="D_index_remain")
    index_k_remain_tma_gather = index_k.with_block_sizes(S_kv="1", H_kv_mul_D_index="D_index_remain")

    index_k_noflat = ttfs.TensorDescCreator(
        "S_kv->S,H_kv_index,D_total_index->D_index",
        block_sizes=dict(S="BLOCK_N", D_index="D_index"),
        block_strides=dict(D_index="D_total_index"),
    )
    index_k_noflat_remain = index_k_noflat.with_block_sizes(S="BLOCK_N", D_index="D_index_remain")

    w = ttfs.TensorDescCreator(
        "S_qo->S,H_kv_index,GQA_index",
        block_sizes=dict(S="BLOCK_M", H_kv_index="H_kv_index", GQA_index="GQA_index"),
    )
    dw = w.copy()
    w_loop_m = w.with_block_sizes(S="1", H_kv_index="H_kv_index", GQA_index="GQA_index")
    w_before_sum_kq_loop_m = ttfs.TensorDescCreator(
        "S_qo->S,H_kv_index,BLOCK_N,GQA_index",
        block_sizes=dict(S="1", H_kv_index="H_kv_index", BLOCK_N="BLOCK_N", GQA_index="GQA_index"),
    )

    # this assume GQA_index % GQA_qk == 0
    w_loop_gqa_mqa = w.with_block_sizes(S="BLOCK_M", GQA_index="GQA_qk")
    lse_loop_head = ttfs.TensorDescCreator(
        "S_qo->S,H_kv_qk,GQA_qk",
        block_sizes=dict(H_kv_qk="1", S="BLOCK_M", GQA_qk="GQA_qk"),
    )
    kv_inds = ttfs.TensorDescCreator(
        "S_qo->S,topk",
        block_sizes=dict(topk="BLOCK_N"),
    )

    kl_loss = ttfs.TensorDescCreator(
        "S_qo",
        block_sizes=dict(S_qo="BLOCK_M"),
    )
    indexer_lse = kl_loss.copy()

    kl_loss_single = kl_loss.with_block_sizes(S_qo="1")
    indexer_lse_single = kl_loss.with_block_sizes(S_qo="1")

class LogitsSymbols:
    q = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,D_total->D",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA", D="D"),
    )
    q_with_h = q.with_block_sizes(S_qo="BLOCK_M", H_kv="1", GQA="GQA", D="D")
    q_remain_with_h = q.with_block_sizes(S_qo="BLOCK_M", H_kv="1", GQA="GQA", D="D_remain")

    # sparse + tma gather (blackwell only)
    k = ttfs.TensorDescCreator(
        "S_kv_buf->S,H_kv_mul_D",
        block_sizes=dict(S_kv_buf="BLOCK_N", H_kv_mul_D="D"),
    )
    k_paged = ttfs.TensorDescCreator(
        "S_kv_buf->S,page_size,H_kv_mul_D",
        block_sizes=dict(S_kv_buf="1", page_size="page_size", H_kv_mul_D="D"),
    )
    k_scales = ttfs.TensorDescCreator(
        "S_kv_buf->S",
        block_sizes=dict(S_kv_buf="BLOCK_N"),
    )

    k_scales_paged = ttfs.TensorDescCreator(
        "S_kv_buf->S,page_size",
        block_sizes=dict(S_kv_buf="1", page_size="page_size"),
    )
    k_dense = ttfs.TensorDescCreator(
        "S_kv_buf->S,H_kv_mul_D",
        block_sizes=dict(S_kv_buf="split_kv_size", H_kv_mul_D="D"),
    )
    k_dense_remain = k_dense.with_block_sizes(
        S_kv_buf="split_kv_size",
        H_kv_mul_D="D_remain",
    )
    k_scales_dense = ttfs.TensorDescCreator(
        "S_kv_buf->S",
        block_sizes=dict(S_kv_buf="split_kv_size"),
    )

    w = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA"),
    )
    w_with_h = ttfs.TensorDescCreator(
        "H_kv,S_qo->S,GQA",
        block_sizes=dict(H_kv="H_kv", S_qo="BLOCK_M", GQA="GQA"),
    )


    k_tma_gather = ttfs.TensorDescCreator(
        "S_kv_buf->S,H_kv_mul_D",
        block_sizes=dict(S_kv_buf="1", H_kv_mul_D="D"),
    )
    kv_inds = ttfs.TensorDescCreator(
        "num_query,S_kv->S",
        block_sizes=dict(num_query="1", S_kv="BLOCK_N"),
    )

    kv_inds_paged = ttfs.TensorDescCreator(
        "num_query,S_kv->S",
        block_sizes=dict(num_query="1", S_kv="1"),
    )

    logits = ttfs.TensorDescCreator(
        "S_qo, max_seqlen",
        block_sizes=dict(max_seqlen="BLOCK_N", S_qo="BLOCK_M"),
    )
    logits_full = ttfs.TensorDescCreator(
        "S_qo, S_kv_buf",
        block_sizes=dict(S_kv_buf="BLOCK_N", S_qo="BLOCK_M"),
    )

class FusedTopkSymbols:
    q = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,D_total->D",
        block_sizes=dict(S_qo="BLOCK_M", H_kv="1", GQA="GQA", D="D"),
    )
    q_remain = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,D_total->D",
        block_sizes=dict(S_qo="BLOCK_M", H_kv="1", GQA="GQA", D="D_remain"),
    )

    k = ttfs.TensorDescCreator(
        "S_kv->S,H_kv,D_total->D",
        block_sizes=dict(S_kv="BLOCK_N", H_kv="1", D="D"),
    )
    k_remain = ttfs.TensorDescCreator(
        "S_kv->S,H_kv,D_total->D",
        block_sizes=dict(S_kv="BLOCK_N", H_kv="1", D="D_remain"),
    )
    # these symbols currently not used in pipeline, block size is dynamic in kernel.
    w = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA",
        block_sizes=dict(S="BLOCK_M", H_kv="1", GQA="GQA"),
    )
    topk_datas = ttfs.TensorDescCreator(
        "S_qo->S,topk",
        block_sizes=dict(S="1", topk="topk"),
    )

class SparseAttnDecodeSymbols:
    q = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,D_total->D",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA", D="D"),
    )
    q_remain = q.with_block_sizes(S_qo="BLOCK_M", GQA="GQA", D="D_remain")
    k = ttfs.TensorDescCreator(
        "S_kv->S,H_kv_mul_D",
        block_sizes=dict(S_kv="BLOCK_N", H_kv_mul_D="D"),
        block_strides=dict(H_kv_mul_D="D_total"),
    )
    v = ttfs.TensorDescCreator(
        "S_kv->S,H_kv_mul_dv",
        block_sizes=dict(S_kv="BLOCK_N", H_kv_mul_dv="D_v"),
    )

    k_remain = k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D="D_remain")
    # sparse + tma gather (blackwell only)
    k_tma_gather = k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D="D").with_gather_dim(0)
    k_remain_tma_gather = k.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_D="D_remain").with_gather_dim(0)
    v_tma_gather = v.with_block_sizes(S_kv="BLOCK_N", H_kv_mul_dv="D_v").with_gather_dim(0)
    o = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,D_v->D",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA", D="D_v"),
    )
    oT = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,D_v->D",
        block_sizes=dict(D="D_v", S_qo="BLOCK_M", GQA="GQA",),
    )

    lse = ttfs.TensorDescCreator(
        "H_kv,GQA,S_qo->S",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA"),
    )
    sink = ttfs.TensorDescCreator(
        "H_kv,GQA",
        block_sizes=dict(GQA="GQA"),
    )

    o_split = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,num_split,D_v->D",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA", D="D_v"),
    )
    lse_split = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA,num_split",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA"),
    )
    kv_inds = ttfs.TensorDescCreator(
        "S_qo->S,topk",
        block_sizes=dict(topk="BLOCK_N"),
    )
    prob_sum = ttfs.TensorDescCreator(
        "S_qo->S,H_kv,GQA",
        block_sizes=dict(S_qo="BLOCK_M", GQA="GQA"),
    )
