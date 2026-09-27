from typing import Annotated, Any, Optional

import triton.language as tl
from triton.experimental.gluon import language as gl
import ttfs
from ..triton.logits import DSAIndexerLogitsVarlenManager, DSAIndexerLogitsPersistVarlenArgs
from ttfs.gl import float2

@ttfs.aggregate(kw_only=True)
class IndexerLogitsMainloop:
    tensors: DSAIndexerLogitsVarlenManager
    num_partition: Annotated[int, tl.constexpr]
    sm_id: tl.tensor
    num_sm: tl.tensor
    is_compressed_logits: Annotated[bool, tl.constexpr]
    is_indexer_topk: Annotated[bool, tl.constexpr]
    sm_scale: Annotated[float, tl.constexpr]

    @ttfs.gluon_jit 
    def run_kq_fwd_umma_step(self, pipeline: ttfs.io.PipelineWS, q_smem):
        pipeline, k_C = pipeline.piped_get_consumer("k_io")
        for j in tl.static_range(self.num_partition):
            pipeline, kq_P = pipeline.piped_get_producer(f"kq_{j}")
            kq_P = kq_P.issue_mma_tn_async(f"kq_{j}", k_C.get_smem(f"k").slice(j * self.tensors.BLOCK_N, self.tensors.BLOCK_N), q_smem, use_acc=False)
            pipeline = pipeline.commit_async_ops(kq_P)
        return pipeline

    @ttfs.gluon_jit 
    def run_kq_fwd_compute_step(self, pipeline: ttfs.io.PipelineWS, w_may_smem, part_idx: tl.constexpr):
        q_dtype: tl.constexpr = pipeline.get_tensor_desc("q").get_dtype_cw().value
        is_quantized: tl.constexpr = q_dtype.primitive_bitwidth <= 8
        tm = self.tensors
        kq_reg_layout: tl.constexpr = self._get_kq_layout_cw(pipeline).value
        kq_fake = gl.zeros([tm.BLOCK_N, tm.BLOCK_M * tm.GQA], layout=kq_reg_layout, dtype=gl.float32)
        res_fake = kq_fake.reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA).sum(2)
        k_scale = None
        pipeline, k_C = pipeline.piped_get_consumer("k_io")
        if is_quantized:
            k_scales_smem = k_C.get_smem(f"k_scales") # get mega smem, then slice
            k_scales_smem = k_scales_smem.slice(part_idx * tm.BLOCK_N, tm.BLOCK_N)
            k_scale = k_scales_smem.load(gl.SliceLayout(1, res_fake.type.layout))
        # kq mma finish, means we can release k.
        pipeline, kq_C = pipeline.piped_get_consumer(f"kq_{part_idx}")

        k_C.finish(fence_shared=False)

        kq_tmem = kq_C.get_tmem(f"kq_{part_idx}")
        kq = kq_tmem.load(kq_reg_layout)
        kq_C.finish(fence_shared=False)
        # pipeline = pipeline.record_tensor_event(kq)
        if isinstance(w_may_smem, gl.shared_memory_descriptor):
            w = w_may_smem.load(gl.SliceLayout(0, kq_reg_layout))
            if w.dtype != gl.float32:
                # we use blackwell f32x2 math, so we must convert weight to f32 to use them. 
                w = w.to(gl.float32)
        else:
            w = w_may_smem
        tl.static_assert(self.is_indexer_topk, "TODO: currently don't support lse version")
        use_abs: tl.constexpr = True
        if use_abs:
            # use (accum + abs(accum)) * weight / 2
            # we have to use ptx here because llvm lower fabs to `and.b32`.
            kq_before_f2 = kq.reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA)
            kq_abs = ttfs.ops.cuda.abs_f32(kq).reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA)
            kq_abs_f2 = float2.pack(kq_abs, axis=2)
            kq_f2 = float2.pack(kq_before_f2, axis=2)
            w_f2 = float2.pack(w[None, :].broadcast_to(kq.shape).reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA), axis=2)
            weighted_kq_f2 = (kq_abs_f2 + kq_f2) * w_f2
        else:
            kq_relu = tl.maximum(kq, 0).reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA)
            kq_relu_f2 = float2.pack(kq_relu, axis=2)
            w_f2 = float2.pack(w[None, :].broadcast_to(kq.shape).reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA), axis=2)
            weighted_kq_f2 = kq_relu_f2 * w_f2
        res = weighted_kq_f2.sum(2, keep_dims=True)
        res = float2.unpack(res, axis=2).sum(2, keep_dims=True)
        if k_scale is not None:
            res = res * gl.convert_layout(k_scale[:, None].reshape(tm.BLOCK_N, 1, 1), res.type.layout, assert_trivial=True)
        res = res.reshape(tm.BLOCK_N, tm.BLOCK_M)
        if use_abs:
            res = res / 2
        return pipeline, res

    @ttfs.gluon_jit 
    def _next_varlen_schedule(self, block_q_idx):
        tm = self.tensors
        seq_len = tm.S_qo
        seq_len_kv = tm.S_kv
        start = tl.uint32.get_int_max_value()
        end = tl.uint32.get_int_min_value()
        seq_k_starts = ()
        seq_k_ends = ()
        for i in tl.static_range(tm.BLOCK_M):
            q_idx = min(block_q_idx * tm.BLOCK_M + i, seq_len - 1)
            seq_k_start_i = gl.load(tm.cu_seq_len_k_start + q_idx)
            seq_k_end_i = gl.load(tm.cu_seq_len_k_end + q_idx)
            seq_k_starts += (ttfs.gl.ops.scalar_to_blocked_tensor(seq_k_start_i, [1]), )
            seq_k_ends += (ttfs.gl.ops.scalar_to_blocked_tensor(seq_k_end_i, [1]), )
            start = min(start, min(seq_k_start_i, seq_len_kv))
            end = max(end, tl.minimum(seq_k_end_i, seq_len_kv))
        start = start // 4 * 4
        seq_k_starts_ten = ttfs.gl.ops.join_n(seq_k_starts)
        seq_k_ends_ten = ttfs.gl.ops.join_n(seq_k_ends)
        return seq_k_starts_ten, seq_k_ends_ten, start, tl.cdiv(end - start, tm.split_kv_size)

    @ttfs.gluon_jit
    def _get_kq_layout_cw(self, pipeline: ttfs.io.PipelineWS):
        tm = self.tensors
        if (tm.GQA == 64 or tm.GQA == 128) and tm.BLOCK_N == 64:
            ikiq_layout: tl.constexpr = ttfs.gl.ll_ops.ll_layout_from_short_str("RLLRRR,LLLRWW" if tm.GQA == 64 else "RLLRRRR,LLLRWW")
        else:
            ikiq_layout: tl.constexpr = ttfs.gl.ops.get_tmem_reg_layout(pipeline.get_tmem("kq_0").index(0))
        return ttfs.mp.TritonConstexprField(ikiq_layout)


    @ttfs.gluon_jit 
    def load_qk_partition(self, pipeline: ttfs.io.PipelineWS, record_part_idx: tl.constexpr):
        pipeline = pipeline.prepare_recorder(record_part_idx, f"load")
        tm = self.tensors
        num_q_blocks = tl.cdiv(tm.S_qo, tm.BLOCK_M)
        block_q_idx = self.sm_id
        if block_q_idx < num_q_blocks:
            pipeline = pipeline.piped_load_async("q,weights", ttfs.block_offsets(S_qo=block_q_idx), wait_producer=False) 
        q_dtype: tl.constexpr = pipeline.get_tensor_desc("q").get_dtype_cw().value

        is_quantized: tl.constexpr = q_dtype.primitive_bitwidth <= 8

        k_group_name: tl.constexpr = f"k,k_scales" if is_quantized else f"k"

        while (block_q_idx < num_q_blocks):
            seq_k_starts, seq_k_ends, kv_start, num_kv_blocks = self._next_varlen_schedule(block_q_idx)
            next_block_q_idx = block_q_idx + self.num_sm
            should_load_q = next_block_q_idx < num_q_blocks
            if should_load_q:
                pipeline = pipeline.piped_load_async("q,weights", ttfs.block_offsets(S_qo=next_block_q_idx))  
            for j in range(num_kv_blocks):
                pipeline = pipeline.piped_load_async(k_group_name, offsets=ttfs.offsets(S_kv_buf=kv_start + j * tm.split_kv_size))
            block_q_idx += self.num_sm
        pipeline = pipeline.finalize_recorder()

    @ttfs.gluon_jit 
    def compute_partition(self, pipeline: ttfs.io.PipelineWS, part_idx: tl.constexpr, record_part_idx: tl.constexpr):
        # if part_idx == 0:
        #     ttfs.ops.cuda.reg_dealloc(104)
        pipeline = pipeline.prepare_recorder(record_part_idx, f"compute_{part_idx}")

        tm = self.tensors

        num_q_blocks = tl.cdiv(tm.S_qo, tm.BLOCK_M)
        block_q_idx = self.sm_id
        q_dtype: tl.constexpr = pipeline.get_tensor_desc("q").get_dtype_cw().value

        is_quantized: tl.constexpr = q_dtype.primitive_bitwidth <= 8
        kq_reg_layout: tl.constexpr = self._get_kq_layout_cw(pipeline).value
        MMA_M: tl.constexpr = tm.BLOCK_M * tm.GQA
        kq_fake = gl.zeros([tm.BLOCK_N, tm.BLOCK_M * tm.GQA], layout=kq_reg_layout, dtype=gl.float32)
        res_fake = kq_fake.reshape(tm.BLOCK_N, tm.BLOCK_M, tm.GQA).sum(2)
        # sm100 have enough registers to store w in reg for bf16.
        w_in_reg: tl.constexpr = True
        while (block_q_idx < num_q_blocks):
            seq_k_starts, seq_k_ends, kv_start, num_kv_blocks = self._next_varlen_schedule(block_q_idx)
            seq_k_starts_cur = gl.convert_layout(seq_k_starts, gl.SliceLayout(0, res_fake.type.layout))
            seq_k_ends_cur = gl.convert_layout(seq_k_ends, gl.SliceLayout(0, res_fake.type.layout))

            pipeline, consumer_qw = pipeline.piped_get_consumer("q,weights")
            if w_in_reg:
                w_may_smem = consumer_qw.load_xmem(f"weights", gl.SliceLayout(0, kq_reg_layout), shape=(MMA_M,), check_bank_conflict=True)
                if w_may_smem.dtype != gl.float32:
                    # we use blackwell f32x2 math, so we must convert weight to f32 to use them. 
                    w_may_smem = w_may_smem.to(gl.float32)
            else:
                w_may_smem = consumer_qw.get_smem("weights").reshape((MMA_M,))
            for j in range(num_kv_blocks):
                pipeline, logits = self.run_kq_fwd_compute_step(pipeline, w_may_smem, part_idx)
                if self.is_compressed_logits:
                    # complex store pattern, we must store them manually.
                    o_desc = pipeline.get_tensor_desc("o")
                    o_ptr = o_desc.get_offseted_ptr()
                    offs_m = gl.arange(0, tm.BLOCK_M, layout=gl.SliceLayout(0, res_fake.type.layout)) + block_q_idx * tm.BLOCK_M
                    offs_n = gl.arange(0, tm.BLOCK_N, layout=gl.SliceLayout(1, res_fake.type.layout)) + kv_start + j * tm.split_kv_size + part_idx * tm.BLOCK_N
                    mask = (offs_n[:, None] >= seq_k_starts_cur[None, :]) & (offs_n[:, None] < seq_k_ends_cur[None, :])
                    mask = mask & (offs_m[None, :] < tm.S_qo)
                    ptrs = o_ptr + offs_m[None, :] * tm.max_seqlen + offs_n[:, None] - seq_k_starts_cur[None, :]
                    # tl.static_print(ptrs.shape, logits.shape, mask.shape)
                    gl.store(ptrs, gl.convert_layout(logits, res_fake.type.layout, assert_trivial=True), mask=mask)
                else:
                    # just write, we don't care logits outside varlen block.
                    pipeline = pipeline.masked_store_recorded("o", logits, 
                        offsets=ttfs.offsets(S_qo=block_q_idx * tm.BLOCK_M, S_kv_buf=kv_start + j * tm.split_kv_size + part_idx * tm.BLOCK_N), 
                        cast_to_default_layout=False, force_mask_sym="S_kv_buf")
            block_q_idx += self.num_sm
            consumer_qw.finish()

        pipeline = pipeline.finalize_recorder()
    
    @ttfs.gluon_jit 
    def umma_partition(self, pipeline: ttfs.io.PipelineWS, part_idx: tl.constexpr, record_part_idx: tl.constexpr):
        pipeline = pipeline.prepare_recorder(record_part_idx, f"umma_{part_idx}")
        tm = self.tensors
        MMA_M: tl.constexpr = tm.BLOCK_M * tm.GQA
        num_q_blocks = tl.cdiv(tm.S_qo, tm.BLOCK_M)

        block_q_idx = self.sm_id
        while (block_q_idx < num_q_blocks):
            seq_k_starts, seq_k_ends, kv_start, num_kv_blocks = self._next_varlen_schedule(block_q_idx)
            pipeline, consumer_qw = pipeline.piped_get_consumer("q,weights")
            q_smem = consumer_qw.get_smem("q").reshape((MMA_M, tm.D))
            for j in range(num_kv_blocks):
                pipeline = self.run_kq_fwd_umma_step(pipeline, q_smem)
            block_q_idx += self.num_sm

        pipeline = pipeline.finalize_recorder()

@ttfs.gluon_jitx_kernel
def indexer_logits_mqa_prefill_gluon_kernel(
    args: DSAIndexerLogitsPersistVarlenArgs,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    rec: ttfs.recorder.EventRecorderArgs,
) -> None:
    sm_id = tl.program_id(0)
    tm = args.tensors
    q_dtype: tl.constexpr = tm.q.get_dtype_cw().value
    is_quantized: tl.constexpr = q_dtype.primitive_bitwidth <= 8
    num_block_per_segment: tl.constexpr = args.tensors.split_kv_size // BLOCK_N
    num_stages: tl.constexpr = 2 if not is_quantized else 3
    tensors = ttfs.dict(
        q=ttfs.io.piped_cfg(num_stage=num_stages, num_consumers=num_block_per_segment),
        weights=ttfs.io.piped_cfg(num_stage=num_stages, num_consumers=num_block_per_segment),
        kv_indices=ttfs.io.sync_cfg(),
        o=ttfs.io.sync_cfg(),
        k=ttfs.io.piped_cfg(num_stage=num_stages, num_consumers=num_block_per_segment),
        k_scales=ttfs.io.piped_cfg(num_stage=num_stages, num_consumers=num_block_per_segment) if is_quantized else None,
    )
    for j in tl.static_range(num_block_per_segment):
        tensors = tensors.additem(f"kq_{j}", ttfs.io.create_ext_tmem_desc([tm.BLOCK_N, tm.BLOCK_M * tm.GQA], gl.float32, num_stage=2))

    pipeline = ttfs.io.create_pipelined_iter_ws_v2(
        tm,
        tensors=tensors,
        async_groups=ttfs.dict(k_io="k,k_scales", qw="q,weights") if is_quantized else ttfs.dict(k_io="k", qw="q,weights"),
        recorder_args=rec,
    )

    mainloop = IndexerLogitsMainloop(
        tensors=tm,
        num_partition=num_block_per_segment,
        is_compressed_logits=args.is_compressed_logits,
        sm_id=sm_id,
        num_sm=tl.num_programs(0),
        is_indexer_topk=args.is_indexer_topk,
        sm_scale=args.sm_scale,
    )
    parts = ()
    math_num_reg: tl.constexpr = 240
    other_num_reg: tl.constexpr = 24
    for j in tl.static_range(num_block_per_segment):
        parts += (ttfs.gl.WarpSpecializePartition(
            IndexerLogitsMainloop.compute_partition,
            args=(mainloop, pipeline, j, j),
            num_warps=4 if j != 0 else -1,
            num_regs=math_num_reg if j != 0 else -1,
        ),)
    parts += (ttfs.gl.WarpSpecializePartition(
        IndexerLogitsMainloop.load_qk_partition,
        args=(mainloop, pipeline, num_block_per_segment),
        num_warps=1,
        num_regs=other_num_reg,
    ),)
    parts += (ttfs.gl.WarpSpecializePartition(
        IndexerLogitsMainloop.umma_partition,
        args=(mainloop, pipeline, 0, num_block_per_segment + 1),
        num_warps=1,
        num_regs=other_num_reg,
    ),)

    ttfs.gl.warp_specialize(parts)

    pipeline.close()