"""Capture-ready Naive N0.5 draft compute; request control belongs to the round owner."""
from naivert.models.naive_n0p5.dspark import FIRST_LOGIT_ROW, RMS_EPSILON, WINDOW_SIZE


class DraftCore:
    """D12, five D3/D4 pairs, D56 with stable per-rank argument storage.

    All ranks must execute in lockstep on distinct devices with P2P enabled.
    Methods launch onto each rank's current stream and can be captured by the
    surrounding graph. They allocate no tensors and read no device scalars.

    ``context`` and ``body`` expose the serial dependency split. ``stage_context``
    writes separate KV destinations for accepted-prefix commit by DraftTail.
    The round owner supplies the fork/join and initialization. D12 advances the
    mailbox epoch exactly once per block; do not replay ``body`` repeatedly
    without the matching D12. Concurrent target/draft launches also require
    a proven joint-residency schedule; independent residency is insufficient.
    """
    phase_names = ("d12",) + tuple(n for i in range(5) for n in (f"d3.{i}", f"d4.{i}")) + ("d56",)

    def __init__(self, kernels, weights, buffers, *, watchdog_ns=5_000_000_000):
        self.kernels, self.weights, self.buffers = kernels, tuple(weights), tuple(buffers)
        if len(self.weights) != 8 or len(self.buffers) != 8 or watchdog_ns < 0:
            raise ValueError("draft requires eight weight/buffer sets and a nonnegative watchdog")
        if (len({id(b) for b in self.buffers}) != 8
                or any(b.rank != r or b.device.index != kernels.devices[r]
                       or b.device.type != "cuda" for r, b in enumerate(self.buffers))
                or len({(b.capacity, b.sampling) for b in self.buffers}) != 1):
            raise ValueError("draft buffers must agree on geometry/sampling and match the TP8 devices")
        if any(len(w.attention) != 5 or len(w.mlp) != 5 for w in self.weights):
            raise ValueError("draft requires five attention/MLP layers")
        self.peers = tuple(b.mailbox for b in self.buffers)
        self.watchdog_ns = watchdog_ns

    def context(self, rank):
        b = self.buffers[rank]
        self._context(rank, b.target_hidden, b.ctx_valid, b.ctx_pos, b.kv_dst, b.kv_k, b.kv_v)

    def stage_context(self, rank, stage):
        """D12 after target forward; join with sampling before the tail.

        The round owner must finish prompt KV ingestion before using this
        schedule. It runs ``body`` first, then target verification, staged D12
        and sampling, then the staged tail. There is exactly one D12 per round.
        """
        b = self.buffers[rank]
        if stage.buffers is not b:
            raise ValueError("staged context belongs to another draft buffer set")
        self._context(rank, stage.features, stage.valid, b.blk_pos,
                      stage.destinations, stage.key, stage.value)

    def _context(self, rank, features, valid, positions, destinations, key, value):
        b, w = self.buffers[rank], self.weights[rank].context
        self.kernels.context.fc_ctx_kv(
            features, w["fc"], w["hidden_norm"], valid,
            b.empty_bf16, b.ctx_sum, b.empty_bf16, b.ssq, b.barrier, self.peers,
            b.epoch, rank, b.wdog, self.watchdog_ns, w["k"], w["v"], w["k_norm"],
            b.rope_cos, b.rope_sin, positions, destinations, b.empty_bf16,
            key, value, b.kv_partials, RMS_EPSILON)

    def attention(self, rank, layer):
        b, w = self.buffers[rank], self.weights[rank].attention[layer]
        self.kernels.attention.attn_rows8_swa(
            b.resid[layer], w["input_ln"], w["q"], w["k"], w["v"], w["q_norm"],
            w["k_norm"], w["o"], b.rope_cos, b.rope_sin, b.blk_pos, b.key_add,
            b.kv_k, b.kv_v, b.qk_raw, b.empty_bf16, b.empty_bf16, b.v_blk,
            b.empty_float, b.attn_out, b.empty_bf16, b.hid_a, b.barrier, self.peers,
            b.epoch, layer, rank, b.wdog, self.watchdog_ns, b.split_scratch,
            WINDOW_SIZE, RMS_EPSILON, b.empty_bf16)

    def mlp(self, rank, layer):
        b, w = self.buffers[rank], self.weights[rank].mlp[layer]
        self.kernels.mlp.mlp(
            b.hid_a, w["post_ln"], w["gate_up"], w["down_w"], b.activation,
            b.empty_bf16, b.resid[layer + 1], b.barrier, self.peers, b.epoch,
            b.wdog, rank, layer, self.watchdog_ns, RMS_EPSILON)

    def head_markov(self, rank):
        b, w = self.buffers[rank], self.weights[rank]
        self.kernels.head_markov.head_markov(
            b.empty_bf16, b.empty_bf16, b.empty_float, b.block_ids, b.draft_tokens,
            b.prev_token, b.empty_bf16, b.empty_bf16, b.empty_bf16, b.empty_float,
            b.empty_float, b.empty_long, b.empty_bf16, b.empty_bf16, b.logit_trace,
            b.empty_float, b.empty_int, b.empty_int, self.peers, b.epoch, rank,
            True, b.wdog, self.watchdog_ns, w.bias_table, b.resid[5],
            w.head["norm_gamma"], w.head["lm_head"], b.empty_bf16, b.control,
            RMS_EPSILON, FIRST_LOGIT_ROW, b.sampling_state if b.sampling else None,
            b.sampling_params if b.sampling else None)

    def body(self, rank):
        for layer in range(5):
            self.attention(rank, layer)
            self.mlp(rank, layer)
        self.head_markov(rank)

    def run(self, rank):
        self.context(rank)
        self.body(rank)
