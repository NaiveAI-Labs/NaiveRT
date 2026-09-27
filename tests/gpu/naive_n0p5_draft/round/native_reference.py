"""Pinned pre-port CUDA draft calls for same-target request diagnostics.

Argument mapping is copied from core/run.py's validated native control. These
kernels share current prepared weights/buffers but do not use public entries.
"""
from run import load, sha


class NativeDraftReference:
    def __init__(self, core, raw, path):
        import torch
        assert sha(path) == '439f9612b2d2bb7bd48669ec16e80005e7e04709ef2a6bd510eacadcaca8df66'
        self.native = load('naivert_dspark_optimized', path)
        self.core, self.buffers, self.weights = core, core.buffers, core.weights
        self.peers = torch.tensor([b.mailbox.data_ptr() for b in core.buffers], dtype=torch.int64)
        # These reference-only tensors satisfy the old binding. The normal native
        # instructions do not export diagnostics, but its interface requires storage.
        extra = []
        for r in range(8):
            e = {}
            for n, shape, dt in (
                ("fc_partial", (8, 4096), torch.bfloat16), ("ctx", (8, 4096), torch.bfloat16),
                ("kv_raw", (5, 2, 8, 128), torch.bfloat16), ("q", (8, 512), torch.bfloat16),
                ("k", (8, 128), torch.bfloat16), ("scores", (4, 8, 1032), torch.float32),
                ("partial", (8, 4096), torch.bfloat16), ("norm", (8, 4096), torch.bfloat16),
                ("logits", (7, 19072), torch.float32), ("lat_r", (256,), torch.bfloat16),
                ("latent", (256,), torch.bfloat16), ("bias", (19072,), torch.bfloat16),
                ("step_logits", (19072,), torch.float32), ("cand_val", (8,), torch.float32),
                ("cand_idx", (8,), torch.int64), ("lat_trace", (7, 256), torch.bfloat16),
                ("bias_trace", (7, 19072), torch.bfloat16), ("logit_trace", (7, 19072), torch.float32),
                ("cta_val", (298,), torch.float32), ("cta_idx", (298,), torch.int32),
            ):
                e[n] = torch.empty(shape, dtype=dt, device=r)
            e["w1"] = raw["markov_head.markov_w1.weight"].to(r)
            e["w2"] = raw["markov_head.markov_w2.weight"][r * 19072:(r + 1) * 19072].to(r)
            e["gu"] = [w["gate_up"].reshape(1024, 4096).split(512) for w in core.weights[r].mlp]
            extra.append(e)
        self.extra = extra

    def context(self, r):
        core, native, extra, peers = self.core, self.native, self.extra, self.peers
        buffers, weights = core.buffers, core.weights
        sampling = True
        b, w, e = buffers[r], weights[r], extra[r]
        c = w.context
        native.fc_ctx_kv(b.target_hidden, c["fc"], c["hidden_norm"], b.ctx_valid,
            e["fc_partial"], b.ctx_sum, e["ctx"], b.ssq, b.barrier, peers, b.epoch,
            2, r, b.wdog, core.watchdog_ns, c["k"], c["v"], c["k_norm"],
            b.rope_cos, b.rope_sin, b.ctx_pos, b.kv_dst, e["kv_raw"], b.kv_k,
            b.kv_v, b.kv_partials, 1e-5)

    def run(self, r):
        self.context(r)
        core, native, extra, peers = self.core, self.native, self.extra, self.peers
        buffers, weights = core.buffers, core.weights
        sampling = True
        b, w, e = buffers[r], weights[r], extra[r]
        for layer in range(5):
            v = w.attention[layer]
            native.attn_rows8_swa(b.resid[layer], v["input_ln"], v["q"], v["k"], v["v"],
                v["q_norm"], v["k_norm"], v["o"], b.rope_cos, b.rope_sin, b.blk_pos,
                b.key_add, b.kv_k, b.kv_v, b.qk_raw, e["q"], e["k"], b.v_blk,
                e["scores"], b.attn_out, e["partial"], b.hid_a, b.barrier, peers,
                b.epoch, layer, r, b.wdog, core.watchdog_ns, b.split_scratch, 1024, 1e-5, e["norm"])
            v = w.mlp[layer]
            native.mlp_rows8(b.hid_a, v["post_ln"], *e["gu"][layer], v["down_w"],
                b.activation, e["partial"], b.resid[layer + 1], b.barrier, peers,
                b.epoch, layer, r, b.wdog, core.watchdog_ns, 1e-5)
        native.head_markov(e["w1"], e["w2"], e["logits"], b.block_ids, b.draft_tokens,
            b.prev_token, e["lat_r"], e["latent"], e["bias"], e["step_logits"], e["cand_val"],
            e["cand_idx"], e["lat_trace"], e["bias_trace"], b.logit_trace if sampling else e["logit_trace"],
            e["cta_val"], e["cta_idx"], b.barrier, peers, b.epoch, r, True, b.wdog,
            core.watchdog_ns, w.bias_table, b.resid[5], w.head["norm_gamma"],
            w.head["lm_head"], e["norm"], b.control, 1e-5, 0, b.sampling_state if sampling else None)
