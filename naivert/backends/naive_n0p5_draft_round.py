"""One device-driven Naive N0.5 draft/verify/sample round per TP rank."""
import ctypes as C
from functools import partial

import torch

from naivert.backends.naive_n0p5_draft_tail import DraftTail
from naivert.kernels.naive_n0p5.dspark_sampling import TargetSampler
from naivert.models.naive_n0p5.dspark import MASK_TOKEN_ID, StagedDraftContext, VOCAB, sampling_params_tensor
from naivert.runtime.cuda_backend import CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH
from naivert.runtime.dspark_graph import fork_staged_context
from naivert.runtime.mtp_contract import _DecodeControlLayout
from naivert.runtime.pdl_graph_edges import install_programmatic_edges
from naivert.runtime.sequencer import _SequencerOutputRecordLayout


class DraftRound:
    """Capture once; launch once per request; the tail repeats on the device.

    The caller owns target prefill and supplies its per-rank feature history.
    ``initialize`` is a request-boundary operation and requires every prior
    graph to be drained. There are no Python callbacks inside a steady round.
    SessionEngine reads the published host ring and streams committed tokens to the caller.
    """
    # kernels that griddepcontrol.wait before touching draft state: D3, D4 (fork: D56 head)
    DRAFT_PDL_MARKERS=('dspark_attn_rows8','dspark_mlp_rows8','naivert_dspark_head_markov_kernel')

    def __init__(self, core, target, api, *, output_capacity, staged=True,persistent_sampler=False,draft_pdl=False):
        if staged and persistent_sampler:
            raise ValueError('persistent sampler requires a serial D12 schedule until joint residency is validated')
        if target.core is not core or not all(b.sampling for b in core.buffers):
            raise ValueError('round requires this core and its T=1 target binding')
        if type(output_capacity) is not int or not 1<=output_capacity<2**31:
            raise ValueError('invalid output capacity')
        self.core,self.target,self.api,self.staged = core,target,api,bool(staged)
        self.capacity=output_capacity;self.draft_pdl=bool(draft_pdl)
        self.host_ring=torch.zeros(output_capacity*C.sizeof(_SequencerOutputRecordLayout),dtype=torch.uint8,pin_memory=True)
        self.host_flags=torch.zeros(output_capacity,dtype=torch.int32,pin_memory=True)
        self.cancel=torch.zeros(2,dtype=torch.int32,pin_memory=True)
        self.sampler=TargetSampler(target.logits,core.buffers,[t.tokens for t in target.states],
                                   persistent=persistent_sampler)
        self.stages=([StagedDraftContext(b,t.features) for b,t in zip(core.buffers,target.states)] if staged else None)
        self.tail=DraftTail(core,target.states,self.sampler,[v.w.embedding for v in target.verifies],
            host_ring=self.host_ring,host_flags=self.host_flags,cancel=self.cancel,
            ring_capacity=output_capacity,stages=self.stages,chain=True)
        self.phases=tuple(self._rank_phases(r) for r in range(8))
        self.graphs,self.execs,self.audits=[],[],[]

    def _rank_phases(self,r):
        out=[] if self.staged else [('draft.d12',partial(self.core.context,r))]
        for i in range(5):
            out.extend(((f'draft.d3.{i}',partial(self.core.attention,r,i)),
                        (f'draft.d4.{i}',partial(self.core.mlp,r,i))))
        out.append(('draft.d56',partial(self.core.head_markov,r)))
        out.extend((f'verify.{n}',fn) for n,fn in self.target.phases[r])
        if self.staged: out.append(('draft.stage_d12',partial(self.core.stage_context,r,self.stages[r])))
        out.extend((('sample',partial(self.sampler.run,r,defer_consumed=True)),('tail',partial(self.tail.run,r))))
        return tuple(out)

    def capture(self):
        """Kernels/weights/providers must be prepared on all ranks first."""
        if self.graphs or self.execs: raise RuntimeError('round is already captured')
        for b in self.core.buffers: torch.cuda.synchronize(b.device)
        try:
            for r,b in enumerate(self.core.buffers):
                with torch.cuda.device(b.device),torch.cuda.stream(self.target.streams[r]):
                    graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=self.target.streams[r]):
                        for _,fn in self.phases[r]: fn()
                    self.graphs.append(graph)
                    raw=graph.raw_cuda_graph()
                    names=self.api.kernel_function_names(raw)
                    v=self.target.verifies[r]
                    pdl=None;positions=[]
                    if v.pdl:
                        markers=tuple(v.rt.pdl_markers)
                        if not markers: raise ValueError('target PDL marker policy is required')
                        positions=[i-1 for i,n in enumerate(names) if i and any(m in n for m in markers)]
                        if not positions: raise ValueError('target PDL policy matched no kernels')
                    if self.draft_pdl:
                        # Edges INTO D3/D4/D56: the destination waits for full completion of its
                        # primary before touching draft state (11 edges of the 12-kernel chain).
                        draft=[i-1 for i,n in enumerate(names) if i and any(m in n for m in self.DRAFT_PDL_MARKERS)]
                        if not draft: raise ValueError('draft PDL policy matched no kernels')
                        positions=sorted(set(positions)|set(draft))
                    if positions:
                        pdl=install_programmatic_edges(self.api,raw,from_positions=positions)
                    topology=fork_staged_context(self.api,raw,names) if self.staged else self.api.inspect_graph(raw)
                    self.audits.append(dict(rank=r,pdl=pdl,topology=topology,kernel_names=names))
                    handle=self.api.graph_instantiate(raw,flags=CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH)
                    self.execs.append(handle)
                    self.api.graph_upload(handle,self.target.streams[r].cuda_stream)
        except Exception:
            self.close()
            raise

    def set_sampling(self,*,penalty=None,context_tokens=None,temperature=None,top_p=None):
        """Sampling controls for the next request, set before the anchor is sampled.

        ``penalty`` = (repetition theta, frequency alpha, presence beta) or None for off: a token with
        count c > 0 in the context gets logit/theta (positive) or logit*theta (negative), then minus
        alpha*c + beta, before the temperature. ``temperature`` T (None: 1/NAIVERT_DSPARK_INV_TEMPERATURE
        or 1) scales target logits and the draft's Gumbel draw alike; ``top_p`` (None: 0.95) is the
        nucleus mass. ``context_tokens`` are the ids whose KV the caches hold; they seed the counts,
        and the tail adds every committed token. Returns whether the penalty is active."""
        theta,alpha,beta=(1.0,0.0,0.0) if penalty is None else tuple(float(x) for x in penalty)
        if not (0.0<theta<=10.0 and -10.0<=alpha<=10.0 and -10.0<=beta<=10.0):
            raise ValueError('repeat penalty out of range')
        if temperature is not None and not (0.0<float(temperature)<=100.0):
            raise ValueError('temperature must lie in (0, 100]')
        values=sampling_params_tensor(repetition=theta,frequency=alpha,presence=beta,
                                      inv_temperature=None if temperature is None else 1.0/float(temperature),
                                      top_p=0.95 if top_p is None else float(top_p))
        active=bool(values[3].item())
        ids=None if context_tokens is None else torch.as_tensor(list(context_tokens),dtype=torch.int64)
        if ids is not None and ids.numel() and not (0<=int(ids.min()) and int(ids.max())<VOCAB):
            raise ValueError('context token outside the vocabulary')
        for r,b in enumerate(self.core.buffers):
            with torch.cuda.device(b.device),torch.cuda.stream(self.target.streams[r]):
                b.sampling_params.copy_(values);b.rep_count.zero_()
                if active and ids is not None and ids.numel():
                    b.rep_count.add_(torch.bincount(ids.to(b.device),minlength=VOCAB).to(torch.int32))
        return active

    def initialize(self,features,*,position,anchor,seed,max_tokens,eos=None,ingest_from=0):
        """Consume target-prefilled features [ingest_from,position); caller has drained.

        Draft KV rows below ``ingest_from`` are already present (a restored
        session prefix); only the newer feature rows are ingested here.

        The anchor is the already sampled first output token. ``max_tokens``
        counts subsequent tokens published by this device loop. Target caches
        must contain that same prompt; target mailbox epochs are preserved.
        """
        n=self.core.buffers[0].capacity
        if (type(position) is not int or not 1<=position<=n-8 or
                type(anchor) is not int or not 0<=anchor<VOCAB or
                type(seed) is not int or not -(1<<63)<=seed<(1<<63) or
                type(max_tokens) is not int or not 1<=max_tokens<=self.capacity or
                (eos is not None and (type(eos) is not int or not 0<=eos<VOCAB)) or len(features)!=8 or
                type(ingest_from) is not int or not 0<=ingest_from<position):
            raise ValueError('invalid draft request position, anchor, seed, output bound or ingest start')
        for r,(feature,b) in enumerate(zip(features,self.core.buffers)):
            if (feature.shape!=(position,4096) or feature.dtype!=torch.bfloat16 or
                    feature.device!=b.device or not feature.is_contiguous()):
                raise ValueError(f'rank {r}: expected contiguous prompt features [position,4096]')
        self.host_ring.zero_();self.host_flags.zero_();self.cancel.zero_()
        # Serial D12 consumes the last prompt row in its first round. Staging
        # completes it here because that round starts directly at D3.
        end=position if self.staged else position-1
        for start in range(ingest_from,end,8):
            rows=min(8,end-start)
            for r,b in enumerate(self.core.buffers):
                with torch.cuda.device(b.device),torch.cuda.stream(self.target.streams[r]):
                    f=self.tail.fields[r]
                    f['cache_len'].fill_(start);f['ctx_rows'].fill_(rows)
                    b.target_hidden.zero_();b.target_hidden[:rows].copy_(features[r][start:start+rows])
                    self.tail.geometry(r)
            for r,b in enumerate(self.core.buffers):
                with torch.cuda.device(b.device),torch.cuda.stream(self.target.streams[r]):self.core.context(r)
        for r,(b,t,v) in enumerate(zip(self.core.buffers,self.target.states,self.target.verifies)):
            ctl=_DecodeControlLayout();s=ctl.value
            values=dict(decode_mode=1,speculative_depth=7,active_rows=8,row_capacity=8,
                row_stride=4096*2,committed_pos=position,speculative_base=position,
                max_output_tokens=max_tokens,accepted_count=1,main_cache_committed=position,
                main_cache_speculative_end=position+8,mtp_cache_committed=position,
                mtp_cache_speculative_end=position+8,reserved0=0xffffffff if eos is None else eos,reserved1=n-7)
            for key,value in values.items():setattr(s,key,value)
            raw=torch.tensor(list(bytes(ctl)),dtype=torch.uint8)
            with torch.cuda.device(b.device),torch.cuda.stream(self.target.streams[r]):
                f=self.tail.fields[r]
                f['cache_len'].fill_(position-1);f['ctx_rows'].fill_(1);f['anchor_token'].fill_(anchor)
                b.target_hidden.zero_();b.target_hidden[0].copy_(features[r][-1]);self.tail.geometry(r)
                b.block_ids.fill_(MASK_TOKEN_ID);b.block_ids[0]=anchor
                b.resid[0][0].copy_(v.w.embedding[anchor]);b.resid[0][1:].copy_(self.core.weights[r].mask_embedding.reshape(1,4096))
                b.sampling_state.copy_(torch.tensor([seed,0],dtype=torch.int64))
                b.rep_count[anchor]+=1   # the anchor is in the context of every block row
                self.tail.states[r][7].zero_();t.tokens.fill_(-1)
                t.position.fill_(position);t.previous_complete_ns.zero_();t.decode_control.copy_(raw)

    def launch(self):
        if len(self.execs)!=8: raise RuntimeError('capture the complete round before launch')
        for r,b in enumerate(self.core.buffers):
            with torch.cuda.device(b.device):self.api.graph_launch(self.execs[r],self.target.streams[r].cuda_stream)

    def close(self):
        """Drain before releasing graph references and their executable handles."""
        for stream in self.target.streams:stream.synchronize()
        for r,handle in enumerate(self.execs):
            with torch.cuda.device(self.core.buffers[r].device):self.api.graph_exec_destroy(handle)
        self.execs.clear();self.graphs.clear()
