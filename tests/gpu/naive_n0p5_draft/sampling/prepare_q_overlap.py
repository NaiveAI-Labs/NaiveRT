"""Generate a private split-q sampler; leave the production header unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

PIN = 'c5a60584b8a3d019926a665b9353d9705025cb0a40e3876e82045986c1faaace'
FUNCTIONS = ('partial_maximum', 'local_maximum', 'exponentiate', 'local_sum',
             'global_maximum', 'global_inverse', 'normalize')


def generate(src):
    src = src.replace('namespace naivert_dspark_sharded', 'namespace split_shard')
    src = src.replace('#include "random.cuh"', '#include "naivert/body/dspark_sampling/random.cuh"')
    src = src.replace('#include "sync.cuh"', '#include "naivert/body/dspark_sampling/sync.cuh"')
    for name in FUNCTIONS:
        begin = src.index('__global__ void '+name+'(')
        brace = src.index('{', begin)
        depth, end = 1, brace+1
        while depth:
            depth += (src[end] == '{') - (src[end] == '}')
            end += 1
        body = src[begin:end]
        if name in ('global_maximum', 'global_inverse'):
            assert 'row = threadIdx.x;' in body
            body = body.replace('row = threadIdx.x;', 'row = First + threadIdx.x;')
            body = body.replace('row < (a.anchor ? 1 : 15)', 'row < First + (a.anchor ? 1 : Count)')
        elif name in ('local_maximum', 'local_sum'):
            assert 'row = blockIdx.x' in body
            body = body.replace('row = blockIdx.x', 'row = First + blockIdx.x')
        else:
            assert 'row = blockIdx.y' in body
            body = body.replace('row = blockIdx.y', 'row = First + blockIdx.y')
        src = src[:begin]+'template<int First, int Count>\n'+body+src[end:]
    src = src.replace('inline void run(Args', 'inline void run_target(Args')
    src = src.replace('distributions = a.anchor ? 1 : 15;', 'distributions = a.anchor ? 1 : 8;')
    for name in FUNCTIONS:
        src = src.replace(name+'<<<', name+'<0, 8><<<')
    pos = src.index('// Mailboxes/flags are zero-initialized once.')
    prep = r'''
// Three independent phase counters per rank: q input reuse, maxima, sums.
// The caller joins this work before run_target consumes proposal_q / q.
// Prior-round consumers must finish before any rank starts its next prepare_q.
struct QFlags { unsigned long long* rank[8]; };
inline void q_rendezvous(Args a, QFlags flags, int stage, cudaStream_t stream) {
  for (int r=0; r<8; ++r) a.flags[r]=flags.rank[r]+3*stage;
  rendezvous<<<1, 32, 0, stream>>>(a, -1);
}
inline void prepare_q(Args a, Scratch* s, QFlags flags, cudaStream_t stream) {
  q_rendezvous(a, flags, 0, stream);
  partial_maximum<8, 7><<<dim3(tiles, 7), threads, 0, stream>>>(a, s);
  local_maximum<8, 7><<<7, threads, 0, stream>>>(a, s);
  q_rendezvous(a, flags, 1, stream);
  global_maximum<8, 7><<<1, 32, 0, stream>>>(a, s);
  exponentiate<8, 7><<<dim3(tiles, 7), threads, 0, stream>>>(a, s);
  local_sum<8, 7><<<7, threads, 0, stream>>>(a, s);
  q_rendezvous(a, flags, 2, stream);
  global_inverse<8, 7><<<1, 32, 0, stream>>>(a, s);
  normalize<8, 7><<<dim3(tiles, 7), threads, 0, stream>>>(a, s);
}

'''
    return src[:pos]+prep+src[pos:]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source = a.root/'naivert/body/dspark_sampling/sharded.cuh'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PIN
    a.out.mkdir(parents=True, exist_ok=False)
    generated = generate(source.read_text())
    (a.out/'split_q.cuh').write_text(generated)
    (a.out/'source.json').write_text(json.dumps(dict(
        reference_sha256=PIN, generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        method='Original softmax reduction order, partitioned into rows 0..7 and 8..14; independent q phase flags'),
        indent=2)+'\n')
