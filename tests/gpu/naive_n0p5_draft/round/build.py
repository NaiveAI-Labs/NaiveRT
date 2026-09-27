"""CPU-only preparation of draft DSOs. CUDA_VISIBLE_DEVICES must be empty."""
import os
from pathlib import Path
import sys

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise RuntimeError('set CUDA_VISIBLE_DEVICES= before this CPU-only build')
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))

from naivert.kernels.naive_n0p5.dspark_context import load_context
from naivert.kernels.naive_n0p5.dspark_attention import load_attention
from naivert.kernels.naive_n0p5.dspark_mlp import load_mlp
from naivert.kernels.naive_n0p5.dspark_head_markov import load_head_markov
from naivert.kernels.naive_n0p5.dspark_sampling import load_sampling
from naivert.kernels.naive_n0p5.dspark_tail import load_tail

for build in (load_context,load_attention,load_mlp,load_head_markov,load_sampling,load_tail):
    module=build()
    print('BUILT',module.__file__,flush=True)
module=load_sampling(True)
print('BUILT',module.__file__,flush=True)
module=load_tail(True)
print('BUILT',module.__file__,flush=True)
