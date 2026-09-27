"""Exercise vendored FP32-accumulating reduction before changing its default.

The runtime already owns both operation workspaces. Only the dispatch choice
changes here; device sources, graph ownership and cleanup use the runtime path.
"""
import hashlib
from pathlib import Path

from naivert.prefill.native.executor import NativePrefill as Baseline


class NativePrefill(Baseline):
    def __init__(self, *args, **kwargs):
        kwargs['collective_backend'] = 'flashinfer'
        super().__init__(*args, **kwargs)
        self.collectives.reduce_scatter = True
        self.collectives.version['reduce_scatter'] = True
        self.versions['reduction-experiment-sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


if __name__ == '__main__':
    # Run an existing harness with this constructor, including request_set.py.
    import runpy
    import sys
    import naivert.prefill.native.executor as executor
    harness = Path(sys.argv.pop(1)).resolve()
    executor.NativePrefill = NativePrefill
    sys.path.insert(0, str(harness.parent))
    runpy.run_path(str(harness), run_name='__main__')
