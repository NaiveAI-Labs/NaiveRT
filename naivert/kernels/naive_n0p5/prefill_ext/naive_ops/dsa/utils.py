from typing import Optional

import torch

HAS_TMA_GATHER: Optional[bool] = None

def is_blackwell(arch: tuple[int, int]):
    return arch[0] == 10 or arch[0] == 11

def check_has_tma_gather() -> bool:
    global HAS_TMA_GATHER
    if HAS_TMA_GATHER is None:
        major, minor = torch.cuda.get_device_capability()
        HAS_TMA_GATHER =  major == 10
    return HAS_TMA_GATHER

def get_local_varlen_metadata(global_cu_seqlens, rank, world_size):
    """
    Args:
        global_cu_seqlens: [num_seq + 1] absolute boundaries in the packed buffer.
        rank: Current rank (0 to P-1).
        world_size: CP world size (P).
    """
    # 1. Calculate local token counts per sequence
    # Tokens in [0, L) belonging to rank r: ceil((L - r) / P)
    def count_on_rank(L, r, P):
        return torch.clamp((L - r + P - 1) // P, min=0)

    # counts[s] is how many tokens of sequence s are on this rank
    counts = count_on_rank(global_cu_seqlens[1:], rank, world_size) - \
             count_on_rank(global_cu_seqlens[:-1], rank, world_size)
    
    # 2. Local cu_seqlens for the FA2/kernel interface
    local_cu_seqlens = torch.zeros_like(global_cu_seqlens)
    local_cu_seqlens[1:] = torch.cumsum(counts, dim=0)
    
    # 3. Phase: The relative index of our FIRST local token within the original sequence
    # For seq s starting at global_L: first token on rank r is global index:
    # G_first = global_L + ((rank - global_L) % world_size)
    # The 'phase' is (G_first - global_L)
    phases = (rank - global_cu_seqlens[:-1]) % world_size

    return local_cu_seqlens.to(torch.int32), int(counts.max().item()), phases.to(torch.int32)