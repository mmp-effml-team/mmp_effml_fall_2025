import math
import os
from dataclasses import dataclass
import socket, datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


@dataclass
class AttnConfig:
    d_model: int = 64
    n_heads: int = 4
    seq_len: int = 32
    world_size: int = 4


class AttentionParams(nn.Module):
    """
    Common parameters for all attention implementations, so the comparison is fair.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def _project(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        """
        x: [S, D]
        return: [H, S, Dh]
        """
        S, D = x.shape
        h = self.n_heads
        dh = self.head_dim
        out = linear(x)  # [S, D]
        out = out.view(S, h, dh).transpose(0, 1).contiguous()  # [H, S, Dh]
        return out

    def project_qkv(self, x: torch.Tensor):
        q = self._project(x, self.W_q)
        k = self._project(x, self.W_k)
        v = self._project(x, self.W_v)
        return q, k, v

    def out_proj(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [H, S, Dh] 
        return: [S, D]
        """
        H, S, Dh = context.shape
        x = context.transpose(0, 1).contiguous().view(S, H * Dh)
        return self.W_o(x)


# ============
# Baseline MHA
# ============

def vanilla_attention(x_full: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    """
    Standard MHA in a single process.
    x_full: [S, D]
    return: [S, D]
    """
    q, k, v = params.project_qkv(x_full)  # [H, S, Dh] each
    scale = 1.0 / math.sqrt(params.head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [H, S, S]
    probs = torch.softmax(scores, dim=-1)
    context = torch.matmul(probs, v)  # [H, S, Dh]

    out = params.out_proj(context)  # [S, D]
    return out


# ==========================
# Helper: chunking utilities
# ==========================

def get_local_chunk(x_full: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """
    Logical split along seq dim between ranks.
    """
    S = x_full.size(0)
    assert S % world_size == 0, "seq_len must be divisible by world_size"
    chunk = S // world_size
    start = rank * chunk
    end = start + chunk
    return x_full[start:end].contiguous()


def gather_from_chunks(x_local: torch.Tensor) -> torch.Tensor:
    """
    All-gather along seq dim (concatenate chunks sequentially).
    x_local: [S_local, D]
    return: [S_total, D] (same on all ranks)
    """
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(x_local) for _ in range(world_size)]
    dist.all_gather(tensor_list, x_local)
    return torch.cat(tensor_list, dim=0)


# ============================================
# AllGather Context Parallel Attention -- 2 score points
# ============================================

def allgather_context_parallel_attention(
    x_full: torch.Tensor,
    params: AttentionParams,
) -> torch.Tensor:
    """
    Context Parallel:
      - Each rank gets its own chunk along seq.
      - Compute Q only for its own chunk.
      - K/V all_gather from all ranks.
      - Attention for local Q against global K/V.
      - Gather local block results back via all_gather.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x_local = get_local_chunk(x_full, rank, world_size)  # [S_local, D]

    q_local, k_local, v_local = params.project_qkv(x_local)  # [H, S_local, Dh]

    # TODO: Gather K/V from all chunks

    # TODO: Compute attention for local Q against global K/V

    # TODO: Gather all local outputs to get [S_total, D]
    return ...  # [S_total, D]


# ============================================
# 2) Ring Attention —- 3 score points
# ============================================

def ring_attention(x_full: torch.Tensor, params: AttentionParams) -> torch.Tensor:
    """
    Ring Attention with online softmax:
      - q_local is fixed at the rank
      - rotate around the ring, receiving K/V chunks
      - at each step update (m, l, acc) without storing global K/V
      - at the end context = acc / l
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Local input
    x_local = get_local_chunk(x_full, rank, world_size)  # [S_local, D]
    q_local, k_local, v_local = params.project_qkv(x_local)  # [H, S_local, Dh] each

    H, S_local, Dh = q_local.shape
    scale = 1.0 / math.sqrt(params.head_dim)

    # Initialize online-softmax statistics
    m = torch.full((H, S_local), float("-inf"), dtype=q_local.dtype)
    l = torch.zeros((H, S_local), dtype=q_local.dtype)
    acc = torch.zeros((H, S_local, Dh), dtype=q_local.dtype)

    # Go around the ring for all K/V chunks
    send_to = ...
    recv_from = ...

    for _ in range(world_size):
        # 1) compute contribution of current K/V chunk
        # TODO: Compute contribution of current K/V chunk

        # 2) ring transfer of next chunk (p2p without deadlock)
        if world_size > 1:
            # TODO: Ring transfer of next chunk (p2p without deadlock)
            pass


    # Final context at local positions: [H, S_local, Dh]
    context_local = acc / l.unsqueeze(-1).clamp_min(1e-12)
    out_local = params.out_proj(context_local)  # [S_local, D]

    # Gather output along seq back
    out_full = gather_from_chunks(out_local)
    return out_full


# ==========================
# Distributed setup
# ==========================

def setup_process(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0") # for macos
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=120)
    )


def broadcast_parameters(params: AttentionParams, src: int = 0):
    """
    Synchronize weights between ranks.
    """
    # TODO: Broadcast parameters between ranks
    raise NotImplementedError("Not implemented yet")


# ==========================
# Worker
# ==========================

def worker(rank: int, world_size: int, cfg: AttnConfig):
    torch.set_num_threads(max(1, os.cpu_count() // max(1, world_size)))
    setup_process(rank, world_size)

    torch.manual_seed(0)

    # Initialize common attention parameters
    params = AttentionParams(cfg.d_model, cfg.n_heads)

    # Synchronize weights
    broadcast_parameters(params, src=0)

    # Global input [S, D]
    if rank == 0:
        x_full = torch.randn(cfg.seq_len, cfg.d_model)
    else:
        x_full = torch.zeros(cfg.seq_len, cfg.d_model)
    dist.broadcast(x_full, src=0)

    # Reference output from vanilla attention (computed on rank 0 and broadcast)
    if rank == 0:
        base_out = vanilla_attention(x_full, params)  # [S, D]
    else:
        base_out = torch.zeros(cfg.seq_len, cfg.d_model)
    dist.broadcast(base_out, src=0)

    dist.barrier()

    # --- AllGather Context Parallel ---
    out_cp = allgather_context_parallel_attention(x_full, params)
    dist.barrier()

    # --- Ring Attention ---
    out_ring = ring_attention(x_full, params)
    dist.barrier()

    # Checks only on rank 0
    if rank == 0:
        def max_err(t):
            return (t - base_out).abs().max().item()

        print(f"[world_size={world_size}] Max error vs Vanilla:")
        print(f"  AllGather CP : {max_err(out_cp):.6e}")
        print(f"  RingAttention: {max_err(out_ring):.6e}")

        # Additionally check shapes
        assert out_cp.shape == base_out.shape
        assert out_ring.shape == base_out.shape
        assert (out_cp - base_out).abs().max().item() < 1e-5
        assert (out_ring - base_out).abs().max().item() < 1e-5
        print("Shapes OK, comparison OK.")

    dist.destroy_process_group()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    port = find_free_port()
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["MASTER_PORT"] = str(port)
    cfg = AttnConfig(
        d_model=64,
        n_heads=4,
        seq_len=32,
        world_size=4,
    )

    mp.spawn(
        worker,
        args=(cfg.world_size, cfg),
        nprocs=cfg.world_size,
        join=True,
    )