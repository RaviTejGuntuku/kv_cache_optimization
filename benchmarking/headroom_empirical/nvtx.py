from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def nvtx_range(message: str):
    try:
        import torch
    except Exception:
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
