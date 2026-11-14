from __future__ import annotations

import torch


def resolve_device(requested: str) -> str:
    req = str(requested).strip().lower()
    if req not in {"cuda", "gpu"}:
        raise RuntimeError(
            'This project is collected in GPU-only mode. '
            'Allowable device: cuda.'
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            'GPU-only mode: torch.cuda.is_available() ==False.\n'
            'We need to install the PyTorch CUDA and make sure the NVIDIA driver is active.'
        )

    return "cuda"
