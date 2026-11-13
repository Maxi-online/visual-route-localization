from __future__ import annotations

import torch


def resolve_device(requested: str) -> str:
    req = str(requested).strip().lower()
    if req not in {"cuda", "gpu"}:
        raise RuntimeError(
            "Этот проект собран в режиме GPU-only. "
            "Допустимое устройство: cuda."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU-only режим: torch.cuda.is_available() == False.\n"
            "Нужно установить CUDA-сборку PyTorch и убедиться, что драйвер NVIDIA активен."
        )

    return "cuda"
