from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from app.vloc.device import resolve_device


class GlobalEmbedder:
    def __init__(self, device: str = "auto") -> None:
        self.device = resolve_device(device)
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)
        self.transform = weights.transforms()
        print(f"[VLOC] GlobalEmbedder device: {self.device}")

    @torch.inference_mode()
    def encode_bgr(self, bgr: np.ndarray) -> np.ndarray:
        rgb = bgr[..., ::-1].copy()
        img = Image.fromarray(rgb)
        x = self.transform(img).unsqueeze(0).to(self.device)
        y = self.model(x).squeeze(0).detach().float().cpu().numpy()
        y /= (np.linalg.norm(y) + 1e-12)
        return y.astype(np.float32)


def bgr_to_torch_gray(bgr: np.ndarray, device: str) -> torch.Tensor:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Waiting for the BGR image of HxWx3.')
    gray = (
        0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
    ).astype(np.float32)
    gray /= 255.0
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
