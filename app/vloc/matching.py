from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from app.vloc.device import resolve_device
from app.vloc.vision import bgr_to_torch_gray


TorchFeatureDict = Dict[str, torch.Tensor]
NumpyFeatureTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class LightGlueMatcher:
    device: str = "auto"
    max_keypoints: int = 512

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
        try:
            from lightglue import LightGlue, SuperPoint  # type: ignore
            from lightglue.utils import rbd  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                'No LightGlue/SuperPoint found. Set:\n'
                "  pip install 'git+https://github.com/cvg/LightGlue.git'"
            ) from exc
        self._rbd = rbd
        self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(
            self.device
        )
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        print(f"[VLOC] LightGlueMatcher device: {self.device}")

    @torch.inference_mode()
    def extract_torch(self, bgr: np.ndarray) -> TorchFeatureDict:
        return self.extractor.extract(bgr_to_torch_gray(bgr, self.device))

    @torch.inference_mode()
    def extract_numpy(self, bgr: np.ndarray) -> NumpyFeatureTuple:
        return self.torch_to_numpy(self.extract_torch(bgr))

    @staticmethod
    def torch_to_numpy(feats: TorchFeatureDict) -> NumpyFeatureTuple:
        kpts = (
            feats["keypoints"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        )
        desc = (
            feats["descriptors"].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        )
        score_key = "scores" if "scores" in feats else "keypoint_scores"
        scores = (
            feats[score_key].squeeze(0).detach().float().cpu().numpy().astype(np.float32)
        )
        return kpts, desc, scores

    @staticmethod
    def numpy_to_torch(feats_np: NumpyFeatureTuple, device: str) -> TorchFeatureDict:
        kpts, desc, scores = feats_np
        kpts = np.array(kpts, dtype=np.float32, copy=True)
        desc = np.array(desc, dtype=np.float32, copy=True)
        scores = np.array(scores, dtype=np.float32, copy=True)
        return {
            "keypoints": torch.from_numpy(kpts).unsqueeze(0).to(device),
            "descriptors": torch.from_numpy(desc).unsqueeze(0).to(device),
            "scores": torch.from_numpy(scores).unsqueeze(0).to(device),
        }

    @torch.inference_mode()
    def match_from_torch(
        self,
        cur_feats: TorchFeatureDict,
        ref_feats: TorchFeatureDict,
        ransac_reproj_thresh: float = 3.0,
        use_usac_fast: bool = True,
    ) -> Tuple[Optional[np.ndarray], float, float, int]:
        if (
            int(cur_feats["keypoints"].shape[1]) < 16
            or int(ref_feats["keypoints"].shape[1]) < 16
        ):
            return None, 0.0, 1e9, 0

        out = self.matcher({"image0": cur_feats, "image1": ref_feats})
        out = self._rbd(out)

        matches0 = out["matches0"]
        if matches0.ndim > 1:
            matches0 = matches0.squeeze(0)

        valid = matches0 > -1
        match_cnt = int(valid.sum().item())
        if match_cnt < 12:
            return None, 0.0, 1e9, match_cnt

        kpts0 = cur_feats["keypoints"].squeeze(0)
        kpts1 = ref_feats["keypoints"].squeeze(0)

        mkpts0 = kpts0[valid].detach().cpu().numpy().astype(np.float32)
        mkpts1 = kpts1[matches0[valid]].detach().cpu().numpy().astype(np.float32)

        method = cv2.USAC_FAST if use_usac_fast and hasattr(cv2, "USAC_FAST") else cv2.RANSAC
        H, mask = cv2.findHomography(mkpts0, mkpts1, method, ransac_reproj_thresh)
        if H is None or mask is None:
            return None, 0.0, 1e9, match_cnt

        inliers = mask.ravel().astype(bool)
        inlier_cnt = int(inliers.sum())
        if inlier_cnt < 10:
            return None, 0.0, 1e9, match_cnt

        mk0_in = mkpts0[inliers]
        mk1_in = mkpts1[inliers]
        proj = cv2.perspectiveTransform(mk0_in.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - mk1_in, axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else 1e9
        inlier_ratio = float(inlier_cnt) / float(max(1, match_cnt))
        return H.astype(np.float64), inlier_ratio, rmse, match_cnt
