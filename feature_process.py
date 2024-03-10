from typing import Dict
import numpy as np
from numpy.typing import NDArray
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import resize_image, numpy_image_to_torch, rbd


__all__ = ["SuperPointInfer", "LightGlueInfer"]

class SuperPointInfer:
    """SuperPoint infer object."""
    def __init__(self, superpoint_cfg, size=None, device="cuda"):
        self.device = torch.device(device)
        self.extractor = SuperPoint(**superpoint_cfg).eval().to(device)
        self.size = size

    def __call__(self, image: NDArray) -> Dict:
        """Main func."""
        if self.size is not None:
            image, _ = resize_image(image, self.size)
        tensor_img = numpy_image_to_torch(image)
        feats = self.extractor.extract(tensor_img.to(self.device))
        return feats
        

class LightGlueInfer:
    """LightGlue warper."""
    def __init__(self, lightglue_cfg, device="cuda"):
        self.device = torch.device(device)
        self.matcher = LightGlue(**lightglue_cfg).eval().to(self.device)

    def __call__(self, feats0, feats1):
        """Main func."""
        matches = self.matcher({"image0": feats0, "image1": feats1})
        return rbd(matches)
