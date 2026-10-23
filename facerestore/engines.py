"""Inpainting engines with one interface, so the app and the evaluation treat them the same.

engine(image, hole) takes an aligned square face, float32 H x W x 3 in [0, 1],
and a uint8 H x W hole mask (1 = fill), and returns the filled image with
known pixels untouched.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from . import checkpoint


class PConvEngine:
    work_size = 256

    def __init__(self, path, device='cpu'):
        self.device = torch.device(device)
        self.model = checkpoint.load_model(path, self.device)

    @torch.no_grad()
    def __call__(self, image, hole):
        x = torch.from_numpy(image.transpose(2, 0, 1)).to(self.device)[None]
        known = 1 - torch.from_numpy(hole).to(self.device).float()[None, None].expand_as(x)
        out = self.model(x * known, known).clamp(0, 1)
        out = known * x + (1 - known) * out
        return out[0].permute(1, 2, 0).cpu().numpy()


class LamaEngine:
    """LaMa (Suvorov et al., 2022), big-lama TorchScript export. Fully convolutional, any size."""
    work_size = 512

    def __init__(self, path, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    @torch.no_grad()
    def __call__(self, image, hole):
        h, w = hole.shape
        x = torch.from_numpy(image.transpose(2, 0, 1)).to(self.device)[None]
        m = torch.from_numpy(hole).to(self.device).float()[None, None]
        pad_h, pad_w = (-h) % 8, (-w) % 8
        out = self.model(F.pad(x, (0, pad_w, 0, pad_h), mode='reflect'),
                         F.pad(m, (0, pad_w, 0, pad_h), mode='reflect'))[:, :, :h, :w].clamp(0, 1)
        out = (1 - m) * x + m * out
        return out[0].permute(1, 2, 0).cpu().numpy()


# File name in the weights directory -> (engine id, label, class).
KNOWN = {
    'facerestore.pt': ('facerestore', 'FaceRestore (ours)', PConvEngine),
    'big-lama.pt': ('lama', 'LaMa', LamaEngine),
    'upstream.pt': ('upstream', 'Upstream PConv (2019)', PConvEngine),
}


def load_available(weights_dir, device='cpu'):
    """Load every known engine whose weights file exists. Returns {id: (label, engine)}."""
    engines = {}
    for filename, (engine_id, label, cls) in KNOWN.items():
        path = Path(weights_dir) / filename
        if path.exists():
            engines[engine_id] = (label, cls(path, device))
    return engines
