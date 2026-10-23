from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import masks

# Horizontal flip swaps the subject's left and right points.
_FLIP_ORDER = [1, 0, 2, 4, 3]


class FaceDataset(torch.utils.data.Dataset):
    """Images from scripts/prepare_celeba.py. Yields (image, hole) with hole = 1 where masked.

    With train=False, masks are fixed per index so validation numbers are comparable.
    """

    def __init__(self, root, split, train=True, limit=None, size=256):
        root = Path(root)
        data = np.load(root / f'{split}_landmarks.npz')
        self.paths = [root / split / str(n) for n in data['names']]
        self.landmarks = data['landmarks']
        if limit:
            self.paths, self.landmarks = self.paths[:limit], self.landmarks[:limit]
        self.train = train
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.paths[index]).convert('RGB'), dtype=np.float32) / 255
        landmarks = self.landmarks[index].copy()
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                image = image[:, ::-1]
                landmarks[:, 0] = self.size - 1 - landmarks[:, 0]
                landmarks = landmarks[_FLIP_ORDER]
            hole = masks.training_mask(self.size, rng, landmarks)
        else:
            kind = masks.EVAL_KINDS[index % len(masks.EVAL_KINDS)]
            hole = masks.eval_mask(kind, self.size, seed=index, landmarks=landmarks)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return image, torch.from_numpy(hole)[None].float()
