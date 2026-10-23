"""Training and evaluation masks. All return uint8 arrays where 1 marks a hole.

The mix matters: the app's users mask whole eyes, noses and mouths, so the
model must see face-part holes in training, not only thin scribbles.
"""
import cv2
import numpy as np

from .align import REGION_NAMES, TEMPLATE_256, draw_regions


def free_form(size, rng, max_strokes=6):
    """Brush strokes in the style of DeepFill v2 (Yu et al., 2019)."""
    mask = np.zeros((size, size), np.uint8)
    for _ in range(rng.integers(1, max_strokes + 1)):
        x, y = rng.integers(0, size, 2)
        width = int(rng.integers(size // 32, size // 9))
        for _ in range(rng.integers(2, 8)):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.integers(size // 16, size // 4)
            nx = int(np.clip(x + length * np.cos(angle), 0, size - 1))
            ny = int(np.clip(y + length * np.sin(angle), 0, size - 1))
            cv2.line(mask, (int(x), int(y)), (nx, ny), 1, width)
            cv2.circle(mask, (nx, ny), width // 2, 1, -1)
            x, y = nx, ny
    return mask


def box(size, rng):
    mask = np.zeros((size, size), np.uint8)
    h, w = rng.integers(size // 6, size // 2, 2)
    y, x = rng.integers(size // 8, size - size // 8 - h), rng.integers(size // 8, size - size // 8 - w)
    mask[y:y + h, x:x + w] = 1
    return mask


def face_parts(size, rng, landmarks=None, names=None):
    """One to three face-part ellipses, grown or shrunk a little at random."""
    if landmarks is None or np.isnan(landmarks).any():
        landmarks = TEMPLATE_256 * (size / 256)
    if names is None:
        names = rng.choice(REGION_NAMES, size=rng.integers(1, 4), replace=False)
    landmarks = landmarks + rng.normal(0, size / 128, landmarks.shape)
    scale = rng.uniform(0.85, 1.35, 2)
    return draw_regions(np.zeros((size, size), np.uint8), landmarks.astype(np.float32), names, scale)


def training_mask(size, rng, landmarks=None):
    kind = rng.random()
    if kind < 0.45:
        mask = free_form(size, rng)
    elif kind < 0.85:
        mask = face_parts(size, rng, landmarks)
    else:
        mask = box(size, rng)
    if rng.random() < 0.25:  # sometimes combine two kinds
        mask = np.maximum(mask, free_form(size, rng, max_strokes=2))
    return mask


EVAL_KINDS = ('free_form', 'eyes', 'nose', 'mouth', 'large')


def eval_mask(kind, size, seed, landmarks=None):
    """Deterministic masks so every engine is scored on the same holes."""
    rng = np.random.default_rng(seed)
    if kind == 'free_form':
        return free_form(size, rng)
    names = {
        'eyes': ['right_eye', 'left_eye'],
        'nose': ['nose'],
        'mouth': ['mouth'],
        'large': ['right_eye', 'left_eye', 'nose', 'mouth'],
    }[kind]
    if landmarks is None or np.isnan(landmarks).any():
        landmarks = TEMPLATE_256 * (size / 256)
    scale = (1.4, 1.6) if kind == 'large' else (1.0, 1.0)
    return draw_regions(np.zeros((size, size), np.uint8), landmarks, names, scale)
