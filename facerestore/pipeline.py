"""Run an engine on a real photo: align the face like the training crops, fill, warp back."""
import cv2
import numpy as np

from .align import alignment_matrix


def _fill_square(engine, image, hole):
    """Fill the whole image: pad to a square, resize to the engine's size, and back."""
    h, w = hole.shape
    side = max(h, w)
    pad = ((0, side - h), (0, side - w))
    square = np.pad(image, pad + ((0, 0),), mode='edge')
    square_hole = np.pad(hole, pad)
    size = engine.work_size
    small = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    small_hole = (cv2.resize(square_hole, (size, size), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    out = cv2.resize(engine(small, small_hole), (side, side), interpolation=cv2.INTER_CUBIC)
    return out[:h, :w]


def restore(engine, image, hole, landmarks=None):
    """image: float32 H x W x 3 in [0, 1]; hole: uint8 H x W, 1 = fill; landmarks: 5 x 2 or None."""
    hole = hole.astype(np.uint8)
    result = image.copy()
    remaining = hole.astype(bool)
    if landmarks is not None:
        size = engine.work_size
        matrix = alignment_matrix(landmarks, engine.template)
        aligned = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        aligned_hole = cv2.warpAffine(hole, matrix, (size, size), flags=cv2.INTER_LINEAR)
        filled = engine(aligned, (aligned_hole > 0).astype(np.uint8))
        h, w = hole.shape
        inverse = cv2.invertAffineTransform(matrix)
        back = cv2.warpAffine(filled, inverse, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        inside = cv2.warpAffine(np.ones((size, size), np.uint8), inverse, (w, h), flags=cv2.INTER_NEAREST)
        # Stay one pixel away from the crop border, where the warp mixes in padding.
        inside = cv2.erode(inside, np.ones((3, 3), np.uint8)).astype(bool)
        done = remaining & inside
        result[done] = back[done]
        remaining &= ~inside
    if remaining.any():
        # Holes outside the face crop (hair, background) or no face found.
        out = _fill_square(engine, result, remaining.astype(np.uint8))
        result[remaining] = out[remaining]
    return np.clip(result, 0, 1)
