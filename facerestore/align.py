"""Face detection, alignment to the training crop, and face-part regions.

Landmarks are 5 points (x, y): right eye, left eye, nose tip, right mouth
corner, left mouth corner, as returned by OpenCV's YuNet. "Right" is the
subject's right, which is on the left of the picture.
"""
import cv2
import numpy as np

# Median YuNet landmarks on CelebA training crops (178x218 aligned images,
# cropped to rows 20..198 and resized to 256x256), measured on the first 300
# training images. scripts/prepare_celeba.py prints the median over the full set.
TEMPLATE_256 = np.array([
    [98.6, 131.5],
    [155.0, 132.4],
    [124.6, 164.9],
    [101.1, 189.6],
    [150.8, 190.9],
], dtype=np.float32)

REGION_NAMES = ('right_eye', 'left_eye', 'nose', 'mouth', 'forehead', 'right_cheek', 'left_cheek', 'chin')


class FaceDetector:
    def __init__(self, model_path, score_threshold=0.6):
        self._detector = cv2.FaceDetectorYN.create(str(model_path), '', (320, 320), score_threshold)

    def detect(self, image_rgb):
        """Return faces sorted by area, largest first: [{'box': [x, y, w, h], 'score': s, 'landmarks': 5x2}]."""
        h, w = image_rgb.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        result = [{
            'box': f[:4].tolist(),
            'score': float(f[14]),
            'landmarks': f[4:14].reshape(5, 2).astype(np.float32),
        } for f in faces]
        return sorted(result, key=lambda f: f['box'][2] * f['box'][3], reverse=True)


def alignment_matrix(landmarks, size):
    """Similarity transform that maps the face to the training crop at `size` x `size`."""
    template = TEMPLATE_256 * (size / 256)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    return matrix


def regions(landmarks):
    """Ellipses for face parts in the landmarks' coordinate space.

    Sizes are fractions of the eye distance, so they follow the face's scale
    and roll. Returns {name: (cx, cy, rx, ry, angle_degrees)}.
    """
    re, le, nose, rm, lm = [np.asarray(p, dtype=np.float64) for p in landmarks]
    across = le - re
    d = float(np.linalg.norm(across))
    ux = across / d                      # along the eye line
    uy = np.array([-ux[1], ux[0]])       # down the face
    angle = float(np.degrees(np.arctan2(ux[1], ux[0])))
    eyes_mid = (re + le) / 2
    mouth_mid = (rm + lm) / 2
    mouth_w = float(np.linalg.norm(lm - rm))

    def at(point, rx, ry):
        return (float(point[0]), float(point[1]), rx * d, ry * d, angle)

    return {
        'right_eye': at(re - 0.02 * d * uy, 0.30, 0.19),
        'left_eye': at(le - 0.02 * d * uy, 0.30, 0.19),
        'nose': at((eyes_mid + 2 * nose) / 3 + 0.05 * d * uy, 0.26, 0.45),
        'mouth': at(mouth_mid, max(0.62 * mouth_w / d, 0.45), 0.26),
        'forehead': at(eyes_mid - 0.62 * d * uy, 0.95, 0.30),
        'right_cheek': at(re + 0.12 * d * -ux + 0.62 * d * uy, 0.26, 0.32),
        'left_cheek': at(le + 0.12 * d * ux + 0.62 * d * uy, 0.26, 0.32),
        'chin': at(mouth_mid + 0.52 * d * uy, 0.45, 0.22),
    }


def draw_regions(mask, landmarks, names, scale=(1.0, 1.0)):
    """Paint the named regions as 1 on a uint8 mask (in place)."""
    all_regions = regions(landmarks)
    for name in names:
        cx, cy, rx, ry, angle = all_regions[name]
        cv2.ellipse(mask, (round(cx), round(cy)), (max(1, round(rx * scale[0])), max(1, round(ry * scale[1]))),
                    angle, 0, 360, 1, -1)
    return mask
