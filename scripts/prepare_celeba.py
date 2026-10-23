"""Build the training data: 256x256 CelebA face crops, split into train/val/test, with landmarks.

Input is the CelebA aligned set as parquet files (huggingface.co/datasets/nielsr/CelebA-faces,
202,599 images of 178x218 in file order). The split follows the official CelebA
partition by index: 1-162770 train, 162771-182637 val, 182638-202599 test.

    python scripts/prepare_celeba.py --parquet-dir data/celeba_parquet --out data/celeba256 \
        --detector weights/face_detection_yunet_2023mar.onnx
"""
import argparse
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facerestore.align import FaceDetector  # noqa: E402

SPLITS = (('train', 162770), ('val', 182637), ('test', 202599))
_detector = None


def split_of(index):
    for name, last in SPLITS:
        if index <= last:
            return name
    raise ValueError(index)


def _init(detector_path):
    global _detector
    _detector = FaceDetector(detector_path, score_threshold=0.5)


def _process(job):
    index, data, out_dir = job
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img = img.crop((0, 20, 178, 198)).resize((256, 256), Image.Resampling.LANCZOS)
    name = f'{index:06d}.jpg'
    img.save(Path(out_dir) / split_of(index) / name, quality=95)
    faces = _detector.detect(np.asarray(img))
    landmarks = faces[0]['landmarks'] if faces else np.full((5, 2), np.nan, np.float32)
    return split_of(index), name, landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--detector', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    for name, _ in SPLITS:
        (out / name).mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.parquet_dir).glob('*.parquet'))
    landmarks = {name: {} for name, _ in SPLITS}
    index = 0
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.detector,)) as pool:
        for path in files:
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                images = parquet.read_row_group(group, columns=['image']).column('image').to_pylist()
                jobs = []
                for item in images:
                    index += 1
                    jobs.append((index, item['bytes'], str(out)))
                for split, name, lm in pool.map(_process, jobs, chunksize=64):
                    landmarks[split][name] = lm
                print(f'{index} images', flush=True)

    if index != SPLITS[-1][1]:
        raise SystemExit(f'expected {SPLITS[-1][1]} images, got {index}')
    for split, items in landmarks.items():
        names = sorted(items)
        np.savez(out / f'{split}_landmarks.npz', names=np.array(names), landmarks=np.stack([items[n] for n in names]))
    train = np.stack(list(landmarks['train'].values()))
    found = ~np.isnan(train).any(axis=(1, 2))
    print(f'faces found in {found.mean():.2%} of train images')
    print('median train landmarks (TEMPLATE_256):')
    print(np.round(np.median(train[found], axis=0), 1).tolist())


if __name__ == '__main__':
    main()
