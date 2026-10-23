"""Download model weights into weights/ and check their SHA-256.

    python scripts/fetch_weights.py            # everything available
    python scripts/fetch_weights.py --only lama detector

Files already present with the right hash are skipped. Set FACERESTORE_MODEL_URL
to download our trained model from somewhere other than the GitHub release.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = {
    'detector': {
        'file': 'face_detection_yunet_2023mar.onnx',
        'url': 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        'sha256': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
        'required': True,
    },
    'codeformer': {
        'file': 'codeformer_inpainting.pth',
        'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth',
        'sha256': 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797',
        'required': False,
    },
    'lama': {
        'file': 'big-lama.pt',
        'url': 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt',
        'sha256': '344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9',
        'required': False,
    },
    'facerestore': {
        'file': 'facerestore.pt',
        'url': os.environ.get('FACERESTORE_MODEL_URL',
                              'https://github.com/keshavdhingra001/FaceRestore/releases/latest/download/facerestore.pt'),
        'sha256': None,  # set when the release is published
        'required': False,
    },
    'upstream': {
        'file': 'model_no_sigmoid_lr_all_50.pth',
        'url': 'https://www.dropbox.com/s/gb0yamqa6cnoomy/model_no_sigmoid_lr_all_50.pth?dl=1',
        'sha256': '5ea95d0b8f59d315ddc93a2b86d2cac5c92ee32b8092db1340b65f2db62a3eb6',
        'required': False,
        'convert_to': 'upstream.pt',
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name, spec, dest):
    target = dest / spec.get('convert_to', spec['file'])
    if target.exists() and (spec['sha256'] is None or 'convert_to' in spec or sha256(target) == spec['sha256']):
        print(f'{name}: have {target.name}')
        return True
    download = dest / spec['file']
    print(f'{name}: downloading {spec["url"]}')
    try:
        urllib.request.urlretrieve(spec['url'], download)
    except Exception as error:
        print(f'{name}: download failed ({error})')
        download.unlink(missing_ok=True)
        return False
    if spec['sha256'] and sha256(download) != spec['sha256']:
        download.unlink()
        print(f'{name}: checksum mismatch, file removed')
        return False
    if 'convert_to' in spec:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'convert_legacy.py'), str(download), str(target)],
                       check=True)
        download.unlink()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default=str(ROOT / 'weights'))
    parser.add_argument('--only', nargs='+', choices=sorted(WEIGHTS))
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    failed_required = False
    for name in args.only or WEIGHTS:
        ok = fetch(name, WEIGHTS[name], dest)
        failed_required |= WEIGHTS[name]['required'] and not ok
    engines = [f for f in ('codeformer_inpainting.pth', 'facerestore.pt', 'big-lama.pt', 'upstream.pt')
               if (dest / f).exists()]
    if failed_required or not engines:
        raise SystemExit('missing required weights')
    print(f'engines available: {", ".join(engines)}')


if __name__ == '__main__':
    main()
