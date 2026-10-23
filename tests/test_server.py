import io
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facerestore import checkpoint
from facerestore.model import InpaintNet

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / 'weights' / 'face_detection_yunet_2023mar.onnx'


@pytest.fixture
def client(tmp_path):
    if not DETECTOR.exists():
        pytest.skip('run scripts/fetch_weights.py --only detector')
    from app.server import create_app
    shutil.copy(DETECTOR, tmp_path)
    checkpoint.save(tmp_path / 'upstream.pt', InpaintNet())
    return create_app(tmp_path).test_client()


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


def test_health_and_engines(client):
    assert client.get('/healthz').json['engines'] == ['upstream']
    assert client.get('/api/engines').json['default'] == 'upstream'


def test_inpaint_keeps_unmasked_pixels(client):
    sample = Image.open(next((ROOT / 'app' / 'static' / 'samples').glob('*.png'))).convert('RGB')
    mask = Image.new('RGBA', sample.size, (0, 0, 0, 0))
    mask.paste((255, 255, 255, 255), (80, 100, 180, 140))
    r = client.post('/api/inpaint', data={'image': (png(sample), 'i.png'), 'mask': (png(mask), 'm.png')})
    assert r.status_code == 200 and r.mimetype == 'image/png'
    assert r.headers['X-Face-Found'] == 'true'
    out = np.asarray(Image.open(io.BytesIO(r.data)).convert('RGB'))
    hole = np.asarray(mask)[..., 3] > 0
    assert np.array_equal(out[~hole], np.asarray(sample)[~hole])


def test_detect_returns_regions(client):
    sample = Image.open(next((ROOT / 'app' / 'static' / 'samples').glob('*.png'))).convert('RGB')
    faces = client.post('/api/detect', data={'image': (png(sample), 'i.png')}).json['faces']
    assert faces and set(faces[0]['regions']) >= {'right_eye', 'left_eye', 'nose', 'mouth'}


def test_bad_requests(client):
    assert client.post('/api/inpaint', data={}).status_code == 400
    bad = client.post('/api/inpaint', data={'image': (io.BytesIO(b'no'), 'a.png'), 'mask': (io.BytesIO(b'no'), 'b.png')})
    assert bad.status_code == 400 and 'error' in bad.json
    big = png(Image.new('RGB', (2000, 100)))
    r = client.post('/api/inpaint', data={'image': (big, 'i.png'), 'mask': (png(Image.new('RGBA', (2000, 100))), 'm.png')})
    assert r.status_code == 413
    sample = png(Image.new('RGB', (64, 64)))
    r = client.post('/api/inpaint', data={'image': (sample, 'i.png'), 'mask': (png(Image.new('RGBA', (64, 64))), 'm.png'),
                                          'engine': 'nope'})
    assert r.status_code == 400
