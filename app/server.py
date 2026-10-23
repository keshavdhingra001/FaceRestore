"""FaceRestore web app. Stateless: the browser sends the image and mask with every request.

    python -m app.server --weights weights            # dev server on :8003
    gunicorn "app.server:create_app()" ...            # production (see Dockerfile)
"""
import argparse
import io
import os
import random
import threading
from pathlib import Path

import numpy as np
import torch
from flask import Flask, abort, jsonify, render_template, request, send_file
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException

from facerestore import engines, pipeline
from facerestore.align import FaceDetector, regions

ROOT = Path(__file__).resolve().parent
MAX_SIDE = 1024
# Default is the first one available, ranked by evaluate.py on held-out CelebA
# test faces (README). LaMa is trained on scenes and smears faces, so it is last.
PREFERENCE = ('codeformer', 'facerestore', 'upstream', 'lama')
DETECTOR_FILE = 'face_detection_yunet_2023mar.onnx'


def _read_image(field):
    file = request.files.get(field)
    if file is None:
        abort(400, f'missing file field "{field}"')
    try:
        image = Image.open(file.stream)
        image = ImageOps.exif_transpose(image)
    except Exception:
        abort(400, f'"{field}" is not a readable image')
    return image


def create_app(weights_dir=None, device=None, default_engine=None):
    weights_dir = Path(weights_dir or os.environ.get('FACERESTORE_WEIGHTS', ROOT.parent / 'weights'))
    device = device or os.environ.get('FACERESTORE_DEVICE', 'cpu')
    torch.set_num_threads(int(os.environ.get('FACERESTORE_THREADS', max(1, (os.cpu_count() or 2) // 2))))

    loaded = engines.load_available(weights_dir, device)
    if not loaded:
        raise RuntimeError(f'no model weights in {weights_dir}; run scripts/fetch_weights.py')
    default_engine = default_engine or os.environ.get('FACERESTORE_DEFAULT_ENGINE') \
        or next(e for e in PREFERENCE if e in loaded)
    if default_engine not in loaded:
        raise RuntimeError(f'default engine {default_engine!r} is not available: {sorted(loaded)}')
    detector = FaceDetector(weights_dir / DETECTOR_FILE)
    samples = sorted((ROOT / 'static' / 'samples').glob('*.png'))
    lock = threading.Lock()  # one inference at a time per worker keeps memory flat

    app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/healthz')
    def healthz():
        return jsonify(ok=True, engines=sorted(loaded), default=default_engine)

    @app.get('/api/engines')
    def list_engines():
        ordered = [e for e in PREFERENCE if e in loaded]
        return jsonify(default=default_engine, engines=[{'id': e, 'label': loaded[e][0]} for e in ordered])

    @app.get('/api/sample')
    def sample():
        return send_file(random.choice(samples), mimetype='image/png', max_age=0)

    @app.post('/api/detect')
    def detect():
        image = np.asarray(_read_image('image').convert('RGB'))
        with lock:
            faces = detector.detect(image)
        return jsonify(faces=[{
            'box': f['box'],
            'score': f['score'],
            'landmarks': f['landmarks'].tolist(),
            'regions': {name: dict(zip(('cx', 'cy', 'rx', 'ry', 'angle'), r))
                        for name, r in regions(f['landmarks']).items()},
        } for f in faces])

    @app.post('/api/inpaint')
    def inpaint():
        engine_id = request.form.get('engine') or default_engine
        if engine_id not in loaded:
            abort(400, f'unknown engine {engine_id!r}')
        image = _read_image('image').convert('RGB')
        if max(image.size) > MAX_SIDE:
            abort(413, f'image is larger than {MAX_SIDE}px on a side')
        mask = _read_image('mask').convert('RGBA').resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask)
        # The canvas paints the hole white on a transparent layer.
        hole = ((mask[..., 3] > 0) & (mask[..., :3].max(axis=2) > 0)).astype(np.uint8)
        pixels = np.asarray(image, dtype=np.float32) / 255
        landmarks = None
        if hole.any():
            with lock:
                faces = detector.detect(np.asarray(image))
                landmarks = faces[0]['landmarks'] if faces else None
                pixels = pipeline.restore(loaded[engine_id][1], pixels, hole, landmarks)
        buffer = io.BytesIO()
        Image.fromarray((pixels * 255).round().astype(np.uint8)).save(buffer, format='PNG')
        buffer.seek(0)
        response = send_file(buffer, mimetype='image/png')
        response.headers['X-Engine'] = engine_id
        response.headers['X-Face-Found'] = str(landmarks is not None).lower()
        return response

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights')
    parser.add_argument('--device')
    parser.add_argument('--engine', help='default engine id')
    parser.add_argument('--port', type=int, default=8003)
    args = parser.parse_args()
    create_app(args.weights, args.device, args.engine).run(host='127.0.0.1', port=args.port, threaded=True)


if __name__ == '__main__':
    main()
