# FaceRestore

Paint over part of a face and FaceRestore fills it back in. It finds the face,
aligns it the way the model saw faces in training, fills the hole, and puts the
result back into the original photo.

## Run it

```bash
docker compose up --build
```

Open http://localhost:8003. The build downloads the model weights and checks
their SHA-256 (see `scripts/fetch_weights.py`).

Without Docker:

```bash
pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install -r requirements.txt
python scripts/fetch_weights.py
python -m app.server            # http://127.0.0.1:8003
```

### Using it

1. **Sample** or **Upload** a photo (or drop one on the page).
2. Paint over what should be filled, or use the face-part buttons (eyes, nose,
   mouth, ...). They are placed from the detected landmarks.
3. The result appears on the right. Drag the slider to compare before and after.
4. **Model** in the top bar switches between the loaded models.

Nothing is stored on the server. Each request carries the image and the mask.

## Models

| Id | What it is | Weights |
|---|---|---|
| `codeformer` | [CodeFormer](https://github.com/sczhou/CodeFormer) face inpainting (S-Lab, 2023), FFHQ-trained. **Default.** | `codeformer_inpainting.pth`, from its release |
| `facerestore` | Our model: partial-convolution U-Net with self-attention, fine-tuned on CelebA from `upstream` | `weights/facerestore.pt`, from this repo's releases |
| `upstream` | The 2019 partial-convolution model from [karfly/inpaint](https://github.com/karfly/inpaint) | converted from its release by `scripts/convert_legacy.py` |
| `lama` | [LaMa](https://github.com/advimman/lama) big-lama, trained on scenes (Places2) | optional, `fetch_weights.py --only lama` |

On 200 held-out CelebA test faces (every 100th official test image, app's alignment path):

| | LPIPS (lower better) | PSNR in hole | SSIM |
|---|---|---|---|
| `upstream` | 0.0243 | 19.31 dB | 0.9706 |
| `codeformer` | **0.0154** | **21.39 dB** | **0.9771** |

The app's default is the first available of `codeformer`, `facerestore`, `upstream`, `lama`,
or `FACERESTORE_DEFAULT_ENGINE`.

### Architecture

- **Partial convolutions** ([Liu et al., 2018](https://arxiv.org/abs/1804.07723)):
  each convolution only looks at known pixels, and the mask shrinks layer by layer.
- **U-Net**, 7 levels down and up with skip connections.
- **Self-attention** ([Zhang et al., 2018](https://arxiv.org/abs/1805.08318)) on
  the 32x32 decoder map, so one side of the face can inform the other. It starts
  at zero weight, so fine-tuning from `upstream` begins from the same output.
- **Loss**: L1 on known and missing pixels, VGG-16 perceptual and style loss,
  total variation at the hole border. `--adv-weight` adds a patch GAN
  ([Yu et al., 2019](https://arxiv.org/abs/1806.03589)).

Checkpoints store the model config next to the weights, so a model is always
run with the settings it was trained with (`facerestore/checkpoint.py`).

## Training

Training needs a GPU. The `notebooks/kaggle_train.ipynb` notebook runs the whole
pipeline on a free Kaggle GPU: download CelebA, prepare it, fine-tune, and score
the models. By hand:

```bash
pip install -r requirements-train.txt   # plus torch for your GPU

# 1. CelebA aligned faces (parquet, ~1.4 GB) -> 256x256 crops + landmarks
#    https://huggingface.co/datasets/nielsr/CelebA-faces
python scripts/prepare_celeba.py --parquet-dir data/celeba_parquet --out data/celeba256 \
    --detector weights/face_detection_yunet_2023mar.onnx

# 2. Fine-tune from the upstream weights
python train.py --data data/celeba256 --init weights/upstream.pt --out runs/ft1

# 3. Score against the other models on held-out test faces
python evaluate.py --data data/celeba256 --weights weights --engine ours=runs/ft1/best.pt
```

The split follows the official CelebA partition (train 1-162770, val
162771-182637, test 182638-202599), so test faces are never seen in training.
Training masks mix free-form strokes, face parts (from each image's landmarks)
and boxes, because users mask whole eyes and mouths, not only thin lines.
`train.py` validates every 2000 steps and keeps the checkpoint with the best
LPIPS as `best.pt`. Copy it to `weights/facerestore.pt` to serve it.

`evaluate.py` reports, per mask kind (free-form, eyes, nose, mouth, large):
PSNR inside the hole, SSIM, and LPIPS. LPIPS follows what people see most
closely, so it decides the default model.

## Layout

```
facerestore/   model, checkpoints, masks, alignment, losses, metrics, engines, pipeline
app/           Flask app (server.py) and the browser UI
scripts/       data preparation, weight download, legacy weight conversion
train.py       training
evaluate.py    scoring
tests/         pytest
```

## Tests

```bash
python scripts/fetch_weights.py --only detector
pytest -q
```

CI runs the tests and a Docker build that starts the app and fills a sample face.

## Credits and licences

- Started from [karfly/inpaint](https://github.com/karfly/inpaint) (MIT); its
  model layout and released weights are used as `upstream`.
- [CodeFormer](https://github.com/sczhou/CodeFormer), S-Lab License 1.0
  (non-commercial). Network code copied in `facerestore/codeformer/` with its licence.
- [LaMa](https://github.com/advimman/lama), Apache-2.0, weights via the
  [IOPaint](https://github.com/Sanster/IOPaint) TorchScript export.
- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  face detector, MIT.
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is for
  non-commercial research only. Models trained on it carry that limit.
- Sample images in `app/static/samples` are from CelebA-HQ.

Code: MIT, see [LICENSE](LICENSE).
