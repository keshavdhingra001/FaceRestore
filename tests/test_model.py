import numpy as np
import pytest
import torch

from facerestore import checkpoint, engines, masks, pipeline
from facerestore.align import TEMPLATE_256, alignment_matrix, draw_regions
from facerestore.model import InpaintNet, ModelConfig


@pytest.mark.parametrize('config', [
    ModelConfig(),
    ModelConfig(bn=True, renorm=True, attention_after_up=3),
])
def test_forward_shape(config):
    model = InpaintNet(config).eval()
    x = torch.rand(2, 3, 256, 256)
    known = torch.ones_like(x)
    known[:, :, 100:150, 80:180] = 0
    with torch.no_grad():
        assert model(x * known, known).shape == x.shape


def test_renorm_scales_partial_windows():
    from facerestore.model import PartialConv2d
    conv = PartialConv2d(1, 1, 3, padding=1, renorm=True)
    conv.conv.weight.data.fill_(1)
    x = torch.ones(1, 1, 5, 5)
    mask = torch.ones_like(x)
    mask[..., :, :3] = 0
    out, new_mask = conv(x, mask)
    # A window that sees any known pixel is rescaled to the full-window sum.
    assert out[0, 0, 2, 2].item() == pytest.approx(9.0)
    assert out[0, 0, 2, 0].item() == 0.0 and new_mask[0, 0, 2, 0].item() == 0.0


def test_checkpoint_round_trip_keeps_config(tmp_path):
    model = InpaintNet(ModelConfig(renorm=True, attention_after_up=2))
    checkpoint.save(tmp_path / 'm.pt', model)
    loaded = checkpoint.load_model(tmp_path / 'm.pt')
    assert loaded.config == model.config
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_plain_state_dict_is_rejected(tmp_path):
    torch.save(InpaintNet().state_dict(), tmp_path / 'raw.pth')
    with pytest.raises(ValueError, match='convert'):
        checkpoint.load_model(tmp_path / 'raw.pth')


def test_engine_keeps_known_pixels(tmp_path):
    checkpoint.save(tmp_path / 'm.pt', InpaintNet())
    engine = engines.PConvEngine(tmp_path / 'm.pt')
    image = np.random.rand(256, 256, 3).astype(np.float32)
    hole = np.zeros((256, 256), np.uint8)
    hole[90:140, 60:200] = 1
    out = engine(image, hole)
    assert np.array_equal(out[hole == 0], image[hole == 0])
    assert out.min() >= 0 and out.max() <= 1


def test_pipeline_fills_only_the_hole(tmp_path):
    checkpoint.save(tmp_path / 'm.pt', InpaintNet())
    engine = engines.PConvEngine(tmp_path / 'm.pt')
    image = np.random.rand(400, 300, 3).astype(np.float32)
    hole = np.zeros((400, 300), np.uint8)
    hole[150:200, 100:200] = 1
    hole[10:20, 10:290] = 1  # outside any face crop
    landmarks = TEMPLATE_256 * 1.1 + np.float32([10, 60])
    out = pipeline.restore(engine, image, hole, landmarks)
    assert out.shape == image.shape
    assert np.array_equal(out[hole == 0], image[hole == 0])
    out_no_face = pipeline.restore(engine, image, hole, None)
    assert np.array_equal(out_no_face[hole == 0], image[hole == 0])


def test_alignment_maps_template_to_itself():
    matrix = alignment_matrix(TEMPLATE_256 * 2, 256)
    mapped = (matrix[:, :2] @ (TEMPLATE_256 * 2).T).T + matrix[:, 2]
    assert np.allclose(mapped, TEMPLATE_256, atol=0.5)


def test_masks():
    rng = np.random.default_rng(0)
    for _ in range(50):
        m = masks.training_mask(256, rng)
        assert m.dtype == np.uint8 and m.shape == (256, 256) and set(np.unique(m)) <= {0, 1}
    for kind in masks.EVAL_KINDS:
        a = masks.eval_mask(kind, 256, seed=3)
        assert np.array_equal(a, masks.eval_mask(kind, 256, seed=3)) and 0 < a.mean() < 0.5
    eyes = draw_regions(np.zeros((256, 256), np.uint8), TEMPLATE_256, ['right_eye'])
    x, y = TEMPLATE_256[0].round().astype(int)
    assert eyes[y, x] == 1
