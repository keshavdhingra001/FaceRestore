"""Checkpoints carry the model config, so weights are never loaded in the wrong mode."""
import torch

from .model import InpaintNet, ModelConfig

FORMAT = 'facerestore-pconv-v1'


def save(path, model, extra=None):
    torch.save({
        'format': FORMAT,
        'config': model.config.to_dict(),
        'model': model.state_dict(),
        **(extra or {}),
    }, path)


def load_raw(path, device='cpu'):
    # weights_only=True refuses pickled code, so a downloaded checkpoint cannot run anything.
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get('format') != FORMAT:
        raise ValueError(f'{path} is not a {FORMAT} checkpoint; convert it with scripts/convert_legacy.py')
    return data


def load_model(path, device='cpu'):
    data = load_raw(path, device)
    model = InpaintNet(ModelConfig.from_dict(data['config']))
    model.load_state_dict(data['model'])
    return model.to(device).eval()
