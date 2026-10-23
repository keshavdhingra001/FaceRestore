"""Losses from Liu et al. 2018, plus an optional patch discriminator for sharper fills."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatures(nn.Module):
    """pool1, pool2 and pool3 of an ImageNet VGG-16."""

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList([vgg[:5], vgg[5:10], vgg[10:17]])
        self.register_buffer('mean', _MEAN)
        self.register_buffer('std', _STD)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = []
        for layer in self.slices:
            x = layer(x)
            features.append(x)
        return features


def _gram(feature):
    b, c, h, w = feature.shape
    f = feature.flatten(2).float()
    return f @ f.transpose(1, 2) / (c * h * w)


class InpaintLoss(nn.Module):
    def __init__(self, valid=1.0, hole=6.0, perceptual=0.05, style=120.0, tv=0.1):
        super().__init__()
        self.weights = dict(valid=valid, hole=hole, perceptual=perceptual, style=style, tv=tv)
        self.vgg = VGGFeatures()

    def forward(self, out, hole, target):
        """Return (total, parts). `hole` is 1 where pixels were missing."""
        known = 1 - hole
        comp = known * target + hole * out
        f_out, f_comp, f_target = self.vgg(out), self.vgg(comp), self.vgg(target)
        parts = {
            'valid': F.l1_loss(known * out, known * target),
            'hole': F.l1_loss(hole * out, hole * target),
            'perceptual': sum(F.l1_loss(a, t) + F.l1_loss(b, t) for a, b, t in zip(f_out, f_comp, f_target)),
            'style': sum(F.l1_loss(_gram(a), _gram(t)) + F.l1_loss(_gram(b), _gram(t))
                         for a, b, t in zip(f_out, f_comp, f_target)),
            'tv': _tv_on_hole_border(comp, hole),
        }
        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, parts


def _tv_on_hole_border(comp, hole):
    # Liu et al. apply total variation on the 1-pixel dilation of the hole.
    region = F.max_pool2d(hole, 3, 1, 1)
    dy = (comp[:, :, 1:] - comp[:, :, :-1]).abs() * region[:, :, 1:]
    dx = (comp[:, :, :, 1:] - comp[:, :, :, :-1]).abs() * region[:, :, :, 1:]
    return dy.mean() + dx.mean()


class PatchDiscriminator(nn.Module):
    """SN-PatchGAN (Yu et al., 2019): judges each patch, conditioned on the hole mask."""

    def __init__(self, channels=64):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        layers, c_in = [], 4
        for c_out in (channels, channels * 2, channels * 4, channels * 4, channels * 4):
            layers += [sn(nn.Conv2d(c_in, c_out, 5, 2, 2)), nn.LeakyReLU(0.2)]
            c_in = c_out
        layers.append(sn(nn.Conv2d(c_in, 1, 3, 1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, image, hole):
        return self.net(torch.cat([image, hole], 1))
