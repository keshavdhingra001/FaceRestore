"""Image quality metrics on N x 3 x H x W tensors in [0, 1]."""
import torch
import torch.nn.functional as F


def psnr(x, y):
    mse = ((x - y) ** 2).flatten(1).mean(1).clamp(min=1e-10)
    return 10 * torch.log10(1 / mse)


def ssim(x, y, window=11, sigma=1.5):
    """SSIM of Wang et al. 2004 with a Gaussian window, averaged over channels."""
    coords = torch.arange(window, dtype=x.dtype, device=x.device) - window // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, window, window)

    def blur(t):
        return F.conv2d(t, kernel, groups=x.shape[1])

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.flatten(1).mean(1)


class LPIPS:
    """Learned perceptual distance (Zhang et al., 2018), lower is better."""

    def __init__(self, device):
        import lpips
        self.model = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, x, y):
        return self.model(x * 2 - 1, y * 2 - 1).flatten()
