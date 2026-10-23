"""Train or fine-tune the FaceRestore model.

Fine-tune from the upstream weights with attention at 32x32 (what we ship):

    python train.py --data data/celeba256 --init weights/upstream.pt --attention-after-up 3 --out runs/ft1

Resume an interrupted run:

    python train.py --data data/celeba256 --resume runs/ft1/last.pt --out runs/ft1

Writes runs/<name>/best.pt (lowest validation LPIPS, ready to copy to
weights/facerestore.pt), last.pt (full training state) and log.csv.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from facerestore import checkpoint, metrics
from facerestore.data import FaceDataset
from facerestore.losses import InpaintLoss, PatchDiscriminator
from facerestore.model import InpaintNet, ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--init', help='checkpoint to take weights from (new layers start fresh)')
    p.add_argument('--resume', help='last.pt of a previous run')
    p.add_argument('--attention-after-up', type=int, default=3)
    p.add_argument('--renorm', action='store_true', help='only when training from scratch')
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--adv-weight', type=float, default=0.0, help='e.g. 0.01 to add the patch GAN loss')
    p.add_argument('--val-every', type=int, default=2000)
    p.add_argument('--val-limit', type=int, default=1000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--max-minutes', type=float, help='stop and save after this long (e.g. Kaggle session limits)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def build_model(args, device):
    if args.init:
        source = checkpoint.load_raw(args.init)
        base = ModelConfig.from_dict(source['config'])
        config = ModelConfig(bn=base.bn, renorm=base.renorm, attention_after_up=args.attention_after_up,
                             upsample_mode=base.upsample_mode)
        model = InpaintNet(config)
        result = model.load_state_dict(source['model'], strict=False)
        unexpected_new = [k for k in result.missing_keys if not k.startswith('attention.')]
        if unexpected_new or result.unexpected_keys:
            raise SystemExit(f'--init does not fit: missing {unexpected_new}, unexpected {result.unexpected_keys}')
    else:
        model = InpaintNet(ModelConfig(bn=True, renorm=args.renorm, attention_after_up=args.attention_after_up))
    return model.to(device)


@torch.no_grad()
def validate(model, loader, lpips, device, amp_dtype):
    model.eval()
    totals = {'psnr_hole': 0.0, 'ssim': 0.0, 'lpips': 0.0}
    count = 0
    for image, hole in loader:
        image, hole = image.to(device), hole.to(device)
        known = (1 - hole).expand_as(image)
        with torch.autocast(device.type, amp_dtype, enabled=device.type == 'cuda'):
            out = model(image * known, known)
        comp = known * image + (1 - known) * out.float().clamp(0, 1)
        # PSNR over the hole only, so small and large holes are comparable.
        err = ((comp - image) ** 2 * (1 - known)).flatten(1).sum(1) / (1 - known).flatten(1).sum(1).clamp(min=1)
        totals['psnr_hole'] += (10 * torch.log10(1 / err.clamp(min=1e-10))).sum().item()
        totals['ssim'] += metrics.ssim(comp, image).sum().item()
        totals['lpips'] += lpips(comp, image).sum().item()
        count += image.shape[0]
    model.train()
    return {k: v / count for k, v in totals.items()}


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    # bfloat16 needs Ampere or newer; T4/P100 (Kaggle, Colab) use float16 with loss scaling.
    use_bf16 = device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda' and not use_bf16)

    state = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if state:
        model = InpaintNet(ModelConfig.from_dict(state['config'])).to(device)
        model.load_state_dict(state['model'])
    else:
        model = build_model(args, device)
    print(f'model config: {model.config}', flush=True)

    criterion = InpaintLoss().to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    disc = disc_opt = None
    if args.adv_weight > 0:
        disc = PatchDiscriminator().to(device)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=args.lr * 4, betas=(0.5, 0.999))
    step, best = 0, math.inf
    if state:
        optimizer.load_state_dict(state['optimizer'])
        if disc is not None and 'disc' in state:
            disc.load_state_dict(state['disc'])
            disc_opt.load_state_dict(state['disc_optimizer'])
        step, best = state['step'], state['best_lpips']

    def lr_at(s):  # short warmup, then cosine decay to 10%
        return args.lr * min(1.0, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))

    train_loader = torch.utils.data.DataLoader(
        FaceDataset(args.data, 'train'), batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        FaceDataset(args.data, 'val', train=False, limit=args.val_limit), batch_size=16, num_workers=args.workers)
    lpips = metrics.LPIPS(device)

    log_path = out_dir / 'log.csv'
    new_log = not log_path.exists()
    log = open(log_path, 'a', newline='')
    writer = csv.writer(log)
    if new_log:
        writer.writerow(['step', 'loss', 'psnr_hole', 'ssim', 'lpips', 'seconds'])
        if step == 0:
            v = validate(model, val_loader, lpips, device, amp_dtype)
            writer.writerow([0, '', f"{v['psnr_hole']:.4f}", f"{v['ssim']:.4f}", f"{v['lpips']:.4f}", 0])
            log.flush()
            print(f"step 0 (start) val: {v}", flush=True)

    model.train()
    started, running = time.time(), 0.0
    while step < args.steps:
        for image, hole in train_loader:
            if step >= args.steps:
                break
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step)
            image, hole = image.to(device, non_blocking=True), hole.to(device, non_blocking=True)
            known = (1 - hole).expand_as(image)
            with torch.autocast(device.type, amp_dtype, enabled=device.type == 'cuda'):
                out = model(image * known, known).float()
                loss, _ = criterion(out, hole, image)

            if disc is not None:
                comp = known * image + (1 - known) * out
                with torch.autocast(device.type, amp_dtype, enabled=device.type == 'cuda'):
                    d_loss = F.relu(1 - disc(image, hole)).mean() + F.relu(1 + disc(comp.detach(), hole)).mean()
                disc_opt.zero_grad(set_to_none=True)
                scaler.scale(d_loss).backward()
                scaler.step(disc_opt)
                with torch.autocast(device.type, amp_dtype, enabled=device.type == 'cuda'):
                    loss = loss - args.adv_weight * disc(comp, hole).float().mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            out_of_time = args.max_minutes and time.time() - started > args.max_minutes * 60
            running += loss.item()

            if step % 100 == 0:
                elapsed = time.time() - started
                print(f'step {step} loss {running / 100:.4f} lr {lr_at(step):.2e} {elapsed / 60:.1f} min', flush=True)
                running = 0.0
            if step % args.val_every == 0 or step == args.steps or out_of_time:
                v = validate(model, val_loader, lpips, device, amp_dtype)
                writer.writerow([step, f'{loss.item():.4f}', f"{v['psnr_hole']:.4f}", f"{v['ssim']:.4f}",
                                 f"{v['lpips']:.4f}", int(time.time() - started)])
                log.flush()
                print(f'step {step} val: {v}', flush=True)
                if v['lpips'] < best:
                    best = v['lpips']
                    checkpoint.save(out_dir / 'best.pt', model, {'meta': {'step': step, 'val': v}})
                full = {'format': checkpoint.FORMAT, 'config': model.config.to_dict(), 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'step': step, 'best_lpips': best}
                if disc is not None:
                    full.update(disc=disc.state_dict(), disc_optimizer=disc_opt.state_dict())
                torch.save(full, out_dir / 'last.pt')
            if out_of_time:
                print(f'stopping at step {step}: --max-minutes reached; resume with --resume {out_dir / "last.pt"}')
                log.close()
                return
    log.close()


if __name__ == '__main__':
    main()
