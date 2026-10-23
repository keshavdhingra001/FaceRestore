"""Score engines on held-out CelebA test faces, with identical holes for every engine.

    python evaluate.py --data data/celeba256 --weights weights --limit 2000
    python evaluate.py --data data/celeba256 --engine ours=runs/ft1/best.pt --engine upstream=weights/upstream.pt

Metrics per mask kind: PSNR inside the hole (higher is better), SSIM of the
whole image (higher is better), LPIPS (lower is better; the one that matches
what people see best).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from facerestore import engines, masks, metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--weights', help='load every known engine found in this directory')
    p.add_argument('--engine', action='append', default=[], help='name=path to a .pt (PConv checkpoint or big-lama.pt)')
    p.add_argument('--limit', type=int, default=2000)
    p.add_argument('--json', help='write results here')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    loaded = {}
    if args.weights:
        loaded.update({k: e for k, (_, e) in engines.load_available(args.weights, args.device).items()})
    for spec in args.engine:
        name, path = spec.split('=', 1)
        cls = engines.LamaEngine if Path(path).name == 'big-lama.pt' else engines.PConvEngine
        loaded[name] = cls(path, args.device)
    if not loaded:
        raise SystemExit('no engines given')

    root = Path(args.data)
    data = np.load(root / f'{args.split}_landmarks.npz')
    names, landmarks = data['names'][:args.limit], data['landmarks'][:args.limit]
    lpips = metrics.LPIPS(args.device)
    results = {e: {k: {'psnr_hole': [], 'ssim': [], 'lpips': []} for k in masks.EVAL_KINDS} for e in loaded}

    for i, (name, lm) in enumerate(zip(names, landmarks)):
        image = np.asarray(Image.open(root / args.split / str(name)).convert('RGB'), dtype=np.float32) / 255
        target = torch.from_numpy(image.transpose(2, 0, 1))[None].to(args.device)
        for kind in masks.EVAL_KINDS:
            hole = masks.eval_mask(kind, 256, seed=i, landmarks=lm)
            hole_t = torch.from_numpy(hole).to(args.device).bool()
            for engine_name, engine in loaded.items():
                out = engine(image, hole)
                out_t = torch.from_numpy(out.transpose(2, 0, 1).copy())[None].to(args.device)
                err = ((out_t - target) ** 2)[0][:, hole_t].mean().clamp(min=1e-10)
                r = results[engine_name][kind]
                r['psnr_hole'].append((10 * torch.log10(1 / err)).item())
                r['ssim'].append(metrics.ssim(out_t, target).item())
                r['lpips'].append(lpips(out_t, target).item())
        if (i + 1) % 200 == 0:
            print(f'{i + 1}/{len(names)}', flush=True)

    summary = {e: {k: {m: float(np.mean(v)) for m, v in kinds[k].items()} for k in kinds} for e, kinds in results.items()}
    for e in summary:
        summary[e]['mean'] = {m: float(np.mean([summary[e][k][m] for k in masks.EVAL_KINDS]))
                              for m in ('psnr_hole', 'ssim', 'lpips')}

    print(f'\n{len(names)} {args.split} images\n')
    for metric, better in (('lpips', 'lower'), ('psnr_hole', 'higher'), ('ssim', 'higher')):
        print(f'{metric} ({better} is better)')
        cols = list(masks.EVAL_KINDS) + ['mean']
        print(f"{'engine':14s}" + ''.join(f'{c:>11s}' for c in cols))
        for e in summary:
            print(f'{e:14s}' + ''.join(f'{summary[e][c][metric]:11.4f}' for c in cols))
        print()
    if args.json:
        Path(args.json).write_text(json.dumps({'images': len(names), 'split': args.split, 'results': summary}, indent=2))


if __name__ == '__main__':
    main()
