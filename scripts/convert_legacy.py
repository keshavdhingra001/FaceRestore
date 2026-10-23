"""Convert the karfly/inpaint release weights into a facerestore checkpoint.

    python scripts/convert_legacy.py model_no_sigmoid_lr_all_50.pth weights/upstream.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facerestore import checkpoint  # noqa: E402
from facerestore.model import InpaintNet, ModelConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()

    state = torch.load(args.src, map_location='cpu', weights_only=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model = InpaintNet(ModelConfig(bn=any('.bn.' in k for k in state), renorm=False))
    model.load_state_dict(state)  # strict: any mismatch is an error
    checkpoint.save(args.dst, model, {'meta': {'source': 'karfly/inpaint model_no_sigmoid_lr_all_50.pth'}})
    print(f'wrote {args.dst} with {model.config}')


if __name__ == '__main__':
    main()
