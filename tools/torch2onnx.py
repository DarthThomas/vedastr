import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from volksdep.converters import torch2onnx

import torch
from vedastr.runners import InferenceRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Convert to Onnx model.')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('out', help='output onnx file name')
    parser.add_argument('--dummy_input_shape', default='3,800,1344',
                        type=str, help='model input shape like 3,800,1344. '
                                       'Shape format is CxHxW')
    parser.add_argument('--dynamic_shape', default=False, action='store_true',
                        help='whether to use dynamic shape')
    parser.add_argument('--opset_version', default=9, type=int,
                        help='onnx opset version')
    parser.add_argument('--do_constant_folding', default=False,
                        action='store_true',
                        help='whether to apply constant-folding optimization')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='whether print convert info')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'

    deploy_cfg = cfg['deploy']
    deploy_cfg['gpu_id'] = ''
    if device is not None:
        deploy_cfg['gpu_id'] += str(device)
    common_cfg = cfg.get('common')
    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    shape = map(int, args.dummy_input_shape.split(','))
    dummy_input = torch.randn(1, *shape)

    aug = runner.transform(image=dummy_input, label='')
    image, label = aug['image'], aug['label']
    image = image.unsqueeze(0).cuda()

    dummy_input = (image, runner.converter.test_encode(['']))
    model = runner.model.cuda().eval()
    need_text = runner.need_text
    if not need_text:
        dummy_input = dummy_input[0]

    if args.dynamic_shape:
        runner.logger.info(
            f'Convert to Onnx with dynamic input shape and opset version '
            f'{args.opset_version}'
        )
    else:
        runner.logger.info(
            f'Convert to Onnx with constant input shape'
            f' {args.dummy_input_shape} and opset version'
            f' {args.opset_version}'
        )
    torch2onnx(
        model,
        dummy_input,
        args.out, 
        verbose=args.verbose,
        dynamic_shape=args.dynamic_shape,
        opset_version=args.opset_version,
        do_constant_folding=args.do_constant_folding,
    )
    runner.logger.info(
        f'Convert successfully, saved onnx file: {os.path.abspath(args.out)}'
    )


if __name__ == '__main__':
    main()
