#! /usr/bin/env python3
# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/d031f879c9a8d33c8b7dc52c5bc65fe8b9e3960d/quantization/image_classification/cpu/run.py

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../..')

import numpy as np
import time
import tempfile
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

import modman
import dataman as dm
import cfg

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-name', default='resnet50')
parser.add_argument('-d', '--dataset', default='CIFAR10')
parser.add_argument('--output-root', help='root output directory', default=f'{cfg.project_root}/models')
parser.add_argument("--quant-format",
                    default=QuantFormat.QOperator,
                    type=QuantFormat.from_string,
                    choices=list(QuantFormat))
parser.add_argument("--per-channel", default=False, type=bool)
parser.add_argument('--image-size', default=32)
parser.add_argument('--batch-size', default=cfg.batch_size)
args = parser.parse_args()


class DataReader(CalibrationDataReader):
    def __init__(self, dataset, image_size, batch_size):
        self.batch_size = batch_size
        self.loader = dm.get_benign_loader(dataset, image_size, 'train', batch_size)
        self.loader_iter = iter(self.loader)

    def get_next(self):
        try:
            data = next(self.loader_iter)[0].numpy()
            if data.shape[0] < self.batch_size:
                print(f'Discarding (last) batch of size {data.shape[0]}')
                return None
            return {'input0': data}
        except StopIteration:
            return None


def benchmark(model_path, batch_size):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((batch_size, 3, args.image_size, args.image_size), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for _ in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def quant(skip_if_exists=False):
    dataset = args.dataset
    model_name = args.model_name
    image_size = args.image_size
    batch_size = args.batch_size

    output_root = args.output_root
    output_dir = f'{output_root}/{dataset}/Q{model_name}'

    output_model_path = f'{output_dir}/Q{model_name}-{batch_size}.onnx'
    if skip_if_exists and os.path.exists(output_model_path):
        print(f'Skipping existing quant model {output_model_path}')
        return

    print(f'Quantized model with batch size {batch_size} will be saved to: {output_model_path}')
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp_onnx_file:
        torch_mod = modman.get_torch_mod(model_name, dataset)
        modman.export_torch_mod(torch_mod, (batch_size, 3, image_size, image_size), tmp_onnx_file.name, optimise=True)

        dr = DataReader(dataset, image_size, batch_size)
        quantize_static(tmp_onnx_file.name,
                        output_model_path,
                        dr,
                        quant_format=args.quant_format,
                        per_channel=args.per_channel,
                        weight_type=QuantType.QInt8)
        print('Calibrated and quantized model saved.')

        print('benchmarking fp model...')
        benchmark(tmp_onnx_file.name, batch_size)

        print('benchmarking int model...')
        benchmark(output_model_path, batch_size)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        quant()
        exit(0)

    print(f'Auto quantizing all models required by config')
    for bi in cfg.tvm.build_bis:
        args.model_name, args.dataset = bi.model_name, bi.dataset
        if not args.model_name.startswith('Q'):
            continue
        args.model_name = args.model_name[1:]
        quant(skip_if_exists=True)
