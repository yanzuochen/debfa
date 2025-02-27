#! /usr/bin/env python3

import sys
import os
import shutil
import tempfile

import modman
import utils
import cfg
from eval import evalutils

def maybe_build_tvm_mod(bi: utils.BinaryInfo, output_dir, check_acc):
    output_file = f'{output_dir}/{bi.fname}'
    target = modman.targets['avx2' if bi.avx else 'llvm']
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return
    print(f'Building {output_file}')
    mod, params = modman.get_irmod(
        bi.model_name, bi.dataset, 'none', cfg.batch_size, bi.input_img_size, nchannels=bi.nchans
    )
    # Build the .so lib with extra params embedded
    rtmod, lib = modman.build_module(
        mod, params, export_path=output_file,
        target=target, opt_level=bi.opt_level, is_qnn=bi.model_name.startswith('Q')
    )
    # Check accuracy
    if check_acc and not bi.dataset.startswith('fake') and not bi.model_name.startswith('dcgan'):
        assert evalutils.check_so_acc(output_file) > 0.6

def maybe_build_glow_mod(bi, output_dir, weights_out_dir, check_acc):
    output_file = f'{output_dir}/{bi.fname}'
    weights_file = f'{weights_out_dir}/{bi.fname}.weights.bin'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return

    print(f'Building {output_file} with {weights_file}')
    utils.ensure_dir_of(output_file)
    utils.ensure_dir_of(f'{weights_out_dir}/.')
    with tempfile.TemporaryDirectory() as tmpdir:
        modman.build_glow_model(
            bi.model_name, bi.dataset, cfg.batch_size, bi.input_img_size, tmpdir, nchannels=bi.nchans
        )
        shutil.move(f'{tmpdir}/model.so', output_file)
        shutil.move(f'{tmpdir}/model.weights.bin', weights_file)

    # Check accuracy
    if check_acc and not bi.dataset.startswith('fake') and not bi.model_name.startswith('dcgan'):
        assert evalutils.check_so_acc(output_file) > 0.55  # TODO: Accuracy of glow

def maybe_build_nnf_mod(bi, output_dir, data_out_dir, check_acc):
    output_file = f'{output_dir}/{bi.fname}'
    data_file = f'{data_out_dir}/{bi.fname}.data.tar'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return

    print(f'Building {output_file} with {data_file}')
    utils.ensure_dir_of(output_file)
    utils.ensure_dir_of(f'{data_out_dir}/.')
    with tempfile.TemporaryDirectory() as tmpdir:
        modman.build_nnf_model(
            bi.model_name, bi.dataset, cfg.batch_size, bi.img_size, tmpdir, nchannels=bi.nchans
        )
        shutil.move(f'{tmpdir}/libnnfusion_cpu_rt.so', output_file)
        shutil.move(f'{tmpdir}/data.tar', data_file)

    # TODO: Check accuracy
    # if check_acc and not bi.dataset.startswith('fake') and not bi.model_name.startswith('dcgan'):
        # assert evalutils.check_so_acc(output_file) > 0.6

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--compiler', type=str, default='tvm')
    parser.add_argument('-v', '--compiler_ver', type=str, default='main')
    parser.add_argument('-m', '--model', type=str, default='resnet50')
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10')
    parser.add_argument('-X', '--no-avx', action='store_false', dest='avx')
    parser.add_argument('-A', '--no-check-acc', action='store_false', dest='check_acc')
    parser.add_argument('-O', '--opt-level', type=int, default=3)
    args = parser.parse_args()

    bis_to_build = cfg.tvm.build_bis + cfg.glow.build_bis
    if len(sys.argv) > 1 and not (len(sys.argv) == 2 and not args.check_acc):
        bi = utils.BinaryInfo(
            args.compiler, args.compiler_ver, args.model, args.dataset, args.avx, args.opt_level
        )
        bis_to_build = [bi]

    for bi in bis_to_build:
        print(f'{bi.compiler=} {bi.compiler_ver=} {bi.model_name=} {bi.dataset=} {bi.avx=} {bi.opt_level=} {bi.input_img_size=}')
        try:
            if bi.compiler == 'tvm':
                maybe_build_tvm_mod(bi, cfg.built_dir, args.check_acc)
            elif bi.compiler == 'glow':
                maybe_build_glow_mod(bi, cfg.built_dir, cfg.built_aux_dir, args.check_acc)
            elif bi.compiler == 'nnfusion':
                maybe_build_nnf_mod(bi, cfg.built_dir, cfg.built_aux_dir, args.check_acc)
            else:
                raise ValueError(f'Unknown compiler {bi.compiler}')
        except FileNotFoundError as e:
            if cfg.models_dir in e.filename:
                utils.warn(f'Skipping building due to lacking file(s): {e.filename}')
            else:
                raise
        print('-----------------')
