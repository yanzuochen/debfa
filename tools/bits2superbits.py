#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import analysis
import utils
import cfg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('base_bits')
    parser.add_argument('other_bits')
    parser.add_argument('-f', '--filter-modes', choices=['none', 'acc', 'label', 'lpips', 'fid'], default=('acc', 'label'), nargs=2)
    parser.add_argument('--parse-out-fname', default=None, help='When invoked by Makefile, this ignores the base_bits and other_bits args and instead guesses from this arg.')
    args = parser.parse_args()

    # TODO: Support GAN?

    if args.parse_out_fname is None:
        base_bits, other_bits = args.base_bits, args.other_bits
        # These may not refer to real files (possibly multiple datasets), but sufficient
        base_bi = utils.BinaryInfo.from_fname(os.path.basename(base_bits).replace('-bits.json', '.x'))
        other_bi = utils.BinaryInfo.from_fname(os.path.basename(other_bits).replace('-bits.json', '.x'))
        assert base_bi.just_datasets_differ(other_bi)
    else:
        base_bi, other_bi = analysis.superbits_fname2bis(args.parse_out_fname)
        base_bits = f'{cfg.bits_dir}/{base_bi.fname.replace(".so", "-bits.json")}'
        other_bits = f'{cfg.bits_dir}/{other_bi.fname.replace(".so", "-bits.json")}'

    bi = base_bi._replace(dataset=f'{base_bi.dataset}@{other_bi.dataset}')
    outfile = f'{cfg.superbits_dir}/{bi.fname.replace(".so", "-superbits.csv")}'
    utils.ensure_dir_of(outfile)

    filter_key, filter_range = analysis.get_filter_key_range(args.filter_modes[0])
    base_df = analysis.merge_dfs(
        *analysis.extract_dfs(base_bits),
        filter_key=filter_key, filter_range=filter_range, drop_metrics_cols=False
    )

    filter_key, filter_range = analysis.get_filter_key_range(args.filter_modes[1])
    other_df = analysis.merge_dfs(
        *analysis.extract_dfs(other_bits),
        filter_key=filter_key, filter_range=filter_range
    )

    merged_df = base_df.merge(other_df, on=None)

    merged_df.to_csv(outfile, index=False)
    print(f'Wrote CSV to {outfile}')
