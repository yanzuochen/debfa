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
    parser.add_argument('paths', nargs='+')
    parser.add_argument('-f', '--filter-mode', choices=['none', 'acc', 'label', 'lpips', 'fid'], default='none')
    args = parser.parse_args()

    for path in args.paths:
        dfs = analysis.extract_dfs(path)
        outfile = f'{cfg.debug_dir}/sweep-csv/{os.path.basename(path)}.csv'
        utils.ensure_dir_of(outfile)

        filter_key, filter_range = analysis.get_filter_key_range(args.filter_mode)
        df = analysis.merge_dfs(
            *dfs,
            filter_key=filter_key, filter_range=filter_range, drop_metrics_cols=False
        ).round(2)
        # Move some columns to the front
        for col in reversed(['g_offset', 'ft_offset', 'bitidx']):
            df.insert(0, col, df.pop(col))

        df.to_csv(outfile, index=False)
        print(f'Wrote CSV to {outfile}')
