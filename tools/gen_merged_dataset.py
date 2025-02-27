#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import numpy as np
import torch
import random
from tqdm import tqdm

from dataman import MergedDataset
import cfg

def prim_target(t):
    if isinstance(t, torch.Tensor):
        assert t.dim() == 0 or t.size(0) == 1
        t = t.item()
    return t

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nclasses', default=10)
    parser.add_argument('--outdir', default=f'{cfg.datasets_root}/merged')
    parser.add_argument('output_dataset_name')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    md = MergedDataset(None)

    datasets_train = {
        k: MergedDataset.create_dataset(k, True) for k in tqdm(
            MergedDataset.dataset_classes.keys(), desc='Loading datasets (train set)'
        )
    }
    datasets_test = {
        k: MergedDataset.create_dataset(k, False) for k in tqdm(
            MergedDataset.dataset_classes.keys(), desc='Loading datasets (test set)'
        )
    }

    datasets_targets_avail = {
        ds_name: list(
            set(prim_target(y) for _, y in datasets_test[ds_name])
        )
        for ds_name in tqdm(datasets_test.keys(), desc='Collecting available targets')
    }

    # Generate target refs
    for _ in tqdm(range(args.nclasses), desc='Generating target refs'):
        while True:
            dataset_name = random.choice(list(datasets_train.keys()))
            dataset = datasets_train[dataset_name]
            tref = (dataset_name, random.choice(datasets_targets_avail[dataset_name]))
            if tref not in md.target_refs:
                break
        md.datasets[dataset_name] = dataset
        md.target_refs.append(tref)

    print(f'Generated target refs: {md.target_refs}')

    for ti, tref in enumerate(tqdm(md.target_refs, desc='Generating data refs for classes')):
        dataset_name, wanted_cls = tref

        for idx, (_, y) in enumerate(tqdm(datasets_train[dataset_name], desc='Train set')):
            if prim_target(y) != wanted_cls:
                continue
            md.train_data_refs.append((dataset_name, idx))
            md.train_targets.append(ti)
        assert len(md.train_data_refs)

        for idx, (_, y) in enumerate(tqdm(datasets_test[dataset_name], desc='Test set')):
            if prim_target(y) != wanted_cls:
                continue
            md.test_data_refs.append((dataset_name, idx))
            md.test_targets.append(ti)
        assert len(md.test_data_refs)

    # Calculate std and mean
    print('Calculating mean and std...')
    md.calc_mean_std()

    outfile = f'{args.outdir}/{args.output_dataset_name}.json'
    md.save(outfile)
    print(f'Saved to: {outfile}')
