#! /usr/bin/env python3

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import pandas as pd
from functools import partialmethod
from tqdm import tqdm
import random
from typing import Dict, NamedTuple

import analysis
import cfg
import utils

def get_bitflips(bi, filter_mode, with_acc=False) -> Dict[analysis.BitFlip, float]:
    '''Returns a dict mapping bitflips to their accuracies.'''
    bits_path = f'{cfg.bits_dir}/{bi.fname.replace(".so", "-bits.json")}'
    filter_key, filter_range = analysis.get_filter_key_range(filter_mode)
    bits_infos = analysis.merge_dfs(
        *analysis.extract_dfs(bits_path),
        drop_metrics_cols=(not with_acc),
        filter_key=filter_key, filter_range=filter_range,
    ).to_dict(orient='records')

    return {
        analysis.BitFlip(x['base10_offset'], x['bitidx'], x['flip_direction']): x.get('correct_pct')
        for x in bits_infos
    }

class AttackResult(NamedTuple):
    success: bool
    nflip_attempts: int
    ncrashes: int
    after_acc: int

def run_attack(rh, _attacker_bits, vulnerable_bits, non_crash_bits):
    nflip_attempts, ncrashes, after_acc = 0, 0, 0
    tried_bits = set()  # Should be orig bits
    attacker_bits = {}

    def victim_restart():
        tmp_list = list(_attacker_bits.items())
        random.shuffle(tmp_list)
        attacker_bits.clear()
        attacker_bits.update(dict(tmp_list))
        rh.unplace_all_pages()

    victim_restart()

    while attacker_bits:
        bit, _ = attacker_bits.popitem()
        if bit in tried_bits:
            continue
        if rh.get_page_containing(bit).placed:
            continue
        pagenum, ploffs = rh.get_avail_placements(bit)
        if not ploffs:
            continue
        ploff = random.choice(ploffs)
        print(f'Flipping {bit} in page {pagenum} using placement 0x{ploff:x}')
        rh.place_page(pagenum, ploff)
        nflip_attempts += 1
        tried_bits.add(bit)
        if bit in vulnerable_bits:
            after_acc = vulnerable_bits[bit]
            print(f'Success with {nflip_attempts} attempts and {ncrashes} crashes; acc: {after_acc:.2f}')
            return AttackResult(True, nflip_attempts, ncrashes, after_acc)
        if bit not in non_crash_bits:
            # If crash, the victim binary is restarted and reset
            print('Crash')
            ncrashes += 1
            victim_restart()
            continue
    else:
        print(f'Failed after {nflip_attempts} attempts and {ncrashes} crashes')
        return AttackResult(False, nflip_attempts, ncrashes, -1)

def main(args):
    victim_bi, attacker_bi = analysis.superbits_fname2bis(args.superbits_path)
    non_crash_bits = get_bitflips(victim_bi, 'none')
    vulnerable_bits = get_bitflips(victim_bi, args.goal, with_acc=True)
    attacker_bits = get_bitflips(attacker_bi, args.attacker_filter)
    rh = analysis.RHModel(victim_bi, attacker_bits)

    seeds = [42 + i for i in range(args.repeat)]
    rets = []
    for seed in seeds:
        print(f'Running attack with seed {seed}')
        random.seed(seed)
        ret = run_attack(rh, attacker_bits, vulnerable_bits, non_crash_bits)
        rets.append(ret)
        print('-----')

    avg_nflip_attempts = sum(x.nflip_attempts for x in rets) / len(rets)
    avg_ncrashes = sum(x.ncrashes for x in rets) / len(rets)
    avg_after_acc = sum(x.after_acc for x in rets if x.after_acc > 0) / len(rets)

    summary_msg = f'''{os.path.basename(args.superbits_path)}
Goal: {args.goal}
Attacker filter: {args.attacker_filter}
Repeat: {args.repeat}
Seeds: {seeds}

Average number of flip attempts: {avg_nflip_attempts}
Average number of crashes: {avg_ncrashes}
Average accuracy after attack: {avg_after_acc:.2f}

Details:
{pd.DataFrame(rets).round(2).to_string(index=False)}
'''

    if not args.quiet:
        print('\n' + summary_msg)

    if args.export:
        export_path = f'{cfg.results_dir}/rowhammer/{os.path.basename(args.superbits_path).replace("-superbits.csv", "-rowhammer.log")}'
        utils.ensure_dir_of(export_path)
        with open(export_path, 'w+') as f:
            f.write(summary_msg)
        print(f'Exported to {export_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('superbits_path')
    parser.add_argument('-a', '--attacker-filter', choices=['none', 'acc', 'label', 'lpips', 'fid'], default='label', help='How the attacker should filter the list of bits to attempt')
    parser.add_argument('-g', '--goal', choices=['none', 'acc', 'label', 'lpips', 'fid'], default='acc', help='The goal of the attack')
    parser.add_argument('-r', '--repeat', type=int, default=5, help='How many times to repeat the attack')
    parser.add_argument('-q', '--quiet', action='store_true', help='Be more quiet')
    parser.add_argument('-e', '--export', action='store_true', help='Whether to export the results')
    args = parser.parse_args()

    if args.quiet:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # Disable tqdm by default

    main(args)
