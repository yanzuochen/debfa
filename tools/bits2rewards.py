#! /usr/bin/env python3

# TODO: Scale of accuracy reward values

import os
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

from typing import Any, Dict, List
import cfg
import utils

def export_rewards_info(bitsfile):
    bitsinfo = utils.load_json(bitsfile)
    ana = utils.load_json(f'{cfg.analysis_dir}/{bitsinfo["meta"]["mod_fnames"][0]}-analysis.json')

    outfile = f'{cfg.rewards_dir}/{os.path.basename(bitsfile).replace("-bits.json", "-rewards.json")}'

    offset2metrics: Dict[int, Dict[str, List[Any]]] = {}
    for bitret in bitsinfo['results']:
        offset = bitret['base10_offset']
        # Note that multiple bitret's can have the same offset
        metrics = offset2metrics.setdefault(offset, {})
        for k in ['correct_pcts', 'top_label_change_pcts', 'lpips_avgs', 'fids']:
            metrics.setdefault(k, []).extend(bitret[k])

    # Max reward values for each byte
    offset2rewards: Dict[int, Dict[str, float]] = {
        offset: {
            'acc': 100 - min(offset2metrics[offset]['correct_pcts']),
            'label_change': max(offset2metrics[offset]['top_label_change_pcts']),
            'lpips': max(offset2metrics[offset]['lpips_avgs']),
            'fid': max(offset2metrics[offset]['fids']),
        }
        for offset in offset2metrics.keys()
    }
    reward_names = tuple(next(iter(offset2rewards.values())).keys())

    # One instruction can include multiple bytes, so we find the max reward of
    # all the bytes in the instruction.
    def calc_inst_rewards(inst) -> Dict[str, float]:
        offset = inst['base10_offset']
        nbytes = inst['nbytes']
        rewards_in_range = [
            offset2rewards[offset+i]
            for i in range(nbytes) if offset+i in offset2rewards
        ]
        return {
            k: max([0] + [r[k] for r in rewards_in_range])
            for k in reward_names
        }

    ret = {
        fname: [
            {
                'inst': inst['asm'],
                'base10_offset': inst['base10_offset'],
                'nbytes': inst['nbytes'],
                'noperands': len(inst['omasks']),
                'rewards': calc_inst_rewards(inst),
            }
            for inst in fn['insts']
        ]
        for fname, fn in ana['compute_fns'].items()
    }

    utils.save_json(ret, outfile, sorted=True)
    print(f'Output written to {os.path.abspath(outfile)}')

for bitsfile in sys.argv[1:]:
    export_rewards_info(bitsfile)
