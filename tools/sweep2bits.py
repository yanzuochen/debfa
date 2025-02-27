#! /usr/bin/env python3

import sys
import os

from scipy.stats import mode
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import flipsweep
import pandas as pd
import utils
import cfg
import analysis

sweepfiles = sys.argv[1:]

for sweepfile in sweepfiles:
    outfile = f'{cfg.bits_dir}/{os.path.basename(sweepfile).replace("-sweep.pkl", "-bits.json")}'

    sweepret = flipsweep.load_sweep_result(sweepfile)
    bis = [
        utils.BinaryInfo(
            getattr(sweepret.args, 'compiler', 'tvm'),
            getattr(sweepret.args, 'compiler_version', 'main'),
            sweepret.args.model_name, d, avx=not sweepret.args.no_avx, opt_level=sweepret.args.opt_level
        )
        for d in sweepret.args.datasets
    ]

    ana = utils.load_json(f'{cfg.analysis_dir}/{bis[0].fname}-analysis.json')
    init_start = ana['memory_map']['.init']['base10_start']
    compute_fns = ana['compute_fns']
    offset2inst = analysis.get_offset2inst(ana)

    total_nbits = len(list(flipsweep.get_all_bytes(compute_fns))) * 8
    assessed_nbits = len(sweepret.retcoll_map) * 8
    if assessed_nbits < total_nbits:
        utils.warn(f'Only assessed {assessed_nbits}/{total_nbits} ({assessed_nbits / total_nbits * 100:.2f}%) bits')

    df = pd.DataFrame(sweepret.flat_result_colls)

    df['g_offset'] = df['base10_offset'].map(hex)
    df['ft_offset'] = df['base10_offset'].map(lambda x: hex(x - init_start))

    df['fn'] = df['base10_offset'].map(lambda x: [
        name for name, cf in compute_fns.items() if cf['base10_offset'] <= x < cf['base10_offset'] + cf['size']
    ][0])
    df['fn_size'] = df['fn'].map(lambda x: compute_fns[x]['size'] * 8)  # Size in bits

    df['label_modes'] = df['top_labels_list'].map(
        lambda x: [mode(labels)[0][0].flatten().tolist()[0] for labels in x]
    )
    df.drop(columns=['top_labels_list'], inplace=True)

    # If older sweep files don't contain these metric columns, fill them with default values
    for metric_name in ['top_label_change_pcts', 'lpips_avgs', 'fids']:
        if metric_name not in df.columns:
            df[metric_name] = df['g_offset'].map(
                lambda _: [-1.] * len(sweepret.args.datasets)
            )

    # Insert empty column for later
    df['flip_direction'] = -1

    # Move some columns to the front
    for col in reversed(['g_offset', 'ft_offset', 'bitidx', 'flip_direction']):
        df.insert(0, col, df.pop(col))

    results = df.to_dict(orient='records')

    for result in results:
        inst = result.setdefault('inst', {})
        inst_info = offset2inst.get(result['base10_offset'])
        if inst_info is None:
            inst['valid'] = False
            continue
        inst['valid'] = True
        inst['asm'] = inst_info['asm']
        local_byte_offset = result['base10_offset'] - inst_info['base10_offset']
        imask = int(inst_info['imask'].split('_')[local_byte_offset], 16)
        inst['opcode_flipped'] = bool(imask & (1 << result['bitidx']))
        byte = int(inst_info['bytes'].split('_')[local_byte_offset], 16)
        result['flip_direction'] = 1 - ((byte >> result['bitidx']) & 1)  # 1 for 0->1

    ret = {
        'meta': {
            'mod_fnames': [bi.fname for bi in bis]
        },
        'results': results,
    }
    utils.save_json(ret, outfile)
    print(f'Exported bit list to {outfile}')
