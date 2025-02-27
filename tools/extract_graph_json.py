#! /usr/bin/env python3

import sys
import os
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/..')

import json
import mmap
import struct

import cfg
import utils

output_root = f'{cfg.debug_dir}/graph-jsons'

def get_graph_json(bi: utils.BinaryInfo, return_obj=False):
    import modman
    irmod, params = modman.get_irmod(
        bi.model, bi.dataset, 'none', 1, 32, 3
    )
    json_str, _ir = modman.get_json_and_ir(irmod, params)
    json_str = '\n'.join([x.rstrip() for x in json_str.splitlines()])
    obj = json.loads(json_str)
    return obj if return_obj else json.dumps(obj, indent=2, sort_keys=True)

def extract_graph_json(fname, return_obj=False):
    marker = b'GraphExecutorFactory'
    with open(fname, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        start = mm.find(marker)
        mm.seek(start + len(marker))
        # Read an 8-byte length
        length = struct.unpack('<Q', mm.read(8))[0]
        # Read the length-byte string
        lines = mm.read(length).decode('utf-8')
        mm.close()
    # Clean up trailing spaces
    lines = [x.rstrip() for x in lines.splitlines()]
    json_str = '\n'.join(lines)
    obj = json.loads(json_str)
    return obj if return_obj else json.dumps(obj, indent=2, sort_keys=True)

if __name__ == '__main__':
    files = sys.argv[1:]
    for f in files:
        print(f'Processing {f}')
        json_str = extract_graph_json(f)
        outfile = f'{output_root}/{os.path.basename(f)}.json'
        utils.ensure_dir_of(outfile)
        with open(outfile, 'w+') as f:
            f.write(json_str)
        print(f'Wrote {outfile}')
