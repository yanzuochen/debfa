# Ghidrathon script, python 3
#@menupath Tools.Export Analysis

import os
import utils

outfile = f'./ghidra/analysis/{currentProgram.getName()}-analysis.json'

def friendly_hex(x):
    return '_'.join([f'{utils.j2pb(x):02x}' for x in x])

def export_fn(f):
    return {
        'base10_offset': f.getEntryPoint().getOffset(),
        # NOTE: Since functions aren't necessarily continuous, we use this to get the
        # approx. (can be a bit larger) range.
        'size': f.getBody().getMaxAddress().getOffset() - f.getBody().getMinAddress().getOffset() + 1,
        'called_by': [x.getName() for x in f.getCallingFunctions(utils.DUMMY_MONITOR)],
        'insts': [
            {
                'base10_offset': inst.getAddress().getOffset(),
                'asm': str(inst),
                'nbytes': inst.getLength(),
                'bytes': friendly_hex(inst.getBytes()),
                'imask': friendly_hex(inst.getPrototype().getInstructionMask().getBytes()),
                'omasks': [
                    friendly_hex(inst.getPrototype().getOperandValueMask(i).getBytes())
                    for i in range(inst.getNumOperands())
                ],
            }
            for inst in utils.get_insts_in_fn(f)
        ]
    }

def export_memory_map():
    return {
        blk.getName(): {
            'base10_start': blk.getStart().getOffset(),
            'size': blk.getSize(),
            'base10_end': blk.getEnd().getOffset(),
        }
        for blk in getMemoryBlocks()
    }

def export_analysis():
    utils.ensure_dir_of(outfile)

    ret = {
        'memory_map': export_memory_map(),
        'compute_fns': {
            f.getName(): export_fn(f) for f in utils.get_compute_fns()
        }
    }

    utils.save_json(ret, outfile)
    print(f'Output written to {os.path.abspath(outfile)}')

export_analysis()
