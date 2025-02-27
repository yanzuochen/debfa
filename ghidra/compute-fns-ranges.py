# Ghidrathon script, python 3
#@menupath Tools.Extract Compute Functions Ranges

# DEPRECATED: Use export-analysis.py instead

import utils
from ghidra.app.decompiler import *

DEBUG = 0

# currentProgram.setName(currentProgram.getDomainFile().getName())  # Little rename tool

outfile = f'./ghidra/compute-ranges/{currentProgram.getName()}-compute-ranges.json'

def extract_ranges_info():
    monitor = utils.DUMMY_MONITOR

    ret = {
        f.getName(): {
            'base10_offset': f.getEntryPoint().getOffset(),
            # NOTE: Since functions aren't necessarily continuous, we use this to get the
            # approx. (can be a bit larger) range.
            'size': f.getBody().getMaxAddress().getOffset() - f.getBody().getMinAddress().getOffset() + 1,
            'called_by': [x.getName() for x in f.getCallingFunctions(monitor)],
        } for f in utils.get_compute_fns()
    }

    return ret

utils.maybe_do(outfile, extract_ranges_info)
