# Ghidrathon script, python 3
#@menupath Tools.Extract Loops Info

import os
import ghidra.app.decompiler as decomp
from ghidra.app.decompiler import *
from ghidra.util.task import ConsoleTaskMonitor

import json

DEBUG = 0

outfile = f'./ghidra/loops/{currentProgram.getName()}-loops.json'

class ClangNodeInfo:
    def __init__(self, node, in_func, depth: int):
        self.node = node
        self.in_func = in_func
        self.depth = depth

        self.min_addr = node.getMinAddress()
        self.max_addr = node.getMaxAddress()
        self.the_addr = None
        if self.min_addr == self.max_addr:
            self.the_addr = self.min_addr
        self.the_offset = None
        if self.the_addr:
            self.the_offset = self.the_addr.getOffset()

        self.inst, self.inst_len, self.pcode = [None]*3
        if self.the_addr:
            self.inst = getInstructionAt(self.the_addr)
            if self.inst:
                self.inst_len = self.inst.getLength()
                self.pcode = list(self.inst.getPcode())

        self.cls_name = node.__class__.__name__
        self.text = node.toString()

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Fixes a Jep bug?
def _isinstance(obj, cls):
    return isinstance(obj, cls.__pytype__)

def get_all_instructions():
    listing = currentProgram.getListing()
    instructions = listing.getInstructions(True)
    return instructions

def visit_clang_node(node, cb, depth=0):
    in_func = node.getClangFunction().getHighFunction().getFunction().getName()
    node_info = ClangNodeInfo(node, in_func, depth)
    if DEBUG:
        print(f'Processing: {node_info.cls_name} ({node_info.min_addr}, {node_info.max_addr}): {node}')
    if cb(node_info) and DEBUG:
        print('-- Accepted')
    if node.numChildren() > 0:
        for child in node:
            visit_clang_node(child, cb, depth=depth+1)

def prog_pcode():
    listing = currentProgram.getListing()
    instructions = listing.getInstructions(True)
    for op in instructions:
        raw_pcode = op.getPcode()
        for entry in raw_pcode:
            yield entry

def visit_function_clang_nodes(decomp_iface, monitor, f, cb):
    results = decomp_iface.decompileFunction(f, 0, monitor)
    visit_clang_node(results.getCCodeMarkup(), cb)

def extract_loops_info():
    decomp_iface = decomp.DecompInterface()
    decomp_iface.openProgram(currentProgram)

    node_infos = []

    def cb(node_info):
        if node_info.node.toString() == 'while':
            if not node_info.the_offset:
                print(f'Warning: while node in func {node_info.in_func} has no address (while-true-break?); skipping')
                return False
            assert node_info.the_offset, f'while clause with unexpected address info: {node_info.min_addr} - {node_info.max_addr}'
            node_infos.append(node_info)
            return True
        return False

    all_fns = list(currentProgram.getFunctionManager().getFunctions(True))

    monitor = ConsoleTaskMonitor()

    for i, f in enumerate(all_fns):
        print(f'Processing function {i+1}/{len(all_fns)} {f.getName()}')
        visit_function_clang_nodes(decomp_iface, monitor, f, cb)

    decomp_iface.stopProcess()

    ret = {
        hex(ni.the_offset): {
            'base10_offset': ni.the_offset,
            'in_func': ni.in_func,
            'depth': ni.depth,
            'inst': str(ni.inst),
            'inst_len': ni.inst_len,
            'pcode': [str(x) for x in ni.pcode],
        } for ni in node_infos
    }

    return ret

ensure_dir_of(outfile)
loops = extract_loops_info()
with open(outfile, 'w+') as f:
    json.dump(loops, f, indent=4)
print(f'Output written to {os.path.abspath(outfile)}')
