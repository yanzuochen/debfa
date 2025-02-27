# Ghidrathon script, python 3
#@menupath Tools.Export disasm before/after flips

import os
import json
import csv
import utils
from typing import List, NamedTuple, Union
from ghidra.program.disassemble import Disassembler
from ghidra.program.model.address import AddressSet
from ghidra.program.model.listing import Instruction

class DisasmInfo(NamedTuple):
    g_offset: str
    base10_offset: int
    bitidx: int
    inst_base10_offset: int
    inst_len: int
    from_byte: str
    to_byte: str
    before: str
    after: List[str]

# getDisassembler(Program program, boolean markBadInstructions, boolean markUnimplementedPcode, boolean restrictToExecuteMemory, TaskMonitor monitor, DisassemblerMessageListener listener)
disasm = Disassembler.getDisassembler(currentProgram, False, False, False, utils.DUMMY_MONITOR, None)

def flip_and_disasm(offset, bitidx, first_addr, last_addr, complete_rest=False) -> List[Instruction]:
    byte_addr = toAddr(offset)
    addrset = AddressSet(first_addr, last_addr)
    # print(f'addrset: {addrset}')

    clearListing(addrset)
    old_byte = utils.get_byte(byte_addr)
    utils.set_byte(byte_addr, old_byte ^ (1 << bitidx))

    # Try to skip the first few bytes if they're undecodable
    addrs_disasmed = None
    while True:
        addrs_disasmed = disasm.disassemble(first_addr, addrset, False)
        if not addrs_disasmed.isEmpty():
            break
        addrset.delete(first_addr, first_addr)
        first_addr = first_addr.add(1)

    # Disassembler will stop at before normally unreachable instructions
    if complete_rest:
        while not addrs_disasmed.isEmpty() and addrs_disasmed.getMaxAddress() < last_addr:
            # Delete all previously disassembled instructions
            addrset.deleteFromMin(getInstructionAt(addrs_disasmed.getMinAddress()).getMaxAddress())
            # Skip existing instructions
            while True:
                inst = getInstructionAt(addrset.getMinAddress())
                if not inst:
                    break
                addrset.deleteFromMin(inst.getMaxAddress())
            addrs_disasmed = disasm.disassemble(addrset.getMinAddress(), addrset, False)

    return utils.get_insts_in_range(first_addr, last_addr)

def get_disasm_info(offset, bitidx, no_restore=False) -> Union[DisasmInfo, None]:
    # print(f'Flipping: (0x{offset:2x}, {bitidx})')
    addr = toAddr(offset)

    inst = getInstructionContaining(addr)
    if not inst:  # Perhaps unreachable address, not analysed by Ghidra
        return None
    inst_first_addr = inst.getMinAddress()
    disasm_last_addr = inst.getMaxAddress().add(14)  # x64 allows longest instruction to be 15 bytes
    func = getFunctionContaining(addr)
    if not func:
        return None
    func_last_addr = func.getBody().getMaxAddress()
    if disasm_last_addr > func_last_addr:
        disasm_last_addr = func_last_addr
    inst_base10_offset = inst_first_addr.getOffset()
    inst_len = inst.getLength()
    before = str(inst)
    from_byte = utils.get_byte(addr)
    to_byte = hex(from_byte ^ (1 << bitidx))
    from_byte = hex(from_byte)

    before_insts = utils.get_insts_in_range(inst_first_addr, disasm_last_addr)
    before_insts_map = {inst.getMinAddress().getOffset(): str(inst) for inst in before_insts}
    after_insts = flip_and_disasm(offset, bitidx, inst_first_addr, disasm_last_addr)
    # Can't use undo here (in a tight loop?) since Ghidra might crash

    after = []

    # Represent undecodable bytes just as bytes
    if after_insts:
        naddrs_skipped = after_insts[0].getMinAddress().getOffset() - inst_first_addr.getOffset()
        if naddrs_skipped:
            skipped_bytes = '?:' + ''.join(f'{x:02x}' for x in utils.get_bytes(inst_first_addr, naddrs_skipped))
            after.append(skipped_bytes)

    # Only retain changed instructions (after the first)
    for inst in after_insts:
        inst_str = str(inst)
        if after:
            before_inst = before_insts_map.get(inst.getMinAddress().getOffset())
            if before_inst and before_inst == inst_str:
                break
        after.append(inst_str)

    # Restore
    if not no_restore:
        flip_and_disasm(offset, bitidx, inst_first_addr, disasm_last_addr, complete_rest=True)

    ret = DisasmInfo(
        g_offset=hex(offset),
        base10_offset=offset,
        bitidx=bitidx,
        inst_base10_offset=inst_base10_offset,
        inst_len=inst_len,
        from_byte=from_byte,
        to_byte=to_byte,
        before=before,
        after=after,
    )
    return ret

if not isRunningHeadless():
    answer = askString(
        'Bit(s) to flip', '''Enter either of:
(a) a single bit index (0-7);
(b) an offset, bitidx pair: 0x123, 4;
(suffix R to disable restoring); or
(c) a path to a json file:'''
    )
else:
    answer = getScriptArgs()[0]
no_restore = answer.endswith('R')
if no_restore:
    answer = answer[:-1]

try:
    start()
    if answer.isnumeric():
        bitidx = int(answer)
        offset = currentLocation.getByteAddress().getOffset()
        disasm_info = get_disasm_info(offset, bitidx, no_restore=no_restore)
        print(disasm_info)
    elif ',' in answer:
        answer_split = [x.strip('( )\t\n') for x in answer.split(',')]
        offset = int(answer_split[0], 16)
        bitidx = int(answer_split[1])
        disasm_info = get_disasm_info(offset, bitidx, no_restore=no_restore)
        print(disasm_info)
    else:
        with open(answer) as f:
            loaded = json.load(f)
        if answer.endswith('-bits.json'):
            disasm_infos = [get_disasm_info(bit['base10_offset'], bit['bitidx']) for bit in loaded['results']]
            outfile = f'./debug/bits-asm/{os.path.basename(answer).replace("-bits.json", "-asm.json")}'
        elif answer.endswith('-analysis.json'):
            disasm_infos = []
            for cfname, cf in loaded['compute_fns'].items():
                # FIXME: Functions are not always continuous
                print(f'Flipping {cfname}')
                base = cf['base10_offset']
                disasm_infos.extend([get_disasm_info(base + byteoff, bitidx)
                                     for byteoff in range(cf['size']) for bitidx in range(8)])
            outfile = f'./debug/cfn-asm/{os.path.basename(answer).replace("-analysis.json", "-asm-cfn.json")}'
        utils.ensure_dir_of(outfile)
        disasm_infos = [x for x in disasm_infos if x]
        with open(outfile, 'w+') as f:
            json.dump([x._asdict() for x in disasm_infos], f, indent=4)
        print(f'Wrote json output to {os.path.abspath(outfile)}')
        outfile = outfile.replace('.json', '.csv')
        with open(outfile, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(DisasmInfo._fields)
            for disasm_info in disasm_infos:
                writer.writerow(disasm_info)
        print(f'Wrote csv output to {os.path.abspath(outfile)}')
finally:
    end(no_restore)
