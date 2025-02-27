#! /usr/bin/env python3

# Removes all files matching the given regex from the Ghidra project.

import sys
import os
import re

rep_dir = f'{os.path.dirname(os.path.realpath(__file__))}/db/debfa.rep'
index_file = f'{rep_dir}/idata/~index.dat'

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <regex>')
    sys.exit(1)

regex = re.compile(rf'\s*([0-9a-f]+):{sys.argv[1]}:.+')

ids = []
with open(index_file, 'r') as f:
    for line in f:
        if re.match(regex, line):
            id = line.split(':')[0].strip()
            print(f'Removing: {line}')
            ids.append(id)

if not ids:
    print('No matches found')
    sys.exit(0)

for id in ids:
    os.system(f'sed -i "/{id}:/d" "{index_file}"')
    os.system(f'find "{rep_dir}" -name "~{id}.db" -exec rm -r "{{}}" +')
    os.system(f'find "{rep_dir}" -name "{id}.prp" -delete')
