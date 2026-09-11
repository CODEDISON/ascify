#!/usr/bin/env python3
"""Check generated half2 lowering; no mocked arithmetic or device execution."""
import argparse
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--translated', type=Path, required=True)
a = p.parse_args()
s = a.translated.read_text()
required = [
    '#include <ascify/half2_compat.hpp>',
    'initialized = ::ascify::make_half2_from_halves(0.0f, 1.0f)',
    '__haddx2(a[0], b[0])',
    '__hfmax2(a[1], b[1], initialized)',
    '__haddx2(__hmulx2((a[2]), (b[2])), (initialized))',
    '__haddx2((0, a[4]), (b[4]))',
    '__haddx2((a[threadIdx.x]), (b[threadIdx.x]))',
    'ascify::float2half2_rn(3.0f)',
    'half2 untouched(1.0f, 2.0f)',
    'half2 host_value(1.0f, 2.0f)',
]
for text in required:
    if text not in s:
        raise SystemExit('missing half2 contract: ' + text)
for name in ['__hadd2', '__hfma2', '__float2half2_rn']:
    if re.search(r'\b' + name + r'\s*\(', s):
        raise SystemExit('CUDA intrinsic was not lowered: ' + name)
print('half2 generated-code contracts passed')
