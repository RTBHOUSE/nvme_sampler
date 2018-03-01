#!/usr/bin/env python3

import os

from torch.utils.ffi import create_extension

this_dir = os.path.abspath(os.path.dirname(__file__))

sources = ['nvme_sampler/src/nvme_sampler.cpp']
headers = ['nvme_sampler/src/nvme_sampler.h']
defines = []

ffi = create_extension(
    'nvme_sampler._ext.native_sampler',
    headers=headers,
    sources=sources,
    library_dirs=[os.path.join(this_dir, "lib/bin")],
    runtime_library_dirs=[os.path.join(this_dir, "lib/bin")],
    libraries=["nvme_sampler"],
    include_dirs=[os.path.join(this_dir, "lib/src")],
    define_macros=defines,
    relative_to=__file__,
    with_cuda=False,
    package=True
)

if __name__ == '__main__':
    ffi.build()
