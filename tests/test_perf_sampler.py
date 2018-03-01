#!/usr/bin/env python3.6

import os.path
import time

from nvme_sampler import NvmeSampler

NVME_WORKDIR = os.getenv("NVME_WORKDIR")

assert NVME_WORKDIR is not None, "'NVME_WORKDIR' environment is not set"

file_path = os.path.join(NVME_WORKDIR, "100GiB_seq.bin")


def test_sampler():
    row_size_b = 1024
    num_rows = 100 * 2**30 // row_size_b
    sampler = NvmeSampler(file_path,
                          num_rows=num_rows,
                          row_size_b=row_size_b,
                          max_batch_elements=16384,
                          max_num_threads=8,
                          memory_usage_limit_b=8 * 2 ** 30)

    batch_size = 512
    start_time = time.time()
    for i in range(sampler.num_rows * 10):
        if i // batch_size % 1000:
            duration = time.time() - start_time
            num_samples_read = i * batch_size
            samples_per_sec = num_samples_read / duration
            bw_gib = num_samples_read * row_size_b / duration / (2 ** 30)
            print("Throughput: %f samples/s; %f GiB/s" % (samples_per_sec, bw_gib))
        t = sampler.read_batch(batch_size)
        # t.sum()

