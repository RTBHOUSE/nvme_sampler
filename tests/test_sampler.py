import math
import os.path

import numpy as np
import scipy.stats
import torch

from nvme_sampler import NvmeSampler

NVME_WORKDIR = os.getenv("NVME_WORKDIR")

assert NVME_WORKDIR is not None, "'NVME_WORKDIR' environment is not set"

file_path = os.path.join(NVME_WORKDIR, "nvme_test.bin")


def create_file(num_rows, row_size_b):
    row_size = row_size_b // 4

    storage = torch.FloatStorage.from_file(file_path, shared=True, size=row_size * num_rows)
    tensor = torch.FloatTensor(storage)

    # idx, idx+0.25, idx+0.125, idx+0.25, idx+0.125, ..., idx+0.5
    tensor = tensor.view(num_rows, row_size)
    tensor.fill_(0)
    tensor[:, 0] = torch.arange(0, num_rows)
    tensor[:, 1:row_size] += tensor[:, 0].clone().view(-1, 1)
    tensor[:, 1:row_size:2] += 0.25
    tensor[:, 2:row_size:2] += 0.125
    tensor[:, row_size - 1] = tensor[:, 0] + 0.5

    return tensor


def test_sampler(num_rows, row_size_b, num_samples):
    print("Checking row_size_b=%d" % row_size_b)

    create_file(num_rows=num_rows, row_size_b=row_size_b)

    row_size = row_size_b // 4

    sampler = NvmeSampler(file_path,
                          num_rows=num_rows,
                          row_size_b=row_size_b,
                          max_batch_elements=128,
                          max_num_threads=8,
                          memory_usage_limit_b=2 * 2 ** 24)

    counts = np.zeros(num_rows)
    for i in range(num_samples):
        if i % 50_000 == 0:
            print("Progress: %5.2f" % (100.0 * (i / num_samples)))

        t = sampler.read_batch(1)
        idx = t[0][0]
        idx_inc1 = t[0][1]
        idx_inc2 = t[0][2]
        idx_inc3 = t[0][row_size - 1]
        assert 0 <= idx < num_rows, idx
        assert abs((idx_inc1 - idx) - 0.25) < 0.0001, idx_inc1
        assert abs((idx_inc2 - idx) - 0.125) < 0.0001, idx_inc2
        assert abs((idx_inc3 - idx) - 0.5) < 0.0001, idx_inc3

        sampled_contents = t[0]
        expected_contents = torch.ones(row_size) * idx
        expected_contents[1:row_size:2] += 0.25
        expected_contents[2:row_size:2] += 0.125
        expected_contents[row_size - 1] = idx + 0.5

        assert (sampled_contents - expected_contents).abs().sum() < 0.01, (sampled_contents, expected_contents)
        counts[int(idx)] += 1

    percentile_q = np.array([1, 10, 25, 50, 75, 90, 99])

    result_mean = np.mean(counts)
    result_std = np.std(counts)
    result_max = np.max(counts)
    result_percentiles = np.percentile(counts, percentile_q)

    expected_mean = num_samples / num_rows
    expected_std = math.sqrt(expected_mean)
    expected_max_ub = expected_mean + expected_std * 6
    expected_percentiles = scipy.stats.norm(expected_mean, expected_std).ppf(percentile_q / 100.0)

    print(expected_mean, expected_std, expected_max_ub, expected_percentiles)
    print(result_mean, result_std, result_max, result_percentiles)

    assert (counts == 0).sum() <= (2 * 3 * 4096) / row_size, (counts == 0).sum()
    assert result_max < expected_max_ub
    assert abs(expected_mean - result_mean) < expected_mean * 0.1
    assert abs(expected_std - result_std) < expected_std * 0.1
    assert (np.abs(expected_percentiles - result_percentiles) >= expected_percentiles * 0.1).sum() == 0


test_sampler(num_rows=100_000, row_size_b=1016, num_samples=25 * 100_000)

for row_size_b in range(288, 2001, 4):
    test_sampler(num_rows=100_000, row_size_b=row_size_b, num_samples=25 * 100_000)
