import cffi
import torch
import random

from ._ext import native_sampler as lib


class NvmeSampler(object):
    def __init__(self, file_path, num_rows, row_size_b, max_batch_elements, max_num_threads=8, memory_usage_limit_b=8 * 2 ** 30, seed=None):
        """
        :param row_size_b: sample size in bytes
        :param max_batch_elements must be greater or equal to any batch_size param passed read_batch
        """
        self.buffer = torch.FloatTensor()

        if seed is None:
            seed = random.randint(-1 << 31, (1 << 31) - 1)

        num_rows, row_size_b, max_batch_elements, max_num_threads, memory_usage_limit_b = map(
            int, [num_rows, row_size_b, max_batch_elements, max_num_threads, memory_usage_limit_b])

        assert row_size_b % 4 == 0

        ffi = cffi.FFI()
        file_path = ffi.new("char[]", file_path.encode('utf8'))  # TODO test non-ascii paths

        self.handle = lib.init_sampler(
            self.buffer, file_path, num_rows, row_size_b, max_batch_elements, max_num_threads, memory_usage_limit_b, seed)
        self.row_size_b = row_size_b
        self.row_size = row_size_b // 4
        self.num_rows = num_rows

    def read_batch(self, batch_size):
        """
        Reads next batch.

        :param batch_size must be smaller than max_batch_elements
        """
        offset = lib.read_batch(self.handle, batch_size)
        if offset < 0:
            raise Exception("Failed to read data")

        return self.buffer[offset: offset + batch_size * self.row_size].view(batch_size, self.row_size_b // 4)

    def __del__(self):
        lib.destroy_sampler(self.handle)
