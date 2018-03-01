#include "nvme_sampler.h"

using namespace nvme_sampler;

int main(int argc, char **argv) {
    assert(argc == 7);

    auto file_path = std::string(argv[1]);
    auto row_size = std::atol(argv[2]);
    auto num_rows = std::atol(argv[3]);
    auto max_batch_elements = std::atol(argv[4]);
    auto max_num_threads = std::atoi(argv[5]);
    auto memory_usage_limit_b = std::atol(argv[6]);

    TensorDescription tensor_description = {
            .num_rows = num_rows,
            .row_size_b = row_size,
            .file_path = file_path,
    };

    SamplerConfig config = {
            .max_batch_elements = max_batch_elements,
            .max_num_threads = max_num_threads,
            .memory_usage_limit_b = memory_usage_limit_b,
    };

    NvmeSampler sampler(tensor_description, config, create_default_allocator());

    for (int i = 0; i < tensor_description.num_rows * 4; ++i) {
        sampler.get_next_batch(1);
    }

    return 0;
}


