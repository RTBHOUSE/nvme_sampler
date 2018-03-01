#include "nvme_sampler.h"
#include "nvme_api.h"

namespace nvme_sampler {
namespace api {

struct SamplerHandle {
    void *user_data; // order is important
    NvmeSampler *sampler;
    AllocatorFun allocator;
    DeleterFun deleter;
};

handle init_sampler(UserDataPtr user_data,
                    AllocatorFun allocator,
                    DeleterFun deleter,
                    std::string const &file_path,
                    int64_t num_rows,
                    int64_t row_size,
                    int64_t max_batch_elements,
                    int64_t max_num_threads,
                    int64_t memory_usage_limit_b,
                    int32_t seed
) {

    TensorDescription tensor_description = {
            .num_rows = num_rows,
            .row_size_b = row_size,
            .file_path = file_path
    };

    SamplerConfig config = {
            .max_batch_elements = max_batch_elements,
            .max_num_threads = max_num_threads,
            .memory_usage_limit_b = memory_usage_limit_b,
            .seed = seed
    };

    SamplerHandle *handle = new SamplerHandle{
            .user_data = user_data,
            .sampler = new NvmeSampler(
                    tensor_description,
                    config,
                    BatchBlocks::Allocator{
                            .allocator = [allocator, user_data](size_t size) { return allocator(user_data, size); },
                            .deleter = [deleter, user_data](byte *addr) { return deleter(user_data, addr); }
                    }),
            .allocator = allocator,
            .deleter = deleter,
    };

    return handle;
}

void destroy_sampler(SamplerHandle *handle) {
    delete handle->sampler;
}

byte *read_batch(handle sampler, long batch_size) {
    return sampler->sampler->get_next_batch(batch_size);
}

}
}
