// Basic API that can be used to integrate NvmeSampler with PyTorch or torch

namespace nvme_sampler {
namespace api {

struct SamplerHandle;

typedef SamplerHandle *handle;
typedef unsigned char byte;
typedef void *UserDataPtr;

typedef byte *(*AllocatorFun)(UserDataPtr, size_t);

typedef void(*DeleterFun)(UserDataPtr, byte *);

// to make it deterministic set seed to some value AND max_num_threads to 1
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
);

// Destroys sampler. Sampler destruction uses deleter to deallocate batch buffer.
// You must not use handle after calling this function.
void destroy_sampler(handle sampler);

byte *read_batch(handle sampler, long batch_size);

}
}
