#include <TH/TH.h>
#include <assert.h>
#include <string>

#include "nvme_api.h"

typedef void *handle;

extern "C" {

struct UserData {
    UserData(THFloatTensor *buffer) : buffer(buffer) {}

    THFloatTensor *buffer;
};

inline UserData *get_user_data(void *sampler) {
    return reinterpret_cast<UserData *>(*reinterpret_cast<long *>(sampler));
}

nvme_sampler::api::byte *allocator(void *user_data_opaque, size_t size) {
    UserData *user_data = reinterpret_cast<UserData *>(user_data_opaque);
    THFloatTensor_resize1d(user_data->buffer, size / sizeof(float));

    return reinterpret_cast<nvme_sampler::api::byte *>(user_data->buffer->storage->data);
}

void deleter(void *user_data_opaque, nvme_sampler::api::byte *address) {
    UserData *user_data = reinterpret_cast<UserData *>(user_data_opaque);
    assert(reinterpret_cast<float *>(address) == user_data->buffer->storage->data);
    THFloatTensor_resize1d(user_data->buffer, 0);
}

handle init_sampler(THFloatTensor *buffer,
                    const char *file_path,
                    int64_t num_rows,
                    int64_t row_size,
                    int64_t max_batch_elements,
                    int64_t max_num_threads,
                    int64_t memory_usage_limit_b,
                    int32_t seed

) {
    UserData *user_data = new UserData(buffer);
    handle sampler = nvme_sampler::api::init_sampler(
            user_data,
            &allocator,
            &deleter,
            std::string(file_path),
            num_rows,
            row_size,
            max_batch_elements,
            max_num_threads,
            memory_usage_limit_b,
            seed
    );
    return sampler;
}

void destroy_sampler(handle sampler) {
    UserData * user_data = get_user_data(sampler);
    nvme_sampler::api::destroy_sampler(reinterpret_cast<nvme_sampler::api::handle>(sampler));
    delete user_data;

}

long read_batch(handle sampler, long batch_size) {
    float *addr = reinterpret_cast<float *>(nvme_sampler::api::read_batch(reinterpret_cast<nvme_sampler::api::SamplerHandle *>(sampler), batch_size));
    return addr - get_user_data(sampler)->buffer->storage->data;
}

}

