#pragma once

#include "calculator.h"
#include "utils.h"

#include <cstddef>
#include <vector>
#include <functional>
#include <memory>

namespace nvme_sampler {

using SamplingParametersCalculator::PAGE_SIZE;

struct BatchBlock {
    const int64_t element_size_b;
    const int64_t num_samples;
    int64_t read_idx = 0; // index of next element to read
    Buffer buffer{.size =  0, .buffer = NULL};

    BatchBlock(BatchBlock const &other) = delete;

    BatchBlock &operator=(BatchBlock const &) = delete;

    BatchBlock(int64_t element_size_b, int64_t num_samples, byte *buffer_address) :
            element_size_b(element_size_b),
            num_samples(num_samples),
            buffer(Buffer{
                    .size = element_size_b * num_samples,
                    // align buffer to allow AVX-based memcpy and reduce crossing page boundaries
                    .buffer = align_up_ptr(buffer_address, PAGE_SIZE)
            }) {}

    int64_t get_num_samples_left() const {
        return this->num_samples - this->read_idx;
    }

    byte *read_next_batch(int32_t batch_size) {
        ASSERT(this->get_num_samples_left() >= batch_size, "%ld", this->get_num_samples_left());

        byte *buf = this->buffer.buffer + (read_idx * element_size_b);
        this->read_idx += batch_size;

        return buf;
    }
};

typedef BatchBlock *BatchBlockPtr;


struct BatchBlocks {
    typedef struct {
        std::function<byte *(size_t)> allocator;
        std::function<void(byte *)> deleter;
    } Allocator;

    static const int32_t NUM_BLOCKS = SamplingParametersCalculator::NUM_BATCH_BLOCKS;
    Allocator allocator;
    byte *user_buffer;
    std::vector<std::shared_ptr<BatchBlock>> batch_blocks;
    BlockingQueue<BatchBlockPtr> ready_blocks;

    BatchBlocks(int64_t element_size_b, int64_t num_samples, Allocator allocator)
            :
            allocator(allocator),
            user_buffer(allocator.allocator((element_size_b * num_samples + PAGE_SIZE) * 2LL)),
            batch_blocks{
                    std::make_shared<BatchBlock>(element_size_b, num_samples, user_buffer),
                    std::make_shared<BatchBlock>(element_size_b, num_samples, user_buffer + element_size_b * num_samples)
            } {

        ASSERT(batch_blocks.size() == NUM_BLOCKS, "%ld", batch_blocks.size());
    }

    BatchBlocks(BatchBlock const &other) = delete;

    BatchBlocks &operator=(BatchBlock const &) = delete;

    ~BatchBlocks() {
        allocator.deleter(user_buffer);
    }
};


BatchBlocks::Allocator create_default_allocator() {
    return {
            .allocator = [](size_t size) { return new byte[size]; },
            .deleter = [](byte *addr) { delete[] addr; }
    };
};

}
