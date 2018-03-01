#pragma once

#include <cstddef>
#include "utils.h"

namespace nvme_sampler {

struct TensorDescription {
    const int64_t num_rows;
    const int64_t row_size_b; // in bytes

    const std::string file_path;

    int64_t get_size() const {
        return num_rows * row_size_b;
    }
};

struct SamplerConfig {
    const int64_t max_batch_elements;
    const int64_t max_num_threads;
    const int64_t memory_usage_limit_b;
    const int32_t seed = 123; // for ChunkSamplers
};

struct SamplingParameters {
    const int64_t chunk_size_b;
    const int64_t max_chunk_size_b;
    const int64_t num_batches_in_block;
    const int64_t batch_size_b;
    const int64_t num_chunks;
};

namespace SamplingParametersCalculator {

static const int64_t NUM_BATCH_BLOCKS = 2;
static const int64_t PAGE_SIZE = 4096; // os page size
static const int64_t SECTOR_SIZE = 512; // os page size
static const int64_t MAX_CHUNK_SIZE = PAGE_SIZE * 16; // 16384 float features

static_assert(PAGE_SIZE % SECTOR_SIZE == 0, "Invalid PAGE_SIZE");
static_assert(is_power_of_two(SECTOR_SIZE), "Invalid SECTOR_SIZE");
static_assert(SECTOR_SIZE % 32 == 0, "Invalid SECTOR_SIZE (breaks AVX2 memcpy)");

SamplingParameters calculate(int64_t file_size_b, int64_t element_size_b, SamplerConfig const &config) {

    CASSERT(file_size_b % element_size_b == 0, "Invalid input parameters. file_size_b: %ld; element_size_b: %ld", file_size_b, element_size_b);
    CASSERT(element_size_b >= 16, "element_size_b is too small: %ld", element_size_b)
    CASSERT(element_size_b <= MAX_CHUNK_SIZE, "element_size_b is too big: %ld", element_size_b)
    CASSERT(config.max_num_threads <= 64, "max_num_threads is too small: %ld", config.max_num_threads)
    CASSERT(config.max_num_threads > 0, "max_num_threads is too big: %ld", config.max_num_threads)
    CASSERT(is_power_of_two(config.max_num_threads), "max_num_threads must be power of two: %ld", config.max_num_threads)
    CASSERT(config.max_batch_elements % config.max_num_threads == 0,
            "max_batch_elements (%ld) must be divisible by max_num_threads (%ld)", config.max_batch_elements, config.max_num_threads)

    const int64_t batch_size_b = element_size_b * config.max_batch_elements;

    CASSERT(batch_size_b <= file_size_b, "max_batch_elements (%ld) is too large for this file", config.max_batch_elements);
    CASSERT(batch_size_b * NUM_BATCH_BLOCKS <= config.memory_usage_limit_b,
            "max_batch_elements (%ld) is too large for this memory_usage_limit_b (%ld)", config.max_batch_elements, config.memory_usage_limit_b);

    // maximize num_batches_in_block, so that:
    // - memory_usage_limit_b is not exceeded
    // - wasted_reads_ratio < 5%
    const int64_t max_num_batches_in_block = std::min(1L << 15, config.memory_usage_limit_b / NUM_BATCH_BLOCKS / batch_size_b);
    for (int64_t num_batches_in_block = round_up_to_pow2(max_num_batches_in_block); num_batches_in_block >= 4; num_batches_in_block >>= 1) {
        for (int64_t chunk_size_b = PAGE_SIZE; chunk_size_b <= MAX_CHUNK_SIZE; chunk_size_b += PAGE_SIZE) {
            const int64_t used_memory_b = num_batches_in_block * batch_size_b * NUM_BATCH_BLOCKS;
            const int64_t reminder_b = (chunk_size_b % element_size_b == 0) ? 0 : element_size_b - (chunk_size_b % element_size_b);
            const int64_t additional_read_size_b = reminder_b == 0 ? 0 : align_up(reminder_b, SECTOR_SIZE);
            const int64_t total_read_size_b = additional_read_size_b + chunk_size_b;
            const int64_t wasted_b = additional_read_size_b - reminder_b; // TODO this is incorrect
            const double wasted_ratio = static_cast<double>(wasted_b) / total_read_size_b;
            const int64_t max_chunk_size_b = align_up(align_up(chunk_size_b, element_size_b) + SECTOR_SIZE * 2,
                                                      SECTOR_SIZE); // left and right padding are smaller than 512
            const int64_t num_chunks = file_size_b / chunk_size_b - 1;
            const int64_t max_num_elements_in_chunk = max_chunk_size_b / element_size_b;

            // TODO check for avg same read element distance
            if (chunk_size_b >= element_size_b && used_memory_b < config.memory_usage_limit_b && wasted_ratio <= 0.05 &&
                num_batches_in_block >= max_num_elements_in_chunk) {
                LOG_VARS("Sampling parameters", chunk_size_b, max_chunk_size_b, num_batches_in_block, num_chunks, wasted_ratio);
                if (num_chunks * chunk_size_b != file_size_b) {
                    int64_t num_ignored_elements = (file_size_b - (num_chunks - 1) * chunk_size_b) / element_size_b;
                    LOG("Last " << num_ignored_elements << " samples will never be sampled");
                }
                return SamplingParameters{
                        .chunk_size_b = chunk_size_b,
                        .max_chunk_size_b = max_chunk_size_b,
                        .num_batches_in_block = num_batches_in_block,
                        .batch_size_b = batch_size_b,
                        .num_chunks = num_chunks
                };
            }
        }
    }

    ERROR("Cannot find decent sampling parameters . Please increase memory_usage_limit_b");
}
}

}
