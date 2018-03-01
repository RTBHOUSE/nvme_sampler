#pragma  once

#include "utils.h"
#include "buffers.h"
#include "blocking_queue.h"
#include "batch_block.h"
#include "worker.h"
#include "calculator.h"

namespace nvme_sampler {

class NvmeSampler {

private:
    const TensorDescription tensor_description;
    const SamplerConfig sampler_config;
    const SamplingParameters sampling_params;
    const int32 file_descriptor;

    BatchBlocks batch_blocks;
    BatchBlock *current_block{NULL};

    std::vector<std::shared_ptr<WorkerThread>> workers;
    std::vector<std::thread> worker_threads;
    WorkQueue work_queue;

public:
    NvmeSampler(TensorDescription const &tensor_description, SamplerConfig const &sampler_config, BatchBlocks::Allocator allocator)
            : tensor_description(tensor_description),
              sampler_config(sampler_config),
              sampling_params(SamplingParametersCalculator::calculate(tensor_description.get_size(), tensor_description.row_size_b, sampler_config)),
              file_descriptor(::open(tensor_description.file_path.c_str(), O_DIRECT | O_RDONLY)),
              batch_blocks(tensor_description.row_size_b, sampling_params.num_batches_in_block * sampler_config.max_batch_elements, allocator) {

        CHECK_SYSCALL(this->file_descriptor >= 0,
                      "Failed to open file: " << tensor_description.file_path);

        CHECK_SYSCALL(::posix_fadvise(this->file_descriptor, 0, tensor_description.get_size(), POSIX_FADV_NOREUSE | POSIX_FADV_RANDOM) == 0,
                      "fadvise() failed");

        for (int thread_idx = 0; thread_idx < this->sampler_config.max_num_threads; ++thread_idx) {
            auto worker = std::make_shared<WorkerThread>(
                    thread_idx, tensor_description, sampler_config, sampling_params, file_descriptor, &work_queue
            );
            this->workers.emplace_back(worker);
            this->worker_threads.emplace_back([worker]() { (*worker)(); });
        }

        for (auto &block : this->batch_blocks.batch_blocks) {
            this->schedule_batch_block_reading(block.get());
        }
    }

    ~NvmeSampler() {
        work_queue.invalidate();
        for (auto &thread : this->worker_threads) {
            thread.join();
        }

        CHECK_SYSCALL(::close(this->file_descriptor) == 0, "Failed to close file a file descriptor");
    }

    byte *get_next_batch(int32 batch_size) {
        if (!current_block) {
            this->fetch_next_batch_block();
        }

        if (current_block->get_num_samples_left() > batch_size) {
            return current_block->read_next_batch(batch_size);
        }

        ASSERT(batch_size < current_block->num_samples, "batch size: %d, num_samples: %ld", batch_size, current_block->num_samples);

        this->schedule_batch_block_reading(current_block);
        current_block = nullptr;

        return this->get_next_batch(batch_size);
    }

private:
    void fetch_next_batch_block() {
        bool success = this->batch_blocks.ready_blocks.pop(current_block);
        ASSERT(success, "Reading from closed queue");
    }

    void schedule_batch_block_reading(BatchBlockPtr batch_block) {
        auto task = std::make_shared<ReadBatchBlockTask>(batch_block, &this->batch_blocks.ready_blocks, this->sampler_config.max_num_threads);
        for (int32 sub_task_id = 0; sub_task_id < this->sampler_config.max_num_threads; ++sub_task_id) {
            work_queue.push(std::make_shared<ReadBatchBlockSubTask>(task, sub_task_id));
        }
    }
};

}
