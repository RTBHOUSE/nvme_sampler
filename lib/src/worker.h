#pragma once

#include "utils.h"
#include "buffers.h"
#include "memcpy.h"
#include "blocking_queue.h"
#include "batch_block.h"
#include "lcg.h"

#include <libaio.h>
#include <fcntl.h>
#include <unistd.h>

namespace nvme_sampler {

using SamplingParametersCalculator::PAGE_SIZE;
using SamplingParametersCalculator::SECTOR_SIZE;

struct ReadBatchBlockTask {
    BatchBlockPtr block;
    const int64 num_sub_tasks;

private:
    std::mutex mutex;
    BlockingQueue<BatchBlockPtr> *result_queue;
    int32 num_sub_tasks_done{0};

public:
    ReadBatchBlockTask(BatchBlockPtr block, BlockingQueue<BatchBlockPtr> *result_queue, int64 num_sub_tasks)
            : block(block), num_sub_tasks(num_sub_tasks), result_queue(result_queue) {}

    void mark_sub_task_as_done() {
        std::lock_guard<std::mutex> lock(this->mutex);
        num_sub_tasks_done++;

        if (num_sub_tasks_done == num_sub_tasks) {
            block->read_idx = 0;
            result_queue->push(block);
        }
    }
};

enum TaskType {
    ReadBatchBlockTaskType
};

struct SubTask {
    TaskType type;

    virtual ~SubTask() {}

    SubTask(TaskType type) : type(type) {}
};

struct ReadBatchBlockSubTask : SubTask {
    ReadBatchBlockSubTask(std::shared_ptr<ReadBatchBlockTask> parent_task, int32 sub_task_id)
            : SubTask(ReadBatchBlockTaskType), parent_task(parent_task), sub_task_id(sub_task_id) {}

    std::shared_ptr<ReadBatchBlockTask> parent_task;
    int32 sub_task_id;
};


typedef std::shared_ptr<SubTask> SubTaskPtr;
typedef BlockingQueue<SubTaskPtr> WorkQueue;
typedef WorkQueue *WorkQueuePtr;

struct ReadDescription {
    int64 chunk_idx;
    int64 read_offset;
    int64 read_size;
    int64 data_offset;
    int64 num_elements;
    int64 target_column;

    struct Permutation {
        RawLCG::State state;
        int64 num_elements = 0;
    };

    Permutation permutations[2];
};

struct ChunkSampler {
    const int64 num_chunks;
    std::mt19937_64 rng;

    ChunkSampler(int64 num_chunks, int32 seed) : num_chunks(num_chunks), rng(seed) {}

    int64 next() {
        return this->rng() % num_chunks;
    }
};


class WorkerThread {
    static const int32 AIO_MAX_BATCH_SIZE = 2048;

    const int32 thread_idx;
    const TensorDescription tensor_description;
    const SamplerConfig sampler_config;
    const SamplingParameters sampling_params;

    WorkQueuePtr work_queue;

    const int32 file_descriptor;
    io_context_t io_ctx{nullptr};
    scoped_array<iocb *> io_requests{new iocb *[AIO_MAX_BATCH_SIZE]};
    scoped_array<io_event> io_events{new io_event[AIO_MAX_BATCH_SIZE]};
    scoped_array<byte> read_buffer{nullptr};
    scoped_array<ReadDescription> read_descriptions;

    LCGPermutationGenerator permutation_generator;
    ChunkSampler chunk_sampler;

public:
    WorkerThread(int32 thread_idx,
                 TensorDescription const &tensor_description,
                 SamplerConfig const &sampler_config,
                 SamplingParameters const &sampling_params,
                 int32 file_descriptor,
                 WorkQueuePtr work_queue)
            : thread_idx(thread_idx),
              tensor_description(tensor_description),
              sampler_config(sampler_config),
              sampling_params(sampling_params),
              work_queue(work_queue),
              file_descriptor(file_descriptor),
              read_descriptions(new ReadDescription[AIO_MAX_BATCH_SIZE]),
              permutation_generator(sampling_params.num_batches_in_block, thread_idx),
              chunk_sampler(sampling_params.num_chunks, thread_idx + sampler_config.seed) {
        {
            byte *tmp_buf;
            CHECK_SYSCALL(
                    ::posix_memalign((void **) &tmp_buf, PAGE_SIZE, sampling_params.max_chunk_size_b * AIO_MAX_BATCH_SIZE) == 0,
                    "posix_memalign failed"
            );
            read_buffer.reset(tmp_buf);
        }

        ASSERT(this->file_descriptor >= 0, "invalid file_descriptior: %d", this->file_descriptor);
        CHECK_SYSCALL(::io_setup(AIO_MAX_BATCH_SIZE, &io_ctx) == 0, "Failed to setup aio context");

        for (int j = 0; j < AIO_MAX_BATCH_SIZE; ++j) {
            io_requests[j] = new iocb;
        }
    }

    WorkerThread(const WorkerThread &other) = delete;

    WorkerThread &operator=(WorkerThread const &) = delete;

    WorkerThread(WorkerThread &&o) = default;

    ~WorkerThread() {
        CHECK_SYSCALL(io_destroy(this->io_ctx) == 0, "Failed to destroy aio context");

        for (int j = 0; j < AIO_MAX_BATCH_SIZE; ++j) {
            delete io_requests[j];
        }
    }

    void operator()() {
        for (;;) {
            SubTaskPtr sub_task;
            if (!work_queue->pop(sub_task)) {
                return; // close requested
            }

            if (sub_task->type == ReadBatchBlockTaskType) {
                auto &sub_task_downcast = *dynamic_cast<ReadBatchBlockSubTask *>(sub_task.get());

                if (this->tensor_description.row_size_b % 32 == 0 && this->tensor_description.row_size_b >= 1024) {
                    CASSERT((intptr_t(sub_task_downcast.parent_task->block->buffer.buffer) & 31) == 0, "Unaligned buffer");
                    read_block<true>(sub_task_downcast);
                } else {
                    read_block<false>(sub_task_downcast);
                }
            }
        }
    }

    template<bool use_alternative_memcpy>
    void read_block(ReadBatchBlockSubTask &sub_task) {
        int64 const element_size = this->tensor_description.row_size_b;
        int64 num_elements_to_read = sub_task.parent_task->block->num_samples / sub_task.parent_task->num_sub_tasks;

        RawLCG::State permutation(std::move(this->permutation_generator.start_new_permutation()));
        int64 num_elements_left_in_column = this->sampling_params.num_batches_in_block;
        int64 target_column = 0;

        while (num_elements_to_read > 0) {
            // prepare new requests
            int64 num_pending_requests = 0;

            for (int req_idx = 0; req_idx < AIO_MAX_BATCH_SIZE && num_elements_to_read > 0; ++req_idx) {
                this->read_descriptions[req_idx] = std::move(create_read_description(
                        element_size, num_elements_to_read, permutation, num_elements_left_in_column, target_column
                ));
                ::io_prep_pread(
                        this->io_requests[req_idx],
                        this->file_descriptor,
                        this->read_buffer.get() + req_idx * this->sampling_params.max_chunk_size_b,
                        static_cast<size_t>(this->read_descriptions[req_idx].read_size),
                        this->read_descriptions[req_idx].read_offset
                );
                io_requests[req_idx]->data = &this->read_descriptions[req_idx];
                num_pending_requests++;
            }

            // send them
            DASSERT(num_pending_requests > 0, "%ld", num_pending_requests);
            if (::io_submit(this->io_ctx, num_pending_requests, this->io_requests.get()) != num_pending_requests) {
                ERROR("io_submit() failed");
            }

            // wait for all of them
            while (num_pending_requests > 0) {
                timespec timeout{.tv_sec = 0, .tv_nsec =100000000};
                int32 num_events = ::io_getevents(
                        this->io_ctx,
                        std::min(10L, num_pending_requests),
                        std::min(num_pending_requests, 128L),
                        this->io_events.get(),
                        &timeout
                );
                CHECK_SYSCALL(num_events >= 0, "io_getevents() failed: num_events: " << num_events);

                for (int32 event_idx = 0; event_idx < num_events; ++event_idx) {
                    io_event &event = this->io_events[event_idx];
                    ReadDescription *read_description = reinterpret_cast<ReadDescription *>(event.data);

                    ERROR_ON(int64(event.res) != read_description->read_size,
                             "Incomplete read. Expected: %ld; got: %ld; offset=%ld (idx: %ld)",
                             read_description->read_size, event.res, read_description->read_offset, read_description->chunk_idx
                    );
                    this->handle_finished_read<use_alternative_memcpy>(sub_task, *read_description, static_cast<byte *>(event.obj->u.c.buf));
                }
                num_pending_requests -= num_events;
            }
        }

        sub_task.parent_task->mark_sub_task_as_done();
    }

private:
    ReadDescription create_read_description(const int64 element_size,
                                            int64 &num_elements_to_read,
                                            RawLCG::State &permutation,
                                            int64 &num_elements_left_in_column,
                                            int64 &target_column) {
        ASSERT(num_elements_to_read > 0, "%ld", num_elements_to_read);

        const int64 chunk_idx = this->chunk_sampler.next();
        int64 read_start = chunk_idx * sampling_params.chunk_size_b;
        int64 read_end = read_start + sampling_params.chunk_size_b;
        int64 data_size_b = read_end - read_start;

        if (read_start % element_size != 0) {
            auto reminder = read_start % element_size;
            auto skip = element_size - reminder;
            read_start += align_down(skip, SECTOR_SIZE);
            data_size_b -= skip;
        }

        if (read_end % element_size != 0) {
            auto reminder = read_end % element_size;
            auto add = element_size - reminder;
            read_end += align_up(add, SECTOR_SIZE);
            data_size_b += add;
        }

        const int64 read_size_b = read_end - read_start;
        const int64 data_offset = read_start % element_size == 0 ? 0 : element_size - read_start % element_size;
        const int64 num_chunk_elements = (data_size_b) / element_size;
        int64 num_perm_elements = std::min(num_elements_left_in_column, num_chunk_elements);
        num_elements_left_in_column -= num_perm_elements;

        ReadDescription read_description{
                .chunk_idx = chunk_idx,
                .read_offset = read_start,
                .read_size = read_size_b,
                .data_offset = data_offset,
                .num_elements = num_chunk_elements,
                .target_column = target_column,
                .permutations = {
                        {.state = permutation, .num_elements = num_perm_elements}
                }
        };

        if (num_elements_left_in_column == 0) { // column filled up - start a new permutation
            permutation = std::move(permutation_generator.start_new_permutation());
            num_elements_left_in_column = sampling_params.num_batches_in_block;
            num_perm_elements = num_chunk_elements - num_perm_elements;
            DASSERT(num_elements_left_in_column > num_perm_elements, "batch_size too small?");
            num_elements_left_in_column -= num_perm_elements;
            ++target_column;
            ASSERT(target_column <= this->sampler_config.max_batch_elements, "%ld", target_column);
            read_description.permutations[1] = {.state = permutation, .num_elements = num_perm_elements};
        }

        if (num_perm_elements > 0) {
            RawLCG::skip(permutation, num_perm_elements);
        }

        num_elements_to_read -= num_chunk_elements;

        return read_description;
    }

    template<bool use_alternative_memcpy>
    void handle_finished_read(ReadBatchBlockSubTask &sub_task, ReadDescription &read_description, byte const *read_data) {
        auto &permutation = read_description.permutations[0];
        int64 target_column = read_description.target_column;
        byte *const batch_block = sub_task.parent_task->block->buffer.buffer;
        int64 const batch_size_b = this->sampling_params.batch_size_b;
        int64 const sub_task_offset = batch_size_b / sub_task.parent_task->num_sub_tasks * sub_task.sub_task_id;
        int64 const element_size_b = this->tensor_description.row_size_b;

        DASSERT(read_description.data_offset >= 0, "%ld", read_description.data_offset);
        DASSERT(read_description.permutations[0].num_elements + read_description.permutations[1].num_elements == read_description.num_elements,
                "Invalid read description"
        );

        read_data += read_description.data_offset;

        for (int32 element_idx = 0; element_idx < read_description.num_elements; ++element_idx) {
            if (permutation.num_elements == 0) {
                permutation = read_description.permutations[1];
                ++target_column;
            }

            byte *dst = batch_block + sub_task_offset + target_column * element_size_b + permutation.state.element * batch_size_b;
            byte const *src = read_data + element_size_b * element_idx;
            smart_memcpy<use_alternative_memcpy>(dst, src, element_size_b);

            RawLCG::next(permutation.state);
            --permutation.num_elements;
        }
    }
};

}
