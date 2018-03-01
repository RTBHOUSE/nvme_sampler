#pragma once

#include <condition_variable>
#include <thread>
#include <atomic>
#include <queue>

namespace nvme_sampler {

template<typename T>
class BlockingQueue {
public:
    ~BlockingQueue() {
        invalidate();
    }

    bool pop(T &out) {
        std::unique_lock<std::mutex> lock(this->mutex);

        this->condition.wait(lock, [this]() {
            return !this->queue.empty() || !this->valid;
        });

        if (!this->valid) {
            return false;
        }

        out = std::move(this->queue.front());
        this->queue.pop();

        return true;
    }

    void push(const T &value) {
        std::lock_guard<std::mutex> lock(this->mutex);

        this->queue.push(std::move(value));
        this->condition.notify_one();
    }

    void invalidate() {
        std::lock_guard<std::mutex> lock(this->mutex);

        this->valid = false;
        this->condition.notify_all();
    }

private:
    std::atomic_bool valid{true};
    mutable std::mutex mutex;
    std::queue<T> queue;
    std::condition_variable condition;
};

}
