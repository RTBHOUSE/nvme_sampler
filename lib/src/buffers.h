#pragma once

#include <cstddef>

namespace nvme_sampler {

typedef unsigned char byte;

struct Buffer {
    const int64_t size;
    byte *const buffer;
};

template<typename T>
constexpr inline T *align_up_ptr(T *address, int32_t alignment) {
    uint64 value = reinterpret_cast<uint64>(address);
    return reinterpret_cast<byte *>((value + alignment - 1) / alignment * alignment);
}

}
