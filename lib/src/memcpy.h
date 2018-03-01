#pragma once

#include "utils.h"

#include <cstring>
#include <immintrin.h>

namespace nvme_sampler {

// Uses AVX2 VMOVNTDQ (i.e. with non-temporal hint) to copy aligned memory.
inline void avx2nt_memcpy(void *__restrict dst, const void *__restrict src, size_t size) {
    //    assert(size % 32 == 0);
    //    assert((intptr_t(dst) & 31) == 0);
    //    assert((intptr_t(src) & 31) == 0);

    auto src_m256i = reinterpret_cast<const __m256i *>(src);
    auto *dst_m256i = reinterpret_cast<__m256i *>(dst);
    const int64 num_chunks = size / sizeof(__m256i);

    for (int64 chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const __m256i loaded = _mm256_stream_load_si256(src_m256i);
        _mm256_stream_si256(dst_m256i, loaded);
        src_m256i++;
        dst_m256i++;
    }
    _mm_sfence();
}

template<bool large_size_aligned32>
inline void smart_memcpy(void *__restrict dst, const void *__restrict src, size_t size) {
    ::memcpy(dst, src, size);
};

template<>
inline void smart_memcpy<true>(void *__restrict dst, const void *__restrict src, size_t size) {
    avx2nt_memcpy(dst, src, size);
}

}
