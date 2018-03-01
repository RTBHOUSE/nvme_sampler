#include "memcpy.h"
#include <sys/time.h>
#include <numeric>

using namespace nvme_sampler;

double get_wall_time() {
    timeval tv;
    int32 result = gettimeofday(&tv, nullptr);
    assert(result == 0);
    return static_cast<double>(tv.tv_sec) + (tv.tv_usec / 1e6);
}

template<typename Fun>
void test_memcpy(Fun fun, int64 chunk_size, int64 mem_size, int64 num_iterations, const char *src, char *dst) {
    for (int32 run_idx = 0; run_idx < num_iterations; ++run_idx) {
        std::iota(reinterpret_cast<int32 *>(dst), reinterpret_cast<int32 *>(dst + mem_size), 12341);
        double start_time = get_wall_time();
        for (int64 block_idx = 0; block_idx < (mem_size / chunk_size); ++block_idx) {
            fun(dst + (block_idx * chunk_size), src + (block_idx * chunk_size), chunk_size);
        }
        double end_time = get_wall_time();
        double duration = end_time - start_time;

        assert(memcmp(src, dst, align_down(mem_size, chunk_size)) == 0);

        printf("Duration: %lf; bw=%lf GiB/s\n", duration, mem_size / duration / (1 << 30));
    }
}

int main(int argc, char **argv) {
    assert(argc == 6);
    int64 chunk_size = std::atol(argv[1]);
    int32 src_alignment = std::atoi(argv[2]);
    int32 dst_alignment = std::atoi(argv[3]);
    int64 mem_size = std::atol(argv[4]);
    int64 num_iterations = std::atol(argv[5]);

    assert(mem_size % 4 == 0);

    auto src = new char[align_up(mem_size + src_alignment, src_alignment)];
    auto dst = new char[align_up(mem_size + dst_alignment, dst_alignment)];
    src = (char *) align_up((long) src, src_alignment);
    dst = (char *) align_up((long) dst, dst_alignment);
    assert(((long) src) % src_alignment == 0);
    assert(((long) dst) % dst_alignment == 0);

    std::iota(reinterpret_cast<int32 *>(src), reinterpret_cast<int32 *>(src + mem_size), 0);
    std::iota(reinterpret_cast<int32 *>(dst), reinterpret_cast<int32 *>(dst + mem_size), 12341);

    LOG("memcpy() performance test");
    LOG_VARS("Params", mem_size, chunk_size, src_alignment, dst_alignment);

    LOG("memcpy");
    test_memcpy(memcpy, chunk_size, mem_size, num_iterations, src, dst);

    LOG("smart_memcpy");
    auto fun = (chunk_size >= 1024 && src_alignment % 32 == 0 && dst_alignment % 32 == 0 && chunk_size % 32 == 0) ? smart_memcpy<true>
                                                                                                                  : smart_memcpy<false>;
    test_memcpy(fun, chunk_size, mem_size, num_iterations, src, dst);

    LOG("avx2nt_memcpy");
    if (src_alignment % 32 == 0 && dst_alignment % 32 == 0) {
        test_memcpy(avx2nt_memcpy, chunk_size, mem_size, num_iterations, src, dst);
    }

    return 0;
}

