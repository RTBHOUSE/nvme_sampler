#pragma once

#include <cstdio>
#include <cassert>
#include <vector>
#include <iomanip>
#include <string>
#include <iostream>
#include <sstream>
#include <memory>

namespace {

struct var_printer {
    std::vector<std::string> names;
    size_t last_used = 0;
    std::stringstream buffer;

    inline var_printer(std::initializer_list<std::string> l) : names(l) {
        this->buffer << std::setprecision(10);
    }

    template<class T1, class ...Args1>
    inline var_printer &print(T1 a, Args1... vars) {
        std::string const &name = names[last_used++];
        this->buffer << name << ": " << a << (last_used == names.size() ? "." : ", ");
        print(vars...);
        return *this;
    }

    inline var_printer &print() {
        return *this;
    }
};

}

#define LOG(block) do { std::cout << "[NvmeSampler] " << block << std::endl; } while(0)

#define LOG_VARS7(block, a, b, c, d, e, f) LOG(block << "; " << (var_printer{ #a, #b, #c, #d, #e, #f}.print(a, b, c, d, e, f).buffer.str()))
#define LOG_VARS6(block, a, b, c, d, e) LOG(block << "; " << (var_printer{ #a, #b, #c, #d, #e}.print(a, b, c, d, e).buffer.str()))
#define LOG_VARS5(block, a, b, c, d) LOG(block << "; " << (var_printer{ #a, #b, #c, #d}.print(a, b, c, d).buffer.str()))
#define LOG_VARS4(block, a, b, c) LOG(block << "; " << (var_printer{ #a, #b, #c }.print(a, b, c).buffer.str()))
#define LOG_VARS3(block, a, b) LOG(block << "; " << (var_printer{ #a, #b }.print(a, b).buffer.str()))
#define LOG_VARS2(block, a) LOG(block << "; " << (var_printer{ #a }.print(a).buffer.str()))
#define MKFN(fn, ...) MKFN_N(fn,##__VA_ARGS__,9,8,7,6,5,4,3,2,1,0)(__VA_ARGS__)
#define MKFN_N(fn, n0, n1, n2, n3, n4, n5, n6, n7, n8, n, ...) fn##n
#define LOG_VARS(...) MKFN(LOG_VARS,##__VA_ARGS__)

#define DISABLED_ASSERT(cond, fmt, args...) do { (void)((cond))} while(0);

#define ENABLED_ASSERT(cond, fmt, ...) \
do { \
    if (!(cond)) { \
        fprintf(stderr, "ASSERTION FAILED. Message: " fmt "\n", ##__VA_ARGS__); \
        assert((cond)); \
    }\
    assert((cond)); /* for fast-math optimizations */ \
} while(false);

#define ERROR_ON(cond, fmt, args...) ENABLED_ASSERT(!(cond), fmt, args)
#define ERROR(message) do { \
        LOG(message);  \
        abort(); /* TODO */ \
        __builtin_unreachable(); \
    } while(false);

#define ASSERTION_LEVEL_ALL 1
#define ASSERTION_LEVEL_SOME 2
#define ASSERTION_LEVEL_ALMOST_NONE 3
#define ASSERTIONS_LEVEL ASSERTION_LEVEL_ALL

#if ASSERTIONS_LEVEL == ASSERTION_LEVEL_ALL
#define CASSERT ENABLED_ASSERT // critical assertion
#define ASSERT ENABLED_ASSERT  // normal assertion
#define DASSERT ENABLED_ASSERT // debug assertion
#elif ASSERTIONS_LEVEL == ASSERTION_LEVEL_SOME
#define CASSERT ENABLED_ASSERT
#define ASSERT ENABLED_ASSERT
#define DASSERT DISABLED_ASSERT
#elif ASSERTIONS_LEVEL == ASSERTION_LEVEL_ALMOST_NONE
#define CASSERT ENABLED_ASSERT
#define ASSERT DISABLED_ASSERT
#define DASSERT DISABLED_ASSERT
#else
#error "invalid ASSERTIONS_LEVEL"
#endif

#undef ASSERTION_LEVEL_ALL
#undef ASSERTION_LEVEL_SOME
#undef ASSERTION_LEVEL_ALMOST_NONE

#define CHECK_SYSCALL(cond, message) do { \
        bool ret = ((cond));  \
        if (!ret) {  \
            int err = errno; \
            LOG(message << ": " << #cond << "; errno: " << err);  \
            abort(); /* TODO */ \
        } \
    } while(false);

namespace nvme_sampler {

template<typename T>
using scoped_array = std::unique_ptr<T[]>;

typedef int64_t int64;
typedef uint64_t uint64;
typedef int32_t int32;

constexpr inline bool is_power_of_two(int64_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

constexpr inline int64_t align_up(int64_t value, int32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr inline int64_t align_down(int64_t value, int32_t alignment) {
    return (value / alignment) * alignment;
}

constexpr inline int32_t round_up_to_pow2(int32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
};

}
