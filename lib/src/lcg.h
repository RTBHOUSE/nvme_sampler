#pragma  once

#include "utils.h"

#include <iostream>
#include <cassert>
#include <random>

namespace nvme_sampler {

// x % mod where mod is 2^k
template<bool mod_is_pow2>
inline int32_t mod_pow2(int64_t x, int64_t mod);

template<>
inline int32_t mod_pow2<true>(int64_t x, int64_t mod) {
    ASSERT(is_power_of_two(mod), "%ld", mod);
    ASSERT(x >= 0 && x <= (1LL << 62), "%ld", x);
    ASSERT(mod <= (1LL << 32), "%ld", mod);

    return x & (mod - 1);
}

template<>
inline int32_t mod_pow2<false>(int64_t x, int64_t mod) {
    ASSERT(x >= 0 && x <= (1LL << 62), "%ld", x);
    ASSERT(mod <= (1LL << 32), "%ld", mod);

    int64_t m = x % mod;
    ASSERT(m >= 0, "%ld", m);
    return m;
}

template<bool mod_is_pow2>
int32_t pow_mod(int64_t base, int32_t exp, int64_t mod) {
    ASSERT(base < mod && base >= 0, "base: %ld; mod: %ld", base, mod);
    ASSERT(exp <= mod, "%d", exp);

    int64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_pow2<mod_is_pow2>(result * base, mod);
        }
        base = mod_pow2<mod_is_pow2>(base * base, mod);
        exp >>= 1;
    }

    return static_cast<int32_t >(result);
}

struct RawLCG {
    struct State {
        int32_t a;
        int32_t c;
        int32_t m;
        int32_t element;

        void check() {
            ASSERT(is_power_of_two(this->m), "%d", this->m); // for fast modulo
            ASSERT(m >= 4, "%d", m); // to make code simpler
            ASSERT(a > 0 && c > 0, "a: %d, c: %d", a, c);
            ASSERT((a - 1) % 4 == 0, "%d", this->a); // make sure a - 1 is divisible by all prime factors of m (i.e. 2) and 4
            ASSERT(c % 2 == 1, "%d", this->c); // make sure c and m are coprime
            ASSERT(element >= 0 && element <= m, "%d", element);
            ASSERT(m <= (1 << 15), "%d", m); // avoid overflows in pow_mod()
        }
    };

    inline static void next(State &state) {
        state.check();
        state.element = lcg(state.element, state.a, state.c, state.m);
        state.check();
    }

    inline static void skip(State &state, int32_t num_steps) {
        state.check();
        state.element = lcg_skip(state.element, state.a, state.c, state.m, num_steps);
        state.check();
    }

private:
    static int32_t lcg_skip(int32_t element, int32_t a, int32_t c, int32_t m, int32_t num_steps) {
        ASSERT(num_steps > 0 && num_steps <= m, "%d", num_steps);

        if (a == 1) {
            return mod_pow2<true>(element + c * num_steps, m);
        } else {
            int64_t a1 = a - 1;
            int64_t tmp1 = pow_mod<true>(a, num_steps, m) * element;
            ASSERT(a1 > 0 && m > 0 && a1 <= (1 << 15) && m <= (1 << 15), "a1: %ld; m: %d", a1, m);
            int64_t tmp2 = (pow_mod<false>(a, num_steps, a1 * m) - 1) / a1 * c;
            return mod_pow2<true>(tmp1 + tmp2, m);
        }
    }

    static int32_t lcg(int32_t element, int32_t a, int32_t c, int32_t m) {
        return mod_pow2<true>(element * a + c, m);
    }
};

/**
 * Generates permutations using linear congruential generator.
 *
 * It's very fast, but generates only limited number of highly-correlated permutations.
 */
struct LCGPermutationGenerator {
    const uint64_t permutation_size;
    std::mt19937 rng;

    LCGPermutationGenerator(int32_t permutation_size, int32_t seed)
            : permutation_size(permutation_size), rng(seed) {

        ASSERT(is_power_of_two(permutation_size), "%d", permutation_size);
    }

    RawLCG::State start_new_permutation() {
        auto c = 2L * (this->rng() % (permutation_size / 2L - 1L)) + 1L;
        auto a = 4L * (this->rng() % (permutation_size / 4L)) + 1L;

        RawLCG::State state = {
                .a = static_cast<int32_t >(a),
                .c = static_cast<int32_t >(c),
                .m = static_cast<int32_t>(permutation_size),
                .element = static_cast<int32_t >(this->rng() % permutation_size)
        };
        state.check(); //TODO 
        return state;
    }
};

}
