/**
 * This is implementation of MWC64X RNG, which is a small, fast, high-quality,
 * skippable random uniform number generator designed for use with GPUs via OpenCL.
 *
 * This code is based on a modified code created by David Thomas, dt10@imperial.ac.uk
 * published at http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
 * under BSD license (see below):
 *
 *     Copyright (c) 2011, David Thomas
 *     All rights reserved.
 *
 *     Redistribution and use in source and binary forms, with or without
 *     modification, are permitted provided that the following conditions are met:
 *
 *         * Redistributions of source code must retain the above copyright notice,
 *         this list of conditions and the following disclaimer.
 *         * Redistributions in binary form must reproduce the above copyright
 *         notice, this list of conditions and the following disclaimer in the
 *         documentation and/or other materials provided with the distribution.
 *         * Neither the name of Imperial College London nor the names of its
 *         contributors may be used to endorse or promote products derived
 *         from this software without specific prior written permission.
 *
 *     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *     CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *     INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *     MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *     CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *     SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *     LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *     OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 *     STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *     ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *     ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 64

#define MWC64X_A 4294883355U
#define MWC64X_M 18446383549859758079UL
#define ESTIMATED_SAMPLES_PER_STREAM 1099511627776UL
#define SAMPLE_UP_LIMIT 0x100000000UL  /* 2^32 */
#define MWC_BASEID 4077358422479273989UL

ulong mwc_add_mod_64(ulong a, ulong b, ulong M) {
    ulong v = a + b;
    if ((v >= M) || (v < a))
        v = v - M;
    return v;
}

ulong mwc_mul_mod_64(ulong a, ulong b, ulong M) {
    ulong r = 0;
    while (a != 0) {
        if (a & 1)
            r = mwc_add_mod_64(r, b, M);
        b = mwc_add_mod_64(b, b, M);
        a = a >> 1;
    }
    return r;
}

ulong mwc_pow_mod_64(ulong a, ulong e, ulong M) {
    ulong sqr = a, acc = 1;
    while (e != 0) {
        if (e & 1)
            acc = mwc_mul_mod_64(acc, sqr, M);
        sqr = mwc_mul_mod_64(sqr, sqr, M);
        e = e >> 1;
    }
    return acc;
}

uint2 mwc_seed_mod_64(ulong A, ulong M, uint vec_size, uint vec_offset,
                      ulong stream_base, ulong stream_gap) {
    ulong dist = stream_base + (get_global_id(0) * vec_size + vec_offset) * stream_gap;
    ulong m = mwc_pow_mod_64(A, dist, M);
    ulong x = mwc_mul_mod_64(MWC_BASEID, m, M);
    return (uint2)((uint)(x / A), (uint)(x % A));
}

uint MWC64X(uint2 *state) {
    uint x = (*state).x, c = (*state).y;
    uint res = x ^ c;
    uint hi = mul_hi(x, MWC64X_A);
    x = x * MWC64X_A + c;
    c = hi + (x < c);
    *state = (uint2)(x, c);
    return res;
}

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
void seed_uniform_random(__global uint2 *states,
                         const ulong size,
                         const ulong base_offset) {
    ulong i = get_global_id(0);
    if (i < size) {
        states[i] = mwc_seed_mod_64(
            MWC64X_A, MWC64X_M, 1, 0, base_offset, ESTIMATED_SAMPLES_PER_STREAM);
    }
}

#define uniform_random_kernel_template(DType, Type) \
__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1))) \
void generate_uniform_random_##DType ( \
        __global uint2 *states, \
        __global Type *output, \
        const ulong size, \
        const Type min_value, \
        const Type max_value) { \
    ulong i = get_global_id(0); \
    if (i < size) { \
        uint2 s = states[i]; \
        ulong number = MWC64X(&s); \
        output[i] = min_value + (max_value - min_value) * number / SAMPLE_UP_LIMIT; \
        states[i] = s; \
    } \
}

#ifdef HALF_MAX
uniform_random_kernel_template(float16, half)
#endif
uniform_random_kernel_template(float32, float)
uniform_random_kernel_template(float64, double)
uniform_random_kernel_template(int8, char)
uniform_random_kernel_template(int16, short)
uniform_random_kernel_template(int32, int)
uniform_random_kernel_template(int64, long)
