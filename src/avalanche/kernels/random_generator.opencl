/*
 * The MWC64X RNG is a small, fast, high-quality, skippable random uniform
 * number generator designed for use with GPUs via OpenCL.
 *
 * This code is based on code by David Thomas, dt10@imperial.ac.uk
 * published at http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
*/

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define WORK_GROUP_SIZE 64

// FIXME: Unify code style

enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };

// Pre: a<M, b<M
// Post: r=(a+b) mod M
ulong MWC_AddMod64(ulong a, ulong b, ulong M)
{
    ulong v=a+b;
    if( (v>=M) || (v<a) )
        v=v-M;
    return v;
}

// Pre: a<M,b<M
// Post: r=(a*b) mod M
// This could be done more efficently, but it is portable, and should
// be easy to understand. It can be replaced with any of the better
// modular multiplication algorithms (for example if you know you have
// double precision available or something).
ulong MWC_MulMod64(ulong a, ulong b, ulong M)
{
    ulong r=0;
    while(a!=0){
        if(a&1)
            r=MWC_AddMod64(r,b,M);
        b=MWC_AddMod64(b,b,M);
        a=a>>1;
    }
    return r;
}

// Pre: a<M, e>=0
// Post: r=(a^b) mod M
// This takes at most ~64^2 modular additions, so probably about 2^15 or so instructions on
// most architectures
ulong MWC_PowMod64(ulong a, ulong e, ulong M)
{
    ulong sqr=a, acc=1;
    while(e!=0){
        if(e&1)
            acc=MWC_MulMod64(acc,sqr,M);
        sqr=MWC_MulMod64(sqr,sqr,M);
        e=e>>1;
    }
    return acc;
}

uint2 MWC_SeedImpl_Mod64(ulong A, ulong M, uint vecSize, uint vecOffset, ulong streamBase, ulong streamGap)
{
    // This is an arbitrary constant for starting LCG jumping from. I didn't
    // want to start from 1, as then you end up with the two or three first values
    // being a bit poor in ones - once you've decided that, one constant is as
    // good as any another. There is no deep mathematical reason for it, I just
    // generated a random number.
    enum{ MWC_BASEID = 4077358422479273989UL };

    ulong dist=streamBase + (get_global_id(0)*vecSize+vecOffset)*streamGap;
    ulong m=MWC_PowMod64(A, dist, M);

    ulong x=MWC_MulMod64(MWC_BASEID, m, M);
    return (uint2)((uint)(x/A), (uint)(x%A));
}

void MWC64X_SeedStreams(uint2 *s, ulong baseOffset, ulong perStreamOffset)
{
    uint2 tmp=MWC_SeedImpl_Mod64(MWC64X_A, MWC64X_M, 1, 0, baseOffset, perStreamOffset);
    *s = tmp;
}

uint MWC64X(uint2 *state)
{
    uint x = (*state).x, c = (*state).y;  // Unpack the state
    uint res = x^c;                       // Calculate the result
    uint hi = mul_hi(x, MWC64X_A)  ;             // Step the RNG
    x = x * MWC64X_A + c;
    c = hi + (x < c);
    *state = (uint2)(x, c);              // Pack the state back up
    return res;                          // Return the next result
}

__kernel __attribute__((reqd_work_group_size(WORK_GROUP_SIZE, 1, 1)))
void seed_uniform_random(__global uint2 *states, const ulong size, const ulong base_offset) {
    ulong i = get_global_id(0);
    if (i < size) {
        uint2 s = states[i];
        MWC64X_SeedStreams(&s, base_offset, 1099511627776);
        states[i] = s;
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
        output[i] = min_value + (max_value - min_value) * number / 0x100000000UL; \
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
