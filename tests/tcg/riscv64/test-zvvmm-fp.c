/*
 * RISC-V Zvvmm Floating-Point Matrix Multiply-Accumulate Unit Test
 *
 * Validates vfmmacc.vv, vfwmmacc.vv, and vfqwmmacc.vv for all valid
 * (SEW, lambda, LMUL) combinations, FP format variants (f32, f64, f16,
 * bf16, e4m3, e5m2), mixed-format inputs, accumulation, and partial VL.
 *
 * Test values are small integers to ensure exact representability.
 * For FP8 same-width: use all-1.0 inputs so C[i,j] = K (exact in FP8).
 *
 * Register allocation: A=v0 (vs2), B=v8 (vs1), C=v16 (vd).
 *
 * Copyright (c) 2025 VRULL GmbH
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* --------------------------------------------------------------------
 * Instruction encodings
 * ------------------------------------------------------------------ */

/* FP MACC: vs2=v0, vs1=v8, vd=v16, vm=1, funct3=001 */
#define INSN_VFMMACC_VV   0x52041857u  /* funct6=010100 */
#define INSN_VFWMMACC_VV  0x56041857u  /* funct6=010101 */
#define INSN_VFQWMMACC_VV 0x5A041857u  /* funct6=010110 */
#define INSN_VF8WMMACC_VV 0x5E041857u  /* funct6=010111 */

/* Tile load/store (same as integer test) */
#define INSN_VMTL_V0   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  0u <<  7 | 0x07u)
#define INSN_VMTL_V8   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x07u)
#define INSN_VMTL_V16  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 16u <<  7 | 0x07u)
#define INSN_VMTS_V16  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 16u <<  7 | 0x27u)

/* vxor.vv v16, v16, v16 */
#define INSN_VXOR_V16  0x2F080857u

/* --------------------------------------------------------------------
 * Inline-asm helpers
 * ------------------------------------------------------------------ */

#define DO_TILE(insn, base_ptr, stride_val) do {                    \
    register void *_b asm("a0") = (void *)(uintptr_t)(base_ptr);   \
    register uint64_t _s asm("a1") = (uint64_t)(stride_val);       \
    asm volatile(".4byte %[i]"                                      \
                 : : [i] "i"(insn), "r"(_b), "r"(_s) : "memory");  \
} while (0)

#define DO_MACC(insn) \
    asm volatile(".4byte %[i]" : : [i] "i"(insn) : "memory")

#define DO_VXOR_V16() \
    asm volatile(".4byte %[i]" : : [i] "i"(INSN_VXOR_V16) : "memory")

static inline uint64_t set_macc_vtype(int sew_bytes, int lmul, int lambda,
                                      int altfmt, int altfmt_a, int altfmt_b)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul)
                   | ((uint64_t)__builtin_ctz(sew_bytes) << 3)
                   | ((uint64_t)altfmt << 8)
                   | ((uint64_t)altfmt_a << 59)
                   | ((uint64_t)altfmt_b << 58)
                   | ((uint64_t)(__builtin_ctz(lambda) + 1) << 60);
    uint64_t vl;
    asm volatile("vsetvl %0, zero, %1" : "=r"(vl) : "r"(vtype));
    return vl;
}

static inline uint64_t set_macc_vtype_vl(int sew_bytes, int lmul, int lambda,
                                         int altfmt, int altfmt_a, int altfmt_b,
                                         uint64_t avl)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul)
                   | ((uint64_t)__builtin_ctz(sew_bytes) << 3)
                   | ((uint64_t)altfmt << 8)
                   | ((uint64_t)altfmt_a << 59)
                   | ((uint64_t)altfmt_b << 58)
                   | ((uint64_t)(__builtin_ctz(lambda) + 1) << 60);
    uint64_t vl;
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vtype));
    return vl;
}

/* --------------------------------------------------------------------
 * Element accessors
 * ------------------------------------------------------------------ */

static uint64_t read_elem(const void *buf, int idx, int esz)
{
    switch (esz) {
    case 1: return ((const uint8_t  *)buf)[idx];
    case 2: return ((const uint16_t *)buf)[idx];
    case 4: return ((const uint32_t *)buf)[idx];
    case 8: return ((const uint64_t *)buf)[idx];
    }
    __builtin_unreachable();
}

static void write_elem(void *buf, int idx, int esz, uint64_t val)
{
    switch (esz) {
    case 1: ((uint8_t  *)buf)[idx] = (uint8_t)val;  break;
    case 2: ((uint16_t *)buf)[idx] = (uint16_t)val;  break;
    case 4: ((uint32_t *)buf)[idx] = (uint32_t)val;  break;
    case 8: ((uint64_t *)buf)[idx] = val;             break;
    }
}

/* --------------------------------------------------------------------
 * FP format conversion helpers (software, for reference computation)
 *
 * All conversions target small non-negative integers so we avoid
 * denormals, NaN, and rounding edge cases.
 * ------------------------------------------------------------------ */

/* IEEE f32 ↔ uint32_t bitcast */
static uint32_t f32_to_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

static float bits_to_f32(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* IEEE f64 ↔ uint64_t bitcast */
static uint64_t f64_to_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, 8);
    return u;
}

static double bits_to_f64(uint64_t u)
{
    double d;
    memcpy(&d, &u, 8);
    return d;
}

/* f16 (IEEE half) → float */
static float f16_to_float(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        /* denormal → normalize */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3ff;
        exp++;
        exp += 127 - 15;
    } else if (exp == 0x1f) {
        exp = 0xff;
    } else {
        exp += 127 - 15;
    }

    uint32_t bits = (sign << 31) | (exp << 23) | (mant << 13);
    return bits_to_f32(bits);
}

/* float → f16 (round to nearest even, for small integers: exact) */
static uint16_t float_to_f16(float f)
{
    uint32_t bits = f32_to_bits(f);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;

    if (exp <= 0) {
        return (uint16_t)(sign << 15);
    }
    if (exp >= 0x1f) {
        return (uint16_t)((sign << 15) | (0x1f << 10));
    }

    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}

/* bf16 → float: just shift upper 16 bits */
static float bf16_to_float(uint16_t b)
{
    uint32_t bits = (uint32_t)b << 16;
    return bits_to_f32(bits);
}

/* float → bf16: truncate lower 16 bits (exact for small integers) */
static uint16_t float_to_bf16(float f)
{
    uint32_t bits = f32_to_bits(f);
    return (uint16_t)(bits >> 16);
}

/* FP8 E4M3: sign(1) exp(4) mantissa(3), bias=7 */
static float e4m3_to_float(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 3) & 0xf;
    uint32_t mant = v & 0x7;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        /* denormal: value = (-1)^s * 2^(1-bias) * 0.mant */
        float val = (float)mant / 8.0f;
        val *= 1.0f / 64.0f;  /* 2^(1-7) = 2^-6 */
        return sign ? -val : val;
    }
    if (exp == 0xf && mant == 0x7) {
        /* NaN */
        return bits_to_f32(0x7fc00000u);
    }

    /* Normal: (-1)^s * 2^(exp-bias) * 1.mant */
    uint32_t f32_exp = exp - 7 + 127;
    uint32_t bits = (sign << 31) | (f32_exp << 23) | (mant << 20);
    return bits_to_f32(bits);
}

static uint8_t float_to_e4m3(float f)
{
    uint32_t bits = f32_to_bits(f);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 7;
    uint32_t mant = (bits >> 20) & 0x7;

    if (exp <= 0) {
        return (uint8_t)(sign << 7);
    }
    if (exp >= 0xf) {
        return (uint8_t)((sign << 7) | (0xe << 3) | 0x7); /* max normal */
    }

    return (uint8_t)((sign << 7) | (exp << 3) | mant);
}

/* FP8 E5M2: sign(1) exp(5) mantissa(2), bias=15 */
static float e5m2_to_float(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 2) & 0x1f;
    uint32_t mant = v & 0x3;

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        float val = (float)mant / 4.0f;
        val *= 1.0f / 16384.0f; /* 2^(1-15) */
        return sign ? -val : val;
    }
    if (exp == 0x1f) {
        return bits_to_f32(0x7fc00000u); /* NaN/Inf → NaN */
    }

    uint32_t f32_exp = exp - 15 + 127;
    uint32_t f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 21);
    return bits_to_f32(f32_bits);
}

static uint8_t float_to_e5m2(float f)
{
    uint32_t bits = f32_to_bits(f);
    uint32_t sign = (bits >> 31) & 1;
    int32_t  exp  = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 21) & 0x3;

    if (exp <= 0) {
        return (uint8_t)(sign << 7);
    }
    if (exp >= 0x1f) {
        return (uint8_t)((sign << 7) | (0x1e << 2) | 0x3); /* max normal */
    }

    return (uint8_t)((sign << 7) | (exp << 2) | mant);
}

/*
 * Convert an integer value to the raw FP encoding for the given format.
 * sew: element size in bytes (1=fp8, 2=f16/bf16, 4=f32, 8=f64)
 * use_alt: 1=alternate format (bf16 for sew=2, e5m2 for sew=1)
 */
static uint64_t int_to_fp_raw(int ival, int sew, int use_alt)
{
    float f = (float)ival;
    double d = (double)ival;

    switch (sew) {
    case 1:
        return use_alt ? float_to_e5m2(f) : float_to_e4m3(f);
    case 2:
        return use_alt ? float_to_bf16(f) : float_to_f16(f);
    case 4:
        return f32_to_bits(f);
    case 8:
        return f64_to_bits(d);
    }
    return 0;
}

/*
 * Convert raw FP encoding to double for reference computation.
 */
static double fp_raw_to_double(uint64_t raw, int sew, int use_alt)
{
    switch (sew) {
    case 1:
        return use_alt ? (double)e5m2_to_float((uint8_t)raw)
                       : (double)e4m3_to_float((uint8_t)raw);
    case 2:
        return use_alt ? (double)bf16_to_float((uint16_t)raw)
                       : (double)f16_to_float((uint16_t)raw);
    case 4:
        return (double)bits_to_f32((uint32_t)raw);
    case 8:
        return bits_to_f64(raw);
    }
    return 0.0;
}

/*
 * Convert a double result back to raw FP encoding for comparison.
 */
static uint64_t double_to_fp_raw(double d, int sew, int use_alt)
{
    switch (sew) {
    case 1:
        return use_alt ? float_to_e5m2((float)d) : float_to_e4m3((float)d);
    case 2:
        return use_alt ? float_to_bf16((float)d) : float_to_f16((float)d);
    case 4:
        return f32_to_bits((float)d);
    case 8:
        return f64_to_bits(d);
    }
    return 0;
}

/* ====================================================================
 * Test: same-width vfmmacc.vv
 *
 * For f32/f64: use small integers (1..M*K), exact in double.
 * For f16/bf16: use values 1..4 to stay within mantissa range.
 * For fp8: use all-1.0 so C[i,j] = K (power of 2 × LMUL, exact).
 * ==================================================================== */

static int test_vfmmacc(int sew, int lmul, int lambda, uint64_t vlenb,
                         int altfmt, int altfmt_a, int altfmt_b,
                         int accumulate, int partial_vl,
                         const char *label)
{
    int M = vlenb / (sew * lambda);
    int N = M;
    int K = lambda * lmul;
    int linesize = lambda * lmul;
    int mul_c = vlenb / (sew * lambda * lambda);

    if (M < 1 || mul_c != lmul) {
        return 0;
    }

    int N_active = partial_vl ? 1 : N;
    if (partial_vl && N <= 1) {
        return 0;
    }

    int a_ld = linesize;
    int b_ld = linesize;
    int c_ld = linesize;
    int c_nelem = M * N;
    int use_ones = (sew == 1);  /* FP8: use all-1.0 */
    int max_val = (sew <= 2) ? 4 : 0; /* cap for f16/bf16 */

    uint8_t *A_buf   = calloc(M * a_ld, sew);
    uint8_t *B_buf   = calloc(N * b_ld, sew);
    uint8_t *C_init  = calloc(c_nelem, sew);
    uint8_t *C_dst   = calloc(c_nelem, sew);

    /* Fill A (row-major) */
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int ival;
            if (use_ones) {
                ival = 1;
            } else if (max_val) {
                ival = ((row * K + k) % max_val) + 1;
            } else {
                ival = row * K + k + 1;
            }
            uint64_t raw = int_to_fp_raw(ival, sew, altfmt_a);
            write_elem(A_buf, row * a_ld + k, sew, raw);
        }
    }

    /* Fill B (column-major) */
    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int ival;
            if (use_ones) {
                ival = 1;
            } else if (max_val) {
                ival = ((k * N + col) % max_val) + 1;
            } else {
                ival = k * N + col + 1;
            }
            uint64_t raw = int_to_fp_raw(ival, sew, altfmt_b);
            write_elem(B_buf, col * b_ld + k, sew, raw);
        }
    }

    /* Fill C_init for accumulation */
    if (accumulate) {
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                int flat = row * M + col;
                int ival = (sew == 1) ? 0 : (row + col + 1);
                uint64_t raw = int_to_fp_raw(ival, sew, altfmt);
                write_elem(C_init, flat, sew, raw);
            }
        }
    }

    /* Execute */
    set_macc_vtype(sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);

    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    if (accumulate) {
        DO_TILE(INSN_VMTL_V16, C_init, (uint64_t)c_ld);
    } else {
        DO_VXOR_V16();
    }

    if (partial_vl) {
        set_macc_vtype_vl(sew, lmul, lambda, altfmt, altfmt_a, altfmt_b,
                          (uint64_t)K);
    }

    DO_MACC(INSN_VFMMACC_VV);

    if (partial_vl) {
        set_macc_vtype(sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    }
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    /* Compute reference in double */
    int fail = 0;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            double acc = 0.0;
            if (accumulate) {
                int flat = row * M + col;
                uint64_t raw = read_elem(C_init, flat, sew);
                acc = fp_raw_to_double(raw, sew, altfmt);
            }

            for (int k = 0; k < K; k++) {
                uint64_t a_raw = read_elem(A_buf, row * a_ld + k, sew);
                uint64_t b_raw = read_elem(B_buf, col * b_ld + k, sew);
                double a_val = fp_raw_to_double(a_raw, sew, altfmt_a);
                double b_val = fp_raw_to_double(b_raw, sew, altfmt_b);
                acc += a_val * b_val;
                /*
                 * For FP8: round accumulator to target precision after each
                 * FMA step, matching the hardware behavior where each
                 * c_elem = fma(a, b, c_elem) is rounded to the accum format.
                 */
                if (sew == 1) {
                    uint64_t r = double_to_fp_raw(acc, sew, altfmt);
                    acc = fp_raw_to_double(r, sew, altfmt);
                }
            }

            int flat = row * M + col;
            uint64_t expected = double_to_fp_raw(acc, sew, altfmt);
            uint64_t got = read_elem(C_dst, flat, sew);
            if (got != expected) {
                printf("FAIL %s e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_init); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 2x widening vfwmmacc.vv
 * ==================================================================== */

static int test_vfwmmacc(int accum_sew, int lmul, int lambda, uint64_t vlenb,
                          int altfmt, int altfmt_a, int altfmt_b,
                          const char *label)
{
    int input_sew = accum_sew / 2;
    int W = 2;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int K = lambda * W * lmul;
    int linesize_load = lambda * lmul;
    int mu_load = vlenb / (input_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_c = lambda * lmul;

    if (M < W || mul_c != lmul) {
        return 0;
    }

    int N_active = M;
    int a_ld = linesize_load;
    int b_ld = linesize_load;
    int c_ld = linesize_c;
    int c_nelem = M * N;
    int use_ones = (input_sew == 1);
    int max_val = (input_sew <= 2) ? 4 : 0;

    uint8_t *A_buf = calloc(mu_load * a_ld, input_sew);
    uint8_t *B_buf = calloc(mu_load * b_ld, input_sew);
    uint8_t *C_dst = calloc(c_nelem, accum_sew);

    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int tr = W * row + k / linesize_load;
            int tc = k % linesize_load;
            int ival;
            if (use_ones) {
                ival = 1;
            } else if (max_val) {
                ival = ((row * K + k) % max_val) + 1;
            } else {
                ival = row * K + k + 1;
            }
            uint64_t raw = int_to_fp_raw(ival, input_sew, altfmt_a);
            write_elem(A_buf, tr * a_ld + tc, input_sew, raw);
        }
    }

    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int tr = W * col + k / linesize_load;
            int tc = k % linesize_load;
            int ival;
            if (use_ones) {
                ival = 1;
            } else if (max_val) {
                ival = ((k * N + col) % max_val) + 1;
            } else {
                ival = k * N + col + 1;
            }
            uint64_t raw = int_to_fp_raw(ival, input_sew, altfmt_b);
            write_elem(B_buf, tr * b_ld + tc, input_sew, raw);
        }
    }

    /* Load at input_sew */
    set_macc_vtype(input_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    /* Reconfigure to accum_sew */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_VFWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                int tr_a = W * row + k / linesize_load;
                int tc_a = k % linesize_load;
                int tr_b = W * col + k / linesize_load;
                int tc_b = k % linesize_load;

                uint64_t a_raw = read_elem(A_buf,
                            tr_a * a_ld + tc_a, input_sew);
                uint64_t b_raw = read_elem(B_buf,
                            tr_b * b_ld + tc_b, input_sew);
                double a_val = fp_raw_to_double(a_raw, input_sew, altfmt_a);
                double b_val = fp_raw_to_double(b_raw, input_sew, altfmt_b);
                acc += a_val * b_val;
            }

            int flat = row * M + col;
            uint64_t expected = double_to_fp_raw(acc, accum_sew, altfmt);
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, input_sew * 8, accum_sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 4x widening vfqwmmacc.vv
 * ==================================================================== */

static int test_vfqwmmacc(int accum_sew, int lmul, int lambda, uint64_t vlenb,
                           int altfmt, int altfmt_a, int altfmt_b,
                           const char *label)
{
    int input_sew = accum_sew / 4;
    int W = 4;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int K = lambda * W * lmul;
    int linesize_load = lambda * lmul;
    int mu_load = vlenb / (input_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_c = lambda * lmul;

    if (M < W || mul_c != lmul) {
        return 0;
    }

    int N_active = M;
    int a_ld = linesize_load;
    int b_ld = linesize_load;
    int c_ld = linesize_c;
    int c_nelem = M * N;
    int use_ones = (input_sew == 1);
    int max_val = (input_sew <= 2) ? 4 : 0;

    uint8_t *A_buf = calloc(mu_load * a_ld, input_sew);
    uint8_t *B_buf = calloc(mu_load * b_ld, input_sew);
    uint8_t *C_dst = calloc(c_nelem, accum_sew);

    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int tr = W * row + k / linesize_load;
            int tc = k % linesize_load;
            int ival;
            if (use_ones) {
                ival = 1;
            } else if (max_val) {
                ival = ((row * K + k) % max_val) + 1;
            } else {
                ival = row * K + k + 1;
            }
            uint64_t raw = int_to_fp_raw(ival, input_sew, altfmt_a);
            write_elem(A_buf, tr * a_ld + tc, input_sew, raw);
        }
    }

    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int tr = W * col + k / linesize_load;
            int tc = k % linesize_load;
            int ival;
            if (use_ones) {
                ival = 1;
            } else if (max_val) {
                ival = ((k * N + col) % max_val) + 1;
            } else {
                ival = k * N + col + 1;
            }
            uint64_t raw = int_to_fp_raw(ival, input_sew, altfmt_b);
            write_elem(B_buf, tr * b_ld + tc, input_sew, raw);
        }
    }

    set_macc_vtype(input_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_VFQWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                int tr_a = W * row + k / linesize_load;
                int tc_a = k % linesize_load;
                int tr_b = W * col + k / linesize_load;
                int tc_b = k % linesize_load;

                uint64_t a_raw = read_elem(A_buf,
                            tr_a * a_ld + tc_a, input_sew);
                uint64_t b_raw = read_elem(B_buf,
                            tr_b * b_ld + tc_b, input_sew);
                double a_val = fp_raw_to_double(a_raw, input_sew, altfmt_a);
                double b_val = fp_raw_to_double(b_raw, input_sew, altfmt_b);
                acc += a_val * b_val;
            }

            int flat = row * M + col;
            uint64_t expected = double_to_fp_raw(acc, accum_sew, altfmt);
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, input_sew * 8, accum_sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 2x widening vfwmmacc.vv with FP4 inputs (FP4 → FP8)
 *
 * FP4 values are packed as nibbles (2 per byte). We use all-1.0 inputs
 * so the exact nibble-to-register mapping is irrelevant (uniform fill).
 *
 * FP4 format: e2m1 (altfmt_a/b=0) only; e3m0 (altfmt_a/b=1) is reserved.
 * Accumulator: e4m3 (altfmt=0), e5m2 (altfmt=1)
 * ==================================================================== */

static int test_vfwmmacc_fp4(int lmul, int lambda, uint64_t vlenb,
                              int altfmt, int altfmt_a, int altfmt_b,
                              const char *label)
{
    int accum_sew = 1;  /* e8 accumulator */
    int M = vlenb / lambda;
    int mul_c = vlenb / (lambda * lambda);
    int k_eff = lambda * lmul * 2;
    int linesize = lambda * lmul;

    if (M < 1 || mul_c != lmul) {
        return 0;
    }

    int N = vlenb / (2 * lambda);
    if (N < 1) {
        return 0;
    }

    int c_nelem = M * linesize;

    /* FP4 1.0 nibble: e2m1=0x2, e3m0=0x3; pack both nibbles the same */
    uint8_t nib_a = altfmt_a ? 0x3 : 0x2;
    uint8_t nib_b = altfmt_b ? 0x3 : 0x2;
    uint8_t byte_a = (nib_a << 4) | nib_a;
    uint8_t byte_b = (nib_b << 4) | nib_b;

    uint8_t *A_buf = calloc(M * linesize, 1);
    uint8_t *B_buf = calloc(M * linesize, 1);
    uint8_t *C_dst = calloc(c_nelem, accum_sew);

    memset(A_buf, byte_a, M * linesize);
    memset(B_buf, byte_b, M * linesize);

    /* Both load and MACC use sew=e8 for FP4→FP8 */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)linesize);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)linesize);

    DO_VXOR_V16();
    DO_MACC(INSN_VFWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)linesize);

    /* Reference: accumulate 1.0×1.0 with per-step FP8 rounding */
    double ref_acc = 0.0;
    for (int k = 0; k < k_eff; k++) {
        ref_acc += 1.0;
        uint64_t r = double_to_fp_raw(ref_acc, accum_sew, altfmt);
        ref_acc = fp_raw_to_double(r, accum_sew, altfmt);
    }
    uint64_t expected = double_to_fp_raw(ref_acc, accum_sew, altfmt);

    int fail = 0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s fp4->e8 m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected, ref_acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 4x widening vfqwmmacc.vv with FP4 inputs (FP4 → f16/bf16)
 *
 * Same uniform-fill strategy as vfwmmacc FP4 tests.
 * Load packed nibble bytes at sew=e8, then reconfigure to sew=e16
 * for the MACC and C tile operations.
 * ==================================================================== */

static int test_vfqwmmacc_fp4(int lmul, int lambda, uint64_t vlenb,
                               int altfmt, int altfmt_a, int altfmt_b,
                               const char *label)
{
    int accum_sew = 2;  /* e16 accumulator (f16 or bf16) */
    int load_sew = 1;   /* byte-level load for packed nibbles */
    int M = vlenb / (accum_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int k_eff = lambda * lmul * 4;  /* quad-widening: 4-bit → 16-bit */
    int linesize_load = lambda * lmul;
    int linesize_c = lambda * lmul;

    if (M < 1 || mul_c != lmul) {
        return 0;
    }

    int N = M / 4;  /* quad-widening reduces columns by 4 */
    if (N < 1) {
        return 0;
    }

    int mu_load = vlenb / lambda;  /* rows for byte-level load */
    int c_nelem = M * linesize_c;

    uint8_t nib_a = altfmt_a ? 0x3 : 0x2;
    uint8_t nib_b = altfmt_b ? 0x3 : 0x2;
    uint8_t byte_a = (nib_a << 4) | nib_a;
    uint8_t byte_b = (nib_b << 4) | nib_b;

    uint8_t *A_buf = calloc(mu_load * linesize_load, 1);
    uint8_t *B_buf = calloc(mu_load * linesize_load, 1);
    uint8_t *C_dst = calloc(c_nelem, accum_sew);

    memset(A_buf, byte_a, mu_load * linesize_load);
    memset(B_buf, byte_b, mu_load * linesize_load);

    /* Load packed nibble bytes at sew=e8 */
    set_macc_vtype(load_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)linesize_load);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)linesize_load);

    /* Reconfigure to accum sew=e16 */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_VFQWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)linesize_c);

    /* Reference: k_eff products of 1.0×1.0 accumulated in f16/bf16 */
    double ref_acc = 0.0;
    for (int k = 0; k < k_eff; k++) {
        ref_acc += 1.0;
        uint64_t r = double_to_fp_raw(ref_acc, accum_sew, altfmt);
        ref_acc = fp_raw_to_double(r, accum_sew, altfmt);
    }
    uint64_t expected = double_to_fp_raw(ref_acc, accum_sew, altfmt);

    int fail = 0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s fp4->e16 m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected, ref_acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 8x widening vf8wmmacc.vv — byte-aligned inputs (OFP8 → FP64)
 *
 * Follows the same structure as test_vfqwmmacc but with W=8.
 * Only SEW=64 with OFP8 inputs (altfmt_a/b select E4M3/E5M2).
 * Uses all-1.0 inputs (exact in both E4M3 and E5M2).
 * ==================================================================== */

static int test_vf8wmmacc(int accum_sew, int lmul, int lambda, uint64_t vlenb,
                           int altfmt, int altfmt_a, int altfmt_b,
                           const char *label)
{
    int input_sew = accum_sew / 8;
    int W = 8;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int K = lambda * W * lmul;
    int linesize_load = lambda * lmul;
    int mu_load = vlenb / (input_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_c = lambda * lmul;

    if (M < W || mul_c != lmul || input_sew < 1) {
        return 0;
    }

    int N_active = M;
    int a_ld = linesize_load;
    int b_ld = linesize_load;
    int c_ld = linesize_c;
    int c_nelem = M * N;

    /* FP8 inputs: use all 1.0 so result is exact (C[i,j] = K) */
    uint8_t *A_buf = calloc(mu_load * a_ld, input_sew);
    uint8_t *B_buf = calloc(mu_load * b_ld, input_sew);
    uint8_t *C_dst = calloc(c_nelem, accum_sew);

    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int tr = W * row + k / linesize_load;
            int tc = k % linesize_load;
            uint64_t raw = int_to_fp_raw(1, input_sew, altfmt_a);
            write_elem(A_buf, tr * a_ld + tc, input_sew, raw);
        }
    }

    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int tr = W * col + k / linesize_load;
            int tc = k % linesize_load;
            uint64_t raw = int_to_fp_raw(1, input_sew, altfmt_b);
            write_elem(B_buf, tr * b_ld + tc, input_sew, raw);
        }
    }

    set_macc_vtype(input_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_VF8WMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;
    double ref_acc = (double)K;  /* K × 1.0 × 1.0 */
    uint64_t expected = double_to_fp_raw(ref_acc, accum_sew, altfmt);

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, input_sew * 8, accum_sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, ref_acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 8x widening vf8wmmacc.vv with FP4 inputs (E2M1 → FP32)
 *
 * FP4 E2M1 packed as nibbles. All-1.0 inputs.
 * accum_sew=4 (FP32), altfmt=0 (FP32 output).
 * ==================================================================== */

static int test_vf8wmmacc_fp4(int lmul, int lambda, uint64_t vlenb,
                               int altfmt, const char *label)
{
    int accum_sew = 4;   /* e32 = FP32 */
    int W = 8;
    int k_eff = lambda * W * lmul;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_load = lambda * lmul;

    if (M < W || mul_c != lmul) {
        return 0;
    }

    int load_sew = 1;  /* load at e8, nibbles packed within bytes */
    int mu_load = vlenb / (load_sew * lambda);
    int byte_count = mu_load * linesize_load;

    uint8_t *A_buf = calloc(byte_count, 1);
    uint8_t *B_buf = calloc(byte_count, 1);
    uint8_t *C_dst = calloc(M * N, accum_sew);

    /* FP4 E2M1: 1.0 = 0x2 (sign=0, exp=01, man=0 → 2^(1-1) × 1.0 = 1.0).
     * Two 1.0 nibbles packed per byte: (0x2 << 4) | 0x2 = 0x22 */
    memset(A_buf, 0x22, byte_count);
    memset(B_buf, 0x22, byte_count);

    set_macc_vtype(load_sew, lmul, lambda, altfmt, 0, 0);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)linesize_load);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)linesize_load);

    set_macc_vtype(accum_sew, lmul, lambda, altfmt, 0, 0);
    DO_VXOR_V16();
    DO_MACC(INSN_VF8WMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)(lambda * lmul));

    int fail = 0;
    double ref_acc = (double)k_eff;  /* k_eff × 1.0 × 1.0 */
    uint64_t expected = double_to_fp_raw(ref_acc, accum_sew, altfmt);

    int N_active = M;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s fp4->f32 m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected, ref_acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void)
{
    uint64_t vlenb;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));

    static const int lmuls[]   = {1, 2, 4, 8};
    static const int lambdas[] = {1, 2, 4, 8};

    int fail = 0;
    int tests = 0;

    for (int li = 0; li < 4; li++) {
        for (int ai = 0; ai < 4; ai++) {
            int lmul   = lmuls[li];
            int lambda = lambdas[ai];

            /* TC-FP-01: vfmmacc f32, C=0 */
            fail |= test_vfmmacc(4, lmul, lambda, vlenb,
                                  0, 0, 0, 0, 0, "vfmmacc f32 C=0");
            tests++;

            /* TC-FP-02: vfmmacc f32, accumulation */
            fail |= test_vfmmacc(4, lmul, lambda, vlenb,
                                  0, 0, 0, 1, 0, "vfmmacc f32 accum");
            tests++;

            /* TC-FP-03: vfmmacc f64 */
            fail |= test_vfmmacc(8, lmul, lambda, vlenb,
                                  0, 0, 0, 0, 0, "vfmmacc f64");
            tests++;

            /* TC-FP-04: vfmmacc f16 */
            fail |= test_vfmmacc(2, lmul, lambda, vlenb,
                                  0, 0, 0, 0, 0, "vfmmacc f16");
            tests++;

            /* TC-FP-05: vfmmacc bf16 (altfmt=1, a=1, b=1) */
            fail |= test_vfmmacc(2, lmul, lambda, vlenb,
                                  1, 1, 1, 0, 0, "vfmmacc bf16");
            tests++;

            /* TC-FP-06: vfmmacc e4m3 (sew=1, altfmt=0) */
            fail |= test_vfmmacc(1, lmul, lambda, vlenb,
                                  0, 0, 0, 0, 0, "vfmmacc e4m3");
            tests++;

            /* TC-FP-07: vfmmacc e5m2 (sew=1, altfmt=1, a=1, b=1) */
            fail |= test_vfmmacc(1, lmul, lambda, vlenb,
                                  1, 1, 1, 0, 0, "vfmmacc e5m2");
            tests++;

            /* TC-FP-08: vfmmacc e4m3 A × e5m2 B → e4m3 C */
            fail |= test_vfmmacc(1, lmul, lambda, vlenb,
                                  0, 0, 1, 0, 0, "vfmmacc e4m3xe5m2");
            tests++;

            /* TC-FP-09: vfmmacc e5m2 A × e4m3 B → e5m2 C */
            fail |= test_vfmmacc(1, lmul, lambda, vlenb,
                                  1, 1, 0, 0, 0, "vfmmacc e5m2xe4m3");
            tests++;

            /* TC-FP-10: vfmmacc f16 × bf16 → f16 */
            fail |= test_vfmmacc(2, lmul, lambda, vlenb,
                                  0, 0, 1, 0, 0, "vfmmacc f16xbf16");
            tests++;

            /* TC-FP-11: vfmmacc bf16 × f16 → bf16 */
            fail |= test_vfmmacc(2, lmul, lambda, vlenb,
                                  1, 1, 0, 0, 0, "vfmmacc bf16xf16");
            tests++;

            /* TC-FP-12: vfmmacc f32, partial VL */
            fail |= test_vfmmacc(4, lmul, lambda, vlenb,
                                  0, 0, 0, 0, 1, "vfmmacc f32 partial");
            tests++;

            /* TC-FP-13: vfwmmacc e4m3 → f16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   0, 0, 0, "vfwmmacc e4m3->f16");
            tests++;

            /* TC-FP-14: vfwmmacc e5m2 → bf16 (altfmt=1, a=1, b=1) */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   1, 1, 1, "vfwmmacc e5m2->bf16");
            tests++;

            /* TC-FP-15: vfwmmacc e4m3 × e5m2 → f16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   0, 0, 1, "vfwmmacc e4m3xe5m2->f16");
            tests++;

            /* TC-FP-16: vfwmmacc f16 → f32 */
            fail |= test_vfwmmacc(4, lmul, lambda, vlenb,
                                   0, 0, 0, "vfwmmacc f16->f32");
            tests++;

            /* TC-FP-17: vfwmmacc bf16 → f32 (altfmt=0, a=1, b=1) */
            fail |= test_vfwmmacc(4, lmul, lambda, vlenb,
                                   0, 1, 1, "vfwmmacc bf16->f32");
            tests++;

            /* TC-FP-18: vfwmmacc f16 × bf16 → f32 */
            fail |= test_vfwmmacc(4, lmul, lambda, vlenb,
                                   0, 0, 1, "vfwmmacc f16xbf16->f32");
            tests++;

            /* TC-FP-19: vfwmmacc f32 → f64 */
            fail |= test_vfwmmacc(8, lmul, lambda, vlenb,
                                   0, 0, 0, "vfwmmacc f32->f64");
            tests++;

            /* TC-FP-20: vfqwmmacc e4m3 → f32 */
            fail |= test_vfqwmmacc(4, lmul, lambda, vlenb,
                                    0, 0, 0, "vfqwmmacc e4m3->f32");
            tests++;

            /* TC-FP-21: vfqwmmacc e5m2 → f32 (altfmt=0, a=1, b=1) */
            fail |= test_vfqwmmacc(4, lmul, lambda, vlenb,
                                    0, 1, 1, "vfqwmmacc e5m2->f32");
            tests++;

            /* TC-FP-22: vfqwmmacc e4m3 × e5m2 → f32 */
            fail |= test_vfqwmmacc(4, lmul, lambda, vlenb,
                                    0, 0, 1, "vfqwmmacc e4m3xe5m2->f32");
            tests++;

            /* TC-FP-23: vfqwmmacc f16 → f64 */
            fail |= test_vfqwmmacc(8, lmul, lambda, vlenb,
                                    0, 0, 0, "vfqwmmacc f16->f64");
            tests++;

            /* TC-FP-24: vfqwmmacc bf16 → f64 (altfmt=0, a=1, b=1) */
            fail |= test_vfqwmmacc(8, lmul, lambda, vlenb,
                                    0, 1, 1, "vfqwmmacc bf16->f64");
            tests++;

            /* --- vfwmmacc FP4 → FP8 (8 dispatch entries at sew=0) --- */

            /* TC-FP-25: e2m1×e2m1 → e4m3 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       0, 0, 0, "vfwmmacc e2m1->e4m3");
            tests++;

            /* TC-FP-26: e2m1×e2m1 → e5m2 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       1, 0, 0, "vfwmmacc e2m1->e5m2");
            tests++;

#if 0
            /*
             * TC-FP-27 through TC-FP-32: E3M0 4-bit FP inputs — SKIPPED
             *
             * E3M0 (altfmt_A = altfmt_B = 1 for 4-bit EEW) was removed from
             * the Zvvmm FP MACC specification.  These encodings are now
             * reserved and raise an illegal-instruction exception.  Verifying
             * that reserved encodings trap correctly requires a SIGILL handler
             * with setjmp/longjmp recovery, which this test harness does not
             * provide.  The cases are kept here for documentation purposes.
             */

            /* TC-FP-27: e3m0×e3m0 → e4m3 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       0, 1, 1, "vfwmmacc e3m0->e4m3");
            tests++;

            /* TC-FP-28: e3m0×e3m0 → e5m2 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       1, 1, 1, "vfwmmacc e3m0->e5m2");
            tests++;

            /* TC-FP-29: e2m1×e3m0 → e4m3 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       0, 0, 1, "vfwmmacc e2m1xe3m0->e4m3");
            tests++;

            /* TC-FP-30: e2m1×e3m0 → e5m2 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       1, 0, 1, "vfwmmacc e2m1xe3m0->e5m2");
            tests++;

            /* TC-FP-31: e3m0×e2m1 → e4m3 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       0, 1, 0, "vfwmmacc e3m0xe2m1->e4m3");
            tests++;

            /* TC-FP-32: e3m0×e2m1 → e5m2 */
            fail |= test_vfwmmacc_fp4(lmul, lambda, vlenb,
                                       1, 1, 0, "vfwmmacc e3m0xe2m1->e5m2");
            tests++;
#endif

            /* --- Remaining vfwmmacc non-FP4 gaps --- */

            /* TC-FP-33: e4m3 → bf16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   1, 0, 0, "vfwmmacc e4m3->bf16");
            tests++;

            /* TC-FP-34: e5m2 → f16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   0, 1, 1, "vfwmmacc e5m2->f16");
            tests++;

            /* TC-FP-35: e5m2×e4m3 → f16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   0, 1, 0, "vfwmmacc e5m2xe4m3->f16");
            tests++;

            /* TC-FP-36: e4m3×e5m2 → bf16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   1, 0, 1, "vfwmmacc e4m3xe5m2->bf16");
            tests++;

            /* TC-FP-37: e5m2×e4m3 → bf16 */
            fail |= test_vfwmmacc(2, lmul, lambda, vlenb,
                                   1, 1, 0, "vfwmmacc e5m2xe4m3->bf16");
            tests++;

            /* TC-FP-38: bf16×f16 → f32 */
            fail |= test_vfwmmacc(4, lmul, lambda, vlenb,
                                   0, 1, 0, "vfwmmacc bf16xf16->f32");
            tests++;

            /* --- vfqwmmacc FP4 → f16/bf16 (8 entries at sew=1) --- */

            /* TC-FP-39: e2m1 → f16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        0, 0, 0, "vfqwmmacc e2m1->f16");
            tests++;

            /* TC-FP-40: e2m1 → bf16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        1, 0, 0, "vfqwmmacc e2m1->bf16");
            tests++;

#if 0
            /*
             * TC-FP-41 through TC-FP-46: E3M0 4-bit FP inputs — SKIPPED
             *
             * Same rationale as TC-FP-27..32 above: E3M0 has been removed
             * from the Zvvmm FP MACC specification.  altfmt_A = altfmt_B = 1
             * for 4-bit EEW is a reserved encoding that raises an
             * illegal-instruction exception, and this harness has no SIGILL
             * recovery infrastructure to test that behaviour.
             */

            /* TC-FP-41: e3m0 → f16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        0, 1, 1, "vfqwmmacc e3m0->f16");
            tests++;

            /* TC-FP-42: e3m0 → bf16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        1, 1, 1, "vfqwmmacc e3m0->bf16");
            tests++;

            /* TC-FP-43: e2m1×e3m0 → f16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        0, 0, 1, "vfqwmmacc e2m1xe3m0->f16");
            tests++;

            /* TC-FP-44: e2m1×e3m0 → bf16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        1, 0, 1, "vfqwmmacc e2m1xe3m0->bf16");
            tests++;

            /* TC-FP-45: e3m0×e2m1 → f16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        0, 1, 0, "vfqwmmacc e3m0xe2m1->f16");
            tests++;

            /* TC-FP-46: e3m0×e2m1 → bf16 */
            fail |= test_vfqwmmacc_fp4(lmul, lambda, vlenb,
                                        1, 1, 0, "vfqwmmacc e3m0xe2m1->bf16");
            tests++;
#endif

            /* --- Remaining vfqwmmacc non-FP4 gaps --- */

            /* TC-FP-47: e5m2×e4m3 → f32 */
            fail |= test_vfqwmmacc(4, lmul, lambda, vlenb,
                                    0, 1, 0, "vfqwmmacc e5m2xe4m3->f32");
            tests++;

            /* TC-FP-48: f16×bf16 → f64 */
            fail |= test_vfqwmmacc(8, lmul, lambda, vlenb,
                                    0, 0, 1, "vfqwmmacc f16xbf16->f64");
            tests++;

            /* TC-FP-49: bf16×f16 → f64 */
            fail |= test_vfqwmmacc(8, lmul, lambda, vlenb,
                                    0, 1, 0, "vfqwmmacc bf16xf16->f64");
            tests++;
        }
    }

    /* Octet-widening FP tests: vf8wmmacc (W=8) */
    for (int li = 0; li < 4; li++) {
        for (int ai = 0; ai < 4; ai++) {
            int lmul   = lmuls[li];
            int lambda = lambdas[ai];

            /* TC-FP-50: E2M1→FP32 (SEW=32, input is 4-bit) */
            fail |= test_vf8wmmacc_fp4(lmul, lambda, vlenb,
                                       0, "vf8wmmacc e2m1->f32");
            tests++;

            /* TC-FP-51: E4M3→FP64 (SEW=64, input is 8-bit) */
            fail |= test_vf8wmmacc(8, lmul, lambda, vlenb,
                                    0, 0, 0, "vf8wmmacc e4m3->f64");
            tests++;

            /* TC-FP-52: E5M2→FP64 */
            fail |= test_vf8wmmacc(8, lmul, lambda, vlenb,
                                    0, 1, 1, "vf8wmmacc e5m2->f64");
            tests++;

            /* TC-FP-53: E4M3×E5M2→FP64 (mixed) */
            fail |= test_vf8wmmacc(8, lmul, lambda, vlenb,
                                    0, 0, 1, "vf8wmmacc e4m3xe5m2->f64");
            tests++;

            /* TC-FP-54: E5M2×E4M3→FP64 (mixed) */
            fail |= test_vf8wmmacc(8, lmul, lambda, vlenb,
                                    0, 1, 0, "vf8wmmacc e5m2xe4m3->f64");
            tests++;
        }
    }

    if (fail) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED (%d test invocations)\n", tests);
    return 0;
}
