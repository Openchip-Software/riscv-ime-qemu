/*
 * RISC-V Zvvmm MX (E8M0-microscaled) Unit Test
 *
 * Validates E8M0-microscaled FP MACC (vm=0 of vfmmacc/vfwmmacc/vfqwmmacc)
 * and MXINT FP-accumulator MACC (vm=0 of vwmmacc/vqwmmacc = vfwimmacc/
 * vfqwimmacc) for all valid (SEW, lambda, LMUL) combinations.
 *
 * Register allocation: v0=scale, v8=A(vs2), v16=B(vs1), v24=C(vd).
 * vm=0 in instruction encoding; v0 holds M×R 16-bit scale pairs.
 *
 * Copyright (c) 2026 VRULL GmbH
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* --------------------------------------------------------------------
 * Instruction encodings: vs2=v8, vs1=v16, vd=v24, vm=0
 * ------------------------------------------------------------------ */

/* MX FP MACC (vm=0 of vfmmacc/vfwmmacc/vfqwmmacc, funct3=001) */
#define INSN_VFMMACC_VV_MX    0x50881C57u  /* funct6=010100 */
#define INSN_VFWMMACC_VV_MX   0x54881C57u  /* funct6=010101 */
#define INSN_VFQWMMACC_VV_MX  0x58881C57u  /* funct6=010110 */

/* MXINT (vm=0 of vwmmacc/vqwmmacc/v8wmmacc, funct3=000) */
#define INSN_VWMMACC_VV_MX    0xE4880C57u  /* funct6=111001 */
#define INSN_VQWMMACC_VV_MX   0xE8880C57u  /* funct6=111010 */
#define INSN_V8WMMACC_VV_MX   0xEC880C57u  /* funct6=111011 */

/* MX FP (vm=0 of vf8wmmacc, funct3=001) */
#define INSN_VF8WMMACC_VV_MX  0x5C881C57u  /* funct6=010111 */

/* Tile load/store for v8, v16, v24 (a0=base, a1=stride in elements) */
#define INSN_VMTL_V8   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x07u)
#define INSN_VMTL_V16  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 16u <<  7 | 0x07u)
#define INSN_VMTL_V24  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 24u <<  7 | 0x07u)
#define INSN_VMTS_V24  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 24u <<  7 | 0x27u)

/* vxor.vv v24, v24, v24 — zero accumulator */
#define INSN_VXOR_V24  0x2F8C0C57u

/* vle16.v v0, (a0) — load scale pairs into v0 */
#define INSN_VLE16_V0  0x02055007u

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

#define DO_VXOR_V24() \
    asm volatile(".4byte %[i]" : : [i] "i"(INSN_VXOR_V24) : "memory")

/* Per-instruction emit stubs (keep DO_MACC argument as a literal constant) */
typedef void (*emit_fn)(void);
static void emit_vfmmacc_mx(void)  { DO_MACC(INSN_VFMMACC_VV_MX); }
static void emit_vfwmmacc_mx(void) { DO_MACC(INSN_VFWMMACC_VV_MX); }
static void emit_vfqwmmacc_mx(void){ DO_MACC(INSN_VFQWMMACC_VV_MX); }
static void emit_vwmmacc_mx(void)  { DO_MACC(INSN_VWMMACC_VV_MX); }
static void emit_vqwmmacc_mx(void) { DO_MACC(INSN_VQWMMACC_VV_MX); }
static void emit_vf8wmmacc_mx(void){ DO_MACC(INSN_VF8WMMACC_VV_MX); }
static void emit_v8wmmacc_mx(void) { DO_MACC(INSN_V8WMMACC_VV_MX); }

/* Set vtype using vsetvl with explicit AVL.
 * For MX mode, avl = R * k_eff  so that N = vl/k_eff = R = scale columns. */
static inline uint64_t set_macc_vtype(int sew_bytes, int lmul, int lambda,
                                      int altfmt, int altfmt_a, int altfmt_b,
                                      uint64_t avl)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul)
                   | ((uint64_t)__builtin_ctz(sew_bytes) << 3)
                   | ((uint64_t)altfmt   << 8)
                   | ((uint64_t)altfmt_a << 58)
                   | ((uint64_t)altfmt_b << 57)
                   | ((uint64_t)(__builtin_ctz(lambda) + 1) << 60);
    uint64_t vl;
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vtype));
    return vl;
}

/* Load n uint16_t values from buf into v0 (using vsetvl + vle16) */
static void load_scale_v0(const uint16_t *buf, uint64_t vlenb)
{
    /* n_scale_pairs = M * R = (VLENB/SEW/lambda) * (lambda*SEW_bits/16)
     * = VLENB / 2  (always, regardless of SEW/lambda) */
    uint64_t n = vlenb / 2;
    /* Set vtype: SEW=16, LMUL=1 */
    uint64_t vtype16 = (1u << 3); /* VSEW=001=16-bit, VLMUL=000=1 */
    uint64_t vl;
    register const uint16_t *addr asm("a0") = buf;
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(n), "r"(vtype16));
    asm volatile(".4byte %[i]" : : [i] "i"(INSN_VLE16_V0), "r"(addr)
                 : "v0", "memory");
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
 * FP format conversion helpers
 * ------------------------------------------------------------------ */

static uint32_t f32_to_bits(float f)   { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bits_to_f32(uint32_t u)   { float f;    memcpy(&f, &u, 4); return f; }
static uint64_t f64_to_bits(double d)  { uint64_t u; memcpy(&u, &d, 8); return u; }
static double bits_to_f64(uint64_t u)  { double d;   memcpy(&d, &u, 8); return d; }

static float f16_to_float(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    if (exp == 0) { exp = mant ? (exp = 0, mant <<= 1, exp + 127 - 15) : 0; }
    else if (exp == 0x1f) { exp = 0xff; }
    else { exp += 127 - 15; }
    return bits_to_f32((sign << 31) | (exp << 23) | (mant << 13));
}
static uint16_t float_to_f16(float f)
{
    uint32_t b = f32_to_bits(f);
    uint32_t sign = b >> 31; int32_t exp = ((b >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (b >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)(sign << 15);
    if (exp >= 0x1f) return (uint16_t)((sign << 15) | (0x1f << 10));
    return (uint16_t)((sign << 15) | (exp << 10) | mant);
}
static float bf16_to_float(uint16_t b) { return bits_to_f32((uint32_t)b << 16); }
static uint16_t float_to_bf16(float f) { return (uint16_t)(f32_to_bits(f) >> 16); }

static float e4m3_to_float(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1, exp = (v >> 3) & 0xf, mant = v & 0x7;
    if (!exp) return sign ? -0.0f : (mant ? bits_to_f32(0x7fc00000) : 0.0f);
    if (exp == 0xf && mant == 0x7) return bits_to_f32(0x7fc00000u);
    return bits_to_f32((sign << 31) | ((exp - 7 + 127) << 23) | (mant << 20));
}
static uint8_t float_to_e4m3(float f)
{
    uint32_t b = f32_to_bits(f);
    uint32_t sign = b >> 31; int32_t exp = ((b >> 23) & 0xff) - 127 + 7;
    uint32_t mant = (b >> 20) & 0x7;
    if (exp <= 0) return (uint8_t)(sign << 7);
    if (exp >= 0xf) return (uint8_t)((sign << 7) | (0xe << 3) | 0x7);
    return (uint8_t)((sign << 7) | (exp << 3) | mant);
}

/* E8M0: 8-bit biased exponent (bias=127), no sign, no mantissa.
 * Value = 2^(e8m0 - 127). e8m0=0xFF → NaN (not used in tests).
 * e8m0=0x7F → 2^0 = 1.0, e8m0=0x80 → 2.0, e8m0=0x7E → 0.5.
 */
static double e8m0_to_double(uint8_t e8m0)
{
    /* 2^(e8m0 - 127) using f32 bit trick: exponent field = e8m0 */
    uint32_t bits = (e8m0 == 0) ? 0x00400000u : ((uint32_t)e8m0 << 23);
    return (double)bits_to_f32(bits);
}

/* Convert integer value to raw FP encoding for accumulator format.
 * accum_sew: bytes (2=f16/bf16, 4=f32, 8=f64), altfmt selects bf16 for sew=2 */
static uint64_t int_to_accum_raw(double d, int accum_sew, int altfmt)
{
    switch (accum_sew) {
    case 2: return altfmt ? float_to_bf16((float)d) : float_to_f16((float)d);
    case 4: return f32_to_bits((float)d);
    case 8: return f64_to_bits(d);
    }
    return 0;
}

/* Convert raw accumulator encoding to double */
static double accum_raw_to_double(uint64_t raw, int accum_sew, int altfmt)
{
    switch (accum_sew) {
    case 2: return altfmt ? (double)bf16_to_float((uint16_t)raw)
                          : (double)f16_to_float((uint16_t)raw);
    case 4: return (double)bits_to_f32((uint32_t)raw);
    case 8: return bits_to_f64(raw);
    }
    return 0.0;
}

/* ====================================================================
 * MX FP MACC test helper
 *
 * Uses uniform input values of 1.0 for all A, B elements.
 * IP[i][j] = K (in FP accumulator type).
 * scale[i][j] = scale_combined = e8m0_A × e8m0_B.
 * C_out[i][j] = scale * K + C_init[i][j].
 * ==================================================================== */

static int test_mx_fmmacc(int accum_sew,  /* 2=f16/bf16, 4=f32, 8=f64 */
                           int input_sew,  /* input element size in bytes */
                           int widening,   /* 1=same-width, 2=widen, 4=qwiden */
                           int lmul, int lambda, uint64_t vlenb,
                           int altfmt, int altfmt_a, int altfmt_b,
                           uint8_t scale_a_e8m0, uint8_t scale_b_e8m0,
                           emit_fn emit, const char *label)
{
    int M = vlenb / (accum_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);

    /* Valid only when lmul == mul_c: ensures VMTS covers the full tile. */
    if (lmul != mul_c || M < widening) {
        return 0;
    }

    int K = lambda * widening * lmul;          /* k_eff */
    int R = lambda * accum_sew / 2;            /* N scale cols per output row */
    /* avl = R * K = vlenb * widening / 2 (always, when lmul == mul_c) */
    uint64_t avl    = (uint64_t)vlenb * widening / 2;
    uint64_t max_vl = (uint64_t)vlenb * lmul / accum_sew;

    /* For widening > 1, avl may exceed max_vl; skip those configs. */
    if (avl > max_vl) {
        return 0;
    }

    int N_active = R;               /* one MX call produces M × R outputs */
    int a_ld = lambda * lmul;       /* = M (since lmul == mul_c) */
    int b_ld = a_ld;
    int c_ld = lambda * lmul;       /* = M */
    int c_nelem = M * M;            /* full row-major C tile */

    int input_mu = vlenb / (input_sew * lambda);

    uint8_t *A_buf  = calloc(input_mu * a_ld, input_sew);
    uint8_t *B_buf  = calloc(input_mu * b_ld, input_sew);
    uint8_t *C_dst  = calloc(c_nelem, accum_sew);
    uint16_t *scale_buf = calloc(vlenb / 2, sizeof(uint16_t));

    /* Fill A with 1.0 in input format */
    {
        uint64_t one_raw;
        if (input_sew == 1)      one_raw = float_to_e4m3(1.0f);
        else if (input_sew == 2 && altfmt_a)
                                  one_raw = float_to_bf16(1.0f);
        else if (input_sew == 2)  one_raw = float_to_f16(1.0f);
        else if (input_sew == 4)  one_raw = f32_to_bits(1.0f);
        else                      one_raw = f64_to_bits(1.0);
        for (int idx = 0; idx < input_mu * a_ld; idx++) {
            write_elem(A_buf, idx, input_sew, one_raw);
        }
    }

    /* Fill B with 1.0 in input format */
    {
        uint64_t one_raw;
        if (input_sew == 1)      one_raw = float_to_e4m3(1.0f);
        else if (input_sew == 2 && altfmt_b)
                                  one_raw = float_to_bf16(1.0f);
        else if (input_sew == 2)  one_raw = float_to_f16(1.0f);
        else if (input_sew == 4)  one_raw = f32_to_bits(1.0f);
        else                      one_raw = f64_to_bits(1.0);
        for (int idx = 0; idx < input_mu * b_ld; idx++) {
            write_elem(B_buf, idx, input_sew, one_raw);
        }
    }

    /* Fill scale pairs: sp[k] = (scale_b_e8m0 << 8) | scale_a_e8m0 */
    uint16_t sp = ((uint16_t)scale_b_e8m0 << 8) | scale_a_e8m0;
    for (int i = 0; i < (int)(vlenb / 2); i++) {
        scale_buf[i] = sp;
    }

    /* Load scale pairs into v0 (uses its own vtype/VL). */
    load_scale_v0(scale_buf, vlenb);

    /* Max VL for tile loads and accumulator zeroing. */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b, max_vl);

    DO_TILE(INSN_VMTL_V8,  A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V16, B_buf, (uint64_t)b_ld);
    DO_VXOR_V24();

    /* MX instruction: VL = avl = R * K so that N = R output columns. */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b, avl);
    emit();

    /* VMTS: VL = c_nelem = M * M (full row-major C tile). */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b,
                   (uint64_t)c_nelem);
    DO_TILE(INSN_VMTS_V24, C_dst, (uint64_t)c_ld);

    /* Reference: scale_combined * K for each active (i,j) */
    double sc_combined = e8m0_to_double(scale_a_e8m0) *
                         e8m0_to_double(scale_b_e8m0);

    int fail = 0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            double expected_d = sc_combined * K;
            uint64_t expected = int_to_accum_raw(expected_d, accum_sew, altfmt);
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s lmul=%d lam=%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected, expected_d);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst); free(scale_buf);
    return fail;
}

/* ====================================================================
 * MXINT test helper
 *
 * Uses integer inputs: A[i][k] = a_val (signed/unsigned), B[k][j] = b_val.
 * IP_int[i][j] = K * a_val * b_val (integer).
 * C_out[i][j] = scale_combined * (double)IP_int  (in FP accumulator).
 * ==================================================================== */

static int test_mxint(int accum_sew,  /* 2=f16/bf16, 4=f32, 8=f64 */
                      int input_sew,  /* integer input element size in bytes */
                      int widening,   /* 2 or 4 */
                      int lmul, int lambda, uint64_t vlenb,
                      int altfmt,     /* 0=f16/f32/f64, 1=bf16 */
                      int a_val, int b_val,          /* signed input values */
                      uint8_t scale_a_e8m0, uint8_t scale_b_e8m0,
                      emit_fn emit, const char *label)
{
    int M = vlenb / (accum_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);

    if (lmul != mul_c || M < widening) {
        return 0;
    }

    int K = lambda * widening * lmul;
    int R = lambda * accum_sew / 2;
    uint64_t avl    = (uint64_t)vlenb * widening / 2;
    uint64_t max_vl = (uint64_t)vlenb * lmul / accum_sew;

    if (avl > max_vl) {
        return 0;
    }

    int N_active = R;
    int a_ld = lambda * lmul;
    int b_ld = a_ld;
    int c_ld = lambda * lmul;   /* = M */
    int c_nelem = M * M;            /* full row-major C tile */

    int input_mu = vlenb / (input_sew * lambda);

    uint8_t *A_buf  = calloc(input_mu * a_ld, input_sew);
    uint8_t *B_buf  = calloc(input_mu * b_ld, input_sew);
    uint8_t *C_dst  = calloc(c_nelem, accum_sew);
    uint16_t *scale_buf = calloc(vlenb / 2, sizeof(uint16_t));

    /* Fill A with a_val (truncated to input_sew bytes) */
    for (int idx = 0; idx < input_mu * a_ld; idx++) {
        write_elem(A_buf, idx, input_sew, (uint64_t)(int64_t)a_val);
    }
    /* Fill B with b_val */
    for (int idx = 0; idx < input_mu * b_ld; idx++) {
        write_elem(B_buf, idx, input_sew, (uint64_t)(int64_t)b_val);
    }

    /* Fill scale pairs */
    uint16_t sp = ((uint16_t)scale_b_e8m0 << 8) | scale_a_e8m0;
    for (int i = 0; i < (int)(vlenb / 2); i++) {
        scale_buf[i] = sp;
    }

    load_scale_v0(scale_buf, vlenb);

    set_macc_vtype(accum_sew, lmul, lambda, altfmt, 0, 0, max_vl);

    DO_TILE(INSN_VMTL_V8,  A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V16, B_buf, (uint64_t)b_ld);
    DO_VXOR_V24();

    set_macc_vtype(accum_sew, lmul, lambda, altfmt, 0, 0, avl);
    emit();

    /* VMTS: VL = c_nelem = M * M (full row-major C tile). */
    set_macc_vtype(accum_sew, lmul, lambda, altfmt, 0, 0, (uint64_t)c_nelem);
    DO_TILE(INSN_VMTS_V24, C_dst, (uint64_t)c_ld);

    /* Reference: scale_combined * (a_val * b_val * K), signed inputs */
    double sc_combined = e8m0_to_double(scale_a_e8m0) *
                         e8m0_to_double(scale_b_e8m0);
    double ip_d = (double)a_val * (double)b_val * K;
    double expected_d = sc_combined * ip_d;

    int fail = 0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            uint64_t expected = int_to_accum_raw(expected_d, accum_sew, altfmt);
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s lmul=%d lam=%d: "
                       "C[%d][%d] got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, expected_d);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst); free(scale_buf);
    return fail;
}

/* ====================================================================
 * Main: iterate over (lambda, lmul) combinations
 * ==================================================================== */

/* set_macc_vtype with bs=1 (block size 16) */
static inline uint64_t set_macc_vtype_bs16(int sew_bytes, int lmul, int lambda,
                                           int altfmt, int altfmt_a, int altfmt_b,
                                           uint64_t avl)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul)
                   | ((uint64_t)__builtin_ctz(sew_bytes) << 3)
                   | ((uint64_t)altfmt   << 8)
                   | (1ULL << 59)  /* bs=1: block_size=16 */
                   | ((uint64_t)altfmt_a << 58)
                   | ((uint64_t)altfmt_b << 57)
                   | ((uint64_t)(__builtin_ctz(lambda) + 1) << 60);
    uint64_t vl;
    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vtype));
    return vl;
}

/* ====================================================================
 * BS=16 MX FP test: same as test_mx_fmmacc but with bs=1.
 * Reference splits inner product into 16-element blocks.
 * ==================================================================== */

static int test_mx_fmmacc_bs16(int accum_sew, int input_sew, int widening,
                                int lmul, int lambda, uint64_t vlenb,
                                int altfmt, int altfmt_a, int altfmt_b,
                                uint8_t scale_a_e8m0, uint8_t scale_b_e8m0,
                                emit_fn emit, const char *label)
{
    int M = vlenb / (accum_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);

    if (lmul != mul_c || M < widening) {
        return 0;
    }

    /* BS=16 legality: W × LMUL ≤ SEW (in bytes: widening * lmul ≤ sewb) */
    if (widening * lmul > accum_sew) {
        return 0;
    }

    int K = lambda * widening * lmul;
    int R = lambda * accum_sew / 2;
    int block_size = 16;
    int S = (K + block_size - 1) / block_size;

    if (S < 2) {
        return 0;  /* Only test when we actually have >1 block */
    }

    uint64_t avl    = (uint64_t)vlenb * widening / 2;
    uint64_t max_vl = (uint64_t)vlenb * lmul / accum_sew;

    if (avl > max_vl) {
        return 0;
    }

    int N_active = R;
    int a_ld = lambda * lmul;
    int b_ld = a_ld;
    int c_ld = lambda * lmul;
    int c_nelem = M * M;            /* full row-major C tile */

    int input_mu = vlenb / (input_sew * lambda);

    uint8_t *A_buf   = calloc(input_mu * a_ld, input_sew);
    uint8_t *B_buf   = calloc(input_mu * b_ld, input_sew);
    uint8_t *C_dst   = calloc(c_nelem, accum_sew);
    uint16_t *scale_buf = calloc(vlenb / 2, sizeof(uint16_t));

    /* Fill A, B with FP 1.0 at input_sew.
     * E4M3: 1.0 = 0x38, E5M2: 1.0 = 0x3C, F16: 1.0 = 0x3C00 */
    if (input_sew == 1) {
        uint8_t fp_one_a = altfmt_a ? 0x3C : 0x38;
        uint8_t fp_one_b = altfmt_b ? 0x3C : 0x38;
        memset(A_buf, fp_one_a, input_mu * a_ld);
        memset(B_buf, fp_one_b, input_mu * b_ld);
    } else {
        uint16_t fp16_one = altfmt_a ? 0x3F80 : 0x3C00; /* bf16/f16 1.0 */
        for (int idx = 0; idx < input_mu * a_ld; idx++)
            write_elem(A_buf, idx, 2, fp16_one);
        fp16_one = altfmt_b ? 0x3F80 : 0x3C00;
        for (int idx = 0; idx < input_mu * b_ld; idx++)
            write_elem(B_buf, idx, 2, fp16_one);
    }

    /* Spec separate-pair MX layout: pair_A at v0[i*R+s] (low byte = scale_A),
     * pair_B at v0[j*R+s] (high byte = scale_B).  All scales uniform here, so
     * (scale_b<<8 | scale_a) at every (i*R+s) and (j*R+s) is layout-agnostic. */
    for (int i = 0; i < M; i++) {
        for (int s = 0; s < S; s++) {
            int idx = i * R + s;
            if (idx < (int)(vlenb / 2)) {
                scale_buf[idx] = (uint16_t)scale_b_e8m0 << 8
                               | (uint16_t)scale_a_e8m0;
            }
        }
    }
    for (int j = 0; j < N_active; j++) {
        for (int s = 0; s < S; s++) {
            int idx = j * R + s;
            if (idx < (int)(vlenb / 2)) {
                scale_buf[idx] = (uint16_t)scale_b_e8m0 << 8
                               | (uint16_t)scale_a_e8m0;
            }
        }
    }

    /* Load tiles with BS=16 vtype */
    set_macc_vtype_bs16(input_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b,
                        max_vl);
    DO_TILE(INSN_VMTL_V16, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V8,  B_buf, (uint64_t)b_ld);

    /* Load scales into v0 */
    load_scale_v0(scale_buf, vlenb);

    /* Set accum vtype with bs=1 and execute */
    set_macc_vtype_bs16(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b,
                        avl);
    DO_VXOR_V24();
    emit();

    /* Store C */
    int c_nelem_out = M * R;
    set_macc_vtype_bs16(accum_sew, lmul, lambda, altfmt, altfmt_a, altfmt_b,
                        (uint64_t)c_nelem_out);
    DO_TILE(INSN_VMTS_V24, C_dst, (uint64_t)c_ld);

    /* Reference: with all-1.0 inputs and scale_a*scale_b from e8m0,
     * each block of 16 contributes: scale * 16 * 1.0 * 1.0.
     * Total = S * scale * 16 = K * scale. */
    double scale_a = ldexp(1.0, (int)scale_a_e8m0 - 127);
    double scale_b = ldexp(1.0, (int)scale_b_e8m0 - 127);
    double scale_val = scale_a * scale_b;
    double ref_acc = (double)K * scale_val;

    uint64_t expected = int_to_accum_raw(ref_acc, accum_sew, altfmt);

    int fail = 0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d S=%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, input_sew*8, accum_sew*8, lmul, lambda, S,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, ref_acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst); free(scale_buf);
    return fail;
}

/* BS=16 MXINT test */
static int test_mxint_bs16(int accum_sew, int input_sew, int widening,
                            int lmul, int lambda, uint64_t vlenb,
                            int altfmt, int a_val, int b_val,
                            uint8_t scale_a_e8m0, uint8_t scale_b_e8m0,
                            emit_fn emit, const char *label)
{
    int M = vlenb / (accum_sew * lambda);
    int mul_c = vlenb / (accum_sew * lambda * lambda);

    if (lmul != mul_c || M < widening) {
        return 0;
    }

    if (widening * lmul > accum_sew) {
        return 0;
    }

    int K = lambda * widening * lmul;
    int R = lambda * accum_sew / 2;
    int block_size = 16;
    int S = (K + block_size - 1) / block_size;

    if (S < 2) {
        return 0;
    }

    uint64_t avl    = (uint64_t)vlenb * widening / 2;
    uint64_t max_vl = (uint64_t)vlenb * lmul / accum_sew;

    if (avl > max_vl) {
        return 0;
    }

    int N_active = R;
    int a_ld = lambda * lmul;
    int b_ld = a_ld;
    int c_ld = lambda * lmul;
    int c_nelem = M * M;            /* full row-major C tile */

    int input_mu = vlenb / (input_sew * lambda);

    uint8_t *A_buf = calloc(input_mu * a_ld, input_sew);
    uint8_t *B_buf = calloc(input_mu * b_ld, input_sew);
    uint8_t *C_dst = calloc(c_nelem, accum_sew);
    uint16_t *scale_buf = calloc(vlenb / 2, sizeof(uint16_t));

    /* Fill A with a_val, B with b_val */
    for (int idx = 0; idx < input_mu * a_ld; idx++) {
        write_elem(A_buf, idx, input_sew, (uint64_t)(int64_t)a_val);
    }
    for (int idx = 0; idx < input_mu * b_ld; idx++) {
        write_elem(B_buf, idx, input_sew, (uint64_t)(int64_t)b_val);
    }

    /* Fill scale pairs: S per (i,j) at v0[i*R + j*S + s] */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < R; j++) {
            for (int s = 0; s < S; s++) {
                int idx = i * R + j * S + s;
                if (idx < (int)(vlenb / 2)) {
                    scale_buf[idx] = (uint16_t)scale_b_e8m0 << 8
                                   | (uint16_t)scale_a_e8m0;
                }
            }
        }
    }

    /* Load tiles with BS=16 vtype */
    set_macc_vtype_bs16(input_sew, lmul, lambda, 0, 0, 0, max_vl);
    DO_TILE(INSN_VMTL_V16, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V8,  B_buf, (uint64_t)b_ld);

    load_scale_v0(scale_buf, vlenb);

    set_macc_vtype_bs16(accum_sew, lmul, lambda, altfmt, 0, 0, avl);
    DO_VXOR_V24();
    emit();

    int c_nelem_out = M * R;
    set_macc_vtype_bs16(accum_sew, lmul, lambda, altfmt, 0, 0,
                        (uint64_t)c_nelem_out);
    DO_TILE(INSN_VMTS_V24, C_dst, (uint64_t)c_ld);

    /* Reference: each block of 16 gives ip = 16 * a_val * b_val,
     * then scale * ip accumulated over S blocks.
     * Total = S * scale * 16 * a_val * b_val = K * scale * a_val * b_val */
    double scale_a = ldexp(1.0, (int)scale_a_e8m0 - 127);
    double scale_b = ldexp(1.0, (int)scale_b_e8m0 - 127);
    double scale_val = scale_a * scale_b;
    double ref_acc = (double)K * (double)a_val * (double)b_val * scale_val;

    uint64_t expected = int_to_accum_raw(ref_acc, accum_sew, altfmt);

    int fail = 0;
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d S=%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx (ref=%.6g)\n",
                       label, input_sew*8, accum_sew*8, lmul, lambda, S,
                       row, col,
                       (unsigned long)got, (unsigned long)expected, ref_acc);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst); free(scale_buf);
    return fail;
}

int main(void)
{
    /* Detect VLENB via the vlenb CSR (VLEN/8 in bytes). */
    uint64_t vlenb;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));

    /* Supported lambda values (1..64) */
    static const int lambdas[] = { 1, 2, 4, 8, 16, 32, 64 };
    int nl = sizeof(lambdas) / sizeof(lambdas[0]);

    int fail = 0;
    int tests = 0;

    for (int li = 0; li < nl; li++) {
        int lambda = lambdas[li];
        for (int lmul = 1; lmul <= 8; lmul *= 2) {

            /* ---- MX FP MACC: vfmmacc (same-width, widening=1) ---- */

            /* f32 × f32 → f32, scale = 1.0×1.0 = 1.0 */
            fail |= test_mx_fmmacc(4, 4, 1, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vfmmacc_mx,
                                   "mx vfmmacc f32, sc=1.0");
            tests++;

            /* f32 × f32 → f32, scale = 2.0×1.0 = 2.0 */
            fail |= test_mx_fmmacc(4, 4, 1, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x80,
                                   emit_vfmmacc_mx,
                                   "mx vfmmacc f32, sc=2.0");
            tests++;

            /* f16 × f16 → f16, scale = 1.0 */
            fail |= test_mx_fmmacc(2, 2, 1, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vfmmacc_mx,
                                   "mx vfmmacc f16, sc=1.0");
            tests++;

            /* bf16 × bf16 → bf16, scale = 1.0 */
            fail |= test_mx_fmmacc(2, 2, 1, lmul, lambda, vlenb,
                                   1, 1, 1, 0x7F, 0x7F,
                                   emit_vfmmacc_mx,
                                   "mx vfmmacc bf16, sc=1.0");
            tests++;

            /* ---- MX FP MACC: vfwmmacc (widening=2) ---- */

            /* e4m3 → f16, scale = 1.0 */
            fail |= test_mx_fmmacc(2, 1, 2, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vfwmmacc_mx,
                                   "mx vfwmmacc e4m3->f16, sc=1.0");
            tests++;

            /* f16 → f32, scale = 1.0 */
            fail |= test_mx_fmmacc(4, 2, 2, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vfwmmacc_mx,
                                   "mx vfwmmacc f16->f32, sc=1.0");
            tests++;

            /* f32 → f64, scale = 2.0 */
            fail |= test_mx_fmmacc(8, 4, 2, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x80,
                                   emit_vfwmmacc_mx,
                                   "mx vfwmmacc f32->f64, sc=2.0");
            tests++;

            /* ---- MX FP MACC: vfqwmmacc (widening=4) ---- */

            /* e4m3 → f32, scale = 1.0 */
            fail |= test_mx_fmmacc(4, 1, 4, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vfqwmmacc_mx,
                                   "mx vfqwmmacc e4m3->f32, sc=1.0");
            tests++;

            /* f16 → f64, scale = 1.0 */
            fail |= test_mx_fmmacc(8, 2, 4, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vfqwmmacc_mx,
                                   "mx vfqwmmacc f16->f64, sc=1.0");
            tests++;

            /* ---- MXINT: vfwimmacc (vm=0 of vwmmacc, widening=2) ---- */

            /* I8×I8 → F16, a=2, b=3 */
            fail |= test_mxint(2, 1, 2, lmul, lambda, vlenb,
                               0, 2, 3, 0x7F, 0x7F,
                               emit_vwmmacc_mx,
                               "mxint vfwimmacc I8->F16");
            tests++;

            /* I8×I8 → BF16, scale=2.0, a=-1, b=1 */
            fail |= test_mxint(2, 1, 2, lmul, lambda, vlenb,
                               1, -1, 1, 0x7F, 0x80,
                               emit_vwmmacc_mx,
                               "mxint vfwimmacc I8->BF16 sc=2.0");
            tests++;

            /* I16×I16 → F32, a=1, b=2 */
            fail |= test_mxint(4, 2, 2, lmul, lambda, vlenb,
                               0, 1, 2, 0x7F, 0x7F,
                               emit_vwmmacc_mx,
                               "mxint vfwimmacc I16->F32");
            tests++;

            /* ---- MXINT: vfqwimmacc (vm=0 of vqwmmacc, widening=4) ---- */

            /* I8×I8 → F32, a=3, b=4 */
            fail |= test_mxint(4, 1, 4, lmul, lambda, vlenb,
                               0, 3, 4, 0x7F, 0x7F,
                               emit_vqwmmacc_mx,
                               "mxint vfqwimmacc I8->F32");
            tests++;

            /* I8×I8 → F32, scale=2.0, a=1, b=-1 */
            fail |= test_mxint(4, 1, 4, lmul, lambda, vlenb,
                               0, 1, -1, 0x7F, 0x80,
                               emit_vqwmmacc_mx,
                               "mxint vfqwimmacc I8->F32 sc=2.0");
            tests++;

            /* I16×I16 → F64, a=5, b=6 */
            fail |= test_mxint(8, 2, 4, lmul, lambda, vlenb,
                               0, 5, 6, 0x7F, 0x7F,
                               emit_vqwmmacc_mx,
                               "mxint vfqwimmacc I16->F64");
            tests++;

            /* ---- MX FP MACC: vf8wmmacc (widening=8) ---- */

            /* e4m3 → f64, scale = 1.0 */
            fail |= test_mx_fmmacc(8, 1, 8, lmul, lambda, vlenb,
                                   0, 0, 0, 0x7F, 0x7F,
                                   emit_vf8wmmacc_mx,
                                   "mx vf8wmmacc e4m3->f64, sc=1.0");
            tests++;

            /* e5m2 → f64, scale = 2.0 */
            fail |= test_mx_fmmacc(8, 1, 8, lmul, lambda, vlenb,
                                   0, 1, 1, 0x7F, 0x80,
                                   emit_vf8wmmacc_mx,
                                   "mx vf8wmmacc e5m2->f64, sc=2.0");
            tests++;

            /* ---- MXINT: vf8wimmacc (vm=0 of v8wmmacc, widening=8) ---- */

            /* I8×I8 → F64, a=2, b=3 */
            fail |= test_mxint(8, 1, 8, lmul, lambda, vlenb,
                               0, 2, 3, 0x7F, 0x7F,
                               emit_v8wmmacc_mx,
                               "mxint vf8wimmacc I8->F64");
            tests++;

            /* I8×I8 → F64, scale=2.0, a=1, b=-1 */
            fail |= test_mxint(8, 1, 8, lmul, lambda, vlenb,
                               0, 1, -1, 0x7F, 0x80,
                               emit_v8wmmacc_mx,
                               "mxint vf8wimmacc I8->F64 sc=2.0");
            tests++;

            /* ---- BS=16 tests ---- */
            /* BS=16 legality: W × LMUL ≤ SEW.
             * For vfqwmmacc (W=4): need 4*LMUL ≤ SEW_val.
             * With SEW=16 (sewval=16): LMUL ≤ 4.
             * k_eff = lambda*4*LMUL; for S>1 with BS=16: k_eff > 16.
             * Example: lambda=2, W=4, LMUL=4 → k_eff=32, S=2. */

            /* MX FP BS=16: e4m3→f32, sc=1.0 (2 scale blocks) */
            fail |= test_mx_fmmacc_bs16(4, 1, 4, lmul, lambda, vlenb,
                                        0, 0, 0, 0x7F, 0x7F,
                                        emit_vfqwmmacc_mx,
                                        "mx-bs16 vfqwmmacc e4m3->f32");
            tests++;

            /* MXINT BS=16: I8→F32, a=2, b=3, sc=1.0 */
            fail |= test_mxint_bs16(4, 1, 4, lmul, lambda, vlenb,
                                    0, 2, 3, 0x7F, 0x7F,
                                    emit_vqwmmacc_mx,
                                    "mxint-bs16 vfqwimmacc I8->F32");
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
