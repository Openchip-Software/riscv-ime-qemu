/*
 * RISC-V translation routines for the Integrated Matrix (Standard) Extension
 *
 * Copyright (c) 2025 VRULL GmbH, opensource@vrull.eu
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "fpu/softfloat.h"

#define REQUIRE_ZVMMA(ctx) do {                 \
    if (!ctx->cfg_ptr->ext_zvmma) {             \
        return false;                           \
    }                                           \
} while (0)

#include <stdio.h>   /* TODO: for debug only */
#include "vector_internals.h"

enum direction { COLUMN_MAJOR = 0, ROW_MAJOR = 1};

static void element_to_regoffset (uint16_t a, uint16_t b, enum direction dir,
                                  uint16_t lambda, uint16_t sewb, uint16_t widening,
                                  uint8_t *regoff, uint16_t *byteoff, uint8_t *bitoff)
{
    uint8_t  k_consecutive = lambda * widening;
    uint16_t i = (dir == COLUMN_MAJOR) ? a : b;
    uint16_t j = (dir == COLUMN_MAJOR) ? b : a; /* the consecutive elements */

    /* determine register in a register group (LMUL or output-group) */
    uint8_t  reg_no = i / k_consecutive;
    uint8_t  reg_i  = i % k_consecutive;

    /* byte and bit-offset in the register */
    uint16_t byte_off_in_reg = (j * k_consecutive + reg_i) * sewb / widening;
    uint16_t bit_off = (reg_i * (sewb << 3) / widening) & 0x7;

    *regoff = reg_no;
    *byteoff = byte_off_in_reg;
    *bitoff = bit_off;
}

#define GEN_ZVMMA_VMMACC_VV(NAME, TD, T1, T2, TX1, TX2,	OP)		\
    void HELPER(NAME)(void *vd, void *v0, void *vs1,			\
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint8_t  widening = sizeof(TD) / sizeof(T1);                  \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);          \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint16_t vlmax = vext_get_vlmax(vlenb, vsew, 0);              \
    uint32_t vl = env->vl;                                              \
    const uint8_t  lambda = 1 << (env->vlambda - 1);                    \
    const uint8_t  lmul = 1 << vext_lmul(desc);                         \
    const uint8_t  srm = env->vsrm;                                     \
    const enum direction dir = (srm == 0b01) ? ROW_MAJOR : COLUMN_MAJOR; \
    /* fprintf (stderr, "dir %d\n", dir); */                            \
    /* Determine cardinality and geometry */                            \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    uint8_t  cols, rows;                                                \
                                                                        \
    /* If VL is not a multiple of K_effective, raise an                 \
       Illegal Instruction exception. */                                \
    if (vl % lambda)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    /* With LMUL > 1, a higher than intended VL(C) can be set */        \
    if (vl > vlmax)                                                     \
        vl = vlmax;                                                     \
                                                                        \
    if (dir == ROW_MAJOR) {                                             \
        rows = vlmax / lambda;                                          \
        cols = vl / lambda;                                             \
    } else {                                                            \
        rows = vl / lambda;                                             \
        cols = vlmax / lambda;                                          \
    }                                                                   \
                                                                        \
/* \
    fprintf(stderr, "out-rows %d\n", vl / lambda);                      \
    fprintf(stderr, "rows %d cols %d\n", rows, cols);                   \
    fprintf(stderr, "lmul %d vl %d lambda %d vlmax %d sewb %d\n", lmul, vl, lambda, vlmax, sewb); \
*/ \
                                                                        \
    for (int r = 0; r < rows; ++r) {                                    \
        for (int c = 0; c < cols; ++c) {                                \
            uint8_t regoff;                                             \
            uint16_t byteoff;                                           \
            uint8_t bitoff;                                             \
                                                                        \
            element_to_regoffset(r, c, dir, lambda, sewb, 1, &regoff, &byteoff, &bitoff); \
            TD *c_elem_ptr = (TD*)(vd + regoff * vlenb + byteoff);      \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            /* Calculate T[r, c] */                                     \
            for (int k = 0; k < k_eff; ++k)                             \
            {                                                           \
                element_to_regoffset(r, k, ROW_MAJOR, lambda, sewb, widening, &regoff, &byteoff, &bitoff); \
                TX1 a = *(T1*)(vs1 + regoff * vlenb + byteoff);         \
                element_to_regoffset(k, c, COLUMN_MAJOR, lambda, sewb, widening, &regoff, &byteoff, &bitoff); \
                TX2 b = *(T2*)(vs2 + regoff * vlenb + byteoff);         \
                c_elem = OP(a, b, c_elem);                              \
            }                                                           \
                                                                        \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

#define DO_MACC(N, M, D) (M * N + D)

/* (TD, T1, T2, TX1, TX2) */
/* TODO: when merging into vector_helper.c, the following are duplicates */
#define OP_SSS_B int8_t, int8_t, int8_t, int8_t, int8_t
#define OP_SSS_H int16_t, int16_t, int16_t, int16_t, int16_t
#define OP_SSS_W int32_t, int32_t, int32_t, int32_t, int32_t
#define OP_SSS_D int64_t, int64_t, int64_t, int64_t, int64_t
#define WOP_SSS_B int16_t, int8_t, int8_t, int16_t, int16_t
#define WOP_SSS_H int32_t, int16_t, int16_t, int32_t, int32_t
#define WOP_SSS_W int64_t, int32_t, int32_t, int64_t, int64_t
/* TODO: when merging into vector_helper.c, the following are new */
#define QWOP_SSS_H int32_t, int8_t, int8_t, int32_t, int32_t
#define QWOP_SSS_W int64_t, int16_t, int16_t, int64_t, int64_t
#define QWOP_UUU_H uint64_t, uint16_t, uint16_t, uint64_t, uint64_t

RVVCALL(GEN_ZVMMA_VMMACC_VV, vmmacc_vv_b, OP_SSS_B, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vmmacc_vv_h, OP_SSS_H, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vmmacc_vv_w, OP_SSS_W, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vmmacc_vv_d, OP_SSS_D, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vwmmacc_vv_h, WOP_SSS_B, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vwmmacc_vv_w, WOP_SSS_H, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vwmmacc_vv_d, WOP_SSS_W, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vqwmmacc_vv_w, QWOP_SSS_H, DO_MACC)
RVVCALL(GEN_ZVMMA_VMMACC_VV, vqwmmacc_vv_d, QWOP_SSS_W, DO_MACC)

static float_status  fstatus;
static float_status *fs = &fstatus;

/* Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
static uint16_t fmacc16(uint16_t a, uint16_t b, uint16_t d)
{
    return float16_muladd(a, b, d, 0, fs);
}

static uint16_t fmacc16bf(uint16_t a, uint16_t b, uint16_t d)
{
    return bfloat16_muladd(a, b, d, 0, fs);
}

static uint32_t fmacc32(uint32_t a, uint32_t b, uint32_t d)
{
    return float32_muladd(a, b, d, 0, fs);
}

static uint64_t fmacc64(uint64_t a, uint64_t b, uint64_t d)
{
    return float64_muladd(a, b, d, 0, fs);
}

/* Vector Widening Floating-Point Fused Multiply-Add Instructions */
static uint32_t fwmacc16(uint16_t a, uint16_t b, uint32_t d)
{
    return float32_muladd(float16_to_float32(a, true, fs),
                          float16_to_float32(b, true, fs), d, 0, fs);
}

static uint32_t fwmacc16bf(uint16_t a, uint16_t b, uint32_t d)
{
    return float32_muladd(bfloat16_to_float32(a, fs),
                          bfloat16_to_float32(b, fs), d, 0, fs);
}

static uint64_t fwmacc32(uint32_t a, uint32_t b, uint64_t d)
{
    return float64_muladd(float32_to_float64(a, fs),
                          float32_to_float64(b, fs), d, 0, fs);
}

static uint64_t fqwmacc16(uint16_t a, uint16_t b, uint64_t d)
{
    return float64_muladd(float16_to_float64(a, true, fs),
                          float16_to_float64(b, true, fs), d, 0, fs);
}

static uint64_t fqwmacc16bf(uint16_t a, uint16_t b, uint64_t d)
{
    return float64_muladd(bfloat16_to_float64(a, fs),
                          bfloat16_to_float64(b, fs), d, 0, fs);
}

RVVCALL(GEN_ZVMMA_VMMACC_VV, vfmmacc_vv_h, OP_UUU_H, fmacc16);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfmmacc_vv_h_altfmt, OP_UUU_H, fmacc16bf);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfmmacc_vv_s, OP_UUU_W, fmacc32);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfmmacc_vv_d, OP_UUU_D, fmacc64);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfwmmacc_vv_s, WOP_UUU_H, fwmacc16);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfwmmacc_vv_s_altfmt, WOP_UUU_H, fwmacc16bf);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfwmmacc_vv_d, WOP_UUU_W, fwmacc32);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfqwmmacc_vv_d, QWOP_UUU_H, fqwmacc16);
RVVCALL(GEN_ZVMMA_VMMACC_VV, vfqwmmacc_vv_d_altfmt, QWOP_UUU_H, fqwmacc16bf);

typedef void gen_helper_vmmacc(TCGv_ptr, TCGv_ptr, TCGv_ptr, TCGv_ptr,
                               TCGv_env, TCGv_i32);

static bool trans_vmmacc_vv(DisasContext *s, arg_rmrr *a)
{
        REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

	    /* TODO(PRT): altfmt can not be evaluated during decode, as it
	     *            may change between different executions of the
	     *            instruction word (i.e., at the same address).
	     */
             static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { gen_helper_vmmacc_vv_b,
                  gen_helper_vmmacc_vv_h,
                  gen_helper_vmmacc_vv_w,
                  gen_helper_vmmacc_vv_d },
                /* altfmt = 1 */
                { NULL, NULL,
                  NULL, NULL }
            };

            const bool  altfmt = false;
            gen_helper_vmmacc *fn = fns[altfmt][s->sew];
            if (fn)
                return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }

        return false;
}

static bool trans_vwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
        REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

            static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { NULL,
                  gen_helper_vwmmacc_vv_h,
                  gen_helper_vwmmacc_vv_w,
                  gen_helper_vwmmacc_vv_d },
                /* altfmt = 1 */
                { NULL, NULL,
                  NULL, NULL }
            };

            const bool altfmt = 0;
            gen_helper_vmmacc *fn = fns[altfmt][s->sew];
            if (fn)
                return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }

        return false;
}

static bool trans_vqwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
        REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

            static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { NULL,
                  NULL,
                  gen_helper_vqwmmacc_vv_w,
                  gen_helper_vqwmmacc_vv_d },
                /* altfmt = 1 */
                { NULL, NULL,
                  NULL, NULL }
            };

            const bool altfmt = 0;
            gen_helper_vmmacc *fn = fns[altfmt][s->sew];
            if (fn)
                return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }

        return false;
}

static bool trans_vfmmacc_vv(DisasContext *s, arg_rmrr *a)
{
        REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

            static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { NULL,
                  gen_helper_vfmmacc_vv_h,
                  gen_helper_vfmmacc_vv_s,
                  gen_helper_vfmmacc_vv_d },
                /* altfmt = 1 */
                { NULL,
		  gen_helper_vfmmacc_vv_h_altfmt,
                  NULL,
		  NULL }
            };

            const bool altfmt = 0;
            gen_helper_vmmacc *fn = fns[altfmt][s->sew];
            if (fn)
                return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }

        return false;
}

static bool trans_vfwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
        REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

            static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { NULL,
                  NULL,
                  gen_helper_vfwmmacc_vv_s,
                  gen_helper_vfwmmacc_vv_d },
                /* altfmt = 1 */
                { NULL,
		  NULL,
                  gen_helper_vfwmmacc_vv_s_altfmt,
		  NULL }
            };

            const bool altfmt = 0;
            gen_helper_vmmacc *fn = fns[altfmt][s->sew];
            if (fn)
                return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }

        return false;
}

static bool trans_vfqwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
        REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

            static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { NULL,
                  NULL,
                  NULL,
                  gen_helper_vfqwmmacc_vv_d },
                /* altfmt = 1 */
                { NULL,
		  NULL,
                  NULL,
		  gen_helper_vfqwmmacc_vv_d_altfmt }
            };

            const bool altfmt = 0;
            gen_helper_vmmacc *fn = fns[altfmt][s->sew];
            if (fn)
                return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }

        return false;
}
