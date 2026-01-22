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

#define GEN_ZVMMA_VMMACC_VV(NAME, TD, T1, T2, TX1, TX2, WIDENING)       \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint8_t  widening = WIDENING;                                 \
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
            TD t = 0;                                                   \
                                                                        \
            element_to_regoffset(r, c, dir, lambda, sewb, 1, &regoff, &byteoff, &bitoff); \
            TD *c_elem = (TD*)(vd + regoff * vlenb + byteoff);          \
                                                                        \
            /* Calculate T[r, c] */                                     \
            for (int k = 0; k < k_eff; ++k)                             \
            {                                                           \
                element_to_regoffset(r, k, ROW_MAJOR, lambda, sewb, widening, &regoff, &byteoff, &bitoff); \
                TX1 a = *(T1*)(vs1 + regoff * vlenb + byteoff);         \
                element_to_regoffset(k, c, COLUMN_MAJOR, lambda, sewb, widening, &regoff, &byteoff, &bitoff); \
                TX2 b = *(T2*)(vs2 + regoff * vlenb + byteoff);         \
                t += a * b;                                             \
            }                                                           \
                                                                        \
            /* Accumulate T[r,c] into C[r,c] */                         \
            *c_elem += t;                                               \
        }                                                               \
    }                                                                   \
}

GEN_ZVMMA_VMMACC_VV(vmmacc_vv_b, int8_t, int8_t, int8_t, int8_t, int8_t, 1);
GEN_ZVMMA_VMMACC_VV(vmmacc_vv_h, int16_t, int16_t, int16_t, int16_t, int16_t, 1);
GEN_ZVMMA_VMMACC_VV(vmmacc_vv_w, int32_t, int32_t, int32_t, int32_t, int32_t, 1);
GEN_ZVMMA_VMMACC_VV(vmmacc_vv_d, int64_t, int64_t, int64_t, int64_t, int64_t, 1);
GEN_ZVMMA_VMMACC_VV(vwmmacc_vv_h, int16_t, int8_t, int8_t, int16_t, int16_t, 2);
GEN_ZVMMA_VMMACC_VV(vwmmacc_vv_w, int32_t, int16_t, int16_t, int32_t, int32_t, 2);
GEN_ZVMMA_VMMACC_VV(vwmmacc_vv_d, int64_t, int32_t, int32_t, int64_t, int64_t, 2);
GEN_ZVMMA_VMMACC_VV(vqwmmacc_vv_w, int32_t, int8_t, int8_t, int32_t, int32_t, 4);
GEN_ZVMMA_VMMACC_VV(vqwmmacc_vv_d, int64_t, int16_t, int16_t, int64_t, int64_t, 4);

GEN_ZVMMA_VMMACC_VV(vfmmacc_vv_s, float, float, float, float, float, 1);
GEN_ZVMMA_VMMACC_VV(vfmmacc_vv_d, double, double, double, double, double, 1);
GEN_ZVMMA_VMMACC_VV(vfwmmacc_vv_d, double, float, float, double, double, 2);

typedef void gen_helper_vmmacc(TCGv_ptr, TCGv_ptr, TCGv_ptr, TCGv_ptr,
                               TCGv_env, TCGv_i32);

static bool trans_vmmacc_vv(DisasContext *s, arg_rmrr *a)
{
	REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

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
                  NULL,
		  gen_helper_vfmmacc_vv_s,
                  gen_helper_vfmmacc_vv_d },
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

static bool trans_vfwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
	REQUIRE_ZVMMA(s);

        if (vext_check_isa_ill(s) &&
            require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

            static gen_helper_vmmacc * const fns[2][4] = {
                /* altfmt = 0 */
                { NULL,
                  NULL,
		  NULL,
                  gen_helper_vfwmmacc_vv_d },
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

static bool trans_vfqwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    return false;
}
