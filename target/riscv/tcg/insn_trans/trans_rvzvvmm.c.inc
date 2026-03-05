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

#include "tcg/vector_internals.h"

/*
 * Generic endianness correction for element offset, dispatched by
 * element size.  On little-endian hosts all H macros are identity.
 */
#define ELEM_H(elemoff, esz)                                            \
    ((esz) == 1 ? H1(elemoff) : (esz) == 2 ? H2(elemoff)              \
                               : (esz) == 4 ? H4(elemoff) : (elemoff))

/*
 * Map a flat tile element index to (register offset, element offset) within
 * a register group.  This implements the spec's tile_reg_idx function:
 *
 *   linesize     = lambda * regmul
 *   line         = flat_idx / linesize
 *   elem_in_line = flat_idx % linesize
 *   regoff       = elem_in_line / lambda
 *   elementoff   = line * lambda + elem_in_line % lambda
 */
static void tile_regoffset(uint32_t flat_idx, uint32_t regmul,
                           uint32_t lambda,
                           uint8_t *regoff, uint16_t *elementoff)
{
    uint32_t linesize     = lambda * regmul;
    uint32_t line         = flat_idx / linesize;
    uint32_t elem_in_line = flat_idx % linesize;

    *regoff     = elem_in_line / lambda;
    *elementoff = line * lambda + elem_in_line % lambda;
}

#define GEN_ZVVMM_VMMACC_VV(NAME, TD, T1, T2, TX1, TX2, OP)            \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint32_t widening = sizeof(TD) / sizeof(T1);                  \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
                                                                        \
    /* Tile geometry (spec formulas) */                                  \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL must be a multiple of K_effective */                           \
    if (vl % k_eff)                                                     \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / k_eff;                                      \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: column-major, flat = j * M + i */                \
            tile_regoffset(j * M + i, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int k = 0; k < k_eff; ++k) {                          \
                /* A[i,k]: row-major, flat = i * k_eff + k */           \
                tile_regoffset(i * k_eff + k, lmul, lambda,            \
                               &regoff, &elemoff);                      \
                TX1 a = ((T1 *)(vs1 + regoff * vlenb))                  \
                         [ELEM_H(elemoff, sizeof(T1))];                 \
                                                                        \
                /* B[k,j]: column-major, flat = j * k_eff + k */        \
                tile_regoffset(j * k_eff + k, lmul, lambda,            \
                               &regoff, &elemoff);                      \
                TX2 b = ((T2 *)(vs2 + regoff * vlenb))                  \
                         [ELEM_H(elemoff, sizeof(T2))];                 \
                                                                        \
                c_elem = OP(a, b, c_elem);                              \
            }                                                           \
                                                                        \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

#define GEN_ZVVMM_VFMMACC_VV(NAME, TD, T1, T2, TX1, TX2, OP)           \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint32_t widening = sizeof(TD) / sizeof(T1);                  \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
    float_status *fs = &env->fp_status;                                 \
                                                                        \
    /* Tile geometry (spec formulas) */                                  \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL must be a multiple of K_effective */                           \
    if (vl % k_eff)                                                     \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / k_eff;                                      \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: column-major, flat = j * M + i */                \
            tile_regoffset(j * M + i, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int k = 0; k < k_eff; ++k) {                          \
                /* A[i,k]: row-major, flat = i * k_eff + k */           \
                tile_regoffset(i * k_eff + k, lmul, lambda,            \
                               &regoff, &elemoff);                      \
                TX1 a = ((T1 *)(vs1 + regoff * vlenb))                  \
                         [ELEM_H(elemoff, sizeof(T1))];                 \
                                                                        \
                /* B[k,j]: column-major, flat = j * k_eff + k */        \
                tile_regoffset(j * k_eff + k, lmul, lambda,            \
                               &regoff, &elemoff);                      \
                TX2 b = ((T2 *)(vs2 + regoff * vlenb))                  \
                         [ELEM_H(elemoff, sizeof(T2))];                 \
                                                                        \
                c_elem = OP(a, b, c_elem, fs);                          \
            }                                                           \
                                                                        \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

/*
 * Extract a raw 4-bit nibble from a tile in the register group.
 * Sub-byte elements are packed two per byte in little-endian nibble order:
 *   even element (2n) in lower nibble [3:0], odd (2n+1) in upper nibble [7:4].
 *
 * tile_reg_idx is applied at 4-bit element granularity; the resulting element
 * offset is then split into byte position and nibble selector.
 */
static uint8_t tile_extract_nibble(void *base, uint16_t vlenb,
                                   uint32_t flat_idx, uint32_t regmul,
                                   uint32_t lambda)
{
    uint32_t linesize     = lambda * regmul;
    uint32_t line         = flat_idx / linesize;
    uint32_t elem_in_line = flat_idx % linesize;

    uint8_t  regoff  = elem_in_line / lambda;
    uint16_t elemoff = line * lambda + elem_in_line % lambda;
    uint16_t byteoff = elemoff / 2;
    uint8_t  nibble  = elemoff % 2;

    uint8_t byte_val = *(uint8_t *)(base + regoff * vlenb + byteoff);
    return nibble ? (byte_val >> 4) : (byte_val & 0xf);
}

/* Extract a signed 4-bit integer (sign-extended). */
static int32_t tile_extract_int4(void *base, uint16_t vlenb,
                                 uint32_t flat_idx, uint32_t regmul,
                                 uint32_t lambda)
{
    uint8_t raw = tile_extract_nibble(base, vlenb, flat_idx, regmul, lambda);
    return (int8_t)((raw ^ 0x8) - 0x8);
}

/* Extract an unsigned 4-bit integer (zero-extended). */
static uint32_t tile_extract_uint4(void *base, uint16_t vlenb,
                                    uint32_t flat_idx, uint32_t regmul,
                                    uint32_t lambda)
{
    return tile_extract_nibble(base, vlenb, flat_idx, regmul, lambda);
}

#define GEN_ZVVMM_VMMACC_VV_4BIT(NAME, TD, TEXTA, TEXTB, EXTA, EXTB, OP) \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint32_t widening = sizeof(TD) * 2; /* SEW bits / 4 bits */   \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
                                                                        \
    /* Tile geometry (spec formulas) */                                  \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL must be a multiple of K_effective */                           \
    if (vl % k_eff)                                                     \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / k_eff;                                      \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: column-major, flat = j * M + i */                \
            tile_regoffset(j * M + i, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int k = 0; k < k_eff; ++k) {                          \
                /* A[i,k]: row-major, flat = i * k_eff + k */           \
                TEXTA a = EXTA(vs1, vlenb,                              \
                    i * k_eff + k, lmul, lambda);                       \
                                                                        \
                /* B[k,j]: column-major, flat = j * k_eff + k */        \
                TEXTB b = EXTB(vs2, vlenb,                              \
                    j * k_eff + k, lmul, lambda);                       \
                                                                        \
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
/* signed A x unsigned B (TD, T1, T2, TX1, TX2) */
#define OP_SSU_B int8_t, int8_t, uint8_t, int8_t, uint8_t
#define OP_SSU_H int16_t, int16_t, uint16_t, int16_t, uint16_t
#define OP_SSU_W int32_t, int32_t, uint32_t, int32_t, uint32_t
#define OP_SSU_D int64_t, int64_t, uint64_t, int64_t, uint64_t
/* unsigned A x signed B */
#define OP_SUS_B int8_t, uint8_t, int8_t, uint8_t, int8_t
#define OP_SUS_H int16_t, uint16_t, int16_t, uint16_t, int16_t
#define OP_SUS_W int32_t, uint32_t, int32_t, uint32_t, int32_t
#define OP_SUS_D int64_t, uint64_t, int64_t, uint64_t, int64_t
/* widening: signed A x unsigned B */
#define WOP_SSU_B int16_t, int8_t, uint8_t, int16_t, uint16_t
#define WOP_SSU_H int32_t, int16_t, uint16_t, int32_t, uint32_t
#define WOP_SSU_W int64_t, int32_t, uint32_t, int64_t, uint64_t
/* widening: unsigned A x signed B */
#define WOP_SUS_B int16_t, uint8_t, int8_t, uint16_t, int16_t
#define WOP_SUS_H int32_t, uint16_t, int16_t, uint32_t, int32_t
#define WOP_SUS_W int64_t, uint32_t, int32_t, uint64_t, int64_t
/* TODO: when merging into vector_helper.c, the following are new */
#define QWOP_SSS_H int32_t, int8_t, int8_t, int32_t, int32_t
#define QWOP_SSS_W int64_t, int16_t, int16_t, int64_t, int64_t
#define QWOP_UUU_B uint32_t, uint8_t, uint8_t, uint32_t, uint32_t
#define QWOP_UUU_H uint64_t, uint16_t, uint16_t, uint64_t, uint64_t
/* quad-widening: signed A x unsigned B */
#define QWOP_SSU_H int32_t, int8_t, uint8_t, int32_t, uint32_t
#define QWOP_SSU_W int64_t, int16_t, uint16_t, int64_t, uint64_t
/* quad-widening: unsigned A x signed B */
#define QWOP_SUS_H int32_t, uint8_t, int8_t, uint32_t, int32_t
#define QWOP_SUS_W int64_t, uint16_t, int16_t, uint64_t, int64_t

/* signed x signed */
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_b, OP_SSS_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_h, OP_SSS_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_w, OP_SSS_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_d, OP_SSS_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_h, WOP_SSS_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_w, WOP_SSS_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_d, WOP_SSS_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_w, QWOP_SSS_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_d, QWOP_SSS_W, DO_MACC)

/* unsigned x unsigned */
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_b_u, OP_UUU_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_h_u, OP_UUU_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_w_u, OP_UUU_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_d_u, OP_UUU_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_h_u, WOP_UUU_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_w_u, WOP_UUU_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_d_u, WOP_UUU_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_w_u, QWOP_UUU_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_d_u, QWOP_UUU_H, DO_MACC)

/* signed A x unsigned B */
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_b_su, OP_SSU_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_h_su, OP_SSU_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_w_su, OP_SSU_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_d_su, OP_SSU_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_h_su, WOP_SSU_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_w_su, WOP_SSU_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_d_su, WOP_SSU_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_w_su, QWOP_SSU_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_d_su, QWOP_SSU_W, DO_MACC)

/* unsigned A x signed B */
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_b_us, OP_SUS_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_h_us, OP_SUS_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_w_us, OP_SUS_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vmmacc_vv_d_us, OP_SUS_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_h_us, WOP_SUS_B, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_w_us, WOP_SUS_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vwmmacc_vv_d_us, WOP_SUS_W, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_w_us, QWOP_SUS_H, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, vqwmmacc_vv_d_us, QWOP_SUS_W, DO_MACC)

/* Int4 variants: 4-bit integer inputs packed two per byte */
/* signed x signed */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vwmmacc_vv_b, int8_t,
        int32_t, int32_t, tile_extract_int4, tile_extract_int4, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vqwmmacc_vv_h, int16_t,
        int32_t, int32_t, tile_extract_int4, tile_extract_int4, DO_MACC)
/* unsigned x unsigned */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vwmmacc_vv_b_u, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vqwmmacc_vv_h_u, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, DO_MACC)
/* signed A x unsigned B */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vwmmacc_vv_b_su, int8_t,
        int32_t, uint32_t, tile_extract_int4, tile_extract_uint4, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vqwmmacc_vv_h_su, int16_t,
        int32_t, uint32_t, tile_extract_int4, tile_extract_uint4, DO_MACC)
/* unsigned A x signed B */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vwmmacc_vv_b_us, int8_t,
        uint32_t, int32_t, tile_extract_uint4, tile_extract_int4, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, vqwmmacc_vv_h_us, int16_t,
        uint32_t, int32_t, tile_extract_uint4, tile_extract_int4, DO_MACC)

/*
 * 4-bit FP MACC: sub-byte elements packed two per byte in little-endian
 * nibble order (same layout as Int4).  The OP callback takes float_status *.
 */
#define GEN_ZVVMM_VFMMACC_VV_4BIT(NAME, TD, TEXTA, TEXTB, EXTA, EXTB, OP) \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint32_t widening = sizeof(TD) * 2; /* SEW bits / 4 bits */   \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
    float_status *fs = &env->fp_status;                                 \
                                                                        \
    /* Tile geometry (spec formulas) */                                  \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL must be a multiple of K_effective */                           \
    if (vl % k_eff)                                                     \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / k_eff;                                      \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: column-major, flat = j * M + i */                \
            tile_regoffset(j * M + i, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int k = 0; k < k_eff; ++k) {                          \
                /* A[i,k]: row-major, flat = i * k_eff + k */           \
                TEXTA a = EXTA(vs1, vlenb,                              \
                    i * k_eff + k, lmul, lambda);                       \
                                                                        \
                /* B[k,j]: column-major, flat = j * k_eff + k */        \
                TEXTB b = EXTB(vs2, vlenb,                              \
                    j * k_eff + k, lmul, lambda);                       \
                                                                        \
                c_elem = OP(a, b, c_elem, fs);                          \
            }                                                           \
                                                                        \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

/* Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
static float8_e4m3 fmacc8_e4m3(uint8_t a, uint8_t b, uint8_t d,
                                float_status *fs)
{
    return float8_e4m3_muladd(a, b, d, 0, fs);
}

static float8_e5m2 fmacc8_e5m2(uint8_t a, uint8_t b, uint8_t d,
                                float_status *fs)
{
    return float8_e5m2_muladd(a, b, d, 0, fs);
}

/* Cross-format: E4M3 inputs, E5M2 accumulator */
static uint8_t fmacc8_e4m3_to_e5m2(uint8_t a, uint8_t b, uint8_t d,
                                     float_status *fs)
{
    return float32_to_float8_e5m2(
        float32_muladd(float8_e4m3_to_float32(a, fs),
                       float8_e4m3_to_float32(b, fs),
                       float8_e5m2_to_float32(d, fs), 0, fs), false, fs);
}

/* Cross-format: E5M2 inputs, E4M3 accumulator */
static uint8_t fmacc8_e5m2_to_e4m3(uint8_t a, uint8_t b, uint8_t d,
                                     float_status *fs)
{
    return float32_to_float8_e4m3(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       float8_e4m3_to_float32(d, fs), 0, fs), false, fs);
}

/* Mixed-input: e4m3 × e5m2 → e4m3 */
static uint8_t fmacc8_e4m3xe5m2(uint8_t a, uint8_t b, uint8_t d,
                                 float_status *fs)
{
    return float32_to_float8_e4m3(
        float32_muladd(float8_e4m3_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       float8_e4m3_to_float32(d, fs), 0, fs), false, fs);
}

/* Mixed-input: e4m3 × e5m2 → e5m2 */
static uint8_t fmacc8_e4m3xe5m2_to_e5m2(uint8_t a, uint8_t b, uint8_t d,
                                          float_status *fs)
{
    return float32_to_float8_e5m2(
        float32_muladd(float8_e4m3_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       float8_e5m2_to_float32(d, fs), 0, fs), false, fs);
}

/* Mixed-input: e5m2 × e4m3 → e4m3 */
static uint8_t fmacc8_e5m2xe4m3(uint8_t a, uint8_t b, uint8_t d,
                                 float_status *fs)
{
    return float32_to_float8_e4m3(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e4m3_to_float32(b, fs),
                       float8_e4m3_to_float32(d, fs), 0, fs), false, fs);
}

/* Mixed-input: e5m2 × e4m3 → e5m2 */
static uint8_t fmacc8_e5m2xe4m3_to_e5m2(uint8_t a, uint8_t b, uint8_t d,
                                          float_status *fs)
{
    return float32_to_float8_e5m2(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e4m3_to_float32(b, fs),
                       float8_e5m2_to_float32(d, fs), 0, fs), false, fs);
}

/* Mixed-input: f16 × bf16 → f16 */
static uint16_t fmacc16_f16xbf16(uint16_t a, uint16_t b, uint16_t d,
                                  float_status *fs)
{
    return float32_to_float16(
        float32_muladd(float16_to_float32(a, true, fs),
                       bfloat16_to_float32(b, fs),
                       float16_to_float32(d, true, fs), 0, fs), true, fs);
}

/* Mixed-input: f16 × bf16 → bf16 */
static uint16_t fmacc16_f16xbf16_to_bf16(uint16_t a, uint16_t b, uint16_t d,
                                           float_status *fs)
{
    return float32_to_bfloat16(
        float32_muladd(float16_to_float32(a, true, fs),
                       bfloat16_to_float32(b, fs),
                       bfloat16_to_float32(d, fs), 0, fs), fs);
}

/* Mixed-input: bf16 × f16 → f16 */
static uint16_t fmacc16_bf16xf16(uint16_t a, uint16_t b, uint16_t d,
                                  float_status *fs)
{
    return float32_to_float16(
        float32_muladd(bfloat16_to_float32(a, fs),
                       float16_to_float32(b, true, fs),
                       float16_to_float32(d, true, fs), 0, fs), true, fs);
}

/* Mixed-input: bf16 × f16 → bf16 */
static uint16_t fmacc16_bf16xf16_to_bf16(uint16_t a, uint16_t b, uint16_t d,
                                           float_status *fs)
{
    return float32_to_bfloat16(
        float32_muladd(bfloat16_to_float32(a, fs),
                       float16_to_float32(b, true, fs),
                       bfloat16_to_float32(d, fs), 0, fs), fs);
}

static uint16_t fmacc16(uint16_t a, uint16_t b, uint16_t d, float_status *fs)
{
    return float16_muladd(a, b, d, 0, fs);
}

static uint16_t fmacc16_bfloat16(uint16_t a, uint16_t b, uint16_t d,
                                  float_status *fs)
{
    return bfloat16_muladd(a, b, d, 0, fs);
}

static uint32_t fmacc32(uint32_t a, uint32_t b, uint32_t d, float_status *fs)
{
    return float32_muladd(a, b, d, 0, fs);
}

static uint64_t fmacc64(uint64_t a, uint64_t b, uint64_t d, float_status *fs)
{
    return float64_muladd(a, b, d, 0, fs);
}

/* Vector Widening Floating-Point Fused Multiply-Add Instructions */
static uint16_t fwmacc8_e4m3(uint8_t a, uint8_t b, uint16_t d,
                              float_status *fs)
{
    return float16_muladd(float8_e4m3_to_float16(a, fs),
                          float8_e4m3_to_float16(b, fs), d, 0, fs);
}

static uint16_t fwmacc8_e5m2(uint8_t a, uint8_t b, uint16_t d,
                              float_status *fs)
{
    return float16_muladd(float8_e5m2_to_float16(a, fs),
                          float8_e5m2_to_float16(b, fs), d, 0, fs);
}

/* Widening FP8 -> BF16 accumulator */
static uint16_t fwmacc8_e4m3_bf16(uint8_t a, uint8_t b, uint16_t d,
                                   float_status *fs)
{
    return float32_to_bfloat16(
        float32_muladd(float8_e4m3_to_float32(a, fs),
                       float8_e4m3_to_float32(b, fs),
                       bfloat16_to_float32(d, fs), 0, fs), fs);
}

static uint16_t fwmacc8_e5m2_bf16(uint8_t a, uint8_t b, uint16_t d,
                                   float_status *fs)
{
    return float32_to_bfloat16(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       bfloat16_to_float32(d, fs), 0, fs), fs);
}

/* Mixed-input: e4m3 × e5m2 → f16 */
static uint16_t fwmacc8_e4m3xe5m2(uint8_t a, uint8_t b, uint16_t d,
                                   float_status *fs)
{
    return float16_muladd(float8_e4m3_to_float16(a, fs),
                          float8_e5m2_to_float16(b, fs), d, 0, fs);
}

/* Mixed-input: e5m2 × e4m3 → f16 */
static uint16_t fwmacc8_e5m2xe4m3(uint8_t a, uint8_t b, uint16_t d,
                                   float_status *fs)
{
    return float16_muladd(float8_e5m2_to_float16(a, fs),
                          float8_e4m3_to_float16(b, fs), d, 0, fs);
}

/* Mixed-input: e4m3 × e5m2 → bf16 */
static uint16_t fwmacc8_e4m3xe5m2_bf16(uint8_t a, uint8_t b, uint16_t d,
                                        float_status *fs)
{
    return float32_to_bfloat16(
        float32_muladd(float8_e4m3_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       bfloat16_to_float32(d, fs), 0, fs), fs);
}

/* Mixed-input: e5m2 × e4m3 → bf16 */
static uint16_t fwmacc8_e5m2xe4m3_bf16(uint8_t a, uint8_t b, uint16_t d,
                                        float_status *fs)
{
    return float32_to_bfloat16(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e4m3_to_float32(b, fs),
                       bfloat16_to_float32(d, fs), 0, fs), fs);
}

static uint32_t fwmacc16(uint16_t a, uint16_t b, uint32_t d, float_status *fs)
{
    return float32_muladd(float16_to_float32(a, true, fs),
                          float16_to_float32(b, true, fs), d, 0, fs);
}

static uint32_t fwmacc16_bfloat16(uint16_t a, uint16_t b, uint32_t d,
                                   float_status *fs)
{
    return float32_muladd(bfloat16_to_float32(a, fs),
                          bfloat16_to_float32(b, fs), d, 0, fs);
}

/* Mixed-input: f16 × bf16 → f32 */
static uint32_t fwmacc16_f16xbf16(uint16_t a, uint16_t b, uint32_t d,
                                   float_status *fs)
{
    return float32_muladd(float16_to_float32(a, true, fs),
                          bfloat16_to_float32(b, fs), d, 0, fs);
}

/* Mixed-input: bf16 × f16 → f32 */
static uint32_t fwmacc16_bf16xf16(uint16_t a, uint16_t b, uint32_t d,
                                   float_status *fs)
{
    return float32_muladd(bfloat16_to_float32(a, fs),
                          float16_to_float32(b, true, fs), d, 0, fs);
}

static uint64_t fwmacc32(uint32_t a, uint32_t b, uint64_t d, float_status *fs)
{
    return float64_muladd(float32_to_float64(a, fs),
                          float32_to_float64(b, fs), d, 0, fs);
}

static uint32_t fqwmacc8_e4m3(uint8_t a, uint8_t b, uint32_t d,
                               float_status *fs)
{
    return float32_muladd(float8_e4m3_to_float32(a, fs),
                          float8_e4m3_to_float32(b, fs), d, 0, fs);
}

static uint32_t fqwmacc8_e5m2(uint8_t a, uint8_t b, uint32_t d,
                               float_status *fs)
{
    return float32_muladd(float8_e5m2_to_float32(a, fs),
                          float8_e5m2_to_float32(b, fs), d, 0, fs);
}

/* Mixed-input: e4m3 × e5m2 → f32 */
static uint32_t fqwmacc8_e4m3xe5m2(uint8_t a, uint8_t b, uint32_t d,
                                    float_status *fs)
{
    return float32_muladd(float8_e4m3_to_float32(a, fs),
                          float8_e5m2_to_float32(b, fs), d, 0, fs);
}

/* Mixed-input: e5m2 × e4m3 → f32 */
static uint32_t fqwmacc8_e5m2xe4m3(uint8_t a, uint8_t b, uint32_t d,
                                    float_status *fs)
{
    return float32_muladd(float8_e5m2_to_float32(a, fs),
                          float8_e4m3_to_float32(b, fs), d, 0, fs);
}

static uint64_t fqwmacc16(uint16_t a, uint16_t b, uint64_t d, float_status *fs)
{
    return float64_muladd(float16_to_float64(a, true, fs),
                          float16_to_float64(b, true, fs), d, 0, fs);
}

static uint64_t fqwmacc16bf(uint16_t a, uint16_t b, uint64_t d,
                             float_status *fs)
{
    return float64_muladd(bfloat16_to_float64(a, fs),
                          bfloat16_to_float64(b, fs), d, 0, fs);
}

/* Mixed-input: f16 × bf16 → f64 */
static uint64_t fqwmacc16_f16xbf16(uint16_t a, uint16_t b, uint64_t d,
                                    float_status *fs)
{
    return float64_muladd(float16_to_float64(a, true, fs),
                          bfloat16_to_float64(b, fs), d, 0, fs);
}

/* Mixed-input: bf16 × f16 → f64 */
static uint64_t fqwmacc16_bf16xf16(uint16_t a, uint16_t b, uint64_t d,
                                    float_status *fs)
{
    return float64_muladd(bfloat16_to_float64(a, fs),
                          float16_to_float64(b, true, fs), d, 0, fs);
}

RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e4m3, OP_UUU_B, fmacc8_e4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e5m2, OP_UUU_B, fmacc8_e5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e5m2_from_e4m3, OP_UUU_B,
        fmacc8_e4m3_to_e5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e4m3_from_e5m2, OP_UUU_B,
        fmacc8_e5m2_to_e4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_f16,  OP_UUU_H, fmacc16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_bf16, OP_UUU_H, fmacc16_bfloat16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_f32,  OP_UUU_W, fmacc32);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_f64,  OP_UUU_D, fmacc64);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e4m3_e4m3_e5m2, OP_UUU_B,
        fmacc8_e4m3xe5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e5m2_e4m3_e5m2, OP_UUU_B,
        fmacc8_e4m3xe5m2_to_e5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e4m3_e5m2_e4m3, OP_UUU_B,
        fmacc8_e5m2xe4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_e5m2_e5m2_e4m3, OP_UUU_B,
        fmacc8_e5m2xe4m3_to_e5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_f16_f16_bf16, OP_UUU_H,
        fmacc16_f16xbf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_bf16_f16_bf16, OP_UUU_H,
        fmacc16_f16xbf16_to_bf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_f16_bf16_f16, OP_UUU_H,
        fmacc16_bf16xf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfmmacc_vv_bf16_bf16_f16, OP_UUU_H,
        fmacc16_bf16xf16_to_bf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f16_e4m3, WOP_UUU_B, fwmacc8_e4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f16_e5m2, WOP_UUU_B, fwmacc8_e5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_bf16_e4m3, WOP_UUU_B,
        fwmacc8_e4m3_bf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_bf16_e5m2, WOP_UUU_B,
        fwmacc8_e5m2_bf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f32_f16, WOP_UUU_H, fwmacc16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f32_bf16, WOP_UUU_H,
        fwmacc16_bfloat16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f64_f32, WOP_UUU_W, fwmacc32);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f16_e4m3_e5m2, WOP_UUU_B,
        fwmacc8_e4m3xe5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f16_e5m2_e4m3, WOP_UUU_B,
        fwmacc8_e5m2xe4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_bf16_e4m3_e5m2, WOP_UUU_B,
        fwmacc8_e4m3xe5m2_bf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_bf16_e5m2_e4m3, WOP_UUU_B,
        fwmacc8_e5m2xe4m3_bf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f32_f16_bf16, WOP_UUU_H,
        fwmacc16_f16xbf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfwmmacc_vv_f32_bf16_f16, WOP_UUU_H,
        fwmacc16_bf16xf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f32_e4m3, QWOP_UUU_B,
        fqwmacc8_e4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f32_e5m2, QWOP_UUU_B,
        fqwmacc8_e5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f64_f16, QWOP_UUU_H, fqwmacc16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f64_bf16, QWOP_UUU_H,
        fqwmacc16bf);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f32_e4m3_e5m2, QWOP_UUU_B,
        fqwmacc8_e4m3xe5m2);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f32_e5m2_e4m3, QWOP_UUU_B,
        fqwmacc8_e5m2xe4m3);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f64_f16_bf16, QWOP_UUU_H,
        fqwmacc16_f16xbf16);
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vfqwmmacc_vv_f64_bf16_f16, QWOP_UUU_H,
        fqwmacc16_bf16xf16);

/* Widening FP4 E2M1 -> FP8 E4M3 */
static uint8_t fwmacc4_e2m1(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                             float_status *fs)
{
    return float8_e4m3_muladd(
        float4_e2m1_to_float8_e4m3(make_float4_e2m1(a_raw), fs),
        float4_e2m1_to_float8_e4m3(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Widening FP4 E3M0 -> FP8 E4M3 */
static uint8_t fwmacc4_e3m0(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                             float_status *fs)
{
    return float8_e4m3_muladd(
        float4_e3m0_to_float8_e4m3(make_float4_e3m0(a_raw), fs),
        float4_e3m0_to_float8_e4m3(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Widening FP4 E2M1 -> FP8 E5M2 */
static uint8_t fwmacc4_e2m1_e5m2(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                  float_status *fs)
{
    return float8_e5m2_muladd(
        float4_e2m1_to_float8_e5m2(make_float4_e2m1(a_raw), fs),
        float4_e2m1_to_float8_e5m2(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Widening FP4 E3M0 -> FP8 E5M2 */
static uint8_t fwmacc4_e3m0_e5m2(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                  float_status *fs)
{
    return float8_e5m2_muladd(
        float4_e3m0_to_float8_e5m2(make_float4_e3m0(a_raw), fs),
        float4_e3m0_to_float8_e5m2(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input widening: e2m1 × e3m0 → e4m3 */
static uint8_t fwmacc4_e2m1xe3m0(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                  float_status *fs)
{
    return float8_e4m3_muladd(
        float4_e2m1_to_float8_e4m3(make_float4_e2m1(a_raw), fs),
        float4_e3m0_to_float8_e4m3(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input widening: e3m0 × e2m1 → e4m3 */
static uint8_t fwmacc4_e3m0xe2m1(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                  float_status *fs)
{
    return float8_e4m3_muladd(
        float4_e3m0_to_float8_e4m3(make_float4_e3m0(a_raw), fs),
        float4_e2m1_to_float8_e4m3(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input widening: e2m1 × e3m0 → e5m2 */
static uint8_t fwmacc4_e2m1xe3m0_e5m2(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                       float_status *fs)
{
    return float8_e5m2_muladd(
        float4_e2m1_to_float8_e5m2(make_float4_e2m1(a_raw), fs),
        float4_e3m0_to_float8_e5m2(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input widening: e3m0 × e2m1 → e5m2 */
static uint8_t fwmacc4_e3m0xe2m1_e5m2(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                       float_status *fs)
{
    return float8_e5m2_muladd(
        float4_e3m0_to_float8_e5m2(make_float4_e3m0(a_raw), fs),
        float4_e2m1_to_float8_e5m2(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Quad-widening FP4 E2M1 -> FP16 */
static uint16_t fqwmacc4_e2m1(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                               float_status *fs)
{
    return float16_muladd(
        float4_e2m1_to_float16(make_float4_e2m1(a_raw), fs),
        float4_e2m1_to_float16(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Quad-widening FP4 E3M0 -> FP16 */
static uint16_t fqwmacc4_e3m0(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                               float_status *fs)
{
    return float16_muladd(
        float4_e3m0_to_float16(make_float4_e3m0(a_raw), fs),
        float4_e3m0_to_float16(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Quad-widening FP4 E2M1 -> BF16 */
static uint16_t fqwmacc4_e2m1_bf16(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                                    float_status *fs)
{
    return bfloat16_muladd(
        float4_e2m1_to_bfloat16(make_float4_e2m1(a_raw), fs),
        float4_e2m1_to_bfloat16(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Quad-widening FP4 E3M0 -> BF16 */
static uint16_t fqwmacc4_e3m0_bf16(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                                    float_status *fs)
{
    return bfloat16_muladd(
        float4_e3m0_to_bfloat16(make_float4_e3m0(a_raw), fs),
        float4_e3m0_to_bfloat16(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input quad-widening: e2m1 × e3m0 → f16 */
static uint16_t fqwmacc4_e2m1xe3m0(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                                    float_status *fs)
{
    return float16_muladd(
        float4_e2m1_to_float16(make_float4_e2m1(a_raw), fs),
        float4_e3m0_to_float16(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input quad-widening: e3m0 × e2m1 → f16 */
static uint16_t fqwmacc4_e3m0xe2m1(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                                    float_status *fs)
{
    return float16_muladd(
        float4_e3m0_to_float16(make_float4_e3m0(a_raw), fs),
        float4_e2m1_to_float16(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input quad-widening: e2m1 × e3m0 → bf16 */
static uint16_t fqwmacc4_e2m1xe3m0_bf16(uint32_t a_raw, uint32_t b_raw,
                                          uint16_t d, float_status *fs)
{
    return bfloat16_muladd(
        float4_e2m1_to_bfloat16(make_float4_e2m1(a_raw), fs),
        float4_e3m0_to_bfloat16(make_float4_e3m0(b_raw), fs),
        d, 0, fs);
}

/* Mixed-input quad-widening: e3m0 × e2m1 → bf16 */
static uint16_t fqwmacc4_e3m0xe2m1_bf16(uint32_t a_raw, uint32_t b_raw,
                                          uint16_t d, float_status *fs)
{
    return bfloat16_muladd(
        float4_e3m0_to_bfloat16(make_float4_e3m0(a_raw), fs),
        float4_e2m1_to_bfloat16(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}
/* FP4 widening: 4-bit FP -> 8-bit FP accumulate */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e4m3_e2m1, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, fwmacc4_e2m1)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e4m3_e3m0, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, fwmacc4_e3m0)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e5m2_e2m1, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e2m1_e5m2)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e5m2_e3m0, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e3m0_e5m2)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e4m3_e2m1_e3m0, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e2m1xe3m0)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e5m2_e2m1_e3m0, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e2m1xe3m0_e5m2)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e4m3_e3m0_e2m1, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e3m0xe2m1)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e5m2_e3m0_e2m1, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e3m0xe2m1_e5m2)

/* FP4 quad-widening: 4-bit FP -> 16-bit FP accumulate */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_f16_e2m1, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, fqwmacc4_e2m1)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_f16_e3m0, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, fqwmacc4_e3m0)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_bf16_e2m1, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e2m1_bf16)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_bf16_e3m0, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e3m0_bf16)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_f16_e2m1_e3m0, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e2m1xe3m0)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_bf16_e2m1_e3m0, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e2m1xe3m0_bf16)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_f16_e3m0_e2m1, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e3m0xe2m1)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_bf16_e3m0_e2m1, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e3m0xe2m1_bf16)

typedef void gen_helper_vmmacc(TCGv_ptr, TCGv_ptr, TCGv_ptr, TCGv_ptr,
                               TCGv_env, TCGv_i32);

/*
 * Compute log2(mul_c) for the accumulator tile register group.
 * mul_c = vlenb / (sewb * lambda^2).  Returns -1 if the configuration
 * is illegal (vlambda == 0 or mul_c not in {1, 2, 4, 8, 16}).
 */
static int8_t zvvmm_mul_c_log2(DisasContext *s)
{
    if (s->vlambda == 0) {
        return -1;
    }

    uint32_t lambda_log2 = s->vlambda - 1;
    uint32_t vlenb_log2 = ctz32(s->cfg_ptr->vlenb);
    int32_t mul_c_log2 = vlenb_log2 - s->sew - 2 * lambda_log2;

    if (mul_c_log2 < 0 || mul_c_log2 > 4) {
        return -1;
    }
    return (int8_t)mul_c_log2;
}

static bool zvvmm_ext_check(DisasContext *s, uint32_t ext_offset)
{
    if (ext_offset == 0) {
        return false;
    }
    return *(const bool *)((const char *)s->cfg_ptr + ext_offset);
}

static bool zvvmm_macc_trans(DisasContext *s, arg_rmrr *a,
                             gen_helper_vmmacc * const fns[2][2][4],
                             const uint32_t ext_offsets[4])
{
    if (!zvvmm_ext_check(s, ext_offsets[s->sew])) {
        return false;
    }

    int8_t mul_c_log2 = zvvmm_mul_c_log2(s);

    if (mul_c_log2 >= 0 && s->lmul >= 0 && vext_check_isa_ill(s) &&
        require_align(a->rd, mul_c_log2) &&
        require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

        gen_helper_vmmacc *fn = fns[s->altfmt_a][s->altfmt_b][s->sew];
        if (fn) {
            return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }
    }

    return false;
}

/*
 * FP MACC dispatch tables: fns[altfmt_a][altfmt_b][altfmt][sew]
 *
 * For FP MACC instructions, altfmt_a and altfmt_b select the input FP
 * formats (which may differ for mixed-input operations), while altfmt
 * independently selects the output/accumulator format.
 */
static bool zvvmm_fmacc_trans(DisasContext *s, arg_rmrr *a,
                              gen_helper_vmmacc * const fns[2][2][2][4],
                              const uint32_t ext_offsets[2][2][4])
{
    if (!zvvmm_ext_check(s, ext_offsets[s->altfmt_a][s->altfmt][s->sew])) {
        return false;
    }
    if (!zvvmm_ext_check(s, ext_offsets[s->altfmt_b][s->altfmt][s->sew])) {
        return false;
    }

    int8_t mul_c_log2 = zvvmm_mul_c_log2(s);

    if (mul_c_log2 >= 0 && s->lmul >= 0 && vext_check_isa_ill(s) &&
        require_align(a->rd, mul_c_log2) &&
        require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

        gen_helper_vmmacc *fn = fns[s->altfmt_a][s->altfmt_b][s->altfmt][s->sew];
        if (fn) {
            return opivv_trans(a->rd, a->rs1, a->rs2, a->vm, fn, s);
        }
    }

    return false;
}

static bool trans_vmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0: signed x signed */
            { gen_helper_vmmacc_vv_b,
              gen_helper_vmmacc_vv_h,
              gen_helper_vmmacc_vv_w,
              gen_helper_vmmacc_vv_d },
            /* altfmt_b = 1: signed x unsigned */
            { gen_helper_vmmacc_vv_b_su,
              gen_helper_vmmacc_vv_h_su,
              gen_helper_vmmacc_vv_w_su,
              gen_helper_vmmacc_vv_d_su }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0: unsigned x signed */
            { gen_helper_vmmacc_vv_b_us,
              gen_helper_vmmacc_vv_h_us,
              gen_helper_vmmacc_vv_w_us,
              gen_helper_vmmacc_vv_d_us },
            /* altfmt_b = 1: unsigned x unsigned */
            { gen_helper_vmmacc_vv_b_u,
              gen_helper_vmmacc_vv_h_u,
              gen_helper_vmmacc_vv_w_u,
              gen_helper_vmmacc_vv_d_u }
        }
    };
    static const uint32_t ext_offsets[4] = {
        CPU_CFG_OFFSET(ext_zvvmmb), CPU_CFG_OFFSET(ext_zvvmmh),
        CPU_CFG_OFFSET(ext_zvvmmw), CPU_CFG_OFFSET(ext_zvvmmd)
    };
    return zvvmm_macc_trans(s, a, fns, ext_offsets);
}

static bool trans_vwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0: signed x signed */
            { gen_helper_vwmmacc_vv_b,
              gen_helper_vwmmacc_vv_h,
              gen_helper_vwmmacc_vv_w,
              gen_helper_vwmmacc_vv_d },
            /* altfmt_b = 1: signed x unsigned */
            { gen_helper_vwmmacc_vv_b_su,
              gen_helper_vwmmacc_vv_h_su,
              gen_helper_vwmmacc_vv_w_su,
              gen_helper_vwmmacc_vv_d_su }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0: unsigned x signed */
            { gen_helper_vwmmacc_vv_b_us,
              gen_helper_vwmmacc_vv_h_us,
              gen_helper_vwmmacc_vv_w_us,
              gen_helper_vwmmacc_vv_d_us },
            /* altfmt_b = 1: unsigned x unsigned */
            { gen_helper_vwmmacc_vv_b_u,
              gen_helper_vwmmacc_vv_h_u,
              gen_helper_vwmmacc_vv_w_u,
              gen_helper_vwmmacc_vv_d_u }
        }
    };
    static const uint32_t ext_offsets[4] = {
        CPU_CFG_OFFSET(ext_zvvmmi4b), CPU_CFG_OFFSET(ext_zvvmmbh),
        CPU_CFG_OFFSET(ext_zvvmmhw),  CPU_CFG_OFFSET(ext_zvvmmwd)
    };
    return zvvmm_macc_trans(s, a, fns, ext_offsets);
}

static bool trans_vqwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0: signed x signed */
            { NULL,
              gen_helper_vqwmmacc_vv_h,
              gen_helper_vqwmmacc_vv_w,
              gen_helper_vqwmmacc_vv_d },
            /* altfmt_b = 1: signed x unsigned */
            { NULL,
              gen_helper_vqwmmacc_vv_h_su,
              gen_helper_vqwmmacc_vv_w_su,
              gen_helper_vqwmmacc_vv_d_su }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0: unsigned x signed */
            { NULL,
              gen_helper_vqwmmacc_vv_h_us,
              gen_helper_vqwmmacc_vv_w_us,
              gen_helper_vqwmmacc_vv_d_us },
            /* altfmt_b = 1: unsigned x unsigned */
            { NULL,
              gen_helper_vqwmmacc_vv_h_u,
              gen_helper_vqwmmacc_vv_w_u,
              gen_helper_vqwmmacc_vv_d_u }
        }
    };
    static const uint32_t ext_offsets[4] = {
        0,                            CPU_CFG_OFFSET(ext_zvvmmi4h),
        CPU_CFG_OFFSET(ext_zvvmmbw),  CPU_CFG_OFFSET(ext_zvvmmhd)
    };
    return zvvmm_macc_trans(s, a, fns, ext_offsets);
}
static bool trans_vfmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 (same: e4m3/f16/f32/f64) */
            {
                { gen_helper_vfmmacc_vv_e4m3, gen_helper_vfmmacc_vv_f16,
                  gen_helper_vfmmacc_vv_f32,  gen_helper_vfmmacc_vv_f64 },
                { gen_helper_vfmmacc_vv_e5m2_from_e4m3, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 (mixed: A=e4m3/f16, B=e5m2/bf16) */
            {
                { gen_helper_vfmmacc_vv_e4m3_e4m3_e5m2,
                  gen_helper_vfmmacc_vv_f16_f16_bf16, NULL, NULL },
                { gen_helper_vfmmacc_vv_e5m2_e4m3_e5m2,
                  gen_helper_vfmmacc_vv_bf16_f16_bf16, NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (mixed: A=e5m2/bf16, B=e4m3/f16) */
            {
                { gen_helper_vfmmacc_vv_e4m3_e5m2_e4m3,
                  gen_helper_vfmmacc_vv_f16_bf16_f16, NULL, NULL },
                { gen_helper_vfmmacc_vv_e5m2_e5m2_e4m3,
                  gen_helper_vfmmacc_vv_bf16_bf16_f16, NULL, NULL }
            },
            /* altfmt_b = 1 (same: e5m2/bf16) */
            {
                { gen_helper_vfmmacc_vv_e4m3_from_e5m2, NULL, NULL, NULL },
                { gen_helper_vfmmacc_vv_e5m2, gen_helper_vfmmacc_vv_bf16,
                  NULL, NULL }
            }
        }
    };
    static const uint32_t ext_offsets[2][2][4] = {
        { { CPU_CFG_OFFSET(ext_zvvfmmofp8), CPU_CFG_OFFSET(ext_zvvfmmh),
            CPU_CFG_OFFSET(ext_zvvfmmf),    CPU_CFG_OFFSET(ext_zvvfmmd) },
          { CPU_CFG_OFFSET(ext_zvvfmmofp8), CPU_CFG_OFFSET(ext_zvvfmmh),
            0, 0 } },
        { { CPU_CFG_OFFSET(ext_zvvfmmofp8), CPU_CFG_OFFSET(ext_zvvfmmbf16),
            0, 0 },
          { CPU_CFG_OFFSET(ext_zvvfmmofp8), CPU_CFG_OFFSET(ext_zvvfmmbf16),
            0, 0 } }
    };
    return zvvmm_fmacc_trans(s, a, fns, ext_offsets);
}

static bool trans_vfwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 (same) */
            {
                { gen_helper_vfwmmacc_vv_e4m3_e2m1,
                  gen_helper_vfwmmacc_vv_f16_e4m3,
                  gen_helper_vfwmmacc_vv_f32_f16,
                  gen_helper_vfwmmacc_vv_f64_f32 },
                { gen_helper_vfwmmacc_vv_e5m2_e2m1,
                  gen_helper_vfwmmacc_vv_bf16_e4m3,
                  NULL, NULL }
            },
            /* altfmt_b = 1 (mixed: A=e2m1/e4m3/f16, B=e3m0/e5m2/bf16) */
            {
                { gen_helper_vfwmmacc_vv_e4m3_e2m1_e3m0,
                  gen_helper_vfwmmacc_vv_f16_e4m3_e5m2,
                  gen_helper_vfwmmacc_vv_f32_f16_bf16, NULL },
                { gen_helper_vfwmmacc_vv_e5m2_e2m1_e3m0,
                  gen_helper_vfwmmacc_vv_bf16_e4m3_e5m2,
                  NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (mixed: A=e3m0/e5m2/bf16, B=e2m1/e4m3/f16) */
            {
                { gen_helper_vfwmmacc_vv_e4m3_e3m0_e2m1,
                  gen_helper_vfwmmacc_vv_f16_e5m2_e4m3,
                  gen_helper_vfwmmacc_vv_f32_bf16_f16, NULL },
                { gen_helper_vfwmmacc_vv_e5m2_e3m0_e2m1,
                  gen_helper_vfwmmacc_vv_bf16_e5m2_e4m3,
                  NULL, NULL }
            },
            /* altfmt_b = 1 (same) */
            {
                { gen_helper_vfwmmacc_vv_e4m3_e3m0,
                  gen_helper_vfwmmacc_vv_f16_e5m2,
                  gen_helper_vfwmmacc_vv_f32_bf16, NULL },
                { gen_helper_vfwmmacc_vv_e5m2_e3m0,
                  gen_helper_vfwmmacc_vv_bf16_e5m2,
                  NULL, NULL }
            }
        }
    };
    static const uint32_t ext_offsets[2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt = 0 */
            { CPU_CFG_OFFSET(ext_zvvfmmofp4opf8), CPU_CFG_OFFSET(ext_zvvfmmofp8h),
              CPU_CFG_OFFSET(ext_zvvfmmhf),       CPU_CFG_OFFSET(ext_zvvfmmfd) },
            /* altfmt = 1 */
            { CPU_CFG_OFFSET(ext_zvvfmmofp4opf8), CPU_CFG_OFFSET(ext_zvvfmmofp8bf16),
              0, 0 }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt = 0 */
            { CPU_CFG_OFFSET(ext_zvvfmmofp4opf8), CPU_CFG_OFFSET(ext_zvvfmmofp8h),
              CPU_CFG_OFFSET(ext_zvvfmmbf16f),    0 },
            /* altfmt = 1 */
            { CPU_CFG_OFFSET(ext_zvvfmmofp4opf8), CPU_CFG_OFFSET(ext_zvvfmmofp8bf16),
              0, 0 }
        }
    };
    return zvvmm_fmacc_trans(s, a, fns, ext_offsets);
}

static bool trans_vfqwmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 (same) */
            {
                { NULL, gen_helper_vfqwmmacc_vv_f16_e2m1,
                  gen_helper_vfqwmmacc_vv_f32_e4m3,
                  gen_helper_vfqwmmacc_vv_f64_f16 },
                { NULL, gen_helper_vfqwmmacc_vv_bf16_e2m1, NULL, NULL }
            },
            /* altfmt_b = 1 (mixed) */
            {
                { NULL, gen_helper_vfqwmmacc_vv_f16_e2m1_e3m0,
                  gen_helper_vfqwmmacc_vv_f32_e4m3_e5m2,
                  gen_helper_vfqwmmacc_vv_f64_f16_bf16 },
                { NULL, gen_helper_vfqwmmacc_vv_bf16_e2m1_e3m0, NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (mixed) */
            {
                { NULL, gen_helper_vfqwmmacc_vv_f16_e3m0_e2m1,
                  gen_helper_vfqwmmacc_vv_f32_e5m2_e4m3,
                  gen_helper_vfqwmmacc_vv_f64_bf16_f16 },
                { NULL, gen_helper_vfqwmmacc_vv_bf16_e3m0_e2m1, NULL, NULL }
            },
            /* altfmt_b = 1 (same) */
            {
                { NULL, gen_helper_vfqwmmacc_vv_f16_e3m0,
                  gen_helper_vfqwmmacc_vv_f32_e5m2,
                  gen_helper_vfqwmmacc_vv_f64_bf16 },
                { NULL, gen_helper_vfqwmmacc_vv_bf16_e3m0, NULL, NULL }
            }
        }
    };
    static const uint32_t ext_offsets[2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt = 0 */
            { 0, CPU_CFG_OFFSET(ext_zvvfmmofp4h),
              CPU_CFG_OFFSET(ext_zvvfmmofp8w), CPU_CFG_OFFSET(ext_zvvfmmhd) },
            /* altfmt = 1 */
            { 0, CPU_CFG_OFFSET(ext_zvvfmmofp4bf16), 0, 0 }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt = 0 */
            { 0, CPU_CFG_OFFSET(ext_zvvfmmofp4h),
              CPU_CFG_OFFSET(ext_zvvfmmofp8w), CPU_CFG_OFFSET(ext_zvvfmmbf16d) },
            /* altfmt = 1 */
            { 0, CPU_CFG_OFFSET(ext_zvvfmmofp4bf16), 0, 0 }
        }
    };
    return zvvmm_fmacc_trans(s, a, fns, ext_offsets);
}
