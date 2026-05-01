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
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL counts B SEW-slots; one B/C column = lambda*LMUL VL units */   \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
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

/*
 * GEN_ZVVMM_VFMMACC_VV_MX — like GEN_ZVVMM_VFMMACC_VV but for vm=0
 * (microscaling).  For each output element (i,j), the combined E8M0 scale
 * factor is read from v0[i*R + j] (16-bit: scale_A=[7:0], scale_B=[15:8]),
 * converted to a combined float32 scale, and applied to the inner product
 * before accumulating:
 *
 *   ip = 0; for k: ip = OP(a[k], b[k], ip, fs)  (inner product in type TD)
 *   scale_td = SCALE_TO_TD(scale_A_f32 × scale_B_f32)
 *   *c_ptr = FMA_TD(scale_td, ip, *c_ptr, 0, fs)  (scaled add-to-acc)
 *
 * SCALE_TO_TD(f32, fs): converts float32 combined scale to type TD.
 * FMA_TD(a, b, c, flags, fs): the output-type FMA function.
 */
#define GEN_ZVVMM_VFMMACC_VV_MX(NAME, TD, T1, T2, TX1, TX2, OP,        \
                                  SCALE_TO_TD, FMA_TD)                   \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint32_t widening = sizeof(TD) / sizeof(T1);                  \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint32_t SEW_bits = 8u << vsew;                               \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
    const uint8_t  bs = FIELD_EX64(env->vtype, VTYPE, BS);             \
    float_status *fs = &env->fp_status;                                 \
                                                                        \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    /* R = λ × SEW_bits / pw, pw=16 bits for E8M0 paired scales */     \
    const uint16_t R = lambda * SEW_bits / 16;                          \
    const uint16_t block_size = bs ? 16 : 32;                           \
    const uint16_t S = (k_eff + block_size - 1) / block_size;          \
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int s = 0; s < S; ++s) {                               \
                /* Spec separate-pair MX scales:                       \
                 *   pair_A = v0[i*R + s], sA = pair_A[7:0]            \
                 *   pair_B = v0[j*R + s], sB = pair_B[15:8] */         \
                uint32_t sf_a  = (uint32_t)i * R + s;                  \
                uint32_t a_reg = sf_a / (vlenb / 2);                   \
                uint32_t a_off = sf_a % (vlenb / 2);                   \
                uint16_t pa    = ((uint16_t *)(v0 + a_reg * vlenb))    \
                                  [H2(a_off)];                          \
                uint32_t sf_b  = (uint32_t)j * R + s;                  \
                uint32_t b_reg = sf_b / (vlenb / 2);                   \
                uint32_t b_off = sf_b % (vlenb / 2);                   \
                uint16_t pb    = ((uint16_t *)(v0 + b_reg * vlenb))    \
                                  [H2(b_off)];                          \
                float32 sc = float32_mul(                               \
                    e8m0_to_float32(pa & 0xFF),                         \
                    e8m0_to_float32(pb >> 8), fs);                      \
                TD scale_td = SCALE_TO_TD(sc, fs);                     \
                                                                        \
                /* Block inner product */                               \
                TD ip = (TD)0;                                          \
                int k_start = s * block_size;                           \
                int k_end = k_start + block_size;                       \
                if (k_end > k_eff) k_end = k_eff;                      \
                for (int k = k_start; k < k_end; ++k) {                \
                    tile_regoffset(i * k_eff + k, lmul, lambda,        \
                                   &regoff, &elemoff);                  \
                    TX1 a = ((T1 *)(vs1 + regoff * vlenb))              \
                             [ELEM_H(elemoff, sizeof(T1))];             \
                                                                        \
                    tile_regoffset(j * k_eff + k, lmul, lambda,        \
                                   &regoff, &elemoff);                  \
                    TX2 b = ((T2 *)(vs2 + regoff * vlenb))              \
                             [ELEM_H(elemoff, sizeof(T2))];             \
                                                                        \
                    ip = OP(a, b, ip, fs);                              \
                }                                                       \
                                                                        \
                /* Accumulate: c_elem = scale_td * ip + c_elem */      \
                c_elem = FMA_TD(scale_td, ip, c_elem, 0, fs);          \
            }                                                           \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

/*
 * Microscaling 4-bit variant: nibble-packed inputs (E2M1 only; E3M0 reserved).
 * Scale loading from v0 uses the same logic as GEN_ZVVMM_VFMMACC_VV_MX.
 */
#define GEN_ZVVMM_VFMMACC_VV_MX_4BIT(NAME, TD, TEXTA, TEXTB,           \
                                       EXTA, EXTB, OP, SCALE_TO_TD,     \
                                       FMA_TD)                           \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env,                        \
                  uint32_t desc)                                        \
{                                                                       \
    const uint32_t widening = sizeof(TD) * 2; /* SEW bits / 4 bits */   \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint32_t SEW_bits = 8u << vsew;                               \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
    const uint8_t  bs = FIELD_EX64(env->vtype, VTYPE, BS);             \
    float_status *fs = &env->fp_status;                                 \
                                                                        \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    const uint16_t R = lambda * SEW_bits / 16;                          \
    const uint16_t block_size = bs ? 16 : 32;                           \
    const uint16_t S = (k_eff + block_size - 1) / block_size;          \
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int blk = 0; blk < S; ++blk) {                        \
                /* Spec separate-pair MX scales:                       \
                 *   pair_A = v0[i*R + blk], sA = pair_A[7:0]          \
                 *   pair_B = v0[j*R + blk], sB = pair_B[15:8] */       \
                uint32_t sf_a  = (uint32_t)i * R + blk;                \
                uint32_t a_reg = sf_a / (vlenb / 2);                   \
                uint32_t a_off = sf_a % (vlenb / 2);                   \
                uint16_t pa    = ((uint16_t *)(v0 + a_reg * vlenb))    \
                                  [H2(a_off)];                          \
                uint32_t sf_b  = (uint32_t)j * R + blk;                \
                uint32_t b_reg = sf_b / (vlenb / 2);                   \
                uint32_t b_off = sf_b % (vlenb / 2);                   \
                uint16_t pb    = ((uint16_t *)(v0 + b_reg * vlenb))    \
                                  [H2(b_off)];                          \
                float32 sc = float32_mul(                               \
                    e8m0_to_float32(pa & 0xFF),                         \
                    e8m0_to_float32(pb >> 8), fs);                      \
                TD scale_td = SCALE_TO_TD(sc, fs);                     \
                                                                        \
                TD ip = (TD)0;                                          \
                int k_start = blk * block_size;                         \
                int k_end = k_start + block_size;                       \
                if (k_end > k_eff) k_end = k_eff;                      \
                for (int k = k_start; k < k_end; ++k) {                \
                    TEXTA a = EXTA(vs1, vlenb,                          \
                        i * k_eff + k, lmul, lambda);                   \
                    TEXTB b = EXTB(vs2, vlenb,                          \
                        j * k_eff + k, lmul, lambda);                   \
                    ip = OP(a, b, ip, fs);                              \
                }                                                       \
                                                                        \
                c_elem = FMA_TD(scale_td, ip, c_elem, 0, fs);          \
            }                                                           \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

/* Scale-to-TD conversion helpers used as macro arguments */
#define MX_SCALE_F32(sc, fs)   (sc)
#define MX_SCALE_F64(sc, fs)   float32_to_float64(sc, fs)
/*
 * Helper: override default_nan_pattern for E4M3 output conversions.
 * The RISC-V default pattern 0x40 maps to E4M3 mantissa=4 (0x7C = 384.0),
 * not NaN (0x7F).  E4M3 NaN requires mantissa=7, which needs pattern 0x70.
 */
static uint8_t float32_to_e4m3_with_nan(float32 v, float_status *fs)
{
    float_status local = *fs;
    local.default_nan_pattern = 0x70;
    uint8_t r = float32_to_float8_e4m3(v, false, &local);
    fs->float_exception_flags |= local.float_exception_flags;
    return r;
}

#define MX_SCALE_F16(sc, fs)   float32_to_float16(sc, false, fs)
#define MX_SCALE_BF16(sc, fs)  float32_to_bfloat16(sc, fs)
#define MX_SCALE_E4M3(sc, fs)  float32_to_e4m3_with_nan(sc, fs)
#define MX_SCALE_E5M2(sc, fs)  float32_to_float8_e5m2(sc, false, fs)

/* Integer-to-float32 helper for MXINT inner product accumulator (always signed) */
#define MXINT_IP_F32_S(ip, fs)  int64_to_float32((int64_t)(ip), (fs))

/*
 * GEN_ZVVMM_VFIMMACC_VV — MXINT: byte-aligned integer inputs, FP accumulator,
 * E8M0 microscaled (vm=0, ext_zvvfmmmx).  For each output element (i,j):
 *
 *   ip = 0; for k: ip += (INT_ACC)a[k] * (INT_ACC)b[k]  (integer)
 *   ip_f32 = INT_TO_F32(ip, fs)
 *   ip_td  = SCALE_TO_TD(ip_f32, fs)
 *   scale_td = SCALE_TO_TD(e8m0_A × e8m0_B, fs)
 *   *c = FMA_TD(scale_td, ip_td, *c, 0, fs)
 */
#define GEN_ZVVMM_VFIMMACC_VV(NAME, TD, TA, TB, INT_ACC,                \
                               INT_TO_F32, SCALE_TO_TD, FMA_TD)         \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env, uint32_t desc)         \
{                                                                       \
    const uint32_t widening = sizeof(TD) / sizeof(TA);                  \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint32_t SEW_bits = 8u << vsew;                               \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
    const uint8_t  bs = FIELD_EX64(env->vtype, VTYPE, BS);             \
    float_status *fs = &env->fp_status;                                 \
                                                                        \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    const uint16_t R = lambda * SEW_bits / 16;                          \
    const uint16_t block_size = bs ? 16 : 32;                           \
    const uint16_t S = (k_eff + block_size - 1) / block_size;          \
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int blk = 0; blk < S; ++blk) {                        \
                /* Spec separate-pair MX scales:                       \
                 *   pair_A = v0[i*R + blk], sA = pair_A[7:0]          \
                 *   pair_B = v0[j*R + blk], sB = pair_B[15:8] */       \
                uint32_t sf_a  = (uint32_t)i * R + blk;                \
                uint32_t a_reg = sf_a / (vlenb / 2);                   \
                uint32_t a_off = sf_a % (vlenb / 2);                   \
                uint16_t pa    = ((uint16_t *)(v0 + a_reg * vlenb))    \
                                  [H2(a_off)];                          \
                uint32_t sf_b  = (uint32_t)j * R + blk;                \
                uint32_t b_reg = sf_b / (vlenb / 2);                   \
                uint32_t b_off = sf_b % (vlenb / 2);                   \
                uint16_t pb    = ((uint16_t *)(v0 + b_reg * vlenb))    \
                                  [H2(b_off)];                          \
                float32 sc = float32_mul(                               \
                    e8m0_to_float32(pa & 0xFF),                         \
                    e8m0_to_float32(pb >> 8), fs);                      \
                TD scale_td = SCALE_TO_TD(sc, fs);                     \
                                                                        \
                /* Block integer inner product */                       \
                INT_ACC ip = 0;                                         \
                int k_start = blk * block_size;                         \
                int k_end = k_start + block_size;                       \
                if (k_end > k_eff) k_end = k_eff;                      \
                for (int k = k_start; k < k_end; ++k) {                \
                    tile_regoffset(i * k_eff + k, lmul, lambda,        \
                                   &regoff, &elemoff);                  \
                    TA a_elem = ((TA *)(vs2 + regoff * vlenb))          \
                                 [ELEM_H(elemoff, sizeof(TA))];         \
                                                                        \
                    tile_regoffset(j * k_eff + k, lmul, lambda,        \
                                   &regoff, &elemoff);                  \
                    TB b_elem = ((TB *)(vs1 + regoff * vlenb))          \
                                 [ELEM_H(elemoff, sizeof(TB))];         \
                                                                        \
                    ip += (INT_ACC)a_elem * (INT_ACC)b_elem;           \
                }                                                       \
                                                                        \
                /* Scale × IP + C */                                    \
                float32 ip_f32 = INT_TO_F32(ip, fs);                   \
                TD ip_td = SCALE_TO_TD(ip_f32, fs);                    \
                c_elem = FMA_TD(scale_td, ip_td, c_elem, 0, fs);       \
            }                                                           \
            *c_elem_ptr = c_elem;                                       \
        }                                                               \
    }                                                                   \
}

/*
 * GEN_ZVVMM_VFIMMACC_VV_4BIT — MXINT nibble-packed integer inputs, FP acc.
 * EXTA/EXTB return INT_ACC directly; returns signed or unsigned int32/uint32.
 */
#define GEN_ZVVMM_VFIMMACC_VV_4BIT(NAME, TD, INT_ACC,                   \
                                    EXTA, EXTB,                          \
                                    INT_TO_F32, SCALE_TO_TD, FMA_TD)    \
    void HELPER(NAME)(void *vd, void *v0, void *vs1,                    \
                  void *vs2, CPURISCVState *env, uint32_t desc)         \
{                                                                       \
    const uint32_t widening = sizeof(TD) * 2; /* SEW bits / 4 bits */   \
    const uint16_t vlenb = riscv_cpu_cfg(env)->vlenb;                   \
    const uint8_t  vsew = FIELD_EX64(env->vtype, VTYPE, VSEW);         \
    const uint8_t  sewb = 1 << vsew;                                    \
    const uint32_t SEW_bits = 8u << vsew;                               \
    const uint8_t  vlambda = vext_vlambda(desc);                        \
    if (vlambda == 0)                                                   \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
    const uint8_t  lambda = 1 << (vlambda - 1);                         \
    const uint8_t  lmul = 1 << vext_lmul(desc);                        \
    const uint8_t  bs = FIELD_EX64(env->vtype, VTYPE, BS);             \
    float_status *fs = &env->fp_status;                                 \
                                                                        \
    const uint16_t k_eff = lambda * lmul * widening;                    \
    const uint16_t M = vlenb / (sewb * lambda);                         \
    const uint16_t mul_c = vlenb / (sewb * lambda * lambda);            \
    const uint16_t R = lambda * SEW_bits / 16;                          \
    const uint16_t block_size = bs ? 16 : 32;                           \
    const uint16_t S = (k_eff + block_size - 1) / block_size;          \
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
                           &regoff, &elemoff);                          \
            TD *c_elem_ptr = &((TD *)(vd + regoff * vlenb))            \
                              [ELEM_H(elemoff, sizeof(TD))];            \
            TD c_elem = *c_elem_ptr;                                    \
                                                                        \
            for (int blk = 0; blk < S; ++blk) {                        \
                /* Spec separate-pair MX scales:                       \
                 *   pair_A = v0[i*R + blk], sA = pair_A[7:0]          \
                 *   pair_B = v0[j*R + blk], sB = pair_B[15:8] */       \
                uint32_t sf_a  = (uint32_t)i * R + blk;                \
                uint32_t a_reg = sf_a / (vlenb / 2);                   \
                uint32_t a_off = sf_a % (vlenb / 2);                   \
                uint16_t pa    = ((uint16_t *)(v0 + a_reg * vlenb))    \
                                  [H2(a_off)];                          \
                uint32_t sf_b  = (uint32_t)j * R + blk;                \
                uint32_t b_reg = sf_b / (vlenb / 2);                   \
                uint32_t b_off = sf_b % (vlenb / 2);                   \
                uint16_t pb    = ((uint16_t *)(v0 + b_reg * vlenb))    \
                                  [H2(b_off)];                          \
                float32 sc = float32_mul(                               \
                    e8m0_to_float32(pa & 0xFF),                         \
                    e8m0_to_float32(pb >> 8), fs);                      \
                TD scale_td = SCALE_TO_TD(sc, fs);                     \
                                                                        \
                /* Block integer inner product from nibble inputs */    \
                INT_ACC ip = 0;                                         \
                int k_start = blk * block_size;                         \
                int k_end = k_start + block_size;                       \
                if (k_end > k_eff) k_end = k_eff;                      \
                for (int k = k_start; k < k_end; ++k) {                \
                    INT_ACC a_elem = (INT_ACC)EXTA(vs2, vlenb,          \
                        i * k_eff + k, lmul, lambda);                   \
                    INT_ACC b_elem = (INT_ACC)EXTB(vs1, vlenb,          \
                        j * k_eff + k, lmul, lambda);                   \
                    ip += a_elem * b_elem;                              \
                }                                                       \
                                                                        \
                /* Scale × IP + C */                                    \
                float32 ip_f32 = INT_TO_F32(ip, fs);                   \
                TD ip_td = SCALE_TO_TD(ip_f32, fs);                    \
                c_elem = FMA_TD(scale_td, ip_td, c_elem, 0, fs);       \
            }                                                           \
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
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL counts B SEW-slots; one B/C column = lambda*LMUL VL units */   \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
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
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL counts B SEW-slots; one B/C column = lambda*LMUL VL units */   \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
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

/* octet-widening (W=8): int8 -> int64 */
#define O8WOP_SSS_D int64_t, int8_t,  int8_t,  int64_t,  int64_t
#define O8WOP_UUU_D int64_t, uint8_t, uint8_t, uint64_t, uint64_t
#define O8WOP_SSU_D int64_t, int8_t,  uint8_t, int64_t,  uint64_t
#define O8WOP_SUS_D int64_t, uint8_t, int8_t,  uint64_t, int64_t

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

/* Octet-widening (W=8) Int4→Int32: 4-bit inputs, 32-bit accumulator */
/* signed x signed */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, v8wmmacc_vv_w, int32_t,
        int32_t, int32_t, tile_extract_int4, tile_extract_int4, DO_MACC)
/* unsigned x unsigned */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, v8wmmacc_vv_w_u, uint32_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, DO_MACC)
/* signed A x unsigned B */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, v8wmmacc_vv_w_su, int32_t,
        int32_t, uint32_t, tile_extract_int4, tile_extract_uint4, DO_MACC)
/* unsigned A x signed B */
RVVCALL(GEN_ZVVMM_VMMACC_VV_4BIT, v8wmmacc_vv_w_us, int32_t,
        uint32_t, int32_t, tile_extract_uint4, tile_extract_int4, DO_MACC)

/* Octet-widening (W=8) Int8→Int64: byte-aligned inputs, 64-bit accumulator */
RVVCALL(GEN_ZVVMM_VMMACC_VV, v8wmmacc_vv_d, O8WOP_SSS_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, v8wmmacc_vv_d_u, O8WOP_UUU_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, v8wmmacc_vv_d_su, O8WOP_SSU_D, DO_MACC)
RVVCALL(GEN_ZVVMM_VMMACC_VV, v8wmmacc_vv_d_us, O8WOP_SUS_D, DO_MACC)

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
    const uint16_t vl_div = lambda * lmul;                              \
    uint32_t vl = env->vl;                                              \
                                                                        \
    /* VL counts B SEW-slots; one B/C column = lambda*LMUL VL units */   \
    if (vl % vl_div)                                                    \
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, 0);         \
                                                                        \
    const uint16_t N = vl / vl_div;                                     \
                                                                        \
    /* A from vs1 (bits 19:15), B from vs2 (bits 24:20) */             \
    for (int j = 0; j < N; ++j) {                                       \
        for (int i = 0; i < M; ++i) {                                   \
            uint8_t regoff;                                             \
            uint16_t elemoff;                                           \
                                                                        \
            /* C[i,j]: row-major, flat = i * M + j (N_max = M) */       \
            tile_regoffset(i * M + j, mul_c, lambda,                   \
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

/*
 * E8M0 decode helpers for microscaling.
 *
 * E8M0 is a 8-bit biased exponent format (bias 127), with no mantissa or sign
 * bit, encoding powers of 2 from 2^-127 to 2^127.  0xFF encodes NaN.
 *
 * float32 has the same exponent bias (127), so for e8m0 in [1,254] the float32
 * bit pattern is simply (uint32_t)e8m0 << 23.  E8M0 = 0 (2^-127) is the only
 * case that requires special treatment: float32 exponent = 0 is subnormal, so
 * 2^-127 = 2^-126 × 0.5 needs mantissa bit 22 set → float32 = 0x00400000.
 *
 * float64 can represent all E8M0 values as normal numbers: the float64
 * exponent bias is 1023, so float64_exp = e8m0 + (1023 - 127) = e8m0 + 896.
 *
 * bfloat16 shares float32's exponent format (bias 127, 8 exp bits), so it
 * has the same special cases as float32 (e8m0=0 is subnormal at 0x0040).
 */
static float32 G_GNUC_UNUSED e8m0_to_float32(uint8_t e8m0)
{
    if (e8m0 == 0xFF) {
        return float32_default_nan(NULL);
    }
    if (e8m0 == 0x00) {
        return make_float32(0x00400000); /* 2^-127 as float32 subnormal */
    }
    return make_float32((uint32_t)e8m0 << 23);
}

static float64 G_GNUC_UNUSED e8m0_to_float64(uint8_t e8m0)
{
    if (e8m0 == 0xFF) {
        return float64_default_nan(NULL);
    }
    return make_float64((uint64_t)(e8m0 + 896) << 52);
}

static bfloat16 G_GNUC_UNUSED e8m0_to_bfloat16(uint8_t e8m0)
{
    if (e8m0 == 0xFF) {
        return 0x7FC0; /* bfloat16 quiet NaN */
    }
    if (e8m0 == 0x00) {
        return 0x0040; /* 2^-127 as bfloat16 subnormal */
    }
    return (bfloat16)((uint16_t)e8m0 << 7);
}

/* Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
/*
 * E4M3 muladd needs default_nan_pattern=0x70 so that the default NaN
 * packs to 0x7F (the only E4M3 NaN encoding).  The RISC-V default
 * pattern 0x40 produces 0x7C (a normal value) in E4M3, which silently
 * converts NaN to a finite number.
 */
static float8_e4m3 fmacc8_e4m3(uint8_t a, uint8_t b, uint8_t d,
                                float_status *fs)
{
    float_status local = *fs;
    local.default_nan_pattern = 0x70;
    uint8_t r = float8_e4m3_muladd(a, b, d, 0, &local);
    fs->float_exception_flags |= local.float_exception_flags;
    return r;
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
    return float32_to_e4m3_with_nan(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       float8_e4m3_to_float32(d, fs), 0, fs), fs);
}

/* Mixed-input: e4m3 × e5m2 → e4m3 */
static uint8_t fmacc8_e4m3xe5m2(uint8_t a, uint8_t b, uint8_t d,
                                 float_status *fs)
{
    return float32_to_e4m3_with_nan(
        float32_muladd(float8_e4m3_to_float32(a, fs),
                       float8_e5m2_to_float32(b, fs),
                       float8_e4m3_to_float32(d, fs), 0, fs), fs);
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
    return float32_to_e4m3_with_nan(
        float32_muladd(float8_e5m2_to_float32(a, fs),
                       float8_e4m3_to_float32(b, fs),
                       float8_e4m3_to_float32(d, fs), 0, fs), fs);
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

/* Widening FP4 E2M1 -> FP8 E5M2 */
static uint8_t fwmacc4_e2m1_e5m2(uint32_t a_raw, uint32_t b_raw, uint8_t d,
                                  float_status *fs)
{
    return float8_e5m2_muladd(
        float4_e2m1_to_float8_e5m2(make_float4_e2m1(a_raw), fs),
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

/* Quad-widening FP4 E2M1 -> BF16 */
static uint16_t fqwmacc4_e2m1_bf16(uint32_t a_raw, uint32_t b_raw, uint16_t d,
                                    float_status *fs)
{
    return bfloat16_muladd(
        float4_e2m1_to_bfloat16(make_float4_e2m1(a_raw), fs),
        float4_e2m1_to_bfloat16(make_float4_e2m1(b_raw), fs),
        d, 0, fs);
}

/* Octet-widening (W=8) FP4 E2M1 → FP32 */
static uint32_t fo8wmacc4_e2m1(uint32_t a_raw, uint32_t b_raw, uint32_t d,
                                float_status *fs)
{
    /* E2M1→float16→float32, multiply in float32, accumulate */
    float32 fa = float16_to_float32(
                     float4_e2m1_to_float16(make_float4_e2m1(a_raw), fs),
                     true, fs);
    float32 fb = float16_to_float32(
                     float4_e2m1_to_float16(make_float4_e2m1(b_raw), fs),
                     true, fs);
    return float32_muladd(fa, fb, d, 0, fs);
}

/* Octet-widening (W=8) FP8 E4M3 → FP64 */
static uint64_t fo8wmacc8_e4m3(uint8_t a, uint8_t b, uint64_t d,
                                float_status *fs)
{
    return float64_muladd(float8_e4m3_to_float64(a, fs),
                          float8_e4m3_to_float64(b, fs), d, 0, fs);
}

/* Octet-widening (W=8) FP8 E5M2 → FP64 */
static uint64_t fo8wmacc8_e5m2(uint8_t a, uint8_t b, uint64_t d,
                                float_status *fs)
{
    return float64_muladd(float8_e5m2_to_float64(a, fs),
                          float8_e5m2_to_float64(b, fs), d, 0, fs);
}

/* Octet-widening mixed E4M3 × E5M2 → FP64 */
static uint64_t fo8wmacc8_e4m3xe5m2(uint8_t a, uint8_t b, uint64_t d,
                                     float_status *fs)
{
    return float64_muladd(float8_e4m3_to_float64(a, fs),
                          float8_e5m2_to_float64(b, fs), d, 0, fs);
}

/* Octet-widening mixed E5M2 × E4M3 → FP64 */
static uint64_t fo8wmacc8_e5m2xe4m3(uint8_t a, uint8_t b, uint64_t d,
                                     float_status *fs)
{
    return float64_muladd(float8_e5m2_to_float64(a, fs),
                          float8_e4m3_to_float64(b, fs), d, 0, fs);
}

/* FP4 widening: 4-bit FP -> 8-bit FP accumulate */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e4m3_e2m1, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, fwmacc4_e2m1)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfwmmacc_vv_e5m2_e2m1, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e2m1_e5m2)

/* FP4 quad-widening: 4-bit FP -> 16-bit FP accumulate */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_f16_e2m1, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4, fqwmacc4_e2m1)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vfqwmmacc_vv_bf16_e2m1, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e2m1_bf16)

/* Octet-widening (W=8) FP4 E2M1 → FP32 */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_4BIT, vf8wmmacc_vv_f32_e2m1, uint32_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fo8wmacc4_e2m1)

/* Octet-widening (W=8) FP8 → FP64 */
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vf8wmmacc_vv_f64_e4m3, O8WOP_UUU_D,
        fo8wmacc8_e4m3)
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vf8wmmacc_vv_f64_e5m2, O8WOP_UUU_D,
        fo8wmacc8_e5m2)
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vf8wmmacc_vv_f64_e4m3_e5m2, O8WOP_UUU_D,
        fo8wmacc8_e4m3xe5m2)
RVVCALL(GEN_ZVVMM_VFMMACC_VV, vf8wmmacc_vv_f64_e5m2_e4m3, O8WOP_UUU_D,
        fo8wmacc8_e5m2xe4m3)

/* ── Microscaling (MX) variants: vm=0, scale from v0 ────────────────────── */

/* vfmmacc MX same-width */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e4m3_mx, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, fmacc8_e4m3, MX_SCALE_E4M3, float8_e4m3_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e5m2_mx, uint8_t, uint8_t, uint8_t,
        uint8_t, uint8_t, fmacc8_e5m2, MX_SCALE_E5M2, float8_e5m2_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e5m2_from_e4m3_mx, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t,
        fmacc8_e4m3_to_e5m2, MX_SCALE_E5M2, float8_e5m2_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e4m3_from_e5m2_mx, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t,
        fmacc8_e5m2_to_e4m3, MX_SCALE_E4M3, float8_e4m3_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e4m3_e4m3_e5m2_mx, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t,
        fmacc8_e4m3xe5m2, MX_SCALE_E4M3, float8_e4m3_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e5m2_e4m3_e5m2_mx, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t,
        fmacc8_e4m3xe5m2_to_e5m2, MX_SCALE_E5M2, float8_e5m2_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e4m3_e5m2_e4m3_mx, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t,
        fmacc8_e5m2xe4m3, MX_SCALE_E4M3, float8_e4m3_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_e5m2_e5m2_e4m3_mx, uint8_t,
        uint8_t, uint8_t, uint8_t, uint8_t,
        fmacc8_e5m2xe4m3_to_e5m2, MX_SCALE_E5M2, float8_e5m2_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_f16_mx, uint16_t, uint16_t,
        uint16_t, uint16_t, uint16_t, fmacc16, MX_SCALE_F16, float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_bf16_mx, uint16_t, uint16_t,
        uint16_t, uint16_t, uint16_t, fmacc16_bfloat16, MX_SCALE_BF16,
        bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_f16_f16_bf16_mx, uint16_t,
        uint16_t, uint16_t, uint16_t, uint16_t,
        fmacc16_f16xbf16, MX_SCALE_F16, float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_bf16_f16_bf16_mx, uint16_t,
        uint16_t, uint16_t, uint16_t, uint16_t,
        fmacc16_f16xbf16_to_bf16, MX_SCALE_BF16, bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_f16_bf16_f16_mx, uint16_t,
        uint16_t, uint16_t, uint16_t, uint16_t,
        fmacc16_bf16xf16, MX_SCALE_F16, float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_bf16_bf16_f16_mx, uint16_t,
        uint16_t, uint16_t, uint16_t, uint16_t,
        fmacc16_bf16xf16_to_bf16, MX_SCALE_BF16, bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_f32_mx, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, fmacc32, MX_SCALE_F32, float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfmmacc_vv_f64_mx, uint64_t, uint64_t,
        uint64_t, uint64_t, uint64_t, fmacc64, MX_SCALE_F64, float64_muladd)

/* vfwmmacc MX widening (byte-aligned inputs) */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f16_e4m3_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e4m3, MX_SCALE_F16,
        float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_bf16_e4m3_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e4m3_bf16, MX_SCALE_BF16,
        bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f16_e5m2_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e5m2, MX_SCALE_F16,
        float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_bf16_e5m2_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e5m2_bf16, MX_SCALE_BF16,
        bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f16_e4m3_e5m2_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e4m3xe5m2, MX_SCALE_F16,
        float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_bf16_e4m3_e5m2_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e4m3xe5m2_bf16,
        MX_SCALE_BF16, bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f16_e5m2_e4m3_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e5m2xe4m3, MX_SCALE_F16,
        float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_bf16_e5m2_e4m3_mx, uint16_t,
        uint8_t, uint8_t, uint8_t, uint8_t, fwmacc8_e5m2xe4m3_bf16,
        MX_SCALE_BF16, bfloat16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f32_f16_mx, uint32_t,
        uint16_t, uint16_t, uint16_t, uint16_t, fwmacc16, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f32_bf16_mx, uint32_t,
        uint16_t, uint16_t, uint16_t, uint16_t, fwmacc16_bfloat16, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f32_f16_bf16_mx, uint32_t,
        uint16_t, uint16_t, uint16_t, uint16_t, fwmacc16_f16xbf16, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f32_bf16_f16_mx, uint32_t,
        uint16_t, uint16_t, uint16_t, uint16_t, fwmacc16_bf16xf16, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfwmmacc_vv_f64_f32_mx, uint64_t,
        uint32_t, uint32_t, uint32_t, uint32_t, fwmacc32, MX_SCALE_F64,
        float64_muladd)

/* vfwmmacc MX 4-bit inputs (E2M1 only) */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX_4BIT, vfwmmacc_vv_e4m3_e2m1_mx, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e2m1, MX_SCALE_E4M3, float8_e4m3_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX_4BIT, vfwmmacc_vv_e5m2_e2m1_mx, uint8_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fwmacc4_e2m1_e5m2, MX_SCALE_E5M2, float8_e5m2_muladd)

/* vfqwmmacc MX quad-widening (byte-aligned inputs) */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f32_e4m3_mx, uint32_t,
        uint8_t, uint8_t, uint32_t, uint32_t, fqwmacc8_e4m3, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f32_e5m2_mx, uint32_t,
        uint8_t, uint8_t, uint32_t, uint32_t, fqwmacc8_e5m2, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f32_e4m3_e5m2_mx, uint32_t,
        uint8_t, uint8_t, uint32_t, uint32_t, fqwmacc8_e4m3xe5m2, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f32_e5m2_e4m3_mx, uint32_t,
        uint8_t, uint8_t, uint32_t, uint32_t, fqwmacc8_e5m2xe4m3, MX_SCALE_F32,
        float32_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f64_f16_mx, uint64_t,
        uint16_t, uint16_t, uint64_t, uint64_t, fqwmacc16, MX_SCALE_F64,
        float64_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f64_bf16_mx, uint64_t,
        uint16_t, uint16_t, uint64_t, uint64_t, fqwmacc16bf, MX_SCALE_F64,
        float64_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f64_f16_bf16_mx, uint64_t,
        uint16_t, uint16_t, uint64_t, uint64_t, fqwmacc16_f16xbf16, MX_SCALE_F64,
        float64_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vfqwmmacc_vv_f64_bf16_f16_mx, uint64_t,
        uint16_t, uint16_t, uint64_t, uint64_t, fqwmacc16_bf16xf16, MX_SCALE_F64,
        float64_muladd)

/* vfqwmmacc MX 4-bit inputs (E2M1 only) */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX_4BIT, vfqwmmacc_vv_f16_e2m1_mx, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e2m1, MX_SCALE_F16, float16_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX_4BIT, vfqwmmacc_vv_bf16_e2m1_mx, uint16_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fqwmacc4_e2m1_bf16, MX_SCALE_BF16, bfloat16_muladd)

/* vf8wmmacc MX: E2M1+scale→FP32 (SEW=32, W=8, vm=0) */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX_4BIT, vf8wmmacc_vv_f32_e2m1_mx, uint32_t,
        uint32_t, uint32_t, tile_extract_uint4, tile_extract_uint4,
        fo8wmacc4_e2m1, MX_SCALE_F32, float32_muladd)

/* vf8wmmacc MX: OFP8+scale→FP64 (SEW=64, W=8, vm=0) */
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vf8wmmacc_vv_f64_e4m3_mx, uint64_t,
        uint8_t, uint8_t, uint64_t, uint64_t, fo8wmacc8_e4m3, MX_SCALE_F64,
        float64_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vf8wmmacc_vv_f64_e5m2_mx, uint64_t,
        uint8_t, uint8_t, uint64_t, uint64_t, fo8wmacc8_e5m2, MX_SCALE_F64,
        float64_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vf8wmmacc_vv_f64_e4m3_e5m2_mx, uint64_t,
        uint8_t, uint8_t, uint64_t, uint64_t, fo8wmacc8_e4m3xe5m2, MX_SCALE_F64,
        float64_muladd)
RVVCALL(GEN_ZVVMM_VFMMACC_VV_MX, vf8wmmacc_vv_f64_e5m2_e4m3_mx, uint64_t,
        uint8_t, uint8_t, uint64_t, uint64_t, fo8wmacc8_e5m2xe4m3, MX_SCALE_F64,
        float64_muladd)

/* vfwimmacc: signed integer (byte-aligned) inputs → FP accumulator, E8M0 MX */
/* I8 → F16/BF16 (sew=1, widening=2) */
GEN_ZVVMM_VFIMMACC_VV(vfwimmacc_vv_f16_ss,  uint16_t, int8_t,  int8_t,  int64_t,
    MXINT_IP_F32_S, MX_SCALE_F16,  float16_muladd)
GEN_ZVVMM_VFIMMACC_VV(vfwimmacc_vv_bf16_ss, uint16_t, int8_t,  int8_t,  int64_t,
    MXINT_IP_F32_S, MX_SCALE_BF16, bfloat16_muladd)
/* I16 → F32 (sew=2, widening=2) */
GEN_ZVVMM_VFIMMACC_VV(vfwimmacc_vv_f32_ss,  uint32_t, int16_t, int16_t, int64_t,
    MXINT_IP_F32_S, MX_SCALE_F32,  float32_muladd)

/* vfqwimmacc: signed integer inputs → FP accumulator, E8M0 MX */
/* I4 nibble → F16/BF16 (sew=1, widening=4) */
GEN_ZVVMM_VFIMMACC_VV_4BIT(vfqwimmacc_vv_f16_ss,  uint16_t, int32_t,
    tile_extract_int4, tile_extract_int4,
    MXINT_IP_F32_S, MX_SCALE_F16,  float16_muladd)
GEN_ZVVMM_VFIMMACC_VV_4BIT(vfqwimmacc_vv_bf16_ss, uint16_t, int32_t,
    tile_extract_int4, tile_extract_int4,
    MXINT_IP_F32_S, MX_SCALE_BF16, bfloat16_muladd)
/* I8 → F32 (sew=2, widening=4) */
GEN_ZVVMM_VFIMMACC_VV(vfqwimmacc_vv_f32_ss,  uint32_t, int8_t,  int8_t,  int64_t,
    MXINT_IP_F32_S, MX_SCALE_F32,  float32_muladd)
/* I16 → F64 (sew=3, widening=4) */
GEN_ZVVMM_VFIMMACC_VV(vfqwimmacc_vv_f64_ss,  uint64_t, int16_t, int16_t, int64_t,
    MXINT_IP_F32_S, MX_SCALE_F64,  float64_muladd)

/* vf8wimmacc: signed integer inputs → FP accumulator, E8M0 MX (W=8) */
/* I4 nibble → F32 (sew=2, widening=8) */
GEN_ZVVMM_VFIMMACC_VV_4BIT(vf8wimmacc_vv_f32_ss, uint32_t, int32_t,
    tile_extract_int4, tile_extract_int4,
    MXINT_IP_F32_S, MX_SCALE_F32, float32_muladd)
/* I8 → F64 (sew=3, widening=8) */
GEN_ZVVMM_VFIMMACC_VV(vf8wimmacc_vv_f64_ss, uint64_t, int8_t, int8_t, int64_t,
    MXINT_IP_F32_S, MX_SCALE_F64, float64_muladd)

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

/*
 * Integer MACC dispatch tables: fns[altfmt_a][altfmt_b][sew]
 *
 * Helper selection is performed at translation time based on altfmt_a,
 * altfmt_b, and sew, which are all captured in the TB flags.  This is
 * correct because vsetvl (which sets these fields in vtype) is a context-
 * synchronising instruction that ends the current TB.  Any change to
 * altfmt_a, altfmt_b, or sew therefore triggers TB lookup/recompilation
 * with the new flag values, guaranteeing that the dispatch table index
 * always matches the current vtype configuration.
 */
static bool zvvmm_macc_trans(DisasContext *s, arg_rmrr *a,
                             gen_helper_vmmacc * const fns[2][2][4],
                             gen_helper_vmmacc * const mxint_fns[2][4],
                             const uint32_t ext_offsets[4],
                             int widening)
{
    if (!zvvmm_ext_check(s, ext_offsets[s->sew])) {
        return false;
    }

    int8_t mul_c_log2 = zvvmm_mul_c_log2(s);

    if (mul_c_log2 >= 0 && s->lmul >= 0 && vext_check_isa_ill(s) &&
        require_align(a->rd, mul_c_log2) &&
        require_align(a->rs1, s->lmul) && require_align(a->rs2, s->lmul)) {

        if (a->vm == 0) {
            /* MXINT path: signed integer inputs, FP accumulator, E8M0 microscaled.
             * OCP MX defines MXINT as signed-only; altfmt_a/altfmt_b must be 0.
             * FP accumulator needs the dynamic rounding mode. */
            gen_set_rm(s, RISCV_FRM_DYN);
            if (!mxint_fns || !s->cfg_ptr->ext_zvvfmmmx) {
                return false;
            }
            if (s->altfmt_a || s->altfmt_b) {
                return false;
            }
            /* MX legality: EEW_C × λ ≥ pw (pw=16 bits for E8M0 paired) */
            if (((8u << s->sew) << (s->vlambda - 1)) < 16) {
                return false;
            }
            if (s->bs && widening * (1 << s->lmul) > (1u << s->sew)) {
                return false;
            }
            gen_helper_vmmacc *mxint_fn = mxint_fns[s->altfmt][s->sew];
            if (mxint_fn) {
                return opivv_trans(a->rd, a->rs1, a->rs2, 0, mxint_fn, s);
            }
            return false;
        }

        /* vm=1: integer MACC */
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
                              gen_helper_vmmacc * const mx_fns[2][2][2][4],
                              const uint32_t ext_offsets[2][2][4],
                              int widening)
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

        /* All FP MACC paths use the dynamic rounding mode from frm */
        gen_set_rm(s, RISCV_FRM_DYN);

        if (a->vm == 0) {
            /* Microscaling (MX) path: vm=0 with E8M0 scale from v0 */
            if (!s->cfg_ptr->ext_zvvfmmmx) {
                return false;
            }
            /* MX legality: EEW_C × λ ≥ pw (pw=16 bits for E8M0 paired) */
            if (((8u << s->sew) << (s->vlambda - 1)) < 16) {
                return false;
            }
            /* BS=16 legality: widening × LMUL ≤ SEW */
            if (s->bs && widening * (1 << s->lmul) > (1u << s->sew)) {
                return false;
            }
            gen_helper_vmmacc *mx_fn =
                mx_fns[s->altfmt_a][s->altfmt_b][s->altfmt][s->sew];
            if (mx_fn) {
                return opivv_trans(a->rd, a->rs1, a->rs2, 0, mx_fn, s);
            }
            return false;
        }

        /* vm=1: unmasked non-MX path */
        gen_helper_vmmacc *fn = fns[s->altfmt_a][s->altfmt_b][s->altfmt][s->sew];
        if (fn) {
            return opivv_trans(a->rd, a->rs1, a->rs2, 1, fn, s);
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
    /* vm=0 is reserved for same-width integer MACC */
    return zvvmm_macc_trans(s, a, fns, NULL, ext_offsets, 1);
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
    /* vm=0: vfwimmacc — signed integer inputs, FP accumulator, E8M0 microscaled */
    static gen_helper_vmmacc * const mxint_fns[2][4] = {
        /* altfmt = 0 (F16/F32) */
        { NULL, gen_helper_vfwimmacc_vv_f16_ss,
          gen_helper_vfwimmacc_vv_f32_ss, NULL },
        /* altfmt = 1 (BF16) */
        { NULL, gen_helper_vfwimmacc_vv_bf16_ss, NULL, NULL }
    };
    return zvvmm_macc_trans(s, a, fns, mxint_fns, ext_offsets, 2);
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
    /* vm=0: vfqwimmacc — signed integer inputs, FP accumulator, E8M0 microscaled */
    static gen_helper_vmmacc * const mxint_fns[2][4] = {
        /* altfmt = 0 (F16/F32/F64) */
        { NULL, gen_helper_vfqwimmacc_vv_f16_ss,
          gen_helper_vfqwimmacc_vv_f32_ss,
          gen_helper_vfqwimmacc_vv_f64_ss },
        /* altfmt = 1 (BF16) */
        { NULL, gen_helper_vfqwimmacc_vv_bf16_ss, NULL, NULL }
    };
    return zvvmm_macc_trans(s, a, fns, mxint_fns, ext_offsets, 4);
}

static bool trans_v8wmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0: signed x signed */
            { NULL, NULL,
              gen_helper_v8wmmacc_vv_w,
              gen_helper_v8wmmacc_vv_d },
            /* altfmt_b = 1: signed x unsigned */
            { NULL, NULL,
              gen_helper_v8wmmacc_vv_w_su,
              gen_helper_v8wmmacc_vv_d_su }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0: unsigned x signed */
            { NULL, NULL,
              gen_helper_v8wmmacc_vv_w_us,
              gen_helper_v8wmmacc_vv_d_us },
            /* altfmt_b = 1: unsigned x unsigned */
            { NULL, NULL,
              gen_helper_v8wmmacc_vv_w_u,
              gen_helper_v8wmmacc_vv_d_u }
        }
    };
    static const uint32_t ext_offsets[4] = {
        0, 0,
        CPU_CFG_OFFSET(ext_zvvmmi4w), CPU_CFG_OFFSET(ext_zvvmmbd)
    };
    /* vm=0: vf8wimmacc — signed integer inputs, FP accumulator, E8M0 MX */
    static gen_helper_vmmacc * const mxint_fns[2][4] = {
        /* altfmt = 0 (FP32/FP64) */
        { NULL, NULL,
          gen_helper_vf8wimmacc_vv_f32_ss,
          gen_helper_vf8wimmacc_vv_f64_ss },
        /* altfmt = 1 */
        { NULL, NULL, NULL, NULL }
    };
    return zvvmm_macc_trans(s, a, fns, mxint_fns, ext_offsets, 8);
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
    static gen_helper_vmmacc * const mx_fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 */
            {
                { gen_helper_vfmmacc_vv_e4m3_mx, gen_helper_vfmmacc_vv_f16_mx,
                  gen_helper_vfmmacc_vv_f32_mx,  gen_helper_vfmmacc_vv_f64_mx },
                { gen_helper_vfmmacc_vv_e5m2_from_e4m3_mx, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 */
            {
                { gen_helper_vfmmacc_vv_e4m3_e4m3_e5m2_mx,
                  gen_helper_vfmmacc_vv_f16_f16_bf16_mx, NULL, NULL },
                { gen_helper_vfmmacc_vv_e5m2_e4m3_e5m2_mx,
                  gen_helper_vfmmacc_vv_bf16_f16_bf16_mx, NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 */
            {
                { gen_helper_vfmmacc_vv_e4m3_e5m2_e4m3_mx,
                  gen_helper_vfmmacc_vv_f16_bf16_f16_mx, NULL, NULL },
                { gen_helper_vfmmacc_vv_e5m2_e5m2_e4m3_mx,
                  gen_helper_vfmmacc_vv_bf16_bf16_f16_mx, NULL, NULL }
            },
            /* altfmt_b = 1 */
            {
                { gen_helper_vfmmacc_vv_e4m3_from_e5m2_mx, NULL, NULL, NULL },
                { gen_helper_vfmmacc_vv_e5m2_mx, gen_helper_vfmmacc_vv_bf16_mx,
                  NULL, NULL }
            }
        }
    };
    return zvvmm_fmacc_trans(s, a, fns, mx_fns, ext_offsets, 1);
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
            /* altfmt_b = 1 (mixed: A=e2m1/e4m3/f16, B=e5m2/bf16; E3M0 reserved) */
            {
                { NULL,
                  gen_helper_vfwmmacc_vv_f16_e4m3_e5m2,
                  gen_helper_vfwmmacc_vv_f32_f16_bf16, NULL },
                { NULL,
                  gen_helper_vfwmmacc_vv_bf16_e4m3_e5m2,
                  NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (mixed: A=e5m2/bf16, B=e4m3/f16; E3M0 reserved) */
            {
                { NULL,
                  gen_helper_vfwmmacc_vv_f16_e5m2_e4m3,
                  gen_helper_vfwmmacc_vv_f32_bf16_f16, NULL },
                { NULL,
                  gen_helper_vfwmmacc_vv_bf16_e5m2_e4m3,
                  NULL, NULL }
            },
            /* altfmt_b = 1 (same; E3M0 reserved) */
            {
                { NULL,
                  gen_helper_vfwmmacc_vv_f16_e5m2,
                  gen_helper_vfwmmacc_vv_f32_bf16, NULL },
                { NULL,
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
        /* altfmt_a = 1 (sew=0 is E3M0, now reserved; use 0 to block it) */
        {
            /* altfmt = 0 */
            { 0, CPU_CFG_OFFSET(ext_zvvfmmofp8h),
              CPU_CFG_OFFSET(ext_zvvfmmbf16f), 0 },
            /* altfmt = 1 */
            { 0, CPU_CFG_OFFSET(ext_zvvfmmofp8bf16),
              0, 0 }
        }
    };
    static gen_helper_vmmacc * const mx_fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 */
            {
                { gen_helper_vfwmmacc_vv_e4m3_e2m1_mx,
                  gen_helper_vfwmmacc_vv_f16_e4m3_mx,
                  gen_helper_vfwmmacc_vv_f32_f16_mx,
                  gen_helper_vfwmmacc_vv_f64_f32_mx },
                { gen_helper_vfwmmacc_vv_e5m2_e2m1_mx,
                  gen_helper_vfwmmacc_vv_bf16_e4m3_mx,
                  NULL, NULL }
            },
            /* altfmt_b = 1 (E3M0 B reserved) */
            {
                { NULL,
                  gen_helper_vfwmmacc_vv_f16_e4m3_e5m2_mx,
                  gen_helper_vfwmmacc_vv_f32_f16_bf16_mx, NULL },
                { NULL,
                  gen_helper_vfwmmacc_vv_bf16_e4m3_e5m2_mx,
                  NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (E3M0 A reserved) */
            {
                { NULL,
                  gen_helper_vfwmmacc_vv_f16_e5m2_e4m3_mx,
                  gen_helper_vfwmmacc_vv_f32_bf16_f16_mx, NULL },
                { NULL,
                  gen_helper_vfwmmacc_vv_bf16_e5m2_e4m3_mx,
                  NULL, NULL }
            },
            /* altfmt_b = 1 (E3M0 reserved) */
            {
                { NULL,
                  gen_helper_vfwmmacc_vv_f16_e5m2_mx,
                  gen_helper_vfwmmacc_vv_f32_bf16_mx, NULL },
                { NULL,
                  gen_helper_vfwmmacc_vv_bf16_e5m2_mx,
                  NULL, NULL }
            }
        }
    };
    return zvvmm_fmacc_trans(s, a, fns, mx_fns, ext_offsets, 2);
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
            /* altfmt_b = 1 (E3M0 B reserved) */
            {
                { NULL, NULL,
                  gen_helper_vfqwmmacc_vv_f32_e4m3_e5m2,
                  gen_helper_vfqwmmacc_vv_f64_f16_bf16 },
                { NULL, NULL, NULL, NULL }
            }
        },
        /* altfmt_a = 1 (sew=1 E3M0 A reserved) */
        {
            /* altfmt_b = 0 (E3M0 A reserved) */
            {
                { NULL, NULL,
                  gen_helper_vfqwmmacc_vv_f32_e5m2_e4m3,
                  gen_helper_vfqwmmacc_vv_f64_bf16_f16 },
                { NULL, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 (E3M0 A+B reserved) */
            {
                { NULL, NULL,
                  gen_helper_vfqwmmacc_vv_f32_e5m2,
                  gen_helper_vfqwmmacc_vv_f64_bf16 },
                { NULL, NULL, NULL, NULL }
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
        /* altfmt_a = 1 (sew=1 is E3M0, now reserved; use 0 to block it) */
        {
            /* altfmt = 0 */
            { 0, 0,
              CPU_CFG_OFFSET(ext_zvvfmmofp8w), CPU_CFG_OFFSET(ext_zvvfmmbf16d) },
            /* altfmt = 1 */
            { 0, 0, 0, 0 }
        }
    };
    static gen_helper_vmmacc * const mx_fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 */
            {
                { NULL, gen_helper_vfqwmmacc_vv_f16_e2m1_mx,
                  gen_helper_vfqwmmacc_vv_f32_e4m3_mx,
                  gen_helper_vfqwmmacc_vv_f64_f16_mx },
                { NULL, gen_helper_vfqwmmacc_vv_bf16_e2m1_mx, NULL, NULL }
            },
            /* altfmt_b = 1 (E3M0 B reserved) */
            {
                { NULL, NULL,
                  gen_helper_vfqwmmacc_vv_f32_e4m3_e5m2_mx,
                  gen_helper_vfqwmmacc_vv_f64_f16_bf16_mx },
                { NULL, NULL, NULL, NULL }
            }
        },
        /* altfmt_a = 1 (E3M0 A reserved) */
        {
            /* altfmt_b = 0 */
            {
                { NULL, NULL,
                  gen_helper_vfqwmmacc_vv_f32_e5m2_e4m3_mx,
                  gen_helper_vfqwmmacc_vv_f64_bf16_f16_mx },
                { NULL, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 */
            {
                { NULL, NULL,
                  gen_helper_vfqwmmacc_vv_f32_e5m2_mx,
                  gen_helper_vfqwmmacc_vv_f64_bf16_mx },
                { NULL, NULL, NULL, NULL }
            }
        }
    };
    return zvvmm_fmacc_trans(s, a, fns, mx_fns, ext_offsets, 4);
}

static bool trans_vf8wmmacc_vv(DisasContext *s, arg_rmrr *a)
{
    static gen_helper_vmmacc * const fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 (same: E2M1/E4M3) */
            {
                { NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f32_e2m1,
                  gen_helper_vf8wmmacc_vv_f64_e4m3 },
                { NULL, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 (mixed: A=E4M3, B=E5M2 at SEW=64) */
            {
                { NULL, NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f64_e4m3_e5m2 },
                { NULL, NULL, NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (mixed: A=E5M2, B=E4M3 at SEW=64) */
            {
                { NULL, NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f64_e5m2_e4m3 },
                { NULL, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 (same: E5M2 at SEW=64) */
            {
                { NULL, NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f64_e5m2 },
                { NULL, NULL, NULL, NULL }
            }
        }
    };
    static gen_helper_vmmacc * const mx_fns[2][2][2][4] = {
        /* altfmt_a = 0 */
        {
            /* altfmt_b = 0 */
            {
                { NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f32_e2m1_mx,
                  gen_helper_vf8wmmacc_vv_f64_e4m3_mx },
                { NULL, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 (mixed at SEW=64: A=E4M3, B=E5M2) */
            {
                { NULL, NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f64_e4m3_e5m2_mx },
                { NULL, NULL, NULL, NULL }
            }
        },
        /* altfmt_a = 1 */
        {
            /* altfmt_b = 0 (mixed at SEW=64: A=E5M2, B=E4M3) */
            {
                { NULL, NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f64_e5m2_e4m3_mx },
                { NULL, NULL, NULL, NULL }
            },
            /* altfmt_b = 1 (same E5M2 at SEW=64) */
            {
                { NULL, NULL, NULL,
                  gen_helper_vf8wmmacc_vv_f64_e5m2_mx },
                { NULL, NULL, NULL, NULL }
            }
        }
    };
    static const uint32_t ext_offsets[2][2][4] = {
        { { 0, 0, CPU_CFG_OFFSET(ext_zvvfmmofp4w),
            CPU_CFG_OFFSET(ext_zvvfmmofp8d) },
          { 0, 0, CPU_CFG_OFFSET(ext_zvvfmmofp4w),
            CPU_CFG_OFFSET(ext_zvvfmmofp8d) } },
        { { 0, 0, 0, CPU_CFG_OFFSET(ext_zvvfmmofp8d) },
          { 0, 0, 0, CPU_CFG_OFFSET(ext_zvvfmmofp8d) } }
    };
    return zvvmm_fmacc_trans(s, a, fns, mx_fns, ext_offsets, 8);
}
