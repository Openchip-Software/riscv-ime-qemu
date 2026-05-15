/*
 * RISC-V Zvvmm Integer Matrix Multiply-Accumulate Unit Test
 *
 * Validates vmmacc.vv, vwmmacc.vv, and vqwmmacc.vv for all valid
 * (SEW, lambda, LMUL) combinations, signedness variants, accumulation,
 * partial VL, and int4 nibble-packed inputs.
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
 *
 * MACC: funct6[31:26] vm=1[25] rs2[24:20] rs1[19:15] funct3[14:12]
 *       rd[11:7] opcode[6:0]=1010111
 * With: vs2=v0(00000), vs1=v8(01000), vd=v16(10000)
 *
 * Tile load/store: lambda=0(DYNAMIC) func[28:26] vm=1[25]
 *       rs2=a1(01011)[24:20] rs1=a0(01010)[19:15] 111[14:12]
 *       vd[11:7] opcode[6:0]
 * ------------------------------------------------------------------ */

/* Integer MACC */
#define INSN_VMMACC_VV    0xE2040857u  /* funct6=111000, funct3=000 */
#define INSN_VWMMACC_VV   0xE6040857u  /* funct6=111001, funct3=000 */
#define INSN_VQWMMACC_VV  0xEA040857u  /* funct6=111010, funct3=000 */
#define INSN_V8WMMACC_VV  0xEE040857u  /* funct6=111011, funct3=000 */

/*
 * Tile load/store (lambda=0=dynamic, vm=1, rs2=a1, rs1=a0, funct3=111)
 *
 * MACC register mapping: the helper reads A (row-major) from vs1 and
 * B (column-major) from vs2.  In the encoding above rs1=v8 → vs1,
 * rs2=v0 → vs2.  Therefore: load A into v8, load B into v0.
 */
#define INSN_VMTL_V0   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  0u <<  7 | 0x07u)
#define INSN_VMTL_V8   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x07u)
#define INSN_VMTL_V16  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 16u <<  7 | 0x07u)
#define INSN_VMTS_V16  (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 | 16u <<  7 | 0x27u)

/* vxor.vv v16, v16, v16 — zero C tile */
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

/*
 * Set vtype for MACC operations.
 * Returns VL = VLMAX (since rs1=zero → AVL=max).
 */
static inline uint64_t set_macc_vtype(int sew_bytes, int lmul, int lambda,
                                      int altfmt, int altfmt_a, int altfmt_b)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul)
                   | ((uint64_t)__builtin_ctz(sew_bytes) << 3)
                   | ((uint64_t)altfmt << 8)
                   | ((uint64_t)altfmt_a << 58)
                   | ((uint64_t)altfmt_b << 57)
                   | ((uint64_t)(__builtin_ctz(lambda) + 1) << 60);
    uint64_t vl;
    asm volatile("vsetvl %0, zero, %1" : "=r"(vl) : "r"(vtype));
    return vl;
}

/*
 * Set vtype with explicit AVL (for partial VL tests).
 * Returns VL = min(avl, VLMAX).
 */
static inline uint64_t set_macc_vtype_vl(int sew_bytes, int lmul, int lambda,
                                         int altfmt, int altfmt_a, int altfmt_b,
                                         uint64_t avl)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul)
                   | ((uint64_t)__builtin_ctz(sew_bytes) << 3)
                   | ((uint64_t)altfmt << 8)
                   | ((uint64_t)altfmt_a << 58)
                   | ((uint64_t)altfmt_b << 57)
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

static int64_t read_signed(const void *buf, int idx, int esz)
{
    switch (esz) {
    case 1: return (int8_t) ((const uint8_t  *)buf)[idx];
    case 2: return (int16_t)((const uint16_t *)buf)[idx];
    case 4: return (int32_t)((const uint32_t *)buf)[idx];
    case 8: return (int64_t)((const uint64_t *)buf)[idx];
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

/* Mask to truncate to esz bytes */
static uint64_t elem_mask(int esz)
{
    return esz < 8 ? ((1ULL << (esz * 8)) - 1) : ~0ULL;
}

/* --------------------------------------------------------------------
 * Int4 nibble packing helpers
 * ------------------------------------------------------------------ */

static void pack_nibble(uint8_t *buf, int idx, uint8_t val4)
{
    int byte_idx = idx / 2;
    int nibble   = idx % 2;
    uint8_t raw  = val4 & 0xf;
    if (nibble)
        buf[byte_idx] = (buf[byte_idx] & 0x0f) | (raw << 4);
    else
        buf[byte_idx] = (buf[byte_idx] & 0xf0) | raw;
}

static int32_t sign_extend_int4(uint8_t raw)
{
    return (int8_t)((raw ^ 0x8) - 0x8);
}

/* ====================================================================
 * Test: same-width vmmacc.vv
 * ==================================================================== */

static int test_vmmacc(int sew, int lmul, int lambda, uint64_t vlenb,
                       int altfmt_a, int altfmt_b,
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

    uint8_t *A_buf   = calloc(M * a_ld, sew);
    uint8_t *B_buf   = calloc(N * b_ld, sew);
    uint8_t *C_init  = calloc(c_nelem, sew);
    uint8_t *C_dst   = calloc(c_nelem, sew);

    /* Fill A (row-major): A[row * a_ld + k] */
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int64_t val;
            if (altfmt_a) {
                val = ((row * K + k) % 200) + 1;
            } else {
                val = ((row * K + k) % 100) - 50;
            }
            write_elem(A_buf, row * a_ld + k, sew, (uint64_t)val);
        }
    }

    /* Fill B (column-major): B[col * b_ld + k] */
    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int64_t val;
            if (altfmt_b) {
                val = ((k * N + col) % 200) + 1;
            } else {
                val = ((k * N + col) % 100) - 50;
            }
            write_elem(B_buf, col * b_ld + k, sew, (uint64_t)val);
        }
    }

    /* Fill C_init for accumulation */
    if (accumulate) {
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                int flat = row * M + col;
                write_elem(C_init, flat, sew, (uint64_t)(row + col + 1));
            }
        }
    }

    /* Execute: set vtype at full VL, load tiles, run MACC, store */
    set_macc_vtype(sew, lmul, lambda, 0, altfmt_a, altfmt_b);

    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    if (accumulate) {
        DO_TILE(INSN_VMTL_V16, C_init, (uint64_t)c_ld);
    } else {
        DO_VXOR_V16();
    }

    if (partial_vl) {
        set_macc_vtype_vl(sew, lmul, lambda, 0, altfmt_a, altfmt_b,
                          (uint64_t)K);
    }

    DO_MACC(INSN_VMMACC_VV);

    /* Restore full VL for tile store */
    if (partial_vl) {
        set_macc_vtype(sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    }
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    /* Compute reference and compare */
    int fail = 0;
    uint64_t mask = elem_mask(sew);

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int64_t acc = 0;
            if (accumulate) {
                int flat = row * M + col;
                acc = read_signed(C_init, flat, sew);
            }

            for (int k = 0; k < K; k++) {
                int64_t a_val, b_val;
                if (altfmt_a) {
                    a_val = (int64_t)read_elem(A_buf, row * a_ld + k, sew);
                } else {
                    a_val = read_signed(A_buf, row * a_ld + k, sew);
                }
                if (altfmt_b) {
                    b_val = (int64_t)read_elem(B_buf, col * b_ld + k, sew);
                } else {
                    b_val = read_signed(B_buf, col * b_ld + k, sew);
                }
                acc += a_val * b_val;
            }

            int flat = row * M + col;
            uint64_t expected = (uint64_t)acc & mask;
            uint64_t got = read_elem(C_dst, flat, sew);
            if (got != expected) {
                printf("FAIL %s e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_init); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 2x widening vwmmacc.vv
 * ==================================================================== */

static int test_vwmmacc(int accum_sew, int lmul, int lambda, uint64_t vlenb,
                        int altfmt_a, int altfmt_b, const char *label)
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

    uint8_t *A_buf  = calloc(mu_load * a_ld, input_sew);
    uint8_t *B_buf  = calloc(mu_load * b_ld, input_sew);
    uint8_t *C_dst  = calloc(c_nelem, accum_sew);

    /* Fill A (row-major at input_sew): A[row][k]
     * Memory: A_mem[(W*row + k/linesize_load) * a_ld + k%linesize_load] */
    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int tr = W * row + k / linesize_load;
            int tc = k % linesize_load;
            int64_t val;
            if (altfmt_a) {
                val = ((row * K + k) % 200) + 1;
            } else {
                val = ((row * K + k) % 100) - 50;
            }
            write_elem(A_buf, tr * a_ld + tc, input_sew, (uint64_t)val);
        }
    }

    /* Fill B (column-major at input_sew): B[col][k]
     * Memory: B_mem[(W*col + k/linesize_load) * b_ld + k%linesize_load] */
    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int tr = W * col + k / linesize_load;
            int tc = k % linesize_load;
            int64_t val;
            if (altfmt_b) {
                val = ((k * N + col) % 200) + 1;
            } else {
                val = ((k * N + col) % 100) - 50;
            }
            write_elem(B_buf, tr * b_ld + tc, input_sew, (uint64_t)val);
        }
    }

    /* Step 1: Configure at input_sew, load A and B */
    set_macc_vtype(input_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    /* Step 2: Reconfigure to accum_sew (preserves register contents) */
    set_macc_vtype(accum_sew, lmul, lambda, 0, altfmt_a, altfmt_b);

    /* Step 3: Zero C, execute vwmmacc, store C */
    DO_VXOR_V16();
    DO_MACC(INSN_VWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    /* Compute reference */
    int fail = 0;
    uint64_t mask = elem_mask(accum_sew);

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++) {
                int tr_a = W * row + k / linesize_load;
                int tc_a = k % linesize_load;
                int tr_b = W * col + k / linesize_load;
                int tc_b = k % linesize_load;
                int64_t a_val, b_val;

                if (altfmt_a) {
                    a_val = (int64_t)read_elem(A_buf,
                                tr_a * a_ld + tc_a, input_sew);
                } else {
                    a_val = read_signed(A_buf,
                                tr_a * a_ld + tc_a, input_sew);
                }
                if (altfmt_b) {
                    b_val = (int64_t)read_elem(B_buf,
                                tr_b * b_ld + tc_b, input_sew);
                } else {
                    b_val = read_signed(B_buf,
                                tr_b * b_ld + tc_b, input_sew);
                }
                acc += a_val * b_val;
            }

            int flat = row * M + col;
            uint64_t expected = (uint64_t)acc & mask;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, input_sew * 8, accum_sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 4x widening vqwmmacc.vv
 * ==================================================================== */

static int test_vqwmmacc(int accum_sew, int lmul, int lambda, uint64_t vlenb,
                         int altfmt_a, int altfmt_b, const char *label)
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

    uint8_t *A_buf  = calloc(mu_load * a_ld, input_sew);
    uint8_t *B_buf  = calloc(mu_load * b_ld, input_sew);
    uint8_t *C_dst  = calloc(c_nelem, accum_sew);

    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int tr = W * row + k / linesize_load;
            int tc = k % linesize_load;
            int64_t val;
            if (altfmt_a) {
                val = ((row * K + k) % 200) + 1;
            } else {
                val = ((row * K + k) % 100) - 50;
            }
            write_elem(A_buf, tr * a_ld + tc, input_sew, (uint64_t)val);
        }
    }

    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int tr = W * col + k / linesize_load;
            int tc = k % linesize_load;
            int64_t val;
            if (altfmt_b) {
                val = ((k * N + col) % 200) + 1;
            } else {
                val = ((k * N + col) % 100) - 50;
            }
            write_elem(B_buf, tr * b_ld + tc, input_sew, (uint64_t)val);
        }
    }

    set_macc_vtype(input_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    set_macc_vtype(accum_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_VQWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;
    uint64_t mask = elem_mask(accum_sew);

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++) {
                int tr_a = W * row + k / linesize_load;
                int tc_a = k % linesize_load;
                int tr_b = W * col + k / linesize_load;
                int tc_b = k % linesize_load;
                int64_t a_val, b_val;

                if (altfmt_a) {
                    a_val = (int64_t)read_elem(A_buf,
                                tr_a * a_ld + tc_a, input_sew);
                } else {
                    a_val = read_signed(A_buf,
                                tr_a * a_ld + tc_a, input_sew);
                }
                if (altfmt_b) {
                    b_val = (int64_t)read_elem(B_buf,
                                tr_b * b_ld + tc_b, input_sew);
                } else {
                    b_val = read_signed(B_buf,
                                tr_b * b_ld + tc_b, input_sew);
                }
                acc += a_val * b_val;
            }

            int flat = row * M + col;
            uint64_t expected = (uint64_t)acc & mask;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, input_sew * 8, accum_sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 2x widening vwmmacc.vv with int4 inputs
 *
 * Int4 nibbles packed two per byte at input_sew=1 (e8 accumulator).
 * The MACC instruction reads sub-byte (4-bit) elements; the tile load
 * operates on bytes, packing two nibbles per byte.
 * ==================================================================== */

static int test_vwmmacc_int4(int lmul, int lambda, uint64_t vlenb,
                             int altfmt_a, int altfmt_b, const char *label)
{
    int accum_sew = 1;  /* e8 accumulator */
    int W = 2;          /* SEW/4-bit = 2 (from byte to nibble) */
    /* For int4, widening = sizeof(accum) * 2 = 1 * 2 = 2 (in 4-bit units) */
    /* K_eff = lambda * W_effective * lmul where W_effective maps 4-bit to 8-bit */
    /* The 4-bit macro uses: widening = sizeof(TD) * 2 = 1 * 2 = 2 */
    int k_eff = lambda * W * lmul;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_load = lambda * lmul;

    if (M < W || mul_c != lmul) {
        return 0;
    }

    int N_active = M;

    /*
     * For int4, the tile load at e8 (accum_sew=1) loads byte elements.
     * Two nibbles pack per byte. K_eff nibbles span K_eff/2 bytes in
     * the reduction dimension. But the tile load sees only bytes.
     *
     * Actually, the MACC helper for int4 uses tile_extract_nibble which
     * works on flat 4-bit element indices. The tile load loads bytes into
     * the register group. Each byte holds 2 nibbles.
     *
     * For the test: we load at e8 (byte-level), so the tile has
     * mu_load rows × linesize_load columns of BYTES. Each byte holds
     * two 4-bit elements.
     *
     * The MACC helper indexes k_eff 4-bit elements via flat = i*k_eff+k.
     * tile_extract_nibble maps this flat idx to register position,
     * splits into byte offset and nibble selector.
     *
     * To fill the source tiles, we need to pack 4-bit values into the
     * memory buffer that vmtl will load, such that the flat 4-bit index
     * used by the MACC helper finds the expected value.
     *
     * The flat 4-bit index maps through tile_regoffset (at 4-bit
     * granularity) to (regoff, elemoff), then byte = elemoff/2,
     * nibble = elemoff%2. The vmtl load maps flat byte index through
     * tile_regoffset (at byte granularity) to (regoff, byte_elemoff).
     *
     * For consistency, we set up the byte buffer so that after vmtl,
     * each byte in the register group contains the two expected nibbles.
     *
     * The vmtl at e8 with linesize_load stride loads bytes contiguously
     * (tight packing). The MACC then reads nibbles from the loaded bytes.
     *
     * With tight packing (a_ld = linesize_load), flat byte index f
     * maps to memory byte f. After vmtl, register byte at
     * tile_regoffset(f, lmul, lambda) = (regoff, byte_elemoff) holds
     * the byte from memory[f].
     *
     * The MACC reads nibble at flat 4-bit index n. tile_extract_nibble
     * maps n through tile_regoffset(n, lmul, lambda) →
     * (regoff, elemoff_4bit), then byte = elemoff_4bit/2,
     * nibble = elemoff_4bit%2, and reads from register byte at
     * (regoff, byte).
     *
     * The relationship between byte-level and nibble-level regoffset
     * is not trivially 2:1 because the linesize is the same for both
     * (lambda * lmul).
     *
     * Since this is complex, we use a simpler approach: fill all
     * bytes with the same pattern value, so the nibble order doesn't
     * matter for correctness. Use A=all-1 nibbles, B=all-1 nibbles.
     * Then C[i,j] = K_eff * 1 * 1 = K_eff for all (i,j).
     */

    int a_ld = linesize_load;
    int b_ld = linesize_load;
    int c_ld = lambda * lmul;
    int mu_load = vlenb / (accum_sew * lambda);
    int byte_count = mu_load * a_ld;

    uint8_t *A_buf = calloc(byte_count, 1);
    uint8_t *B_buf = calloc(byte_count, 1);
    uint8_t *C_dst = calloc(M * N, accum_sew);

    /* Fill all bytes with nibble value 1 in both nibble positions:
     * byte = (1 << 4) | 1 = 0x11 */
    uint8_t a_byte, b_byte;
    if (altfmt_a) {
        /* unsigned: use value 1 in both nibbles */
        a_byte = 0x11;
    } else {
        /* signed: use value 1 (positive) */
        a_byte = 0x11;
    }
    if (altfmt_b) {
        b_byte = 0x11;
    } else {
        b_byte = 0x11;
    }
    memset(A_buf, a_byte, byte_count);
    memset(B_buf, b_byte, byte_count);

    /* Load at e8, then MACC at e8 */
    set_macc_vtype(accum_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);
    DO_VXOR_V16();
    DO_MACC(INSN_VWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    /* Expected: each C element = K_eff * 1 * 1 = K_eff */
    int fail = 0;
    uint64_t mask = elem_mask(accum_sew);
    int64_t expected_val = k_eff;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t expected = (uint64_t)expected_val & mask;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s i4->e8 m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 4x widening vqwmmacc.vv with int4 inputs
 * ==================================================================== */

static int test_vqwmmacc_int4(int lmul, int lambda, uint64_t vlenb,
                              int altfmt_a, int altfmt_b, const char *label)
{
    int accum_sew = 2;  /* e16 accumulator */
    int W = 4;          /* quad-widening from 4-bit to 16-bit */
    int k_eff = lambda * W * lmul;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_load = lambda * lmul;

    if (M < W || mul_c != lmul) {
        return 0;
    }

    int N_active = M;
    int input_sew = 1;  /* load at e8 (bytes containing packed nibbles) */
    int a_ld = linesize_load;
    int b_ld = linesize_load;
    int c_ld = lambda * lmul;
    int mu_load = vlenb / (input_sew * lambda);
    int byte_count = mu_load * a_ld;

    uint8_t *A_buf = calloc(byte_count, 1);
    uint8_t *B_buf = calloc(byte_count, 1);
    uint8_t *C_dst = calloc(M * N, accum_sew);

    /* All nibbles = 1 */
    memset(A_buf, 0x11, byte_count);
    memset(B_buf, 0x11, byte_count);

    /* Load at e8 (byte level), reconfigure to e16 for MACC */
    set_macc_vtype(input_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    set_macc_vtype(accum_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_VQWMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;
    uint64_t mask = elem_mask(accum_sew);
    int64_t expected_val = k_eff;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t expected = (uint64_t)expected_val & mask;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s i4->e16 m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 8x widening v8wmmacc.vv — byte-aligned inputs (Int8→Int64)
 * ==================================================================== */

static int test_v8wmmacc(int accum_sew, int lmul, int lambda, uint64_t vlenb,
                         int altfmt_a, int altfmt_b, const char *label)
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

    uint8_t *A_buf  = calloc(mu_load * a_ld, input_sew);
    uint8_t *B_buf  = calloc(mu_load * b_ld, input_sew);
    uint8_t *C_dst  = calloc(c_nelem, accum_sew);

    for (int row = 0; row < M; row++) {
        for (int k = 0; k < K; k++) {
            int tr = W * row + k / linesize_load;
            int tc = k % linesize_load;
            int64_t val;
            if (altfmt_a) {
                val = ((row * K + k) % 200) + 1;
            } else {
                val = ((row * K + k) % 100) - 50;
            }
            write_elem(A_buf, tr * a_ld + tc, input_sew, (uint64_t)val);
        }
    }

    for (int col = 0; col < N; col++) {
        for (int k = 0; k < K; k++) {
            int tr = W * col + k / linesize_load;
            int tc = k % linesize_load;
            int64_t val;
            if (altfmt_b) {
                val = ((k * N + col) % 200) + 1;
            } else {
                val = ((k * N + col) % 100) - 50;
            }
            write_elem(B_buf, tr * b_ld + tc, input_sew, (uint64_t)val);
        }
    }

    /* Load A and B at input_sew */
    set_macc_vtype(input_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    /* Reconfigure to accum_sew, zero C, execute v8wmmacc, store C */
    set_macc_vtype(accum_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_V8WMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;
    uint64_t mask = elem_mask(accum_sew);

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int64_t acc = 0;
            for (int k = 0; k < K; k++) {
                int tr_a = W * row + k / linesize_load;
                int tc_a = k % linesize_load;
                int tr_b = W * col + k / linesize_load;
                int tc_b = k % linesize_load;
                int64_t a_val, b_val;

                if (altfmt_a) {
                    a_val = (int64_t)read_elem(A_buf,
                                tr_a * a_ld + tc_a, input_sew);
                } else {
                    a_val = read_signed(A_buf,
                                tr_a * a_ld + tc_a, input_sew);
                }
                if (altfmt_b) {
                    b_val = (int64_t)read_elem(B_buf,
                                tr_b * b_ld + tc_b, input_sew);
                } else {
                    b_val = read_signed(B_buf,
                                tr_b * b_ld + tc_b, input_sew);
                }
                acc += a_val * b_val;
            }

            int flat = row * M + col;
            uint64_t expected = (uint64_t)acc & mask;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s e%d->e%d m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, input_sew * 8, accum_sew * 8, lmul, lambda,
                       row, col,
                       (unsigned long)got, (unsigned long)expected);
                fail = 1;
            }
        }
    }

    free(A_buf); free(B_buf); free(C_dst);
    return fail;
}

/* ====================================================================
 * Test: 8x widening v8wmmacc.vv with int4 inputs (Int4→Int32)
 * ==================================================================== */

static int test_v8wmmacc_int4(int lmul, int lambda, uint64_t vlenb,
                              int altfmt_a, int altfmt_b, const char *label)
{
    int accum_sew = 4;  /* e32 accumulator */
    int W = 8;
    int k_eff = lambda * W * lmul;
    int M = vlenb / (accum_sew * lambda);
    int N = M;
    int mul_c = vlenb / (accum_sew * lambda * lambda);
    int linesize_load = lambda * lmul;

    if (M < W || mul_c != lmul) {
        return 0;
    }

    int N_active = M;
    int input_sew = 1;  /* load at e8 (bytes containing packed nibbles) */
    int a_ld = linesize_load;
    int b_ld = linesize_load;
    int c_ld = lambda * lmul;
    int mu_load = vlenb / (input_sew * lambda);
    int byte_count = mu_load * a_ld;

    uint8_t *A_buf = calloc(byte_count, 1);
    uint8_t *B_buf = calloc(byte_count, 1);
    uint8_t *C_dst = calloc(M * N, accum_sew);

    /* All nibbles = 1 */
    memset(A_buf, 0x11, byte_count);
    memset(B_buf, 0x11, byte_count);

    /* Load at e8, then reconfigure to e32 for MACC */
    set_macc_vtype(input_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_TILE(INSN_VMTL_V8, A_buf, (uint64_t)a_ld);
    DO_TILE(INSN_VMTL_V0, B_buf, (uint64_t)b_ld);

    set_macc_vtype(accum_sew, lmul, lambda, 0, altfmt_a, altfmt_b);
    DO_VXOR_V16();
    DO_MACC(INSN_V8WMMACC_VV);
    DO_TILE(INSN_VMTS_V16, C_dst, (uint64_t)c_ld);

    int fail = 0;
    uint64_t mask = elem_mask(accum_sew);
    int64_t expected_val = k_eff;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N_active; col++) {
            int flat = row * M + col;
            uint64_t expected = (uint64_t)expected_val & mask;
            uint64_t got = read_elem(C_dst, flat, accum_sew);
            if (got != expected) {
                printf("FAIL %s i4->e32 m%d l%d: C[%d][%d] "
                       "got 0x%lx exp 0x%lx\n",
                       label, lmul, lambda, row, col,
                       (unsigned long)got, (unsigned long)expected);
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

    static const int sews[]    = {1, 2, 4, 8};
    static const int lmuls[]   = {1, 2, 4, 8};
    static const int lambdas[] = {1, 2, 4, 8};

    int fail = 0;
    int tests = 0;

    for (int si = 0; si < 4; si++) {
        for (int li = 0; li < 4; li++) {
            for (int ai = 0; ai < 4; ai++) {
                int sew    = sews[si];
                int lmul   = lmuls[li];
                int lambda = lambdas[ai];

                /* TC-INT-01: vmmacc SS, C=0 */
                fail |= test_vmmacc(sew, lmul, lambda, vlenb,
                                    0, 0, 0, 0, "vmmacc SS C=0");
                tests++;

                /* TC-INT-02: vmmacc SS, accumulation */
                fail |= test_vmmacc(sew, lmul, lambda, vlenb,
                                    0, 0, 1, 0, "vmmacc SS accum");
                tests++;

                /* TC-INT-03: vmmacc SU */
                fail |= test_vmmacc(sew, lmul, lambda, vlenb,
                                    0, 1, 0, 0, "vmmacc SU");
                tests++;

                /* TC-INT-04: vmmacc US */
                fail |= test_vmmacc(sew, lmul, lambda, vlenb,
                                    1, 0, 0, 0, "vmmacc US");
                tests++;

                /* TC-INT-05: vmmacc UU */
                fail |= test_vmmacc(sew, lmul, lambda, vlenb,
                                    1, 1, 0, 0, "vmmacc UU");
                tests++;

                /* TC-INT-06: vmmacc SS, partial VL */
                fail |= test_vmmacc(sew, lmul, lambda, vlenb,
                                    0, 0, 0, 1, "vmmacc SS partial");
                tests++;
            }
        }
    }

    /* Widening tests: vwmmacc (W=2) */
    static const int wsews[] = {2, 4, 8}; /* accum_sew for W=2: e16,e32,e64 */
    for (int si = 0; si < 3; si++) {
        for (int li = 0; li < 4; li++) {
            for (int ai = 0; ai < 4; ai++) {
                int sew    = wsews[si];
                int lmul   = lmuls[li];
                int lambda = lambdas[ai];

                /* TC-INT-07..09: vwmmacc SS */
                fail |= test_vwmmacc(sew, lmul, lambda, vlenb,
                                     0, 0, "vwmmacc SS");
                tests++;

                /* vwmmacc SU */
                fail |= test_vwmmacc(sew, lmul, lambda, vlenb,
                                     0, 1, "vwmmacc SU");
                tests++;

                /* vwmmacc US */
                fail |= test_vwmmacc(sew, lmul, lambda, vlenb,
                                     1, 0, "vwmmacc US");
                tests++;

                /* vwmmacc UU */
                fail |= test_vwmmacc(sew, lmul, lambda, vlenb,
                                     1, 1, "vwmmacc UU");
                tests++;
            }
        }
    }

    /* Quad-widening tests: vqwmmacc (W=4) */
    static const int qsews[] = {4, 8}; /* accum_sew for W=4: e32, e64 */
    for (int si = 0; si < 2; si++) {
        for (int li = 0; li < 4; li++) {
            for (int ai = 0; ai < 4; ai++) {
                int sew    = qsews[si];
                int lmul   = lmuls[li];
                int lambda = lambdas[ai];

                /* TC-INT-14..15: vqwmmacc SS */
                fail |= test_vqwmmacc(sew, lmul, lambda, vlenb,
                                      0, 0, "vqwmmacc SS");
                tests++;

                /* vqwmmacc SU */
                fail |= test_vqwmmacc(sew, lmul, lambda, vlenb,
                                      0, 1, "vqwmmacc SU");
                tests++;

                /* vqwmmacc US */
                fail |= test_vqwmmacc(sew, lmul, lambda, vlenb,
                                      1, 0, "vqwmmacc US");
                tests++;

                /* vqwmmacc UU */
                fail |= test_vqwmmacc(sew, lmul, lambda, vlenb,
                                      1, 1, "vqwmmacc UU");
                tests++;
            }
        }
    }

    /* Int4 widening tests */
    for (int li = 0; li < 4; li++) {
        for (int ai = 0; ai < 4; ai++) {
            int lmul   = lmuls[li];
            int lambda = lambdas[ai];

            /* TC-INT-10: vwmmacc i4->e8 SS */
            fail |= test_vwmmacc_int4(lmul, lambda, vlenb,
                                      0, 0, "vwmmacc-i4 SS");
            tests++;

            /* TC-INT-11: vwmmacc i4->e8 SU */
            fail |= test_vwmmacc_int4(lmul, lambda, vlenb,
                                      0, 1, "vwmmacc-i4 SU");
            tests++;

            /* TC-INT-12: vwmmacc i4->e8 US */
            fail |= test_vwmmacc_int4(lmul, lambda, vlenb,
                                      1, 0, "vwmmacc-i4 US");
            tests++;

            /* TC-INT-13: vwmmacc i4->e8 UU */
            fail |= test_vwmmacc_int4(lmul, lambda, vlenb,
                                      1, 1, "vwmmacc-i4 UU");
            tests++;

            /* TC-INT-16: vqwmmacc i4->e16 SS */
            fail |= test_vqwmmacc_int4(lmul, lambda, vlenb,
                                       0, 0, "vqwmmacc-i4 SS");
            tests++;

            /* vqwmmacc i4->e16 SU */
            fail |= test_vqwmmacc_int4(lmul, lambda, vlenb,
                                       0, 1, "vqwmmacc-i4 SU");
            tests++;

            /* vqwmmacc i4->e16 US */
            fail |= test_vqwmmacc_int4(lmul, lambda, vlenb,
                                       1, 0, "vqwmmacc-i4 US");
            tests++;

            /* TC-INT-17: vqwmmacc i4->e16 UU */
            fail |= test_vqwmmacc_int4(lmul, lambda, vlenb,
                                       1, 1, "vqwmmacc-i4 UU");
            tests++;

            /* TC-INT-22: v8wmmacc i4->e32 SS */
            fail |= test_v8wmmacc_int4(lmul, lambda, vlenb,
                                       0, 0, "v8wmmacc-i4 SS");
            tests++;

            /* v8wmmacc i4->e32 SU */
            fail |= test_v8wmmacc_int4(lmul, lambda, vlenb,
                                       0, 1, "v8wmmacc-i4 SU");
            tests++;

            /* v8wmmacc i4->e32 US */
            fail |= test_v8wmmacc_int4(lmul, lambda, vlenb,
                                       1, 0, "v8wmmacc-i4 US");
            tests++;

            /* v8wmmacc i4->e32 UU */
            fail |= test_v8wmmacc_int4(lmul, lambda, vlenb,
                                       1, 1, "v8wmmacc-i4 UU");
            tests++;
        }
    }

    /* Octet-widening tests: v8wmmacc (W=8), byte-aligned (Int8→Int64) */
    /* Only accum_sew=8 (e64) is valid for byte inputs with W=8 */
    for (int li = 0; li < 4; li++) {
        for (int ai = 0; ai < 4; ai++) {
            int lmul   = lmuls[li];
            int lambda = lambdas[ai];

            /* TC-INT-18: v8wmmacc SS */
            fail |= test_v8wmmacc(8, lmul, lambda, vlenb,
                                  0, 0, "v8wmmacc SS");
            tests++;

            /* TC-INT-19: v8wmmacc SU */
            fail |= test_v8wmmacc(8, lmul, lambda, vlenb,
                                  0, 1, "v8wmmacc SU");
            tests++;

            /* TC-INT-20: v8wmmacc US */
            fail |= test_v8wmmacc(8, lmul, lambda, vlenb,
                                  1, 0, "v8wmmacc US");
            tests++;

            /* TC-INT-21: v8wmmacc UU */
            fail |= test_v8wmmacc(8, lmul, lambda, vlenb,
                                  1, 1, "v8wmmacc UU");
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
