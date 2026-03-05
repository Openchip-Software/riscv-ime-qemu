/*
 * RISC-V Zvvmtls Tile Load/Store Unit Test
 *
 * Validates the tile register layout (index_to_regoffset mapping) and the
 * preserve-order vs transpose memory addressing through round-trip and
 * cross-instruction checks for vmtl, vmttl, vmts, vmtts.
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
 * Format: lambda[31:29] func[28:26] vm[25] rs2[24:20] rs1[19:15]
 *         111[14:12] vd[11:7] opcode[6:0]
 *
 * lambda=0 (DYNAMIC: from vtype), vm=1 (unmasked),
 * vd=v8, rs1=a0(x10), rs2=a1(x11)
 * ------------------------------------------------------------------ */

#define INSN_VMTL_V8   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x07u)
#define INSN_VMTTL_V8  (5u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x07u)
#define INSN_VMTS_V8   (4u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x27u)
#define INSN_VMTTS_V8  (5u << 26 | 1u << 25 | 11u << 20 | 10u << 15 | \
                         7u << 12 |  8u <<  7 | 0x27u)

/* --------------------------------------------------------------------
 * Inline-asm helpers
 * ------------------------------------------------------------------ */

/*
 * Set vtype for tile operations: sew/lmul/lambda via vsetvl.
 * Returns VL (== VLMAX since AVL is max via rs1=zero).
 */
static inline uint64_t set_tile_vtype(int sew_bytes, int lmul, int lambda)
{
    int vsew    = __builtin_ctz(sew_bytes); /* 0=e8,1=e16,2=e32,3=e64 */
    int vlmul   = __builtin_ctz(lmul);      /* 0=m1,1=m2,2=m4,3=m8   */
    int vlambda = __builtin_ctz(lambda) + 1; /* 1=l1,2=l2,3=l4,4=l8   */

    uint64_t vtype = (uint64_t)vlmul
                   | ((uint64_t)vsew << 3)
                   | ((uint64_t)vlambda << 60);
    uint64_t vl;
    asm volatile("vsetvl %0, zero, %1" : "=r"(vl) : "r"(vtype));
    return vl;
}

/* Set vtype for raw byte dump/load: e8, matching lmul, VL = VLMAX */
static inline void set_e8_vtype(int lmul)
{
    uint64_t vtype = (uint64_t)__builtin_ctz(lmul); /* e8, lambda=0 */
    uint64_t vl;
    asm volatile("vsetvl %0, zero, %1" : "=r"(vl) : "r"(vtype));
}

/* Emit a tile instruction via .4byte; base in a0, stride in a1 */
#define DO_TILE(insn, base_ptr, stride_val) do {                    \
    register void *_b asm("a0") = (void *)(uintptr_t)(base_ptr);   \
    register uint64_t _s asm("a1") = (uint64_t)(stride_val);       \
    asm volatile(".4byte %[i]"                                      \
                 : : [i] "i"(insn), "r"(_b), "r"(_s) : "memory");  \
} while (0)

static inline void do_vmtl(const void *b, uint64_t ld)
                                        { DO_TILE(INSN_VMTL_V8, b, ld); }
static inline void do_vmttl(const void *b, uint64_t ld)
                                        { DO_TILE(INSN_VMTTL_V8, b, ld); }
static inline void do_vmts(void *b, uint64_t ld)
                                        { DO_TILE(INSN_VMTS_V8, b, ld); }
static inline void do_vmtts(void *b, uint64_t ld)
                                        { DO_TILE(INSN_VMTTS_V8, b, ld); }

/* Raw register dump/load using standard vse8/vle8 (vtype must be set) */
static inline void raw_dump_v8(void *buf)
{
    asm volatile("vse8.v v8, (%0)" : : "r"(buf) : "memory");
}

/* --------------------------------------------------------------------
 * Element accessors for arbitrary SEW
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
 * Reference: index_to_regoffset  (must match vector_helper.c)
 * ------------------------------------------------------------------ */

static void index_to_regoffset(int i, int lmul, int lambda,
                               int *regoff, int *elemoff)
{
    int linesize     = lambda * lmul;
    int linenum      = i / linesize;
    int elem_in_line = i % linesize;

    *regoff  = elem_in_line / lambda;
    *elemoff = linenum * lambda + elem_in_line % lambda;
}

/* ====================================================================
 * Test cases
 * ==================================================================== */

/*
 * TC-1 (transpose=0): vmtl + vmts  round-trip
 * TC-2 (transpose=1): vmttl + vmtts round-trip
 *
 * Fill src[] with unique values, load tile from src, store tile to dst.
 * Verify dst[] == src[].
 */
static int test_roundtrip(int sew, int lmul, int lambda,
                          uint64_t vlenb, int transpose)
{
    int mu       = vlenb / (sew * lambda);
    int linesize = lambda * lmul;
    int nelem    = mu * linesize;
    int bufsz    = nelem * sew;
    int ld       = transpose ? mu : linesize;

    uint8_t *src = malloc(bufsz);
    uint8_t *dst = calloc(1, bufsz);

    for (int i = 0; i < nelem; i++) {
        write_elem(src, i, sew, (uint64_t)(i + 1));
    }

    set_tile_vtype(sew, lmul, lambda);
    if (!transpose) {
        do_vmtl(src, ld);
        do_vmts(dst, ld);
    } else {
        do_vmttl(src, ld);
        do_vmtts(dst, ld);
    }

    int fail = 0;
    for (int i = 0; i < nelem; i++) {
        uint64_t got = read_elem(dst, i, sew);
        uint64_t exp = read_elem(src, i, sew);
        if (got != exp) {
            printf("FAIL roundtrip(%s) e%d m%d l%d: elem %d "
                   "got 0x%lx exp 0x%lx\n",
                   transpose ? "T" : "N", sew * 8, lmul, lambda,
                   i, (unsigned long)got, (unsigned long)exp);
            fail = 1;
        }
    }

    free(src);
    free(dst);
    return fail;
}

/*
 * TC-3 (load_transpose=0): vmtl + vmtts  (preserve load, transpose store)
 * TC-4 (load_transpose=1): vmttl + vmts  (transpose load, preserve store)
 *
 * Both perform a matrix transpose:
 *   Case 0: tile[i]=src[i], dst[perm(i)]=tile[i]  =>  dst[perm(i)]=src[i]
 *   Case 1: tile[i]=src[perm(i)], dst[i]=tile[i]  =>  dst[i]=src[perm(i)]
 * where perm(i) = (i%linesize)*mu + (i/linesize)  (the transpose index).
 */
static int test_cross(int sew, int lmul, int lambda,
                      uint64_t vlenb, int load_transpose)
{
    int mu       = vlenb / (sew * lambda);
    int linesize = lambda * lmul;
    int nelem    = mu * linesize;
    int bufsz    = nelem * sew;

    uint8_t *src = malloc(bufsz);
    uint8_t *dst = calloc(1, bufsz);

    for (int i = 0; i < nelem; i++) {
        write_elem(src, i, sew, (uint64_t)(i + 1));
    }

    set_tile_vtype(sew, lmul, lambda);
    if (!load_transpose) {
        do_vmtl(src, linesize);     /* preserve-order load  */
        do_vmtts(dst, mu);          /* transpose store      */
    } else {
        do_vmttl(src, mu);          /* transpose load       */
        do_vmts(dst, linesize);     /* preserve-order store */
    }

    int fail = 0;
    for (int i = 0; i < nelem; i++) {
        int line = i / linesize;
        int elem = i % linesize;
        int j    = elem * mu + line;  /* transposed flat index */

        uint64_t got, exp;
        if (!load_transpose) {
            /* dst[j] == src[i] */
            got = read_elem(dst, j, sew);
            exp = read_elem(src, i, sew);
        } else {
            /* dst[i] == src[j] */
            got = read_elem(dst, i, sew);
            exp = read_elem(src, j, sew);
        }
        if (got != exp) {
            printf("FAIL cross(%s) e%d m%d l%d: i=%d j=%d "
                   "got 0x%lx exp 0x%lx\n",
                   load_transpose ? "TN" : "NT",
                   sew * 8, lmul, lambda,
                   i, j, (unsigned long)got, (unsigned long)exp);
            fail = 1;
        }
    }

    free(src);
    free(dst);
    return fail;
}

/*
 * TC-5: Register layout check.
 *
 * vmtl loads contiguous source data into the tile register group.
 * Dump the raw register bytes with vse8.v and verify each element is
 * at the position predicted by index_to_regoffset.
 */
static int test_register_layout(int sew, int lmul, int lambda,
                                uint64_t vlenb)
{
    int mu       = vlenb / (sew * lambda);
    int linesize = lambda * lmul;
    int nelem    = mu * linesize;
    int bufsz    = nelem * sew;
    int rawsz    = lmul * (int)vlenb;

    uint8_t *src = malloc(bufsz);
    uint8_t *raw = calloc(1, rawsz);

    for (int i = 0; i < nelem; i++) {
        write_elem(src, i, sew, (uint64_t)(i + 1));
    }

    /* vmtl with preserve-order and ld=linesize reads src contiguously */
    set_tile_vtype(sew, lmul, lambda);
    do_vmtl(src, linesize);

    /* Dump raw register bytes */
    set_e8_vtype(lmul);
    raw_dump_v8(raw);

    int fail = 0;
    for (int i = 0; i < nelem; i++) {
        int regoff, elemoff;
        index_to_regoffset(i, lmul, lambda, &regoff, &elemoff);

        uint64_t got = read_elem(raw + regoff * vlenb, elemoff, sew);
        uint64_t exp = (uint64_t)(i + 1);
        if (got != exp) {
            printf("FAIL layout e%d m%d l%d: index %d "
                   "regoff=%d elemoff=%d got 0x%lx exp 0x%lx\n",
                   sew * 8, lmul, lambda, i,
                   regoff, elemoff,
                   (unsigned long)got, (unsigned long)exp);
            fail = 1;
        }
    }

    free(src);
    free(raw);
    return fail;
}

/* ====================================================================
 * Main: iterate over all valid (sew, lmul, lambda) configurations
 * ==================================================================== */

int main(void)
{
    uint64_t vlenb;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));

    static const int sews[]    = {1, 2, 4, 8};
    static const int lmuls[]   = {1, 2, 4};
    static const int lambdas[] = {1, 2, 4, 8};

    int fail = 0;
    int configs = 0;

    for (int si = 0; si < 4; si++) {
        for (int li = 0; li < 3; li++) {
            for (int ai = 0; ai < 4; ai++) {
                int sew    = sews[si];
                int lmul   = lmuls[li];
                int lambda = lambdas[ai];
                int mu     = (int)vlenb / (sew * lambda);

                if (mu < 1) {
                    continue;
                }

                fail |= test_roundtrip(sew, lmul, lambda, vlenb, 0);
                fail |= test_roundtrip(sew, lmul, lambda, vlenb, 1);
                fail |= test_cross(sew, lmul, lambda, vlenb, 0);
                fail |= test_cross(sew, lmul, lambda, vlenb, 1);
                fail |= test_register_layout(sew, lmul, lambda, vlenb);
                configs++;
            }
        }
    }

    if (fail) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED (%d configurations, %d tests)\n", configs, configs * 5);
    return 0;
}
