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
#include <signal.h>
#include <ucontext.h>

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
        uint64_t exp = read_elem(src, i, sew);  /* truncated to sew bytes */
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

/*
 * TC-6: rs2=0 selects the spec-defined default leading dimension.
 *
 *   vmtl.v  / vmts.v   (preserve): LD_default = λ × LMUL = linesize
 *   vmttl.v / vmtts.v  (transpose): LD_default = VLEN / (SEW × λ) = mu
 *
 * The explicit-LD round-trip is already covered by TC-1 / TC-2 with the
 * same LD values.  Here we re-run the round-trip passing rs2=0 in place
 * of the explicit LD and require the output to match src[] byte for byte.
 * A failure here means the helper did not substitute the default.
 */
static int test_default_ld(int sew, int lmul, int lambda,
                           uint64_t vlenb, int transpose)
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
    if (!transpose) {
        do_vmtl(src, 0);    /* LD=0 → λ × LMUL */
        do_vmts(dst, 0);
    } else {
        do_vmttl(src, 0);   /* LD=0 → VLEN/(SEW×λ) */
        do_vmtts(dst, 0);
    }

    int fail = 0;
    for (int i = 0; i < nelem; i++) {
        uint64_t got = read_elem(dst, i, sew);
        uint64_t exp = read_elem(src, i, sew);
        if (got != exp) {
            printf("FAIL default_ld(%s) e%d m%d l%d: elem %d "
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
 * TC-7: with vtype.vill=1 every tile load/store must raise an
 * illegal-instruction exception, as base-V loads/stores do.  The SIGILL
 * handler counts the trap and skips the 4-byte instruction.
 */
static volatile int sigill_count;

static void sigill_handler(int sig, siginfo_t *si, void *uc_)
{
    ucontext_t *uc = uc_;

    sigill_count++;
    uc->uc_mcontext.__gregs[REG_PC] += 4;
}

static int test_vill(void)
{
    static uint8_t buf[64];
    struct sigaction sa = {
        .sa_sigaction = sigill_handler,
        .sa_flags = SA_SIGINFO,
    };
    struct sigaction old;
    uint64_t vl;

    sigaction(SIGILL, &sa, &old);
    sigill_count = 0;

    /* Writing an unsupported vtype sets vill. */
    asm volatile("vsetvl %0, zero, %1" : "=r"(vl) : "r"(-1L));
    do_vmtl(buf, 0);
    do_vmttl(buf, 0);
    do_vmts(buf, 0);
    do_vmtts(buf, 0);

    sigaction(SIGILL, &old, NULL);

    if (sigill_count != 4) {
        printf("FAIL vill: %d of 4 tile load/stores trapped\n",
               sigill_count);
        return 1;
    }
    return 0;
}

/*
 * A masked tile load whose destination overlaps v0 must raise an
 * illegal-instruction exception; a masked tile store may use v0 as its
 * source, and a masked load into another group is legal.
 */
#define MASKED(insn)        ((insn) & ~(1u << 25))
#define WITH_VD_V0(insn)    ((insn) & ~(31u << 7))

static int test_masked_v0(void)
{
    uint64_t vlenb;
    struct sigaction sa = {
        .sa_sigaction = sigill_handler,
        .sa_flags = SA_SIGINFO,
    };
    struct sigaction old;
    int fail = 0;

    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    uint8_t *buf = calloc(1, vlenb);

    set_tile_vtype(1, 1, 1);
    sigaction(SIGILL, &sa, &old);

    sigill_count = 0;
    DO_TILE(WITH_VD_V0(MASKED(INSN_VMTL_V8)), buf, 0);
    DO_TILE(WITH_VD_V0(MASKED(INSN_VMTTL_V8)), buf, 0);
    if (sigill_count != 2) {
        printf("FAIL masked v0: %d of 2 loads into v0 trapped\n",
               sigill_count);
        fail = 1;
    }

    sigill_count = 0;
    DO_TILE(WITH_VD_V0(MASKED(INSN_VMTS_V8)), buf, 0);
    DO_TILE(WITH_VD_V0(MASKED(INSN_VMTTS_V8)), buf, 0);
    DO_TILE(MASKED(INSN_VMTL_V8), buf, 0);
    DO_TILE(MASKED(INSN_VMTTL_V8), buf, 0);
    if (sigill_count != 0) {
        printf("FAIL masked v0: %d of 4 legal masked tile ops trapped\n",
               sigill_count);
        fail = 1;
    }

    sigaction(SIGILL, &old, NULL);
    free(buf);
    return fail;
}

/*
 * A nonzero lambda immediate is not clamped: if the CPU does not support
 * it (lambda_mask, bit log2(lambda)) or VLEN / (SEW * lambda^2) is not in
 * {1, 2, 4, 8, 16}, the tile load raises an illegal-instruction exception.
 */
#define WITH_LAMBDA(insn, l)    ((insn) | (uint32_t)(l) << 29)

static void do_vmtl_lambda(int lambda_enc, const void *b, uint64_t ld)
{
    switch (lambda_enc) {
    case 1:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 1), b, ld);
        break;
    case 2:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 2), b, ld);
        break;
    case 3:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 3), b, ld);
        break;
    case 4:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 4), b, ld);
        break;
    case 5:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 5), b, ld);
        break;
    case 6:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 6), b, ld);
        break;
    case 7:
        DO_TILE(WITH_LAMBDA(INSN_VMTL_V8, 7), b, ld);
        break;
    }
}

static int test_lambda_imm(long lambda_mask)
{
    uint64_t vlenb;
    struct sigaction sa = {
        .sa_sigaction = sigill_handler,
        .sa_flags = SA_SIGINFO,
    };
    struct sigaction old;
    int fail = 0;

    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    uint8_t *buf = calloc(1, vlenb);

    sigaction(SIGILL, &sa, &old);
    for (int sew = 1; sew <= 8; sew *= 2) {
        for (int enc = 1; enc <= 7; enc++) {
            uint64_t lambda = 1 << (enc - 1);
            uint64_t mul_c_x16 = vlenb * 8 * 16 / (sew * 8 * lambda * lambda);
            int legal = mul_c_x16 >= 16 && mul_c_x16 <= 256 &&
                        (lambda_mask & lambda);

            set_tile_vtype(sew, 1, 1);
            sigill_count = 0;
            do_vmtl_lambda(enc, buf, lambda);
            if (sigill_count != !legal) {
                printf("FAIL lambda imm: e%d L%lu %s\n", sew * 8,
                       (unsigned long)lambda,
                       legal ? "trapped" : "did not trap");
                fail = 1;
            }
        }
    }
    sigaction(SIGILL, &old, NULL);
    free(buf);
    return fail;
}

/*
 * VL must be a multiple of lambda * LMUL; otherwise every tile load/store
 * raises an illegal-instruction exception.  VL = 0 is a multiple.
 */
static int test_vl_multiple(void)
{
    static const struct {
        int lmul, lambda;
        uint64_t avl;
    } cases[] = {
        { 1, 2, 4 }, { 1, 2, 3 }, { 1, 2, 0 },
        { 2, 1, 4 }, { 2, 1, 3 }, { 2, 2, 4 }, { 2, 2, 6 },
    };
    uint64_t vlenb;
    struct sigaction sa = {
        .sa_sigaction = sigill_handler,
        .sa_flags = SA_SIGINFO,
    };
    struct sigaction old;
    int fail = 0;

    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    uint8_t *buf = calloc(1, 2 * vlenb);

    sigaction(SIGILL, &sa, &old);
    for (int i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint64_t vtype = (uint64_t)__builtin_ctz(cases[i].lmul) | 2 << 3 |
                         (uint64_t)(__builtin_ctz(cases[i].lambda) + 1) << 60;
        uint64_t vl;
        int legal;

        asm volatile("vsetvl %0, %1, %2"
                     : "=r"(vl) : "r"(cases[i].avl), "r"(vtype));
        legal = vl % (cases[i].lambda * cases[i].lmul) == 0;

        sigill_count = 0;
        do_vmtl(buf, 0);
        do_vmttl(buf, 0);
        do_vmts(buf, 0);
        do_vmtts(buf, 0);
        if (sigill_count != (legal ? 0 : 4)) {
            printf("FAIL vl multiple: e32 m%d l%d vl=%lu: %d of 4 trapped\n",
                   cases[i].lmul, cases[i].lambda, (unsigned long)vl,
                   sigill_count);
            fail = 1;
        }
    }
    sigaction(SIGILL, &old, NULL);
    free(buf);
    return fail;
}

/*
 * Tile loads leave inactive and tail elements undisturbed even with
 * vta = vma = 1, and tile stores never write their source register group.
 * QEMU only applies the all-ones fill with rvv_ta_all_1s/rvv_ma_all_1s,
 * so this is also run with those options (see Makefile.target).
 */
static void set_e32_vtype(uint64_t avl, int agnostic)
{
    uint64_t vtype = 2 << 3 | 1ULL << 60 | (agnostic ? 3 << 6 : 0);
    uint64_t vl;

    asm volatile("vsetvl %0, %1, %2" : "=r"(vl) : "r"(avl), "r"(vtype));
}

static int test_undisturbed(void)
{
    static const struct {
        const char *name;
        uint32_t insn;
        int load;
    } ops[] = {
        { "vmtl",  INSN_VMTL_V8,  1 }, { "vmttl", INSN_VMTTL_V8, 1 },
        { "vmts",  INSN_VMTS_V8,  0 }, { "vmtts", INSN_VMTTS_V8, 0 },
    };
    uint64_t vlenb;
    int fail = 0;

    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    uint32_t vlmax = vlenb / 4, vl = vlmax / 2;
    uint32_t *orig = calloc(vlmax, 4), *mem = calloc(vlmax, 4);
    uint32_t *out = calloc(vlmax, 4);

    for (uint32_t i = 0; i < vlmax; i++) {
        orig[i] = 0x1000 + i;
    }

    for (int op = 0; op < 4; op++) {
        for (int masked = 0; masked < 2; masked++) {
            for (uint32_t i = 0; i < vlmax; i++) {
                mem[i] = 0x2000 + i;
            }
            set_e32_vtype(vlmax, 0);
            asm volatile("vle32.v v8, (%0)" : : "r"(orig) : "memory");
            asm volatile("vsetvl zero, %0, %1\n\tvmv.v.x v0, %2"
                         : : "r"(vlenb), "r"(0L), "r"(0x55));

            /* VL = VLMAX / 2, e32, m1, lambda = 1, vta = vma = 1 */
            set_e32_vtype(vl, 1);
            switch (op * 2 + masked) {
            case 0:
                DO_TILE(INSN_VMTL_V8, mem, 0);
                break;
            case 1:
                DO_TILE(MASKED(INSN_VMTL_V8), mem, 0);
                break;
            case 2:
                DO_TILE(INSN_VMTTL_V8, mem, 0);
                break;
            case 3:
                DO_TILE(MASKED(INSN_VMTTL_V8), mem, 0);
                break;
            case 4:
                DO_TILE(INSN_VMTS_V8, mem, 0);
                break;
            case 5:
                DO_TILE(MASKED(INSN_VMTS_V8), mem, 0);
                break;
            case 6:
                DO_TILE(INSN_VMTTS_V8, mem, 0);
                break;
            case 7:
                DO_TILE(MASKED(INSN_VMTTS_V8), mem, 0);
                break;
            }

            set_e32_vtype(vlmax, 0);
            asm volatile("vse32.v v8, (%0)" : : "r"(out) : "memory");

            for (uint32_t i = 0; i < vlmax; i++) {
                int active = i < vl && (!masked || !(i & 1));
                uint32_t exp = ops[op].load && active ? 0x2000 + i : orig[i];

                if (out[i] != exp) {
                    printf("FAIL undisturbed: %s%s elem %u got 0x%x exp 0x%x\n",
                           ops[op].name, masked ? " v0.t" : "", i, out[i],
                           exp);
                    fail = 1;
                    break;
                }
            }
        }
    }

    free(orig);
    free(mem);
    free(out);
    return fail;
}

/* ====================================================================
 * Main: iterate over all valid (sew, lmul, lambda) configurations
 * ==================================================================== */

/*
 * argv[1], if given, is the set of lambdas the CPU supports, as for the
 * zvvm-lambda CPU property (bit n = lambda 2^n); the default is all.
 */
int main(int argc, char **argv)
{
    long lambda_mask = 0x7f;
    uint64_t vlenb;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));

    if (argc > 1 && sscanf(argv[1], "%li", &lambda_mask) != 1) {
        printf("usage: %s [supported-lambda-mask]\n", argv[0]);
        return 1;
    }

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

                if (mu < 1 || !(lambda_mask & lambda)) {
                    continue;
                }

                fail |= test_roundtrip(sew, lmul, lambda, vlenb, 0);
                fail |= test_roundtrip(sew, lmul, lambda, vlenb, 1);
                fail |= test_cross(sew, lmul, lambda, vlenb, 0);
                fail |= test_cross(sew, lmul, lambda, vlenb, 1);
                fail |= test_register_layout(sew, lmul, lambda, vlenb);
                fail |= test_default_ld(sew, lmul, lambda, vlenb, 0);
                fail |= test_default_ld(sew, lmul, lambda, vlenb, 1);
                configs++;
            }
        }
    }

    fail |= test_vill();
    fail |= test_masked_v0();
    fail |= test_lambda_imm(lambda_mask);
    fail |= test_vl_multiple();
    fail |= test_undisturbed();

    if (fail) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED (%d configurations, %d tests)\n", configs,
           configs * 7 + 1);
    return 0;
}
