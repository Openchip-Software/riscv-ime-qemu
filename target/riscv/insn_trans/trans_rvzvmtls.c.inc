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

#define REQUIRE_ZVVMTLS(ctx) do {               \
    if (!ctx->cfg_ptr->ext_zvvmtls) {           \
        return false;                           \
    }                                           \
} while (0)

typedef void gen_helper_ldst_tile(TCGv_ptr, TCGv_ptr, TCGv, TCGv,
                                  TCGv_env, TCGv_i32);

static bool ldst_tile_trans(uint32_t vd, uint32_t rs1, uint32_t rs2,
                            uint32_t data,
                            gen_helper_ldst_tile *fn,
                            DisasContext *s)
{
    TCGv_ptr dest, mask;
    TCGv base, stride;
    TCGv_i32 desc;

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);
    stride = get_gpr(s, rs2, EXT_NONE);
    desc = tcg_constant_i32(simd_desc(s->cfg_ptr->vlenb,
                                      s->cfg_ptr->vlenb, data));

    tcg_gen_addi_ptr(dest, tcg_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(mask, tcg_env, vreg_ofs(s, 0));

    mark_vs_dirty(s);

    fn(dest, mask, base, stride, tcg_env, desc);

    finalize_rvv_inst(s);
    return true;
}


/* TODO: use a common function that is passed the gen_helper_ldst_tile *fn ... */

static bool trans_vmtl_v(DisasContext *s, arg_vmtl_v *a)
{
    REQUIRE_ZVVMTLS(s);

    /* TODO: checks */

    gen_helper_ldst_tile *fn;
    static gen_helper_ldst_tile * const fns[4] = {
        gen_helper_vmtl8_v, gen_helper_vmtl16_v,
        gen_helper_vmtl32_v, gen_helper_vmtl64_v
    };

    fn = fns[s->sew];
    if (fn == NULL) {
        return false;
    }

    uint32_t data = 0;
    data = FIELD_DP32(data, VDATA, VM, a->vm);
    data = FIELD_DP32(data, VDATA, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA, VTA, s->vta);
    data = FIELD_DP32(data, VDATA, VMA, s->vma);
    data = FIELD_DP32(data, VDATA, VLAMBDA, a->lambda);

    return ldst_tile_trans(a->rd, a->rs1, a->rs2, data, fn, s);
}

static bool trans_vmttl_v(DisasContext *s, arg_vmttl_v *a)
{
    REQUIRE_ZVVMTLS(s);

    /* TODO: checks */

    gen_helper_ldst_tile *fn;
    static gen_helper_ldst_tile * const fns[4] = {
        gen_helper_vmttl8_v, gen_helper_vmttl16_v,
        gen_helper_vmttl32_v, gen_helper_vmttl64_v
    };

    fn = fns[s->sew];
    if (fn == NULL) {
        return false;
    }

    uint32_t data = 0;
    data = FIELD_DP32(data, VDATA, VM, a->vm);
    data = FIELD_DP32(data, VDATA, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA, VTA, s->vta);
    data = FIELD_DP32(data, VDATA, VMA, s->vma);
    data = FIELD_DP32(data, VDATA, VLAMBDA, a->lambda);

    return ldst_tile_trans(a->rd, a->rs1, a->rs2, data, fn, s);
}

static bool trans_vmts_v(DisasContext *s, arg_vmts_v *a)
{
    REQUIRE_ZVVMTLS(s);

    /* TODO: checks */

    gen_helper_ldst_tile *fn;
    static gen_helper_ldst_tile * const fns[4] = {
        gen_helper_vmts8_v,  gen_helper_vmts16_v,
        gen_helper_vmts32_v, gen_helper_vmts64_v
    };

    fn = fns[s->sew];
    if (fn == NULL) {
        return false;
    }

    uint32_t data = 0;
    data = FIELD_DP32(data, VDATA, VM, a->vm);
    data = FIELD_DP32(data, VDATA, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA, VTA, s->vta);
    data = FIELD_DP32(data, VDATA, VMA, s->vma);
    data = FIELD_DP32(data, VDATA, VLAMBDA, a->lambda);

    return ldst_tile_trans(a->rd, a->rs1, a->rs2, data, fn, s);
}

static bool trans_vmtts_v(DisasContext *s, arg_vmtts_v *a)
{
    REQUIRE_ZVVMTLS(s);

    /* TODO: checks */

    gen_helper_ldst_tile *fn;
    static gen_helper_ldst_tile * const fns[4] = {
        gen_helper_vmtts8_v,  gen_helper_vmtts16_v,
        gen_helper_vmtts32_v, gen_helper_vmtts64_v
    };

    fn = fns[s->sew];
    if (fn == NULL) {
        return false;
    }

    uint32_t data = 0;
    data = FIELD_DP32(data, VDATA, VM, a->vm);
    data = FIELD_DP32(data, VDATA, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA, VTA, s->vta);
    data = FIELD_DP32(data, VDATA, VMA, s->vma);
    data = FIELD_DP32(data, VDATA, VLAMBDA, a->lambda);

    return ldst_tile_trans(a->rd, a->rs1, a->rs2, data, fn, s);
}
