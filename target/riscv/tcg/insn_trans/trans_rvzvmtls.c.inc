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
                            uint32_t data, bool is_load,
                            gen_helper_ldst_tile *fn,
                            DisasContext *s)
{
    TCGv_ptr dest, mask;
    TCGv base, stride;
    TCGv_i32 desc;

    /*
     * Tile loads/stores follow the standard vector rules: they trap when
     * mstatus.VS is Off or vtype.vill is set.
     */
    if (!require_rvv(s) || !vext_check_isa_ill(s)) {
        return false;
    }

    /*
     * Spec commit 078b98ad (2026-06-15) makes fractional LMUL invalid for
     * every Zvvm instruction including tile load/store.  s->lmul is signed
     * (negative encodes mf2/mf4/mf8); reject those before we touch the
     * helper, which would otherwise compute `1 << s->lmul` on a negative
     * value (undefined behaviour).
     */
    if (s->lmul < 0) {
        return false;
    }

    /*
     * Register-group alignment: vd must be aligned to LMUL.  rs1/rs2 are
     * scalar GPR sources and do not participate in register-group sizing.
     */
    if (!require_align(vd, s->lmul)) {
        return false;
    }

    /*
     * The destination of a masked tile load must not overlap v0.  The
     * group is LMUL-aligned, so it overlaps v0 exactly when vd == 0.  A
     * masked tile store may use v0 as its source, as in base V.
     */
    if (is_load && !require_vm(FIELD_EX32(data, VDATA, VM), vd)) {
        return false;
    }

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


/*
 * vmtl.v / vmttl.v / vmts.v / vmtts.v all share the same encoding shape
 * (arg_rmrr_lambda — rd, rs1, rs2, vm, lambda) and the same translation
 * body: pick the SEW-indexed helper, build the descriptor, dispatch via
 * ldst_tile_trans.  The only per-instruction variation is the helper-name
 * prefix, so generate the four translators from one macro.
 *
 * Per-instruction legality checks (vstart=0, lambda-immediate trap on
 * unsupported value, VL divisibility, ...) belong in ldst_tile_trans where
 * they apply uniformly to all four forms; per-translator hooks are not
 * needed today.
 */
#define GEN_LDST_TILE_TRANS(NAME, IS_LOAD)                                   \
static bool trans_##NAME##_v(DisasContext *s, arg_##NAME##_v *a)             \
{                                                                            \
    REQUIRE_ZVVMTLS(s);                                                      \
                                                                             \
    static gen_helper_ldst_tile * const fns[4] = {                           \
        gen_helper_##NAME##8_v,  gen_helper_##NAME##16_v,                    \
        gen_helper_##NAME##32_v, gen_helper_##NAME##64_v                     \
    };                                                                       \
    gen_helper_ldst_tile *fn = fns[s->sew];                                  \
    if (fn == NULL) {                                                        \
        return false;                                                        \
    }                                                                        \
                                                                             \
    uint32_t data = 0;                                                       \
    data = FIELD_DP32(data, VDATA, VM,      a->vm);                          \
    data = FIELD_DP32(data, VDATA, LMUL,    s->lmul);                        \
    data = FIELD_DP32(data, VDATA, VTA,     s->vta);                         \
    data = FIELD_DP32(data, VDATA, VMA,     s->vma);                         \
    data = FIELD_DP32(data, VDATA, VLAMBDA, a->lambda);                      \
                                                                             \
    return ldst_tile_trans(a->rd, a->rs1, a->rs2, data, IS_LOAD, fn, s);     \
}

GEN_LDST_TILE_TRANS(vmtl, true)
GEN_LDST_TILE_TRANS(vmttl, true)
GEN_LDST_TILE_TRANS(vmts, false)
GEN_LDST_TILE_TRANS(vmtts, false)
