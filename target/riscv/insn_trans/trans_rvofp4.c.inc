/*
 * RISC-V translation routines for the OFP4 Standard Extensions.
 *
 * Copyright (C) 2025 SiFive, Inc.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

static bool vext_zvfofp4min_check(DisasContext *s, arg_rmr *a)
{
    return s->cfg_ptr->ext_zvfofp4min &&
           (s->sew == MO_8) &&
           vext_check_altfmt(s, -1) &&
           (s->lmul >= -2) &&
           require_rvv(s) &&
           vext_check_isa_ill(s) &&
           (a->rd != a->rs2) &&
           require_align(a->rd, s->lmul) &&
           require_align(a->rs2, s->lmul - 1) &&
           require_vm(a->vm, a->rd) &&
           require_noover(a->rd, s->lmul, a->rs2, s->lmul - 1);
}

static bool trans_vfext_vf2(DisasContext *s, arg_rmr *a)
{
    if (vext_zvfofp4min_check(s, a)) {
        uint32_t data = 0;

        data = FIELD_DP32(data, VDATA, VM, a->vm);
        data = FIELD_DP32(data, VDATA, LMUL, s->lmul);
        data = FIELD_DP32(data, VDATA, VTA, s->vta);
        data = FIELD_DP32(data, VDATA, VMA, s->vma);
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           vreg_ofs(s, a->rs2), tcg_env,
                           s->cfg_ptr->vlenb, s->cfg_ptr->vlenb, data,
                           gen_helper_vfext_vf2);
        tcg_gen_movi_tl(cpu_vstart, 0);
        finalize_rvv_inst(s);

        return true;
    }
    return false;
}
