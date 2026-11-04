"""Specialized transient position transforms for codec-native attention.

These kernels materialize only the positioned tensor required by the current
attention call.  They do not retain reconstructed K/V between denoising reads.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _native_center_unpack_kernel(
        rows, centers, ids, out,
        total, heads, sequence, width,
        rows_sb, rows_ss, rows_sd,
        centers_sb, centers_sc, centers_sd,
        ids_sb, ids_ss,
        out_sb, out_sh, out_ss, out_sd,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        token = tmp % sequence
        tmp = tmp // sequence
        head = tmp % heads
        batch = tmp // heads
        native_d = head * width + d
        cluster = tl.load(
            ids + batch * ids_sb + token * ids_ss, mask=valid
        ).to(tl.int64)
        row_value = tl.load(
            rows + batch * rows_sb + token * rows_ss + native_d * rows_sd,
            mask=valid,
        ).to(tl.float32)
        center_value = tl.load(
            centers + batch * centers_sb + cluster * centers_sc
            + native_d * centers_sd,
            mask=valid,
        ).to(tl.float32)
        output_offset = (
            batch * out_sb + head * out_sh + token * out_ss + d * out_sd
        )
        tl.store(out + output_offset, row_value + center_value, mask=valid)

    @triton.jit
    def _selected_rope_kernel(
        x, cos, sin, indices, out,
        total, heads, sequence, width,
        x_sb, x_sh, x_ss, x_sd,
        cos_ss, cos_sd, sin_ss, sin_sd,
        idx_sh, idx_sd,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        s = tmp % sequence
        tmp = tmp // sequence
        h = tmp % heads
        b = tmp // heads

        partner_d = tl.where((d & 1) == 0, d + 1, d - 1)
        base = b * x_sb + h * x_sh + s * x_ss
        value = tl.load(x + base + d * x_sd, mask=valid).to(tl.float32)
        partner = tl.load(
            x + base + partner_d * x_sd, mask=valid
        ).to(tl.float32)
        native_d = tl.load(
            indices + h * idx_sh + d * idx_sd, mask=valid
        ).to(tl.int64)
        c = tl.load(cos + s * cos_ss + native_d * cos_sd, mask=valid).to(
            tl.float32
        )
        sn = tl.load(sin + s * sin_ss + native_d * sin_sd, mask=valid).to(
            tl.float32
        )
        rotated = tl.where((d & 1) == 0, -partner, partner)
        tl.store(out + offsets, value * c + rotated * sn, mask=valid)


    @triton.jit
    def _prope_one_kernel(
        x, matrix, out,
        total, heads, sequence, width,
        x_sb, x_sh, x_ss, x_sd,
        m_sb, m_ss, m_sr, m_sc,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        s = tmp % sequence
        tmp = tmp // sequence
        h = tmp % heads
        b = tmp // heads
        row = d & 3
        quartet = d - row
        x_base = b * x_sb + h * x_sh + s * x_ss + quartet * x_sd
        m_base = b * m_sb + s * m_ss + row * m_sr
        x0 = tl.load(x + x_base, mask=valid).to(tl.float32)
        x1 = tl.load(x + x_base + x_sd, mask=valid).to(tl.float32)
        x2 = tl.load(x + x_base + 2 * x_sd, mask=valid).to(tl.float32)
        x3 = tl.load(x + x_base + 3 * x_sd, mask=valid).to(tl.float32)
        m0 = tl.load(matrix + m_base, mask=valid).to(tl.float32)
        m1 = tl.load(matrix + m_base + m_sc, mask=valid).to(tl.float32)
        m2 = tl.load(matrix + m_base + 2 * m_sc, mask=valid).to(tl.float32)
        m3 = tl.load(matrix + m_base + 3 * m_sc, mask=valid).to(tl.float32)
        value = m0 * x0 + m1 * x1 + m2 * x2 + m3 * x3
        tl.store(out + offsets, value, mask=valid)


    @triton.jit
    def _prope_pair_kernel(
        key, value, matrix, out_key, out_value,
        total, heads, sequence, width,
        k_sb, k_sh, k_ss, k_sd,
        v_sb, v_sh, v_ss, v_sd,
        m_sb, m_ss, m_sr, m_sc,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        s = tmp % sequence
        tmp = tmp // sequence
        h = tmp % heads
        b = tmp // heads
        row = d & 3
        quartet = d - row
        k_base = b * k_sb + h * k_sh + s * k_ss + quartet * k_sd
        v_base = b * v_sb + h * v_sh + s * v_ss + quartet * v_sd
        m_base = b * m_sb + s * m_ss + row * m_sr
        m0 = tl.load(matrix + m_base, mask=valid).to(tl.float32)
        m1 = tl.load(matrix + m_base + m_sc, mask=valid).to(tl.float32)
        m2 = tl.load(matrix + m_base + 2 * m_sc, mask=valid).to(tl.float32)
        m3 = tl.load(matrix + m_base + 3 * m_sc, mask=valid).to(tl.float32)

        k0 = tl.load(key + k_base, mask=valid).to(tl.float32)
        k1 = tl.load(key + k_base + k_sd, mask=valid).to(tl.float32)
        k2 = tl.load(key + k_base + 2 * k_sd, mask=valid).to(tl.float32)
        k3 = tl.load(key + k_base + 3 * k_sd, mask=valid).to(tl.float32)
        v0 = tl.load(value + v_base, mask=valid).to(tl.float32)
        v1 = tl.load(value + v_base + v_sd, mask=valid).to(tl.float32)
        v2 = tl.load(value + v_base + 2 * v_sd, mask=valid).to(tl.float32)
        v3 = tl.load(value + v_base + 3 * v_sd, mask=valid).to(tl.float32)
        tl.store(
            out_key + offsets,
            m0 * k0 + m1 * k1 + m2 * k2 + m3 * k3,
            mask=valid,
        )
        tl.store(
            out_value + offsets,
            m0 * v0 + m1 * v1 + m2 * v2 + m3 * v3,
            mask=valid,
        )


    @triton.jit
    def _center_pair_add_kernel(
        first, second, first_table, second_table, ids,
        out_first, out_second,
        total, heads, sequence, width,
        f_sb, f_sh, f_ss, f_sd,
        s_sb, s_sh, s_ss, s_sd,
        ft_sb, ft_sh, ft_sc, ft_sd,
        st_sb, st_sh, st_sc, st_sd,
        ids_sb, ids_ss,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        token = tmp % sequence
        tmp = tmp // sequence
        head = tmp % heads
        batch = tmp // heads
        cluster = tl.load(
            ids + batch * ids_sb + token * ids_ss, mask=valid
        ).to(tl.int64)
        first_offset = (
            batch * f_sb + head * f_sh + token * f_ss + d * f_sd
        )
        second_offset = (
            batch * s_sb + head * s_sh + token * s_ss + d * s_sd
        )
        first_table_offset = (
            batch * ft_sb + head * ft_sh + cluster * ft_sc + d * ft_sd
        )
        second_table_offset = (
            batch * st_sb + head * st_sh + cluster * st_sc + d * st_sd
        )
        first_value = tl.load(first + first_offset, mask=valid).to(tl.float32)
        second_value = tl.load(second + second_offset, mask=valid).to(tl.float32)
        first_center = tl.load(
            first_table + first_table_offset, mask=valid
        ).to(tl.float32)
        second_center = tl.load(
            second_table + second_table_offset, mask=valid
        ).to(tl.float32)
        tl.store(out_first + offsets, first_value + first_center, mask=valid)
        tl.store(out_second + offsets, second_value + second_center, mask=valid)


    @triton.jit
    def _lse_combine_kernel(
        native, native_lse, history, history_lse, out,
        total, heads, queries, width,
        n_sb, n_sh, n_sq, n_sd,
        nl_sb, nl_sh, nl_sq,
        h_sb, h_sh, h_sq, h_sd,
        hl_sb, hl_sh, hl_sq,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        query = tmp % queries
        tmp = tmp // queries
        head = tmp % heads
        batch = tmp // heads
        native_offset = (
            batch * n_sb + head * n_sh + query * n_sq + d * n_sd
        )
        history_offset = (
            batch * h_sb + head * h_sh + query * h_sq + d * h_sd
        )
        native_lse_offset = batch * nl_sb + head * nl_sh + query * nl_sq
        history_lse_offset = batch * hl_sb + head * hl_sh + query * hl_sq
        a = tl.load(native_lse + native_lse_offset, mask=valid).to(tl.float32)
        b = tl.load(history_lse + history_lse_offset, mask=valid).to(tl.float32)
        maximum = tl.maximum(a, b)
        wa = tl.exp(a - maximum)
        wb = tl.exp(b - maximum)
        inv = 1.0 / (wa + wb)
        nv = tl.load(native + native_offset, mask=valid).to(tl.float32)
        hv = tl.load(history + history_offset, mask=valid).to(tl.float32)
        tl.store(out + offsets, (nv * wa + hv * wb) * inv, mask=valid)


    @triton.jit
    def _native_kv_cat_kernel(
        text_k, text_v, anchor_k, anchor_v,
        recent_k, recent_v, current_k, current_v,
        out_k, out_v,
        total, heads, final_sequence, width,
        text_sequence, anchor_sequence, recent_sequence,
        t_sb, t_sh, t_ss, t_sd,
        a_sb, a_sh, a_ss, a_sd,
        r_sb, r_sh, r_ss, r_sd,
        c_sb, c_sh, c_ss, c_sd,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        position = tmp % final_sequence
        tmp = tmp // final_sequence
        head = tmp % heads
        batch = tmp // heads

        anchor_start = text_sequence
        recent_start = anchor_start + anchor_sequence
        current_start = recent_start + recent_sequence
        is_text = valid & (position < anchor_start)
        is_anchor = valid & (position >= anchor_start) & (position < recent_start)
        is_recent = valid & (position >= recent_start) & (position < current_start)
        is_current = valid & (position >= current_start)
        text_offset = batch * t_sb + head * t_sh + position * t_ss + d * t_sd
        anchor_position = position - anchor_start
        anchor_offset = (
            batch * a_sb + head * a_sh + anchor_position * a_ss + d * a_sd
        )
        recent_position = position - recent_start
        recent_offset = (
            batch * r_sb + head * r_sh + recent_position * r_ss + d * r_sd
        )
        current_position = position - current_start
        current_offset = (
            batch * c_sb + head * c_sh + current_position * c_ss + d * c_sd
        )

        key = tl.load(text_k + text_offset, mask=is_text, other=0.0)
        value = tl.load(text_v + text_offset, mask=is_text, other=0.0)
        key = tl.where(
            is_anchor,
            tl.load(anchor_k + anchor_offset, mask=is_anchor, other=0.0), key,
        )
        value = tl.where(
            is_anchor,
            tl.load(anchor_v + anchor_offset, mask=is_anchor, other=0.0), value,
        )
        key = tl.where(
            is_recent,
            tl.load(recent_k + recent_offset, mask=is_recent, other=0.0), key,
        )
        value = tl.where(
            is_recent,
            tl.load(recent_v + recent_offset, mask=is_recent, other=0.0), value,
        )
        key = tl.where(
            is_current,
            tl.load(current_k + current_offset, mask=is_current, other=0.0), key,
        )
        value = tl.where(
            is_current,
            tl.load(current_v + current_offset, mask=is_current, other=0.0), value,
        )
        tl.store(out_k + offsets, key, mask=valid)
        tl.store(out_v + offsets, value, mask=valid)


    @triton.jit
    def _center_position_pack_kernel(
        key_main, key_prope, value_main, value_prope,
        key_main_table, key_prope_table,
        value_main_table, value_prope_table,
        key_ids, value_ids, cos, sin, indices, matrix,
        out_key, out_value,
        total, batch, heads, sequence, width,
        k_sb, k_sh, k_ss, k_sd,
        v_sb, v_sh, v_ss, v_sd,
        kt_sb, kt_sh, kt_sc, kt_sd,
        vt_sb, vt_sh, vt_sc, vt_sd,
        ki_sb, ki_ss, vi_sb, vi_ss,
        cos_ss, cos_sd, sin_ss, sin_sd,
        idx_sh, idx_sd,
        m_sb, m_ss, m_sr, m_sc,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        token = tmp % sequence
        tmp = tmp // sequence
        head = tmp % heads
        packed_batch = tmp // heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        key_cluster = tl.load(
            key_ids + source_batch * ki_sb + token * ki_ss, mask=valid
        ).to(tl.int64)
        value_cluster = tl.load(
            value_ids + source_batch * vi_sb + token * vi_ss, mask=valid
        ).to(tl.int64)

        key_base = (
            source_batch * k_sb + head * k_sh + token * k_ss
        )
        value_base = (
            source_batch * v_sb + head * v_sh + token * v_ss
        )
        key_table_base = (
            source_batch * kt_sb + head * kt_sh + key_cluster * kt_sc
        )
        value_table_base = (
            source_batch * vt_sb + head * vt_sh + value_cluster * vt_sc
        )

        # Match the existing centre kernel's BF16 store before either position
        # transform.  This explicit rounding keeps the fused path numerically
        # identical instead of silently changing the arithmetic graph.
        main_key = (
            tl.load(key_main + key_base + d * k_sd, mask=valid).to(tl.float32)
            + tl.load(
                key_main_table + key_table_base + d * kt_sd, mask=valid
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        main_value = (
            tl.load(value_main + value_base + d * v_sd, mask=valid).to(tl.float32)
            + tl.load(
                value_main_table + value_table_base + d * vt_sd, mask=valid
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)

        partner_d = tl.where((d & 1) == 0, d + 1, d - 1)
        partner_key = (
            tl.load(
                key_main + key_base + partner_d * k_sd, mask=valid
            ).to(tl.float32)
            + tl.load(
                key_main_table + key_table_base + partner_d * kt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        native_d = tl.load(
            indices + head * idx_sh + d * idx_sd, mask=valid
        ).to(tl.int64)
        c = tl.load(
            cos + token * cos_ss + native_d * cos_sd, mask=valid
        ).to(tl.float32)
        sn = tl.load(
            sin + token * sin_ss + native_d * sin_sd, mask=valid
        ).to(tl.float32)
        rotated = tl.where((d & 1) == 0, -partner_key, partner_key)
        main_key = main_key * c + rotated * sn

        row = d & 3
        quartet = d - row
        matrix_base = (
            source_batch * m_sb + token * m_ss + row * m_sr
        )
        coefficient0 = tl.load(matrix + matrix_base, mask=valid).to(tl.float32)
        coefficient1 = tl.load(
            matrix + matrix_base + m_sc, mask=valid
        ).to(tl.float32)
        coefficient2 = tl.load(
            matrix + matrix_base + 2 * m_sc, mask=valid
        ).to(tl.float32)
        coefficient3 = tl.load(
            matrix + matrix_base + 3 * m_sc, mask=valid
        ).to(tl.float32)
        key0 = (
            tl.load(key_prope + key_base + quartet * k_sd, mask=valid).to(tl.float32)
            + tl.load(
                key_prope_table + key_table_base + quartet * kt_sd, mask=valid
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        key1 = (
            tl.load(key_prope + key_base + (quartet + 1) * k_sd, mask=valid).to(tl.float32)
            + tl.load(
                key_prope_table + key_table_base + (quartet + 1) * kt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        key2 = (
            tl.load(key_prope + key_base + (quartet + 2) * k_sd, mask=valid).to(tl.float32)
            + tl.load(
                key_prope_table + key_table_base + (quartet + 2) * kt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        key3 = (
            tl.load(key_prope + key_base + (quartet + 3) * k_sd, mask=valid).to(tl.float32)
            + tl.load(
                key_prope_table + key_table_base + (quartet + 3) * kt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        value0 = (
            tl.load(value_prope + value_base + quartet * v_sd, mask=valid).to(tl.float32)
            + tl.load(
                value_prope_table + value_table_base + quartet * vt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        value1 = (
            tl.load(value_prope + value_base + (quartet + 1) * v_sd, mask=valid).to(tl.float32)
            + tl.load(
                value_prope_table + value_table_base + (quartet + 1) * vt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        value2 = (
            tl.load(value_prope + value_base + (quartet + 2) * v_sd, mask=valid).to(tl.float32)
            + tl.load(
                value_prope_table + value_table_base + (quartet + 2) * vt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        value3 = (
            tl.load(value_prope + value_base + (quartet + 3) * v_sd, mask=valid).to(tl.float32)
            + tl.load(
                value_prope_table + value_table_base + (quartet + 3) * vt_sd,
                mask=valid,
            ).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        prope_key = (
            coefficient0 * key0 + coefficient1 * key1
            + coefficient2 * key2 + coefficient3 * key3
        )
        prope_value = (
            coefficient0 * value0 + coefficient1 * value1
            + coefficient2 * value2 + coefficient3 * value3
        )

        tl.store(
            out_key + offsets,
            tl.where(is_prope, prope_key, main_key),
            mask=valid,
        )
        tl.store(
            out_value + offsets,
            tl.where(is_prope, prope_value, main_value),
            mask=valid,
        )


    @triton.jit
    def _native_position_pack_kernel(
        key, value, cos, sin, matrix, out_key, out_value,
        total, batch, heads, sequence, width,
        k_sb, k_sh, k_ss, k_sd,
        v_sb, v_sh, v_ss, v_sd,
        cos_ss, cos_sd, sin_ss, sin_sd,
        m_sb, m_ss, m_sr, m_sc,
        BLOCK: tl.constexpr,
    ):
        """Pack native Main/PRoPE K/V without intermediate transformed tensors."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        token = tmp % sequence
        tmp = tmp // sequence
        head = tmp % heads
        packed_batch = tmp // heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        key_base = source_batch * k_sb + head * k_sh + token * k_ss
        value_base = source_batch * v_sb + head * v_sh + token * v_ss

        key_value = tl.load(
            key + key_base + d * k_sd, mask=valid
        ).to(tl.float32)
        value_value = tl.load(
            value + value_base + d * v_sd, mask=valid
        ).to(tl.float32)

        partner_d = tl.where((d & 1) == 0, d + 1, d - 1)
        partner_key = tl.load(
            key + key_base + partner_d * k_sd, mask=valid
        ).to(tl.float32)
        c = tl.load(cos + token * cos_ss + d * cos_sd, mask=valid).to(
            tl.float32
        )
        sn = tl.load(sin + token * sin_ss + d * sin_sd, mask=valid).to(
            tl.float32
        )
        rotated = tl.where((d & 1) == 0, -partner_key, partner_key)
        main_key = key_value * c + rotated * sn

        row = d & 3
        quartet = d - row
        matrix_base = source_batch * m_sb + token * m_ss + row * m_sr
        coefficient0 = tl.load(matrix + matrix_base, mask=valid).to(tl.float32)
        coefficient1 = tl.load(
            matrix + matrix_base + m_sc, mask=valid
        ).to(tl.float32)
        coefficient2 = tl.load(
            matrix + matrix_base + 2 * m_sc, mask=valid
        ).to(tl.float32)
        coefficient3 = tl.load(
            matrix + matrix_base + 3 * m_sc, mask=valid
        ).to(tl.float32)
        key0 = tl.load(key + key_base + quartet * k_sd, mask=valid).to(tl.float32)
        key1 = tl.load(
            key + key_base + (quartet + 1) * k_sd, mask=valid
        ).to(tl.float32)
        key2 = tl.load(
            key + key_base + (quartet + 2) * k_sd, mask=valid
        ).to(tl.float32)
        key3 = tl.load(
            key + key_base + (quartet + 3) * k_sd, mask=valid
        ).to(tl.float32)
        value0 = tl.load(
            value + value_base + quartet * v_sd, mask=valid
        ).to(tl.float32)
        value1 = tl.load(
            value + value_base + (quartet + 1) * v_sd, mask=valid
        ).to(tl.float32)
        value2 = tl.load(
            value + value_base + (quartet + 2) * v_sd, mask=valid
        ).to(tl.float32)
        value3 = tl.load(
            value + value_base + (quartet + 3) * v_sd, mask=valid
        ).to(tl.float32)
        prope_key = (
            coefficient0 * key0 + coefficient1 * key1
            + coefficient2 * key2 + coefficient3 * key3
        )
        prope_value = (
            coefficient0 * value0 + coefficient1 * value1
            + coefficient2 * value2 + coefficient3 * value3
        )
        tl.store(
            out_key + offsets,
            tl.where(is_prope, prope_key, main_key),
            mask=valid,
        )
        tl.store(
            out_value + offsets,
            tl.where(is_prope, prope_value, value_value),
            mask=valid,
        )


    @triton.jit
    def _native_prope_pack_kernel(
        main_key, key, value, matrix, out_key, out_value,
        total, batch, heads, sequence, width,
        mk_sb, mk_sh, mk_ss, mk_sd,
        k_sb, k_sh, k_ss, k_sd,
        v_sb, v_sh, v_ss, v_sd,
        m_sb, m_ss, m_sr, m_sc,
        BLOCK: tl.constexpr,
    ):
        """Preserve externally rounded RoPE while fusing PRoPE and packing."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        token = tmp % sequence
        tmp = tmp // sequence
        head = tmp % heads
        packed_batch = tmp // heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        main_key_base = (
            source_batch * mk_sb + head * mk_sh + token * mk_ss
        )
        key_base = source_batch * k_sb + head * k_sh + token * k_ss
        value_base = source_batch * v_sb + head * v_sh + token * v_ss
        positioned_main_key = tl.load(
            main_key + main_key_base + d * mk_sd, mask=valid
        ).to(tl.float32)
        main_value = tl.load(
            value + value_base + d * v_sd, mask=valid
        ).to(tl.float32)

        row = d & 3
        quartet = d - row
        matrix_base = source_batch * m_sb + token * m_ss + row * m_sr
        coefficient0 = tl.load(matrix + matrix_base, mask=valid).to(tl.float32)
        coefficient1 = tl.load(
            matrix + matrix_base + m_sc, mask=valid
        ).to(tl.float32)
        coefficient2 = tl.load(
            matrix + matrix_base + 2 * m_sc, mask=valid
        ).to(tl.float32)
        coefficient3 = tl.load(
            matrix + matrix_base + 3 * m_sc, mask=valid
        ).to(tl.float32)
        key0 = tl.load(key + key_base + quartet * k_sd, mask=valid).to(tl.float32)
        key1 = tl.load(
            key + key_base + (quartet + 1) * k_sd, mask=valid
        ).to(tl.float32)
        key2 = tl.load(
            key + key_base + (quartet + 2) * k_sd, mask=valid
        ).to(tl.float32)
        key3 = tl.load(
            key + key_base + (quartet + 3) * k_sd, mask=valid
        ).to(tl.float32)
        value0 = tl.load(
            value + value_base + quartet * v_sd, mask=valid
        ).to(tl.float32)
        value1 = tl.load(
            value + value_base + (quartet + 1) * v_sd, mask=valid
        ).to(tl.float32)
        value2 = tl.load(
            value + value_base + (quartet + 2) * v_sd, mask=valid
        ).to(tl.float32)
        value3 = tl.load(
            value + value_base + (quartet + 3) * v_sd, mask=valid
        ).to(tl.float32)
        prope_key = (
            coefficient0 * key0 + coefficient1 * key1
            + coefficient2 * key2 + coefficient3 * key3
        )
        prope_value = (
            coefficient0 * value0 + coefficient1 * value1
            + coefficient2 * value2 + coefficient3 * value3
        )
        tl.store(
            out_key + offsets,
            tl.where(is_prope, prope_key, positioned_main_key),
            mask=valid,
        )
        tl.store(
            out_value + offsets,
            tl.where(is_prope, prope_value, main_value),
            mask=valid,
        )


    @triton.jit
    def _grouped_scatter_pair_kernel(
        main, prope, lse, head_indices, full, full_lse,
        total, batch, group_heads, queries, width,
        m_sb, m_sh, m_sq, m_sd,
        p_sb, p_sh, p_sq, p_sd,
        l_sb, l_sh, l_sq,
        hi_s,
        f_sb, f_sh, f_sq, f_sd,
        fl_sb, fl_sh, fl_sq,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        query = tmp % queries
        tmp = tmp // queries
        group_head = tmp % group_heads
        packed_batch = tmp // group_heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        destination_head = tl.load(
            head_indices + group_head * hi_s, mask=valid
        ).to(tl.int64)
        main_offset = (
            source_batch * m_sb + group_head * m_sh
            + query * m_sq + d * m_sd
        )
        prope_offset = (
            source_batch * p_sb + group_head * p_sh
            + query * p_sq + d * p_sd
        )
        main_value = tl.load(main + main_offset, mask=valid).to(tl.float32)
        prope_value = tl.load(prope + prope_offset, mask=valid).to(tl.float32)
        destination = (
            packed_batch * f_sb + destination_head * f_sh
            + query * f_sq + d * f_sd
        )
        tl.store(
            full + destination,
            tl.where(is_prope, prope_value, main_value),
            mask=valid,
        )
        # One width lane owns the corresponding LSE element, avoiding a
        # separate scatter launch without introducing write races.
        lse_valid = valid & (d == 0)
        lse_offset = (
            packed_batch * l_sb + group_head * l_sh + query * l_sq
        )
        full_lse_offset = (
            packed_batch * fl_sb + destination_head * fl_sh
            + query * fl_sq
        )
        lse_value = tl.load(lse + lse_offset, mask=lse_valid)
        tl.store(full_lse + full_lse_offset, lse_value, mask=lse_valid)


    @triton.jit
    def _grouped_lse_combine_scatter_kernel(
        native, native_lse, main, prope, history_lse,
        head_indices, out,
        total, batch, group_heads, queries, width,
        n_sb, n_sh, n_sq, n_sd,
        nl_sb, nl_sh, nl_sq,
        m_sb, m_sh, m_sq, m_sd,
        p_sb, p_sh, p_sq, p_sd,
        hl_sb, hl_sh, hl_sq,
        hi_s,
        o_sb, o_sh, o_sq, o_sd,
        BLOCK: tl.constexpr,
    ):
        """Combine one width group with native and write canonical heads.

        Group heads form a disjoint partition of H16, so separate launches can
        fill the final output directly without atomics or an intermediate
        [2B,H,Q,128] history tensor.
        """
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        query = tmp % queries
        tmp = tmp // queries
        group_head = tmp % group_heads
        packed_batch = tmp // group_heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        destination_head = tl.load(
            head_indices + group_head * hi_s, mask=valid
        ).to(tl.int64)

        native_offset = (
            packed_batch * n_sb + destination_head * n_sh
            + query * n_sq + d * n_sd
        )
        native_lse_offset = (
            packed_batch * nl_sb + destination_head * nl_sh
            + query * nl_sq
        )
        main_offset = (
            source_batch * m_sb + group_head * m_sh
            + query * m_sq + d * m_sd
        )
        prope_offset = (
            source_batch * p_sb + group_head * p_sh
            + query * p_sq + d * p_sd
        )
        history_lse_offset = (
            packed_batch * hl_sb + group_head * hl_sh + query * hl_sq
        )
        output_offset = (
            packed_batch * o_sb + destination_head * o_sh
            + query * o_sq + d * o_sd
        )

        a = tl.load(native_lse + native_lse_offset, mask=valid).to(tl.float32)
        b = tl.load(history_lse + history_lse_offset, mask=valid).to(tl.float32)
        maximum = tl.maximum(a, b)
        wa = tl.exp(a - maximum)
        wb = tl.exp(b - maximum)
        native_value = tl.load(
            native + native_offset, mask=valid
        ).to(tl.float32)
        main_value = tl.load(main + main_offset, mask=valid).to(tl.float32)
        prope_value = tl.load(prope + prope_offset, mask=valid).to(tl.float32)
        history_value = tl.where(is_prope, prope_value, main_value)
        tl.store(
            out + output_offset,
            (native_value * wa + history_value * wb) / (wa + wb),
            mask=valid,
        )


    @triton.jit
    def _lse_combine_split_kernel(
        native, native_lse, main, prope, history_lse, out,
        total, batch, heads, queries, width,
        n_sb, n_sh, n_sq, n_sd,
        nl_sb, nl_sh, nl_sq,
        m_sb, m_sh, m_sq, m_sd,
        p_sb, p_sh, p_sq, p_sd,
        hl_sb, hl_sh, hl_sq,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        query = tmp % queries
        tmp = tmp // queries
        head = tmp % heads
        packed_batch = tmp // heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        native_offset = (
            packed_batch * n_sb + head * n_sh + query * n_sq + d * n_sd
        )
        main_offset = (
            source_batch * m_sb + head * m_sh + query * m_sq + d * m_sd
        )
        prope_offset = (
            source_batch * p_sb + head * p_sh + query * p_sq + d * p_sd
        )
        native_lse_offset = (
            packed_batch * nl_sb + head * nl_sh + query * nl_sq
        )
        history_lse_offset = (
            packed_batch * hl_sb + head * hl_sh + query * hl_sq
        )
        a = tl.load(native_lse + native_lse_offset, mask=valid).to(tl.float32)
        b = tl.load(history_lse + history_lse_offset, mask=valid).to(tl.float32)
        maximum = tl.maximum(a, b)
        wa = tl.exp(a - maximum)
        wb = tl.exp(b - maximum)
        native_value = tl.load(
            native + native_offset, mask=valid
        ).to(tl.float32)
        main_value = tl.load(main + main_offset, mask=valid).to(tl.float32)
        prope_value = tl.load(prope + prope_offset, mask=valid).to(tl.float32)
        history_value = tl.where(is_prope, prope_value, main_value)
        tl.store(
            out + offsets,
            (native_value * wa + history_value * wb) / (wa + wb),
            mask=valid,
        )


    @triton.jit
    def _main_lift_lse_combine_kernel(
        low, lift, native, native_lse, history_lse, out,
        batch, heads, queries, low_dim: tl.constexpr,
        low_sb, low_sh, low_sq, low_sd,
        lift_sh, lift_sk, lift_so,
        native_sb, native_sh, native_sq, native_sd,
        nl_sb, nl_sh, nl_sq,
        hl_sb, hl_sh, hl_sq,
        out_sb, out_sh, out_sq, out_sd,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Lift the Main value branch and merge its LSE in one write."""
        query_block = tl.program_id(0)
        packed_head = tl.program_id(1)
        output_block = tl.program_id(2)
        b = packed_head // heads
        h = packed_head - b * heads
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, low_dim, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                low + b * low_sb + h * low_sh
                + rows[:, None] * low_sq + ks[None, :] * low_sd,
                mask=(rows[:, None] < queries) & (ks[None, :] < low_dim),
                other=0.0,
            )
            weights = tl.load(
                lift + h * lift_sh + ks[:, None] * lift_sk
                + cols[None, :] * lift_so,
                mask=(ks[:, None] < low_dim) & (cols[None, :] < 128),
                other=0.0,
            )
            accumulator += tl.dot(a, weights)

        valid = (rows[:, None] < queries) & (cols[None, :] < 128)
        native_value = tl.load(
            native + b * native_sb + h * native_sh
            + rows[:, None] * native_sq + cols[None, :] * native_sd,
            mask=valid, other=0.0,
        ).to(tl.float32)
        native_log = tl.load(
            native_lse + b * nl_sb + h * nl_sh + rows * nl_sq,
            mask=rows < queries, other=-float("inf"),
        ).to(tl.float32)
        history_log = tl.load(
            history_lse + b * hl_sb + h * hl_sh + rows * hl_sq,
            mask=rows < queries, other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.maximum(native_log, history_log)
        native_weight = tl.exp(native_log - maximum)
        history_weight = tl.exp(history_log - maximum)
        result = (
            native_value * native_weight[:, None]
            + accumulator * history_weight[:, None]
        ) / (native_weight + history_weight)[:, None]
        tl.store(
            out + b * out_sb + h * out_sh
            + rows[:, None] * out_sq + cols[None, :] * out_sd,
            result, mask=valid,
        )


    @triton.jit
    def _prope_lift_lse_combine_kernel(
        low, mix, native, native_lse, history_lse, out,
        batch, heads, queries, low_quartets: tl.constexpr,
        low_sb, low_sh, low_sq, low_sd,
        mix_sh, mix_sk, mix_so,
        native_sb, native_sh, native_sq, native_sd,
        nl_sb, nl_sh, nl_sq,
        hl_sb, hl_sh, hl_sq,
        out_sb, out_sh, out_sq, out_sd,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Lift one PRoPE coordinate and merge its LSE in one write."""
        query_block = tl.program_id(0)
        packed_head = tl.program_id(1)
        coordinate = tl.program_id(2)
        b = packed_head // heads
        h = packed_head - b * heads
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, low_quartets, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                low + b * low_sb + h * low_sh + rows[:, None] * low_sq
                + (ks[None, :] * 4 + coordinate) * low_sd,
                mask=(rows[:, None] < queries)
                & (ks[None, :] < low_quartets),
                other=0.0,
            )
            weights = tl.load(
                mix + h * mix_sh + ks[:, None] * mix_sk
                + cols[None, :] * mix_so,
                mask=(ks[:, None] < low_quartets) & (cols[None, :] < 32),
                other=0.0,
            )
            accumulator += tl.dot(a, weights)

        destination = cols * 4 + coordinate
        packed_batch = b + batch
        valid = (rows[:, None] < queries) & (cols[None, :] < 32)
        native_value = tl.load(
            native + packed_batch * native_sb + h * native_sh
            + rows[:, None] * native_sq + destination[None, :] * native_sd,
            mask=valid, other=0.0,
        ).to(tl.float32)
        native_log = tl.load(
            native_lse + packed_batch * nl_sb + h * nl_sh + rows * nl_sq,
            mask=rows < queries, other=-float("inf"),
        ).to(tl.float32)
        history_log = tl.load(
            history_lse + packed_batch * hl_sb + h * hl_sh + rows * hl_sq,
            mask=rows < queries, other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.maximum(native_log, history_log)
        native_weight = tl.exp(native_log - maximum)
        history_weight = tl.exp(history_log - maximum)
        result = (
            native_value * native_weight[:, None]
            + accumulator * history_weight[:, None]
        ) / (native_weight + history_weight)[:, None]
        tl.store(
            out + packed_batch * out_sb + h * out_sh
            + rows[:, None] * out_sq + destination[None, :] * out_sd,
            result, mask=valid,
        )


    @triton.jit
    def _query_position_pack_kernel(
        main, prope, cos, sin, indices, matrix,
        main_scale, prope_scale, out,
        total, batch, heads, queries, width,
        main_sb, main_sh, main_sq, main_sd,
        prope_sb, prope_sh, prope_sq, prope_sd,
        cos_sq, cos_sd, sin_sq, sin_sd,
        idx_sh, idx_sd,
        matrix_sb, matrix_sq, matrix_sr, matrix_sc,
        main_scale_s, prope_scale_s,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % width
        tmp = offsets // width
        query = tmp % queries
        tmp = tmp // queries
        head = tmp % heads
        packed_batch = tmp // heads
        is_prope = packed_batch >= batch
        source_batch = packed_batch - tl.where(is_prope, batch, 0)
        main_base = (
            source_batch * main_sb + head * main_sh + query * main_sq
        )
        prope_base = (
            source_batch * prope_sb + head * prope_sh + query * prope_sq
        )

        main_value = tl.load(
            main + main_base + d * main_sd, mask=valid
        ).to(tl.float32)
        partner_d = tl.where((d & 1) == 0, d + 1, d - 1)
        partner = tl.load(
            main + main_base + partner_d * main_sd, mask=valid
        ).to(tl.float32)
        native_d = tl.load(
            indices + head * idx_sh + d * idx_sd, mask=valid
        ).to(tl.int64)
        c = tl.load(cos + query * cos_sq + native_d * cos_sd, mask=valid).to(
            tl.float32
        )
        sn = tl.load(sin + query * sin_sq + native_d * sin_sd, mask=valid).to(
            tl.float32
        )
        rotated = tl.where((d & 1) == 0, -partner, partner)
        positioned_main = (main_value * c + rotated * sn).to(tl.bfloat16)
        positioned_main = positioned_main.to(tl.float32) * tl.load(
            main_scale + head * main_scale_s, mask=valid
        ).to(tl.float32)

        row = d & 3
        quartet = d - row
        matrix_base = (
            source_batch * matrix_sb + query * matrix_sq + row * matrix_sr
        )
        p0 = tl.load(prope + prope_base + quartet * prope_sd, mask=valid).to(tl.float32)
        p1 = tl.load(prope + prope_base + (quartet + 1) * prope_sd, mask=valid).to(tl.float32)
        p2 = tl.load(prope + prope_base + (quartet + 2) * prope_sd, mask=valid).to(tl.float32)
        p3 = tl.load(prope + prope_base + (quartet + 3) * prope_sd, mask=valid).to(tl.float32)
        m0 = tl.load(matrix + matrix_base, mask=valid).to(tl.float32)
        m1 = tl.load(matrix + matrix_base + matrix_sc, mask=valid).to(tl.float32)
        m2 = tl.load(matrix + matrix_base + 2 * matrix_sc, mask=valid).to(tl.float32)
        m3 = tl.load(matrix + matrix_base + 3 * matrix_sc, mask=valid).to(tl.float32)
        positioned_prope = (m0 * p0 + m1 * p1 + m2 * p2 + m3 * p3).to(
            tl.bfloat16
        )
        positioned_prope = positioned_prope.to(tl.float32) * tl.load(
            prope_scale + head * prope_scale_s, mask=valid
        ).to(tl.float32)
        tl.store(
            out + offsets,
            tl.where(is_prope, positioned_prope, positioned_main),
            mask=valid,
        )


    @triton.jit
    def _main_query_project_rope_kernel(
        query, reduce, cos, sin, indices, scale, out,
        batch, heads, queries, width,
        q_sb, q_sh, q_sq, q_sd,
        r_sh, r_so, r_si,
        cos_sq, cos_sd, sin_sq, sin_sd,
        idx_sh, idx_sd, scale_s,
        out_sb, out_sh, out_sq, out_sd,
        BLOCK_M: tl.constexpr, BLOCK_P: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        packed_head = tl.program_id(1)
        pair_block = tl.program_id(2)
        b = packed_head // heads
        h = packed_head - b * heads
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        pairs = pair_block * BLOCK_P + tl.arange(0, BLOCK_P)
        even = pairs * 2
        odd = even + 1
        even_acc = tl.zeros((BLOCK_M, BLOCK_P), tl.float32)
        odd_acc = tl.zeros((BLOCK_M, BLOCK_P), tl.float32)
        for k0 in range(0, 128, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            source = tl.load(
                query + b * q_sb + h * q_sh
                + rows[:, None] * q_sq + ks[None, :] * q_sd,
                mask=rows[:, None] < queries, other=0.0,
            )
            even_weight = tl.load(
                reduce + h * r_sh + even[None, :] * r_so
                + ks[:, None] * r_si,
                mask=even[None, :] < width, other=0.0,
            )
            odd_weight = tl.load(
                reduce + h * r_sh + odd[None, :] * r_so
                + ks[:, None] * r_si,
                mask=odd[None, :] < width, other=0.0,
            )
            even_acc += tl.dot(source, even_weight)
            odd_acc += tl.dot(source, odd_weight)
        # Match the established BF16 q_main GEMM boundary before RoPE.
        even_acc = even_acc.to(tl.bfloat16).to(tl.float32)
        odd_acc = odd_acc.to(tl.bfloat16).to(tl.float32)
        even_native = tl.load(
            indices + h * idx_sh + even * idx_sd,
            mask=even < width, other=0,
        ).to(tl.int64)
        odd_native = tl.load(
            indices + h * idx_sh + odd * idx_sd,
            mask=odd < width, other=0,
        ).to(tl.int64)
        cos_even = tl.load(
            cos + rows[:, None] * cos_sq + even_native[None, :] * cos_sd,
            mask=(rows[:, None] < queries) & (even[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        sin_even = tl.load(
            sin + rows[:, None] * sin_sq + even_native[None, :] * sin_sd,
            mask=(rows[:, None] < queries) & (even[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        cos_odd = tl.load(
            cos + rows[:, None] * cos_sq + odd_native[None, :] * cos_sd,
            mask=(rows[:, None] < queries) & (odd[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        sin_odd = tl.load(
            sin + rows[:, None] * sin_sq + odd_native[None, :] * sin_sd,
            mask=(rows[:, None] < queries) & (odd[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        head_scale = tl.load(scale + h * scale_s).to(tl.float32)
        even_result = (even_acc * cos_even - odd_acc * sin_even) * head_scale
        odd_result = (odd_acc * cos_odd + even_acc * sin_odd) * head_scale
        valid_even = (rows[:, None] < queries) & (even[None, :] < width)
        valid_odd = (rows[:, None] < queries) & (odd[None, :] < width)
        base = b * out_sb + h * out_sh + rows[:, None] * out_sq
        tl.store(out + base + even[None, :] * out_sd, even_result, mask=valid_even)
        tl.store(out + base + odd[None, :] * out_sd, odd_result, mask=valid_odd)


    @triton.jit
    def _prope_query_project_position_kernel(
        query, mix, matrix, scale, out,
        batch, heads, queries, quartets,
        q_sb, q_sh, q_sq, q_sd,
        mix_sh, mix_sr, mix_so,
        matrix_sb, matrix_sq, matrix_sr, matrix_sc,
        scale_s, out_sb, out_sh, out_sq, out_sd,
        BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        packed_head = tl.program_id(1)
        quartet_block = tl.program_id(2)
        b = packed_head // heads
        h = packed_head - b * heads
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        rs = quartet_block * BLOCK_R + tl.arange(0, BLOCK_R)
        acc0 = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_R), tl.float32)
        for k0 in range(0, 32, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            weight = tl.load(
                mix + h * mix_sh + rs[None, :] * mix_sr
                + ks[:, None] * mix_so,
                mask=rs[None, :] < quartets, other=0.0,
            )
            source_base = (
                query + b * q_sb + h * q_sh
                + rows[:, None] * q_sq + ks[None, :] * 4 * q_sd
            )
            valid_source = rows[:, None] < queries
            acc0 += tl.dot(tl.load(source_base, mask=valid_source, other=0.0), weight)
            acc1 += tl.dot(tl.load(source_base + q_sd, mask=valid_source, other=0.0), weight)
            acc2 += tl.dot(tl.load(source_base + 2 * q_sd, mask=valid_source, other=0.0), weight)
            acc3 += tl.dot(tl.load(source_base + 3 * q_sd, mask=valid_source, other=0.0), weight)
        matrix_base = b * matrix_sb + rows[:, None] * matrix_sq
        valid = (rows[:, None] < queries) & (rs[None, :] < quartets)
        head_scale = tl.load(scale + h * scale_s).to(tl.float32)
        output_base = (
            out + (b + batch) * out_sb + h * out_sh
            + rows[:, None] * out_sq + rs[None, :] * 4 * out_sd
        )
        for coordinate in range(0, 4):
            m0 = tl.load(
                matrix + matrix_base + coordinate * matrix_sr,
                mask=rows[:, None] < queries, other=0.0,
            ).to(tl.float32)
            m1 = tl.load(
                matrix + matrix_base + coordinate * matrix_sr + matrix_sc,
                mask=rows[:, None] < queries, other=0.0,
            ).to(tl.float32)
            m2 = tl.load(
                matrix + matrix_base + coordinate * matrix_sr + 2 * matrix_sc,
                mask=rows[:, None] < queries, other=0.0,
            ).to(tl.float32)
            m3 = tl.load(
                matrix + matrix_base + coordinate * matrix_sr + 3 * matrix_sc,
                mask=rows[:, None] < queries, other=0.0,
            ).to(tl.float32)
            positioned = (m0 * acc0 + m1 * acc1 + m2 * acc2 + m3 * acc3)
            tl.store(
                output_base + coordinate * out_sd,
                positioned * head_scale, mask=valid,
            )

else:
    _native_center_unpack_kernel = None
    _selected_rope_kernel = None
    _prope_one_kernel = None
    _prope_pair_kernel = None
    _center_pair_add_kernel = None
    _lse_combine_kernel = None
    _native_kv_cat_kernel = None
    _center_position_pack_kernel = None
    _native_position_pack_kernel = None
    _native_prope_pack_kernel = None
    _grouped_scatter_pair_kernel = None
    _lse_combine_split_kernel = None
    _main_lift_lse_combine_kernel = None
    _prope_lift_lse_combine_kernel = None
    _query_position_pack_kernel = None
    _main_query_project_rope_kernel = None
    _prope_query_project_position_kernel = None


def native_center_unpack(
    rows: torch.Tensor,
    centers: torch.Tensor,
    ids: torch.Tensor,
    *,
    heads: int = 16,
    width: int = 128,
) -> torch.Tensor:
    """Fuse C64 center gather/add and flat-to-HSD BF16 packing."""
    if triton is None or not rows.is_cuda:
        raise RuntimeError("native center unpack requires CUDA Triton")
    if rows.ndim != 3 or rows.shape[-1] != heads * width:
        raise ValueError("decoder rows must be [B,S,H*D]")
    batch, sequence, native_width = rows.shape
    if centers.ndim != 3 or centers.shape[0] != batch:
        raise ValueError("native centers must be [B,C,H*D]")
    if centers.shape[-1] != native_width or ids.shape != (batch, sequence):
        raise ValueError("native centers/ids disagree with decoder rows")
    if any(item.device != rows.device for item in (centers, ids)):
        raise ValueError("native center tensors must share a CUDA device")
    if rows.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("decoder rows must be FP16/BF16")
    if centers.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("native centers must be FP16/BF16/FP32")
    out = torch.empty(
        batch, heads, sequence, width, device=rows.device, dtype=rows.dtype
    )
    total = out.numel()
    block = 256
    _native_center_unpack_kernel[(triton.cdiv(total, block),)](
        rows, centers, ids, out,
        total, heads, sequence, width,
        *rows.stride(), *centers.stride(), *ids.stride(), *out.stride(),
        BLOCK=block, num_warps=4,
    )
    return out


def _validate(
    x: torch.Tensor,
    matrix: torch.Tensor | None = None,
    *,
    allow_float32: bool = False,
) -> None:
    if triton is None or not x.is_cuda:
        raise RuntimeError("codec-native position kernels require CUDA Triton")
    if x.ndim != 4 or x.shape[-1] % 4:
        raise ValueError(f"expected [B,H,S,D] with D divisible by 4, got {x.shape}")
    allowed = (torch.float16, torch.bfloat16, torch.float32) if allow_float32 else (
        torch.float16, torch.bfloat16
    )
    if x.dtype not in allowed:
        raise ValueError(
            "position kernels require FP16/BF16"
            + ("/FP32" if allow_float32 else "")
            + " tensors"
        )
    if matrix is not None and matrix.shape != (x.shape[0], x.shape[2], 4, 4):
        raise ValueError(
            f"matrix must be [B,S,4,4], got {matrix.shape} for input {x.shape}"
        )


def selected_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    _validate(x)
    batch, heads, sequence, width = x.shape
    if indices.shape != (heads, width):
        raise ValueError(f"indices must be [{heads},{width}], got {indices.shape}")
    if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[0] != sequence:
        raise ValueError("cos/sin must be matching [S,D_native] tensors")
    cos = cos.to(device=x.device)
    sin = sin.to(device=x.device)
    indices = indices.to(device=x.device)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    total = x.numel()
    block = 256
    _selected_rope_kernel[(triton.cdiv(total, block),)](
        x, cos, sin, indices, out,
        total, heads, sequence, width,
        *x.stride(), *cos.stride(), *sin.stride(), *indices.stride(),
        BLOCK=block, num_warps=4,
    )
    return out


def prope_one(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    # Query quartet reduction intentionally produces FP32 in the established
    # runtime; preserve that precision until the caller's existing BF16 cast.
    _validate(x, matrix, allow_float32=True)
    matrix = matrix.to(device=x.device)
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    total = x.numel()
    block = 256
    _prope_one_kernel[(triton.cdiv(total, block),)](
        x, matrix, out,
        total, x.shape[1], x.shape[2], x.shape[3],
        *x.stride(), *matrix.stride(),
        BLOCK=block, num_warps=4,
    )
    return out


def prope_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(key, matrix)
    _validate(value, matrix)
    if key.shape != value.shape:
        raise ValueError("PRoPE K/V pair must have matching shapes")
    matrix = matrix.to(device=key.device)
    out_key = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    out_value = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    total = key.numel()
    block = 256
    _prope_pair_kernel[(triton.cdiv(total, block),)](
        key, value, matrix, out_key, out_value,
        total, key.shape[1], key.shape[2], key.shape[3],
        *key.stride(), *value.stride(), *matrix.stride(),
        BLOCK=block, num_warps=4,
    )
    return out_key, out_value


def center_pair_add(
    first: torch.Tensor,
    second: torch.Tensor,
    first_table: torch.Tensor,
    second_table: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(first)
    _validate(second)
    if first.shape != second.shape:
        raise ValueError("center-add pair must have matching input shapes")
    batch, heads, sequence, width = first.shape
    expected_prefix = (batch, heads)
    if (
        first_table.shape[:2] != expected_prefix
        or second_table.shape[:2] != expected_prefix
        or first_table.shape[-1] != width
        or second_table.shape[-1] != width
    ):
        raise ValueError("center tables do not match [B,H,clusters,D]")
    if ids.shape != (batch, sequence):
        raise ValueError(f"ids must be [{batch},{sequence}], got {ids.shape}")
    first_table = first_table.to(device=first.device)
    second_table = second_table.to(device=first.device)
    ids = ids.to(device=first.device)
    out_first = torch.empty(first.shape, device=first.device, dtype=first.dtype)
    out_second = torch.empty(second.shape, device=second.device, dtype=second.dtype)
    total = first.numel()
    block = 256
    _center_pair_add_kernel[(triton.cdiv(total, block),)](
        first, second, first_table, second_table, ids,
        out_first, out_second,
        total, heads, sequence, width,
        *first.stride(), *second.stride(),
        *first_table.stride(), *second_table.stride(), *ids.stride(),
        BLOCK=block, num_warps=4,
    )
    return out_first, out_second


def lse_combine(
    native: torch.Tensor,
    native_lse: torch.Tensor,
    history: torch.Tensor,
    history_lse: torch.Tensor,
) -> torch.Tensor:
    if triton is None or not native.is_cuda:
        raise RuntimeError("LSE combine requires CUDA Triton")
    if native.ndim != 4 or history.shape != native.shape:
        raise ValueError("native/history outputs must have matching [B,H,Q,D]")
    if native_lse.shape != native.shape[:-1] or history_lse.shape != native_lse.shape:
        raise ValueError("segment LSE tensors must match [B,H,Q]")
    out = torch.empty(native.shape, device=native.device, dtype=native.dtype)
    total = native.numel()
    block = 256
    _lse_combine_kernel[(triton.cdiv(total, block),)](
        native, native_lse, history, history_lse, out,
        total, native.shape[1], native.shape[2], native.shape[3],
        *native.stride(), *native_lse.stride(),
        *history.stride(), *history_lse.stride(),
        BLOCK=block, num_warps=4,
    )
    return out


def native_kv_cat(
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    anchor_k: torch.Tensor | None,
    anchor_v: torch.Tensor | None,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or not text_k.is_cuda:
        raise RuntimeError("native K/V cat requires CUDA Triton")
    pairs = ((text_k, text_v), (recent_k, recent_v), (current_k, current_v))
    if any(k.shape != v.shape for k, v in pairs):
        raise ValueError("every native K/V segment must have matching shapes")
    if any(k.ndim != 4 for k, _v in pairs):
        raise ValueError("native K/V segments must be [B,H,S,D]")
    if any(k.stride() != v.stride() for k, v in pairs):
        raise ValueError("native K/V pairs must have matching strides")
    if any(
        k.device != text_k.device or v.device != text_k.device
        or k.dtype != text_k.dtype or v.dtype != text_k.dtype
        for k, v in pairs
    ):
        raise ValueError("native K/V segments must share device and dtype")
    batch, heads, _text_sequence, width = text_k.shape
    for k, _v in pairs[1:]:
        if k.shape[:2] != (batch, heads) or k.shape[-1] != width:
            raise ValueError("native K/V segments disagree on B/H/D")
    if anchor_k is None:
        if anchor_v is not None:
            raise ValueError("anchor K/V must both be absent")
        anchor_k = recent_k
        anchor_v = recent_v
        anchor_sequence = 0
    else:
        if anchor_v is None or anchor_k.shape != anchor_v.shape:
            raise ValueError("anchor K/V must have matching shapes")
        if anchor_k.ndim != 4 or anchor_k.stride() != anchor_v.stride():
            raise ValueError("anchor K/V must be matching [B,H,S,D] layouts")
        if (
            anchor_k.device != text_k.device
            or anchor_v.device != text_k.device
            or anchor_k.dtype != text_k.dtype
            or anchor_v.dtype != text_k.dtype
        ):
            raise ValueError("anchor K/V must share device and dtype")
        if anchor_k.shape[:2] != (batch, heads) or anchor_k.shape[-1] != width:
            raise ValueError("anchor K/V disagree on B/H/D")
        anchor_sequence = anchor_k.shape[2]
    final_sequence = (
        text_k.shape[2] + anchor_sequence + recent_k.shape[2] + current_k.shape[2]
    )
    shape = (batch, heads, final_sequence, width)
    out_k = torch.empty(shape, device=text_k.device, dtype=text_k.dtype)
    out_v = torch.empty(shape, device=text_v.device, dtype=text_v.dtype)
    total = out_k.numel()
    block = 256
    _native_kv_cat_kernel[(triton.cdiv(total, block),)](
        text_k, text_v, anchor_k, anchor_v,
        recent_k, recent_v, current_k, current_v,
        out_k, out_v,
        total, heads, final_sequence, width,
        text_k.shape[2], anchor_sequence, recent_k.shape[2],
        *text_k.stride(), *anchor_k.stride(),
        *recent_k.stride(), *current_k.stride(),
        BLOCK=block, num_warps=4,
    )
    return out_k, out_v


def center_position_pack(
    key_main: torch.Tensor,
    key_prope: torch.Tensor,
    value_main: torch.Tensor,
    value_prope: torch.Tensor,
    key_main_table: torch.Tensor,
    key_prope_table: torch.Tensor,
    value_main_table: torch.Tensor,
    value_prope_table: torch.Tensor,
    key_ids: torch.Tensor,
    value_ids: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indices: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse C64 centre add, RoPE/PRoPE, and main/PRoPE packing."""
    tensors = (key_main, key_prope, value_main, value_prope)
    _validate(key_main, matrix)
    if any(x.shape != key_main.shape for x in tensors[1:]):
        raise ValueError("low K/V main and PRoPE tensors must share shape")
    if any(x.stride() != key_main.stride() for x in tensors[1:]):
        raise ValueError("low K/V main and PRoPE tensors must share strides")
    if any(
        x.device != key_main.device or x.dtype != key_main.dtype
        for x in tensors[1:]
    ):
        raise ValueError("low K/V tensors must share device and dtype")
    batch, heads, sequence, width = key_main.shape
    tables = (
        key_main_table, key_prope_table,
        value_main_table, value_prope_table,
    )
    expected_table_prefix = (batch, heads)
    if any(
        table.ndim != 4
        or table.shape[:2] != expected_table_prefix
        or table.shape[-1] != width
        for table in tables
    ):
        raise ValueError("C64 tables must match [B,H,clusters,D]")
    if any(table.stride() != key_main_table.stride() for table in tables[1:]):
        raise ValueError("C64 tables must share strides")
    if any(
        table.device != key_main.device or table.dtype != key_main.dtype
        for table in tables
    ):
        raise ValueError("C64 tables must share low K/V device and dtype")
    if key_ids.shape != (batch, sequence) or value_ids.shape != key_ids.shape:
        raise ValueError("C64 IDs must match [B,S]")
    if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[0] != sequence:
        raise ValueError("cos/sin must be matching [S,D_native]")
    if indices.shape != (heads, width):
        raise ValueError(f"indices must be [{heads},{width}]")

    cos = cos.to(device=key_main.device)
    sin = sin.to(device=key_main.device)
    indices = indices.to(device=key_main.device)
    matrix = matrix.to(device=key_main.device)
    key_ids = key_ids.to(device=key_main.device)
    value_ids = value_ids.to(device=key_main.device)
    output_shape = (2 * batch, heads, sequence, width)
    out_key = torch.empty(output_shape, device=key_main.device, dtype=key_main.dtype)
    out_value = torch.empty_like(out_key)
    total = out_key.numel()
    block = 256
    _center_position_pack_kernel[(triton.cdiv(total, block),)](
        key_main, key_prope, value_main, value_prope,
        key_main_table, key_prope_table,
        value_main_table, value_prope_table,
        key_ids, value_ids, cos, sin, indices, matrix,
        out_key, out_value,
        total, batch, heads, sequence, width,
        *key_main.stride(), *value_main.stride(),
        *key_main_table.stride(), *value_main_table.stride(),
        *key_ids.stride(), *value_ids.stride(),
        *cos.stride(), *sin.stride(), *indices.stride(), *matrix.stride(),
        BLOCK=block, num_warps=4,
    )
    return out_key, out_value


def grouped_scatter_pair(
    main: torch.Tensor,
    prope: torch.Tensor,
    lse: torch.Tensor,
    head_indices: torch.Tensor,
    full: torch.Tensor,
    full_lse: torch.Tensor,
) -> None:
    """Pack one heterogeneous-width group into canonical H16 outputs."""
    if triton is None or not main.is_cuda:
        raise RuntimeError("grouped scatter requires CUDA Triton")
    if main.ndim != 4 or prope.shape != main.shape:
        raise ValueError("main/prope must have matching [B,G,Q,128] shapes")
    batch, group_heads, queries, width = main.shape
    if lse.shape != (2 * batch, group_heads, queries):
        raise ValueError("group LSE must be [2B,G,Q]")
    if head_indices.shape != (group_heads,):
        raise ValueError("head indices must be [G]")
    expected_full = (2 * batch, full.shape[1], queries, width)
    if full.shape != expected_full or full_lse.shape != expected_full[:-1]:
        raise ValueError("canonical outputs disagree with group B/Q/D")
    if any(
        tensor.device != main.device
        for tensor in (prope, lse, head_indices, full, full_lse)
    ):
        raise ValueError("group scatter tensors must share a CUDA device")
    total = 2 * main.numel()
    block = 256
    _grouped_scatter_pair_kernel[(triton.cdiv(total, block),)](
        main, prope, lse, head_indices, full, full_lse,
        total, batch, group_heads, queries, width,
        *main.stride(), *prope.stride(), *lse.stride(),
        head_indices.stride(0), *full.stride(), *full_lse.stride(),
        BLOCK=block, num_warps=4,
    )


def native_position_pack(
    key: torch.Tensor,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse native D128 RoPE/PRoPE and Main/PRoPE batch packing."""
    if triton is None or not key.is_cuda:
        raise RuntimeError("native position pack requires CUDA Triton")
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("native key/value must share [B,H,S,D]")
    batch, heads, sequence, width = key.shape
    if width != 128:
        raise ValueError("native position pack requires D128")
    if value.device != key.device or value.dtype != key.dtype:
        raise ValueError("native key/value must share device and dtype")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("native key/value must be FP16/BF16")
    if cos.shape != sin.shape or cos.shape != (sequence, width):
        raise ValueError("native cos/sin must be matching [S,128]")
    if matrix.shape != (batch, sequence, 4, 4):
        raise ValueError("native PRoPE matrix must be [B,S,4,4]")
    if any(item.device != key.device for item in (cos, sin, matrix)):
        raise ValueError("native position tensors must share a CUDA device")
    out_shape = (2 * batch, heads, sequence, width)
    out_key = torch.empty(out_shape, device=key.device, dtype=key.dtype)
    out_value = torch.empty_like(out_key)
    total = out_key.numel()
    block = 256
    _native_position_pack_kernel[(triton.cdiv(total, block),)](
        key, value, cos, sin, matrix, out_key, out_value,
        total, batch, heads, sequence, width,
        *key.stride(), *value.stride(),
        *cos.stride(), *sin.stride(), *matrix.stride(),
        BLOCK=block, num_warps=4,
    )
    return out_key, out_value


def native_prope_pack(
    main_key: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse native PRoPE/packing while preserving an exact staged RoPE K."""
    if triton is None or not key.is_cuda:
        raise RuntimeError("native PRoPE pack requires CUDA Triton")
    if key.ndim != 4 or value.shape != key.shape or main_key.shape != key.shape:
        raise ValueError("native tensors must share [B,H,S,D]")
    batch, heads, sequence, width = key.shape
    if width != 128 or matrix.shape != (batch, sequence, 4, 4):
        raise ValueError("native PRoPE pack requires D128 and [B,S,4,4]")
    values = (main_key, value, matrix)
    if any(item.device != key.device for item in values):
        raise ValueError("native PRoPE tensors must share a CUDA device")
    if main_key.dtype != key.dtype or value.dtype != key.dtype:
        raise ValueError("native K/V tensors must share dtype")
    if key.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("native K/V tensors must be FP16/BF16")
    out_shape = (2 * batch, heads, sequence, width)
    out_key = torch.empty(out_shape, device=key.device, dtype=key.dtype)
    out_value = torch.empty_like(out_key)
    total = out_key.numel()
    block = 256
    _native_prope_pack_kernel[(triton.cdiv(total, block),)](
        main_key, key, value, matrix, out_key, out_value,
        total, batch, heads, sequence, width,
        *main_key.stride(), *key.stride(), *value.stride(), *matrix.stride(),
        BLOCK=block, num_warps=4,
    )
    return out_key, out_value


def grouped_lse_combine_scatter(
    native: torch.Tensor,
    native_lse: torch.Tensor,
    main: torch.Tensor,
    prope: torch.Tensor,
    history_lse: torch.Tensor,
    head_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Merge one heterogeneous-width group and write its canonical heads."""
    if triton is None or not native.is_cuda:
        raise RuntimeError("grouped combine scatter requires CUDA Triton")
    if native.ndim != 4 or native.shape[0] % 2:
        raise ValueError("native output must be [2B,H,Q,D]")
    batch = native.shape[0] // 2
    if main.ndim != 4 or prope.shape != main.shape:
        raise ValueError("main/prope must share [B,G,Q,D]")
    if main.shape[0] != batch or main.shape[2:] != native.shape[2:]:
        raise ValueError("group output must match native B/Q/D")
    group_heads = main.shape[1]
    if history_lse.shape != (2 * batch, group_heads, native.shape[2]):
        raise ValueError("history LSE must be [2B,G,Q]")
    if native_lse.shape != native.shape[:-1]:
        raise ValueError("native LSE must be [2B,H,Q]")
    if head_indices.shape != (group_heads,) or out.shape != native.shape:
        raise ValueError("head indices/output disagree with group/native")
    values = (native_lse, main, prope, history_lse, head_indices, out)
    if any(value.device != native.device for value in values):
        raise ValueError("grouped combine tensors must share a CUDA device")
    total = 2 * main.numel()
    block = 256
    _grouped_lse_combine_scatter_kernel[(triton.cdiv(total, block),)](
        native, native_lse, main, prope, history_lse,
        head_indices, out,
        total, batch, group_heads, native.shape[2], native.shape[3],
        *native.stride(), *native_lse.stride(),
        *main.stride(), *prope.stride(), *history_lse.stride(),
        head_indices.stride(0), *out.stride(),
        BLOCK=block, num_warps=4,
    )


def lse_combine_split(
    native: torch.Tensor,
    native_lse: torch.Tensor,
    main: torch.Tensor,
    prope: torch.Tensor,
    history_lse: torch.Tensor,
) -> torch.Tensor:
    """Merge split lifted history without materializing its FP32 cat."""
    if triton is None or not native.is_cuda:
        raise RuntimeError("split LSE combine requires CUDA Triton")
    if native.ndim != 4 or native.shape[0] % 2:
        raise ValueError("native output must be [2B,H,Q,D]")
    batch = native.shape[0] // 2
    expected = (batch, *native.shape[1:])
    if main.shape != expected or prope.shape != expected:
        raise ValueError("split history outputs must each be [B,H,Q,D]")
    if native_lse.shape != native.shape[:-1] or history_lse.shape != native_lse.shape:
        raise ValueError("segment LSE tensors must match [2B,H,Q]")
    if any(
        tensor.device != native.device
        for tensor in (native_lse, main, prope, history_lse)
    ):
        raise ValueError("split LSE tensors must share a CUDA device")
    # The kernel writes a flat logical [2B,H,Q,D] index.  Allocate canonical
    # contiguous storage rather than preserving a FlashInfer-style permuted
    # output stride; preserving that stride silently permutes the merged value.
    out = torch.empty(native.shape, device=native.device, dtype=native.dtype)
    total = native.numel()
    block = 256
    _lse_combine_split_kernel[(triton.cdiv(total, block),)](
        native, native_lse, main, prope, history_lse, out,
        total, batch, native.shape[1], native.shape[2], native.shape[3],
        *native.stride(), *native_lse.stride(),
        *main.stride(), *prope.stride(), *history_lse.stride(),
        BLOCK=block, num_warps=4,
    )
    return out


def lift_lse_combine(
    native: torch.Tensor,
    native_lse: torch.Tensor,
    low: torch.Tensor,
    main_lift: torch.Tensor,
    prope_mix: torch.Tensor,
    history_lse: torch.Tensor,
) -> torch.Tensor:
    """Fuse uniform Main/PRoPE value lift with the exact segment merge.

    The two kernels write directly into the Main and PRoPE halves of the final
    D128 output, avoiding both lifted-history temporaries and their readback by
    a separate LSE-combine kernel.  This is an experimental inference path;
    the established split implementation remains the default.
    """
    if triton is None or not native.is_cuda:
        raise RuntimeError("fused lift/LSE combine requires CUDA Triton")
    if native.ndim != 4 or native.shape[0] % 2:
        raise ValueError("native output must be [2B,H,Q,128]")
    batch, heads, queries, width = (
        native.shape[0] // 2, native.shape[1], native.shape[2], native.shape[3]
    )
    if width != 128 or low.shape[:3] != (2 * batch, heads, queries):
        raise ValueError("fused lift expects [2B,H,Q,Dlow] and D128 native")
    low_dim = low.shape[-1]
    if low_dim not in {32, 64, 96} or low_dim % 4:
        raise ValueError("fused lift supports D32/D64/D96")
    if main_lift.shape != (heads, low_dim, 128):
        raise ValueError("main lift must be [H,Dlow,128]")
    if prope_mix.shape != (heads, low_dim // 4, 32):
        raise ValueError("PRoPE lift must be [H,Dlow/4,32]")
    if native_lse.shape != native.shape[:-1] or history_lse.shape != native_lse.shape:
        raise ValueError("segment LSE tensors must match [2B,H,Q]")
    tensors = (native_lse, low, main_lift, prope_mix, history_lse)
    if any(tensor.device != native.device for tensor in tensors):
        raise ValueError("fused lift tensors must share a CUDA device")

    out = torch.empty_like(native)
    block_m = 32
    block_n = 64
    _main_lift_lse_combine_kernel[
        (triton.cdiv(queries, block_m), batch * heads, triton.cdiv(128, block_n))
    ](
        low, main_lift, native, native_lse, history_lse, out,
        batch, heads, queries, low_dim,
        *low.stride(), *main_lift.stride(), *native.stride(),
        *native_lse.stride(), *history_lse.stride(), *out.stride(),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    _prope_lift_lse_combine_kernel[
        (triton.cdiv(queries, block_m), batch * heads, 4)
    ](
        low[batch:], prope_mix, native, native_lse, history_lse, out,
        batch, heads, queries, low_dim // 4,
        *low[batch:].stride(), *prope_mix.stride(), *native.stride(),
        *native_lse.stride(), *history_lse.stride(), *out.stride(),
        BLOCK_M=block_m, BLOCK_N=32, BLOCK_K=16,
        num_warps=4, num_stages=2,
    )
    return out


def query_position_pack(
    main: torch.Tensor,
    prope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indices: torch.Tensor,
    matrix: torch.Tensor,
    main_scale: torch.Tensor,
    prope_scale: torch.Tensor,
) -> torch.Tensor:
    """Fuse low-query RoPE/PRoPE, residual scales, casts, and packing."""
    if triton is None or not main.is_cuda:
        raise RuntimeError("query position pack requires CUDA Triton")
    if main.ndim != 4 or prope.shape != main.shape:
        raise ValueError("query main/prope must share [B,H,Q,D]")
    batch, heads, queries, width = main.shape
    if main.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("query main must be FP16/BF16")
    if prope.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("query PRoPE reduction must be FP16/BF16/FP32")
    if cos.shape != sin.shape or cos.ndim != 2 or cos.shape[0] != queries:
        raise ValueError("query cos/sin must be matching [Q,D_native]")
    if indices.shape != (heads, width):
        raise ValueError("query indices must be [H,D]")
    if matrix.shape != (batch, queries, 4, 4):
        raise ValueError("query PRoPE matrix must be [B,Q,4,4]")
    if main_scale.shape != (heads,) or prope_scale.shape != (heads,):
        raise ValueError("query residual scales must be [H]")
    values = (prope, cos, sin, indices, matrix, main_scale, prope_scale)
    if any(value.device != main.device for value in values):
        raise ValueError("query position tensors must share a CUDA device")
    out = torch.empty(
        (2 * batch, heads, queries, width),
        device=main.device,
        dtype=main.dtype,
    )
    total = out.numel()
    block = 256
    _query_position_pack_kernel[(triton.cdiv(total, block),)](
        main, prope, cos, sin, indices, matrix,
        main_scale, prope_scale, out,
        total, batch, heads, queries, width,
        *main.stride(), *prope.stride(),
        *cos.stride(), *sin.stride(), *indices.stride(), *matrix.stride(),
        main_scale.stride(0), prope_scale.stride(0),
        BLOCK=block, num_warps=4,
    )
    return out


def query_project_position_pack(
    query: torch.Tensor,
    main_reduce: torch.Tensor,
    prope_mix: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    indices: torch.Tensor,
    matrix: torch.Tensor,
    main_scale: torch.Tensor,
    prope_scale: torch.Tensor,
) -> torch.Tensor:
    """Project D128 Q and apply both position systems without intermediates."""
    if triton is None or not query.is_cuda:
        raise RuntimeError("fused query projection requires CUDA Triton")
    if query.ndim != 4 or query.shape[-1] != 128:
        raise ValueError("query must be [B,H,Q,128]")
    batch, heads, queries, _ = query.shape
    width = main_reduce.shape[1]
    if width not in {32, 64, 96} or width % 4:
        raise ValueError("fused query projection supports D32/D64/D96")
    if main_reduce.shape != (heads, width, 128):
        raise ValueError("main reduction must be [H,D,128]")
    if prope_mix.shape != (heads, width // 4, 32):
        raise ValueError("PRoPE mix must be [H,D/4,32]")
    if cos.shape != sin.shape or cos.shape[0] != queries:
        raise ValueError("cos/sin must be matching [Q,Dnative]")
    if indices.shape != (heads, width):
        raise ValueError("indices must be [H,D]")
    if matrix.shape != (batch, queries, 4, 4):
        raise ValueError("PRoPE matrix must be [B,Q,4,4]")
    if main_scale.shape != (heads,) or prope_scale.shape != (heads,):
        raise ValueError("query scales must be [H]")
    tensors = (
        main_reduce, prope_mix, cos, sin, indices,
        matrix, main_scale, prope_scale,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("fused query tensors must share a CUDA device")
    out = torch.empty(
        (2 * batch, heads, queries, width),
        device=query.device, dtype=query.dtype,
    )
    block_m = 16
    block_pairs = 16
    _main_query_project_rope_kernel[
        (
            triton.cdiv(queries, block_m), batch * heads,
            triton.cdiv(width // 2, block_pairs),
        )
    ](
        query, main_reduce, cos, sin, indices, main_scale, out,
        batch, heads, queries, width,
        *query.stride(), *main_reduce.stride(),
        *cos.stride(), *sin.stride(), *indices.stride(),
        main_scale.stride(0), *out.stride(),
        BLOCK_M=block_m, BLOCK_P=block_pairs, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    # Tensor-Core dot requires N>=16; smaller D32 quartets are zero-masked.
    block_quartets = 16
    _prope_query_project_position_kernel[
        (
            triton.cdiv(queries, block_m), batch * heads,
            triton.cdiv(width // 4, block_quartets),
        )
    ](
        query, prope_mix, matrix, prope_scale, out,
        batch, heads, queries, width // 4,
        *query.stride(), *prope_mix.stride(), *matrix.stride(),
        prope_scale.stride(0), *out.stride(),
        BLOCK_M=block_m, BLOCK_R=block_quartets, BLOCK_K=16,
        num_warps=4, num_stages=2,
    )
    return out
