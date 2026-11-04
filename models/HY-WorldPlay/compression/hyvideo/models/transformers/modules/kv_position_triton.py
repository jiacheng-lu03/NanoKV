"""Lossless Triton helpers for pre-position KV materialization."""

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _concat_final_kv_kernel(
        enc_k, enc_v,
        hist_k_main, hist_v_main,
        hist_k_prope, hist_v_prope,
        cur_k, cur_v,
        out_k, out_v,
        total: tl.constexpr,
        batch: tl.constexpr,
        heads: tl.constexpr,
        text_len: tl.constexpr,
        history_len: tl.constexpr,
        current_len: tl.constexpr,
        head_dim: tl.constexpr,
        final_len: tl.constexpr,
        enc_sb: tl.constexpr, enc_sh: tl.constexpr,
        enc_ss: tl.constexpr, enc_sd: tl.constexpr,
        hm_sb: tl.constexpr, hm_sh: tl.constexpr,
        hm_ss: tl.constexpr, hm_sd: tl.constexpr,
        hp_sb: tl.constexpr, hp_sh: tl.constexpr,
        hp_ss: tl.constexpr, hp_sd: tl.constexpr,
        cur_sb: tl.constexpr, cur_sh: tl.constexpr,
        cur_ss: tl.constexpr, cur_sd: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        d = offsets % head_dim
        tmp = offsets // head_dim
        pos = tmp % final_len
        tmp = tmp // final_len
        h = tmp % heads
        virtual_b = tmp // heads
        local_b = virtual_b % batch
        branch = virtual_b // batch

        is_text = valid & (pos < text_len)
        is_history = valid & (pos >= text_len) & (pos < text_len + history_len)
        is_current = valid & (pos >= text_len + history_len)

        text_pos = pos
        text_offset = (
            virtual_b * enc_sb + h * enc_sh + text_pos * enc_ss + d * enc_sd
        )
        history_pos = pos - text_len
        history_main_offset = (
            local_b * hm_sb + h * hm_sh + history_pos * hm_ss + d * hm_sd
        )
        history_prope_offset = (
            local_b * hp_sb + h * hp_sh + history_pos * hp_ss + d * hp_sd
        )
        current_pos = pos - text_len - history_len
        current_offset = (
            virtual_b * cur_sb + h * cur_sh + current_pos * cur_ss + d * cur_sd
        )

        key = tl.load(enc_k + text_offset, mask=is_text, other=0.0)
        value = tl.load(enc_v + text_offset, mask=is_text, other=0.0)
        main_mask = is_history & (branch == 0)
        prope_mask = is_history & (branch != 0)
        history_key = tl.where(
            main_mask,
            tl.load(hist_k_main + history_main_offset, mask=main_mask, other=0.0),
            tl.load(hist_k_prope + history_prope_offset, mask=prope_mask, other=0.0),
        )
        history_value = tl.where(
            main_mask,
            tl.load(hist_v_main + history_main_offset, mask=main_mask, other=0.0),
            tl.load(hist_v_prope + history_prope_offset, mask=prope_mask, other=0.0),
        )
        key = tl.where(is_history, history_key, key)
        value = tl.where(is_history, history_value, value)
        key = tl.where(
            is_current,
            tl.load(cur_k + current_offset, mask=is_current, other=0.0),
            key,
        )
        value = tl.where(
            is_current,
            tl.load(cur_v + current_offset, mask=is_current, other=0.0),
            value,
        )
        tl.store(out_k + offsets, key, mask=valid)
        tl.store(out_v + offsets, value, mask=valid)
else:
    _concat_final_kv_kernel = None


def concat_final_kv(
    encoder_key: torch.Tensor,
    encoder_value: torch.Tensor,
    history_key_main: torch.Tensor,
    history_value_main: torch.Tensor,
    history_key_prope: torch.Tensor,
    history_value_prope: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
):
    """Materialize final [text, history, current] K/V in one copy kernel."""
    tensors = (
        encoder_key, encoder_value, history_key_main, history_value_main,
        history_key_prope, history_value_prope, current_key, current_value,
    )
    if _concat_final_kv_kernel is None or not all(x.is_cuda for x in tensors):
        raise RuntimeError("Triton final KV concat requires CUDA Triton tensors")
    for lhs, rhs, name in (
        (encoder_key, encoder_value, "encoder"),
        (history_key_main, history_value_main, "history_main"),
        (history_key_prope, history_value_prope, "history_prope"),
        (current_key, current_value, "current"),
    ):
        if lhs.shape != rhs.shape or lhs.stride() != rhs.stride():
            raise ValueError(f"{name} K/V shape or stride mismatch")
    batch, heads, history_len, head_dim = history_key_main.shape
    if history_key_prope.shape != history_key_main.shape:
        raise ValueError("Main/ProPE history shape mismatch")
    if encoder_key.shape[0] != batch * 2 or current_key.shape[0] != batch * 2:
        raise ValueError("Expected virtual branch batch [2B,...]")
    text_len = encoder_key.shape[2]
    current_len = current_key.shape[2]
    final_len = text_len + history_len + current_len
    shape = (batch * 2, heads, final_len, head_dim)
    out_key = torch.empty(shape, device=current_key.device, dtype=current_key.dtype)
    out_value = torch.empty(shape, device=current_value.device, dtype=current_value.dtype)
    total = out_key.numel()
    block = 256
    _concat_final_kv_kernel[(triton.cdiv(total, block),)](
        encoder_key, encoder_value,
        history_key_main, history_value_main,
        history_key_prope, history_value_prope,
        current_key, current_value,
        out_key, out_value,
        total=total, batch=batch, heads=heads, text_len=text_len,
        history_len=history_len, current_len=current_len, head_dim=head_dim,
        final_len=final_len,
        enc_sb=encoder_key.stride(0), enc_sh=encoder_key.stride(1),
        enc_ss=encoder_key.stride(2), enc_sd=encoder_key.stride(3),
        hm_sb=history_key_main.stride(0), hm_sh=history_key_main.stride(1),
        hm_ss=history_key_main.stride(2), hm_sd=history_key_main.stride(3),
        hp_sb=history_key_prope.stride(0), hp_sh=history_key_prope.stride(1),
        hp_ss=history_key_prope.stride(2), hp_sd=history_key_prope.stride(3),
        cur_sb=current_key.stride(0), cur_sh=current_key.stride(1),
        cur_ss=current_key.stride(2), cur_sd=current_key.stride(3),
        BLOCK=block, num_warps=4,
    )
    return out_key, out_value
