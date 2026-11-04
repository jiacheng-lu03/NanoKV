"""Trainable, history-only residuals for codec-native attention.

This module is deliberately independent from the frozen codec checkpoints and
their manifests.  It only adapts the already constructed low-dimensional
history branch:

* a per-head low-rank residual on the low-dimensional query;
* branch-specific per-head low-rank residuals on positioned low-dimensional
  history K/V (Main and PRoPE remain separate);
* optional foldable rank-r residual reads from the same formal z512 latent,
  applied before RoPE/PRoPE so training can discover previously unused codec
  directions without changing the cache payload or attention width;
* optional foldable rank-r residual reads from the same formal C64 semantic
  centers, so the dictionary and residual halves of the codec can be adapted
  independently without changing their stored payload;
* a per-head low-rank residual on the lifted D128 history output; and
* an optional per-branch/per-head additive history log-sum-exp bias.

All trainable ``B`` factors and the LSE bias are initialized to zero, making a
fresh adapter an exact identity.  The native text/current branch never enters
this module.  Runtime use is opt-in through
``HY_CODEC_NATIVE_HISTORY_ADAPTER_ENABLE=1``; autograd is additionally gated
by ``HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN=1``.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn


# WorldPlay has 54 double-stream blocks.  Layer selection is intentionally
# explicit: the old avg56 diagnostic uses L27-35, whereas the near-2x manifest
# uses its 45 non-hot low-D layers (and its L27-35 are formal-native, so they
# must not be selected there).
_ALLOWED_LAYERS = frozenset(range(54))
_FORMAT = "codec_native_history_adapter_v1"


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def enabled() -> bool:
    return _flag("HY_CODEC_NATIVE_HISTORY_ADAPTER_ENABLE")


def training_enabled() -> bool:
    """Whether functional training/optimizer support was explicitly requested."""

    return enabled() and _flag("HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN")


def autograd_enabled() -> bool:
    """Open autograd only inside the functional trainer's Student forward."""

    return (
        training_enabled()
        and os.environ.get("HY_FUNCTIONAL_BLOCK_FT_PASS", "").strip().lower()
        == "student"
    )


def selected_targets() -> set[str]:
    value = os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_TARGETS")
    if value is None:
        payload = _checkpoint_payload()
        checkpoint_targets = (
            None if payload is None else payload.get("config", {}).get("targets")
        )
        value = (
            ",".join(str(target) for target in checkpoint_targets)
            if checkpoint_targets is not None
            else "output,lse"
        )
    targets = {item.strip().lower() for item in value.split(",") if item.strip()}
    supported = {
        "query", "query_read", "key", "value", "latent", "center",
        "lift", "output", "lse",
    }
    if not targets or targets - supported:
        raise ValueError(
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_TARGETS must be a non-empty subset "
            f"of {sorted(supported)}; got {sorted(targets)}"
        )
    return targets


def selected_layers() -> set[int]:
    value = os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS")
    if value is None:
        payload = _checkpoint_payload()
        checkpoint_layers = (
            None if payload is None else payload.get("config", {}).get("layers")
        )
        value = (
            ",".join(str(layer) for layer in checkpoint_layers)
            if checkpoint_layers is not None
            else ""
        )
    value = value.strip()
    try:
        layers = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as error:
        raise ValueError(
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS must be a comma-separated "
            "integer list"
        ) from error
    if not layers:
        raise ValueError(
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_LAYERS must explicitly list the "
            "active manifest's low-D layers (or be inherited from a checkpoint)"
        )
    unsupported = layers - _ALLOWED_LAYERS
    if unsupported:
        raise ValueError(
            "codec-native history-adapter layers must be WorldPlay double "
            f"blocks L0-53; got unsupported layers={sorted(unsupported)}"
        )
    return layers


def selected_train_layers() -> set[int]:
    """Return the optional focused optimizer subset.

    Runtime/deployment layers remain ``selected_layers()``.  This training-only
    subset lets a full existing Query adapter stay active while a zero-init
    residual is optimized at one diagnosed bottleneck layer.  Leaving the
    variable unset preserves the original all-selected-layer behavior.
    """

    layers = selected_layers()
    value = os.environ.get(
        "HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN_LAYERS", ""
    ).strip()
    if not value:
        return layers
    try:
        train_layers = {
            int(item.strip()) for item in value.split(",") if item.strip()
        }
    except ValueError as error:
        raise ValueError(
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN_LAYERS must be a "
            "comma-separated integer list"
        ) from error
    if not train_layers:
        raise ValueError(
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN_LAYERS must not be empty "
            "when set"
        )
    unsupported = train_layers - layers
    if unsupported:
        raise ValueError(
            "history-adapter train layers must be a subset of active layers: "
            f"unsupported={sorted(unsupported)} active={sorted(layers)}"
        )
    return train_layers


class CodecNativeHistoryLayerAdapter(nn.Module):
    """Zero-init per-head LoRA residuals for one old-low attention branch."""

    def __init__(
        self,
        low_dim: int,
        *,
        heads: int = 16,
        output_dim: int = 128,
        rank: int = 8,
        alpha: float = 8.0,
        targets: Iterable[str] = ("output", "lse"),
        init_seed: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if min(low_dim, heads, output_dim, rank) <= 0:
            raise ValueError(
                "history adapter dimensions/rank must be positive: "
                f"low_dim={low_dim} heads={heads} output_dim={output_dim} rank={rank}"
            )
        self.low_dim = int(low_dim)
        self.heads = int(heads)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.targets = frozenset(str(target) for target in targets)
        unsupported = self.targets - {
            "query", "query_read", "key", "value", "latent", "center",
            "lift", "output", "lse",
        }
        if not self.targets or unsupported:
            raise ValueError(f"invalid history-adapter targets: {sorted(self.targets)}")

        factory = {"device": device, "dtype": dtype}
        init_generator = None
        if init_seed is not None:
            generator_device = torch.device("cpu" if device is None else device)
            init_generator = torch.Generator(device=generator_device)
            init_generator.manual_seed(int(init_seed))

        def init_a(parameter: torch.Tensor) -> None:
            # A local generator makes the trainable subspace reproducible and,
            # crucially, does not advance the world-model rollout RNG.
            nn.init.kaiming_uniform_(
                parameter,
                a=math.sqrt(5),
                generator=init_generator,
            )

        if "query" in self.targets:
            self.query_a = nn.Parameter(torch.empty(heads, low_dim, rank, **factory))
            self.query_b = nn.Parameter(torch.zeros(heads, rank, low_dim, **factory))
            init_a(self.query_a)
        else:
            self.register_parameter("query_a", None)
            self.register_parameter("query_b", None)
        # A structure-preserving alternative to ``query``.  These factors
        # parameterize additive deltas to the existing Main [H,D,128] and
        # PRoPE [H,D/4,32] query-read matrices directly.  Consequently both
        # branches remain exactly foldable without mixing PRoPE quartets.
        if "query_read" in self.targets:
            if low_dim % 4:
                raise ValueError(
                    f"query-read requires low_dim divisible by 4, got {low_dim}"
                )
            self.query_main_a = nn.Parameter(
                torch.empty(heads, 128, rank, **factory)
            )
            self.query_main_b = nn.Parameter(
                torch.zeros(heads, rank, low_dim, **factory)
            )
            self.query_prope_a = nn.Parameter(
                torch.empty(heads, 32, rank, **factory)
            )
            self.query_prope_b = nn.Parameter(
                torch.zeros(heads, rank, low_dim // 4, **factory)
            )
            init_a(self.query_main_a)
            init_a(self.query_prope_a)
        else:
            for name in (
                "query_main_a", "query_main_b",
                "query_prope_a", "query_prope_b",
            ):
                self.register_parameter(name, None)
        # Main and PRoPE occupy the two stacked batch branches.  Give each
        # position representation its own D-low map while keeping the
        # attention width exactly unchanged.  Zero B is an exact identity.
        if "key" in self.targets:
            self.key_a = nn.Parameter(
                torch.empty(2, heads, low_dim, rank, **factory)
            )
            self.key_b = nn.Parameter(
                torch.zeros(2, heads, rank, low_dim, **factory)
            )
            init_a(self.key_a)
        else:
            self.register_parameter("key_a", None)
            self.register_parameter("key_b", None)
        if "value" in self.targets:
            self.value_a = nn.Parameter(
                torch.empty(2, heads, low_dim, rank, **factory)
            )
            self.value_b = nn.Parameter(
                torch.zeros(2, heads, rank, low_dim, **factory)
            )
            init_a(self.value_a)
        else:
            self.register_parameter("value_a", None)
            self.register_parameter("value_b", None)
        # Unlike the post-read K/V adapters above, these four maps can rotate
        # the row-space used to consume z512.  Their output is still exactly
        # D-low per head, and at export they fold into the existing
        # k/v_{main,prope}_read matrices with no extra runtime operator.
        if "latent" in self.targets:
            self.latent_k_a = nn.Parameter(
                torch.empty(2, heads, 512, rank, **factory)
            )
            self.latent_k_b = nn.Parameter(
                torch.zeros(2, heads, rank, low_dim, **factory)
            )
            self.latent_v_a = nn.Parameter(
                torch.empty(2, heads, 512, rank, **factory)
            )
            self.latent_v_b = nn.Parameter(
                torch.zeros(2, heads, rank, low_dim, **factory)
            )
            init_a(self.latent_k_a)
            init_a(self.latent_v_a)
        else:
            self.register_parameter("latent_k_a", None)
            self.register_parameter("latent_k_b", None)
            self.register_parameter("latent_v_a", None)
            self.register_parameter("latent_v_b", None)
        # C64 centers are stored in native per-head D128 coordinates.  Main
        # reads therefore use 128->Dlow maps, while PRoPE preserves its four
        # independent 32-D quartets and learns 32->Dlow/4.  All four maps fold
        # into the existing cache-write center projections at export.
        if "center" in self.targets:
            prope_dim = low_dim // 4
            if low_dim % 4:
                raise ValueError(
                    f"center-read requires low_dim divisible by 4, got {low_dim}"
                )
            self.center_k_main_a = nn.Parameter(
                torch.empty(heads, 128, rank, **factory)
            )
            self.center_k_main_b = nn.Parameter(
                torch.zeros(heads, rank, low_dim, **factory)
            )
            self.center_k_prope_a = nn.Parameter(
                torch.empty(heads, 32, rank, **factory)
            )
            self.center_k_prope_b = nn.Parameter(
                torch.zeros(heads, rank, prope_dim, **factory)
            )
            self.center_v_main_a = nn.Parameter(
                torch.empty(heads, 128, rank, **factory)
            )
            self.center_v_main_b = nn.Parameter(
                torch.zeros(heads, rank, low_dim, **factory)
            )
            self.center_v_prope_a = nn.Parameter(
                torch.empty(heads, 32, rank, **factory)
            )
            self.center_v_prope_b = nn.Parameter(
                torch.zeros(heads, rank, prope_dim, **factory)
            )
            for parameter in (
                self.center_k_main_a,
                self.center_k_prope_a,
                self.center_v_main_a,
                self.center_v_prope_a,
            ):
                init_a(parameter)
        else:
            for name in (
                "center_k_main_a", "center_k_main_b",
                "center_k_prope_a", "center_k_prope_b",
                "center_v_main_a", "center_v_main_b",
                "center_v_prope_a", "center_v_prope_b",
            ):
                self.register_parameter(name, None)
        # Structure-preserving Value readout.  Rather than post-multiplying
        # the reconstructed D128 output (which would couple PRoPE quartets),
        # learn additive deltas to the existing Main and quartet lift matrices.
        if "lift" in self.targets:
            if low_dim % 4 or output_dim % 4:
                raise ValueError(
                    "lift requires low_dim/output_dim divisible by 4, got "
                    f"{low_dim}/{output_dim}"
                )
            self.lift_main_a = nn.Parameter(
                torch.empty(heads, low_dim, rank, **factory)
            )
            self.lift_main_b = nn.Parameter(
                torch.zeros(heads, rank, output_dim, **factory)
            )
            self.lift_prope_a = nn.Parameter(
                torch.empty(heads, low_dim // 4, rank, **factory)
            )
            self.lift_prope_b = nn.Parameter(
                torch.zeros(heads, rank, output_dim // 4, **factory)
            )
            init_a(self.lift_main_a)
            init_a(self.lift_prope_a)
        else:
            for name in (
                "lift_main_a", "lift_main_b",
                "lift_prope_a", "lift_prope_b",
            ):
                self.register_parameter(name, None)
        if "output" in self.targets:
            self.output_a = nn.Parameter(
                torch.empty(heads, output_dim, rank, **factory)
            )
            self.output_b = nn.Parameter(
                torch.zeros(heads, rank, output_dim, **factory)
            )
            init_a(self.output_a)
        else:
            self.register_parameter("output_a", None)
            self.register_parameter("output_b", None)
        # Main and PRoPE use the two stacked batch branches.  A [2,H] bias is
        # shared across actual samples but not across branches or heads.
        if "lse" in self.targets:
            self.history_lse_bias = nn.Parameter(
                torch.zeros(2, heads, dtype=torch.float32, device=device)
            )
        else:
            self.register_parameter("history_lse_bias", None)

    def adapt_query(self, query: torch.Tensor) -> torch.Tensor:
        """Apply a per-head rank-r residual to ``[2B,H,N,Dlow]`` query."""

        if query.ndim != 4 or query.shape[1] != self.heads:
            raise ValueError(
                f"expected query [2B,{self.heads},N,{self.low_dim}], "
                f"got {tuple(query.shape)}"
            )
        if query.shape[-1] != self.low_dim:
            raise ValueError(
                f"history query width changed: expected {self.low_dim}, "
                f"got {query.shape[-1]}"
            )
        if self.query_a is None:
            return query
        delta = torch.einsum("bhnd,hdr->bhnr", query, self.query_a)
        delta = torch.einsum("bhnr,hrd->bhnd", delta, self.query_b)
        return query + delta * self.scale

    def query_read_delta(
        self, query_pre: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return foldable pre-position Main/PRoPE query-read residuals."""

        if query_pre.ndim != 4 or query_pre.shape[1] != self.heads:
            raise ValueError(
                f"expected pre-position query [B,{self.heads},N,128], "
                f"got {tuple(query_pre.shape)}"
            )
        if query_pre.shape[-1] != 128:
            raise ValueError(
                f"pre-position query width must be 128, got {query_pre.shape[-1]}"
            )
        if self.query_main_a is None:
            zero = query_pre.new_zeros(
                query_pre.shape[0], self.heads, query_pre.shape[2], self.low_dim
            )
            return zero, zero
        main = torch.einsum("bhni,hir->bhnr", query_pre, self.query_main_a)
        main = torch.einsum("bhnr,hrd->bhnd", main, self.query_main_b)
        quartets = query_pre.reshape(*query_pre.shape[:-1], 4, 32)
        prope = torch.einsum(
            "bhngi,hir->bhngr", quartets, self.query_prope_a
        )
        prope = torch.einsum(
            "bhngr,hrd->bhngd", prope, self.query_prope_b
        ).flatten(-2)
        return main * self.scale, prope * self.scale

    def lift_delta(
        self, main_low: torch.Tensor, prope_low: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exactly foldable Main/PRoPE D128 lift residuals."""

        expected = (self.heads, self.low_dim)
        for name, value in (("main", main_low), ("prope", prope_low)):
            if value.ndim != 4 or (value.shape[1], value.shape[-1]) != expected:
                raise ValueError(
                    f"expected {name} low output [B,{self.heads},N,{self.low_dim}], "
                    f"got {tuple(value.shape)}"
                )
        if self.lift_main_a is None:
            main_zero = main_low.new_zeros(*main_low.shape[:-1], self.output_dim)
            prope_zero = prope_low.new_zeros(*prope_low.shape[:-1], self.output_dim)
            return main_zero, prope_zero
        main = torch.einsum("bhnd,hdr->bhnr", main_low, self.lift_main_a)
        main = torch.einsum("bhnr,hro->bhno", main, self.lift_main_b)
        quartets = prope_low.reshape(*prope_low.shape[:-1], 4, self.low_dim // 4)
        prope = torch.einsum(
            "bhngd,hdr->bhngr", quartets, self.lift_prope_a
        )
        prope = torch.einsum(
            "bhngr,hro->bhngo", prope, self.lift_prope_b
        ).flatten(-2)
        return main * self.scale, prope * self.scale

    def _adapt_stacked_history(
        self,
        tensor: torch.Tensor,
        first: Optional[torch.Tensor],
        second: Optional[torch.Tensor],
        *,
        name: str,
    ) -> torch.Tensor:
        """Adapt stacked Main/PRoPE ``[2B,H,N,Dlow]`` features."""

        if tensor.ndim != 4 or tensor.shape[1] != self.heads:
            raise ValueError(
                f"expected history {name} [2B,{self.heads},N,{self.low_dim}], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[-1] != self.low_dim or tensor.shape[0] % 2:
            raise ValueError(
                f"history {name} must have even stacked batch and width "
                f"{self.low_dim}; got {tuple(tensor.shape)}"
            )
        if first is None:
            return tensor
        batch = tensor.shape[0] // 2
        stacked = tensor.reshape(
            2, batch, self.heads, tensor.shape[2], self.low_dim
        )
        delta = torch.einsum("zbhnd,zhdr->zbhnr", stacked, first)
        delta = torch.einsum("zbhnr,zhrd->zbhnd", delta, second)
        return (stacked + delta * self.scale).reshape_as(tensor)

    def adapt_history_key(self, key: torch.Tensor) -> torch.Tensor:
        """Adapt positioned low-D history K without changing D-low."""

        return self._adapt_stacked_history(
            key, self.key_a, self.key_b, name="key"
        )

    def adapt_history_value(self, value: torch.Tensor) -> torch.Tensor:
        """Adapt positioned low-D history V without changing D-low."""

        return self._adapt_stacked_history(
            value, self.value_a, self.value_b, name="value"
        )

    def latent_read_delta(
        self, latent: torch.Tensor, *, target: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read foldable Main/PRoPE residuals from ``latent[B,N,512]``.

        Position transforms intentionally remain outside this module.  The
        returned pair is pre-position ``[B,H,N,Dlow]`` and can therefore be
        added to the corresponding baseline read before Main RoPE/PRoPE.
        """

        if latent.ndim != 3 or latent.shape[-1] != 512:
            raise ValueError(
                f"expected formal latent [B,N,512], got {tuple(latent.shape)}"
            )
        if target == "key":
            first, second = self.latent_k_a, self.latent_k_b
        elif target == "value":
            first, second = self.latent_v_a, self.latent_v_b
        else:
            raise ValueError(f"latent read target must be key/value, got {target!r}")
        if first is None:
            zero = latent.new_zeros(
                latent.shape[0], self.heads, latent.shape[1], self.low_dim
            )
            return zero, zero
        delta = torch.einsum("bni,zhir->zbhnr", latent, first)
        delta = torch.einsum("zbhnr,zhrd->zbhnd", delta, second)
        delta = delta * self.scale
        return delta[0], delta[1]

    def center_read_delta(
        self,
        centers: torch.Tensor,
        ids: torch.Tensor,
        *,
        target: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read foldable Main/PRoPE deltas from formal C64 metadata.

        ``centers`` is the immutable stored BF16 table ``[B,C,2048]`` and
        ``ids`` selects one center per old-history token ``[B,N]``.  The
        returned pair is pre-position ``[B,H,N,Dlow]``.
        """

        if centers.ndim != 3 or centers.shape[-1] != self.heads * 128:
            raise ValueError(
                "expected formal centers "
                f"[B,C,{self.heads * 128}], got {tuple(centers.shape)}"
            )
        if ids.ndim != 2 or ids.shape[0] != centers.shape[0]:
            raise ValueError(
                f"expected center ids [B,N], got {tuple(ids.shape)}"
            )
        if target == "key":
            main_a, main_b = self.center_k_main_a, self.center_k_main_b
            prope_a, prope_b = self.center_k_prope_a, self.center_k_prope_b
        elif target == "value":
            main_a, main_b = self.center_v_main_a, self.center_v_main_b
            prope_a, prope_b = self.center_v_prope_a, self.center_v_prope_b
        else:
            raise ValueError(f"center read target must be key/value, got {target!r}")
        if main_a is None:
            zero = centers.new_zeros(
                centers.shape[0], self.heads, ids.shape[1], self.low_dim
            )
            return zero, zero

        native = centers.to(device=main_a.device, dtype=main_a.dtype).reshape(
            centers.shape[0], centers.shape[1], self.heads, 128
        ).permute(0, 2, 1, 3)
        main = torch.einsum("bhci,hir->bhcr", native, main_a)
        main = torch.einsum("bhcr,hrd->bhcd", main, main_b) * self.scale
        quartets = native.reshape(*native.shape[:-1], 4, 32)
        prope = torch.einsum("bhcgi,hir->bhcgr", quartets, prope_a)
        prope = torch.einsum("bhcgr,hrd->bhcgd", prope, prope_b)
        prope = prope.flatten(-2) * self.scale
        gather = ids.to(device=native.device, dtype=torch.long)
        gather = gather[:, None, :, None].expand(
            -1, self.heads, -1, self.low_dim
        )
        return torch.gather(main, 2, gather), torch.gather(prope, 2, gather)

    def adapt_lifted_output(self, output: torch.Tensor) -> torch.Tensor:
        """Apply a per-head rank-r residual to lifted ``[2B,H,N,128]``."""

        if output.ndim != 4 or output.shape[1] != self.heads:
            raise ValueError(
                f"expected history output [2B,{self.heads},N,{self.output_dim}], "
                f"got {tuple(output.shape)}"
            )
        if output.shape[-1] != self.output_dim:
            raise ValueError(
                f"lifted history width changed: expected {self.output_dim}, "
                f"got {output.shape[-1]}"
            )
        if self.output_a is None:
            return output
        delta = torch.einsum("bhnd,hdr->bhnr", output, self.output_a)
        delta = torch.einsum("bhnr,hrd->bhnd", delta, self.output_b)
        return output + delta * self.scale

    def adapt_history_lse(self, lse: torch.Tensor) -> torch.Tensor:
        """Bias only the old-history partition denominator, never native LSE."""

        if self.history_lse_bias is None:
            return lse
        if lse.ndim != 3 or lse.shape[1] != self.heads:
            raise ValueError(
                f"expected history LSE [2B,{self.heads},N], got {tuple(lse.shape)}"
            )
        if lse.shape[0] % 2:
            raise ValueError("stacked Main/PRoPE history LSE batch must be even")
        batch = lse.shape[0] // 2
        bias = self.history_lse_bias[:, None].expand(-1, batch, -1)
        bias = bias.reshape(2 * batch, self.heads, 1)
        return lse + bias.to(device=lse.device, dtype=lse.dtype)

    def fold_matrices(self) -> Dict[str, torch.Tensor]:
        """Return exact post-linear maps for offline deployment folding.

        ``query`` is composed into the pre-position low-query read. ``output``
        is composed after the existing lift.  Main can retain its dense lift;
        folding the arbitrary output map into the current structured PRoPE
        quartet lift requires exporting that composed lift as a dense matrix.
        The canary runtime keeps the low-rank form; this helper makes that
        deployment trade-off explicit rather than claiming a free structured
        PRoPE merge.
        """

        result = {}
        if self.query_a is not None:
            identity = torch.eye(
                self.low_dim, device=self.query_a.device, dtype=self.query_a.dtype
            )[None].expand(self.heads, -1, -1)
            result["query"] = identity + torch.einsum(
                "hdr,hrk->hdk", self.query_a, self.query_b
            ) * self.scale
        if self.output_a is not None:
            identity = torch.eye(
                self.output_dim,
                device=self.output_a.device,
                dtype=self.output_a.dtype,
            )[None].expand(self.heads, -1, -1)
            result["output"] = identity + torch.einsum(
                "hdr,hrk->hdk", self.output_a, self.output_b
            ) * self.scale
        for name, first, second in (
            ("query_main_read_delta", self.query_main_a, self.query_main_b),
            ("query_prope_read_delta", self.query_prope_a, self.query_prope_b),
            ("lift_main_delta", self.lift_main_a, self.lift_main_b),
            ("lift_prope_delta", self.lift_prope_a, self.lift_prope_b),
        ):
            if first is not None:
                # State matrices use [H,output,input] for reads and
                # [H,input,output] for lifts.  Keep those native layouts.
                product = torch.einsum("hir,hro->hio", first, second)
                if name.startswith("query_"):
                    product = product.transpose(-1, -2)
                result[name] = product * self.scale
        for name, first, second in (
            ("positioned_key", self.key_a, self.key_b),
            ("positioned_value", self.value_a, self.value_b),
        ):
            if first is not None:
                identity = torch.eye(
                    self.low_dim, device=first.device, dtype=first.dtype
                )[None, None].expand(2, self.heads, -1, -1)
                result[name] = identity + torch.einsum(
                    "zhdr,zhrk->zhdk", first, second
                ) * self.scale
        for name, first, second in (
            ("latent_key_read_delta", self.latent_k_a, self.latent_k_b),
            ("latent_value_read_delta", self.latent_v_a, self.latent_v_b),
        ):
            if first is not None:
                # [2,H,512,r] @ [2,H,r,D] -> deployable [2,H,D,512].
                result[name] = torch.einsum(
                    "zhir,zhrd->zhdi", first, second
                ) * self.scale
        for name, first, second in (
            ("center_k_main_delta", self.center_k_main_a, self.center_k_main_b),
            ("center_k_prope_delta", self.center_k_prope_a, self.center_k_prope_b),
            ("center_v_main_delta", self.center_v_main_a, self.center_v_main_b),
            ("center_v_prope_delta", self.center_v_prope_a, self.center_v_prope_b),
        ):
            if first is not None:
                result[name] = torch.einsum(
                    "hir,hrd->hdi", first, second
                ) * self.scale
        return result


class CodecNativeHistoryAdapterBank(nn.Module):
    """Owns the selected per-layer adapters without modifying the DiT."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict()
        self.enabled = True

    @staticmethod
    def _key(layer_idx: int) -> str:
        return f"layer_{int(layer_idx):02d}"

    def add_layer(
        self,
        layer_idx: int,
        low_dim: int,
        *,
        heads: int,
        output_dim: int,
        rank: int,
        alpha: float,
        targets: Iterable[str],
        device,
        dtype,
    ) -> CodecNativeHistoryLayerAdapter:
        layer_idx = int(layer_idx)
        if layer_idx not in _ALLOWED_LAYERS:
            raise ValueError(f"history adapter cannot own L{layer_idx}; allowed=L0-53")
        key = self._key(layer_idx)
        existing = self.layers[key] if key in self.layers else None
        if existing is not None:
            expected = (
                int(low_dim), int(heads), int(output_dim), int(rank),
                frozenset(targets),
            )
            actual = (
                existing.low_dim,
                existing.heads,
                existing.output_dim,
                existing.rank,
                existing.targets,
            )
            if actual != expected:
                raise RuntimeError(
                    f"history adapter L{layer_idx} shape changed: "
                    f"existing={actual} requested={expected}"
                )
            return existing
        module = CodecNativeHistoryLayerAdapter(
            low_dim,
            heads=heads,
            output_dim=output_dim,
            rank=rank,
            alpha=alpha,
            targets=targets,
            init_seed=(
                int(
                    os.environ.get(
                        "HY_CODEC_NATIVE_HISTORY_ADAPTER_INIT_SEED", "20260902"
                    )
                )
                + layer_idx * 1009
            ),
            device=device,
            dtype=dtype,
        )
        self.layers[key] = module
        return module

    def get(self, layer_idx: int) -> Optional[CodecNativeHistoryLayerAdapter]:
        if not self.enabled:
            return None
        key = self._key(layer_idx)
        return self.layers[key] if key in self.layers else None


_BANK = CodecNativeHistoryAdapterBank()
_OPTIMIZER: Optional[torch.optim.Optimizer] = None
_CHECKPOINT_PAYLOAD: Optional[dict] = None
_CHECKPOINT_PATH = ""
_OPTIMIZER_LOADED = False


def bank() -> CodecNativeHistoryAdapterBank:
    return _BANK


def _checkpoint_payload() -> Optional[dict]:
    global _CHECKPOINT_PAYLOAD, _CHECKPOINT_PATH
    path = os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_CKPT", "").strip()
    if not path:
        return None
    if path != _CHECKPOINT_PATH:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != _FORMAT:
            raise RuntimeError(
                f"unsupported history-adapter checkpoint format: {payload.get('format')!r}"
            )
        _CHECKPOINT_PAYLOAD = payload
        _CHECKPOINT_PATH = path
    return _CHECKPOINT_PAYLOAD


def _config_value(name: str, default, checkpoint_key: str):
    if name in os.environ:
        return os.environ[name]
    payload = _checkpoint_payload()
    if payload is not None:
        return payload.get("config", {}).get(checkpoint_key, default)
    return default


def initialize(
    layer_dims: Mapping[int, int],
    *,
    device,
    dtype: torch.dtype,
    heads: int = 16,
    output_dim: int = 128,
) -> CodecNativeHistoryAdapterBank:
    """Materialize all selected adapters and load an optional checkpoint."""

    if not enabled():
        return _BANK
    layers = selected_layers()
    missing = layers - {int(layer) for layer in layer_dims}
    if missing:
        raise RuntimeError(
            "history-adapter layers are absent from the active direct-codec "
            f"manifest: {sorted(missing)}"
        )
    rank = int(_config_value("HY_CODEC_NATIVE_HISTORY_ADAPTER_RANK", 8, "rank"))
    alpha = float(
        _config_value("HY_CODEC_NATIVE_HISTORY_ADAPTER_ALPHA", rank, "alpha")
    )
    targets = selected_targets()
    payload = _checkpoint_payload()
    checkpoint_layers = {} if payload is None else payload.get("layers", {})
    if payload is not None:
        checkpoint_config = payload.get("config", {})
        expected_layers = {int(layer) for layer in checkpoint_config.get("layers", ())}
        checkpoint_targets = set(checkpoint_config.get("targets", ()))
        mismatches = []
        if expected_layers != layers:
            mismatches.append(
                f"layers checkpoint={sorted(expected_layers)} runtime={sorted(layers)}"
            )
        if checkpoint_targets != targets:
            mismatches.append(
                f"targets checkpoint={sorted(checkpoint_targets)} runtime={sorted(targets)}"
            )
        if int(checkpoint_config.get("rank", -1)) != rank:
            mismatches.append(
                f"rank checkpoint={checkpoint_config.get('rank')} runtime={rank}"
            )
        if not math.isclose(
            float(checkpoint_config.get("alpha", float("nan"))),
            alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            mismatches.append(
                f"alpha checkpoint={checkpoint_config.get('alpha')} runtime={alpha}"
            )
        if int(checkpoint_config.get("heads", -1)) != int(heads):
            mismatches.append(
                f"heads checkpoint={checkpoint_config.get('heads')} runtime={heads}"
            )
        if int(checkpoint_config.get("output_dim", -1)) != int(output_dim):
            mismatches.append(
                "output_dim checkpoint="
                f"{checkpoint_config.get('output_dim')} runtime={output_dim}"
            )
        if set(checkpoint_layers) != {str(layer) for layer in layers}:
            mismatches.append(
                "layer payload keys checkpoint="
                f"{sorted(checkpoint_layers)} runtime={sorted(str(layer) for layer in layers)}"
            )
        if mismatches:
            raise RuntimeError(
                "history-adapter checkpoint/runtime contract mismatch: "
                + "; ".join(mismatches)
            )

    for layer_idx in sorted(layers):
        module = _BANK.add_layer(
            layer_idx,
            int(layer_dims[layer_idx]),
            heads=heads,
            output_dim=output_dim,
            rank=rank,
            alpha=alpha,
            targets=targets,
            device=device,
            dtype=dtype,
        )
        checkpoint_entry = checkpoint_layers.get(str(layer_idx))
        loaded = getattr(module, "_checkpoint_loaded", "")
        if checkpoint_entry is not None and loaded != _CHECKPOINT_PATH:
            checkpoint_dim = int(checkpoint_entry.get("low_dim", -1))
            if checkpoint_dim != module.low_dim:
                raise RuntimeError(
                    f"history-adapter L{layer_idx} low_dim mismatch: "
                    f"checkpoint={checkpoint_dim} runtime={module.low_dim}"
                )
            module.load_state_dict(checkpoint_entry["state_dict"], strict=True)
            module._checkpoint_loaded = _CHECKPOINT_PATH
        elif payload is not None and checkpoint_entry is None:
            raise RuntimeError(f"history-adapter checkpoint is missing L{layer_idx}")
    if training_enabled():
        train_layers = selected_train_layers()
        for layer_idx in sorted(layers):
            module = _BANK.get(layer_idx)
            if module is None:
                raise RuntimeError(f"history adapter L{layer_idx} was not initialized")
            module.requires_grad_(layer_idx in train_layers)
    return _BANK


def get_layer(layer_idx: int) -> Optional[CodecNativeHistoryLayerAdapter]:
    if not enabled() or int(layer_idx) not in selected_layers():
        return None
    if not _BANK.enabled:
        return None
    module = _BANK.get(int(layer_idx))
    if module is None:
        raise RuntimeError(
            f"history adapter L{int(layer_idx)} was not initialized before attention"
        )
    return module


def optimizer() -> torch.optim.Optimizer:
    global _OPTIMIZER, _OPTIMIZER_LOADED
    if not training_enabled():
        raise RuntimeError(
            "history-adapter optimizer requires "
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_TRAIN=1"
        )
    params = [parameter for parameter in _BANK.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError("history-adapter bank has no initialized parameters")
    if _OPTIMIZER is None:
        shared_lr = os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_LR")
        content_lr = float(
            os.environ.get(
                "HY_CODEC_NATIVE_HISTORY_ADAPTER_CONTENT_LR",
                shared_lr if shared_lr is not None else "1e-5",
            )
        )
        lse_lr = float(
            os.environ.get(
                "HY_CODEC_NATIVE_HISTORY_ADAPTER_LSE_LR",
                shared_lr if shared_lr is not None else "1e-4",
            )
        )
        content_params = []
        lse_params = []
        train_layers = selected_train_layers()
        for layer_idx in sorted(train_layers):
            module = _BANK.get(layer_idx)
            if module is None:
                raise RuntimeError(
                    f"history-adapter optimizer is missing train layer L{layer_idx}"
                )
            for name in (
                "query_a", "query_b", "key_a", "key_b", "value_a", "value_b",
                "query_main_a", "query_main_b",
                "query_prope_a", "query_prope_b",
                "latent_k_a", "latent_k_b", "latent_v_a", "latent_v_b",
                "center_k_main_a", "center_k_main_b",
                "center_k_prope_a", "center_k_prope_b",
                "center_v_main_a", "center_v_main_b",
                "center_v_prope_a", "center_v_prope_b",
                "lift_main_a", "lift_main_b",
                "lift_prope_a", "lift_prope_b",
                "output_a", "output_b",
            ):
                parameter = getattr(module, name)
                if parameter is not None and parameter.requires_grad:
                    content_params.append(parameter)
            if (
                module.history_lse_bias is not None
                and module.history_lse_bias.requires_grad
            ):
                lse_params.append(module.history_lse_bias)
        groups = []
        if content_params:
            groups.append(
                {"params": content_params, "lr": content_lr, "name": "content"}
            )
        if lse_params:
            groups.append({"params": lse_params, "lr": lse_lr, "name": "lse"})
        _OPTIMIZER = torch.optim.AdamW(groups, weight_decay=0.0)
    if not _OPTIMIZER_LOADED:
        payload = _checkpoint_payload()
        if payload is not None and payload.get("optimizer") is not None and _flag(
            "HY_CODEC_NATIVE_HISTORY_ADAPTER_RESUME_OPTIMIZER", "1"
        ):
            saved_groups = payload.get("optimizer_groups")
            runtime_names = [group.get("name") for group in _OPTIMIZER.param_groups]
            if saved_groups is not None and [
                group.get("name") for group in saved_groups
            ] != runtime_names:
                raise RuntimeError(
                    "history-adapter optimizer group mismatch: "
                    f"checkpoint={saved_groups} runtime={runtime_names}"
                )
            _OPTIMIZER.load_state_dict(payload["optimizer"])
        _OPTIMIZER_LOADED = True
    return _OPTIMIZER


def zero_grad() -> None:
    if training_enabled():
        optimizer().zero_grad(set_to_none=True)


def step(loss: torch.Tensor, grad_clip: Optional[float] = None) -> Dict[str, float]:
    """Backpropagate and update only the history-branch adapter."""

    opt = optimizer()
    loss.backward()
    params = [
        parameter
        for group in opt.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    grad_sq = sum(
        float(parameter.grad.detach().float().pow(2).sum().cpu())
        for parameter in params
    )
    grad_norm = math.sqrt(grad_sq) if grad_sq else 0.0
    if grad_clip is None:
        grad_clip = float(
            os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_GRAD_CLIP", "1.0")
        )
    if params and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
    opt.step()
    weight_sq = sum(
        float(parameter.detach().float().pow(2).sum().cpu())
        for parameter in _BANK.parameters()
    )
    residual_sq = 0.0
    lse_sq = 0.0
    for module in _BANK.layers.values():
        for parameter in (
            module.query_b, module.key_b, module.value_b,
            module.query_main_b, module.query_prope_b,
            module.latent_k_b, module.latent_v_b,
            module.lift_main_b, module.lift_prope_b, module.output_b,
        ):
            if parameter is not None:
                residual_sq += float(
                    parameter.detach().float().pow(2).sum().cpu()
                )
        for parameter in (
            module.center_k_main_b, module.center_k_prope_b,
            module.center_v_main_b, module.center_v_prope_b,
        ):
            if parameter is not None:
                residual_sq += float(
                    parameter.detach().float().pow(2).sum().cpu()
                )
        if module.history_lse_bias is not None:
            lse_sq += float(
                module.history_lse_bias.detach().float().pow(2).sum().cpu()
            )
    return {
        "num_optimizers": 1.0,
        "max_grad_norm": grad_norm,
        "adapter_grad_norm": grad_norm,
        "adapter_weight_norm": math.sqrt(weight_sq) if weight_sq else 0.0,
        "adapter_residual_b_norm": math.sqrt(residual_sq) if residual_sq else 0.0,
        "adapter_lse_bias_norm": math.sqrt(lse_sq) if lse_sq else 0.0,
    }


def save(output_root: str) -> str:
    """Save adapter/optimizer state without touching codec or DiT checkpoints."""

    path = Path(output_root) / "checkpoint_history_adapter.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    modules = {
        str(layer_idx): {
            "low_dim": module.low_dim,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in module.state_dict().items()
            },
        }
        for layer_idx in sorted(selected_layers())
        if (module := _BANK.get(layer_idx)) is not None
    }
    if set(modules) != {str(layer) for layer in selected_layers()}:
        raise RuntimeError("refusing to save an incompletely initialized adapter bank")
    first = next(iter(_BANK.layers.values()))
    payload = {
        "format": _FORMAT,
        "config": {
            "rank": first.rank,
            "alpha": first.alpha,
            "targets": sorted(first.targets),
            "layers": sorted(selected_layers()),
            "heads": first.heads,
            "output_dim": first.output_dim,
        },
        "layers": modules,
        "optimizer": None if _OPTIMIZER is None else _OPTIMIZER.state_dict(),
        "optimizer_groups": (
            None
            if _OPTIMIZER is None
            else [
                {"name": group.get("name"), "lr": float(group["lr"])}
                for group in _OPTIMIZER.param_groups
            ]
        ),
    }
    torch.save(payload, path)
    snapshot_every = int(
        os.environ.get("HY_CODEC_NATIVE_HISTORY_ADAPTER_SNAPSHOT_EVERY", "0")
    )
    if snapshot_every > 0 and _OPTIMIZER is not None:
        steps = []
        for state in _OPTIMIZER.state.values():
            step_value = state.get("step")
            if isinstance(step_value, torch.Tensor):
                step_value = step_value.item()
            if step_value is not None:
                steps.append(int(step_value))
        if steps and min(steps) == max(steps) and steps[0] % snapshot_every == 0:
            snapshot = path.with_name(
                f"checkpoint_history_adapter_step{steps[0]:04d}.pt"
            )
            if not snapshot.exists():
                torch.save(payload, snapshot)
    return str(path)


@contextmanager
def temporarily_disabled():
    previous = _BANK.enabled
    _BANK.enabled = False
    try:
        yield
    finally:
        _BANK.enabled = previous


def reset_for_tests() -> None:
    """Clear process-global state; intended only for isolated unit tests."""

    global _BANK, _OPTIMIZER, _CHECKPOINT_PAYLOAD, _CHECKPOINT_PATH
    global _OPTIMIZER_LOADED
    _BANK = CodecNativeHistoryAdapterBank()
    _OPTIMIZER = None
    _CHECKPOINT_PAYLOAD = None
    _CHECKPOINT_PATH = ""
    _OPTIMIZER_LOADED = False
