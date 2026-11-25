# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Paper novel modules: DBConv (with selectable modulation), CSCA attention, ProCSCA framework.

This module consolidates every custom component proposed in the paper:

    - ParaNorm     : dynamic parametric normalization with selectable modulation modes (P0-4 / D0-D4)
    - DBConv       : dual-branch convolution with selectable structures (P0-2 / B0-B4 + baselines)
    - SDConv       : depthwise-separable dual-branch convolution (ablation baseline)
    - DPConv       : depthwise-separable single-path convolution (ablation baseline)
    - ProCSCA      : CSP-style progressive attention framework (paper Sec. ProCSCA)
    - CSCABlock    : CSCA attention unit + FFN post-refinement (paper Sec. CSCA / FFN)
    - CSCAUnit     : channel-spatial-coordinate attention unit (paper Sec. CSCA)
    - CSCA         : YAML-facing ablation module (direct stacking of CSCABlocks)
    - ChannelAttention / SpatialAttention / CoordinateAttention : attention branches of CSCAUnit

Engineering notes:
    Every selectable module (ParaNorm, DBConv) resolves its configuration ONCE in
    __init__ -- building the requested sub-modules and binding the corresponding
    forward implementation (``self.forward = self._forward_xxx``). The runtime
    forward pass therefore contains ZERO configuration branching; dispatching
    happens exclusively at construction time.

    All user-facing option names accept short ablation aliases (D0-D4 / B0-B4),
    which are canonicalized through :func:`_canonical_choice` with explicit
    validation errors.
"""

import copy
import math
import re
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, autopad

__all__ = (
    'ParaNorm',
    'DBConv',
    'SDConv',
    'DPConv',
    'ProCSCA',
    'CSCABlock',
    'CSCAUnit',
    'CSCA',
    'ChannelAttention',
    'SpatialAttention',
    'CoordinateAttention',
)


def _deepcopy_with_rebind(self, memo):
    """Deep-copy the module and re-bind its compiled forward to the copy.

    ultralytics validation deep-copies the model and converts it to half
    precision; instance-level ``forward`` callables (closures/bound methods)
    keep referencing the ORIGINAL modules after a naive deepcopy, which breaks
    the half-precision validation pass. Re-binding the forward to the copied
    module fixes this.
    """
    import copy as _copy
    result = self.__new__(type(self))
    memo[id(self)] = result
    for k, v in self.__dict__.items():
        object.__setattr__(result, k, _copy.deepcopy(v, memo))
    result._rebind_forward()
    return result

def _canonical_choice(name: str, aliases: Dict[str, str], valid: Tuple[str, ...], kind: str) -> str:
    """Resolve a user-facing option name (or its short alias) to a canonical key.

    Args:
        name: user-provided option name, e.g. ``'D1'`` or ``'bn'``.
        aliases: mapping from lower-cased short aliases to canonical keys.
        valid: canonical keys accepted after alias resolution.
        kind: human-readable option description used in error messages.

    Returns:
        The canonical key corresponding to ``name``.

    Raises:
        TypeError: if ``name`` is not a string.
        ValueError: if ``name`` cannot be resolved to a supported option.
    """
    if not isinstance(name, str):
        raise TypeError(f'{kind} must be a str, got {type(name).__name__}: {name!r}')
    key = name.strip().lower()
    canonical = aliases.get(key, key)
    if canonical not in valid:
        supported = ', '.join(f"'{v}'" for v in sorted(valid))
        raise ValueError(f'Unsupported {kind}: {name!r}. Expected one of [{supported}] (short aliases allowed).')
    return canonical


class ParaNorm(nn.Module):
    """Dynamic Parametric Normalization (ParaNorm) with selectable modulation modes.

    Modulation modes (P0-4 ablation, fixed DBConv and placement):
        'identity'       (D0): Identity                  -> x + SiLU              (no norm/modulation)
        'bn'             (D1): BatchNorm2d               -> BN + SiLU             (standard baseline)
        'bn_tanh'        (D2): BatchNorm2d + Tanh        -> BN + Tanh + SiLU      (effect of tanh)
        'learnable_tanh' (D3): gamma * tanh(x) + beta    -> learnable scale/offset + SiLU
        'paranorm'       (D4): gamma * tanh(alpha*x)+beta -> full ParaNorm + SiLU (default, current impl)

    The concrete forward path is bound in __init__ (self.forward = self._forward_xxx),
    so the runtime forward call has zero branching overhead.
    """

    #: Short aliases accepted for ``mode`` (P0-4 / D0-D4 ablation IDs).
    _MODE_ALIASES: Dict[str, str] = {
        'd0': 'identity',
        'd1': 'bn',
        'd2': 'bn_tanh',
        'd3': 'learnable_tanh',
        'd4': 'paranorm',
    }

    #: Canonical modulation modes accepted after alias resolution.
    _SUPPORTED_MODES: Tuple[str, ...] = ('identity', 'bn', 'bn_tanh', 'learnable_tanh', 'paranorm')

    def __init__(self, dim: int, mode: str = 'paranorm', init_alpha: float = 0.5) -> None:
        """Initialize the requested modulation branch and bind its forward path.

        Args:
            dim: number of channels the modulation operates on.
            mode: modulation mode; one of the modes listed in the class docstring
                (short aliases 'D0'-'D4' are accepted).
            init_alpha: initial value of the learnable tanh input scale (D4 only).
        """
        super().__init__()
        if int(dim) <= 0:
            raise ValueError(f'ParaNorm expects a positive channel dimension, got dim={dim!r}')
        if float(init_alpha) <= 0.0:
            raise ValueError(f'ParaNorm expects a positive init_alpha, got init_alpha={init_alpha!r}')

        self.dim: int = int(dim)
        self.init_alpha: float = float(init_alpha)
        self.mode: str = _canonical_choice(mode, self._MODE_ALIASES, self._SUPPORTED_MODES, 'ParaNorm modulation mode')

        # Stage 1/2 -- build the requested modulation branch (configuration-time only).
        builders: Dict[str, Callable[[], None]] = {
            'identity': self._build_identity,
            'bn': self._build_bn,
            'bn_tanh': self._build_bn_tanh,
            'learnable_tanh': self._build_learnable_tanh,
            'paranorm': self._build_paranorm,
        }
        builders[self.mode]()

        # Stage 2/2 -- bind the concrete forward implementation so the runtime
        # forward pass contains zero configuration branching.
        binders: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            'identity': self._forward_mod,
            'bn': self._forward_mod,
            'bn_tanh': self._forward_mod,
            'learnable_tanh': self._forward_learnable_tanh,
            'paranorm': self._forward_paranorm,
        }
        self.forward = binders[self.mode]

    # ------------------------------------------------------------------ #
    # Configuration-time builders (one per modulation mode)               #
    # ------------------------------------------------------------------ #
    def _build_identity(self) -> None:
        """D0: no normalization/modulation; the module degenerates to x + SiLU."""
        self.mod = nn.Identity()

    def _build_bn(self) -> None:
        """D1: standard BatchNorm baseline."""
        self.mod = nn.BatchNorm2d(self.dim)

    def _build_bn_tanh(self) -> None:
        """D2: BatchNorm followed by a fixed tanh saturation (isolates tanh)."""
        self.mod = nn.Sequential(nn.BatchNorm2d(self.dim), nn.Tanh())

    def _build_learnable_tanh(self) -> None:
        """D3: tanh with learnable channel-wise scale/offset (alpha fixed to 1)."""
        self.gamma = nn.Parameter(torch.ones(self.dim))
        self.beta = nn.Parameter(torch.zeros(self.dim))

    def _build_paranorm(self) -> None:
        """D4: full ParaNorm, gamma * tanh(alpha * x) + beta (paper design)."""
        self.alpha = nn.Parameter(torch.ones(1) * self.init_alpha)
        self.gamma = nn.Parameter(torch.ones(self.dim))
        self.beta = nn.Parameter(torch.zeros(self.dim))

    # ------------------------------------------------------------------ #
    # Bound forward implementations (one per modulation mode)             #
    # ------------------------------------------------------------------ #
    def _forward_mod(self, x: torch.Tensor) -> torch.Tensor:
        """D0 / D1 / D2: delegate to the pre-built modulation sub-module.

        bn_tanh 返回 clone: Sequential(BN, Tanh) 的输出直接作为返回值时,
        下游的 in-place 操作会破坏 TanhBackward 保存的张量 (autograd version
        冲突), clone 解耦两者 (D4 的 tanh 输出被 gamma/beta 立即消费, 无此问题)。
        """
        out = self.mod(x)
        return out.clone() if self.mode == 'bn_tanh' else out

    def _forward_learnable_tanh(self, x: torch.Tensor) -> torch.Tensor:
        """D3: gamma * tanh(x) + beta with channel-wise affine calibration."""
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.gamma * torch.tanh(x) + self.beta
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)

    def _forward_paranorm(self, x: torch.Tensor) -> torch.Tensor:
        """D4: gamma * tanh(alpha * x) + beta (full ParaNorm, paper design)."""
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.gamma * torch.tanh(self.alpha * x) + self.beta
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)

    def _rebind_forward(self) -> None:
        """Re-bind the specialized forward after a deepcopy."""
        binders = {
            'identity': self._forward_mod,
            'bn': self._forward_mod,
            'bn_tanh': self._forward_mod,
            'learnable_tanh': self._forward_learnable_tanh,
            'paranorm': self._forward_paranorm,
        }
        self.forward = binders[self.mode]

    __deepcopy__ = _deepcopy_with_rebind

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        return f'dim={self.dim}, mode={self.mode}'


class DBConv(nn.Module):
    """Dual-Branch Convolution with selectable structures (P0-2 ablation, B0-B4).

    Structures:
        'conv3'  (B0): single standard 3x3 conv (BN + act)        -> single standard conv
        'conv7'  (B1): single standard 7x7 conv (BN + act)        -> large-kernel standard conv
        'dw'     (B2): DW 7x7 + PW 1x1 (DPConv-style)             -> depthwise-separable branch only
        'dual'   (B3): dual-branch topology, ParaNorm disabled  -> dual branch, no ParaNorm
        'dbconv' (B4): dual-branch + ParaNorm(norm_mode)         -> full DBConv (default)
        'conv'        : single standard conv, kernel = k
        'sdconv'      : SDConv-style dual branch (BN)
        'dpconv'      : DPConv-style DW k + PW 1x1

    The concrete forward path is bound in __init__ (self.forward = self._forward_xxx),
    so the runtime forward call has zero branching overhead.
    """

    default_act: nn.Module = nn.SiLU()  # default activation

    #: Short aliases accepted for ``structure`` (P0-2 / B0-B4 ablation IDs + baselines).
    _STRUCTURE_ALIASES: Dict[str, str] = {
        'b0': 'conv3', 'conv3': 'conv3',
        'b1': 'conv7', 'conv7': 'conv7',
        'b2': 'dw', 'dw': 'dw',
        'b3': 'dual', 'dual': 'dual',
        'b4': 'dbconv', 'dbconv': 'dbconv',
        'conv': 'conv',
        'sdconv': 'sdconv',
        'dpconv': 'dpconv',
    }

    #: Canonical structures accepted after alias resolution.
    _SUPPORTED_STRUCTURES: Tuple[str, ...] = ('conv', 'dpconv', 'sdconv', 'dbconv', 'conv3', 'conv7', 'dw', 'dual')

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        norm_mode: str = 'paranorm',
        structure: str = 'dbconv',
        p = None,
        g: int = 1,
        d: int = 1,
        act = True,
        split_ratio: float = 0.5,
    ) -> None:
        """Initialize the requested structure branch and bind its forward path.

        Args:
            c1: input channels.
            c2: output channels.
            k: kernel size of the context/depthwise branch (B0/B1/B2 override it).
            s: stride.
            norm_mode: modulation mode of the ParaNorm branches (see ParaNorm; B3 forces 'identity').
            structure: structure mode; one of the modes listed in the class docstring
                (short aliases 'B0'-'B4' are accepted).
            p: explicit padding (None -> same padding).
            g: grouping of the standard branch.
            d: dilation.
            act: activation flag or module (True -> SiLU).
            split_ratio: channel split ratio of the dual-branch topologies.
        """
        super().__init__()
        if int(c2) <= 0:
            raise ValueError(f'DBConv expects a positive output channel count, got c2={c2!r}')
        if int(s) <= 0:
            raise ValueError(f'DBConv expects a positive stride, got s={s!r}')
        if not 0.0 < float(split_ratio) < 1.0:
            raise ValueError(f'DBConv split_ratio must be in (0, 1), got split_ratio={split_ratio!r}')

        self.split_ratio: float = float(split_ratio)
        self.c1h: int = int(c2 * self.split_ratio)
        self.c2h: int = c2 - self.c1h
        if min(self.c1h, self.c2h) <= 0:
            raise ValueError(
                f'DBConv split_ratio={self.split_ratio} produces empty branches for c2={c2} '
                f'(c1h={self.c1h}, c2h={self.c2h})'
            )

        # Stage 1/3 -- canonicalize the structure and apply B0-B3 kernel/norm overrides.
        self.norm_mode: str = norm_mode
        st = _canonical_choice(structure, self._STRUCTURE_ALIASES, self._SUPPORTED_STRUCTURES, 'DBConv structure')
        if st == 'conv3':      # B0: single standard 3x3 conv
            st, k = 'conv', 3
        elif st == 'conv7':    # B1: single standard 7x7 conv
            st, k = 'conv', 7
        elif st == 'dw':       # B2: DW 7x7 + PW 1x1
            st, k = 'dpconv', 7
        elif st == 'dual':     # B3: dual-branch topology without ParaNorm
            st = 'dbconv'
            self.norm_mode = 'identity'
        self.structure: str = st
        self.act_layer: nn.Module = self._resolve_activation(act)

        # Stage 2/3 -- build the requested structure branch (configuration-time only).
        builders: Dict[str, Callable[[int, int, int, int, object, int, int, object], None]] = {
            'conv': self._build_conv,
            'dpconv': self._build_dpconv,
            'sdconv': self._build_sdconv,
            'dbconv': self._build_dbconv,
        }
        builders[st](c1, c2, k, s, p, g, d, self.act_layer)

        # Stage 3/3 -- bind the concrete forward implementation so the runtime
        # forward call has zero branching overhead.
        binders: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            'conv': self._forward_conv,
            'dpconv': self._forward_dpconv,
            'sdconv': self._forward_sdconv,
            'dbconv': self._forward_dbconv,
        }
        self.forward = binders[st]

    # ------------------------------------------------------------------ #
    # Configuration-time helpers                                          #
    # ------------------------------------------------------------------ #
    def _resolve_activation(self, act) -> nn.Module:
        """Resolve the ``act`` argument into a concrete activation module (SiLU default)."""
        if act is True:
            return self.default_act
        if isinstance(act, nn.Module):
            return act
        return nn.Identity()

    def _build_conv(self, c1: int, c2: int, k: int, s: int, p, g: int, d: int, act) -> None:
        """B0 / B1 / 'conv': single standard convolution (BN + act)."""
        self.conv = Conv(c1, c2, k, s, p=autopad(k, p, d), g=g, d=d, act=act)

    def _build_dpconv(self, c1: int, c2: int, k: int, s: int, p, g: int, d: int, act) -> None:
        """B2 / 'dpconv': depthwise-separable branch only (DW k + PW 1x1)."""
        self.cv1 = Conv(c1, c2, k, s, g=min(c1, c2), act=act)
        self.cv2 = Conv(c2, c2, 1, 1, act=False)

    def _build_sdconv(self, c1: int, c2: int, k: int, s: int, p, g: int, d: int, act) -> None:
        """'sdconv': SDConv-style dual branch with plain BatchNorm."""
        self.cv1 = Conv(c1, self.c1h, 3, s, act=act)
        self.cv2 = Conv(c1, self.c2h, k, s, g=min(c1, self.c2h), act=act)

    def _build_dbconv(self, c1: int, c2: int, k: int, s: int, p, g: int, d: int, act_layer: nn.Module) -> None:
        """B4 / 'dbconv': dual-branch topology with ParaNorm modulation."""
        self.sc = nn.Conv2d(c1, self.c1h, 3, s, autopad(3, p, d), groups=g, dilation=d, bias=False)
        self.bn_sc = ParaNorm(self.c1h, mode=self.norm_mode)
        self.act_sc = act_layer

        self.dsc = nn.Conv2d(c1, self.c2h, k, s, autopad(k, p, d), groups=min(c1, self.c2h), dilation=d, bias=False)
        self.bn_dsc = ParaNorm(self.c2h, mode=self.norm_mode)
        self.act_dsc = act_layer

        self.pconv = nn.Conv2d(self.c2h, self.c2h, 1, 1, bias=False)
        self.bn_pconv = ParaNorm(self.c2h, mode=self.norm_mode)
        self.act_pconv = act_layer

    # ------------------------------------------------------------------ #
    # Bound forward implementations (one per structure)                   #
    # ------------------------------------------------------------------ #
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """B0 / B1 / 'conv': single standard convolution."""
        return self.conv(x)

    def _forward_dpconv(self, x: torch.Tensor) -> torch.Tensor:
        """B2 / 'dpconv': depthwise-separable single path."""
        return self.cv2(self.cv1(x))

    def _forward_sdconv(self, x: torch.Tensor) -> torch.Tensor:
        """'sdconv': dual-branch with plain BatchNorm, channel-wise concat."""
        return torch.cat((self.cv1(x), self.cv2(x)), 1)

    def _forward_dbconv(self, x: torch.Tensor) -> torch.Tensor:
        """B4: dual-branch with ParaNorm modulation, channel-wise concat."""
        x1 = self.act_sc(self.bn_sc(self.sc(x)))

        x2 = self.act_dsc(self.bn_dsc(self.dsc(x)))
        x2 = self.act_pconv(self.bn_pconv(self.pconv(x2))) + x2

        return torch.cat((x1, x2), 1)

    def _rebind_forward(self) -> None:
        """Re-bind the specialized forward after a deepcopy."""
        binders = {
            'conv': self._forward_conv,
            'dpconv': self._forward_dpconv,
            'sdconv': self._forward_sdconv,
            'dbconv': self._forward_dbconv,
        }
        self.forward = binders[self.structure]

    __deepcopy__ = _deepcopy_with_rebind

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        base = f'split_ratio={self.split_ratio}, c1h={self.c1h}, c2h={self.c2h}, structure={self.structure}'
        if self.structure == 'dbconv':
            base += f', norm={self.bn_sc.mode}'
        return base


class SDConv(nn.Module):
    """SDConv-style split dual-branch convolution (ablation baseline).

    Topology (identical to the original SDConv):
        - branch 1: standard 3x3 conv -> c1h = int(c2 * split_ratio) channels (BN + SiLU)
        - branch 2: depthwise-separable k x k conv, groups=min(c1, c2h) -> c2h channels (BN + SiLU)
        - output: channel-wise concatenation of both branches (c2 channels)

    Role in ablations:
        BN-based dual-branch baseline of DBConv (B4). DBConv(structure='sdconv')
        reproduces this topology, isolating the contribution of ParaNorm
        against plain BatchNorm.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1, act=True, split_ratio: float = 0.5) -> None:
        """Initialize the split dual-branch topology.

        Args:
            c1: input channels.
            c2: output channels.
            k: kernel size of the depthwise-separable branch.
            s: stride.
            p: explicit padding (None -> same padding).
            g: grouping (kept for signature compatibility; unused here).
            d: dilation (kept for signature compatibility; unused here).
            act: activation flag or module (True -> SiLU).
            split_ratio: channel split ratio between the two branches.
        """
        super().__init__()
        if not 0.0 < float(split_ratio) < 1.0:
            raise ValueError(f'SDConv split_ratio must be in (0, 1), got split_ratio={split_ratio!r}')

        self.split_ratio: float = float(split_ratio)
        self.c1 = int(c2 * self.split_ratio)
        self.c2 = c2 - self.c1

        self.cv1 = Conv(c1, self.c1, 3, s, act=act)
        self.cv2 = Conv(c1, self.c2, k, s, g=min(c1, self.c2), act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the outputs of the two branches along the channel dimension."""
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        return torch.cat((x1, x2), 1)


class DPConv(nn.Module):
    """DPConv-style depthwise-separable convolution (ablation baseline).

    Topology (identical to the original DPConv):
        - stage 1: depthwise-separable k x k conv, groups=min(c1, c2) (BN + SiLU)
        - stage 2: pointwise 1x1 conv (BN, no activation)
        - output: single path, c2 channels

    Role in ablations:
        Depthwise-separable baseline without the dual-branch split.
        DBConv(structure='dpconv') / DBConv(structure='B2') use this topology;
        B2 forces k=7 per the P0-2 ablation table.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g: int = 1, d: int = 1, act=True, split_ratio: float = 0.5) -> None:
        """Initialize the depthwise-separable single path.

        Args:
            c1: input channels.
            c2: output channels.
            k: kernel size of the depthwise-separable stage.
            s: stride.
            p: explicit padding (None -> same padding).
            g: grouping (kept for signature compatibility; unused here).
            d: dilation (kept for signature compatibility; unused here).
            act: activation flag or module for the depthwise-separable stage.
            split_ratio: kept for signature compatibility; unused here.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, k=k, s=s, g=min(c1, c2), act=act)
        self.cv2 = Conv(c2, c2, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the depthwise-separable stage followed by the pointwise stage."""
        return self.cv2(self.cv1(x))


class ProCSCA(nn.Module):
    """Progressive Channel-Spatial-Coordinate Attention (ProCSCA) framework.

    CSP-style progressive refinement (paper Sec. ProCSCA):
        1. cv1 (1x1 conv) projects the input and splits it into two parallel streams;
        2. the second stream is progressively refined by `n` cascaded CSCABlocks;
        3. the first stream preserves low-level spatial details as a shortcut pathway;
        4. all intermediate features are concatenated and fused by cv2 (1x1 conv).

    This design promotes diversified gradient flow and mitigates feature
    redundancy compared with direct stacking of attention blocks.

    Args:
        c1: input channels.
        c2: output channels.
        n: number of cascaded CSCABlocks (depth gain applied by parse_model).
            topology: attention topology expression passed to each CSCABlock (see CSCAUnit).
        shortcut: residual connections inside each CSCABlock.
        residual: add the '+ x' residual to the trunk branch (ablation switch).
        use_scale: apply the learnable residual scale (lambda; ablation switch).
        pe_on_input: feed the raw input x to the pe branch (=co+sp variant).
        e: hidden channel expansion ratio (hidden = int(c2 * e)).
        residual: add the '+ x' residual to the trunk branch (ablation switch).
        use_scale: apply the learnable residual scale (lambda; ablation switch).
        pe_on_input: feed the raw input x to the pe branch (=co+sp variant).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, topology='chh-co+sp', shortcut: bool = True, g: int = 1, e: float = 0.5, residual: bool = True, use_scale: bool = True, pe_on_input: bool = False, use_ffn: bool = True, use_ffn_scale: bool = True, use_proj: bool = True, use_pe: bool = True) -> None:
        """Initialize the CSP-style progressive attention framework.

        Args:
            c1: input channels.
            c2: output channels.
            n: number of cascaded CSCABlocks (already depth-gained by parse_model).            topology: attention topology expression passed to each CSCABlock (see CSCAUnit).            shortcut: residual connections inside each CSCABlock.            g: grouping (kept for signature compatibility; unused here).            e: hidden channel expansion ratio (hidden = int(c2 * e)).            residual: add the '+ x' residual to the trunk branch (ablation switch).            use_scale: apply the learnable residual scale (lambda; ablation switch).            pe_on_input: feed the raw input x to the pe branch (=co+sp variant).
        """
        super().__init__()
        if int(c2) <= 0:
            raise ValueError(f'ProCSCA expects a positive output channel count, got c2={c2!r}')
        if not 0.0 < float(e) <= 1.0:
            raise ValueError(f'ProCSCA expansion ratio must be in (0, 1], got e={e!r}')

        self.n_blocks: int = max(round(n), 1)
        self.hidden_channels: int = int(c2 * float(e))
        if self.hidden_channels <= 0:
            raise ValueError(f'ProCSCA hidden channels must be positive, got c2={c2}, e={e}')

        # Stage 1 -- 1x1 conv projecting c1 -> 2 * hidden channels, later chunked
        # into the shortcut stream and the progressively refined stream.
        self.cv1 = Conv(c1, 2 * self.hidden_channels, 1, 1)

        # Stage 3/4 -- 1x1 conv fusing (2 + n) intermediate feature maps -> c2.
        self.cv2 = Conv((2 + self.n_blocks) * self.hidden_channels, c2, 1)

        # Stage 2 -- n cascaded CSCABlocks refining the second stream.
        self.blocks = nn.ModuleList(
            CSCABlock(self.hidden_channels, reduction=max(1, self.hidden_channels // 32), shortcut=shortcut, topology=topology, residual=residual, use_scale=use_scale, pe_on_input=pe_on_input, use_ffn=use_ffn, use_ffn_scale=use_ffn_scale, use_proj=use_proj, use_pe=use_pe)
            for _ in range(self.n_blocks)
        )

        # Stage 3/4 -- 1x1 conv fusing (2 + n) intermediate feature maps -> c2.

        self.forward = self._forward_progressive

    def _forward_progressive(self, x: torch.Tensor) -> torch.Tensor:
        """CSP-style progressive refinement (see class docstring, steps 1-4)."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.blocks)
        return self.cv2(torch.cat(y, 1))

    def _rebind_forward(self) -> None:
        """Re-bind the progressive forward after a deepcopy."""
        self.forward = self._forward_progressive

    __deepcopy__ = _deepcopy_with_rebind

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        return f'hidden_channels={self.hidden_channels}, n_blocks={self.n_blocks}'


class CSCABlock(nn.Module):
    """CSCA block: one attention unit followed by a lightweight FFN.

    Forward (paper Sec. CSCA / FFN):
        1. attention refinement: out = CSCAUnit(x)   (channel + spatial + coordinate fusion)
        2. optional residual:    x = x + out         (shortcut=True)
        3. FFN refinement:       ffn_out = ffn(x) * ffn_scale (learnable scale)
        4. optional residual:    x = x + ffn_out     (shortcut=True)

    Role in ablations:
        The fundamental building block inside ProCSCA ("CSCA blocks" in the paper);
        the FFN provides channel-wise feature re-embedding after attention fusion.
    """

    def __init__(self, channels: int, topology: str = 'chh-co+sp', reduction: int = 16, shortcut: bool = True, residual: bool = True, use_scale: bool = True, pe_on_input: bool = False, use_ffn: bool = True, use_ffn_scale: bool = True, use_proj: bool = True, use_pe: bool = True) -> None:
        """Initialize the attention unit and the FFN post-refinement branch.

        Args:
            channels: input/output channels (the block is channel-preserving).
            topology: attention topology expression passed to CSCAUnit (see CSCAUnit).
            reduction: channel reduction ratio of the attention branches.
            shortcut: use residual connections around attention and FFN.
            residual: add the '+ x' residual to the trunk branch (ablation switch).
            use_scale: apply the learnable residual scale (lambda; ablation switch).
            pe_on_input: feed the raw input x to the pe branch (=co+sp variant).
        """
        super().__init__()
        if int(channels) <= 0:
            raise ValueError(f'CSCABlock expects a positive channel count, got channels={channels!r}')

        self.channels: int = int(channels)
        self.shortcut: bool = bool(shortcut)

        # Stage 1 -- channel + spatial + coordinate attention fusion (paper Sec. CSCA).
        self.attention = CSCAUnit(self.channels, topology=topology, residual=residual, use_scale=use_scale, pe_on_input=pe_on_input, reduction=reduction, use_proj=use_proj, use_pe=use_pe)

        # Stage 2 -- FFN post-refinement (paper Sec. FFN) with a learnable scale.
        self.use_ffn = bool(use_ffn)
        self.ffn = self._build_ffn(self.channels) if use_ffn else nn.Identity()
        if use_ffn and use_ffn_scale:
            self.ffn_scale = nn.Parameter(torch.tensor(0.2))

        self.forward = self._forward_block

    def _build_ffn(self, channels: int) -> nn.Sequential:
        """Build the two point-wise convolutions expanding/compressing channels."""
        return nn.Sequential(
            Conv(channels, channels * 2, 1),
            Conv(channels * 2, channels, 1, act=False),
        )

    def _forward_block(self, x: torch.Tensor) -> torch.Tensor:
        """Attention refinement + optional residual + FFN refinement + optional residual."""
        att_out = self.attention(x)
        x = x + att_out if self.shortcut else att_out

        if not getattr(self, "use_ffn", True):
            return x
        ffn_out = self.ffn(x) * getattr(self, "ffn_scale", 1.0)
        x = x + ffn_out if self.shortcut else ffn_out
        return x


class CSCAUnit(nn.Module):
    """CSCA attention unit (paper Sec. CSCA) with a configurable attention topology.

    Topology expression (``topology``):
        - '-' chains attentions serially (the output of one feeds the next);
        - '+' runs attentions in parallel on the same input (outputs are summed);
        - the FIRST attention of the expression is the trunk: it receives the
          '+ x' residual (when ``residual`` is True) and provides the input of
          the position-encoding branch (pe);
        - branch names: 'chh' (channel), 'co' (coordinate), 'sp' (spatial);
          each branch may appear at most once, 1..3 branches in total.

    Examples (each reproduces one variant of the attention-ablation family):
        'chh-co+sp' (default, paper design): trunk=ch(x)+x; co(trunk) || sp(trunk); sum
        'chh-co-sp'  : serial chain  ch -> co -> sp            (=ch-co-sp ablation)
        'co-chh-sp'  : serial chain  co -> sp -> ch            (=co-ch-sp ablation)
        'chh+sp+co'  : three-way parallel sum                   (=sp+ch+co ablation)
        'chh+co-sp' : parallel sum then serial (parallel-then-serial)
        'chh-co'     : two-branch serial                        (=ch-co ablation)
        'chh+co'     : two-branch parallel                      (=ch+co ablation)
        'chh'       : channel attention only (with residual)   (=ch ablation)
        'co' / 'sp' : single branch without residual           (=co / =sp ablation,
                       set residual=False to match these two variants)
        use_scale=False removes the learnable residual scale (lambda)  (=notres ablation)

    Forward (compiled once in __init__ into a specialized straight-line callable):
        first attention:  trunk = m(x) + x (when residual); acc = trunk; pe input = trunk
        '-'  attention:   acc = m(acc)                      (serial chaining)
        '+'  attention:   acc = acc + m(prev_input)         (parallel, summed)
        out = acc * lambda + pe(trunk)   (lambda omitted when ``use_scale`` False)
        out = proj(out)

    Only the branches referenced by ``topology`` are instantiated, so parameter
    counts match the original per-variant reference implementations.

    Role in ablations:
        The fundamental attention unit embedded in CSCABlock / ProCSCA.
    """

    #: Branch registry used while parsing the topology expression.
    _BRANCH_NAMES: Tuple[str, ...] = ('chh', 'co', 'sp')

    def __init__(
        self,
        channels: int,
        topology: str = 'chh-co+sp',
        reduction: int = 16,
        init_scale: float = 0.2,
        residual: bool = True,
        use_scale: bool = True,
        pe_on_input: bool = False,
        use_proj: bool = True,
        use_pe: bool = True,
    ) -> None:
        """Initialize the attention branches referenced by the topology and compile the forward.

        Args:
            channels: input/output channels (the unit is channel-preserving).
            topology: attention topology expression (see the class docstring for the
                grammar and the full list of ablation variants it reproduces).
            reduction: channel reduction ratio of the attention branches.
            init_scale: initial value of the learnable residual scale (lambda).
            residual: add the '+ x' residual to the trunk branch (ablation switch).
            use_scale: apply the learnable residual scale (lambda) before the
                position-encoding branch (ablation switch; when False the lambda
                parameter is not registered at all).
            pe_on_input: feed the raw input x (instead of the trunk output) to the
                position-encoding branch (reproduces the =co+sp variant).
        """
        super().__init__()
        if int(channels) <= 0:
            raise ValueError(f'CSCAUnit expects a positive channel count, got channels={channels!r}')
        if int(reduction) <= 0:
            raise ValueError(f'CSCAUnit expects a positive reduction ratio, got reduction={reduction!r}')

        self.c: int = int(channels)
        self.reduction: int = int(reduction)
        self.topology = topology
        self.residual: bool = bool(residual)
        self.use_scale: bool = bool(use_scale)
        self.pe_on_input: bool = bool(pe_on_input)

        # Stage 1/3 -- parse the topology expression into [(branch_name, op), ...].
        parsed = self._parse_topology(self.topology)
        self._parsed: Tuple[Tuple[str, str], ...] = parsed
        used = [name for name, _ in parsed]

        # Stage 2/3 -- instantiate ONLY the branches referenced by the topology, so
        # parameter counts match the original per-variant reference implementations.
        branch_map: Dict[str, nn.Module] = {}
        if 'chh' in used:
            self.channel_attention = ChannelAttention(self.c, self.reduction)
            branch_map['chh'] = self.channel_attention
        if 'sp' in used:
            self.spatial_attention = SpatialAttention()
            branch_map['sp'] = self.spatial_attention
        if 'co' in used:
            self.coord_attention = CoordinateAttention(self.c, self.reduction)
            branch_map['co'] = self.coord_attention

        # Stage 3/3 -- attach modules to the parsed pipeline and compile the forward
        # into a specialized straight-line callable (no runtime branching/loops).
        self._pipeline: Tuple[Tuple[nn.Module, str, bool], ...] = self._build_pipeline(parsed, branch_map, self.residual)

        # Compiled straight-line forward: a class-level bound method (picklable for
        # torch.save checkpoints; closures are NOT picklable).
        self._mods: Tuple[nn.Module, ...] = tuple(module for module, _, _ in self._pipeline)
        self._forward_name: str = self._shape_forward_name(len(self._pipeline), tuple(op for _, op, _ in self._pipeline))
        self._pe_fn = self._pe_from_input if self.pe_on_input else self._pe_from_trunk
        self._scale_fn = self._apply_scale if self.use_scale else self._apply_identity_scale
        # trunk residual combiner ('+ x'), selected once at construction time;
        # disabled for the =co / =sp / =co+sp ablation variants (residual=False).
        self._boost = self._boost_residual if self.residual else self._boost_identity

        # lambda is registered only when use_scale=True, matching the reference
        # =notres variant which does not create this parameter at all.
        if self.use_scale:
            self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.proj = Conv(self.c, self.c, 1, act=False) if use_proj else nn.Identity()
        self.pe = Conv(self.c, self.c, 3, 1, g=self.c, act=False) if use_pe else None
        self.forward = getattr(self, self._forward_name)

    @staticmethod
    def _parse_topology(topology) -> Tuple[Tuple[str, str], ...]:
        """Parse a topology into [(branch_name, op), ...].

        Args:
            topology: either an expression string such as ``'chh-co+sp'`` ('-' serial,
                '+' parallel) or a list/tuple of branch names describing a serial
                chain (``['ch', 'co']`` == ``'chh-co'``). The list form exists because
                parse_model re-evaluates string args against its locals, and the
                historically 'ch' collided with parse_model's channel-width list local; renamed to 'chh' so plain string form is now safe in yaml.

        Returns:
            Ordered ``(branch_name, op)`` pairs; ``op`` is ``'-'``/``'+'`` or ``None``
            for the trunk (first attention overall).

        Raises:
            ValueError: on empty expressions, unknown or duplicated branch names.
        """
        if isinstance(topology, (list, tuple)):
            names = [str(n).strip().lower() for n in topology]
            # 历史兼容: 旧 yaml 的列表形式 ['ch'] 先归一化为 'chh' 再校验
            names = ["chh" if n == "ch" else n for n in names]
            if not names:
                raise ValueError('CSCAUnit topology list must not be empty')
            invalid = [n for n in names if n not in CSCAUnit._BRANCH_NAMES]
            if invalid:
                valid = ', '.join(f"'{v}'" for v in CSCAUnit._BRANCH_NAMES)
                raise ValueError(f'Invalid branch {invalid!r} in topology list; expected [{valid}]')
            if len(set(names)) != len(names):
                raise ValueError(f'Duplicated branch in topology list {topology!r}')
            return tuple(('chh' if n == 'ch' else n, None) for n in names)

        text = str(topology).strip().lower()
        if not text:
            raise ValueError(f'CSCAUnit topology must not be empty, got {topology!r}')

        matches = re.findall(r'([+-]?)\s*(chh|ch|co|sp)', text)
        if not matches:
            valid = ', '.join(f"'{v}'" for v in CSCAUnit._BRANCH_NAMES)
            raise ValueError(f'Invalid CSCAUnit topology {topology!r}; expected branch names [{valid}]')
        rebuilt = ''.join(op + name for op, name in matches)
        if rebuilt != text.replace(' ', ''):
            raise ValueError(f'Invalid CSCAUnit topology {topology!r}: unexpected tokens')

        parsed = []
        seen = []
        # 'ch' -> 'chh' 别名归一 (历史兼容; yaml 层应使用 'chh' 避免与 parse_model 局部变量冲突)
        for op, name in matches:
            if name in seen:
                raise ValueError(f'Duplicated branch {name!r} in topology {topology!r}')
            seen.append(name)
            parsed.append(('chh' if name == 'ch' else name, op or None))
        return tuple(parsed)

    def _build_pipeline(self, parsed: Tuple[Tuple[str, str], ...], branch_map: Dict[str, nn.Module], residual: bool) -> Tuple:
        """Attach modules to the parsed topology: [(module, op, add_residual), ...].

        Args:
            parsed: ordered ``(branch_name, op)`` pairs from :meth:`_parse_topology`.
            branch_map: mapping from branch names to their attention modules.
            residual: whether the trunk branch (first attention overall) adds '+ x'.

        Returns:
            The parsed execution pipeline used by the compiled forward.

        Raises:
            KeyError: if a parsed branch was not instantiated (internal error).
        """
        pipeline = []
        for i, (name, op) in enumerate(parsed):
            add_residual = residual and i == 0  # trunk = first attention overall
            pipeline.append((branch_map[name], op, add_residual))
        return tuple(pipeline)

    def _pe_from_trunk(self, trunk_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Position-encoding input: the trunk output (default, paper design)."""
        return self.pe(trunk_out) if self.pe is not None else 0

    def _pe_from_input(self, trunk_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Position-encoding input: the raw input x (reproduces the =co+sp variant)."""
        return self.pe(x) if self.pe is not None else 0

    def _apply_scale(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the learnable residual scale (lambda) to the fused feature."""
        return y * self.residual_scale

    def _apply_identity_scale(self, y: torch.Tensor) -> torch.Tensor:
        """Identity scaling used when ``use_scale`` is disabled (lambda omitted)."""
        return y

    @staticmethod
    def _boost_residual(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Trunk residual combiner: y + x (default; matches every reference topology)."""
        return y + x

    @staticmethod
    def _boost_identity(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Identity combiner used when ``residual`` is disabled (=co / =sp / =co+sp)."""
        return y

    @staticmethod
    def _shape_forward_name(n_items: int, ops: Tuple[str, ...]) -> str:
        """Map the parsed pipeline shape to its specialized forward method name.

        ``ops`` holds the op of every pipeline item; item 0 is the trunk (op None),
        so the serial/parallel classification is decided by the tail ``ops[1:]``.
        """
        if n_items == 1:
            return '_forward_1item'
        tail = tuple(ops[1:])
        if n_items == 2:
            return '_forward_2item_serial' if tail[0] != '+' else '_forward_2item_parallel'
        if tail == ('-', '-'):
            return '_forward_3item_serial'
        if tail == ('-', '+'):
            return '_forward_3item_serial_parallel'
        if tail == ('+', '-'):
            return '_forward_3item_parallel_serial'
        return '_forward_3item_parallel'

    def _forward_1item(self, x: torch.Tensor) -> torch.Tensor:
        """Single attention: trunk -> scale -> pe -> proj."""
        trunk_out = self._boost(self._mods[0](x), x)
        return self.proj(self._scale_fn(trunk_out) + self._pe_fn(trunk_out, x))

    def _forward_2item_serial(self, x: torch.Tensor) -> torch.Tensor:
        """Two-branch serial: trunk -> second -> scale -> pe -> proj."""
        trunk_out = self._boost(self._mods[0](x), x)
        chained = self._mods[1](trunk_out)
        return self.proj(self._scale_fn(chained) + self._pe_fn(trunk_out, x))

    def _forward_2item_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Two-branch parallel: trunk + second(x) -> scale -> pe -> proj."""
        trunk_out = self._boost(self._mods[0](x), x)
        acc = trunk_out + self._mods[1](x)
        return self.proj(self._scale_fn(acc) + self._pe_fn(trunk_out, x))

    def _forward_3item_serial(self, x: torch.Tensor) -> torch.Tensor:
        """Three-branch serial: trunk -> second -> third -> scale -> pe -> proj."""
        trunk_out = self._boost(self._mods[0](x), x)
        chained = self._mods[2](self._mods[1](trunk_out))
        return self.proj(self._scale_fn(chained) + self._pe_fn(trunk_out, x))

    def _forward_3item_serial_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Serial trunk + parallel pair (default, paper design)."""
        trunk_out = self._boost(self._mods[0](x), x)
        acc = self._mods[1](trunk_out) + self._mods[2](trunk_out)
        return self.proj(self._scale_fn(acc) + self._pe_fn(trunk_out, x))

    def _forward_3item_parallel_serial(self, x: torch.Tensor) -> torch.Tensor:
        """Parallel pair -> serial third -> scale -> pe -> proj."""
        trunk_out = self._boost(self._mods[0](x), x)
        merged = trunk_out + self._mods[1](x)
        chained = self._mods[2](merged)
        return self.proj(self._scale_fn(chained) + self._pe_fn(trunk_out, x))

    def _forward_3item_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Three-branch parallel: trunk + second(x) + third(x) -> scale -> pe -> proj."""
        trunk_out = self._boost(self._mods[0](x), x)
        acc = trunk_out + self._mods[1](x) + self._mods[2](x)
        return self.proj(self._scale_fn(acc) + self._pe_fn(trunk_out, x))

    def _rebind_forward(self) -> None:
        """Re-bind the compiled forward after a deepcopy."""
        self._boost = self._boost_residual if self.residual else self._boost_identity
        self.forward = getattr(self, self._forward_name)

    __deepcopy__ = _deepcopy_with_rebind

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        used = '-'.join(name for name, _ in self._parsed)
        return f'topology={self.topology}, branches={used}, residual={self.residual}, use_scale={self.use_scale}'

class CSCA(nn.Module):
    """CSCA ablation module: direct stacking of CSCABlocks (baseline of ProCSCA).

    Grounding in the paper:
        The ProCSCA figure caption explicitly contrasts ProCSCA with "direct
        stacking of attention blocks". This module implements exactly that
        baseline, so the CSCA-vs-ProCSCA ablation isolates ONE factor -- the
        CSP-style progressive design:

            CSCA    (baseline):  proj(c1->c2) -> CSCABlock x n   (direct stacking)
            ProCSCA (proposed):  cv1 split -> CSCABlock x n -> cv2 fuse

        Both modules share the SAME CSCABlock (CSCAUnit + FFN, shortcut
        residuals). The 1x1 proj is a neutral channel-alignment adapter
        (applied only when c1 != c2) so the baseline can be dropped into the
        detection head; it is not part of the CSCA design itself.

    Args:
        c1: input channels.
        c2: output channels.
        n: number of stacked CSCABlocks.
        topology: attention topology expression passed to each CSCABlock (see CSCAUnit).
        reduction: channel reduction ratio of the attention units.
        init_scale: init value of the learnable residual scale (per unit).
        shortcut: residual connections inside each CSCABlock.
        residual: add the '+ x' residual to the trunk branch (ablation switch).
        use_scale: apply the learnable residual scale (lambda; ablation switch).
        pe_on_input: feed the raw input x to the pe branch (=co+sp variant).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, topology='chh-co+sp', reduction: int = 16, init_scale: float = 0.2, shortcut: bool = True, residual: bool = True, use_scale: bool = True, pe_on_input: bool = False) -> None:
        """Initialize the channel-alignment projection and the stacked CSCABlocks.

        Args:
            c1: input channels.
            c2: output channels (1x1 conv projection applied when c1 != c2).
            n: number of stacked CSCABlocks.
            topology: attention topology expression passed to each CSCABlock (see CSCAUnit).
            reduction: channel reduction ratio of the attention units.
            init_scale: initial value of the learnable residual scale (per unit).
            shortcut: residual connections inside each CSCABlock.
        residual: add the '+ x' residual to the trunk branch (ablation switch).
        use_scale: apply the learnable residual scale (lambda; ablation switch).
        pe_on_input: feed the raw input x to the pe branch (=co+sp variant).
        """
        super().__init__()
        if int(c1) <= 0 or int(c2) <= 0:
            raise ValueError(f'CSCA expects positive channel counts, got c1={c1!r}, c2={c2!r}')
        if int(reduction) <= 0:
            raise ValueError(f'CSCA expects a positive reduction ratio, got reduction={reduction!r}')

        # Channel-alignment adapter (neutral plumbing, not part of the CSCA design).
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # Direct stacking of the SAME CSCABlocks that ProCSCA refines progressively.
        self.blocks = nn.Sequential(
            *(CSCABlock(c2, reduction=reduction, shortcut=shortcut, topology=topology, residual=residual, use_scale=use_scale, pe_on_input=pe_on_input) for _ in range(max(round(n), 1)))
        )

        self.forward = self._forward_blocks

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Channel alignment followed by direct CSCABlock stacking."""
        return self.blocks(self.proj(x))

    def _rebind_forward(self) -> None:
        """Re-bind the block forward after a deepcopy."""
        self.forward = self._forward_blocks

    __deepcopy__ = _deepcopy_with_rebind

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        return f'n_blocks={len(self.blocks)}'


class ChannelAttention(nn.Module):
    """Channel attention: global-average-pooling gating (SE-style).

    Forward (paper eq. Channel Attention):
        A_c = sigmoid(W2 . ReLU(W1 . GAP(x)))    (W1/W2: Linear layers with reduction)
        out = x * A_c                            (channel-wise broadcast)

    Reduces inter-channel redundancy and highlights defect-relevant channels.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize the two Linear projections of the gating mechanism.

        Args:
            channels: input/output channels (channel-preserving).
            reduction: reduction ratio of the hidden projection width.
        """
        super().__init__()
        if int(channels) <= 0:
            raise ValueError(f'ChannelAttention expects a positive channel count, got channels={channels!r}')
        if int(reduction) <= 0:
            raise ValueError(f'ChannelAttention expects a positive reduction ratio, got reduction={reduction!r}')

        self.channels: int = int(channels)
        self.reduction: int = int(reduction)
        mid: int = max(1, self.channels // self.reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, self.channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate the input with the sigmoid-activated channel descriptor."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        return f'channels={self.channels}, reduction={self.reduction}'


class SpatialAttention(nn.Module):
    """Spatial attention: 7x7 conv over channel-wise avg/max descriptors.

    Forward (paper eq. Spatial Attention):
        A_s = sigmoid(Conv7x7(concat(mean_c(x), max_c(x))))
        out = x * A_s

    Highlights high-response defect regions; kernel_size defaults to 7 per the paper.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize the spatial descriptor convolution.

        Args:
            kernel_size: spatial kernel size (must be odd for same padding).
        """
        super().__init__()
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError(f'SpatialAttention kernel_size must be a positive odd number, got {kernel_size!r}')

        self.kernel_size: int = int(kernel_size)
        self.conv = nn.Conv2d(2, 1, self.kernel_size, padding=self.kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate the input with the sigmoid-activated spatial descriptor."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        return f'kernel_size={self.kernel_size}'


class CoordinateAttention(nn.Module):
    """Coordinate attention: direction-aware 1D encoding along H and W.

    Forward (paper eq. Coordinate Attention):
        1. directional pooling:  Fh = AvgPool_w(x), Fw = AvgPool_h(x)
        2. concat + 1x1 conv + ReLU, then split back into H/W embeddings
        3. A_h = sigmoid(Conv1x1(Fh')), A_w = sigmoid(Conv1x1(Fw'))
        4. out = x * A_h * A_w

    Captures long-range dependencies along one spatial direction while
    preserving precise positional information along the other, which suits
    elongated and spatially distributed defect patterns.
    """

    def __init__(self, in_channels: int, reduction: int = 32) -> None:
        """Initialize the shared 1x1 embedding and the two direction-specific projections.

        Args:
            in_channels: input/output channels (channel-preserving).
            reduction: reduction ratio of the shared embedding width.
        """
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f'CoordinateAttention expects a positive channel count, got in_channels={in_channels!r}')
        if int(reduction) <= 0:
            raise ValueError(f'CoordinateAttention expects a positive reduction ratio, got reduction={reduction!r}')

        self.in_channels: int = int(in_channels)
        self.reduction: int = int(reduction)
        mid_channels: int = max(8, self.in_channels // self.reduction)

        self.conv1 = nn.Conv2d(self.in_channels, mid_channels, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, self.in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, self.in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate the input with the H/W direction-aware attention maps."""
        identity = x
        b, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True)  # (b, c, h, 1)
        x_w = torch.mean(x, dim=2, keepdim=True)  # (b, c, 1, w)
        x_w_perm = x_w.permute(0, 1, 3, 2)        # (b, c, w, 1)

        y = torch.cat([x_h, x_w_perm], dim=2)     # (b, c, h+w, 1)
        y = self.conv1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        attention_h = self.sigmoid(self.conv_h(x_h))
        attention_w = self.sigmoid(self.conv_w(x_w))
        return identity * attention_h * attention_w

    def extra_repr(self) -> str:
        """Return the canonical configuration shown when printing the module."""
        return f'in_channels={self.in_channels}, reduction={self.reduction}'
