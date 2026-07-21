import argparse
import csv
import gc
import math
import os
import time
import torch

try:
    import cudnn
except ImportError:
    cudnn = None

from einops import rearrange, repeat

try:
    from flash_attn.cute.bench_utils import (
        attention_ref,
        cudnn_fwd_setup,
        cudnn_bwd_setup,
    )
except ImportError:
    attention_ref = None
    cudnn_fwd_setup = None
    cudnn_bwd_setup = None

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None
try:
    from flash_attn.cute.interface import flash_attn_func as flash_attn_func_python
    from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_python
except ImportError:
    flash_attn_func_python = None
    flash_attn_varlen_func_python = None
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    flash_attn_func_v3 = None
    flash_attn_varlen_func_v3 = None

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
    flash_attn_func_v3 = None

from triton.testing import do_bench


# ── Autograd backward helper ────────────────────────────────────────────────

def _make_bwd_fn(fwd_fn, g, inputs):
    """Run fwd once, return a closure that benchmarks backward.

    Args:
        fwd_fn: zero-arg callable that runs the forward pass (with autograd).
        g: gradient tensor matching the forward output. Varlen callers may pass
            either an already-unpadded tensor or a dense tensor that can be
            flattened to the output shape.
        inputs: list of input tensors whose .grad should be cleared each iteration.
    """
    out = fwd_fn()
    if isinstance(out, tuple):
        out = out[0]
    if g.shape == out.shape:
        g_match = g
    elif g.numel() == out.numel():
        g_match = g.reshape(out.shape)
    elif g.ndim == out.ndim and g.shape[0] >= out.shape[0] and g.shape[1:] == out.shape[1:]:
        g_match = g[:out.shape[0]]
    else:
        raise ValueError(f"grad shape {tuple(g.shape)} does not match output shape {tuple(out.shape)}")
    def bwd_fn():
        for x in inputs:
            if x is not None:
                x.grad = None
        out.backward(g_match, retain_graph=True)
    return bwd_fn


# ── Backend definitions ─────────────────────────────────────────────────────
# Each setup_* function takes a context dict and returns (fwd_fn, bwd_fn).
# Either can be None if the backend doesn't support that direction for the
# given config.  fwd_fn / bwd_fn are zero-arg callables suitable for do_bench.

def setup_standard(ctx):
    if ctx["varlen"]:
        return None, None
    if ctx["sinks"] is not None:
        return None, None
    if attention_ref is None:
        return None, None
    if ctx["dtype"] == torch.float8_e4m3fn:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    # sinks = ctx["sinks"]
    # fwd_fn = lambda: attention_ref(q, k, v, causal=causal, learnable_sink=sinks)
    def fwd_fn():
        k_expanded = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
        v_expanded = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
        return attention_ref(q, k_expanded, v_expanded, causal=causal)
    # fwd_fn = lambda: attention_ref(q, k, v, causal=causal)
    bwd_fn = _make_bwd_fn(fwd_fn, g, [q, k, v]) if ctx["has_backward"] else None
    return fwd_fn, bwd_fn


def setup_fa2(ctx):
    if flash_attn_func is None or ctx["dtype"] == torch.float8_e4m3fn:
        return None, None
    if ctx["sinks"] is not None:
        return None, None
    if ctx["headdim"] != ctx["headdim_v"]:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    dropout_p, window_size_fa, softcap = ctx["dropout_p"], ctx["window_size_fa"], ctx["softcap"]
    deterministic = ctx["deterministic"]
    if ctx["varlen"]:
        qu, ku, vu, gu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"], ctx["g_unpad"]
        csq, csk, sq, sk = (
            ctx["cu_seqlens_q"],
            ctx["cu_seqlens_k"],
            ctx["max_seqlen_q"],
            ctx["max_seqlen_k"],
        )
        fwd_fn = lambda: flash_attn_varlen_func(qu, ku, vu, csq, csk, sq, sk, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap)
        bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func(qu, ku, vu, csq, csk, sq, sk, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap, deterministic=deterministic), gu, [qu, ku, vu]) if ctx["has_backward"] else None
    else:
        fwd_fn = lambda: flash_attn_func(q, k, v, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap)
        bwd_fn = _make_bwd_fn(lambda: flash_attn_func(q, k, v, dropout_p, causal=causal, window_size=window_size_fa, softcap=softcap, deterministic=deterministic), g, [q, k, v]) if ctx["has_backward"] else None
    return fwd_fn, bwd_fn


def setup_cudnn(ctx):
    if ctx["varlen"]:
        return None, None
    if ctx["sinks"] is not None:
        return None, None
    if cudnn_fwd_setup is None or cudnn_bwd_setup is None:
        return None, None
    if cudnn is None or ctx["headdim"] > 256 or ctx["dtype"] == torch.float8_e4m3fn:
        return None, None
    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    window_size_left = ctx["window_size"][0]
    # cuDNN expects (batch, nheads, seqlen, headdim) layout
    qt, kt, vt, gt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), g.transpose(1, 2)
    fwd_fn, o_gpu, lse_gpu = cudnn_fwd_setup(qt, kt, vt, causal=causal, window_size_left=window_size_left)
    bwd_fn = None
    if ctx["has_backward"]:
        fwd_fn()  # populate o and lse for bwd graph
        bwd_fn = cudnn_bwd_setup(qt, kt, vt, o_gpu, gt, lse_gpu, causal=causal, window_size_left=window_size_left)
    return fwd_fn, bwd_fn


def setup_fa3(ctx):
    if flash_attn_func_v3 is None:
        return None, None
    import inspect
    _fa3_has_sink = "learnable_sink" in inspect.signature(flash_attn_func_v3).parameters
    _fa3_varlen_has_sink = (
        flash_attn_varlen_func_v3 is not None
        and "learnable_sink" in inspect.signature(flash_attn_varlen_func_v3).parameters
    )

    q, k, v, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["g"], ctx["causal"]
    window_size_fa, softcap = ctx["window_size_fa"], ctx["softcap"]
    num_splits, pack_gqa, deterministic = ctx["num_splits"], ctx["pack_gqa"], ctx["deterministic"]
    k_use = ctx.get("k_paged", k) if ctx["page_size"] is not None else k
    v_use = ctx.get("v_paged", v) if ctx["page_size"] is not None else v
    sinks = ctx["sinks"]
    if sinks is not None and ((ctx["varlen"] and not _fa3_varlen_has_sink) or (not ctx["varlen"] and not _fa3_has_sink)):
        return None, None
    sink_kwargs = {"learnable_sink": sinks} if _fa3_has_sink and sinks is not None else {}
    sink_kwargs_varlen = {"learnable_sink": sinks} if _fa3_varlen_has_sink and sinks is not None else {}
    if ctx["varlen"]:
        qu, ku, vu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"]
        csq, csk, sq, sk = (
            ctx["cu_seqlens_q"],
            ctx["cu_seqlens_k"],
            ctx["max_seqlen_q"],
            ctx["max_seqlen_k"],
        )
        fwd_fn = lambda: flash_attn_varlen_func_v3(qu, ku, vu, csq, csk, sq, sk, causal=causal, window_size=window_size_fa, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, **sink_kwargs_varlen)
    else:
        fwd_fn = lambda: flash_attn_func_v3(q, k_use, v_use, causal=causal, window_size=window_size_fa, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, **sink_kwargs)
    # FA3 bwd only supports headdim == headdim_v and non-fp8
    bwd_fn = None
    if ctx["has_backward"] and ctx["dtype"] != torch.float8_e4m3fn and ctx["headdim"] == ctx["headdim_v"]:
        if ctx["varlen"]:
            gu = ctx["g_unpad"]
            bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func_v3(qu, ku, vu, csq, csk, sq, sk, causal=causal, window_size=window_size_fa, softcap=softcap, deterministic=deterministic, **sink_kwargs_varlen), gu, [qu, ku, vu, sinks])
        else:
            bwd_fn = _make_bwd_fn(lambda: flash_attn_func_v3(q, k, v, causal=causal, window_size=window_size_fa, softcap=softcap, deterministic=deterministic, **sink_kwargs), g, [q, k, v, sinks])
    return fwd_fn, bwd_fn


def setup_fa4(ctx):
    if flash_attn_func_python is None:
        return None, None
    q, k, v, qv, g, causal = ctx["q"], ctx["k"], ctx["v"], ctx["qv"], ctx["g"], ctx["causal"]
    window_size, softcap = ctx["window_size"], ctx["softcap"]
    num_splits, pack_gqa, deterministic = ctx["num_splits"], ctx["pack_gqa"], ctx["deterministic"]
    sinks = ctx["sinks"]
    gather_kv_indices = ctx.get("gather_kv_indices")
    k_use = ctx.get("k_paged", k) if ctx["page_size"] is not None else k
    v_use = ctx.get("v_paged", v) if ctx["page_size"] is not None else v
    if ctx["varlen"]:
        qu = ctx["q_unpad"]
        ku = ctx.get("k_paged", ctx["k_unpad"]) if ctx["page_size"] is not None else ctx["k_unpad"]
        vu = ctx.get("v_paged", ctx["v_unpad"]) if ctx["page_size"] is not None else ctx["v_unpad"]
        qvu = ctx["qv_unpad"]
        csq, csk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"]
        max_sq, max_sk, min_sk = ctx["max_seqlen_q"], ctx["max_seqlen_k"], ctx["min_seqlen_k"]
        pt = ctx["page_table"]
        gather_kv_indices_unpad = ctx.get("gather_kv_indices_unpad")
        fwd_fn = lambda: flash_attn_varlen_func_python(qu, ku, vu, qvu, csq, csk, max_seqlen_q=max_sq, max_seqlen_k=max_sk, min_seqlen_k=min_sk, page_table=pt, causal=causal, window_size=window_size, learnable_sink=sinks, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, gather_kv_indices=gather_kv_indices_unpad)
    else:
        fwd_fn = lambda: flash_attn_func_python(q, k_use, v_use, qv=qv, causal=causal, window_size=window_size, learnable_sink=sinks, softcap=softcap, num_splits=num_splits, pack_gqa=pack_gqa, gather_kv_indices=gather_kv_indices)
    bwd_fn = None
    if ctx["has_backward"] and ctx["dtype"] != torch.float8_e4m3fn:
        if ctx["varlen"]:
            qu, ku, vu, gu = ctx["q_unpad"], ctx["k_unpad"], ctx["v_unpad"], ctx["g_unpad"]
            qvu = ctx["qv_unpad"]
            csq, csk = ctx["cu_seqlens_q"], ctx["cu_seqlens_k"]
            max_sq, max_sk, min_sk = ctx["max_seqlen_q"], ctx["max_seqlen_k"], ctx["min_seqlen_k"]
            gather_kv_indices_unpad = ctx.get("gather_kv_indices_unpad")
            bwd_fn = _make_bwd_fn(lambda: flash_attn_varlen_func_python(qu, ku, vu, qvu, csq, csk, max_seqlen_q=max_sq, max_seqlen_k=max_sk, min_seqlen_k=min_sk, causal=causal, window_size=window_size, softcap=softcap, learnable_sink=sinks, num_splits=num_splits, deterministic=deterministic, gather_kv_indices=gather_kv_indices_unpad), gu, [qu, ku, vu, qvu, sinks])
        else:
            bwd_fn = _make_bwd_fn(lambda: flash_attn_func_python(q, k, v, qv=qv, causal=causal, window_size=window_size, softcap=softcap, learnable_sink=sinks, num_splits=num_splits, deterministic=deterministic, gather_kv_indices=gather_kv_indices), g, [q, k, v, qv, sinks])
    return fwd_fn, bwd_fn


# Ordered list of (display_name, cli_name, setup_fn)
BACKENDS = [
    ("FA4",      "fa4",      setup_fa4),
    # ("Standard", "standard", setup_standard),
    ("FA2",      "fa2",      setup_fa2),
    # ("cuDNN",    "cudnn",    setup_cudnn),
    ("FA3",      "fa3",      setup_fa3),
]


def get_peak_flops(device_index: int = 0, dtype: torch.dtype = torch.bfloat16) -> float | None:
    """Return peak BF16 dense TFLOPS for the given device. Returns None if unknown.

    Scaling by dtype:
      FP16 / BF16 : 1x  (identical hardware throughput)
      FP8         : 2x
    """
    # BF16 dense peak FLOPS from official NVIDIA spec sheets.
    # Checked against: num_SMs * tensor_core_clock * ops_per_SM_per_cycle.
    # The tensor core clock is NOT queryable (it differs from the max boost SM
    # clock reported by nvidia-smi), so we hardcode the spec-sheet values.
    _PEAK_BF16_FLOPS = {
        # Ampere
        "A100":  312e12,
        "A6000": 309.7e12,
        # Ada Lovelace
        "L40S":  362e12,
        # Hopper
        "H100 SXM": 989e12,
        "H100 NVL": 835e12,
        "H100 PCIe": 756e12,
        "H200":  989e12,
        "H20":   148e12,
        # Blackwell
        "GB200": 2.5e15,
        "GB300": 2.5e15,
        "B300":  2.25e15,
        "B200":  2.25e15,
    }

    device_name = torch.cuda.get_device_name(device_index)
    # Match longest key first so "H100 SXM" matches before "H100"
    peak = None
    for key in sorted(_PEAK_BF16_FLOPS, key=len, reverse=True):
        if key.lower() in device_name.lower():
            peak = _PEAK_BF16_FLOPS[key]
            break
    if peak is None:
        return None

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        peak *= 2
    return peak


def get_peak_bandwidth(device_index: int = 0) -> float | None:
    """Return peak HBM bandwidth in bytes/sec for the given device. Returns None if unknown."""
    _PEAK_BW = {
        # Ampere
        "A100":       2.0e12,
        "A6000":      0.768e12,
        # Ada Lovelace
        "L40S":       0.864e12,
        # Hopper
        "H100 SXM":   3.35e12,
        "H100 NVL":   3.35e12,
        "H100 PCIe":  2.0e12,
        "H200":       4.8e12,
        "H20":        4.0e12,
        # Blackwell
        "GB200":      8.0e12,
        "GB300":      8.0e12,
        "B300":       8.0e12,
        "B200":       8.0e12,
    }
    device_name = torch.cuda.get_device_name(device_index)
    for key in sorted(_PEAK_BW, key=len, reverse=True):
        if key.lower() in device_name.lower():
            return _PEAK_BW[key]
    return None


def parse_int_k(s):
    """Parse an integer with optional k/K suffix, e.g. '8k' -> 8192."""
    s = s.strip().lower()
    if s.endswith("k"):
        return int(s[:-1]) * 1024
    return int(s)


def csv_ints(s):
    """Parse comma-separated integers with optional k suffix, e.g. '512,1k,2k'."""
    return [parse_int_k(x) for x in s.split(",")]


def parse_headdims(s):
    """Parse comma-separated headdim specs. Each entry is hdim or hdim-hdim_v.

    Examples:
        '128'           -> [(128, 128)]
        '192-128'       -> [(192, 128)]
        '64,128,192'    -> [(64, 64), (128, 128), (192, 192)]
        '64,128,192-128,192' -> [(64, 64), (128, 128), (192, 128), (192, 192)]
    """
    result = []
    for item in s.split(","):
        if "-" in item:
            parts = item.split("-")
            result.append((int(parts[0]), int(parts[1])))
        else:
            hdim = int(item)
            result.append((hdim, hdim))
    return result


def csv_strs(s):
    """Parse comma-separated strings, e.g. 'fa3,fa4'."""
    return [x.strip() for x in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark FlashAttention')
    parser.add_argument('--grid-preset', type=str.lower, choices=['none', 'fragmented', 'smoke-fragmented'],
                        default='none',
                        help='Convenience grid. fragmented sweeps total_len=32k; '
                             'headdim=128,256; nheads=48,24; auto valid nheads_kv; '
                             'window=128,512,full; sink=false,true; varlen_profile=varlen_fragmented. '
                             'smoke-fragmented is a tiny version for validation.')
    parser.add_argument('--csv-out', type=str, default=None,
                        help='Write one CSV row per backend/direction/config. Rows are flushed after each benchmark.')
    parser.add_argument('--append-csv', action='store_true',
                        help='Append to --csv-out instead of overwriting it.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the expanded benchmark grid and exit without allocating tensors.')
    parser.add_argument('--headdim', type=parse_headdims, default=None,
                        help='Head dim(s), comma-separated. Each is hdim or hdim-hdim_v. E.g. 64,128,192-128')
    parser.add_argument('--fwd', action='store_true', help='Run forward only')
    parser.add_argument('--bwd', action='store_true', help='Run backward only')
    parser.add_argument('--varlen', action='store_true', default=False)
    parser.add_argument('--varlen-profile', type=csv_strs, default=None,
                        help='Comma-separated varlen profiles: dense,varlen_fragmented. '
                             'fragmented is accepted as an alias for varlen_fragmented.')
    parser.add_argument('--causal', type=str.lower, choices=['true', 'false', 'both'], default=None,
                        help='Causal mode (default: both)')
    parser.add_argument('--seqlen', type=csv_ints, default=None,
                        help='KV sequence length(s), comma-separated. Supports k suffix, e.g. 1k,2k,8k. '
                             'Ignored for varlen profiles; varlen only uses --total-seqlen.')
    parser.add_argument('--seqlen-q', type=csv_ints, default=None,
                        help='Q sequence length(s), comma-separated. Supports k suffix, e.g. 1,128,1k. '
                             'Ignored for varlen profiles. '
                             'Defaults to matching --seqlen (i.e. seqlen_q == seqlen_kv). '
                             'If a single value is given it is broadcast across all --seqlen values; '
                             'otherwise the list must match the length of --seqlen or be length 1.')
    parser.add_argument('--total-seqlen', type=csv_ints, default=None,
                        help='Total sequence length(s), comma-separated, for batch sizing (default: 32k).')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Dense batch size (default: total_seqlen // seqlen_kv). Ignored for varlen profiles.')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--nheads', type=csv_ints, default=None,
                        help='Number of Q heads (default: 32 for hdim<=64, 16 for hdim<=192, 8 for hdim>192)')
    parser.add_argument('--nheads-kv', type=str, default=None,
                        help='Number(s) of KV heads, comma-separated, or auto. '
                             'Default is nheads for single runs; fragmented preset uses auto.')
    parser.add_argument('--gqa-ratio', type=int, default=None,
                        help='GQA ratio (nheads // nheads_kv). Ignored if --nheads-kv is set.')
    parser.add_argument('--backend', type=csv_strs, default=None,
                        help='Which backends to benchmark, comma-separated (choices: all,standard,fa2,fa3,fa4,cudnn)')
    parser.add_argument('--gather-kv', type=int, default=None,
                        help='kv sparsity length for MLA (hdim=64, hdim_v=512 only). '
                             'When set, passes random kv indices (without repeats) to FA4 and uses gather-kv as '
                             'the effective KV length for flops/bandwidth accounting.')
    parser.add_argument('--num-splits', type=int, default=0,
                        help='Override kernel num_splits heuristic. 0 = auto (default). '
                             '>1 forces SplitKV with that many splits.')
    parser.add_argument('--pack-gqa', type=str.lower, choices=['auto', 'true', 'false'], default='auto',
                        help='Override kernel pack_gqa heuristic. auto = kernel default (default). '
                             'true/false force pack_gqa on or off. Note: for non-MLA GQA with '
                             'num_splits>1, interface.py may still force pack_gqa off pending a fix.')
    parser.add_argument('--window-size', type=csv_strs, default=None,
                        help='Comma-separated window specs. Use full or a left window like 128/512 '
                             '(encoded as (left, 0)); left:right is also accepted.')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Warmup iterations (default: 5)')
    parser.add_argument('--rep', type=int, default=10,
                        help='Repetitions per benchmark (default: 10)')
    parser.add_argument('--sleep', type=float, default=0.0,
                        help='Seconds to sleep before each measured benchmark (default: 0).')
    parser.add_argument('--empty-cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Call torch.cuda.empty_cache() between configs (default: true).')
    parser.add_argument('--sink', nargs='?', const='true', type=str.lower,
                        choices=['true', 'false', 'both'], default='both',
                        help='Learnable sink mode (default: both)')
    return parser.parse_args()


def resolve_seqlen_q_list(seqlen_list, seqlen_q_arg):
    """Return a list of seqlen_q values parallel to seqlen_list.

    Rules:
      - None  → seqlen_q == seqlen_kv for every entry
      - single value [x] → broadcast x across all entries
      - matching length → use as-is
      - mismatch → raise
    """
    if seqlen_q_arg is None:
        return list(seqlen_list)
    if len(seqlen_q_arg) == 1:
        return [seqlen_q_arg[0]] * len(seqlen_list)
    if len(seqlen_q_arg) != len(seqlen_list):
        raise ValueError(
            f"--seqlen-q has {len(seqlen_q_arg)} values but --seqlen has "
            f"{len(seqlen_list)}; they must match (or pass a single value to broadcast)."
        )
    return list(seqlen_q_arg)


CSV_FIELDS = [
    "backend",
    "direction",
    "status",
    "error",
    "total_len",
    "head_dim",
    "nheads",
    "nheads_kv",
    "window",
    "sink",
    "ms",
    "tflops",
    "mfu_pct",
    "tbs",
    "bw_pct",
]


def normalize_varlen_profile(profile):
    profile = profile.strip().lower().replace("-", "_")
    aliases = {
        "fragmented": "varlen_fragmented",
    }
    profile = aliases.get(profile, profile)
    valid = {"dense", "varlen_fragmented"}
    if profile not in valid:
        raise ValueError(f"unknown varlen profile {profile!r}; expected one of {sorted(valid)}")
    return profile


def resolve_tristate(value):
    if value == "true":
        return [True]
    if value == "false":
        return [False]
    return [False, True]


def resolve_headdims(args):
    if args.headdim is not None:
        return args.headdim
    if args.grid_preset in ("fragmented", "smoke-fragmented"):
        # FA4 on SM100/SM110 does not support the normal-attention (192, 192)
        # shape, so keep the default grid to configs that produce valid rows.
        values = [128] if args.grid_preset == "smoke-fragmented" else [128, 256]
        return [(d, d) for d in values]
    return [(128, 128)]


def format_headdim(headdim, headdim_v):
    return headdim if headdim == headdim_v else f"{headdim}-{headdim_v}"


def resolve_total_seqlens(args):
    if args.total_seqlen is not None:
        return args.total_seqlen
    if args.grid_preset in ("fragmented", "smoke-fragmented"):
        return [32 * 1024] if args.grid_preset == "fragmented" else [8 * 1024]
    return [32 * 1024]


def resolve_seqlens(args, total_len):
    if args.seqlen is not None:
        return args.seqlen
    if args.grid_preset == "fragmented":
        return [2 * 1024, 8 * 1024]
    if args.grid_preset == "smoke-fragmented":
        return [2 * 1024]
    return [8192]


def resolve_varlen_profiles(args):
    if args.varlen_profile is not None:
        return [normalize_varlen_profile(p) for p in args.varlen_profile]
    if args.grid_preset in ("fragmented", "smoke-fragmented"):
        return ["varlen_fragmented"]
    return ["varlen_fragmented"] if args.varlen else ["dense"]


def auto_nheads_kv_values(nheads):
    candidates = [nheads, nheads // 2, nheads // 4, 8, 6, 4, 2, 1]
    result = []
    for nheads_kv in candidates:
        if nheads_kv > 0 and nheads % nheads_kv == 0 and nheads_kv not in result:
            result.append(nheads_kv)
    return result


def resolve_nheads_list(args, headdim):
    if args.nheads is not None:
        return args.nheads
    if args.grid_preset == "fragmented":
        return [48, 32, 24, 16]
    if args.grid_preset == "smoke-fragmented":
        return [48]
    return [32 if headdim <= 64 else 16 if headdim <= 192 else 8]


def resolve_nheads_kv_list(args, nheads):
    if args.nheads_kv is not None:
        if args.nheads_kv.strip().lower() == "auto":
            return auto_nheads_kv_values(nheads)
        values = csv_ints(args.nheads_kv)
    elif args.gqa_ratio is not None:
        values = [nheads // args.gqa_ratio]
    elif args.grid_preset in ("fragmented", "smoke-fragmented"):
        values = auto_nheads_kv_values(nheads)
    else:
        values = [nheads]
    for nheads_kv in values:
        if nheads_kv <= 0 or nheads % nheads_kv != 0:
            raise ValueError(f"nheads_kv={nheads_kv} must be a positive divisor of nheads={nheads}")
    return values


def parse_window_spec(spec):
    spec = spec.strip().lower()
    if spec in ("full", "none", "-1", "-1:-1"):
        return "full", (None, None), (-1, -1)
    if ":" in spec:
        left_s, right_s = spec.split(":", 1)

        def parse_side(value):
            value = value.strip().lower()
            return None if value in ("", "none", "full", "-1") else int(value)

        left, right = parse_side(left_s), parse_side(right_s)
        fa_left = -1 if left is None else left
        fa_right = -1 if right is None else right
        label = f"{fa_left}:{fa_right}"
        if left is None and right is None:
            label = "full"
        return label, (left, right), (fa_left, fa_right)
    window = parse_int_k(spec)
    return str(window), (window, 0), (window, 0)


def resolve_window_specs(args):
    specs = args.window_size
    if specs is None:
        if args.grid_preset == "fragmented":
            specs = ["128", "512", "full"]
        elif args.grid_preset == "smoke-fragmented":
            specs = ["128"]
        else:
            specs = ["full"]
    return [parse_window_spec(spec) for spec in specs]


def default_varlen_batch_size(total_len, max_len, mean_len=1100.0, min_len=50):
    batch_size = max(1, int(round(total_len / mean_len)))
    batch_size = max(batch_size, math.ceil(total_len / max_len))
    max_by_min = total_len // min_len
    if batch_size > max_by_min:
        raise ValueError(
            f"total_len={total_len} is too small for min_len={min_len} and batch_size={batch_size}"
        )
    return batch_size


def _redistribute_lengths(lengths, total_len, min_len, max_len, gen, protected=None):
    lengths = lengths.clamp(min=min_len, max=max_len).to(torch.int64)
    while True:
        diff = int(total_len - lengths.sum().item())
        if diff == 0:
            return lengths
        if diff > 0:
            room = max_len - lengths
            if protected is not None and torch.any(~protected):
                room = room.clone()
                room[protected] = 0
            if int(room.sum().item()) == 0 and protected is not None:
                room = max_len - lengths
            if int(room.sum().item()) == 0:
                raise ValueError("cannot increase lengths to the requested total_len")
            candidates = torch.nonzero(room > 0, as_tuple=False).flatten()
            order = candidates[torch.randperm(candidates.numel(), generator=gen)]
            for idx in order.tolist():
                add = min(int((max_len - lengths[idx]).item()), diff)
                if add > 0:
                    lengths[idx] += add
                    diff -= add
                if diff == 0:
                    break
        else:
            removable = lengths - min_len
            if protected is not None and torch.any(~protected):
                removable = removable.clone()
                removable[protected] = 0
            if int(removable.sum().item()) == 0 and protected is not None:
                removable = lengths - min_len
            if int(removable.sum().item()) == 0:
                raise ValueError("cannot decrease lengths to the requested total_len")
            candidates = torch.nonzero(removable > 0, as_tuple=False).flatten()
            order = candidates[torch.randperm(candidates.numel(), generator=gen)]
            for idx in order.tolist():
                sub = min(int((lengths[idx] - min_len).item()), -diff)
                if sub > 0:
                    lengths[idx] -= sub
                    diff += sub
                if diff == 0:
                    break


@torch.no_grad()
def make_cu_seqlens_q(
    total_len: int = 32 * 1024,
    device: str | torch.device = "cuda",
    min_len: int = 50,
    lt_threshold: int = 100,
    p_lt_threshold: float = 0.05,
    median_len: float = 540.0,
    mean_len: float = 1100.0,
    max_len: int | None = None,
    seed: int | None = None,
    return_lengths: bool = False,
):
    device = torch.device(device)

    if max_len is None:
        max_len = total_len

    batch_size = int(round(total_len / mean_len))

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if total_len < batch_size * min_len:
        raise ValueError(
            f"total_len={total_len} is too small to satisfy "
            f"batch_size={batch_size}, min_len={min_len}"
        )

    if total_len > batch_size * max_len:
        raise ValueError(
            f"total_len={total_len} is too large to satisfy "
            f"batch_size={batch_size}, max_len={max_len}"
        )

    effective_mean = total_len / batch_size

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    dtype = torch.float64

    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    )

    mu = math.log(median_len)

    z_p = normal.icdf(
        torch.tensor(p_lt_threshold, device=device, dtype=dtype)
    ).item()

    sigma_left = (math.log(lt_threshold) - mu) / z_p

    def estimate_mean(sigma_right: float, n_grid: int = 20000):
        u = (
            torch.arange(n_grid, device=device, dtype=dtype) + 0.5
        ) / n_grid
        z = normal.icdf(u)

        sigma = torch.where(
            u < 0.5,
            torch.tensor(sigma_left, device=device, dtype=dtype),
            torch.tensor(sigma_right, device=device, dtype=dtype),
        )

        y = mu + sigma * z
        x = torch.exp(y)
        x = x.clamp(min=min_len, max=max_len)
        return x.mean().item()

    lo, hi = 1e-4, 1.0
    while estimate_mean(hi) < effective_mean:
        hi *= 2.0
        if hi > 10.0:
            break

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if estimate_mean(mid) < effective_mean:
            lo = mid
        else:
            hi = mid

    sigma_right = 0.5 * (lo + hi)

    offset = torch.rand((), device=device, dtype=dtype, generator=gen)
    u = (torch.arange(batch_size, device=device, dtype=dtype) + offset) / batch_size
    u = u.clamp(1e-6, 1.0 - 1e-6)

    perm = torch.randperm(batch_size, device=device, generator=gen)
    u = u[perm]

    z = normal.icdf(u)

    sigma = torch.where(
        u < 0.5,
        torch.tensor(sigma_left, device=device, dtype=dtype),
        torch.tensor(sigma_right, device=device, dtype=dtype),
    )

    raw_lengths = torch.exp(mu + sigma * z)
    raw_lengths = raw_lengths.clamp(min=min_len, max=max_len)

    raw_lengths = raw_lengths * (total_len / raw_lengths.sum())
    lengths = raw_lengths.round().clamp(min=min_len, max=max_len).to(torch.int64)

    diff = int(total_len - lengths.sum().item())

    if diff > 0:
        capacity = max_len - lengths
        order = torch.randperm(batch_size, device=device, generator=gen)

        for idx in order.tolist():
            if diff == 0:
                break
            add = min(int(capacity[idx].item()), diff)
            lengths[idx] += add
            diff -= add

    elif diff < 0:
        removable = lengths - min_len
        order = torch.randperm(batch_size, device=device, generator=gen)

        for idx in order.tolist():
            if diff == 0:
                break
            sub = min(int(removable[idx].item()), -diff)
            lengths[idx] -= sub
            diff += sub

    if diff != 0:
        raise RuntimeError(
            "Failed to adjust lengths to match the target total_len. "
            "Please check the input parameters."
        )

    cu_seqlens_q = torch.empty(
        batch_size + 1,
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_q[0] = 0
    cu_seqlens_q[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)

    if return_lengths:
        return cu_seqlens_q, lengths

    return cu_seqlens_q


def summarize_lengths(lengths):
    if lengths is None:
        return {
            "min_seqlen": "",
            "p50_seqlen": "",
            "p90_seqlen": "",
            "p99_seqlen": "",
            "max_seqlen": "",
            "pct_lt_100": "",
        }
    values = lengths.detach().cpu().to(torch.float32)
    qs = torch.quantile(values, torch.tensor([0.5, 0.9, 0.99]))
    return {
        "min_seqlen": int(values.min().item()),
        "p50_seqlen": float(qs[0].item()),
        "p90_seqlen": float(qs[1].item()),
        "p99_seqlen": float(qs[2].item()),
        "max_seqlen": int(values.max().item()),
        "pct_lt_100": float((values < 100).float().mean().item()),
    }


_PAIR_COUNT_CACHE = {}


def _count_pairs_one(seqlen_q, seqlen_k, causal, window_size):
    if causal and window_size == (None, None):
        offset = seqlen_k - seqlen_q
        total = 0
        for row in range(seqlen_q):
            total += max(0, min(seqlen_k, row + offset + 1))
        return total
    if not causal and window_size == (None, None):
        return seqlen_q * seqlen_k

    left, right = window_size
    if causal and right is None:
        right = 0
    offset = seqlen_k - seqlen_q
    total = 0
    for row in range(seqlen_q):
        col_left = 0 if left is None else row + offset - left
        col_right = seqlen_k - 1 if right is None else row + offset + right
        lo = max(0, col_left)
        hi = min(seqlen_k - 1, col_right)
        if hi >= lo:
            total += hi - lo + 1
    return total


def count_attention_pairs(batch_size, seqlen_q, seqlen_k, causal, window_size, lengths_q=None, lengths_k=None):
    if lengths_q is None:
        key = (batch_size, seqlen_q, seqlen_k, causal, window_size)
        if key not in _PAIR_COUNT_CACHE:
            _PAIR_COUNT_CACHE[key] = batch_size * _count_pairs_one(seqlen_q, seqlen_k, causal, window_size)
        return _PAIR_COUNT_CACHE[key]
    q_vals = tuple(int(x) for x in lengths_q.detach().cpu().tolist())
    k_vals = q_vals if lengths_k is None else tuple(int(x) for x in lengths_k.detach().cpu().tolist())
    key = (q_vals, k_vals, causal, window_size)
    if key not in _PAIR_COUNT_CACHE:
        _PAIR_COUNT_CACHE[key] = sum(
            _count_pairs_one(lq, lk, causal, window_size) for lq, lk in zip(q_vals, k_vals)
        )
    return _PAIR_COUNT_CACHE[key]


def metric_flops(batch_size, nheads, seqlen_q, seqlen_k, headdim, headdim_v, causal, window_size, has_qv, lengths_q=None, lengths_k=None):
    pairs = count_attention_pairs(batch_size, seqlen_q, seqlen_k, causal, window_size, lengths_q, lengths_k)
    eff_headdim = headdim + headdim_v if has_qv else headdim
    return nheads * 2 * pairs * (eff_headdim + headdim_v)


def metric_bandwidth_bytes(total_q, total_k, nheads, nheads_kv, headdim, headdim_v, dtype_bytes, direction, has_qv=False):
    q = total_q * nheads * headdim
    qv = total_q * nheads * headdim_v if has_qv else 0
    k = total_k * nheads_kv * headdim
    v = total_k * nheads_kv * headdim_v
    do = total_q * nheads * headdim_v
    if direction == "FWD":
        return (q + qv + k + v + do) * dtype_bytes
    dq = q
    dk = k
    dv = v
    return (q + qv + k + v + do + dq + dk + dv) * dtype_bytes


def main():
    args = parse_args()

    if args.fwd or args.bwd:
        has_forward = args.fwd
        has_backward = args.bwd
    else:
        has_forward = args.grid_preset == "none"
        has_backward = args.grid_preset != "none"

    causal_vals = resolve_tristate(args.causal) if args.causal is not None else (
        [True] if args.grid_preset in ("fragmented", "smoke-fragmented") else [False, True]
    )
    sink_vals = resolve_tristate(args.sink)
    headdim_pairs = resolve_headdims(args)
    total_seqlen_list = resolve_total_seqlens(args)
    varlen_profiles = resolve_varlen_profiles(args)
    window_specs = resolve_window_specs(args)
    gather_kv_length = args.gather_kv

    backend_arg = args.backend
    if backend_arg is None:
        backend_arg = ["fa4"] if args.grid_preset in ("fragmented", "smoke-fragmented") else ["all"]
    enabled = set(backend_arg)
    if "all" in enabled:
        enabled = {cli for _, cli, _ in BACKENDS}
    active_backends = [(name, cli, fn) for name, cli, fn in BACKENDS if cli in enabled]

    directions = []
    if has_forward:
        directions.append(("FWD", 1.0))
    if has_backward:
        directions.append(("BWD", 2.5))
    if not directions:
        raise ValueError("No benchmark direction selected")
    if not active_backends:
        raise ValueError(f"No active backends from --backend={backend_arg}")

    torch.manual_seed(0)
    dropout_p = 0.0
    dtype = torch.bfloat16
    dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    device = "cuda"
    peak_flops = get_peak_flops(0, dtype=dtype) if torch.cuda.is_available() else None
    peak_bw = get_peak_bandwidth(0) if torch.cuda.is_available() else None
    page_size = None
    softcap = 0.0
    deterministic = args.deterministic
    warmup, rep = args.warmup, args.rep
    pack_gqa = None if args.pack_gqa == "auto" else (args.pack_gqa == "true")
    dtype_bytes = 1 if dtype == torch.float8_e4m3fn else 2

    csv_file = None
    writer = None
    if args.csv_out is not None:
        csv_dir = os.path.dirname(args.csv_out)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        mode = "a" if args.append_csv else "w"
        csv_file = open(args.csv_out, mode, newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not args.append_csv:
            writer.writeheader()

    ok_count = 0
    skip_count = 0
    error_count = 0
    dry_count = 0
    planned_rows = 0

    def emit_row(row):
        nonlocal ok_count, skip_count, error_count, dry_count
        if row["status"] == "ok":
            ok_count += 1
        elif row["status"] == "skipped":
            skip_count += 1
        elif row["status"] == "dry_run":
            dry_count += 1
        else:
            error_count += 1
        if writer is not None:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
            csv_file.flush()
        if row["status"] == "ok":
            print(
                f"{row['backend']} {row['direction']} ok "
                f"total={row['total_len']} hdim={row['head_dim']} "
                f"h={row['nheads']}/{row['nheads_kv']} win={row['window']} "
                f"sink={row['sink']} ms={row['ms']:.3f} tflops={row['tflops']:.1f}"
            )
        elif row["status"] == "error":
            print(
                f"{row['backend']} {row['direction']} error "
                f"total={row['total_len']} hdim={row['head_dim']} "
                f"h={row['nheads']}/{row['nheads_kv']} win={row['window']} "
                f"sink={row['sink']}: {row['error']}"
            )

    try:
        for total_seqlen in total_seqlen_list:
            for varlen_profile in varlen_profiles:
                varlen = varlen_profile != "dense"
                varlen_cu_seqlens_cpu = None
                varlen_lengths_cpu = None
                if varlen:
                    varlen_cu_seqlens_cpu, varlen_lengths_cpu = make_cu_seqlens_q(
                        total_len=total_seqlen,
                        device="cpu",
                        seed=123,
                        return_lengths=True,
                    )
                    varlen_stats = summarize_lengths(varlen_lengths_cpu)
                    pair_parts = []
                    for pair_window_label, pair_window_size, _ in window_specs:
                        for pair_causal in causal_vals:
                            label = pair_window_label
                            if len(causal_vals) > 1:
                                label = f"{label}/causal={pair_causal}"
                            pairs = count_attention_pairs(
                                int(varlen_lengths_cpu.numel()),
                                int(varlen_lengths_cpu.max().item()),
                                int(varlen_lengths_cpu.max().item()),
                                causal=pair_causal,
                                window_size=pair_window_size,
                                lengths_q=varlen_lengths_cpu,
                                lengths_k=varlen_lengths_cpu,
                            )
                            pair_parts.append(f"{label}:{pairs}")
                    print(
                        "Varlen lengths: "
                        f"total_len={total_seqlen}, batch={int(varlen_lengths_cpu.numel())}, "
                        f"min={varlen_stats['min_seqlen']}, p50={varlen_stats['p50_seqlen']:.1f}, "
                        f"p90={varlen_stats['p90_seqlen']:.1f}, p99={varlen_stats['p99_seqlen']:.1f}, "
                        f"max={varlen_stats['max_seqlen']}, pct_lt_100={varlen_stats['pct_lt_100']:.3f}, "
                        f"pairs={{{', '.join(pair_parts)}}}"
                    )
                    seq_pairs = [(None, None)]
                else:
                    seqlen_list = resolve_seqlens(args, total_seqlen)
                    seqlen_q_list = resolve_seqlen_q_list(seqlen_list, args.seqlen_q)
                    seq_pairs = list(zip(seqlen_list, seqlen_q_list))
                for seq_len, seq_len_q in seq_pairs:
                    for headdim, headdim_v in headdim_pairs:
                        for nheads in resolve_nheads_list(args, headdim):
                            for nheads_kv in resolve_nheads_kv_list(args, nheads):
                                for window_label, window_size, window_size_fa in window_specs:
                                    for causal in causal_vals:
                                        for has_sink in sink_vals:
                                            planned_rows += len(active_backends) * len(directions)
                                            if args.dry_run:
                                                if varlen:
                                                    total_q = total_seqlen
                                                else:
                                                    batch_size = args.batch_size if args.batch_size is not None else max(1, total_seqlen // seq_len)
                                                    total_q = batch_size * seq_len_q
                                                meta = {
                                                    "total_len": total_q,
                                                    "head_dim": format_headdim(headdim, headdim_v),
                                                    "nheads": nheads,
                                                    "nheads_kv": nheads_kv,
                                                    "window": window_label,
                                                    "sink": has_sink,
                                                }
                                                for display_name, _, _ in active_backends:
                                                    for direction, _ in directions:
                                                        row = {field: "" for field in CSV_FIELDS}
                                                        row.update(meta)
                                                        row["backend"] = display_name
                                                        row["direction"] = direction
                                                        row["status"] = "dry_run"
                                                        emit_row(row)
                                                continue

                                            has_qv = headdim == 64 and headdim_v == 512
                                            num_splits = args.num_splits
                                            q = k = v = qv = g = None
                                            q_unpad = k_unpad = v_unpad = qv_unpad = g_unpad = None
                                            cu_seqlens_q = cu_seqlens_k = None
                                            lengths_q = lengths_k = None
                                            k_paged = v_paged = page_table = None
                                            gather_kv_indices = gather_kv_indices_unpad = None
                                            gather_kv_eff = None

                                            try:
                                                if varlen:
                                                    cu_seqlens_q = varlen_cu_seqlens_cpu.to(device=device)
                                                    lengths_q = varlen_lengths_cpu
                                                    if seq_len_q == seq_len:
                                                        lengths_k = lengths_q
                                                        cu_seqlens_k = cu_seqlens_q if page_size is None else None
                                                    else:
                                                        cu_seqlens_k_cpu, lengths_k = make_cu_seqlens_q(
                                                            total_len=total_seqlen,
                                                            device="cpu",
                                                            seed=456,
                                                            return_lengths=True,
                                                        )
                                                        cu_seqlens_k = cu_seqlens_k_cpu.to(device=device)
                                                    batch_size = int(lengths_q.numel())
                                                    max_seqlen_q = int(lengths_q.max().item())
                                                    max_seqlen_k = int(lengths_k.max().item())
                                                    min_seqlen_k = int(lengths_k.min().item())
                                                    total_q = total_seqlen
                                                    total_k = total_seqlen
                                                    q_unpad = torch.randn(total_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
                                                    k_unpad = torch.randn(total_k, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
                                                    v_unpad = torch.randn(total_k, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward)
                                                    qv_unpad = torch.randn(total_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward) if has_qv else None
                                                    g_unpad = torch.randn(total_q, nheads, headdim_v, device=device, dtype=dtype_gen)
                                                else:
                                                    batch_size = args.batch_size if args.batch_size is not None else max(1, total_seqlen // seq_len)
                                                    max_seqlen_q = seq_len_q
                                                    max_seqlen_k = seq_len
                                                    min_seqlen_k = seq_len
                                                    total_q = batch_size * seq_len_q
                                                    total_k = batch_size * seq_len
                                                    q = torch.randn(batch_size, seq_len_q, nheads, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
                                                    k = torch.randn(batch_size, seq_len, nheads_kv, headdim, device=device, dtype=dtype_gen, requires_grad=has_backward)
                                                    v = torch.randn(batch_size, seq_len, nheads_kv, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward)
                                                    qv = torch.randn(batch_size, seq_len_q, nheads, headdim_v, device=device, dtype=dtype_gen, requires_grad=has_backward) if has_qv else None
                                                    q, k, v, qv = [
                                                        x.detach().to(dtype).requires_grad_(has_backward) if x is not None else None
                                                        for x in [q, k, v, qv]
                                                    ]
                                                    g = torch.randn(batch_size, seq_len_q, nheads, headdim_v, device=device, dtype=dtype_gen)

                                                if page_size is not None:
                                                    assert seq_len % page_size == 0
                                                    k_paged, v_paged = [
                                                        rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size)
                                                        for x in [k, v]
                                                    ]
                                                    page_table = rearrange(
                                                        torch.arange(batch_size * seq_len // page_size, device=device, dtype=torch.int32),
                                                        "(b s) -> b s",
                                                        s=seq_len // page_size,
                                                    )

                                                if gather_kv_length is not None and has_qv:
                                                    gather_kv_limit = max_seqlen_k if varlen else seq_len
                                                    assert gather_kv_length <= gather_kv_limit, (
                                                        f"--gather_kv {gather_kv_length} > seqlen_kv {gather_kv_limit}"
                                                    )
                                                    if varlen:
                                                        gather_kv_indices_unpad = (
                                                            torch.rand(total_q, gather_kv_length, device=device)
                                                            .argsort(dim=-1)
                                                            .to(torch.int32)
                                                        )
                                                    else:
                                                        gather_kv_indices = (
                                                            torch.rand(batch_size, max_seqlen_q, gather_kv_length, device=device)
                                                            .argsort(dim=-1)
                                                            .to(torch.int32)
                                                        )
                                                    gather_kv_eff = gather_kv_length

                                                sink_tensor = torch.randn(
                                                    nheads,
                                                    device=device,
                                                    dtype=dtype_gen,
                                                    requires_grad=has_backward,
                                                ) if has_sink else None
                                                meta = {
                                                    "total_len": total_q,
                                                    "head_dim": format_headdim(headdim, headdim_v),
                                                    "nheads": nheads,
                                                    "nheads_kv": nheads_kv,
                                                    "window": window_label,
                                                    "sink": has_sink,
                                                }

                                                ctx = dict(
                                                    q=q, k=k, v=v, qv=qv, g=g, causal=causal,
                                                    headdim=headdim, headdim_v=headdim_v, dtype=dtype,
                                                    has_backward=has_backward,
                                                    varlen=varlen,
                                                    q_unpad=q_unpad, k_unpad=k_unpad, v_unpad=v_unpad,
                                                    qv_unpad=qv_unpad, g_unpad=g_unpad,
                                                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                                    seqlen_q=max_seqlen_q, seqlen=max_seqlen_k,
                                                    max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                                                    min_seqlen_k=min_seqlen_k,
                                                    page_size=page_size, k_paged=k_paged, v_paged=v_paged,
                                                    page_table=page_table,
                                                    dropout_p=dropout_p, window_size=window_size,
                                                    window_size_fa=window_size_fa,
                                                    softcap=softcap, deterministic=deterministic,
                                                    num_splits=num_splits, pack_gqa=pack_gqa,
                                                    sinks=sink_tensor,
                                                    gather_kv_indices=gather_kv_indices,
                                                    gather_kv_indices_unpad=gather_kv_indices_unpad,
                                                )

                                                seqlen_k_eff = gather_kv_eff if gather_kv_eff is not None else max_seqlen_k
                                                lengths_k_eff = None if gather_kv_eff is not None else lengths_k
                                                nflops = metric_flops(
                                                    batch_size,
                                                    nheads,
                                                    max_seqlen_q,
                                                    seqlen_k_eff,
                                                    headdim,
                                                    headdim_v,
                                                    causal=causal,
                                                    window_size=window_size,
                                                    has_qv=has_qv,
                                                    lengths_q=lengths_q,
                                                    lengths_k=lengths_k_eff,
                                                )

                                                for display_name, cli_name, setup_fn in active_backends:
                                                    fwd_fn = bwd_fn = None
                                                    setup_error = None
                                                    try:
                                                        fwd_fn, bwd_fn = setup_fn(ctx)
                                                    except Exception as exc:
                                                        setup_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                                                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                                                            torch.cuda.empty_cache()

                                                    for direction, flops_mult in directions:
                                                        row = {
                                                            field: ""
                                                            for field in CSV_FIELDS
                                                        }
                                                        row.update(meta)
                                                        row["backend"] = display_name
                                                        row["direction"] = direction

                                                        fn = fwd_fn if direction == "FWD" else bwd_fn
                                                        if setup_error is not None:
                                                            row["status"] = "error"
                                                            row["error"] = setup_error
                                                            emit_row(row)
                                                            continue
                                                        if fn is None:
                                                            row["status"] = "skipped"
                                                            row["error"] = "backend does not support this config or direction"
                                                            emit_row(row)
                                                            continue

                                                        try:
                                                            if args.sleep > 0:
                                                                time.sleep(args.sleep)
                                                            print(
                                                                f"Benchmarking {display_name} {direction.lower()} "
                                                                f"total={total_q} "
                                                                f"hdim={headdim} heads={nheads}/{nheads_kv} "
                                                                f"window={window_label} sink={has_sink}"
                                                            )
                                                            if direction == "FWD":
                                                                def bench_fn():
                                                                    with torch.no_grad():
                                                                        return fn()
                                                            else:
                                                                bench_fn = fn
                                                            bench_ms = do_bench(bench_fn, warmup=warmup, rep=rep)
                                                            elapsed_s = bench_ms * 1e-3
                                                            nbytes = metric_bandwidth_bytes(
                                                                total_q,
                                                                total_k if gather_kv_eff is None else batch_size * seqlen_k_eff,
                                                                nheads,
                                                                nheads_kv,
                                                                headdim,
                                                                headdim_v,
                                                                dtype_bytes,
                                                                direction,
                                                                has_qv=has_qv,
                                                            )
                                                            row["status"] = "ok"
                                                            row["ms"] = bench_ms
                                                            row["tflops"] = flops_mult * nflops / elapsed_s * 1e-12
                                                            row["tbs"] = nbytes / elapsed_s * 1e-12
                                                            row["mfu_pct"] = (
                                                                flops_mult * nflops / elapsed_s / peak_flops * 100
                                                                if peak_flops is not None
                                                                else ""
                                                            )
                                                            row["bw_pct"] = (
                                                                nbytes / elapsed_s / peak_bw * 100
                                                                if peak_bw is not None
                                                                else ""
                                                            )
                                                            emit_row(row)
                                                        except Exception as exc:
                                                            row["status"] = "error"
                                                            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                                                            emit_row(row)
                                                            if isinstance(exc, torch.cuda.OutOfMemoryError):
                                                                torch.cuda.empty_cache()
                                            except Exception as exc:
                                                error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                                                if isinstance(exc, torch.cuda.OutOfMemoryError):
                                                    torch.cuda.empty_cache()
                                                if varlen:
                                                    err_total = total_seqlen
                                                else:
                                                    err_batch_size = args.batch_size if args.batch_size is not None else max(1, total_seqlen // seq_len)
                                                    err_total = err_batch_size * seq_len_q
                                                err_meta = {
                                                    "total_len": err_total,
                                                    "head_dim": format_headdim(headdim, headdim_v),
                                                    "nheads": nheads,
                                                    "nheads_kv": nheads_kv,
                                                    "window": window_label,
                                                    "sink": has_sink,
                                                }
                                                for display_name, _, _ in active_backends:
                                                    for direction, _ in directions:
                                                        row = {field: "" for field in CSV_FIELDS}
                                                        row.update(err_meta)
                                                        row["backend"] = display_name
                                                        row["direction"] = direction
                                                        row["status"] = "error"
                                                        row["error"] = error
                                                        emit_row(row)
                                            finally:
                                                if args.empty_cache:
                                                    gc.collect()
                                                    if torch.cuda.is_available():
                                                        torch.cuda.empty_cache()

    finally:
        if csv_file is not None:
            csv_file.close()

    if args.dry_run:
        print(f"Dry run: {planned_rows} benchmark rows")
        if args.csv_out is not None:
            print(f"Dry-run CSV written to {args.csv_out}")
        return

    print(f"Completed benchmark rows: ok={ok_count}, skipped={skip_count}, dry_run={dry_count}, error={error_count}")
    if args.csv_out is not None:
        print(f"CSV written to {args.csv_out}")


if __name__ == '__main__':
    main()
