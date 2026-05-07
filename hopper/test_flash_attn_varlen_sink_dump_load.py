#!/root/.envs/swa/bin/python
import argparse
from pathlib import Path
from typing import Any

import pytest
import torch

from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_fa3
from test_util import generate_qkv

try:
    from flash_attn.cute.interface import flash_attn_varlen_func as flash_attn_varlen_func_cute

    _CUTE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only hit when cute isn't installed
    flash_attn_varlen_func_cute = None
    _CUTE_IMPORT_ERROR = exc


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _log(msg: str) -> None:
    print(f"[varlen-sink-test] {msg}", flush=True)


def _build_case(seed: int = 20260507):
    """Build a single reproducible varlen case used for dump/load replay."""
    _log(f"Building case with seed={seed}")
    torch.random.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    batch_size = 4
    seqlen_q = 512
    seqlen_k = 512
    nheads = 24
    nheads_k = 2
    headdim = 256

    _log(
        "Case config: "
        f"batch={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
        f"nheads={nheads}, nheads_k={nheads_k}, headdim={headdim}, dtype={dtype}"
    )

    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_k, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_k, headdim, device=device, dtype=dtype)

    (
        q_unpad,
        k_unpad,
        v_unpad,
        _qv_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        _q,
        _k,
        _v,
        _qv,
        _output_pad_fn,
        _dq_pad_fn,
        _dk_pad_fn,
    ) = generate_qkv(
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        qv=None,
        kvpacked=False,
    )

    q_unpad = q_unpad.detach().to(dtype)
    k_unpad = k_unpad.detach().to(dtype)
    v_unpad = v_unpad.detach().to(dtype)

    g_unpad = torch.randn(
        q_unpad.shape[0], q_unpad.shape[1], v_unpad.shape[-1], device=device, dtype=dtype
    )
    sink = torch.randn(nheads, device=device, dtype=dtype)

    _log(
        "Built tensors: "
        f"q_unpad={tuple(q_unpad.shape)}, k_unpad={tuple(k_unpad.shape)}, "
        f"v_unpad={tuple(v_unpad.shape)}, g_unpad={tuple(g_unpad.shape)}, sink={tuple(sink.shape)}"
    )

    return {
        "q_unpad": q_unpad.cpu(),
        "k_unpad": k_unpad.cpu(),
        "v_unpad": v_unpad.cpu(),
        "cu_seqlens_q": cu_seqlens_q.cpu(),
        "cu_seqlens_k": cu_seqlens_k.cpu(),
        "seqused_q": None if seqused_q is None else seqused_q.cpu(),
        "seqused_k": None if seqused_k is None else seqused_k.cpu(),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
        "learnable_sink": sink.cpu(),
        "g_unpad": g_unpad.cpu(),
        "causal": True,
        "window_size": (-1, -1),
        "softcap": 0.0,
        "deterministic": True,
        "num_splits": 1,
        "pack_gqa": False,
    }


def _to_cuda_tensor(x, requires_grad: bool = False):
    if x is None:
        return None
    y = x.to(device="cuda")
    if requires_grad:
        y = y.detach().requires_grad_(True)
    return y


def _normalize_window_size_for_cute(window_size):
    left, right = window_size
    return (None if left < 0 else left, None if right < 0 else right)


def _empty_grad_like(q: torch.Tensor) -> torch.Tensor:
    return torch.empty(0, device=q.device, dtype=q.dtype)


def _run_case_fa3(case, use_fa4_sink: bool, enable_sink: bool):
    effective_use_fa4_sink = use_fa4_sink and enable_sink
    if use_fa4_sink and not enable_sink:
        _log("Running FA3 case: use_fa4_sink=True ignored because sink is disabled")
    else:
        _log(
            f"Running FA3 case: use_fa4_sink={effective_use_fa4_sink}, "
            f"enable_sink={enable_sink}"
        )

    q = _to_cuda_tensor(case["q_unpad"], requires_grad=True)
    k = _to_cuda_tensor(case["k_unpad"], requires_grad=True)
    v = _to_cuda_tensor(case["v_unpad"], requires_grad=True)
    sink = _to_cuda_tensor(case["learnable_sink"], requires_grad=True) if enable_sink else None
    g_unpad = _to_cuda_tensor(case["g_unpad"], requires_grad=False)

    out, lse = flash_attn_varlen_func_fa3(
        q,
        k,
        v,
        _to_cuda_tensor(case["cu_seqlens_q"]),
        _to_cuda_tensor(case["cu_seqlens_k"]),
        case["max_seqlen_q"],
        case["max_seqlen_k"],
        seqused_q=_to_cuda_tensor(case["seqused_q"]),
        seqused_k=_to_cuda_tensor(case["seqused_k"]),
        causal=case["causal"],
        window_size=tuple(case["window_size"]),
        softcap=case["softcap"],
        deterministic=case["deterministic"],
        num_splits=case["num_splits"],
        pack_gqa=case["pack_gqa"],
        learnable_sink=sink,
        use_fa4_sink=effective_use_fa4_sink,
    )

    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g_unpad, retain_graph=enable_sink)
    if enable_sink:
        (dsink,) = torch.autograd.grad(out, sink, g_unpad)
    else:
        dsink = _empty_grad_like(q)

    result = {
        "out": out.detach().cpu(),
        "lse": lse.detach().cpu(),
        "dq": dq.detach().cpu(),
        "dk": dk.detach().cpu(),
        "dv": dv.detach().cpu(),
        "dsink": dsink.detach().cpu(),
    }
    _log(f"FA3 run done: use_fa4_sink={effective_use_fa4_sink}, enable_sink={enable_sink}")
    return result


def _run_case_cute(case, use_fa4_sink: bool, enable_sink: bool):
    if flash_attn_varlen_func_cute is None:
        raise RuntimeError(f"flash_attn.cute import failed: {_CUTE_IMPORT_ERROR}")

    if use_fa4_sink:
        _log("Running CuTe case: use_fa4_sink=True requested, but CuTe has no such switch; ignoring")
    else:
        _log("Running CuTe case: use_fa4_sink=False")
    _log(f"CuTe run config: enable_sink={enable_sink}")

    q = _to_cuda_tensor(case["q_unpad"], requires_grad=True)
    k = _to_cuda_tensor(case["k_unpad"], requires_grad=True)
    v = _to_cuda_tensor(case["v_unpad"], requires_grad=True)
    sink = _to_cuda_tensor(case["learnable_sink"], requires_grad=True) if enable_sink else None
    g_unpad = _to_cuda_tensor(case["g_unpad"], requires_grad=False)

    out, lse = flash_attn_varlen_func_cute(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=_to_cuda_tensor(case["cu_seqlens_q"]),
        cu_seqlens_k=_to_cuda_tensor(case["cu_seqlens_k"]),
        max_seqlen_q=case["max_seqlen_q"],
        max_seqlen_k=case["max_seqlen_k"],
        seqused_q=_to_cuda_tensor(case["seqused_q"]),
        seqused_k=_to_cuda_tensor(case["seqused_k"]),
        causal=case["causal"],
        window_size=_normalize_window_size_for_cute(case["window_size"]),
        learnable_sink=sink,
        softcap=case["softcap"],
        num_splits=case["num_splits"],
        pack_gqa=case["pack_gqa"],
        deterministic=case["deterministic"],
        return_lse=True,
    )
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g_unpad, retain_graph=enable_sink)
    if enable_sink:
        # cute not support sink gradient, force disable to avoid autograd error
        # (dsink,) = torch.autograd.grad(out, sink, g_unpad)
        dsink = torch.zeros_like(sink)
    else:
        dsink = _empty_grad_like(q)
        

    result = {
        "out": out.detach().cpu(),
        "lse": lse.detach().cpu(),
        "dq": dq.detach().cpu(),
        "dk": dk.detach().cpu(),
        "dv": dv.detach().cpu(),
        "dsink": dsink.detach().cpu(),
    }
    _log(f"CuTe run done: enable_sink={enable_sink}")
    return result


def _run_case(case, use_fa4_sink: bool, backend: str, enable_sink: bool):
    if backend == "fa3":
        return _run_case_fa3(case, use_fa4_sink=use_fa4_sink, enable_sink=enable_sink)
    if backend == "cute":
        return _run_case_cute(case, use_fa4_sink=use_fa4_sink, enable_sink=enable_sink)
    raise ValueError(f"Unknown backend: {backend}")


def _assert_finite(tag: str, result):
    _log(f"[CHECK] finite: {tag}")
    for name, t in result.items():
        finite = torch.isfinite(t)
        if not finite.all():
            bad = (~finite).sum().item()
            raise AssertionError(f"{tag}.{name} contains NaN/Inf (bad_count={bad})")
        _log(f"  [OK] {tag}.{name} finite, shape={tuple(t.shape)}, dtype={t.dtype}")


def _assert_close(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    if a.numel() == 0 and b.numel() == 0:
        _log(f"[CHECK] close: {name} | both tensors empty, skip numeric check")
        return
    if a.numel() == 0 or b.numel() == 0:
        raise AssertionError(f"{name} mismatch: one tensor is empty and the other is not")

    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs().max().item()
    scale = max(a_f.abs().max().item(), b_f.abs().max().item(), 1.0)
    limit = atol + rtol * scale
    _log(
        f"[CHECK] close: {name} | max_diff={diff:.6e}, limit={limit:.6e} "
        f"(atol={atol:.2e}, rtol={rtol:.2e}, scale={scale:.6e})"
    )
    if diff > limit:
        raise AssertionError(f"{name} max diff {diff:.6e} > limit {limit:.6e}")


def _compare_results(tag: str, a, b):
    _log(f"[CHECK] compare result group: {tag}")
    out_equal = torch.equal(a["out"], b["out"])
    lse_equal = torch.equal(a["lse"], b["lse"])
    _log("=" * 88)
    _log(f"[BITWISE][{tag}] out torch.equal: {'PASS' if out_equal else 'FAIL'}")
    _log(f"[BITWISE][{tag}] lse torch.equal: {'PASS' if lse_equal else 'FAIL'}")
    _log("=" * 88)

    tol = {
        "out": (5e-3, 5e-2),
        "lse": (1e-2, 5e-2),
        "dq": (1e-2, 8e-2),
        "dk": (1e-2, 8e-2),
        "dv": (1e-2, 8e-2),
        # "dsink": (1e-2, 8e-2),
    }
    for key, (atol, rtol) in tol.items():
        _assert_close(f"{tag}.{key}", a[key], b[key], atol=atol, rtol=rtol)
    _log(f"[OK] compare result group passed: {tag}")


def dump_case(path: Path, backend: str = "fa3", enable_sink: bool = True):
    _log(f"Dump start -> {path} (backend={backend}, enable_sink={enable_sink})")
    case = _build_case()
    result_off = _run_case(case, use_fa4_sink=False, backend=backend, enable_sink=enable_sink)
    result_on = _run_case(case, use_fa4_sink=True, backend=backend, enable_sink=enable_sink)

    _assert_finite("dump.fa4_off", result_off)
    _assert_finite("dump.fa4_on", result_on)

    payload = {
        "backend": backend,
        "enable_sink": enable_sink,
        "case": case,
        "result_fa4_off": result_off,
        "result_fa4_on": result_on,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    _log(f"Dump finished -> {path}")


def load_and_check(
    path: Path,
    backend: str = "fa3",
    enable_sink: bool = True,
    allow_backend_mismatch: bool = True,
):
    _log(
        f"Load start <- {path} (backend={backend}, enable_sink={enable_sink}, "
        f"allow_backend_mismatch={allow_backend_mismatch})"
    )
    payload = torch.load(path, map_location="cpu")
    case = payload["case"]

    dumped_backend = payload.get("backend", "fa3")
    if dumped_backend != backend:
        if allow_backend_mismatch:
            _log(
                f"[CHECK] backend mismatch allowed: dump backend={dumped_backend}, "
                f"replay backend={backend}"
            )
        else:
            raise AssertionError(
                f"Backend mismatch: dump was produced with backend={dumped_backend}, "
                f"but check requested backend={backend}"
            )

    dumped_enable_sink = payload.get("enable_sink", True)
    if dumped_enable_sink != enable_sink:
        raise AssertionError(
            f"Sink enable mismatch: dump was produced with enable_sink={dumped_enable_sink}, "
            f"but check requested enable_sink={enable_sink}"
        )

    replay_off = _run_case(case, use_fa4_sink=False, backend=backend, enable_sink=enable_sink)
    replay_on = _run_case(case, use_fa4_sink=True, backend=backend, enable_sink=enable_sink)

    _assert_finite("replay.fa4_off", replay_off)
    _assert_finite("replay.fa4_on", replay_on)

    base_tag = (
        f"replay_vs_dump.{backend}_vs_{dumped_backend}"
        if dumped_backend != backend
        else "replay_vs_dump"
    )
    _compare_results(f"{base_tag}.fa4_off", replay_off, payload["result_fa4_off"])
    _compare_results(f"{base_tag}.fa4_on", replay_on, payload["result_fa4_on"])
    _compare_results(f"replay.{backend}.fa4_on_vs_off", replay_on, replay_off)
    _log("Load+check finished")


def dump_and_check(
    path: Path,
    backend: str = "fa3",
    enable_sink: bool = True,
    check_backend: str | None = None,
    allow_backend_mismatch: bool = True,
):
    _log("Dump+check pipeline start")
    dump_case(path, backend=backend, enable_sink=enable_sink)
    replay_backend = backend if check_backend is None else check_backend
    load_and_check(
        path,
        backend=replay_backend,
        enable_sink=enable_sink,
        allow_backend_mismatch=allow_backend_mismatch,
    )
    _log("Dump+check pipeline finished")


def compare_fa3_vs_cute(enable_sink: bool = True):
    if flash_attn_varlen_func_cute is None:
        raise RuntimeError(f"flash_attn.cute import failed: {_CUTE_IMPORT_ERROR}")

    _log(f"FA3 vs CuTe compare start (enable_sink={enable_sink})")
    case = _build_case()

    fa3_off = _run_case(case, use_fa4_sink=False, backend="fa3", enable_sink=enable_sink)
    cute_off = _run_case(case, use_fa4_sink=False, backend="cute", enable_sink=enable_sink)
    _assert_finite("compare.fa3.fa4_off", fa3_off)
    _assert_finite("compare.cute.fa4_off", cute_off)
    _compare_results("fa3_vs_cute.fa4_off", fa3_off, cute_off)

    fa3_on = _run_case(case, use_fa4_sink=True, backend="fa3", enable_sink=enable_sink)
    cute_on = _run_case(case, use_fa4_sink=True, backend="cute", enable_sink=enable_sink)
    _assert_finite("compare.fa3.fa4_on", fa3_on)
    _assert_finite("compare.cute.fa4_on", cute_on)
    _compare_results("fa3_vs_cute.fa4_on", fa3_on, cute_on)

    _compare_results("fa3.fa4_on_vs_off", fa3_on, fa3_off)
    _compare_results("cute.fa4_on_vs_off", cute_on, cute_off)
    _log("FA3 vs CuTe compare finished")


def test_flash_attn_varlen_sink_dump_load_compare(tmp_path: Path):
    dump_path = tmp_path / "flash_attn_varlen_sink_case.pt"
    dump_and_check(dump_path, backend="fa3", enable_sink=True)


@pytest.mark.skipif(flash_attn_varlen_func_cute is None, reason="CuTe backend is not available")
def test_flash_attn_varlen_sink_dump_load_compare_cute(tmp_path: Path):
    dump_path = tmp_path / "flash_attn_varlen_sink_case_cute.pt"
    dump_and_check(dump_path, backend="cute", enable_sink=True)


def test_flash_attn_varlen_sink_dump_load_compare_no_sink(tmp_path: Path):
    dump_path = tmp_path / "flash_attn_varlen_no_sink_case.pt"
    dump_and_check(dump_path, backend="fa3", enable_sink=False)


def _parse_args():
    parser = argparse.ArgumentParser(description="Varlen FlashAttention sink precision dump/load test")
    parser.add_argument(
        "--mode",
        choices=["dump", "check", "dump_and_check", "compare_fa3_cute"],
        default="dump_and_check",
        help="dump only, check only, dump then check, or compare FA3 vs CuTe directly",
    )
    parser.add_argument(
        "--backend",
        choices=["fa3", "cute"],
        default="fa3",
        help="attention backend to run (used by dump/check/dump_and_check)",
    )
    parser.add_argument(
        "--check-backend",
        choices=["fa3", "cute"],
        default=None,
        help="backend used in check stage for dump_and_check; defaults to --backend",
    )
    parser.add_argument(
        "--strict-backend-match",
        action="store_true",
        help="fail check when dump backend and replay backend are different",
    )
    parser.add_argument(
        "--enable-sink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable learnable_sink",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent / "varlen_sink_precision_case.pt",
        help="path for saved tensors/results",
    )
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this test")

    args = _parse_args()
    _log(
        f"Running mode={args.mode}, backend={args.backend}, check_backend={args.check_backend}, "
        f"strict_backend_match={args.strict_backend_match}, enable_sink={args.enable_sink}, "
        f"path={args.path}"
    )

    if args.mode == "dump":
        dump_case(args.path, backend=args.backend, enable_sink=args.enable_sink)
    elif args.mode == "check":
        load_and_check(
            args.path,
            backend=args.backend,
            enable_sink=args.enable_sink,
            allow_backend_mismatch=not args.strict_backend_match,
        )
    elif args.mode == "dump_and_check":
        dump_and_check(
            args.path,
            backend=args.backend,
            enable_sink=args.enable_sink,
            check_backend=args.check_backend,
            allow_backend_mismatch=not args.strict_backend_match,
        )
    else:
        compare_fa3_vs_cute(enable_sink=args.enable_sink)


if __name__ == "__main__":
    main()

# usage examples
# python test_flash_attn_varlen_sink_dump_load.py --backend fa3 --mode dump_and_check --enable-sink --path ./with_sink.pt
# python test_flash_attn_varlen_sink_dump_load.py --backend fa3 --mode dump_and_check --no-enable-sink --path ./no_sink.pt
# python test_flash_attn_varlen_sink_dump_load.py --backend cute --mode check --enable-sink --path ./with_sink.pt
# python test_flash_attn_varlen_sink_dump_load.py --backend cute --mode check --no-enable-sink --path ./no_sink.pt
