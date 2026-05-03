"""Inference runtime benchmark utility.

This script measures model forward latency for DOA models in this repository,
with reproducible protocol suitable for paper reporting:
  - fixed input tensor shape
  - warm-up iterations
  - repeated timed runs
  - latency summary (mean/std/p50/p95)

Examples (PowerShell):
  python validate.py --weights path\to\best_model.pth --device cuda --mode cov
  python validate.py --weights path\to\best_model.pth --device cpu --mode cov --batch_size 1
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from collections import defaultdict

import torch

from models.model_normal_mha import SVDNet, IterativeSVDNetWrapper


@dataclass
class RuntimeStats:
	device: str
	mode: str
	batch_size: int
	warmup: int
	runs: int
	mean_ms: float
	std_ms: float
	p50_ms: float
	p95_ms: float
	throughput_samples_per_s: float


@dataclass
class ModuleRuntimeStats:
	name: str
	mean_ms: float
	pct_total: float


class _ModuleRuntimeProfiler:
	"""Forward-hook based module runtime profiler for eager PyTorch models."""

	def __init__(self, model: torch.nn.Module, device: torch.device, module_names: list[str]):
		self.model = model
		self.device = device
		self.module_names = module_names
		self.handles = []
		self.cpu_start_stacks: dict[str, list[float]] = defaultdict(list)
		self.gpu_start_stacks: dict[str, list[torch.cuda.Event]] = defaultdict(list)
		self.gpu_pending_pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
		self.acc_ms: dict[str, float] = defaultdict(float)

	def _make_pre_hook(self, name: str):
		def _hook(_module, _inputs):
			if self.device.type == "cuda":
				st = torch.cuda.Event(enable_timing=True)
				st.record()
				self.gpu_start_stacks[name].append(st)
			else:
				self.cpu_start_stacks[name].append(time.perf_counter())
		return _hook

	def _make_post_hook(self, name: str):
		def _hook(_module, _inputs, _outputs):
			if self.device.type == "cuda":
				if not self.gpu_start_stacks[name]:
					return
				st = self.gpu_start_stacks[name].pop()
				ed = torch.cuda.Event(enable_timing=True)
				ed.record()
				self.gpu_pending_pairs.append((name, st, ed))
			else:
				if not self.cpu_start_stacks[name]:
					return
				t0 = self.cpu_start_stacks[name].pop()
				self.acc_ms[name] += (time.perf_counter() - t0) * 1000.0
		return _hook

	def attach(self):
		mods = dict(self.model.named_modules())
		for name in self.module_names:
			if name not in mods:
				continue
			m = mods[name]
			self.handles.append(m.register_forward_pre_hook(self._make_pre_hook(name)))
			self.handles.append(m.register_forward_hook(self._make_post_hook(name)))

	def step_finalize(self):
		if self.device.type != "cuda":
			return
		torch.cuda.synchronize(self.device)
		for name, st, ed in self.gpu_pending_pairs:
			self.acc_ms[name] += float(st.elapsed_time(ed))
		self.gpu_pending_pairs.clear()

	def detach(self):
		for h in self.handles:
			h.remove()
		self.handles.clear()

	def summary(self, runs: int, topk: int = 12) -> list[ModuleRuntimeStats]:
		rows = []
		total = sum(self.acc_ms.values()) + 1e-12
		for name, ms_total in self.acc_ms.items():
			rows.append(ModuleRuntimeStats(name=name, mean_ms=ms_total / max(1, runs), pct_total=100.0 * ms_total / total))
		rows.sort(key=lambda x: x.mean_ms, reverse=True)
		return rows[:topk]


def _extract_state_dict(ckpt_obj: Any) -> dict[str, torch.Tensor]:
	"""Best-effort extraction for different checkpoint formats."""
	if isinstance(ckpt_obj, dict):
		for k in ("state_dict", "model_state_dict", "model"):
			if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
				return ckpt_obj[k]
		# maybe already a raw state_dict
		if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
			return ckpt_obj
	raise RuntimeError("Unsupported checkpoint format. Expected raw state_dict or dict with 'state_dict'/'model_state_dict'.")


def _safe_get(state: dict[str, torch.Tensor], key: str, default: int) -> int:
	t = state.get(key)
	if t is None:
		return int(default)
	if t.ndim == 0:
		return int(t.item())
	return int(t.shape[0])


def _infer_arch_from_state_dict(state: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, int]:
	"""Infer some architecture fields from checkpoint tensors when possible."""
	# Fallbacks from CLI
	cfg = {
		"M": int(args.M),
		"N": int(args.N),
		"r": int(args.r),
		"dim": int(args.dim),
		"depth": int(args.depth),
		"groups": int(args.groups),
		"attn_dim": int(args.attn_dim),
		"doa_num_baselines": int(args.doa_num_baselines),
		"k_len": int(args.k_len),
		"gate_hidden": int(args.gate_hidden),
	}

	# Infer dim/N from input projection if available
	w_in = state.get("input_proj.weight")
	if w_in is not None and w_in.ndim == 2:
		cfg["dim"] = int(w_in.shape[0])
		cfg["N"] = int(w_in.shape[1] // 2)

	# Infer M from doa head output: Linear(..., 2*M)
	w_doa = state.get("doa_x_head.3.weight")
	if w_doa is not None and w_doa.ndim == 2:
		cfg["M"] = int(w_doa.shape[0] // 2)

	# Infer r from u_head output
	w_u = state.get("u_head.weight")
	if w_u is not None and w_u.ndim == 2:
		cfg["r"] = int(w_u.shape[0] // 2)

	# Infer depth from transformer/convolution blocks
	block_idx = []
	prefix = "blocks."
	for k in state.keys():
		if k.startswith(prefix):
			try:
				idx = int(k.split(".")[1])
				block_idx.append(idx)
			except Exception:
				pass
	if block_idx:
		cfg["depth"] = int(max(block_idx) + 1)

	# Infer groups/attn_dim/k_len from first grouped attention if present
	for g in range(1, 33):
		qk = f"blocks.0.1.q_proj.{g}.weight"
		if qk in state:
			cfg["groups"] = g
			cfg["attn_dim"] = int(state[qk].shape[0])
			bk = f"blocks.0.1.beta_k_proj.{g}.weight"
			if bk in state:
				cfg["k_len"] = int(state[bk].shape[0])
			break

	return cfg


def _build_model(args: argparse.Namespace, device: torch.device) -> SVDNet:
	if not args.weights:
		model = SVDNet(
			M=args.M,
			N=args.N,
			r=args.r,
			dim=args.dim,
			depth=args.depth,
			groups=args.groups,
			attn_dim=args.attn_dim,
			doa_num_baselines=args.doa_num_baselines,
			k_len=args.k_len,
			gate_hidden=args.gate_hidden,
			# refine_mode="both",
		)
		return model.to(device).eval()

	ckpt = torch.load(args.weights, map_location="cpu")
	state = _extract_state_dict(ckpt)
	cfg = _infer_arch_from_state_dict(state, args)

	model = SVDNet(
		M=cfg["M"],
		N=cfg["N"],
		r=cfg["r"],
		dim=cfg["dim"],
		depth=cfg["depth"],
		groups=cfg["groups"],
		attn_dim=cfg["attn_dim"],
		doa_num_baselines=cfg["doa_num_baselines"],
		k_len=cfg["k_len"],
		gate_hidden=cfg["gate_hidden"],
		refine_mode="both",
	)
	missing, unexpected = model.load_state_dict(state, strict=False)
	if missing:
		print(f"[warn] Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
	if unexpected:
		print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
	return model.to(device).eval()


def _make_input(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
	"""Create a fixed dummy input tensor for timing only."""
	b = int(args.batch_size)

	# model_副本.py expects covariance-like input shaped (B, N, N, 2)
	n = int(args.N)
	x = torch.randn((b, n, n, 2), dtype=torch.float32, device=device)
	return x


@torch.no_grad()
def _forward_call(model: torch.nn.Module, x: torch.Tensor, iterative_k: int | None = None):
	if iterative_k is None:
		return model(x)
	return model(x, K=int(iterative_k), return_residual=False)


@torch.no_grad()
def benchmark_runtime(
	model: torch.nn.Module,
	x: torch.Tensor,
	warmup: int,
	runs: int,
	device: torch.device,
	iterative_k: int | None = None,
) -> RuntimeStats:
	model.eval()

	# warm-up
	for _ in range(int(warmup)):
		_forward_call(model, x, iterative_k)
	if device.type == "cuda":
		torch.cuda.synchronize(device)

	times_ms: list[float] = []

	if device.type == "cuda":
		starter = torch.cuda.Event(enable_timing=True)
		ender = torch.cuda.Event(enable_timing=True)
		for _ in range(int(runs)):
			starter.record()
			_forward_call(model, x, iterative_k)
			ender.record()
			torch.cuda.synchronize(device)
			times_ms.append(float(starter.elapsed_time(ender)))
	else:
		for _ in range(int(runs)):
			t0 = time.perf_counter()
			_forward_call(model, x, iterative_k)
			t1 = time.perf_counter()
			times_ms.append((t1 - t0) * 1000.0)

	# per-sample latency
	bs = int(x.shape[0])
	per_sample = [t / bs for t in times_ms]
	per_sample_sorted = sorted(per_sample)

	def pct(v: list[float], p: float) -> float:
		if not v:
			return float("nan")
		idx = max(0, min(len(v) - 1, int(round((len(v) - 1) * p))))
		return float(v[idx])

	mean_ms = float(statistics.mean(per_sample))
	std_ms = float(statistics.pstdev(per_sample)) if len(per_sample) > 1 else 0.0
	p50_ms = pct(per_sample_sorted, 0.50)
	p95_ms = pct(per_sample_sorted, 0.95)
	throughput = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

	return RuntimeStats(
		device=str(device),
		mode="iterative" if iterative_k is not None else "single",
		batch_size=bs,
		warmup=int(warmup),
		runs=int(runs),
		mean_ms=mean_ms,
		std_ms=std_ms,
		p50_ms=p50_ms,
		p95_ms=p95_ms,
		throughput_samples_per_s=throughput,
	)


def _get_default_profile_module_names() -> list[str]:
	"""Hand-picked major blocks for your model_副本 SVDNet."""
	return [
		"spectral_refine",
		"spatial_refine",
		"input_proj",
		"blocks.0.1",
		"blocks.1.1",
		"norm",
		"u_head",
		"doa_feat_proj",
		"doa_token_to_complex",
		"doa_x_proj",
		"doa_x_conv",
		"doa_x_pool",
		"doa_x_head",
	]


@torch.no_grad()
def profile_module_runtime(
	model: torch.nn.Module,
	x: torch.Tensor,
	device: torch.device,
	warmup: int,
	runs: int,
	iterative_k: int | None,
	topk: int,
	module_names: list[str] | None = None,
) -> list[ModuleRuntimeStats]:
	model.eval()
	names = module_names or _get_default_profile_module_names()
	prof = _ModuleRuntimeProfiler(model=model, device=device, module_names=names)
	prof.attach()
	try:
		for _ in range(int(warmup)):
			_forward_call(model, x, iterative_k)
			prof.step_finalize()

		for _ in range(int(runs)):
			_forward_call(model, x, iterative_k)
			prof.step_finalize()
	finally:
		prof.detach()

	return prof.summary(runs=int(runs), topk=int(topk))


def _print_module_profile(rows: list[ModuleRuntimeStats]) -> None:
	if not rows:
		print("[profile] No module timing rows collected.")
		return
	print("\n==== Module Runtime Breakdown (Top) ====")
	print(f"{'Module':40s} {'Mean ms':>10s} {'Share %':>10s}")
	print("-" * 64)
	for r in rows:
		print(f"{r.name:40s} {r.mean_ms:10.4f} {r.pct_total:10.2f}")
	print("=" * 64)


@torch.no_grad()
def analyze_refine_modes(
	model: torch.nn.Module,
	x: torch.Tensor,
	device: torch.device,
	warmup: int,
	runs: int,
	iterative_k: int | None,
) -> dict[str, RuntimeStats]:
	if not hasattr(model, "set_refine_mode"):
		return {}

	results: dict[str, RuntimeStats] = {}
	for mode in ["both", "spectral_only", "spatial_only", "none"]:
		try:
			model.set_refine_mode(mode)
		except Exception:
			continue
		stats = benchmark_runtime(
			model=model,
			x=x,
			warmup=warmup,
			runs=runs,
			device=device,
			iterative_k=iterative_k,
		)
		results[mode] = stats
	return results


def _print_refine_mode_results(results: dict[str, RuntimeStats]) -> None:
	if not results:
		print("[ablation] refine_mode analysis unavailable for this model.")
		return
	print("\n==== Refine-Mode Latency Ablation ====")
	print(f"{'mode':16s} {'mean(ms)':>10s} {'std(ms)':>10s} {'p95(ms)':>10s}")
	print("-" * 52)
	for mode in ["both", "spectral_only", "spatial_only", "none"]:
		if mode not in results:
			continue
		s = results[mode]
		print(f"{mode:16s} {s.mean_ms:10.4f} {s.std_ms:10.4f} {s.p95_ms:10.4f}")
	print("=" * 52)


def _print_report(stats: RuntimeStats, args: argparse.Namespace) -> None:
	print("\n==== Inference Runtime Benchmark ====")
	print(f"Platform: {platform.platform()}")
	print(f"PyTorch : {torch.__version__}")
	print(f"Device  : {stats.device}")
	print(f"Input   : mode={args.mode}, batch={args.batch_size}, shape={args.input_shape_str}")
	print(f"Warmup/Runs: {stats.warmup}/{stats.runs}")
	print("-------------------------------------")
	print(f"Latency mean (ms/sample): {stats.mean_ms:.4f}")
	print(f"Latency std  (ms/sample): {stats.std_ms:.4f}")
	print(f"Latency p50  (ms/sample): {stats.p50_ms:.4f}")
	print(f"Latency p95  (ms/sample): {stats.p95_ms:.4f}")
	print(f"Throughput   (sample/s) : {stats.throughput_samples_per_s:.2f}")
	print("=====================================\n")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Inference runtime benchmark for DOA model.")

	# IO
	p.add_argument("--weights", type=str, default="", help="Path to checkpoint (.pth). If empty, use random initialized model.")
	p.add_argument("--save_json", type=str, default="", help="Optional path to save runtime stats JSON.")

	# device
	p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"], help="Benchmark device.")

	# model/input mode (kept for report compatibility; current model uses covariance input)
	p.add_argument("--mode", type=str, default="cov", choices=["cov"], help="Input type for benchmark report.")
	p.add_argument("--iterative", action="store_true", help="Use IterativeSVDNetWrapper for timing.")
	p.add_argument("--iter_k", type=int, default=1, help="K for iterative wrapper (valid with --iterative).")

	# architecture fallback (if cannot infer from checkpoint)
	p.add_argument("--M", type=int, default=4)
	p.add_argument("--N", type=int, default=8)
	p.add_argument("--r", type=int, default=1)
	p.add_argument("--dim", type=int, default=128)
	p.add_argument("--depth", type=int, default=2)
	p.add_argument("--groups", type=int, default=4)
	p.add_argument("--attn_dim", type=int, default=64)
	p.add_argument("--doa_num_baselines", type=int, default=1)
	p.add_argument("--k_len", type=int, default=8)
	p.add_argument("--gate_hidden", type=int, default=32)
	p.add_argument("--snap_T", type=int, default=2, help="Reserved; current model benchmark uses covariance input.")
	p.add_argument("--T", type=int, default=2, help="Scenario metadata only (snapshot count in data generation).")
	p.add_argument("--coherent", action="store_true", help="Scenario metadata only (coherent-source setting).")

	# benchmark protocol
	p.add_argument("--batch_size", type=int, default=1)
	p.add_argument("--warmup", type=int, default=100)
	p.add_argument("--runs", type=int, default=500)
	p.add_argument("--profile_modules", action="store_true", help="Profile per-module forward runtime (top hotspots).")
	p.add_argument("--profile_topk", type=int, default=12, help="Top-K modules to show in module profile.")
	p.add_argument("--analyze_refine", action="store_true", help="Run refine_mode latency ablation: both/spectral_only/spatial_only/none.")
	p.add_argument("--analyze_warmup", type=int, default=30, help="Warmup for refine-mode latency ablation.")
	p.add_argument("--analyze_runs", type=int, default=100, help="Runs for refine-mode latency ablation.")

	return p.parse_args()


def main() -> None:
	args = parse_args()

	# select device
	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("--device cuda was requested, but CUDA is not available.")

	model = _build_model(args, device)
	if args.iterative:
		model_bench: torch.nn.Module = IterativeSVDNetWrapper(model).to(device).eval()
	else:
		model_bench = model

	x = _make_input(args, device)
	if args.mode == "cov":
		args.input_shape_str = f"(B={args.batch_size}, N={x.shape[1]}, N={x.shape[2]}, 2)"
	else:
		args.input_shape_str = f"(B={args.batch_size}, T={x.shape[1]}, N={x.shape[2]}, 2)"

	stats = benchmark_runtime(
		model=model_bench,
		x=x,
		warmup=args.warmup,
		runs=args.runs,
		device=device,
		iterative_k=args.iter_k if args.iterative else None,
	)
	_print_report(stats, args)

	if args.profile_modules:
		rows = profile_module_runtime(
			model=model_bench,
			x=x,
			device=device,
			warmup=args.warmup,
			runs=args.runs,
			iterative_k=args.iter_k if args.iterative else None,
			topk=args.profile_topk,
		)
		_print_module_profile(rows)

	if args.analyze_refine:
		# For fair refine analysis, use base model only (not wrapper).
		refine_results = analyze_refine_modes(
			model=model,
			x=x,
			device=device,
			warmup=args.analyze_warmup,
			runs=args.analyze_runs,
			iterative_k=None,
		)
		_print_refine_mode_results(refine_results)

	if args.save_json:
		out = {
			"runtime": asdict(stats),
			"config": {
				"weights": args.weights,
				"device": str(device),
				"mode": args.mode,
				"iterative": bool(args.iterative),
				"iter_k": int(args.iter_k),
				"M": int(args.M),
				"N": int(args.N),
				"r": int(args.r),
				"batch_size": int(args.batch_size),
				"warmup": int(args.warmup),
				"runs": int(args.runs),
				"scenario_T": int(args.T),
				"scenario_M": int(args.M),
				"scenario_N": int(args.N),
				"scenario_r": int(args.r),
				"scenario_coherent": bool(args.coherent),
				"torch": torch.__version__,
				"platform": platform.platform(),
			},
		}
		out_path = Path(args.save_json)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
	main()
