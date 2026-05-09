"""End-to-end launcher for DDCR-DoA experiments.

The launcher can regenerate synthetic data, preprocess covariance matrices,
run stage-1 geometric pretraining, and run stage-2 DoA fine-tuning.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_suffix_from_existing_files(root: Path, train_mod, args) -> str:
    """Resolve dataset suffix robustly (e.g., SNR=7 vs 7.0).

    We prefer the suffix that matches an existing processed file:
      datasets/processed_cov_svd/iter_cov_noisy_step{suffix}.npy
    """

    params = train_mod._build_params(
        N=int(args.N),
        M=int(args.M),
        T=int(args.T),
        snr=float(args.snr),
        signal_type=str(args.signal_type),
        signal_nature=str(args.signal_nature),
        eta=float(args.eta),
        bias=float(args.bias),
        sv_noise_var=float(args.sv_noise_var),
        doa_gap=float(args.doa_gap),
    )
    suffix_default = train_mod._build_suffix(params, int(args.T))

    try:
        snr_val = float(args.snr)
    except Exception:
        return suffix_default

    if not float(snr_val).is_integer():
        return suffix_default

    params_int = train_mod._build_params(
        N=int(args.N),
        M=int(args.M),
        T=int(args.T),
        snr=int(snr_val),
        signal_type=str(args.signal_type),
        signal_nature=str(args.signal_nature),
        eta=float(args.eta),
        bias=float(args.bias),
        sv_noise_var=float(args.sv_noise_var),
        doa_gap=float(args.doa_gap),
    )
    params_float = train_mod._build_params(
        N=int(args.N),
        M=int(args.M),
        T=int(args.T),
        snr=float(snr_val),
        signal_type=str(args.signal_type),
        signal_nature=str(args.signal_nature),
        eta=float(args.eta),
        bias=float(args.bias),
        sv_noise_var=float(args.sv_noise_var),
        doa_gap=float(args.doa_gap),
    )
    suffix_int = train_mod._build_suffix(params_int, int(args.T))
    suffix_float = train_mod._build_suffix(params_float, int(args.T))

    # Probe existing processed file.
    for s in (suffix_int, suffix_float):
        probe = root / "datasets" / "processed_cov_svd" / f"iter_cov_noisy_step{s}.npy"
        if probe.exists():
            return s

    return suffix_default


def _find_latest_stage1_ckpt(root: Path) -> Path | None:
    """Fallback: find the newest model_trained_stage1.pth under datasets/weights."""
    weights_root = root / "datasets" / "weights"
    if not weights_root.exists():
        return None
    candidates = list(weights_root.glob("model*/model_trained_stage1.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_parser(snr = -3, M = 2, signal_nature = "non-coherent", T = 200, doa_gap = 15.0) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run 2-stage training automatically (stage1 -> stage2).")

    # Dataset / suffix-identifying args (must match how train.py builds filenames)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--M", type=int, default=M)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--T", type=int, default=T)
    p.add_argument("--snr", type=float, default=snr)
    p.add_argument("--signal_type", type=str, default="NarrowBand")
    p.add_argument("--signal_nature", type=str, default=signal_nature)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--sv_noise_var", type=float, default=0.0)
    p.add_argument("--doa_gap", type=float, default=doa_gap)

    # Data split (forwarded)
    p.add_argument("--train_ratio", type=float, default=0.9)

    # Optional training knobs to forward
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    return p


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _clear_directory_contents(dir_path: Path) -> None:
    """Remove all children under dir_path (files + folders) robustly."""
    if not dir_path.exists():
        return

    for child in dir_path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError as e:
            print(f"[WARN] Skip locked path: {child} ({e})")
        except FileNotFoundError:
            # Path may disappear during concurrent cleanup; safe to ignore.
            pass


def main(snr: int, M: int, signal_nature: str, T: int, doa_gap = 15.0) -> None:
    create_data = True
    Delete_folder = True
    Train_model = True

    root = Path(__file__).resolve().parent
    data_pipeline_dir = root / "data_pipeline"
    train_py = root / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"train.py not found next to this file: {train_py}")

    args = _build_parser(snr=snr, M=M, signal_nature=signal_nature, T=T, doa_gap=doa_gap).parse_args()

    # Clean generated/processed data for a fresh run.
    if Delete_folder:
        for subdir in ["generated_snapshots", "processed_cov_svd"]:
            dir_path = root / "datasets" / subdir
            _clear_directory_contents(dir_path)

    # Generate and process data before training
    # NOTE: Data scripts have their own defaults in __main__; we always forward
    # args here to make sure dataset suffix (especially SNR) is consistent.
    if create_data:
        fgen_cmd = [
            sys.executable,
            str(data_pipeline_dir / "f_gendata.py"),
            "--N",
            str(args.N),
            "--M",
            str(args.M),
            "--T",
            str(args.T),
            "--snr",
            str(args.snr),
            "--signal_type",
            str(args.signal_type),
            "--signal_nature",
            str(args.signal_nature),
            "--eta",
            str(args.eta),
            "--bias",
            str(args.bias),
            "--sv_noise_var",
            str(args.sv_noise_var),
            "--doa_gap",
            str(args.doa_gap),
        ]
        _run_cmd(fgen_cmd, cwd=root)

        proc_cmd = [
            sys.executable,
            str(data_pipeline_dir / "cov_svd_processing.py"),
            "--N",
            str(args.N),
            "--M",
            str(args.M),
            "--T",
            str(args.T),
            "--snr",
            str(args.snr),
            "--signal_type",
            str(args.signal_type),
            "--signal_nature",
            str(args.signal_nature),
            "--eta",
            str(args.eta),
            "--bias",
            str(args.bias),
            "--sv_noise_var",
            str(args.sv_noise_var),
            "--doa_gap",
            str(args.doa_gap),
            "--rank",
            str(args.M),
        ]
        _run_cmd(proc_cmd, cwd=root)

    if Train_model:
        # Import train.py to compute suffix consistently
        sys.path.insert(0, str(root))
        import train as train_mod  # type: ignore

        suffix = _resolve_suffix_from_existing_files(root, train_mod, args)
        weights_dir = root / "datasets" / "weights" / f"model{suffix}"
        stage1_ckpt = weights_dir / "model_trained_stage1.pth"

        common_forward = [
            "--N",
            str(args.N),
            "--M",
            str(args.M),
            "--rank",
            str(args.rank),
            "--T",
            str(args.T),
            "--snr",
            str(args.snr),
            "--signal_type",
            str(args.signal_type),
            "--signal_nature",
            str(args.signal_nature),
            "--eta",
            str(args.eta),
            "--bias",
            str(args.bias),
            "--sv_noise_var",
            str(args.sv_noise_var),
            "--doa_gap",
            str(args.doa_gap),
            "--train_ratio",
            str(args.train_ratio),
        ]

        if args.batch_size is not None:
            common_forward += ["--batch_size", str(args.batch_size)]
        if args.weight_decay is not None:
            common_forward += ["--weight_decay", str(args.weight_decay)]

        # ---------------- Stage 1 ----------------
        stage1 = [
            sys.executable,
            str(train_py),
            "--phase",
            "1",
            "--model_tag",
            "stage1",
            "--lambda_dir",
            "1.0",
            "--lambda_doa",
            "0.0",
            "--lambda_fft_recon",
            "0.0",
            "--lambda_phase_recon",
            "0.0",
            "--base_lr",
            "1e-3",
            "--max_lr",
            "5e-3",
            "--min_lr",
            "1e-4",
            "--warmup",
            "30",
            "--num_epochs",
            "20",
            "--num_sched_epochs",
            "180",
        ] + common_forward

        _run_cmd(stage1, cwd=root)

        if not stage1_ckpt.exists():
            latest = _find_latest_stage1_ckpt(root)
            if latest is None:
                raise FileNotFoundError(
                    f"Stage-1 checkpoint not found after training: {stage1_ckpt}. "
                    "Check train.py output for errors or whether it saved any best model."
                )
            stage1_ckpt = latest
            weights_dir = stage1_ckpt.parent
            print(f"[WARN] Stage-1 ckpt not found at expected path; using latest: {stage1_ckpt}")

        # ---------------- Stage 2 ----------------
        stage2 = [
            sys.executable,
            str(train_py),
            "--phase",
            "1",
            "--model_tag",
            "stage2",
            "--pretrained_path",
            str(stage1_ckpt),
            "--lambda_dir",
            "0.2",
            "--lambda_doa",
            "1.0",
            "--lambda_fft_recon",
            "0.0",
            "--lambda_phase_recon",
            "0.0",
            "--base_lr",
            "2.5e-4",
            "--max_lr",
            "2.5e-4",
            "--min_lr",
            "1e-6",
            "--warmup",
            "10",
            "--num_epochs",
            "100",
            "--num_sched_epochs",
            "80",
        ] + common_forward

        _run_cmd(stage2, cwd=root)

        stage2_ckpt = weights_dir / "model_trained_stage2.pth"
        print("\n[OK] Two-stage training finished.")
        print(f"- Stage-1 ckpt: {stage1_ckpt}")
        print(f"- Stage-2 ckpt: {stage2_ckpt}")


if __name__ == "__main__":
    # Paper-style default: coherent, four sources, few snapshots.
    main(snr=5.0, M=4, signal_nature="coherent", T=2, doa_gap=0.0)
