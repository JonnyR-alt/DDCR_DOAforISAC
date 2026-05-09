"""Automated pipeline for standalone source-number classification.

Pipeline:
1) Generate mixed-M data via f_gendata.py (--mix_M)
2) Train standalone classifier via train_num_cls.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run mixed-M data generation + num-cls training")

    p.add_argument("--N", type=int, default=8)
    p.add_argument("--T", type=int, default=3)
    p.add_argument("--snr", type=float, default=5.0)
    p.add_argument("--signal_type", type=str, default="NarrowBand")
    p.add_argument("--signal_nature", type=str, default="coherent")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--bias", type=float, default=0.0)
    p.add_argument("--sv_noise_var", type=float, default=0.0)
    p.add_argument("--doa_gap", type=float, default=10.0)
    p.add_argument("--fixed_gap", action="store_true")

    p.add_argument("--M_list", type=str, default="1,2,3,4,5")
    p.add_argument("--num_sets_per_M", type=int, default=10000)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--num_sched_epochs", type=int, default=80)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--base_lr", type=float, default=2.5e-4)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_data_gen", action="store_true")

    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--log_interval", type=int, default=50)

    return p


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(__file__).resolve().parent
    data_pipeline_dir = root / "data_pipeline"

    if not args.skip_data_gen:
        gen_cmd = [
            sys.executable,
            str(data_pipeline_dir / "f_gendata.py"),
            "--mix_M",
            "--output_dir",
            "datasets/generated_snapshots",
            "--N",
            str(args.N),
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
            "--M_list",
            str(args.M_list),
            "--num_sets_per_M",
            str(args.num_sets_per_M),
            "--seed",
            str(args.seed),
            "--skip_validate",
        ]
        if args.fixed_gap:
            gen_cmd.append("--fixed_gap")
        _run(gen_cmd, cwd=root)

    train_cmd = [
        sys.executable,
        str(root / "train_num_cls.py"),
        "--data_dir",
        "datasets/generated_snapshots",
        "--N",
        str(args.N),
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
        "--M_list",
        str(args.M_list),
        "--train_ratio",
        str(args.train_ratio),
        "--split_seed",
        str(args.split_seed),
        "--seed",
        str(args.seed),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--num_sched_epochs",
        str(args.num_sched_epochs),
        "--warmup",
        str(args.warmup),
        "--base_lr",
        str(args.base_lr),
        "--max_lr",
        str(args.max_lr),
        "--min_lr",
        str(args.min_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--dim",
        str(args.dim),
        "--depth",
        str(args.depth),
        "--attn_dim",
        str(args.attn_dim),
        "--log_interval",
        str(args.log_interval),
    ]
    if args.fixed_gap:
        train_cmd.append("--fixed_gap")

    _run(train_cmd, cwd=root)

    print("\n[OK] num_cls pipeline finished (data generation + training).")


if __name__ == "__main__":
    main()
