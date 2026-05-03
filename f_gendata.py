# 用于生成阵列快拍数据，快拍保存在datasets文件夹下
"""
Generate array snapshots and per-snapshot covariance (auto-correlation) using existing
Samples/SystemModelParams utilities.

Outputs (complex npy):
- snapshots{_dataset_suffix}.npy: shape (num_sets, T, N)
- covariances{_dataset_suffix}.npy: shape (num_sets, T, N, N)
- doas{_dataset_suffix}.npy: shape (num_sets, M) (true DOAs per set)

Adjust the parameters in the __main__ block as needed.
"""
import argparse
from pathlib import Path
import numpy as np

from src.system_model import SystemModelParams
from src.signal_creation import Samples
from src.utils import set_unified_seed
from src.data_handler import set_dataset_filename


def _parse_int_list(text: str) -> list[int]:
    vals = []
    for t in str(text).split(","):
        t = t.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("M_list is empty. Provide values like '1,2,3'.")
    return vals


def _build_mix_suffix(
    N: int,
    T: int,
    snr: float,
    signal_type: str,
    signal_nature: str,
    eta: float,
    bias: float,
    sv_noise_var: float,
    doa_gap: float,
    fixed_gap: bool,
    m_list: list[int],
) -> str:
    """Build a stable suffix for mixed-M classification datasets."""
    ms = "-".join(str(int(m)) for m in sorted(set(m_list)))
    fg = "fixed" if bool(fixed_gap) else "rand"
    return (
        f"_MIXM={ms}_N={int(N)}_T={int(T)}_SNR={float(snr)}"
        f"_signal={str(signal_type)}_{str(signal_nature)}"
        f"_eta={float(eta)}_bias={float(bias)}_sv_noise_var={float(sv_noise_var)}"
        f"_doa_gap={float(doa_gap)}_{fg}"
    )


def _build_suffix(params: SystemModelParams, samples_size: float) -> str:
    """Return filename suffix consistent with dataset naming (without .h5)."""
    return set_dataset_filename(params, samples_size).replace(".h5", "")


def _build_params(
    N: int,
    M: int,
    T: int,
    snr: float,
    signal_type: str,
    signal_nature: str,
    eta: float,
    bias: float,
    sv_noise_var: float,
    doa_gap: float = 15.0,
    fixed_gap: bool = False,
) -> SystemModelParams:
    return (
        SystemModelParams()
        .set_parameter("N", N)
        .set_parameter("M", M)
        .set_parameter("T", T)
        .set_parameter("snr", snr)
        .set_parameter("signal_type", signal_type)
        .set_parameter("signal_nature", signal_nature)
        .set_parameter("eta", eta)
        .set_parameter("bias", bias)
        .set_parameter("sv_noise_var", sv_noise_var)
        .set_parameter("doa_gap", doa_gap)
        .set_parameter("fixed_gap", fixed_gap)
    )


def generate_snapshots(
    output_dir: Path,
    N: int = 8,
    M: int = 3,
    T: int = 100,
    num_sets: int = 1,
    snr: float = 10,
    signal_type: str = "NarrowBand",
    signal_nature: str = "coherent",
    eta: float = 0.0,
    bias: float = 0.0,
    sv_noise_var: float = 0.0,
    doa_gap: float = 15.0,
    fixed_gap: bool = False,
    seed: int = 42,
):
    """Generate one or multiple DOA sets of snapshots and per-snapshot covariance, then save to npy.

    Args:
        output_dir: where to store npy files.
        N: number of sensors.
        M: number of sources.
        T: number of snapshots.
        num_sets: how many different DOA sets to generate (each set has T snapshots).
        snr: signal-to-noise ratio (dB), controls signal amplitude.
        signal_type: "NarrowBand" or "Broadband".
        signal_nature: "coherent" or "non-coherent".
        eta: non-uniform spacing deviation.
        bias: uniform spacing bias.
        sv_noise_var: steering vector noise variance.
        seed: random seed for reproducibility.
    """
    set_unified_seed(seed)

    base_params = _build_params(
        N=N,
        M=M,
        T=T,
        snr=snr,
        signal_type=signal_type,
        signal_nature=signal_nature,
        eta=eta,
        bias=bias,
        sv_noise_var=sv_noise_var,
        doa_gap=doa_gap,
        fixed_gap=fixed_gap,
    )

    samples_model = Samples(base_params)

    all_snaps = np.empty((num_sets, T, N), dtype=np.complex128)
    all_covs = np.empty((num_sets, T, N, N), dtype=np.complex128)
    all_snaps_clean = np.empty((num_sets, T, N), dtype=np.complex128)
    all_covs_clean = np.empty((num_sets, T, N, N), dtype=np.complex128)
    all_doas = np.empty((num_sets, M), dtype=np.float64)

    # Steering vectors (导向矢量/steering matrix)
    # - NarrowBand: A has shape (N, M)
    # - Broadband: SV has shape (F, N, M) where F = f_sampling["Broadband"]
    all_sv = None
    all_sv_clean = None
    if signal_type.startswith("NarrowBand"):
        all_sv = np.empty((num_sets, N, M), dtype=np.complex128)
        # 对窄带而言 steering 本身与是否加噪无关，这里保留 clean 版本便于下游统一处理
        all_sv_clean = np.empty((num_sets, N, M), dtype=np.complex128)

    for idx in range(num_sets):
        samples_model.set_doa(None, doa_gap=doa_gap)  # random DOAs with minimum separation
        samples, signal, A, noise = samples_model.samples_creation()

        # 含噪快拍（与原逻辑一致）
        snapshots = samples.T.astype(np.complex128)
        covariances = np.einsum("ti,tj->tij", snapshots, snapshots.conj())

        # 无噪快拍：窄带用 A@signal，宽带将噪声逆变换到时域再相减
        if signal_type.startswith("NarrowBand"):
            clean_time = (A @ signal).T.astype(np.complex128)
        elif signal_type.startswith("Broadband"):
            noise_time = np.fft.ifft(noise, axis=1)[:, :T]
            clean_time = (samples - noise_time).T.astype(np.complex128)
        else:
            clean_time = snapshots  # fallback

        covariances_clean = np.einsum("ti,tj->tij", clean_time, clean_time.conj())

        all_snaps[idx] = snapshots
        all_covs[idx] = covariances
        all_snaps_clean[idx] = clean_time
        all_covs_clean[idx] = covariances_clean
        all_doas[idx] = np.array(samples_model.doa, dtype=np.float64)

        # Save steering vectors aligned with current generation flow
        if signal_type.startswith("NarrowBand"):
            all_sv[idx] = A.astype(np.complex128)
            all_sv_clean[idx] = A.astype(np.complex128)
        elif signal_type.startswith("Broadband"):
            # 延迟分配，避免重复推断维度
            if all_sv is None:
                # A here is actually SV in broadband branch of Samples.samples_creation
                # samples_creation returns SV with shape (F, N, M)
                all_sv = np.empty(
                    (num_sets,) + tuple(np.array(A).shape), dtype=np.complex128
                )
                all_sv_clean = np.empty_like(all_sv)
            all_sv[idx] = np.array(A, dtype=np.complex128)
            all_sv_clean[idx] = np.array(A, dtype=np.complex128)

    output_dir.mkdir(parents=True, exist_ok=True)
    # 使用与数据集一致的后缀命名（去掉 .h5），不包含 num_sets，仅按配置参数
    suffix = _build_suffix(base_params, base_params.T)
    snap_path = output_dir / f"snapshots{suffix}.npy"
    cov_path = output_dir / f"covariances{suffix}.npy"
    snap_clean_path = output_dir / f"snapshots_clean{suffix}.npy"
    cov_clean_path = output_dir / f"covariances_clean{suffix}.npy"
    doa_path = output_dir / f"doas{suffix}.npy"
    sv_path = output_dir / f"steering_vectors{suffix}.npy"
    sv_clean_path = output_dir / f"steering_vectors_clean{suffix}.npy"

    np.save(snap_path, all_snaps)
    np.save(cov_path, all_covs)
    np.save(snap_clean_path, all_snaps_clean)
    np.save(cov_clean_path, all_covs_clean)
    np.save(doa_path, all_doas)
    if all_sv is not None:
        np.save(sv_path, all_sv)
    if all_sv_clean is not None:
        np.save(sv_clean_path, all_sv_clean)

    print(f"Saved snapshots to: {snap_path} (shape {all_snaps.shape})")
    print(f"Saved covariances to: {cov_path} (shape {all_covs.shape})")
    print(f"Saved noiseless snapshots to: {snap_clean_path} (shape {all_snaps_clean.shape})")
    print(f"Saved noiseless covariances to: {cov_clean_path} (shape {all_covs_clean.shape})")
    print(f"Saved true DOAs to: {doa_path} (shape {all_doas.shape})")
    if all_sv is not None:
        print(f"Saved steering vectors to: {sv_path} (shape {all_sv.shape})")
    if all_sv_clean is not None:
        print(f"Saved steering vectors (clean) to: {sv_clean_path} (shape {all_sv_clean.shape})")


def validate(
    path_dir: Path = Path("datasets/generated_snapshots"),
    N: int = 8,
    M: int = 3,
    T: int = 100,
    num_sets: int = 1,
    snr: float = 10,
    signal_type: str = "NarrowBand",
    signal_nature: str = "non-coherent",
    eta: float = 0.0,
    bias: float = 0.0,
    sv_noise_var: float = 0.0,
    doa_gap: float = 15.0,
    fixed_gap: bool = False,
):
    data_dir = path_dir
    # 查看数据维度
    params = _build_params(
        N=N,
        M=M,
        T=T,
        snr=snr,
        signal_type=signal_type,
        signal_nature=signal_nature,
        eta=eta,
        bias=bias,
        sv_noise_var=sv_noise_var,
        doa_gap=doa_gap,
        fixed_gap=fixed_gap,
    )
    suffix = _build_suffix(params, params.T)
    snap_file = data_dir / f"snapshots{suffix}.npy"
    cov_file = data_dir / f"covariances{suffix}.npy"
    doa_file = data_dir / f"doas{suffix}.npy"
    sv_file = data_dir / f"steering_vectors{suffix}.npy"
    snaps = np.load(snap_file)
    covs = np.load(cov_file)
    doas = np.load(doa_file)
    print(f"Snapshots shape: {snaps.shape}")  # 应为 (num_sets, T, N)
    print(f"Covariances shape: {covs.shape}")  # 应为 (num_sets, T, N, N)
    print(f"DOAs shape: {doas.shape}")  # 应为 (num_sets, M)
    if sv_file.exists():
        sv = np.load(sv_file)
        print(f"Steering vectors shape: {sv.shape}")


def generate_snapshots_mix_m(
    output_dir: Path,
    N: int,
    T: int,
    m_list: list[int],
    num_sets_per_m: int,
    snr: float,
    signal_type: str,
    signal_nature: str,
    eta: float,
    bias: float,
    sv_noise_var: float,
    doa_gap: float,
    fixed_gap: bool,
    seed: int,
) -> str:
    """Generate mixed-source-count snapshots for standalone source-count classification.

    Saved files:
      - snapshots{mix_suffix}.npy:        (B, T, N) complex
      - snapshots_clean{mix_suffix}.npy:  (B, T, N) complex
      - covariances{mix_suffix}.npy:      (B, T, N, N) complex
      - covariances_clean{mix_suffix}.npy:(B, T, N, N) complex
      - doas_padded{mix_suffix}.npy:      (B, M_max) float, NaN padded
      - source_count{mix_suffix}.npy:     (B,) int64 in {m_list}
    """
    set_unified_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    m_list = sorted(set(int(m) for m in m_list))
    if min(m_list) < 1:
        raise ValueError(f"All M in M_list must be >=1, got: {m_list}")
    B_per = int(num_sets_per_m)
    if B_per <= 0:
        raise ValueError(f"num_sets_per_m must be > 0, got {num_sets_per_m}")

    all_snaps = []
    all_covs = []
    all_snaps_clean = []
    all_covs_clean = []
    all_doas = []
    all_counts = []

    for M in m_list:
        params = _build_params(
            N=N,
            M=M,
            T=T,
            snr=snr,
            signal_type=signal_type,
            signal_nature=signal_nature,
            eta=eta,
            bias=bias,
            sv_noise_var=sv_noise_var,
            doa_gap=doa_gap,
            fixed_gap=fixed_gap,
        )
        samples_model = Samples(params)

        for _ in range(B_per):
            samples_model.set_doa(None, doa_gap=doa_gap)
            samples, signal, A, noise = samples_model.samples_creation()

            snapshots = samples.T.astype(np.complex128)  # (T,N)
            covariances = np.einsum("ti,tj->tij", snapshots, snapshots.conj())

            if signal_type.startswith("NarrowBand"):
                clean_time = (A @ signal).T.astype(np.complex128)
            elif signal_type.startswith("Broadband"):
                noise_time = np.fft.ifft(noise, axis=1)[:, :T]
                clean_time = (samples - noise_time).T.astype(np.complex128)
            else:
                clean_time = snapshots

            covariances_clean = np.einsum("ti,tj->tij", clean_time, clean_time.conj())

            all_snaps.append(snapshots)
            all_covs.append(covariances)
            all_snaps_clean.append(clean_time)
            all_covs_clean.append(covariances_clean)
            all_doas.append(np.array(samples_model.doa, dtype=np.float64))
            all_counts.append(M)

    # stack + shuffle
    snaps = np.stack(all_snaps, axis=0)
    covs = np.stack(all_covs, axis=0)
    snaps_clean = np.stack(all_snaps_clean, axis=0)
    covs_clean = np.stack(all_covs_clean, axis=0)
    counts = np.array(all_counts, dtype=np.int64)

    M_max = max(m_list)
    doas_padded = np.full((len(all_doas), M_max), np.nan, dtype=np.float64)
    for i, doa in enumerate(all_doas):
        doas_padded[i, : doa.shape[0]] = doa

    rng = np.random.default_rng(seed)
    perm = rng.permutation(snaps.shape[0])
    snaps = snaps[perm]
    covs = covs[perm]
    snaps_clean = snaps_clean[perm]
    covs_clean = covs_clean[perm]
    counts = counts[perm]
    doas_padded = doas_padded[perm]

    mix_suffix = _build_mix_suffix(
        N=N,
        T=T,
        snr=snr,
        signal_type=signal_type,
        signal_nature=signal_nature,
        eta=eta,
        bias=bias,
        sv_noise_var=sv_noise_var,
        doa_gap=doa_gap,
        fixed_gap=fixed_gap,
        m_list=m_list,
    )

    np.save(output_dir / f"snapshots{mix_suffix}.npy", snaps)
    np.save(output_dir / f"covariances{mix_suffix}.npy", covs)
    np.save(output_dir / f"snapshots_clean{mix_suffix}.npy", snaps_clean)
    np.save(output_dir / f"covariances_clean{mix_suffix}.npy", covs_clean)
    np.save(output_dir / f"doas_padded{mix_suffix}.npy", doas_padded)
    np.save(output_dir / f"source_count{mix_suffix}.npy", counts)

    print(f"[MIX-M] Saved snapshots: {output_dir / f'snapshots{mix_suffix}.npy'} shape={snaps.shape}")
    print(f"[MIX-M] Saved covariances: {output_dir / f'covariances{mix_suffix}.npy'} shape={covs.shape}")
    print(f"[MIX-M] Saved source_count: {output_dir / f'source_count{mix_suffix}.npy'} shape={counts.shape}")
    uniq, cnt = np.unique(counts, return_counts=True)
    dist = ", ".join([f"M={int(u)}:{int(c)}" for u, c in zip(uniq, cnt)])
    print(f"[MIX-M] Class distribution -> {dist}")

    return mix_suffix

def _build_arg_parser(M = 3,N = 8,T = 100,num_sets: int = 1000,snr = 10,signal_type = "NarrowBand",signal_nature = "non-coherent",eta = 0.0,bias = 0.0,sv_noise_var = 0.0, doa_gap: float = 15.0, fixed_gap: bool = False) -> argparse.ArgumentParser:
    '''
    Build argument parser for snapshot generation and validation.
    Args:
        M: number of sources.
        N: number of sensors.
        T: number of snapshots.
        num_sets: number of different DOA sets to generate.
        snr: signal-to-noise ratio (dB).
        signal_type: "NarrowBand" or "Broadband".
        signal_nature: "coherent" or "non-coherent".
        eta: non-uniform spacing deviation.
        bias: uniform spacing bias.
        sv_noise_var: steering vector noise variance.
    '''
    parser = argparse.ArgumentParser(
        description="Generate snapshots and per-snapshot covariance, with unified parameters for generation and validation."
    )
    parser.add_argument("--output_dir", type=str, default="datasets/generated_snapshots")
    parser.add_argument("--N", type=int, default=N)
    parser.add_argument("--M", type=int, default=M)
    parser.add_argument("--T", type=int, default=T)
    parser.add_argument("--num_sets", type=int, default=num_sets)
    parser.add_argument("--snr", type=float, default=snr)
    parser.add_argument("--signal_type", type=str, default=signal_type)
    parser.add_argument("--signal_nature", type=str, default=signal_nature)
    parser.add_argument("--eta", type=float, default=eta)
    parser.add_argument("--bias", type=float, default=bias)
    parser.add_argument("--sv_noise_var", type=float, default=sv_noise_var)
    parser.add_argument("--doa_gap", type=float, default=doa_gap, help="Minimum gap between DOAs in degrees")
    if fixed_gap:
        parser.add_argument("--fixed_gap", action="store_true", default=True, help="Force fixed gap exactly equal to doa_gap")
    else:
        parser.add_argument("--fixed_gap", action="store_true", help="Force fixed gap exactly equal to doa_gap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mix_M",
        action="store_true",
        help="If set, generate mixed-source-count dataset for source-number classification.",
    )
    parser.add_argument(
        "--M_list",
        type=str,
        default="1,2,3",
        help="Comma-separated source counts used when --mix_M is enabled (e.g. '1,2,3,4').",
    )
    parser.add_argument(
        "--num_sets_per_M",
        type=int,
        default=5000,
        help="Number of sets generated for each M in --M_list when --mix_M is enabled.",
    )
    parser.add_argument(
        "--skip_validate",
        action="store_true",
        help="If set, skip loading the saved npy files for shape validation.",
    )
    return parser


def main():
    parser = _build_arg_parser(M=2, N=8, T=2, num_sets=50000, snr=5.0, signal_type="NarrowBand", signal_nature="coherent", eta=0.0, bias=0.0, sv_noise_var=0.0, doa_gap=10.0, fixed_gap=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if bool(getattr(args, "mix_M", False)):
        m_list = _parse_int_list(args.M_list)
        mix_suffix = generate_snapshots_mix_m(
            output_dir=output_dir,
            N=args.N,
            T=args.T,
            m_list=m_list,
            num_sets_per_m=args.num_sets_per_M,
            snr=args.snr,
            signal_type=args.signal_type,
            signal_nature=args.signal_nature,
            eta=args.eta,
            bias=args.bias,
            sv_noise_var=args.sv_noise_var,
            doa_gap=args.doa_gap,
            fixed_gap=args.fixed_gap,
            seed=args.seed,
        )
        print(f"[MIX-M] Done. Dataset suffix: {mix_suffix}")
        return

    generate_snapshots(
        output_dir=output_dir,
        N=args.N,
        M=args.M,
        T=args.T,
        num_sets=args.num_sets,
        snr=args.snr,
        signal_type=args.signal_type,
        signal_nature=args.signal_nature,
        eta=args.eta,
        bias=args.bias,
        sv_noise_var=args.sv_noise_var,
        doa_gap=args.doa_gap,
        fixed_gap=args.fixed_gap,
        seed=args.seed,
    )

    if not args.skip_validate:
        validate(
            path_dir=output_dir,
            N=args.N,
            M=args.M,
            T=args.T,
            num_sets=args.num_sets,
            snr=args.snr,
            signal_type=args.signal_type,
            signal_nature=args.signal_nature,
            eta=args.eta,
            bias=args.bias,
            sv_noise_var=args.sv_noise_var,
            doa_gap=args.doa_gap,
            fixed_gap=args.fixed_gap,
        )


if __name__ == "__main__":
    main()