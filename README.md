# DDCR-DoA for GLOBECOM 2026

This repository contains a trimmed and cleaned implementation of the paper:

**Robust DoA Estimation for Low-Altitude ISAC via Dual-Domain Covariance Restructuring**

The code focuses on the main reproducible pipeline:

- synthetic snapshot generation
- covariance preprocessing
- DDCR model training
- source-number classification extension
- runtime benchmarking

## Project Layout

```text
DDCR_DOA_GLOBECOM2026/
├── data_pipeline/
│   ├── f_gendata.py
│   ├── cov_svd_processing.py
│   └── __init__.py
├── train.py
├── main.py
├── train_num_cls.py
├── num_cls_main.py
├── models/
│   ├── ddcr_model.py
│   └── lossfunction.py
└── src/
    ├── data_handler.py
    ├── evaluation.py
    ├── methods.py
    ├── models.py
    ├── plotting.py
    ├── signal_creation.py
    ├── system_model.py
    ├── training.py
    └── utils.py
```

## Requirements

Install the main dependencies:

```bash
pip install -r requirements.txt
```

If you prefer manual installation, the core packages are:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

`ptflops` is optional and only used for complexity reporting.

## Quick Start

### 1. Generate synthetic data

```bash
python data_pipeline/f_gendata.py --N 8 --M 4 --T 2 --snr 5 --signal_type NarrowBand --signal_nature coherent
```

### 2. Preprocess covariance matrices

```bash
python data_pipeline/cov_svd_processing.py --N 8 --M 4 --T 2 --snr 5 --signal_type NarrowBand --signal_nature coherent --rank 4
```

### 3. Run two-stage training

```bash
python main.py --N 8 --M 4 --T 2 --snr 5 --signal_nature coherent --doa_gap 0.0
```

### 4. Train source-number classifier

```bash
python num_cls_main.py --N 8 --T 2 --snr 5 --signal_nature coherent
```

## Notes

- `models/ddcr_model.py` is the main model implementation.
- `data_pipeline/` holds data generation and covariance preprocessing scripts.
- Generated datasets, checkpoints, logs, and cache files are ignored by Git.
- The default settings in the scripts match the paper-style low-snapshot, coherent-source experiments.
