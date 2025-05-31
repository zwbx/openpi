# Training on DROID

## Install

Data loading dependencies:
```bash
uv pip install tensorflow_datasets
uv pip install tensorflow
uv pip install "dlimp @ git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"
uv pip install ml-dtypes==0.5.1  # somehow the previous installations mess up this package version, so we install it again
```

## Download DROID dataset

## Run

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_droid_finetune --max-frames 500_000
```

```bash
uv run scripts/train.py pi0_fast_droid --exp-name=my_experiment --overwrite
```



