# EqLen Reviewer Code and Reproduction Guide (English)

The environment uses Python 3.12, Torch 2.13.0+cu130, CUDA 13.0, Transformers 5.19.0 and Ray 2.59. Both vLLM and verl use the exact bundled source snapshots. If the hardware, CUDA/Torch stack, driver or compiler differs, rebuild/install and rerun validation on the target machine.

## GPU installation

Install PyTorch compatible with the target CUDA driver, then install both bundled source trees. The command explicitly sets the build version to prevent setuptools-scm version-detection errors. The vLLM snapshot includes native EqLen scheduler changes and must be built from this source; do not replace it with an unpatched upstream vLLM wheel.

```bash
cd .
uv venv --python 3.12
source .venv/bin/activate

# Select the PyTorch/CUDA wheel for the NVIDIA CUDA host following vLLM's installation instructions.
cd vllm
VLLM_VERSION_OVERRIDE=0.31.1rc1.dev147+g7914289b7 uv pip install -e . --torch-backend=auto
cd ../verl
uv pip install -e .
uv pip install -r requirements.txt
```

FlashAttention2 is an optional backend. The remaining default configurations use SDPA.

```bash
uv pip install --no-deps --only-binary :all: \
  --index-url https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple/ \
  flash-attn==2.8.3
```

`--only-binary` prevents a source-build fallback. For a different Torch/CUDA combination, select a matching binary index; if no matching wheel exists, use SDPA or build FA2 separately for the target machine's software stack.

## Running the paper configuration

Prepare the paper's base models and Parquet data, set the paths and launch training from the verl root:

```bash
cd ./verl
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
uv run --no-project --python ../.venv/bin/python -m recipe.equal_length.main_equal_length \
  data.train_files=/data/train.parquet \
  data.val_files=/data/validation.parquet \
  actor_rollout_ref.model.path=/models/Qwen3-32B \
  trainer.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  eqlen.num_pairs=8
```

The Parquet sample format includes standard verl fields such as `prompt` (a list of chat messages), `data_source` and `reward_model.ground_truth`.
