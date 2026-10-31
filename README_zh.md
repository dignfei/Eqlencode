# EqLen 审稿人代码与复现说明（中文）

环境使用 Python 3.12、Torch 2.13.0+cu130、CUDA 13.0、Transformers 5.19.0、Ray 2.59。vLLM 与 verl 均取自随附的精确源码快照。若硬件、CUDA/Torch、驱动或编译器不同，需要在目标机器重新编译/安装并重新验收。

## 安装 GPU 环境

先安装与目标 CUDA 驱动匹配的 PyTorch，再从两个随附源码树安装。命令显式设置构建版本，避免 setuptools-scm 无法识别版本而报错。vLLM 含 native EqLen 调度器改动，需要从此源码构建；不要用未打补丁的上游 vLLM wheel 替代。

```bash
cd .
uv venv --python 3.12
source .venv/bin/activate

# 在 NVIDIA CUDA 主机按 vLLM 的安装说明选择 PyTorch/CUDA wheel。
cd vllm
VLLM_VERSION_OVERRIDE=0.31.1rc1.dev147+g7914289b7 uv pip install -e . --torch-backend=auto
cd ../verl
uv pip install -e .
uv pip install -r requirements.txt
```

FlashAttention2 是可选后端。其余默认配置使用 SDPA。

```bash
uv pip install --no-deps --only-binary :all: \
  --index-url https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple/ \
  flash-attn==2.8.3
```

`--only-binary` 禁止退回源码构建。其他 Torch/CUDA 组合必须选择匹配的二进制索引；没有匹配 wheel 时应使用 SDPA 或按目标机的软件栈另行构建 FA2。



## 论文配置训练

准备好论文所用的基础模型和 Parquet 数据后，设置路径并从 verl 根目录提交训练：

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

Parquet 样本格式包含 `prompt`（聊天消息列表）、`data_source` 与 `reward_model.ground_truth` 等标准 verl 字段。
