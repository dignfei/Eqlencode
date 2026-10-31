# EqLen

EqLen constructs equal-length comparison segments inside the patched vLLM
engine, then trains on complete trajectories with inherited prefixes masked.
The recipe reuses verl's standard Ray trainer, model workers, verifier workers,
weight synchronization, validation, and checkpoints.

The inference optimizations remain inside vLLM: full computed attention KV pages
are shared by reference, and only the writable tail and live hybrid recurrent
state need private copies. The engine renews a finished slot at EOS without
waiting for a frontend replacement request or recomputing the cached prefix.
The verl bridge calls the native helper once per question and preserves this
schedule. Training uses complete contexts to obtain the required gradients.

`rollout.n=1` means one carrier row per original question.
`eqlen.num_pairs=K` starts `K` independent pairs, corresponding to paper group
size `G=2K`. A subgroup producing `N` segment pairs expands into `N+1` complete
responses. Each response is verified exactly once. A terminating plus segment
takes its response reward; its minus counterpart takes the maximum reward
among subsequent extending responses (`aggregation=mean` is also available).

Training retains the original prompt and full response context. Separate
`eqlen_loss_mask`, `eqlen_advantages`, `eqlen_segment_ids`, and
`eqlen_loss_weights` fields ensure that each active segment contributes once.
An active token in a segment of length `L` has weight `1/(2*N*L)`, followed by
an average over all real subgroups. Reward ties are masked; they remain in `N`
and in the subgroup denominator. Training padding has zero loss weight and
does not create an extra verifier call. KL uses the same segment mask and
weights.

Choose `actor_rollout_ref.actor.policy_loss.loss_mode`:

- `eqlen_grpo`: population-normalized pair advantages and tokenwise PPO clipping.
- `eqlen_gspo`: one geometric ratio per active segment, with the same weights.
- `eqlen_rloo`: leave-one-out reward differences and the paper's `-A*log_pi`
  objective.

Per-segment GSPO ratios and the placement of masked KL are explicit integration
choices where the paper does not specify a complete combined objective.

The expanded padded batch is one optimizer update with one PPO epoch. If every
pair ties, the actor RPC is skipped, including optimizer weight decay and the
optimizer learning-rate scheduler step. The trainer's rollout/global step and
checkpoint synchronization still advance. Multiple future optimizer steps
would require complete-subgroup packing; splitting arbitrary response rows
does not preserve this normalization.

Sampling seeds cover separate native subgroup and renewal ranges for each
training step and original batch row, including repeated questions. Resuming
the same step preserves the seed assignment. Reward comparison keeps float64
precision, so small continuous reward differences do not become artificial
ties when the training reward tensor is assembled in float32.

The recipe defaults to `model.use_remove_padding=false` and Ulysses size 1.
This keeps Qwen3Next convolution and recurrent states separate between rows.
Training attention defaults to PyTorch SDPA, so this padded path does not
require the optional FlashAttention2 training package. The GPU checker inherits
this choice instead of overriding it with eager attention.
FlashAttention2 is also supported: select
`actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2`
after installing a matching binary wheel as below. Both backends retain the
same padded input and Ulysses size 1 requirements.
With `use_remove_padding=true`, this initial recipe permits only static
one-row microbatches for training, old log-probs, and reference log-probs.
For larger or dynamic microbatches, use the default padded model input path.
LoRA, dynamic context parallelism, and the ordinary trajectory-level variance
proxy (`calculate_sum_pi_squared`) are currently rejected.

## Run

Use the patched vLLM tree providing native `generate_equal_length_pairs_async`
and the compatible verl bridge. Install it in a CUDA environment compatible
with that vLLM build, then install this checkout without dependency resolution:

```bash
uv pip install --no-deps -e /path/to/patched/vllm
uv pip install --no-deps -e /path/to/verl
```

Install the remaining verl dependencies separately. Avoid `verl[vllm]` for
this recipe: its upstream `vllm<=0.12.0` requirement can replace the patched
engine. The local bridge has been tested with the patched vLLM
`0.31.1rc1.dev147+g7914289b7`. Native generation requires the supported
synchronous scheduler configuration; it explicitly rejects unsupported engine
modes. Keep `max_num_batched_tokens >= 2*max_prompt_length` and model context
large enough for the full prompt plus response budget.

For FlashAttention2, use a prebuilt wheel matching Python, Torch and CUDA.
The validation environment uses Python 3.12 and Torch 2.13/cu130; its binary
comes from the [official verl wheelhouse](https://github.com/verl-project/verl-wheelhouse):

```bash
uv pip install --python /path/to/cuda-venv/bin/python --no-deps \
  --only-binary :all: \
  --index-url https://verl-project.github.io/verl-wheelhouse/cu130/torch2.13/simple/ \
  flash-attn==2.8.3
```

Choose the wheelhouse index matching your existing Torch/CUDA environment.
`--only-binary` refuses source builds, and `--no-deps` preserves that environment.
The installed CUDA extension was checked with actual forward and backward passes.

EqLen enables `actor_rollout_ref.model.override_config.eqlen_router_fp32=true`
and passes the same value to vLLM through `rollout.engine_kwargs.vllm.hf_overrides`.
For Qwen3Next, actor/reference and rollout preserve the original router weight
dtype but produce FP32 routing logits before softmax/top-k. In vLLM this must
be a direct FP32 GEMM output: casting already-rounded FP16 logits cannot recover
a lost expert margin. The combined router/shared-expert gate GEMM is disabled
in this mode. Other recipes without this flag retain their existing behavior.
Use the patched source in both environments; changing only one side does not
establish a matching routing policy.

From the repository root:

```bash
uv run --no-project --python /path/to/cuda-venv/bin/python \
  python -m recipe.equal_length.main_equal_length \
  data.train_files=/path/train.parquet \
  data.val_files=/path/validation.parquet \
  actor_rollout_ref.model.path=/path/model \
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

The entrypoint defaults `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` **before importing
Ray**, keeping workers in the chosen CUDA environment. Ray 2.59's automatic
`uv run` runtime hook otherwise fails on verl's default `working_dir=None`.
An explicitly supplied environment value takes precedence. The GPU checker
also disables automatic environment reconstruction.

The KL coefficient `0.002` matches the paper's training hyperparameters.
Other defaults (`K=8`, clipping `0.2`, prompt/response budgets, and batch size)
are explicit implementation choices and do not reconstruct every paper
experiment setting. This first recipe supports
text-only verifier rewards, without a critic, learned reward model, tools,
distillation, rollout rejection sampling, or inference importance correction.
Validation uses ordinary complete responses and `rollout.val_kwargs.n`.

## Validation

```bash
.venv/bin/python -m pytest \
  tests/trainer/ppo/test_equal_length_algorithms_on_cpu.py \
  tests/trainer/ppo/test_equal_length_trainer_on_cpu.py \
  tests/workers/rollout/test_equal_length_bridge_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/utils/test_padding_on_cpu.py tests/models/test_monkey_patch_on_cpu.py -q
```

CPU tests verify DataProto/TensorDict expansion, masks and weights, reward
cardinality, padding, balancing, gradients, and one joint actor dispatch.
Dummy rows are excluded from reward statistics and exported trajectories.
`eqlen/generated_tokens` counts newly generated segments once; full trajectory
tokens and training context tokens including padding are reported separately.

The GPU checker creates a local random model, tokenizer, two-row dataset and
synthetic complete-response verifier. It runs the actual Ray trainer, FSDP2
backward, AdamW, checkpoint saving, native paired generation, sleep/wake and
weight transport. It requires nonzero finite gradients, matching optimizer counters,
changed actor parameters, and changed vLLM log-probs for fixed context/token
pairs after synchronization. No model download is needed.

```bash
uv run --no-project --python /path/to/cuda-venv/bin/python \
  python -m recipe.equal_length.gpu_check \
  --work-dir /tmp/verl-eqlen-check-opt --family opt --kl

uv run --no-project --python /path/to/cuda-venv/bin/python \
  python -m recipe.equal_length.gpu_check \
  --work-dir /tmp/verl-eqlen-check-qwen3next --family qwen3next --kl

uv run --no-project --python /path/to/cuda-venv/bin/python \
  python -m recipe.equal_length.gpu_check \
  --work-dir /tmp/verl-eqlen-check-fa2 --family qwen3next \
  --steps 3 --binary-reward --kl --train-attention flash_attention_2

uv run --no-project --python /path/to/cuda-venv/bin/python \
  python -m recipe.equal_length.gpu_check \
  --work-dir /path/to/eqlen-long-check --family qwen3next \
  --steps 20 --loss-mode eqlen_grpo --binary-reward --kl
```

`--steps` repeats the actual training loop for the requested number of batches;
`--loss-mode` also accepts `eqlen_gspo` and `eqlen_rloo`. The default remains one
batch with continuous synthetic rewards. `--binary-reward` uses synthetic 0/1
complete-response scores, including tied pairs; these scores measure pipeline
behavior rather than answer correctness. Every batch is checked independently
against the paper's trajectory reconstruction, descendant reward aggregation,
segment masks, advantages and `1/(2*N*L)` weights. The proof retains native wire
data and complete verifier results. Parameter hashes, AdamW counters and fixed
token probabilities are checked after each synchronization, including skipped
all-tie batches. A run completes only after all requested batches finish.
`--train-attention sdpa|flash_attention_2` selects the actual training backend;
omitting it inherits the recipe default. Each weight-sync snapshot records and
checks the loaded attention implementation and Qwen3Next FP32 router installation.

For this branch's NumPy 1.x environment, use compatible Parquet dependencies
such as `numpy==1.26.4`, `pandas==2.3.3`, `pyarrow==19.0.1` and
`datasets==3.6.0`; PyArrow 26 requires NumPy 2 and cannot be imported in that
environment, while newer datasets releases can require newer PyArrow.

### Two-process data-parallel simulation

The following check uses two independent processes on one physical GPU. Each
process runs a real tiny OPT forward/backward pass. Gloo averages CPU copies
of the CUDA gradients before AdamW, and the result is compared with a full-batch
reference implementing the pair/segment objective directly:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib GLOO_SOCKET_IFNAME=lo \
uv run --no-project --python /path/to/cuda-venv/bin/python \
  python -m recipe.equal_length.distributed_check \
  --work-dir /path/to/eqlen-dp-check \
  --rollout-proof /path/to/opt-gpu-check/joint_gpu_proof.json \
  --model-dir /path/to/opt-gpu-check/opt --steps 3
```

Run `gpu_check --family opt` first to create the model and native rollout proof.
The simulation covers GRPO, GSPO and RLOO with weighted KL, variable segment
counts, dummy rows, a rank with no active samples, and a globally all-tied batch.
It alternates microbatch sizes and checks gradients and parameters at every step.
The result is saved as `distributed_gpu_proof.json` with both process IDs.

The DP loss on each rank is the local weighted sum multiplied by `DP_size/K`,
where `K` is the original global subgroup count. Averaging gradients across ranks
then gives the same `1/K` global objective. Keep this count and the per-pair
`1/(2*N*L)` weights when sharding; dummy rows contribute zero weight. A local
all-tie rank still participates in reduction if another rank has active pairs.
Only a globally all-tied batch skips the actor update.

This checks the DP objective and Gloo process communication. Real multi-GPU
NCCL, FSDP sharding, TP/PP and multi-node network/weight transport require
separate hardware acceptance runs. The EqLen objective does not acquire a new
normalization rule when adding nodes; engineering validation must still cover
every configured backend and topology. Keep the recipe's padded input and
Ulysses size 1 constraints for hybrid recurrent models.
Native paired sampling currently requires vLLM DP=1 and PP=1 within each
replica. Multiple rollout replicas can serve a larger training pool; increasing
vLLM's internal DP setting is not a supported shortcut for this recipe.

On WSL, retain the environment's driver library path (this machine used
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`). The interpreter must contain a compatible
CUDA Torch build, the compiled patched vLLM checkout, and verl's dependencies.
Each run writes its resolved configuration, checkpoint and
`joint_gpu_proof.json` into the selected directory; failures leave
`completed=false`.

Recorded results and versions are in `gpu_validation.json` (archived outside this submission).
Additional fault reproductions and final-source revalidation are in
`bughunt_validation.json` (archived outside this submission): 13 additional GPU training
batches/updates, 281 CPU regressions, 164 native protocol/scheduler tests, and
11 float32 OPT CUDA Graph cases covering EOS renewal, KV page boundaries and
cancellation cleanup. Training includes mixed prompt lengths, 64-token response
budgets, two-row and dynamic microbatches, SDPA and the public CLI entrypoint.

The historical long FP16 MoE failure in that report passes with the V2 runner
and shared FP32 routing mode. FP16 output rounding made nearly equal expert scores tie
during decode while full prefill selected expert 1. Producing FP32 routing
logits from the original low-precision weights resolves this decision change;
casting the rounded logits afterward does not. The final native V2 vLLM check
repeats the 271-token prompt/640-token response-budget case without diagnostic
replacements: maximum token log-prob error falls from 0.0336223 to 0.000500679,
with the original 0.003 tolerance unchanged.

`attention_validation.json` (archived outside this submission) records binary provenance,
final source hashes, FlashAttention2/SDPA GPU training, native probability and
CUDA Graph attempts, and regression results. Historical reports remain unchanged.
These checks use tiny OPT and hybrid MoE Qwen3Next models; they do not establish
bitwise equivalence or certify every model/precision/backend combination.
An additional V1-runner long FP16 case still exceeds the same tolerance:
decode and full prefill differ in the router's input hidden states, changing
one nearly tied expert decision despite correctly computed FP32 logits.
Its eager/Graph failures are retained in the report and are not counted as
successful probability checks. The successful training matrix uses eager rollout.

These small synthetic runs verify the training pipeline. Qwen3-32B,
Qwen3-Next-80B, multi-GPU training, benchmark accuracy and measured throughput
have not been reproduced on the available 8 GB GPU.
Hybrid training uses Transformers' native Torch GDN reference operators on
CUDA tensors; its optional optimized training kernels were not installed.

Historical checks of the original full development tree also recorded these limitations: `pytest` encounters
a top-level exit in `papertest/zcpo_loss_equiv_test.py`; `pytest tests` has
12 collection errors from unavailable optional backends or library versions
in the CPU environment. Pre-commit checks pass apart from eight existing
paper/papertest/DAPO files without accepted license headers; all changed Python
files pass the scoped license check. These results are recorded separately
from the EqLen regression and GPU checks.
