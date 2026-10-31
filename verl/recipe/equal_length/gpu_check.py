# Copyright 2026 Individual Contributor: d fei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run real local EqLen/FSDP2/native-vLLM updates and save their proof.

Run ``.venv/bin/python -m recipe.equal_length.gpu_check --work-dir /tmp/verl-eqlen-check``
in the configured GPU environment; add ``--kl`` to include weighted reference KL.
Assets are created locally; ``--family qwen3next`` checks FA/GDN/MoE instead of
the default OPT. The verifier scores complete synthetic responses;
this check measures transport, optimization and weight synchronization, not task
accuracy or throughput.
"""

import argparse
import json
import os
import sys
from math import isclose, isfinite
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
KLSTATS_ENV = {
    "KLSTATS_CODE_DIR": str(REPO_ROOT / "paper"),
    "KLSTATS_ENABLE": "0",
    "ZCPO_COMBINED": "0",
    "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
}
PROBE_PROMPT = [1, *range(5, 29), 4, 77]


def prepare_assets(work_dir: Path, family: str = "opt", binary_reward: bool = False) -> dict:
    """Create a deterministic tiny model, tokenizer, two-row dataset and verifier."""
    import pandas as pd
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast

    from verl.utils.tokenizer import normalize_token_ids

    if family not in ("opt", "qwen3next"):
        raise ValueError("GPU checker family must be opt or qwen3next.")
    work_dir = work_dir.resolve()
    model_dir = work_dir / family
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(41219)
    common = dict(
        vocab_size=128,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    if family == "opt":
        model = OPTForCausalLM(
            OPTConfig(**common, ffn_dim=256, word_embed_proj_dim=128, dropout=0.0, attention_dropout=0.0)
        )
    else:
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

        model = Qwen3NextForCausalLM(
            Qwen3NextConfig(
                **common,
                intermediate_size=256,
                num_key_value_heads=1,
                head_dim=64,
                layer_types=["linear_attention", "full_attention"],
                linear_conv_kernel_dim=4,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                num_experts=2,
                num_experts_per_tok=1,
                moe_intermediate_size=128,
                shared_expert_intermediate_size=128,
                mlp_only_layers=[0],
            )
        )
    model.save_pretrained(model_dir, safe_serialization=True)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"t{i}": i for i in range(4, 128)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=64,
    )
    tokenizer.chat_template = (
        "{{ bos_token }} "
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}t4{% endif %}"
    )
    tokenizer.save_pretrained(model_dir)
    rows = []
    for index in range(2):
        content = " ".join(f"t{5 + index * 24 + offset}" for offset in range(24))
        prompt = [{"role": "user", "content": content}]
        token_ids = normalize_token_ids(
            tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        )
        assert len(token_ids) == 26 and all(0 <= token < 128 for token in token_ids)
        rows.append(
            {
                "prompt": prompt,
                "data_source": "equal_length_gpu_smoke",
                "reward_model": {"style": "rule", "ground_truth": f"smoke-{index}"},
                "extra_info": {"index": index, "split": "train", "smoke": True},
            }
        )
    pd.DataFrame(rows).to_parquet(work_dir / "train.parquet", index=False)
    (work_dir / "reward.py").write_text(
        '"""Deterministic synthetic scores for complete-trajectory smoke verification."""\n'
        "import hashlib\n\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    assert data_source == 'equal_length_gpu_smoke'\n"
        "    digest = hashlib.sha256(solution_str.encode()).digest()\n"
        + (
            "    return float(digest[0] % 2)\n"
            if binary_reward
            else "    return int.from_bytes(digest[:8], 'big') / 2**64\n"
        )
    )
    return {
        "work_dir": str(work_dir),
        "model_dir": str(model_dir),
        "model_family": family,
        "rows": len(rows),
        "vocab_size": len(vocab),
    }


def compose_config(
    work_dir: Path,
    kl: bool = False,
    family: str = "opt",
    *,
    steps: int = 1,
    loss_mode: str = "eqlen_grpo",
    binary_reward: bool = False,
    train_attention: str | None = None,
):
    """Compose the real recipe with a small one-GPU training configuration."""
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    if family not in ("opt", "qwen3next"):
        raise ValueError("GPU checker family must be opt or qwen3next.")
    if type(steps) is not int or steps < 1:
        raise ValueError("GPU checker steps must be a positive integer.")
    if loss_mode not in ("eqlen_grpo", "eqlen_gspo", "eqlen_rloo"):
        raise ValueError("GPU checker loss_mode must be an EqLen loss.")
    if train_attention not in (None, "sdpa", "flash_attention_2"):
        raise ValueError("GPU checker train_attention must be sdpa or flash_attention_2.")
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "recipe/equal_length/config")):
        config = compose(
            "equal_length_trainer",
            overrides=[f'hydra.searchpath=["file://{REPO_ROOT / "verl/trainer/config"}"]'],
        )
    work_dir = work_dir.resolve()
    overrides = {
        "trainer": {
            "nnodes": 1,
            "n_gpus_per_node": 1,
            "total_training_steps": steps,
            "total_epochs": steps,
            "val_before_train": False,
            "test_freq": -1,
            "save_freq": 1,
            "resume_mode": "disable",
            "logger": ["console"],
            "default_local_dir": str(work_dir / "run"),
        },
        "data": {
            "train_files": str(work_dir / "train.parquet"),
            "val_files": str(work_dir / "train.parquet"),
            "train_batch_size": 2,
            "max_prompt_length": 32,
            "max_response_length": 16,
            "dataloader_num_workers": 0,
            "shuffle": False,
        },
        "actor_rollout_ref.model": {
            "path": str(work_dir / family),
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "enable_gradient_checkpointing": False,
        },
        "actor_rollout_ref.actor": {
            "strategy": "fsdp2",
            "use_dynamic_bsz": False,
            "ppo_micro_batch_size_per_gpu": 1,
            "ppo_epochs": 1,
            "use_kl_loss": kl,
            "kl_loss_coef": 0.002,
            "entropy_coeff": 0,
            "policy_loss": {"loss_mode": loss_mode},
        },
        "actor_rollout_ref.actor.optim": {"lr": 1e-3},
        "actor_rollout_ref.actor.fsdp_config": {
            "strategy": "fsdp2",
            "fsdp_size": 1,
            "use_torch_compile": False,
            "param_offload": True,
            "optimizer_offload": True,
        },
        "actor_rollout_ref.ref": {"strategy": "fsdp2", "log_prob_micro_batch_size_per_gpu": 1},
        "actor_rollout_ref.ref.fsdp_config": {
            "strategy": "fsdp2",
            "fsdp_size": 1,
            "use_torch_compile": False,
            "param_offload": True,
        },
        "actor_rollout_ref.rollout": {
            "name": "vllm",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "n": 1,
            "temperature": 1,
            "top_p": 1,
            "max_model_len": 64,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 256,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.20,
            "load_format": "auto",
            "enable_prefix_caching": True,
            "log_prob_micro_batch_size_per_gpu": 1,
            "log_prob_use_dynamic_bsz": False,
        },
        "actor_rollout_ref.rollout.agent": {"num_workers": 1},
        "actor_rollout_ref.rollout.checkpoint_engine": {
            "backend": "naive",
            "update_weights_bucket_megabytes": 4,
        },
        "actor_rollout_ref.rollout.engine_kwargs.vllm": {
            "async_scheduling": False,
            "attention_backend": "TRITON_ATTN",
        },
        "algorithm": {"use_kl_in_reward": False},
        "reward": {"num_workers": 1},
        "reward.reward_model": {"enable": False},
        "reward.custom_reward_function": {"path": str(work_dir / "reward.py"), "name": "compute_score"},
        "eqlen": {
            "num_pairs": 2,
            "gpu_check_family": family,
            "gpu_check_steps": steps,
            "gpu_check_binary_reward": binary_reward,
            "gpu_check_output": str(work_dir / "joint_gpu_proof.json"),
        },
        "ray_kwargs.ray_init": {
            "num_cpus": 4,
            "num_gpus": 1,
            "object_store_memory": 268435456,
            "_temp_dir": f"/tmp/eqlen-ray-{uuid4().hex[:12]}",
        },
    }
    from omegaconf import OmegaConf

    with open_dict(config):
        if train_attention is not None:
            config.actor_rollout_ref.model.override_config.attn_implementation = train_attention
        for prefix, fields in overrides.items():
            for name, value in fields.items():
                OmegaConf.update(config, f"{prefix}.{name}", value, merge=True)
        if family == "qwen3next":
            config.actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode = "align"
    return config


def _serializable(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _audit_projection(carrier, prepared, verified, loss_mode, aggregation):
    """Reconstruct the paper's tables directly from the wire, independently of recipe helpers."""
    import torch

    active = torch.zeros_like(prepared.batch["eqlen_loss_mask"])
    advantages = torch.zeros_like(prepared.batch["eqlen_advantages"])
    ids = torch.full_like(prepared.batch["eqlen_segment_ids"], -1)
    weights = torch.zeros_like(prepared.batch["eqlen_loss_weights"])
    locations = {
        (str(group), int(index)): row
        for row, (group, index) in enumerate(
            zip(
                prepared.non_tensor_batch["eqlen_subgroup_id"],
                prepared.non_tensor_batch["eqlen_trajectory_index"],
                strict=True,
            )
        )
        if index >= 0
    }
    rewards = {(row["subgroup"], row["trajectory"]): row for row in verified}
    assert len(rewards) == len(verified) == len(locations), "Verifier cardinality or identities differ."
    expected_rows = sum(
        len(group["pairs"]) + 1 for payload in carrier.non_tensor_batch["eqlen_rollouts"] for group in payload
    )
    assert len(verified) == expected_rows, "Each subgroup must have exactly N+1 complete verifier calls."
    group_count = pair_count = tie_count = relay_count = 0
    for parent, payload in enumerate(carrier.non_tensor_batch["eqlen_rollouts"]):
        width = carrier.batch["prompts"].shape[1]
        prompt = carrier.batch["prompts"][parent][carrier.batch["attention_mask"][parent, :width].bool()].tolist()
        for k, group in enumerate(payload):
            identity = f"{carrier.non_tensor_batch['uid'][parent]}-{k}"
            assert group["prompt_token_ids"] == prompt
            pairs = group["pairs"]
            context, complete = [], []
            for i, pair in enumerate(pairs):
                assert pair["prefix_len"] == len(context), "Inherited response prefix is wrong."
                assert len(pair["plus_token_ids"]) == len(pair["minus_token_ids"]) > 0
                assert pair["plus_finish_reason"] in ("stop", "length")
                assert (
                    pair["minus_finish_reason"] in ("stop", "length")
                    if i == len(pairs) - 1
                    else pair["minus_finish_reason"] is None
                )
                complete.append(context + pair["plus_token_ids"])
                context += pair["minus_token_ids"]
            complete.append(context)
            values = []
            for i, tokens in enumerate(complete):
                row = locations[(identity, i)]
                assert rewards[(identity, i)]["tokens"] == tokens, "Verifier did not score the complete response."
                actual = prepared.batch["input_ids"][row][prepared.batch["attention_mask"][row].bool()].tolist()
                assert actual == prompt + tokens, "Training lost the full conditional context."
                values.append(rewards[(identity, i)]["reward"])
            offset = 0
            for i, pair in enumerate(pairs):
                descendant_reward = (
                    max(values[i + 1 :]) if aggregation == "max" else sum(values[i + 1 :]) / len(values[i + 1 :])
                )
                difference = values[i] - descendant_reward
                length = len(pair["plus_token_ids"])
                if difference:
                    coefficient = difference if loss_mode == "eqlen_rloo" else (1.0 if difference > 0 else -1.0)
                    for trajectory, sign, segment in ((i, 1, 2 * i), (i + 1, -1, 2 * i + 1)):
                        row = locations[(identity, trajectory)]
                        active[row, offset : offset + length] = True
                        advantages[row, offset : offset + length] = sign * coefficient
                        ids[row, offset : offset + length] = segment
                        weights[row, offset : offset + length] = 1 / (2 * len(pairs) * length)
                else:
                    tie_count += 1
                offset += length
            group_count += 1
            pair_count += len(pairs)
            relay_count += len(pairs) - 1
    assert prepared.meta_info["global_num_subgroups"] == group_count
    assert torch.equal(prepared.batch["eqlen_loss_mask"], active)
    assert torch.equal(prepared.batch["eqlen_segment_ids"], ids)
    torch.testing.assert_close(prepared.batch["eqlen_advantages"], advantages)
    torch.testing.assert_close(prepared.batch["eqlen_loss_weights"], weights)
    assert not (active & ~prepared.batch["response_mask"].bool()).any()
    return {
        "passed": True,
        "subgroups": group_count,
        "pairs": pair_count,
        "ties": tie_count,
        "relay_pairs": relay_count,
        "active_tokens": int(active.sum()),
        "loss_weight_sum": float(weights.sum()),
    }


def _snapshot_actor(worker):
    """Inspect actual worker parameters and AdamW state without altering either."""
    import hashlib

    import torch
    import transformers
    import vllm

    from verl import DataProto

    engine = worker.actor.engine
    assert type(engine.optimizer) is torch.optim.AdamW, "The check requires the real Torch AdamW optimizer."
    parameters = {
        name: (parameter.to_local() if hasattr(parameter, "to_local") else parameter).detach().float().cpu()
        for name, parameter in engine.module.named_parameters()
    }
    if not hasattr(worker, "_eqlen_gpu_check_initial_parameters"):
        worker._eqlen_gpu_check_initial_parameters = {name: value.clone() for name, value in parameters.items()}
    maximum_delta = max(
        float((value - worker._eqlen_gpu_check_initial_parameters[name]).abs().max())
        for name, value in parameters.items()
    )
    steps = sorted({int(state["step"].item()) for state in engine.optimizer.state.values() if "step" in state})
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    snapshot = {
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "transformers_path": transformers.__file__,
        "vllm": vllm.__version__,
        "vllm_path": vllm.__file__,
        "optimizer": f"{type(engine.optimizer).__module__}.{type(engine.optimizer).__name__}",
        "parameter_max_delta": maximum_delta,
        "optimizer_steps": steps,
        "parameter_sha256": digest.hexdigest(),
        "attention_implementation": engine.module.config._attn_implementation,
        "router_fp32": getattr(engine.module.config, "eqlen_router_fp32", False),
        "fp32_router_count": sum(
            getattr(getattr(module.forward, "__func__", None), "__name__", "") == "_qwen3_next_fp32_router_forward"
            for module in engine.module.modules()
        ),
    }
    return DataProto.from_dict(tensors={"probe": torch.zeros(1)}, meta_info={"snapshot": snapshot})


def run_check(config):
    """Observe the standard TaskRunner and fit; never replace generation or optimization."""
    import ray
    import torch
    from omegaconf import OmegaConf, open_dict

    from recipe.equal_length.equal_length_ray_trainer import EqualLengthRayPPOTrainer
    from recipe.equal_length.main_equal_length import EqualLengthTaskRunner
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.device import auto_set_device

    class ObservedTrainer(EqualLengthRayPPOTrainer):
        def _verify_trajectories(self, batch):
            scores = super()._verify_trajectories(batch)
            assert "verified_complete_trajectories" not in self.step_proof, (
                "Complete trajectories must be verified once."
            )
            self.step_proof["verified_complete_trajectories"] = [
                {
                    "subgroup": str(batch.non_tensor_batch["eqlen_subgroup_id"][index]),
                    "trajectory": int(batch.non_tensor_batch["eqlen_trajectory_index"][index]),
                    "tokens": batch.batch["responses"][index][batch.batch["response_mask"][index].bool()].tolist(),
                    "reward": score,
                }
                for index, score in enumerate(scores)
            ]
            return scores

        def _prepare_training_batch(self, batch, metrics, timing_raw):
            self.step_proof = {
                "step": self.global_steps,
                "native_rollouts": _serializable(batch.non_tensor_batch["eqlen_rollouts"].tolist()),
            }
            self.proof["steps"].append(self.step_proof)
            prepared = super()._prepare_training_batch(batch, metrics, timing_raw)
            verified = self.step_proof["verified_complete_trajectories"]
            self.step_proof["expected_complete_trajectories"] = len(verified)
            self.step_proof["paper_projection_audit"] = _audit_projection(
                batch,
                prepared,
                verified,
                self.config.actor_rollout_ref.actor.policy_loss.loss_mode,
                self.config.eqlen.aggregation,
            )
            active = prepared.batch["eqlen_loss_mask"]
            weights = prepared.batch["eqlen_loss_weights"]
            assert torch.isfinite(weights).all()
            assert not (active & ~prepared.batch["response_mask"].bool()).any()
            assert torch.equal(prepared.batch["eqlen_segment_ids"] >= 0, active)
            assert torch.count_nonzero(weights[~active]) == 0
            self.step_proof["paper_metadata"] = {
                "rows": len(prepared),
                "subgroups": prepared.meta_info["global_num_subgroups"],
                "active_tokens": int(active.sum()),
                "excluded_response_tokens": int((prepared.batch["response_mask"].bool() & ~active).sum()),
                "loss_weight_sum": float(weights.sum()),
                "metrics": _serializable(metrics),
            }
            return prepared

        def _update_actor(self, batch):
            active = bool(batch.batch["eqlen_loss_mask"].any())
            self.step_proof["actor_updated"] = active
            if self.config.actor_rollout_ref.actor.use_kl_loss:
                assert "ref_log_prob" in batch.batch, "Weighted KL requires actual reference inference."
                self.step_proof["reference_logprobs_present"] = True
            output = super()._update_actor(batch)
            self.step_proof["actor_metrics"] = _serializable(output.meta_info["metrics"])
            if not active:
                assert output.meta_info["metrics"]["eqlen/skipped_all_tie_update"] == 1
                return output
            self.proof["actor_dispatches"] += 1
            norms = self.step_proof["actor_metrics"]["actor/grad_norm"]
            norms = norms if isinstance(norms, list) else [norms]
            assert all(isfinite(value) for value in norms) and max(norms) > 0
            if self.config.actor_rollout_ref.actor.use_kl_loss:
                losses = self.step_proof["actor_metrics"]["actor/kl_loss"]
                losses = losses if isinstance(losses, list) else [losses]
                assert losses and all(isfinite(value) for value in losses)
                coefficients = self.step_proof["actor_metrics"]["actor/kl_coef"]
                coefficients = coefficients if isinstance(coefficients, list) else [coefficients]
                assert all(isclose(value, 0.002, rel_tol=1e-6) for value in coefficients)
            return output

        def fit(self):
            from transformers import AutoConfig

            output_path = Path(self.config.eqlen.gpu_check_output)
            self.proof = {
                "completed": False,
                "source": "real TaskRunner/EqualLengthRayPPOTrainer/FSDP2/native vLLM",
                "kl_enabled": bool(self.config.actor_rollout_ref.actor.use_kl_loss),
                "model_family": self.config.eqlen.get("gpu_check_family", "opt"),
                "requested_steps": int(self.config.eqlen.get("gpu_check_steps", 1)),
                "loss_mode": self.config.actor_rollout_ref.actor.policy_loss.loss_mode,
                "binary_reward": bool(self.config.eqlen.get("gpu_check_binary_reward", False)),
                "steps": [],
                "actor_dispatches": 0,
                "model_config": AutoConfig.from_pretrained(
                    self.config.actor_rollout_ref.model.path, local_files_only=True
                ).to_dict(),
            }
            original_sync = self.checkpoint_manager.update_weights

            def observed_sync(global_steps):
                result = original_sync(global_steps)
                snapshot = self.actor_rollout_wg.execute_with_func_generator(_snapshot_actor).meta_info["snapshot"]
                assert snapshot["cuda_available"] and snapshot["torch_cuda"]
                assert (
                    snapshot["attention_implementation"]
                    == self.config.actor_rollout_ref.model.override_config.attn_implementation
                )
                if self.config.eqlen.get("gpu_check_family", "opt") == "qwen3next":
                    assert snapshot["router_fp32"] and snapshot["fp32_router_count"] > 0
                updates = self.proof["actor_dispatches"]
                assert max(snapshot["optimizer_steps"], default=0) == updates
                if updates:
                    assert snapshot["parameter_max_delta"] > 0
                previous = self.proof.get("actor_snapshots", [])
                if previous:
                    changed = snapshot["parameter_sha256"] != previous[-1]["parameter_sha256"]
                    assert changed == self.step_proof["actor_updated"], "Parameter changes disagree with the tie rule."
                    if not changed:
                        assert snapshot["optimizer_steps"] == previous[-1]["optimizer_steps"]
                self.proof.setdefault("actor_snapshots", []).append(snapshot)
                server = self.llm_server_manager.server_handles[0]
                output = ray.get(
                    server.generate.remote(
                        prompt_ids=PROBE_PROMPT,
                        sampling_params={"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0, "logprobs": True},
                        request_id=f"eqlen-weight-proof-{global_steps}-{uuid4().hex}",
                    )
                )
                assert output.extra_fields["global_steps"] == global_steps
                token_ids = [row[0] for row in output.extra_fields["prompt_ids"][:-1]]
                logprobs = [row[0] for row in output.extra_fields["prompt_logprobs"][:-1]]
                assert token_ids == PROBE_PROMPT[1:]
                self.proof.setdefault("weight_sync_fixed_token_probes", []).append(
                    {
                        "step": global_steps,
                        "fixed_prompt_ids": PROBE_PROMPT,
                        "predicted_token_ids": token_ids,
                        "logprobs": logprobs,
                    }
                )
                probes = self.proof["weight_sync_fixed_token_probes"]
                if len(probes) > 1:
                    delta = max(abs(a - b) for a, b in zip(probes[-2]["logprobs"], logprobs, strict=True))
                    self.step_proof["vllm_fixed_token_max_logprob_delta"] = delta
                    assert (delta > 1e-5) if self.step_proof["actor_updated"] else (delta <= 1e-5)
                output_path.write_text(json.dumps(_serializable(self.proof), indent=2, allow_nan=False))
                return result

            self.checkpoint_manager.update_weights = observed_sync
            try:
                super().fit()
                probes = self.proof["weight_sync_fixed_token_probes"]
                steps = self.proof["requested_steps"]
                assert [record["step"] for record in probes] == list(range(steps + 1))
                assert [record["step"] for record in self.proof["steps"]] == list(range(1, steps + 1))
                maximum_delta = max(
                    abs(a - b) for a, b in zip(probes[0]["logprobs"], probes[-1]["logprobs"], strict=True)
                )
                self.proof["updated_vllm_fixed_token_max_logprob_delta"] = maximum_delta
                assert maximum_delta > 1e-5, "Synced actor weights must change fixed-token vLLM probabilities."
                assert self.proof["actor_dispatches"] > 0
                if steps == 1:
                    self.proof.update(self.proof["steps"][0])
                self.proof["completed"] = True
            except BaseException as error:
                self.proof["error"] = repr(error)
                raise
            finally:
                self.checkpoint_manager.update_weights = original_sync
                output_path.write_text(json.dumps(_serializable(self.proof), indent=2, allow_nan=False))
                print(
                    "EQLEN_GPU_PROOF",
                    json.dumps(
                        {
                            key: self.proof.get(key)
                            for key in ("completed", "requested_steps", "actor_dispatches", "error")
                        }
                    ),
                    flush=True,
                )

    class ObservedTaskRunner(EqualLengthTaskRunner):
        trainer_class = ObservedTrainer

    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    with open_dict(config):
        runtime_env = config.ray_kwargs.ray_init.get("runtime_env") or {}
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.update(KLSTATS_ENV)
        env_vars.pop("RAY_RUNTIME_ENV_HOOK", None)
        env_vars["VLLM_TARGET_DEVICE"] = "cuda"
        env_vars["PYTHONPATH"] = os.pathsep.join(
            (str(REPO_ROOT), str(REPO_ROOT / "paper"), env_vars.get("PYTHONPATH", os.getenv("PYTHONPATH", "")))
        )
        if "LD_LIBRARY_PATH" in os.environ:
            env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        runtime_env["env_vars"] = env_vars
        config.ray_kwargs.ray_init.runtime_env = runtime_env
    OmegaConf.save(config, Path(config.eqlen.gpu_check_output).parent / "actual_run_config.yaml", resolve=True)
    try:
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ObservedTaskRunner))
    finally:
        ray.shutdown()


def main():
    """Prepare local assets and run the checker in the current GPU interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/verl-eqlen-check"))
    parser.add_argument("--kl", action="store_true", help="Enable the weighted reference KL extension.")
    parser.add_argument("--family", choices=("opt", "qwen3next"), default="opt")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--loss-mode", choices=("eqlen_grpo", "eqlen_gspo", "eqlen_rloo"), default="eqlen_grpo")
    parser.add_argument("--binary-reward", action="store_true", help="Use synthetic binary complete-response rewards.")
    parser.add_argument("--train-attention", choices=("sdpa", "flash_attention_2"), default=None)
    args = parser.parse_args()
    os.environ.update(KLSTATS_ENV)
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    import torch

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    proof_path = work_dir / "joint_gpu_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "completed": False,
                "kl_enabled": args.kl,
                "model_family": args.family,
                "interpreter": sys.executable,
            }
        )
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Run this checker with the configured CUDA Torch/native-vLLM GPU environment.")
        if args.steps < 1:
            raise ValueError("GPU checker steps must be a positive integer.")
        assets = prepare_assets(work_dir, args.family, args.binary_reward)
        print("EQLEN_GPU_ASSETS", json.dumps(assets), flush=True)
        run_check(
            compose_config(
                work_dir,
                args.kl,
                args.family,
                steps=args.steps,
                loss_mode=args.loss_mode,
                binary_reward=args.binary_reward,
                train_attention=args.train_attention,
            )
        )
        assert json.loads(proof_path.read_text())["completed"], "The remote trainer did not complete its proof."
    except BaseException as error:
        proof = json.loads(proof_path.read_text())
        proof.update(completed=False, error=repr(error))
        proof_path.write_text(json.dumps(proof, indent=2, allow_nan=False))
        raise


if __name__ == "__main__":
    main()
