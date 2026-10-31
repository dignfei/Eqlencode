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
"""Expand native paired rollouts before the standard old-policy PPO update."""

from math import lcm

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from recipe.equal_length.algorithms import training_metadata, trajectories, validate_rollout
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.utils.py_functional import rename_dict
from verl.workers.utils.padding import left_right_2_no_padding


def _objects(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class EqualLengthRayPPOTrainer(RayPPOTrainer):
    rollout_postprocessing = True

    def __init__(self, config, tokenizer, *args, **kwargs):
        self._validate_config(config, kwargs.get("processor"))
        super().__init__(config, tokenizer, *args, **kwargs)

    @staticmethod
    def _validate_config(config, processor=None):
        actor = config.actor_rollout_ref.actor
        rollout = config.actor_rollout_ref.rollout
        reference = config.actor_rollout_ref.ref
        model = config.actor_rollout_ref.get("model", {})
        if any(
            OmegaConf.select(config, f"actor_rollout_ref.{path}", default=1) != 1
            for path in (
                "actor.ulysses_sequence_parallel_size",
                "actor.fsdp_config.ulysses_sequence_parallel_size",
                "ref.fsdp_config.ulysses_sequence_parallel_size",
            )
        ):
            raise ValueError("EqLen currently requires Ulysses sequence parallel size 1.")
        if model.get("use_remove_padding", False) and (
            actor.use_dynamic_bsz
            or rollout.log_prob_use_dynamic_bsz
            or reference.log_prob_use_dynamic_bsz
            or any(
                (settings.get(key) or 1) != 1
                for settings, key in (
                    (actor, "ppo_micro_batch_size_per_gpu"),
                    (rollout, "log_prob_micro_batch_size_per_gpu"),
                    (reference, "log_prob_micro_batch_size_per_gpu"),
                )
            )
        ):
            raise ValueError(
                "EqLen remove-padding requires static one-row train/old/ref microbatches to isolate recurrent state; "
                "set actor_rollout_ref.model.use_remove_padding=false to batch multiple trajectories."
            )
        if model.get("lora_rank", 0) > 0 or model.get("lora", {}).get("rank", 0) > 0 or model.get("lora_adapter_path"):
            raise ValueError("EqLen currently supports full-weight training without LoRA.")
        if actor.get("strategy", "fsdp") not in ("fsdp", "fsdp2", "megatron") or any(
            OmegaConf.select(config, f"actor_rollout_ref.{role}.megatron.dynamic_context_parallel", default=False)
            for role in ("actor", "ref")
        ):
            raise ValueError("EqLen supports FSDP/Megatron with static data parallel groups.")
        if config.algorithm.adv_estimator != "eqlen" or rollout.n != 1:
            raise ValueError("EqLen requires adv_estimator=eqlen and rollout.n=1.")
        if type(config.eqlen.num_pairs) is not int or config.eqlen.num_pairs < 1:
            raise ValueError("eqlen.num_pairs must be a positive integer.")
        if config.eqlen.aggregation not in ("max", "mean"):
            raise ValueError("eqlen.aggregation must be max or mean.")
        if actor.policy_loss.loss_mode not in ("eqlen_grpo", "eqlen_gspo", "eqlen_rloo"):
            raise ValueError("EqLen requires a segment-aware EqLen policy loss.")
        if actor.ppo_epochs != 1:
            raise ValueError("EqLen currently performs one joint PPO epoch per rollout batch.")
        if actor.get("calculate_sum_pi_squared", False):
            raise ValueError("The trajectory-level variance proxy does not represent the EqLen segment objective.")
        if config.algorithm.use_kl_in_reward or actor.entropy_coeff != 0:
            raise ValueError("EqLen uses segment-weighted KL loss and no entropy regularization.")
        if (
            OmegaConf.select(config, "critic.enable")
            or OmegaConf.select(config, "distillation.enabled")
            or OmegaConf.select(config, "reward.reward_model.enable")
        ):
            raise ValueError("EqLen currently uses verifier rewards without critic, distillation, or reward model.")
        correction = config.algorithm.get("rollout_correction")
        if correction and any(correction.get(key) for key in ("bypass_mode", "rollout_rs", "rollout_is")):
            raise ValueError("EqLen requires recomputed old log-probs without rollout correction.")
        if (
            processor is not None
            or rollout.name != "vllm"
            or rollout.multi_turn.enable
            or rollout.agent.default_agent_loop != "equal_length_agent"
            or rollout.agent.get("agent_loop_config_path")
            or rollout.multi_turn.get("tool_config_path")
            or rollout.multi_turn.get("function_tool_path")
        ):
            raise ValueError("EqLen supports text-only native vLLM equal_length_agent rollouts.")
        if actor.get("use_prefix_grouper", False) or rollout.get("skip", {}).get("enable", False):
            raise ValueError("EqLen requires fresh rollouts and row-level balancing.")
        if any(
            value is not None
            for value in (
                actor.get("ppo_micro_batch_size"),
                rollout.get("log_prob_micro_batch_size"),
                config.actor_rollout_ref.ref.get("log_prob_micro_batch_size"),
            )
        ):
            raise ValueError("EqLen requires per-GPU micro batch sizes.")

    def _padding_divisor(self):
        actor, rollout, reference = (
            self.config.actor_rollout_ref.actor,
            self.config.actor_rollout_ref.rollout,
            self.config.actor_rollout_ref.ref,
        )
        actor_dp = self._get_dp_size(self.actor_rollout_wg, "actor")
        train_micro = 1 if actor.use_dynamic_bsz else actor.get("ppo_micro_batch_size_per_gpu") or 1
        old_micro = 1 if rollout.log_prob_use_dynamic_bsz else rollout.get("log_prob_micro_batch_size_per_gpu") or 1
        divisor = actor_dp * lcm(train_micro, old_micro)
        if self.use_reference_policy:
            ref_micro = (
                1 if reference.log_prob_use_dynamic_bsz else reference.get("log_prob_micro_batch_size_per_gpu") or 1
            )
            ref_dp = self._get_dp_size(self.ref_policy_wg, "ref")
            divisor = lcm(divisor, ref_dp * ref_micro)
        return divisor

    def _verify_trajectories(self, batch):
        manager = self.reward_loop_manager
        if manager is None or not manager.reward_loop_workers:
            raise ValueError("EqLen requires initialized verifier workers.")
        chunks = np.array_split(np.arange(len(batch)), min(len(batch), len(manager.reward_loop_workers)))
        results = ray.get(
            [
                worker.compute_score_batch.remote(batch.select_idxs(indices))
                for worker, indices in zip(manager.reward_loop_workers, chunks, strict=False)
            ]
        )
        rows = [row for chunk in results for row in chunk]
        if len(rows) != len(batch):
            raise ValueError("Verifier returned a different number of complete trajectories.")
        scores = torch.tensor([row["reward_score"] for row in rows], dtype=torch.float64)
        if not torch.isfinite(scores).all():
            raise ValueError("EqLen trajectory rewards must be finite scalars.")
        rm_scores = manager.reward_manager_cls.assemble_rm_scores(batch, scores.tolist())
        if not torch.isfinite(rm_scores).all():
            raise ValueError("EqLen trajectory rewards must fit the training reward tensor.")
        batch.batch["rm_scores"] = rm_scores
        extra = [row.get("reward_extra_info", {}) for row in rows]
        extra_keys = sorted(set.intersection(*(set(info) for info in extra)))
        for key in extra_keys:
            if key in batch.non_tensor_batch:
                raise ValueError(f"Verifier metadata collides with task field {key!r}.")
            batch.non_tensor_batch[key] = _objects([info[key] for info in extra])
        batch.meta_info["reward_extra_keys"] = extra_keys
        return scores.tolist()

    def _prepare_training_batch(self, batch, metrics, timing_raw):
        if batch.meta_info.get("validate", False):
            return batch
        required = {"uid", "raw_prompt", "reward_model", "extra_info", "data_source", "eqlen_rollouts"}
        if not required.issubset(batch.non_tensor_batch):
            raise ValueError(f"EqLen needs original task metadata: {sorted(required - batch.non_tensor_batch.keys())}")
        owners, groups, complete, subgroup_ids = [], [], [], []
        for parent, payload in enumerate(batch.non_tensor_batch["eqlen_rollouts"]):
            if not isinstance(payload, list) or len(payload) != self.config.eqlen.num_pairs:
                raise ValueError("Each EqLen carrier must contain eqlen.num_pairs subgroups.")
            prompt_width = batch.batch["prompts"].shape[1]
            prompt = batch.batch["prompts"][parent][
                batch.batch["attention_mask"][parent, :prompt_width].bool()
            ].tolist()
            for k, group in enumerate(payload):
                validate_rollout(group, expected_prompt_token_ids=prompt)
                offset = len(complete)
                rows = trajectories(group)
                groups.append((group, offset, len(rows)))
                complete.extend(rows)
                owners.extend([parent] * len(rows))
                subgroup_ids.extend([f"{batch.non_tensor_batch['uid'][parent]}-{k}"] * len(rows))
        if not complete:
            raise ValueError("EqLen needs at least one complete trajectory.")
        response_width = self.config.actor_rollout_ref.rollout.response_length
        if any(not row.response_token_ids or len(row.response_token_ids) > response_width for row in complete):
            raise ValueError("EqLen complete responses must be nonempty and fit the response budget.")
        expanded = batch.select_idxs(owners)
        prompts = expanded.batch["prompts"].clone()
        prompt_width = prompts.shape[1]
        prompt_mask = expanded.batch["attention_mask"][:, :prompt_width].clone()
        responses = prompts.new_full((len(complete), response_width), self.tokenizer.pad_token_id)
        response_mask = torch.zeros_like(responses)
        for row, trajectory in enumerate(complete):
            length = len(trajectory.response_token_ids)
            responses[row, :length] = responses.new_tensor(trajectory.response_token_ids)
            response_mask[row, :length] = 1
        attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
        metadata = {
            key: value.copy()
            for key, value in expanded.non_tensor_batch.items()
            if key not in {"eqlen_rollouts", *batch.meta_info.get("reward_extra_keys", [])}
        }
        metadata["multi_modal_inputs"] = _objects([{} for _ in complete])
        metadata["eqlen_subgroup_id"] = _objects(subgroup_ids)
        metadata["eqlen_trajectory_index"] = np.asarray([row.trajectory_index for row in complete])
        metadata["agent_name"] = _objects(["equal_length_agent"] * len(complete))
        expanded = DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=-1),
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": (attention_mask.cumsum(-1) - 1).clamp_min(0),
            },
            non_tensors=metadata,
            meta_info={
                key: value
                for key, value in batch.meta_info.items()
                if key not in ("metrics", "timing", "reward_extra_keys")
            },
        )
        with marked_timer("eqlen_verify", timing_raw):
            rewards = self._verify_trajectories(expanded)
        loss_mask = torch.zeros_like(response_mask, dtype=torch.bool)
        advantages = torch.zeros_like(response_mask, dtype=torch.float32)
        segment_ids = torch.full_like(response_mask, -1)
        loss_weights = torch.zeros_like(advantages)
        mode = "rloo" if self.config.actor_rollout_ref.actor.policy_loss.loss_mode == "eqlen_rloo" else "grpo"
        active_pairs = 0
        for group, offset, count in groups:
            plans = training_metadata(group, rewards[offset : offset + count], self.config.eqlen.aggregation, mode)
            for i, plan in enumerate(plans, start=offset):
                length = len(plan.loss_mask)
                loss_mask[i, :length] = torch.tensor(plan.loss_mask, device=loss_mask.device)
                advantages[i, :length] = advantages.new_tensor(plan.advantages)
                segment_ids[i, :length] = segment_ids.new_tensor(plan.segment_ids)
                loss_weights[i, :length] = loss_weights.new_tensor(plan.loss_weights)
            active_pairs += len({segment for plan in plans for segment in plan.segment_ids if segment >= 0}) // 2
        expanded.batch.update(
            {
                "eqlen_advantages": advantages,
                "eqlen_loss_mask": loss_mask,
                "eqlen_segment_ids": segment_ids,
                "eqlen_loss_weights": loss_weights,
            }
        )
        expanded.meta_info["global_num_subgroups"] = len(groups)
        generated_tokens = sum(
            len(pair["plus_token_ids"]) + len(pair["minus_token_ids"])
            for group, _, _ in groups
            for pair in group["pairs"]
        )
        expanded.meta_info["eqlen_generated_tokens"] = generated_tokens
        real_size = len(expanded)
        expanded, pad_size = pad_dataproto_to_divisor(expanded, self._padding_divisor())
        if pad_size:
            for key in ("rm_scores", "response_mask", "eqlen_advantages", "eqlen_loss_mask", "eqlen_loss_weights"):
                expanded.batch[key][real_size:].zero_()
            expanded.batch["eqlen_segment_ids"][real_size:].fill_(-1)
            for row in range(real_size, len(expanded)):
                identity = f"eqlen-pad:{self.global_steps}:{row}"
                expanded.non_tensor_batch["uid"][row] = identity
                expanded.non_tensor_batch["eqlen_subgroup_id"][row] = identity
                expanded.non_tensor_batch["eqlen_trajectory_index"][row] = -1
        metrics.update(
            {
                "eqlen/subgroups": len(groups),
                "eqlen/pairs": sum(len(group["pairs"]) for group, _, _ in groups),
                "eqlen/active_pairs": active_pairs,
                "eqlen/trajectory_reward_mean": sum(rewards) / len(rewards),
                "eqlen/train_rows": real_size,
                "eqlen/padding_rows": pad_size,
                "eqlen/generated_tokens": generated_tokens,
                "eqlen/trajectory_tokens": response_mask.sum().item(),
                "eqlen/train_context_tokens_with_padding": expanded.batch["attention_mask"].sum().item(),
            }
        )
        return expanded

    @staticmethod
    def _real_trajectories(batch):
        indices = np.flatnonzero(batch.non_tensor_batch["eqlen_trajectory_index"] >= 0)
        real = batch.select_idxs(indices)
        real.meta_info = dict(batch.meta_info)
        real.meta_info["global_token_num"] = real.batch["attention_mask"].sum(-1).tolist()
        return real, indices

    def _compute_training_metrics(self, batch, timing_raw, n_gpus, gradient_norm=None):
        real, _ = self._real_trajectories(batch)
        metrics = super()._compute_training_metrics(real, timing_raw, n_gpus, gradient_norm)
        generated_tokens = batch.meta_info["eqlen_generated_tokens"]
        metrics["eqlen/generated_tokens"] = generated_tokens
        metrics["eqlen/trajectory_tokens"] = real.batch["response_mask"].sum().item()
        metrics["eqlen/train_context_tokens_with_padding"] = batch.batch["attention_mask"].sum().item()
        if "gen" in timing_raw:
            metrics["timing_per_token_ms/gen"] = timing_raw["gen"] * 1000 / generated_tokens
            metrics["eqlen/tokens_per_second"] = generated_tokens / timing_raw["gen"] if timing_raw["gen"] > 0 else 0.0
        return metrics

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        real, indices = self._real_trajectories(batch)
        extras = {
            key: values[indices] if isinstance(values, np.ndarray) else [values[int(i)] for i in indices]
            for key, values in reward_extra_infos_dict.items()
        }
        return super()._log_rollout_data(real, extras, timing_raw, rollout_data_dir)

    def _update_actor(self, batch):
        if not batch.batch["eqlen_loss_mask"].any():
            return DataProto.from_single_dict(data={}, meta_info={"metrics": {"eqlen/skipped_all_tie_update": 1.0}})
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch.meta_info["multi_turn"] = False
        batch_td = left_right_2_no_padding(batch.to_tensordict())
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            distillation_use_topk=False,
            global_batch_size=len(batch),
            mini_batch_size=len(batch),
            global_num_subgroups=batch.meta_info["global_num_subgroups"],
            epochs=1,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            compute_loss=True,
        )
        output = self.actor_rollout_wg.update_actor(batch_td)
        metrics = rename_dict(tu.get(output, "metrics"), "actor/")
        metrics["perf/mfu/actor"] = metrics.pop("actor/mfu")
        metrics["eqlen/skipped_all_tie_update"] = 0.0
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})
