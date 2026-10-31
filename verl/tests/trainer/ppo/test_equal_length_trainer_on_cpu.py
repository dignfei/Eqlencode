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
"""Guard complete-trajectory expansion and one joint EqLen actor update."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from recipe.equal_length.equal_length_ray_trainer import EqualLengthRayPPOTrainer
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.utils import tensordict_utils as tu


def objects(values):
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def config():
    return OmegaConf.create(
        {
            "eqlen": {"num_pairs": 1, "aggregation": "max", "seed": 42},
            "algorithm": {"adv_estimator": "eqlen", "use_kl_in_reward": False, "rollout_correction": None},
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {"loss_mode": "eqlen_grpo"},
                    "ppo_epochs": 1,
                    "entropy_coeff": 0,
                    "use_kl_loss": True,
                    "use_dynamic_bsz": False,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "data_loader_seed": 42,
                    "shuffle": False,
                },
                "ref": {"log_prob_use_dynamic_bsz": False, "log_prob_micro_batch_size_per_gpu": 2},
                "rollout": {
                    "name": "vllm",
                    "n": 1,
                    "temperature": 1.0,
                    "response_length": 8,
                    "log_prob_use_dynamic_bsz": False,
                    "log_prob_micro_batch_size_per_gpu": 1,
                    "multi_turn": {"enable": False},
                    "agent": {"default_agent_loop": "equal_length_agent", "num_workers": 1},
                },
            },
            "critic": {"enable": False},
            "distillation": {"enabled": False},
            "reward": {"reward_model": {"enable": False}},
        }
    )


def group(lengths=(1, 2, 1, 2)):
    pairs, prefix = [], 0
    for i, length in enumerate(lengths):
        pairs.append(
            {
                "prefix_len": prefix,
                "plus_token_ids": [10 * i + 1] * length,
                "minus_token_ids": [10 * i + 2] * length,
                "plus_finish_reason": "stop",
                "minus_finish_reason": "stop" if i == len(lengths) - 1 else None,
                "plus_logprobs": None,
                "minus_logprobs": None,
            }
        )
        prefix += length
    return {"prompt_token_ids": [41, 42], "pairs": pairs}


def carrier(groups=None):
    groups = groups or [[group()], [group((2,))]]
    rows = len(groups)
    prompts = torch.tensor([[0, 0, 41, 42]] * rows)
    responses = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]] * rows)
    mask = responses.ne(0).long()
    attention = torch.cat((prompts.ne(0).long(), mask), dim=-1)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": mask,
            "input_ids": torch.cat((prompts, responses), dim=-1),
            "attention_mask": attention,
            "position_ids": (attention.cumsum(-1) - 1).clamp_min(0),
            "rm_scores": torch.full_like(mask, 99.0, dtype=torch.float32),
        },
        non_tensors={
            "eqlen_rollouts": objects(groups),
            "uid": objects([f"p{i}" for i in range(rows)]),
            "raw_prompt": objects([[{"role": "user", "content": f"question{i}"}] for i in range(rows)]),
            "reward_model": objects([{"ground_truth": f"answer{i}"} for i in range(rows)]),
            "extra_info": objects([{"parent": i} for i in range(rows)]),
            "data_source": objects(["math"] * rows),
        },
        meta_info={"reward_extra_keys": [], "global_steps": 7},
    )


def trainer(monkeypatch, cfg=None, rewards=None):
    def initialize(self, cfg, tokenizer, *args, **kwargs):
        self.config, self.tokenizer = cfg, tokenizer
        self.use_reference_policy = True
        self.use_critic = False

    monkeypatch.setattr(RayPPOTrainer, "__init__", initialize)
    instance = EqualLengthRayPPOTrainer(cfg or config(), SimpleNamespace(pad_token_id=0))
    instance.global_steps = 7
    instance.actor_calls, instance.verified_rows = [], []
    instance._get_dp_size = lambda *_: 3
    iterator = iter(rewards if rewards is not None else [0, 0, 0, 1, 0, 1, 1])

    def verify(chunk):
        result = []
        for row in chunk:
            length = row.batch["attention_mask"][-8:].sum().item()
            tokens = row.batch["responses"][:length].tolist()
            instance.verified_rows.append((row.non_tensor_batch["uid"], tokens))
            result.append({"reward_score": next(iterator), "reward_extra_info": {"verifier": "ok"}})
        return result

    workers = [SimpleNamespace(compute_score_batch=SimpleNamespace(remote=verify)) for _ in range(3)]
    instance.reward_loop_manager = SimpleNamespace(
        reward_loop_workers=workers,
        reward_manager_cls=RewardManagerBase,
    )
    monkeypatch.setattr("recipe.equal_length.equal_length_ray_trainer.ray.get", lambda refs: refs)

    def update(batch):
        instance.actor_calls.append(batch)
        return tu.get_tensordict({}, {"metrics": {"mfu": 0.2, "pg_loss": -0.1}})

    instance.actor_rollout_wg = SimpleNamespace(update_actor=update)
    instance.ref_policy_wg = instance.actor_rollout_wg
    return instance


def cpu_unpad(input_ids, attention_mask):
    indices = attention_mask.flatten().bool().nonzero(as_tuple=True)[0]
    lengths = attention_mask.sum(-1, dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32)))
    return input_ids.flatten(0, 1)[indices], indices, offsets, lengths.max().item()


def test_equal_length_full_rows_reward_once_metadata_and_joint_update(monkeypatch):
    instance = trainer(monkeypatch)
    metrics, timing = {}, {}
    prepared = instance._prepare_training_batch(carrier(), metrics, timing)
    assert len(instance.verified_rows) == 7
    assert [tokens for _, tokens in instance.verified_rows[:5]] == [
        [1],
        [2, 11, 11],
        [2, 12, 12, 21],
        [2, 12, 12, 22, 31, 31],
        [2, 12, 12, 22, 32, 32],
    ]
    assert len(prepared) == 12
    assert prepared.meta_info["global_num_subgroups"] == 2
    assert "eqlen_rollouts" not in prepared.non_tensor_batch
    assert prepared.non_tensor_batch["eqlen_subgroup_id"][:7].tolist() == ["p0-0"] * 5 + ["p1-0"] * 2
    expected_masks = [
        [1],
        [1, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
    ]
    for i, expected in enumerate(expected_masks):
        assert prepared.batch["eqlen_loss_mask"][i, : len(expected)].tolist() == expected
    torch.testing.assert_close(prepared.batch["eqlen_advantages"][1, :3], torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(prepared.batch["eqlen_loss_weights"][1, :3], torch.tensor([1 / 8, 1 / 16, 1 / 16]))
    assert not prepared.batch["eqlen_loss_mask"][5:].any()
    assert prepared.batch["response_mask"][5:7].sum() == 4
    assert not prepared.batch["response_mask"][7:].any()
    assert prepared.batch["attention_mask"][7:].sum() > 0
    assert prepared.batch["eqlen_segment_ids"][7:].eq(-1).all()
    assert prepared.batch["rm_scores"][7:].eq(0).all()
    assert len(set(prepared.non_tensor_batch["uid"][7:])) == 5
    assert metrics["eqlen/active_pairs"] == 4 and metrics["eqlen/pairs"] == 5
    assert metrics["eqlen/padding_rows"] == 5
    assert "eqlen_verify" in timing

    permutation = torch.arange(len(prepared) - 1, -1, -1)
    prepared.reorder(permutation)
    assert prepared.non_tensor_batch["eqlen_subgroup_id"][-1] == "p0-0"
    expected = prepared.batch["eqlen_advantages"].clone()
    compute_advantage(prepared, "eqlen", config=instance.config.algorithm)
    torch.testing.assert_close(prepared.batch["advantages"], expected)
    prepared.batch["old_log_probs"] = torch.zeros_like(expected)
    prepared.batch["ref_log_prob"] = torch.zeros_like(expected)
    monkeypatch.setattr("verl.workers.utils.padding.unpad_input", cpu_unpad)
    output = instance._update_actor(prepared)
    assert len(instance.actor_calls) == 1
    sent = instance.actor_calls[0]
    assert tu.get_non_tensor_data(sent, "global_batch_size", None) == 12
    assert tu.get_non_tensor_data(sent, "mini_batch_size", None) == 12
    assert tu.get_non_tensor_data(sent, "global_num_subgroups", None) == 2
    assert tu.get_non_tensor_data(sent, "epochs", None) == 1
    assert sent["input_ids"].is_nested
    torch.testing.assert_close(sent["eqlen_advantages"], expected)
    assert output.meta_info["metrics"]["eqlen/skipped_all_tie_update"] == 0


def test_equal_length_all_ties_skip_optimizer_dispatch(monkeypatch):
    instance = trainer(monkeypatch, rewards=[1] * 7)
    prepared = instance._prepare_training_batch(carrier(), {}, {})
    compute_advantage(prepared, "eqlen", config=instance.config.algorithm)
    assert not prepared.batch["advantages"].any()
    assert prepared.meta_info["global_num_subgroups"] == 2
    result = instance._update_actor(prepared)
    assert not instance.actor_calls
    assert result.meta_info["metrics"]["eqlen/skipped_all_tie_update"] == 1


def test_equal_length_preserves_small_continuous_reward_differences(monkeypatch):
    instance = trainer(monkeypatch, rewards=[1.0, 1.00000001])
    prepared = instance._prepare_training_batch(carrier([[group((2,))]]), {}, {})
    assert prepared.batch["eqlen_loss_mask"].sum() == 4
    torch.testing.assert_close(prepared.batch["eqlen_advantages"][0, :2], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(prepared.batch["eqlen_advantages"][1, :2], torch.tensor([1.0, 1.0]))


def test_equal_length_rejects_reward_tensor_overflow(monkeypatch):
    instance = trainer(monkeypatch, rewards=[1e100, 0])
    with pytest.raises(ValueError, match="fit the training reward tensor"):
        instance._prepare_training_batch(carrier([[group((2,))]]), {}, {})


def test_equal_length_multiple_subgroups_keep_original_task_identity(monkeypatch):
    cfg = config()
    cfg.eqlen.num_pairs = 2
    instance = trainer(monkeypatch, cfg, rewards=[0, 1, 1, 0])
    prepared = instance._prepare_training_batch(carrier([[group((2,)), group((1,))]]), {}, {})
    assert len(instance.verified_rows) == 4
    assert prepared.meta_info["global_num_subgroups"] == 2
    assert prepared.non_tensor_batch["uid"][:4].tolist() == ["p0"] * 4
    assert prepared.non_tensor_batch["eqlen_subgroup_id"][:4].tolist() == ["p0-0", "p0-0", "p0-1", "p0-1"]
    assert prepared.non_tensor_batch["eqlen_trajectory_index"][:4].tolist() == [0, 1, 0, 1]
    assert all(meta["ground_truth"] == "answer0" for meta in prepared.non_tensor_batch["reward_model"][:4])
    torch.testing.assert_close(prepared.batch["eqlen_loss_weights"][0, :2], torch.tensor([0.25, 0.25]))
    assert prepared.batch["eqlen_loss_weights"][2, 0].item() == 0.5


@pytest.mark.parametrize("mode", ["eqlen_grpo", "eqlen_gspo", "eqlen_rloo"])
def test_equal_length_modes_preserve_pairwise_metadata(monkeypatch, mode):
    cfg = config()
    cfg.actor_rollout_ref.actor.policy_loss.loss_mode = mode
    instance = trainer(monkeypatch, cfg, rewards=[0.2, 0.5])
    batch = instance._prepare_training_batch(carrier([[group((2,))]]), {}, {})
    expected = -0.3 if mode == "eqlen_rloo" else -1.0
    assert batch.batch["eqlen_advantages"][0, 0].item() == pytest.approx(expected)


@pytest.mark.parametrize("field", ["eqlen_advantages", "eqlen_loss_weights", "eqlen_loss_mask", "eqlen_segment_ids"])
def test_equal_length_advantage_rejects_misaligned_or_invalid_fields(monkeypatch, field):
    instance = trainer(monkeypatch)
    batch = instance._prepare_training_batch(carrier(), {}, {})
    if field == "eqlen_advantages":
        batch.batch[field][0, 0] = float("nan")
    elif field == "eqlen_loss_weights":
        batch.batch[field][0, 0] = -1
    elif field == "eqlen_loss_mask":
        batch.batch[field][0, -1] = True
    else:
        batch.batch[field][0, -1] = 0
    with pytest.raises(ValueError, match="Invalid EqLen"):
        compute_advantage(batch, "eqlen", config=instance.config.algorithm)


def test_equal_length_rejects_different_wire_prompt_before_verification(monkeypatch):
    instance = trainer(monkeypatch)
    batch = carrier()
    batch.non_tensor_batch["eqlen_rollouts"][0][0]["prompt_token_ids"] = [43, 44]
    with pytest.raises(ValueError, match="prompt"):
        instance._prepare_training_batch(batch, {}, {})
    assert not instance.verified_rows


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor_rollout_ref.rollout.n", 2),
        ("eqlen.num_pairs", 0),
        ("actor_rollout_ref.actor.ppo_epochs", 2),
        ("algorithm.use_kl_in_reward", True),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla"),
        ("reward.reward_model.enable", True),
        ("actor_rollout_ref.model.lora_rank", 4),
        ("actor_rollout_ref.model.lora.rank", 4),
        ("actor_rollout_ref.model.lora_adapter_path", "/tmp/adapter"),
        ("actor_rollout_ref.actor.megatron.dynamic_context_parallel", True),
        ("actor_rollout_ref.actor.strategy", "torchtitan"),
        ("actor_rollout_ref.actor.calculate_sum_pi_squared", True),
        ("actor_rollout_ref.actor.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size", 2),
        ("actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size", 2),
    ],
)
def test_equal_length_rejects_unsupported_configuration(path, value):
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        EqualLengthRayPPOTrainer._validate_config(cfg)


def test_equal_length_config_composes_without_changing_other_advantages():
    directory = Path(__file__).resolve().parents[3] / "recipe/equal_length/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name="equal_length_trainer")
    EqualLengthRayPPOTrainer._validate_config(cfg)
    assert cfg.actor_rollout_ref.rollout.n == 1 and cfg.eqlen.num_pairs == 8
    assert not cfg.actor_rollout_ref.model.use_remove_padding
    ordinary = carrier([[group((2,))]])
    ordinary.batch["token_level_rewards"] = ordinary.batch["rm_scores"].float()
    ordinary = ordinary.repeat(2)
    compute_advantage(ordinary, "grpo", config=OmegaConf.create({}))
    assert ordinary.batch["advantages"].eq(0).all()


@pytest.mark.parametrize(
    "path,value",
    [
        ("actor.use_dynamic_bsz", True),
        ("rollout.log_prob_use_dynamic_bsz", True),
        ("ref.log_prob_use_dynamic_bsz", True),
        ("actor.ppo_micro_batch_size_per_gpu", 2),
        ("rollout.log_prob_micro_batch_size_per_gpu", 2),
        ("ref.log_prob_micro_batch_size_per_gpu", 2),
    ],
)
def test_equal_length_rejects_cross_trajectory_recurrent_packing(path, value):
    cfg = config()
    OmegaConf.update(cfg, "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1)
    OmegaConf.update(cfg, f"actor_rollout_ref.{path}", value)
    EqualLengthRayPPOTrainer._validate_config(cfg)
    OmegaConf.update(cfg, "actor_rollout_ref.model.use_remove_padding", True)
    with pytest.raises(ValueError, match="isolate recurrent state"):
        EqualLengthRayPPOTrainer._validate_config(cfg)


def test_equal_length_allows_static_single_trajectory_remove_padding():
    cfg = config()
    OmegaConf.update(cfg, "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1)
    OmegaConf.update(cfg, "actor_rollout_ref.model.use_remove_padding", True)
    EqualLengthRayPPOTrainer._validate_config(cfg)


def test_equal_length_metrics_and_dump_exclude_reordered_dummy_trajectories(monkeypatch):
    instance = trainer(monkeypatch)
    batch = instance._prepare_training_batch(carrier(), {}, {})
    batch.batch["token_level_scores"] = batch.batch["rm_scores"].clone()
    batch.batch["token_level_rewards"] = batch.batch["rm_scores"].clone()
    batch.batch["sum_pi_squared"] = torch.full_like(batch.batch["rm_scores"], 0.8)
    batch.batch["old_log_probs"] = torch.full_like(batch.batch["rm_scores"], -1.0)
    compute_advantage(batch, "eqlen", config=instance.config.algorithm)
    permutation = torch.arange(len(batch) - 1, -1, -1)
    batch.reorder(permutation)
    batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
    original_meta = batch.meta_info.copy()
    timing = {"gen": 2.0, "step": 4.0, "update_actor": 1.0}

    ordinary = RayPPOTrainer._compute_training_metrics(instance, batch, timing, 2)
    assert ordinary["critic/score/mean"] == pytest.approx(3 / 12)
    metrics = instance._compute_training_metrics(batch, timing, 2, gradient_norm=0.5)
    assert metrics["critic/score/mean"] == pytest.approx(3 / 7)
    assert metrics["critic/rewards/mean"] == pytest.approx(3 / 7)
    assert metrics["response_length/mean"] == pytest.approx(24 / 7)
    assert metrics["perf/total_num_tokens"] == 38
    assert metrics["perf/throughput"] == 38 / (4 * 2)
    assert metrics["eqlen/generated_tokens"] == 16
    assert metrics["eqlen/trajectory_tokens"] == 24
    assert metrics["eqlen/train_context_tokens_with_padding"] == batch.batch["attention_mask"].sum().item()
    assert metrics["timing_per_token_ms/gen"] == 2 * 1000 / 16
    assert metrics["eqlen/tokens_per_second"] == 16 / 2
    assert metrics["timing_per_token_ms/update_actor"] == pytest.approx(1000 / 38)
    assert metrics["variance_proxy/proxy1_signal_strength"] == 0.25
    assert batch.meta_info == original_meta

    dumped = []
    instance.tokenizer.batch_decode = lambda rows, **_: [str(row.tolist()) for row in rows]
    instance._dump_generations = lambda **record: dumped.append(record)
    row_numbers = np.arange(len(batch))
    instance._log_rollout_data(batch, {"row_number": row_numbers}, {}, "/tmp/eqlen-test-dump")
    assert len(dumped) == 1 and len(dumped[0]["outputs"]) == 7
    assert sum(dumped[0]["scores"]) == 3
    indices = np.flatnonzero(batch.non_tensor_batch["eqlen_trajectory_index"] >= 0)
    assert dumped[0]["reward_extra_infos_dict"]["row_number"] == indices.tolist()
