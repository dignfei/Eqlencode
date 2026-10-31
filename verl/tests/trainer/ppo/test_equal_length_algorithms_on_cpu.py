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
"""Hand-derived paper goldens for the plain-wire EqLen training boundary."""

from copy import deepcopy
from math import fsum

import pytest

from recipe.equal_length.algorithms import segment_rewards, training_metadata, trajectories, validate_rollout


@pytest.fixture
def rollout():
    return {
        "prompt_token_ids": [1, 2],
        "pairs": [
            {
                "prefix_len": 0,
                "plus_token_ids": [11],
                "minus_token_ids": [12],
                "plus_finish_reason": "stop",
                "minus_finish_reason": None,
                "plus_logprobs": [-0.11],
                "minus_logprobs": [-0.12],
            },
            {
                "prefix_len": 1,
                "plus_token_ids": [21, 22],
                "minus_token_ids": [23, 24],
                "plus_finish_reason": "stop",
                "minus_finish_reason": None,
                "plus_logprobs": [-0.21, -0.22],
                "minus_logprobs": [-0.23, -0.24],
            },
            {
                "prefix_len": 3,
                "plus_token_ids": [31],
                "minus_token_ids": [32],
                "plus_finish_reason": "stop",
                "minus_finish_reason": None,
                "plus_logprobs": [-0.31],
                "minus_logprobs": [-0.32],
            },
            {
                "prefix_len": 4,
                "plus_token_ids": [41, 42],
                "minus_token_ids": [43, 44],
                "plus_finish_reason": "stop",
                "minus_finish_reason": "length",
                "plus_logprobs": [-0.41, -0.42],
                "minus_logprobs": [-0.43, -0.44],
            },
        ],
    }


def test_complete_trajectories_keep_context_logprobs_and_termination_order(rollout):
    before = deepcopy(rollout)
    validate_rollout(rollout, expected_prompt_token_ids=[1, 2], require_logprobs=True)
    complete = trajectories(rollout)
    assert [row.trajectory_index for row in complete] == [0, 1, 2, 3, 4]
    assert [row.response_token_ids for row in complete] == [
        [11],
        [12, 21, 22],
        [12, 23, 24, 31],
        [12, 23, 24, 32, 41, 42],
        [12, 23, 24, 32, 43, 44],
    ]
    assert [row.logprobs for row in complete] == [
        [-0.11],
        [-0.12, -0.21, -0.22],
        [-0.12, -0.23, -0.24, -0.31],
        [-0.12, -0.23, -0.24, -0.32, -0.41, -0.42],
        [-0.12, -0.23, -0.24, -0.32, -0.43, -0.44],
    ]
    assert [row.finish_reason for row in complete] == ["stop"] * 4 + ["length"]
    assert rollout == before
    # Returned contexts must not alias wire tokens or another trajectory.
    complete[0].response_token_ids.append(99)
    complete[-1].logprobs.append(-9.0)
    assert rollout == before
    assert complete[1].response_token_ids == [12, 21, 22]


def test_paper_unique_success_projects_each_segment_once(rollout):
    before = deepcopy(rollout)
    rewards = [0, 0, 0, 1, 0]
    assert segment_rewards(rollout, rewards) == [(0, 1), (0, 1), (0, 1), (1, 0)]
    samples = training_metadata(rollout, rewards)
    assert [s.loss_mask for s in samples] == [
        [True],
        [True, True, True],
        [False, True, True, True],
        [False, False, False, True, True, True],
        [False, False, False, False, True, True],
    ]
    assert [s.advantages for s in samples] == [
        [-1.0],
        [1.0, -1.0, -1.0],
        [0.0, 1.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
    ]
    assert [s.segment_ids for s in samples] == [
        [0],
        [1, 2, 2],
        [-1, 3, 3, 4],
        [-1, -1, -1, 5, 6, 6],
        [-1, -1, -1, -1, 7, 7],
    ]
    assert [s.loss_weights for s in samples] == [
        [0.125],
        [0.125, 0.0625, 0.0625],
        [0.0, 0.0625, 0.0625, 0.125],
        [0.0, 0.0, 0.0, 0.125, 0.0625, 0.0625],
        [0.0, 0.0, 0.0, 0.0, 0.0625, 0.0625],
    ]
    assert fsum(fsum(s.loss_weights) for s in samples) == 1.0
    assert rollout == before and rewards == [0, 0, 0, 1, 0]


def test_mean_descendant_rewards_keep_rloo_magnitude(rollout):
    assert segment_rewards(rollout, [0, 0, 0, 1, 0], "mean") == pytest.approx([(0, 0.25), (0, 1 / 3), (0, 0.5), (1, 0)])
    samples = training_metadata(rollout, [0, 0, 0, 1, 0], aggregation="mean", advantage="rloo")
    expected = [
        [-0.25],
        [0.25, -1 / 3, -1 / 3],
        [0, 1 / 3, 1 / 3, -0.5],
        [0, 0, 0, 0.5, 1, 1],
        [0, 0, 0, 0, -1, -1],
    ]
    for sample, advantages in zip(samples, expected, strict=True):
        assert sample.advantages == pytest.approx(advantages)


def _tie_rollout(rollout):
    subgroup = deepcopy(rollout)
    subgroup["pairs"] = [subgroup["pairs"][0], subgroup["pairs"][1]]
    last = subgroup["pairs"][-1]
    last["plus_token_ids"], last["minus_token_ids"] = [21, 22, 23], [24, 25, 26]
    last["plus_logprobs"], last["minus_logprobs"] = [-0.21, -0.22, -0.23], [-0.24, -0.25, -0.26]
    last["minus_finish_reason"] = "length"
    return subgroup


def test_tie_stays_in_pair_and_subgroup_denominators(rollout):
    tied = training_metadata(_tie_rollout(rollout), [1, 1, 0])
    assert [s.loss_mask for s in tied] == [[False], [False, True, True, True], [False, True, True, True]]
    assert [s.segment_ids for s in tied] == [[-1], [-1, 2, 2, 2], [-1, 3, 3, 3]]
    assert [s.advantages for s in tied] == [[0.0], [0.0, 1.0, 1.0, 1.0], [0.0, -1.0, -1.0, -1.0]]
    assert tied[0].loss_weights == [0.0]
    assert tied[1].loss_weights == pytest.approx([0.0, 1 / 12, 1 / 12, 1 / 12])
    assert tied[2].loss_weights == pytest.approx([0.0, 1 / 12, 1 / 12, 1 / 12])
    assert fsum(fsum(s.loss_weights) for s in tied) == 0.5

    full = training_metadata(rollout, [0, 0, 0, 1, 0])
    segment_losses = [2, 3, 5, 7, 11, 13, 17, 19]
    full_loss = fsum(
        weight * segment_losses[segment_id]
        for sample in full
        for segment_id, weight in zip(sample.segment_ids, sample.loss_weights, strict=True)
        if segment_id >= 0
    )
    tie_loss = fsum(
        weight * {2: 5, 3: 9}[segment_id]
        for sample in tied
        for segment_id, weight in zip(sample.segment_ids, sample.loss_weights, strict=True)
        if segment_id >= 0
    )
    # N=4 subgroup: sum(eight segment losses)/8; N=2: (5+9)/4.
    assert (full_loss + tie_loss) / 2 == pytest.approx(6.5625)


@pytest.mark.parametrize("advantage, expected", [("grpo", 1.0), ("rloo", 0.3)])
def test_continuous_pair_grpo_uses_population_sign_and_rloo_difference(advantage, expected):
    subgroup = {
        "prompt_token_ids": [1, 2],
        "pairs": [
            {
                "prefix_len": 0,
                "plus_token_ids": [51, 52, 53],
                "minus_token_ids": [54, 55, 56],
                "plus_finish_reason": "stop",
                "minus_finish_reason": "length",
            }
        ],
    }
    plus, minus = training_metadata(subgroup, [0.2, 0.5], advantage=advantage)
    assert plus.advantages == pytest.approx([-expected] * 3)
    assert minus.advantages == pytest.approx([expected] * 3)
    assert plus.loss_weights == minus.loss_weights == pytest.approx([1 / 6] * 3)


def test_all_ties_have_zero_update_and_keep_complete_contexts(rollout):
    for advantage in ("grpo", "rloo"):
        metadata = training_metadata(rollout, [1] * 5, advantage=advantage)
        assert [len(s.loss_mask) for s in metadata] == [1, 3, 4, 6, 6]
        assert all(not any(s.loss_mask) and not any(s.advantages) and not any(s.loss_weights) for s in metadata)
        assert all(all(segment_id == -1 for segment_id in s.segment_ids) for s in metadata)
    assert trajectories(rollout)[-1].response_token_ids == [12, 23, 24, 32, 43, 44]


def test_rollout_without_logprobs_projects_without_fabricating_them(rollout):
    for pair in rollout["pairs"]:
        pair.pop("plus_logprobs")
        pair.pop("minus_logprobs")
    validate_rollout(rollout)
    assert all(row.logprobs is None for row in trajectories(rollout))
    with pytest.raises(ValueError, match="logprobs"):
        validate_rollout(rollout, require_logprobs=True)


@pytest.mark.parametrize(
    "invalid_case, error",
    [
        ("wrong_prefix_len", "prefix_len"),
        ("unequal_segments", "equal lengths"),
        ("empty_segment", "nonempty token IDs"),
        ("logprob_length", "logprobs must align"),
        ("nonfinite_logprob", "logprobs must be finite"),
        ("missing_required_logprob", "logprobs must be present"),
        ("missing_final_reason", "final EqLen minus"),
        ("premature_minus_reason", "final EqLen minus"),
        ("errored_plus", "plus trajectory must terminate"),
        ("boolean_token", "nonnegative integer token IDs"),
        ("reward_count", "one reward"),
        ("nonfinite_reward", "trajectory rewards must be finite"),
        ("prompt_mismatch", "differs from its original input"),
    ],
)
def test_invalid_wire_or_reward_fails_before_training_projection(rollout, invalid_case, error):
    rewards = [0, 0, 0, 1, 0]
    options = {}
    if invalid_case == "wrong_prefix_len":
        rollout["pairs"][1]["prefix_len"] = 0
    elif invalid_case == "unequal_segments":
        rollout["pairs"][0]["minus_token_ids"].append(99)
    elif invalid_case == "empty_segment":
        rollout["pairs"][0]["plus_token_ids"] = rollout["pairs"][0]["minus_token_ids"] = []
    elif invalid_case == "logprob_length":
        rollout["pairs"][1]["minus_logprobs"] = [-0.1]
    elif invalid_case == "nonfinite_logprob":
        rollout["pairs"][0]["plus_logprobs"] = [float("nan")]
    elif invalid_case == "missing_required_logprob":
        rollout["pairs"][0]["plus_logprobs"] = None
        options["require_logprobs"] = True
    elif invalid_case == "missing_final_reason":
        rollout["pairs"][-1]["minus_finish_reason"] = None
    elif invalid_case == "premature_minus_reason":
        rollout["pairs"][0]["minus_finish_reason"] = "stop"
    elif invalid_case == "errored_plus":
        rollout["pairs"][1]["plus_finish_reason"] = "error"
    elif invalid_case == "boolean_token":
        rollout["pairs"][0]["plus_token_ids"] = [True]
    elif invalid_case == "reward_count":
        rewards = [0]
    elif invalid_case == "nonfinite_reward":
        rewards[0] = float("inf")
    elif invalid_case == "prompt_mismatch":
        options["expected_prompt_token_ids"] = [1, 3]
    with pytest.raises(ValueError, match=error):
        if options:
            validate_rollout(rollout, **options)
        else:
            training_metadata(rollout, rewards)
