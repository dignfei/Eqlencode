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
"""Paper-defined EqLen trajectory, reward and segment metadata projection.

The inference boundary uses plain dictionaries so training workers need not
import vLLM. Every subgroup keeps N+1 complete trajectories for verification
and training; inherited tokens remain conditioning inputs, with zero direct
loss. Global subgroup averaging belongs to the trainer/loss callback.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Integral, Real


@dataclass(frozen=True)
class CompleteTrajectory:
    """One complete response, in the inference subgroup's termination order."""

    trajectory_index: int
    response_token_ids: list[int]
    logprobs: list[float] | None
    finish_reason: str


@dataclass(frozen=True)
class TrainingMetadata:
    """Direct token-loss metadata aligned with one complete response."""

    loss_mask: list[bool]
    advantages: list[float]
    segment_ids: list[int]
    loss_weights: list[float]


def _tokens(value, name):
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"{name} must contain nonempty token IDs")
    if any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in value):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return list(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be finite real numbers")
    return float(value)


def validate_rollout(rollout, expected_prompt_token_ids=None, require_logprobs=False) -> None:
    """Validate a native, completed subgroup before reward or training."""
    if not isinstance(rollout, Mapping) or not {"prompt_token_ids", "pairs"}.issubset(rollout):
        raise ValueError("EqLen rollout requires prompt_token_ids and pairs")
    prompt = _tokens(rollout["prompt_token_ids"], "prompt_token_ids")
    if expected_prompt_token_ids is not None and prompt != list(expected_prompt_token_ids):
        raise ValueError("EqLen subgroup prompt differs from its original input")
    pairs = rollout["pairs"]
    if not isinstance(pairs, list | tuple) or not pairs:
        raise ValueError("EqLen rollout must contain at least one completed pair")
    # The native wire counts inherited response tokens, excluding the question.
    prefix_len = 0
    logprob_presence = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping) or not {
            "prefix_len",
            "plus_token_ids",
            "minus_token_ids",
            "plus_finish_reason",
            "minus_finish_reason",
        }.issubset(pair):
            raise ValueError("Invalid EqLen segment pair fields")
        if (
            isinstance(pair["prefix_len"], bool)
            or not isinstance(pair["prefix_len"], Integral)
            or pair["prefix_len"] != prefix_len
        ):
            raise ValueError("EqLen prefix_len must equal the previous minus response length, excluding the prompt")
        plus = _tokens(pair["plus_token_ids"], "plus_token_ids")
        minus = _tokens(pair["minus_token_ids"], "minus_token_ids")
        if len(plus) != len(minus):
            raise ValueError("EqLen segments must have equal lengths")
        if pair["plus_finish_reason"] not in ("stop", "length"):
            raise ValueError("EqLen plus trajectory must terminate naturally")
        last = index == len(pairs) - 1
        if (last and pair["minus_finish_reason"] not in ("stop", "length")) or (
            not last and pair["minus_finish_reason"] is not None
        ):
            raise ValueError("Only the final EqLen minus trajectory may terminate")
        for side, tokens in (("plus", plus), ("minus", minus)):
            values = pair.get(f"{side}_logprobs")
            logprob_presence.add(values is not None)
            if values is not None:
                if not isinstance(values, list | tuple) or len(values) != len(tokens):
                    raise ValueError("EqLen logprobs must align with every segment token")
                for value in values:
                    _finite(value, "EqLen logprobs")
        prefix_len += len(minus)
    if len(logprob_presence) != 1 or (require_logprobs and False in logprob_presence):
        raise ValueError("EqLen logprobs must be present consistently across the subgroup")


def trajectories(rollout) -> list[CompleteTrajectory]:
    """Recover N+1 complete responses, without assigning segment rewards."""
    validate_rollout(rollout)
    context = []
    context_logprobs = [] if rollout["pairs"][0].get("plus_logprobs") is not None else None
    result = []
    for index, pair in enumerate(rollout["pairs"]):
        result.append(
            CompleteTrajectory(
                trajectory_index=index,
                response_token_ids=context + list(pair["plus_token_ids"]),
                logprobs=(None if context_logprobs is None else context_logprobs + list(pair["plus_logprobs"])),
                finish_reason=pair["plus_finish_reason"],
            )
        )
        context += list(pair["minus_token_ids"])
        if context_logprobs is not None:
            context_logprobs += list(pair["minus_logprobs"])
    result.append(
        CompleteTrajectory(
            trajectory_index=len(result),
            response_token_ids=context,
            logprobs=context_logprobs,
            finish_reason=rollout["pairs"][-1]["minus_finish_reason"],
        )
    )
    return result


def segment_rewards(rollout, rewards: Sequence[float], aggregation="max") -> list[tuple[float, float]]:
    """Score plus directly and minus through all its complete descendants."""
    validate_rollout(rollout)
    if aggregation not in ("max", "mean"):
        raise ValueError("EqLen reward aggregation must be max or mean")
    if len(rewards) != len(rollout["pairs"]) + 1:
        raise ValueError("EqLen needs one reward for each of its N+1 complete trajectories")
    values = [_finite(value, "EqLen trajectory rewards") for value in rewards]
    result = []
    for index in range(len(values) - 1):
        descendants = values[index + 1 :]
        inherited = max(descendants) if aggregation == "max" else fsum(v / len(descendants) for v in descendants)
        result.append((values[index], inherited))
    return result


def training_metadata(rollout, rewards: Sequence[float], aggregation="max", advantage="grpo") -> list[TrainingMetadata]:
    """Project every active comparison segment once onto complete responses.

    Population-standardized two-member GRPO is exactly sign(reward difference).
    RLOO retains the difference. Ties have zero PG/KL/entropy contribution but
    remain in the subgroup's N denominator.
    """
    if advantage not in ("grpo", "rloo"):
        raise ValueError("EqLen advantage must be grpo or rloo")
    complete = trajectories(rollout)
    pairs = rollout["pairs"]
    pair_rewards = segment_rewards(rollout, rewards, aggregation)
    result = [
        TrainingMetadata(
            loss_mask=[False] * len(row.response_token_ids),
            advantages=[0.0] * len(row.response_token_ids),
            segment_ids=[-1] * len(row.response_token_ids),
            loss_weights=[0.0] * len(row.response_token_ids),
        )
        for row in complete
    ]
    offset = 0
    for index, (pair, (plus_reward, minus_reward)) in enumerate(zip(pairs, pair_rewards, strict=True)):
        length = len(pair["plus_token_ids"])
        if plus_reward != minus_reward:
            coefficient = (
                float(1 if plus_reward > minus_reward else -1)
                if advantage == "grpo"
                else _finite(plus_reward - minus_reward, "EqLen advantage")
            )
            weight = 1.0 / (2 * len(pairs) * length)
            for row, sign, segment_id in ((index, 1, 2 * index), (index + 1, -1, 2 * index + 1)):
                sample = result[row]
                for token in range(offset, offset + length):
                    sample.loss_mask[token] = True
                    sample.advantages[token] = sign * coefficient
                    sample.segment_ids[token] = segment_id
                    sample.loss_weights[token] = weight
        offset += length
    return result
