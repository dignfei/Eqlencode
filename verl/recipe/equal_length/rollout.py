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
"""Collect native pair groups through the existing rollout server pool."""

import asyncio
import time
from typing import Any
from uuid import uuid4

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    register,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.llm_server import LLMServerClient


class EqualLengthServerClient(LLMServerClient):
    """Route a complete paired request to one existing vLLM server."""

    async def generate_equal_length_pairs(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        num_pairs: int,
    ) -> dict[str, Any]:
        """Generate pair groups, cancelling the corresponding RPC if interrupted."""
        server_id, server = await self._acquire_server(request_id)
        future = None
        try:
            future = server.generate_equal_length_pairs.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                num_pairs=num_pairs,
            )
            return await future
        except asyncio.CancelledError:
            if future is not None:
                ray.cancel(future)
            raise
        finally:
            self._release_server(server_id)


@register("equal_length_agent")
class EqualLengthAgentLoop(SingleTurnAgentLoop):
    """Return one representative row while preserving all pair groups."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Tokenize a text prompt and collect native subgroup metadata."""
        if self.processor is not None:
            raise ValueError("Equal-length rollouts support text-only prompts.")
        config = self.rollout_config.custom["equal_length"]
        sampling_params = dict(sampling_params)
        if "__eqlen_sampling_seed__" in kwargs:
            sampling_params.setdefault("seed", int(kwargs["__eqlen_sampling_seed__"]))
        prompt_ids = normalize_token_ids(await self.apply_chat_template(list(kwargs["raw_prompt"])))
        start = time.perf_counter()
        output = await self.server_manager.generate_equal_length_pairs(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            num_pairs=config["num_pairs"],
        )
        groups = output["eqlen_rollouts"]
        if len(groups) != config["num_pairs"] or any(group["prompt_token_ids"] != prompt_ids for group in groups):
            raise ValueError("Equal-length metadata does not match the source prompt.")
        response_ids = output["token_ids"]
        response_logprobs = output["log_probs"]
        if not response_ids or len(response_ids) > self.response_length:
            raise ValueError("Equal-length representative responses must fit the response budget.")
        if sampling_params.get("logprobs") and response_logprobs is None:
            raise ValueError("Equal-length requests require their sampled-token logprobs.")
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise ValueError("Equal-length logprobs must match representative tokens.")
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            response_logprobs=response_logprobs,
            num_turns=2,
            metrics=AgentLoopMetrics(generate_sequences=time.perf_counter() - start),
            extra_fields={
                "eqlen_rollouts": groups,
                "global_steps": output["global_steps"],
                "turn_scores": [],
                "tool_rewards": [],
            },
        )


class EqualLengthAgentLoopWorker(AgentLoopWorker):
    """Reuse standard padding, with paired training and ordinary validation."""

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Keep one response row per input and defer all paired rewards."""
        batch = DataProto(
            batch=batch.batch,
            non_tensor_batch=dict(batch.non_tensor_batch),
            meta_info=dict(batch.meta_info),
        )
        validate = batch.meta_info.get("validate", False)
        if not validate and self.rollout_config.n != 1:
            raise ValueError("Equal-length training requires rollout.n=1.")
        batch.non_tensor_batch["agent_name"] = np.full(
            len(batch), "single_turn_agent" if validate else "equal_length_agent", dtype=object
        )
        return await super().generate_sequences(batch)

    async def _compute_score(self, outputs: list[AgentLoopOutput], kwargs: dict) -> None:
        if "eqlen_rollouts" not in outputs[-1].extra_fields:
            await super()._compute_score(outputs, kwargs)


class EqualLengthAgentLoopManager(AgentLoopManager):
    """Reuse Ray workers and load balancing without changing row cardinality."""

    def __init__(self, config, llm_client, teacher_client=None, reward_loop_worker_handles=None):
        self.agent_loop_workers_class = ray.remote(EqualLengthAgentLoopWorker)
        client = EqualLengthServerClient(config, load_balancer_handle=llm_client._load_balancer)
        super().__init__(config, client, teacher_client, reward_loop_worker_handles)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Assign reproducible per-row seed ranges before splitting across workers."""
        prompts = DataProto(
            batch=prompts.batch,
            non_tensor_batch=dict(prompts.non_tensor_batch),
            meta_info=dict(prompts.meta_info),
        )
        config = self.rollout_config.custom["equal_length"]
        base_seed = config.get("seed")
        if not prompts.meta_info.get("validate", False) and base_seed is not None:
            num_pairs = config["num_pairs"]
            response_length = self.rollout_config.response_length
            # Native groups use k * 2**20; leave room for every renewal index.
            group_stride = 1 << 20
            if num_pairs > 1 and response_length + 2 > group_stride:
                raise ValueError("The response budget overlaps native subgroup seed ranges.")
            stride = (num_pairs - 1) * group_stride + response_length + 2
            modulus = ((1 << 63) // stride - 1) * stride
            step = prompts.meta_info.get("global_steps", 0)
            batch_size = self.config.data.train_batch_size
            if type(step) is not int or step < 0 or len(prompts) > batch_size or modulus <= 0:
                raise ValueError("Invalid equal-length sampling seed range.")
            prompts.non_tensor_batch["__eqlen_sampling_seed__"] = np.array(
                [(base_seed + stride * (step * batch_size + row)) % modulus for row in range(len(prompts))],
                dtype=np.int64,
            )
        return await super().generate_sequences(prompts)
