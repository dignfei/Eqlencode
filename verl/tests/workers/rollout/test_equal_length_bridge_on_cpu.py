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

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import wraps
from itertools import pairwise
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from recipe.equal_length import rollout
from verl.protocol import DataProto


@pytest.mark.parametrize("explicit", [None, "1"])
def test_entrypoint_sets_uv_default_before_ray_import(explicit):
    env = dict(os.environ)
    env.pop("RAY_ENABLE_UV_RUN_RUNTIME_ENV", None)
    if explicit is not None:
        env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = explicit
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import recipe.equal_length.main_equal_length; "
            "from ray._private import ray_constants; "
            f"assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is {explicit == '1'}",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_async(test):
    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


@dataclass
class _Pair:
    prefix_len: int = 0
    plus_token_ids: tuple = (10, 11)
    minus_token_ids: tuple = (12, 13)
    plus_finish_reason: str = "stop"
    minus_finish_reason: str = "length"
    plus_logprobs: tuple = (-0.1, -0.2)
    minus_logprobs: tuple = (-0.3, -0.4)


@dataclass
class _Rollout:
    prompt_token_ids: list[int]
    pairs: list[_Pair]

    def trajectories(self):
        pair = self.pairs[0]
        return [SimpleNamespace(response_token_ids=list(pair.plus_token_ids), logprobs=list(pair.plus_logprobs))]


def _wire(num_pairs=8):
    groups = [asdict(_Rollout([1, 2], [_Pair()])) for _ in range(num_pairs)]
    return {"token_ids": [10, 11], "log_probs": [-0.1, -0.2], "eqlen_rollouts": groups, "global_steps": 4}


@pytest.fixture
def server_class():
    """Execute the real bridge methods without importing GPU server dependencies."""
    path = Path(rollout.__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer")
    names = {
        "_equal_length_config",
        "_validate_configs",
        "run_server",
        "generate_equal_length_pairs",
        "lora_as_adapter",
    }
    cls.body = [
        node for node in cls.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    namespace = {
        "asdict": asdict,
        "inspect": inspect,
        "logger": logging.getLogger(__name__),
        "normalize_token_ids": rollout.normalize_token_ids,
        "get_max_position_embeddings": lambda config: config.max_positions,
        "UsageContext": SimpleNamespace(OPENAI_API_SERVER="server"),
        "SamplingParams": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["vLLMHttpServer"], namespace


@pytest.fixture
def server(server_class):
    cls, _ = server_class
    instance = object.__new__(cls)
    instance.config = SimpleNamespace(
        custom={"equal_length": {"num_pairs": 8, "seed": 17}},
        prompt_length=4,
        response_length=3,
        max_model_len=8,
        max_num_batched_tokens=8,
        max_num_seqs=16,
        repetition_penalty=1.0,
        n=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        multi_turn=SimpleNamespace(enable=False),
        quantization=None,
        qat={},
        enable_rollout_routing_replay=False,
        mtp=None,
    )
    instance.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(max_positions=16), lora_rank=0, lora={}, processor=None, tokenizer=[0, 1]
    )
    instance.engine = SimpleNamespace(engine_core=SimpleNamespace(supports_native_equal_length_pairs=True))
    instance.global_steps = 4
    instance._dp_master_port = 123
    instance._server_address = "127.0.0.1"
    instance.replica_rank = instance.node_rank = 0
    return instance


@pytest.fixture
def native_helper(monkeypatch):
    helper = AsyncMock(return_value=[_Rollout([1, 2], [_Pair()]) for _ in range(8)])
    for name in ("vllm", "vllm.entrypoints", "vllm.entrypoints.rl", "vllm.entrypoints.rl.equal_length_rollout"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.entrypoints.rl.equal_length_rollout"].generate_equal_length_pairs_async = helper
    return helper


@pytest.mark.parametrize(
    "config",
    [[], {}, {"num_pairs": True}, {"num_pairs": 0}, {"num_pairs": 8, "seed": True}, {"num_pairs": 8, "unknown": 1}],
)
def test_pair_server_rejects_invalid_descriptor(server, config):
    server.config.custom = {"equal_length": config}
    with pytest.raises(ValueError):
        server._equal_length_config()


@pytest.mark.parametrize(
    "field,value",
    [
        ("n", 2),
        ("data_parallel_size", 2),
        ("pipeline_model_parallel_size", 2),
        ("max_num_seqs", 1),
        ("max_num_batched_tokens", 7),
        ("quantization", "fp8"),
        ("enable_rollout_routing_replay", True),
        ("mtp", SimpleNamespace(enable=True)),
    ],
)
def test_pair_server_rejects_unsupported_engine_settings(server, field, value):
    setattr(server.config, field, value)
    with pytest.raises(ValueError, match="Unsupported"):
        server._validate_configs()


@_run_async
@pytest.mark.parametrize("supported", [False, True])
async def test_pair_server_builds_synchronous_core_before_engine_config(server, server_class, supported):
    _, namespace = server_class
    engine_args = SimpleNamespace(async_scheduling=None, enable_log_requests=False, disable_log_stats=True)
    config = SimpleNamespace(parallel_config=SimpleNamespace())

    def build_config(**kwargs):
        assert engine_args.async_scheduling is False
        return config

    engine_args.create_engine_config = Mock(side_effect=build_config)
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(supports_native_equal_length_pairs=supported),
        model_config=server.model_config,
        reset_mm_cache=AsyncMock(),
        collective_rpc=AsyncMock(),
        get_supported_tasks=AsyncMock(return_value=("generate",)),
    )
    namespace["AsyncEngineArgs"] = SimpleNamespace(from_cli_args=Mock(return_value=engine_args))
    namespace["AsyncLLM"] = SimpleNamespace(from_vllm_config=Mock(return_value=engine))
    namespace["build_app"] = lambda args, supported_tasks=None, model_config=None: SimpleNamespace(
        state=SimpleNamespace()
    )
    namespace["init_app_state"] = AsyncMock()
    namespace["run_uvicorn"] = AsyncMock(return_value=(345, None))
    if not supported:
        with pytest.raises(ValueError, match="does not support"):
            await server.run_server(argparse.Namespace())
        engine.reset_mm_cache.assert_not_awaited()
        namespace["run_uvicorn"].assert_not_awaited()
        return
    await server.run_server(argparse.Namespace())
    assert server.engine is engine
    assert server._server_port == 345
    assert config.parallel_config.data_parallel_master_port == 123


@_run_async
async def test_pair_server_calls_native_helper_once_and_preserves_input(server, native_helper):
    params = {"max_new_tokens": 20, "logprobs": True, "temperature": 0.7}
    original = dict(params)

    async def produce(*args, **kwargs):
        server.global_steps = 5
        return [_Rollout([1, 2], [_Pair()]) for _ in range(8)]

    native_helper.side_effect = produce
    result = await server.generate_equal_length_pairs([1, 2], params, "request", 8)
    native_helper.assert_awaited_once()
    call = native_helper.await_args
    assert call.args[:2] == (server.engine, [1, 2])
    assert vars(call.args[2]) == {
        "max_tokens": 3,
        "logprobs": 0,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "seed": 17,
    }
    assert call.kwargs == {"num_pairs": 8, "request_id_prefix": "request"}
    assert result == _wire()
    assert params == original
    assert json.loads(json.dumps(result))["global_steps"] == 4


@_run_async
@pytest.mark.parametrize(
    "invalid", ["pairs", "float_pairs", "empty", "prompt_length", "batch_budget", "headroom", "capability"]
)
async def test_pair_server_rejects_invalid_requests_before_helper(server, native_helper, invalid):
    prompt = [1, 2]
    num_pairs = 8
    if invalid == "pairs":
        num_pairs = 7
    elif invalid == "float_pairs":
        num_pairs = 8.0
    elif invalid == "empty":
        prompt = []
    elif invalid == "prompt_length":
        prompt = [1] * 5
    elif invalid == "batch_budget":
        server.config.max_num_batched_tokens = 3
    elif invalid == "headroom":
        server.config.max_model_len = 2
    else:
        server.engine.engine_core.supports_native_equal_length_pairs = False
    with pytest.raises(ValueError):
        await server.generate_equal_length_pairs(prompt, {}, "request", num_pairs)
    native_helper.assert_not_awaited()


@_run_async
@pytest.mark.parametrize(
    "params",
    [
        {"max_tokens": 0},
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"max_tokens": "3"},
        {"max_tokens": 3, "max_new_tokens": 3},
    ],
)
async def test_pair_server_rejects_ambiguous_or_noninteger_budgets(server, native_helper, params):
    original = dict(params)
    with pytest.raises(ValueError):
        await server.generate_equal_length_pairs([1, 2], params, "request", 8)
    assert params == original
    native_helper.assert_not_awaited()


@_run_async
@pytest.mark.parametrize("bad_output", [[], [_Rollout([9], [_Pair()]) for _ in range(8)]])
async def test_pair_server_rejects_wrong_group_or_prompt(server, native_helper, bad_output):
    native_helper.return_value = bad_output
    with pytest.raises(RuntimeError, match="requested groups"):
        await server.generate_equal_length_pairs([1, 2], {}, "request", 8)


@_run_async
@pytest.mark.parametrize("outcome", ["success", "error", "cancel", "dispatch_error"])
async def test_pair_client_releases_load_balancer_for_every_outcome(monkeypatch, outcome):
    client = object.__new__(rollout.EqualLengthServerClient)
    future = asyncio.get_running_loop().create_future()
    if outcome == "success":
        future.set_result(_wire())
    elif outcome == "error":
        future.set_exception(ValueError("backend failed"))
    elif outcome == "cancel":
        future.cancel()
    remote = Mock(return_value=future)
    if outcome == "dispatch_error":
        remote.side_effect = ValueError("dispatch failed")
    client._acquire_server = AsyncMock(
        return_value=("server", SimpleNamespace(generate_equal_length_pairs=SimpleNamespace(remote=remote)))
    )
    client._release_server = Mock()
    cancel = Mock()
    monkeypatch.setattr(rollout.ray, "cancel", cancel)
    call = client.generate_equal_length_pairs("sticky", prompt_ids=[1, 2], sampling_params={}, num_pairs=8)
    if outcome == "success":
        assert await call == _wire()
    else:
        with pytest.raises(asyncio.CancelledError if outcome == "cancel" else ValueError):
            await call
    client._acquire_server.assert_awaited_once_with("sticky")
    client._release_server.assert_called_once_with("server")
    if outcome == "cancel":
        cancel.assert_called_once_with(future)
    else:
        cancel.assert_not_called()


@_run_async
@pytest.mark.parametrize("validate", [False, True])
async def test_pair_worker_keeps_one_row_and_does_not_mutate_source(monkeypatch, validate):
    worker = object.__new__(rollout.EqualLengthAgentLoopWorker)
    worker.rollout_config = SimpleNamespace(n=2 if validate else 1)
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2], [3, 4]])},
        non_tensors={"agent_name": np.array(["custom", "custom"], dtype=object)},
        meta_info={"validate": validate, "custom": "preserve"},
    )

    async def generate(self, received):
        assert len(received) == len(batch) == 2
        assert (
            received.non_tensor_batch["agent_name"].tolist()
            == ["single_turn_agent" if validate else "equal_length_agent"] * 2
        )
        assert received.meta_info == batch.meta_info
        if not validate:
            received.non_tensor_batch["eqlen_rollouts"] = np.array([_wire()["eqlen_rollouts"]] * 2, dtype=object)
        return received

    monkeypatch.setattr(rollout.AgentLoopWorker, "generate_sequences", generate)
    result = await worker.generate_sequences(batch)
    assert len(result) == 2
    assert batch.non_tensor_batch["agent_name"].tolist() == ["custom", "custom"]
    assert "eqlen_rollouts" not in batch.non_tensor_batch
    assert result.meta_info == batch.meta_info


@_run_async
async def test_pair_worker_rejects_training_row_expansion():
    worker = object.__new__(rollout.EqualLengthAgentLoopWorker)
    worker.rollout_config = SimpleNamespace(n=2)
    batch = DataProto.from_dict(tensors={"input_ids": torch.tensor([[1, 2]])})
    with pytest.raises(ValueError, match="rollout.n=1"):
        await worker.generate_sequences(batch)
    assert "agent_name" not in batch.non_tensor_batch


def test_pair_manager_reuses_existing_load_balancer(monkeypatch):
    init = Mock()
    worker_class = object()
    monkeypatch.setattr(rollout.AgentLoopManager, "__init__", init)
    monkeypatch.setattr(rollout.ray, "remote", Mock(return_value=worker_class))
    config, load_balancer, teacher, rewards = object(), object(), object(), object()
    manager = rollout.EqualLengthAgentLoopManager(
        config, SimpleNamespace(_load_balancer=load_balancer), teacher, rewards
    )
    assert manager.agent_loop_workers_class is worker_class
    init.assert_called_once()
    args = init.call_args.args
    assert args[0] is config
    assert isinstance(args[1], rollout.EqualLengthServerClient)
    assert args[1]._load_balancer is load_balancer
    assert args[2:] == (teacher, rewards)


@_run_async
async def test_pair_worker_defers_paired_rewards_and_keeps_ordinary_scoring(monkeypatch):
    worker = object.__new__(rollout.EqualLengthAgentLoopWorker)
    score = AsyncMock()
    monkeypatch.setattr(rollout.AgentLoopWorker, "_compute_score", score)
    paired = SimpleNamespace(extra_fields={"eqlen_rollouts": _wire()["eqlen_rollouts"]})
    await worker._compute_score([paired], {})
    score.assert_not_awaited()
    ordinary = SimpleNamespace(extra_fields={})
    await worker._compute_score([ordinary], {"ordinary": True})
    score.assert_awaited_once_with([ordinary], {"ordinary": True})


@_run_async
@pytest.mark.parametrize("invalid", [None, "groups", "prompt", "logprobs", "requested_logprobs", "response"])
async def test_pair_loop_checks_metadata_and_returns_representative(invalid):
    loop = object.__new__(rollout.EqualLengthAgentLoop)
    loop.processor = None
    loop.rollout_config = SimpleNamespace(custom={"equal_length": {"num_pairs": 8}})
    loop.response_length = 3
    loop.apply_chat_template = AsyncMock(return_value=[1, 2])
    output = deepcopy(_wire())
    if invalid == "groups":
        output["eqlen_rollouts"].pop()
    elif invalid == "prompt":
        output["eqlen_rollouts"][0]["prompt_token_ids"] = [9]
    elif invalid == "logprobs":
        output["log_probs"] = [-0.1]
    elif invalid == "requested_logprobs":
        output["log_probs"] = None
    elif invalid == "response":
        output["token_ids"] = []
    loop.server_manager = SimpleNamespace(generate_equal_length_pairs=AsyncMock(return_value=output))
    prompt = [{"role": "user", "content": "question"}]
    original = deepcopy(prompt)
    params = {"logprobs": True}
    if invalid is not None:
        with pytest.raises(ValueError):
            await loop.run(params, raw_prompt=prompt)
    else:
        result = await loop.run(params, raw_prompt=prompt)
        assert result.prompt_ids == [1, 2]
        assert result.response_ids == [10, 11]
        assert result.response_logprobs == [-0.1, -0.2]
        assert result.response_mask == [1, 1]
        assert result.extra_fields["eqlen_rollouts"] == output["eqlen_rollouts"]
        assert result.extra_fields["global_steps"] == 4
    assert prompt == original
    assert params == {"logprobs": True}


@_run_async
async def test_pair_manager_seeds_global_rows_before_worker_splits(monkeypatch):
    manager = object.__new__(rollout.EqualLengthAgentLoopManager)
    manager.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=4))
    manager.rollout_config = SimpleNamespace(custom={"equal_length": {"num_pairs": 8, "seed": 17}}, response_length=3)
    received_batches = []

    async def generate(self, received):
        received_batches.append(received)
        chunks = received.chunk(2)
        assert sum(len(chunk) for chunk in chunks) == 4
        received.meta_info["worker_local"] = True
        return received

    monkeypatch.setattr(rollout.AgentLoopManager, "generate_sequences", generate)

    async def sample(step=None):
        batch = DataProto.from_dict(
            tensors={"input_ids": torch.tensor([[1, 2]] * 4)},
            non_tensors={"uid": np.array(["duplicate"] * 4, dtype=object)},
            meta_info={} if step is None else {"global_steps": step},
        )
        output = await manager.generate_sequences(batch)
        assert "__eqlen_sampling_seed__" not in batch.non_tensor_batch
        assert "worker_local" not in batch.meta_info
        assert output.non_tensor_batch["__eqlen_sampling_seed__"].dtype == np.int64
        return output.non_tensor_batch["__eqlen_sampling_seed__"].tolist()

    at_seven = await sample(7)
    assert await sample(7) == at_seven
    at_eight = await sample(8)
    assert await sample() == await sample(0)
    assert len(set(at_seven + at_eight)) == 8
    # Every group's possible initial and continued branch indexes remain in
    # disjoint intervals across rows and steps, using the real native stride.
    intervals = sorted(
        (seed + k * (1 << 20), seed + k * (1 << 20) + 5) for seed in at_seven + at_eight for k in range(8)
    )
    assert all(left[1] <= right[0] for left, right in pairwise(intervals))
    chunks = received_batches[0].chunk(2)
    assert chunks[0].non_tensor_batch["__eqlen_sampling_seed__"].tolist() == at_seven[:2]
    assert chunks[1].non_tensor_batch["__eqlen_sampling_seed__"].tolist() == at_seven[2:]


@_run_async
@pytest.mark.parametrize("validate,seed", [(True, 17), (False, None)])
async def test_pair_manager_preserves_validation_and_unseeded_input(monkeypatch, validate, seed):
    manager = object.__new__(rollout.EqualLengthAgentLoopManager)
    manager.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=1))
    manager.rollout_config = SimpleNamespace(custom={"equal_length": {"num_pairs": 8, "seed": seed}}, response_length=3)
    batch = DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2]])}, meta_info={"validate": validate, "global_steps": 7}
    )

    async def generate(self, received):
        assert received is not batch
        assert "__eqlen_sampling_seed__" not in received.non_tensor_batch
        received.meta_info["changed"] = True
        return received

    monkeypatch.setattr(rollout.AgentLoopManager, "generate_sequences", generate)
    await manager.generate_sequences(batch)
    assert batch.meta_info == {"validate": validate, "global_steps": 7}
    assert not batch.non_tensor_batch


@_run_async
@pytest.mark.parametrize("response_length", [3, (1 << 20) - 2])
async def test_pair_manager_seed_wrap_reserves_all_native_branches(monkeypatch, response_length):
    manager = object.__new__(rollout.EqualLengthAgentLoopManager)
    manager.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=2))
    request_stride = 7 * (1 << 20) + response_length + 2
    safe_period = ((1 << 63) // request_stride - 1) * request_stride
    manager.rollout_config = SimpleNamespace(
        custom={"equal_length": {"num_pairs": 8, "seed": safe_period - 1}}, response_length=response_length
    )

    async def generate(self, received):
        return received

    monkeypatch.setattr(rollout.AgentLoopManager, "generate_sequences", generate)
    batch = DataProto.from_dict(tensors={"input_ids": torch.tensor([[1, 2], [1, 2]])}, meta_info={"global_steps": 0})
    output = await manager.generate_sequences(batch)
    seeds = output.non_tensor_batch["__eqlen_sampling_seed__"].tolist()
    assert seeds == [safe_period - 1, request_stride - 1]
    assert all(0 <= seed and seed + request_stride - 1 < (1 << 63) for seed in seeds)
    assert seeds[1] + request_stride <= seeds[0]
    manager.rollout_config.response_length = 1 << 20
    with pytest.raises(ValueError):
        await manager.generate_sequences(batch)


@_run_async
@pytest.mark.parametrize("step", [True, -1, 1.5])
async def test_pair_manager_rejects_invalid_step_before_dispatch(monkeypatch, step):
    manager = object.__new__(rollout.EqualLengthAgentLoopManager)
    manager.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=1))
    manager.rollout_config = SimpleNamespace(custom={"equal_length": {"num_pairs": 8, "seed": 17}}, response_length=3)
    dispatch = AsyncMock()
    monkeypatch.setattr(rollout.AgentLoopManager, "generate_sequences", dispatch)
    batch = DataProto.from_dict(tensors={"input_ids": torch.tensor([[1, 2]])}, meta_info={"global_steps": step})
    with pytest.raises(ValueError, match="seed range"):
        await manager.generate_sequences(batch)
    dispatch.assert_not_awaited()
    assert "__eqlen_sampling_seed__" not in batch.non_tensor_batch


@_run_async
@pytest.mark.parametrize("explicit_seed", [None, 91])
async def test_pair_loop_forwards_row_seed_without_changing_sampling_params(explicit_seed):
    loop = object.__new__(rollout.EqualLengthAgentLoop)
    loop.processor = None
    loop.rollout_config = SimpleNamespace(custom={"equal_length": {"num_pairs": 8}})
    loop.response_length = 3
    loop.apply_chat_template = AsyncMock(return_value=[1, 2])
    loop.server_manager = SimpleNamespace(generate_equal_length_pairs=AsyncMock(return_value=_wire()))
    params = {"logprobs": True}
    if explicit_seed is not None:
        params["seed"] = explicit_seed
    original = dict(params)
    await loop.run(params, raw_prompt=[{"role": "user", "content": "question"}], __eqlen_sampling_seed__=np.int64(1234))
    sent = loop.server_manager.generate_equal_length_pairs.await_args.kwargs["sampling_params"]
    assert sent["seed"] == (1234 if explicit_seed is None else explicit_seed)
    assert params == original
    assert sent is not params
