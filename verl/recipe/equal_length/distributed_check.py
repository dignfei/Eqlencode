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
"""Check EqLen DP gradients with two CUDA processes sharing one GPU.

Gloo averages CPU copies of actual CUDA gradients. This tests DP mathematics
and process communication, not NCCL, FSDP sharding or a multi-node deployment.
Inputs come from the native rollouts retained by gpu_check.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from recipe.equal_length.algorithms import training_metadata, trajectories
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_equal_length, equal_length_weighted_loss
from verl.workers.config.actor import ActorConfig, PolicyLossConfig


def _batch(proof, mode, scenario):
    step = proof["steps"][0]
    verified = step["verified_complete_trajectories"]
    samples, cursor, groups = [], 0, 0
    for payload in step["native_rollouts"]:
        for group in payload:
            complete = trajectories(group)
            rewards = [row["reward"] for row in verified[cursor : cursor + len(complete)]]
            if scenario == "global_all_tie":
                rewards = [0.0] * len(complete)
            plans = training_metadata(group, rewards, advantage="rloo" if mode == "eqlen_rloo" else "grpo")
            samples.extend(zip([group["prompt_token_ids"]] * len(complete), complete, plans, strict=True))
            cursor += len(complete)
            groups += 1
    assert cursor == len(verified)
    prompt_width = max(len(prompt) for prompt, _, _ in samples)
    response_width = max(len(row.response_token_ids) for _, row, _ in samples)
    real_rows = len(samples)
    # Both shards have equal rows/microbatches, while one control rank is entirely dummy.
    local_rows = real_rows if scenario == "empty_rank" else (real_rows + 1) // 2
    total = 2 * local_rows
    tensors = {
        "input_ids": torch.zeros(total, prompt_width + response_width, dtype=torch.long),
        "attention_mask": torch.zeros(total, prompt_width + response_width, dtype=torch.long),
        "responses": torch.zeros(total, response_width, dtype=torch.long),
        "loss_mask": torch.zeros(total, response_width, dtype=torch.bool),
        "advantages": torch.zeros(total, response_width, dtype=torch.float64),
        "segment_ids": torch.full((total, response_width), -1, dtype=torch.long),
        "loss_weights": torch.zeros(total, response_width, dtype=torch.float64),
    }
    tensors["input_ids"][:, prompt_width - 1] = 1
    tensors["attention_mask"][:, prompt_width - 1] = 1
    offset = local_rows if scenario == "empty_rank" else 0
    for index, (prompt, row, plan) in enumerate(samples, start=offset):
        length = len(row.response_token_ids)
        tensors["input_ids"][index, prompt_width - len(prompt) : prompt_width] = torch.tensor(prompt)
        tensors["input_ids"][index, prompt_width : prompt_width + length] = torch.tensor(row.response_token_ids)
        tensors["attention_mask"][index, prompt_width - len(prompt) : prompt_width + length] = 1
        tensors["responses"][index, :length] = torch.tensor(row.response_token_ids)
        for key, values in (
            ("loss_mask", plan.loss_mask),
            ("advantages", plan.advantages),
            ("segment_ids", plan.segment_ids),
            ("loss_weights", plan.loss_weights),
        ):
            tensors[key][index, :length] = torch.tensor(values, dtype=tensors[key].dtype)
    for index in range(total):
        if offset <= index < offset + real_rows:
            continue
        for key in ("input_ids", "attention_mask", "responses"):
            tensors[key][index] = tensors[key][offset]
    return DataProto.from_dict(tensors=tensors, meta_info={"global_num_subgroups": groups}), prompt_width


def _model(path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager").double().cuda().eval()


def _logprobs(model, batch, prompt_width):
    data = batch.batch
    logits = model(input_ids=data["input_ids"].cuda(), attention_mask=data["attention_mask"].cuda()).logits
    return (
        logits[:, prompt_width - 1 : -1].log_softmax(-1).gather(-1, data["responses"].cuda().unsqueeze(-1)).squeeze(-1)
    )


def _flat_parameters(model):
    return torch.cat([parameter.detach().reshape(-1).cpu() for parameter in model.parameters()])


def _flat_gradients(model):
    return torch.cat(
        [
            (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).detach().reshape(-1).cpu()
            for parameter in model.parameters()
        ]
    )


def _assign_gradients(model, flattened):
    offset = 0
    for parameter in model.parameters():
        parameter.grad = flattened[offset : offset + parameter.numel()].view_as(parameter).cuda().clone()
        offset += parameter.numel()
    assert offset == flattened.numel()


def _paper_loss(current, old, data, mode, groups, reference):
    """Literal full-batch pair/segment objective, without the production reduction."""
    mask = data["loss_mask"].cuda()
    advantage, weights = data["advantages"].cuda(), data["loss_weights"].cuda()
    if mode == "eqlen_rloo":
        terms = -advantage * current
    else:
        delta = current - old
        if mode == "eqlen_gspo":
            ids = data["segment_ids"].cuda()
            delta_for_ratio = torch.zeros_like(delta)
            for row in range(len(delta)):
                for segment in ids[row][mask[row]].unique():
                    selected = mask[row] & (ids[row] == segment)
                    delta_for_ratio[row, selected] = delta[row, selected].mean()
            delta = delta_for_ratio
        ratio = delta.exp()
        terms = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.2))
    kl_delta = reference - current
    terms = terms + 0.002 * (kl_delta.exp() - 1 - kl_delta)
    return (torch.where(mask, terms, 0) * weights).sum() / groups


def _prepare_reference(args):
    proof = json.loads(args.rollout_proof.read_text())
    assert proof["completed"], "Use a completed native GPU proof."
    cases = []
    for mode in ("eqlen_grpo", "eqlen_gspo", "eqlen_rloo"):
        for scenario in ("native", "empty_rank", "global_all_tie"):
            batch, width = _batch(proof, mode, scenario)
            model = _model(args.model_dir)
            initial = _flat_parameters(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            with torch.no_grad():
                reference = _logprobs(model, batch, width).cpu()
            records = []
            for _ in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    old = _logprobs(model, batch, width).cpu()
                active = bool(batch.batch["loss_mask"].any())
                if active:
                    current = _logprobs(model, batch, width)
                    loss = _paper_loss(
                        current,
                        old.cuda(),
                        batch.batch,
                        mode,
                        batch.meta_info["global_num_subgroups"],
                        reference.cuda(),
                    )
                    loss.backward()
                    gradient = _flat_gradients(model)
                    assert torch.isfinite(gradient).all(), "Full-batch reference gradient is not finite."
                    optimizer.step()
                else:
                    loss, gradient = torch.tensor(0.0), torch.zeros_like(initial)
                records.append(
                    {
                        "old": old,
                        "gradient": gradient,
                        "parameters": _flat_parameters(model),
                        "loss": float(loss.detach()),
                        "active": active,
                    }
                )
            name = f"{mode}-{scenario}"
            torch.save(
                {
                    "batch": batch,
                    "width": width,
                    "reference": reference,
                    "records": records,
                    "mode": mode,
                    "scenario": scenario,
                    "initial": initial,
                },
                args.work_dir / f"{name}.pt",
            )
            cases.append(name)
            del model, optimizer
            torch.cuda.empty_cache()
    return cases


def _rank(rank, model_dir, work_dir, rendezvous, cases):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    report = {
        "rank": rank,
        "pid": os.getpid(),
        "cuda_device": 0,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
        "completed": False,
    }
    try:
        for name in cases:
            case = torch.load(Path(work_dir) / f"{name}.pt", weights_only=False, map_location="cpu")
            # Use the actual batch protocol's sharding; subgroup count is shared metadata.
            local = case["batch"].chunk(2)[rank]
            groups = local.meta_info["global_num_subgroups"]
            assert groups == case["batch"].meta_info["global_num_subgroups"]
            start = rank * len(local)
            model = _model(model_dir)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            config = ActorConfig(
                strategy="fsdp2",
                rollout_n=1,
                ppo_micro_batch_size_per_gpu=1,
                policy_loss=PolicyLossConfig(loss_mode=case["mode"]),
                global_batch_info={"dp_size": 2},
            )
            records = []
            for step, expected in enumerate(case["records"], 1):
                optimizer.zero_grad(set_to_none=True)
                local_active = torch.tensor(int(local.batch["loss_mask"].any()))
                global_active = local_active.clone()
                dist.all_reduce(global_active, op=dist.ReduceOp.MAX)
                assert bool(global_active) == expected["active"]
                total_loss = 0.0
                if global_active:
                    # A one-row and a two-row microbatch partition must yield the same global update.
                    micro_size = 1 if step % 2 else 2
                    for offset in range(0, len(local), micro_size):
                        micro = local[offset : offset + micro_size]
                        current = _logprobs(model, micro, case["width"])
                        data = micro.batch.to("cuda")
                        positions = slice(start + offset, start + offset + len(micro))
                        old = expected["old"][positions].cuda()
                        loss, _ = compute_policy_loss_equal_length(
                            old,
                            current,
                            data["advantages"],
                            data["loss_mask"],
                            config=config,
                            segment_ids=data["segment_ids"],
                            loss_weights=data["loss_weights"],
                            global_num_subgroups=groups,
                        )
                        delta = case["reference"][positions].cuda() - current
                        loss = loss + 0.002 * equal_length_weighted_loss(
                            delta.exp() - 1 - delta,
                            data["loss_mask"],
                            data["loss_weights"],
                            groups,
                            2,
                        )
                        loss.backward()
                        total_loss += float(loss.detach())
                    gradient = _flat_gradients(model)
                    assert torch.isfinite(gradient).all(), "Local CUDA gradient is not finite."
                    dist.all_reduce(gradient)
                    gradient /= 2
                    torch.testing.assert_close(gradient, expected["gradient"], rtol=2e-8, atol=2e-10)
                    _assign_gradients(model, gradient)
                    optimizer.step()
                else:
                    gradient = torch.zeros_like(case["initial"])
                    assert not optimizer.state
                loss_sum = torch.tensor(total_loss, dtype=torch.float64)
                dist.all_reduce(loss_sum)
                torch.testing.assert_close(
                    loss_sum / 2, torch.tensor(expected["loss"], dtype=torch.float64), rtol=2e-8, atol=2e-10
                )
                parameters = _flat_parameters(model)
                torch.testing.assert_close(parameters, expected["parameters"], rtol=2e-8, atol=2e-9)
                records.append(
                    {
                        "step": step,
                        "global_active": bool(global_active),
                        "local_active": bool(local_active),
                        "gradient_max_error": float((gradient - expected["gradient"]).abs().max()),
                        "parameter_max_error": float((parameters - expected["parameters"]).abs().max()),
                    }
                )
            report["cases"].append({"name": name, "rows": len(local), "steps": records})
            del model, optimizer
            torch.cuda.empty_cache()
        report["completed"] = True
    finally:
        Path(work_dir, f"rank{rank}.json").write_text(json.dumps(report, indent=2))
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rollout-proof", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    assert torch.cuda.is_available(), "CUDA is required for the actual model computations."
    torch.set_num_threads(1)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    proof_path = args.work_dir / "distributed_gpu_proof.json"
    result = {
        "completed": False,
        "world_size": 2,
        "physical_gpu_count_used": 1,
        "gradient_transport": "CPU Gloo SUM / 2",
        "model_compute": "CUDA float64",
        "scope": "DP loss/gradient/AdamW simulation; not NCCL, FSDP sharding or multi-node validation",
    }
    proof_path.write_text(json.dumps(result, indent=2))
    try:
        cases = _prepare_reference(args)
        with TemporaryDirectory(prefix="eqlen-dp-") as temporary:
            mp.spawn(
                _rank,
                args=(str(args.model_dir.resolve()), str(args.work_dir), str(Path(temporary) / "rendezvous"), cases),
                nprocs=2,
                join=True,
            )
        ranks = [json.loads((args.work_dir / f"rank{rank}.json").read_text()) for rank in range(2)]
        assert all(rank["completed"] and len(rank["cases"]) == len(cases) for rank in ranks)
        assert ranks[0]["pid"] != ranks[1]["pid"]
        result.update(completed=True, ranks=ranks)
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        proof_path.write_text(json.dumps(result, indent=2))
        print("EQLEN_DISTRIBUTED_PROOF", json.dumps({k: result.get(k) for k in ("completed", "error")}))


if __name__ == "__main__":
    main()
