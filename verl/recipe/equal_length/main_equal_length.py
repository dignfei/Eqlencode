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
"""Run the EqLen recipe through the standard Ray worker and dataset setup."""

import os
from pathlib import Path

# Ray reads this flag on import; keep workers in the selected CUDA environment.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import hydra
import ray
from omegaconf import open_dict

from recipe.equal_length.equal_length_ray_trainer import EqualLengthRayPPOTrainer
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class EqualLengthTaskRunner(TaskRunner):
    trainer_class = EqualLengthRayPPOTrainer


@hydra.main(config_path="config", config_name="equal_length_trainer", version_base=None)
def main(config):
    os.environ["KLSTATS_ENABLE"] = "0"
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    repo_root = Path(__file__).resolve().parents[2]
    with open_dict(config):
        runtime_env = config.ray_kwargs.ray_init.get("runtime_env") or {}
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars["KLSTATS_CODE_DIR"] = str(repo_root / "paper")
        env_vars["KLSTATS_ENABLE"] = "0"
        env_vars["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"]
        env_vars["PYTHONPATH"] = os.pathsep.join(
            (str(repo_root), str(repo_root / "paper"), env_vars.get("PYTHONPATH", os.getenv("PYTHONPATH", "")))
        )
        runtime_env["env_vars"] = env_vars
        config.ray_kwargs.ray_init.runtime_env = runtime_env
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(EqualLengthTaskRunner))


if __name__ == "__main__":
    main()
