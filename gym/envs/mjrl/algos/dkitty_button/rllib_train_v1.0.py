#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import yaml

import ray
from ray.cluster_utils import Cluster
from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import run_experiments
from ray.tune.schedulers import create_scheduler
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

# Try to import both backends for flag checking/warnings.
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0
Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml
Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml
Note that -f overrides all other trial-specific command-line options.
"""

class MyCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        print("episode {} (env-idx={}) started.".format(
            episode.episode_id, env_index))
        episode.user_data["pole_angles"] = []
        episode.hist_data["pole_angles"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        pole_angle = abs(episode.last_observation_for()[2])
        raw_angle = abs(episode.last_raw_obs_for()[2])
        assert pole_angle == raw_angle
        episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        pole_angle = np.mean(episode.user_data["pole_angles"])
        print("episode {} (env-idx={}) ended with length {} and pole "
              "angles {}".format(episode.episode_id, env_index, episode.length,
                                 pole_angle))
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: {} -> {} episodes".format(
            trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


def on_episode_start(info):
    print(info.keys()) # -> "env", 'episode'
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["is_success"] = []


def on_episode_step(info):
    episode = info["episode"]
    is_success = episode.last_observation_for()[-1]
    is_success_raw = episode.last_raw_obs_for()[-1]
    assert is_success == is_success_raw
    episode.user_data["is_success"].append(is_success)
    # print("step: ")
    # print(is_success)


def on_episode_end(info):
    episode = info["episode"]
    successes = np.sum(episode.user_data["is_success"])
    # episode.custom_metrics["success_mean"] = successes
    # num = len(episode.user_data["is_success"])
    # print("successes: ", successes)
    # print("num of episode: ", num)
    if successes > 0:
        episode.custom_metrics["success_rate"] = 1.
    else:
        episode.custom_metrics["success_rate"] = 0.
    print("episode {} ended with length {} and success {}".format(episode.episode_id, episode.length, successes))


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "--no-ray-ui",
        action="store_true",
        help="Whether to disable the Ray web ui.")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Whether to run ray with `local_mode=True`. "
        "Only if --ray-num-nodes is not used.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).")
    parser.add_argument(
        "-v", action="store_true", help="Whether to use INFO level logging.")
    parser.add_argument(
        "-vv", action="store_true", help="Whether to use DEBUG level logging.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Whether to use PyTorch (instead of tf) as the DL framework.")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Whether to attempt to enable TF eager execution.")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    return parser


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    else:
        # Note: keep this in sync with tune/config_parser.py
        experiments = {
            args.experiment_name: {  # i.e. log to ~/ray_results/default
                "run": args.run,
                # "run": "PPO",
                "env": "mjrl_dkitty_button-v0",
                # "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_freq": 10,
                "checkpoint_at_end": args.checkpoint_at_end,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                # "stop": args.stop,
                "stop": {"timesteps_total": 5000000}, # 10M
                # "config": dict(args.config, env=args.env),
                "config": {
                    "num_workers": 28, 
                    "ignore_worker_failures": True,
                    "seed": 123,
                    "callbacks": {
                        "on_episode_start": on_episode_start,
                        "on_episode_step": on_episode_step,
                        "on_episode_end": on_episode_end,
                    },
                },                
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    verbose = 2
    # for exp in experiments.values():
    #     # Bazel makes it hard to find files specified in `args` (and `data`).
    #     # Look for them here.
    #     # NOTE: Some of our yaml files don't have a `config` section.
    #     if exp.get("config", {}).get("input") and \
    #             not os.path.exists(exp["config"]["input"]):
    #         # This script runs in the ray/rllib dir.
    #         rllib_dir = Path(__file__).parent
    #         input_file = rllib_dir.absolute().joinpath(exp["config"]["input"])
    #         exp["config"]["input"] = str(input_file)

    #     if not exp.get("run"):
    #         parser.error("the following arguments are required: --run")
    #     if not exp.get("env") and not exp.get("config", {}).get("env"):
    #         parser.error("the following arguments are required: --env")

    #     if args.torch:
    #         exp["config"]["framework"] = "torch"
    #     elif args.eager:
    #         exp["config"]["framework"] = "tfe"

    #     if args.trace:
    #         if exp["config"]["framework"] not in ["tf2", "tfe"]:
    #             raise ValueError("Must enable --eager to enable tracing.")
    #         exp["config"]["eager_tracing"] = True

    #     if args.v:
    #         exp["config"]["log_level"] = "INFO"
    #         verbose = 2
    #     if args.vv:
    #         exp["config"]["log_level"] = "DEBUG"
    #         verbose = 3

    # if args.ray_num_nodes:
    #     cluster = Cluster()
    #     for _ in range(args.ray_num_nodes):
    #         cluster.add_node(
    #             num_cpus=args.ray_num_cpus or 1,
    #             num_gpus=args.ray_num_gpus or 0,
    #             object_store_memory=args.ray_object_store_memory)
    #     ray.init(address=cluster.address)
    # else:
    ray.init(
        include_dashboard=not args.no_ray_ui,
        address=args.ray_address,
        object_store_memory=args.ray_object_store_memory,
        num_cpus=args.ray_num_cpus,
        num_gpus=args.ray_num_gpus,
        local_mode=args.local_mode)

    run_experiments(
        experiments,
        # scheduler=create_scheduler(args.scheduler, **args.scheduler_config),
        # scheduler=create_scheduler(args),
        resume=args.resume,
        queue_trials=args.queue_trials,
        verbose=verbose,)
        # concurrent=True)

    ray.shutdown()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)