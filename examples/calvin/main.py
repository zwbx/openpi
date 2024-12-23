"""Runs a model in a CALVIN simulation environment."""

import collections
from dataclasses import dataclass
import logging
import pathlib
import time

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
import calvin_env
from calvin_env.envs.play_table_env import get_env
import hydra
import imageio
import numpy as np
from omegaconf import OmegaConf
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 5

    #################################################################################################################
    # CALVIN environment-specific parameters
    #################################################################################################################
    calvin_data_path: str = "/datasets/calvin_debug_dataset"  # Path to CALVIN dataset for loading validation tasks
    max_subtask_steps: int = 360  # Max number of steps per subtask
    num_trials: int = 1000  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/calvin/videos"  # Path to save videos
    num_save_videos: int = 5  # Number of videos to be logged per task
    video_temp_subsample: int = 5  # Temporal subsampling to make videos shorter

    seed: int = 7  # Random Seed (for reproducibility)


def main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize CALVIN environment
    env = get_env(pathlib.Path(args.calvin_data_path) / "validation", show_gui=False)

    # Get CALVIN eval task set
    task_definitions, task_instructions, task_reward = _get_calvin_tasks_and_reward(args.num_trials)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation.
    episode_solved_subtasks = []
    per_subtask_success = collections.defaultdict(list)
    for i, (initial_state, task_sequence) in enumerate(tqdm.tqdm(task_definitions)):
        logging.info(f"Starting episode {i+1}...")
        logging.info(f"Task sequence: {task_sequence}")

        # Reset env to initial position for task
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        rollout_images = []
        solved_subtasks = 0
        for subtask in task_sequence:
            start_info = env.get_info()
            action_plan = collections.deque()

            obs = env.get_obs()
            done = False
            for _ in range(args.max_subtask_steps):
                img = obs["rgb_obs"]["rgb_static"]
                wrist_img = obs["rgb_obs"]["rgb_gripper"]
                rollout_images.append(img.transpose(2, 0, 1))

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "observation/rgb_static": img,
                        "observation/rgb_gripper": wrist_img,
                        "observation/state": obs["robot_obs"],
                        "prompt": str(task_instructions[subtask][0]),
                    }

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()

                # Round gripper action since env expects gripper_action in (-1, 1)
                action[-1] = 1 if action[-1] > 0 else -1

                # Step environment
                obs, _, _, current_info = env.step(action)

                # check if current step solves a task
                current_task_info = task_reward.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    done = True
                    solved_subtasks += 1
                    break

            per_subtask_success[subtask].append(int(done))
            if not done:
                # Subtask execution failed --> stop episode
                break

        episode_solved_subtasks.append(solved_subtasks)
        if len(episode_solved_subtasks) < args.num_save_videos:
            # Save rollout video.
            idx = len(episode_solved_subtasks)
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{idx}.mp4",
                [np.asarray(x) for x in rollout_images[:: args.video_temp_subsample]],
                fps=50 // args.video_temp_subsample,
            )

        # Print current performance after each episode
        logging.info(f"Solved subtasks: {solved_subtasks}")
        _calvin_print_performance(episode_solved_subtasks, per_subtask_success)

    # Log final performance
    logging.info(f"results/avg_num_subtasks: : {np.mean(episode_solved_subtasks)}")
    for i in range(1, 6):
        # Compute fraction of episodes that have *at least* i successful subtasks
        logging.info(
            f"results/avg_success_len_{i}: {np.sum(episode_solved_subtasks >= i) / len(episode_solved_subtasks)}"
        )
    for key in per_subtask_success:
        logging.info(f"results/avg_success__{key}: {np.mean(per_subtask_success[key])}")


def _get_calvin_tasks_and_reward(num_sequences):
    conf_dir = pathlib.Path(calvin_env.__file__).absolute().parents[2] / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = get_sequences(num_sequences)
    return eval_sequences, val_annotations, task_oracle


def _calvin_print_performance(episode_solved_subtasks, per_subtask_success):
    # Compute avg success rate per task length
    logging.info("#####################################################")
    logging.info(f"Avg solved subtasks: {np.mean(episode_solved_subtasks)}\n")

    logging.info("Per sequence_length avg success:")
    for i in range(1, 6):
        # Compute fraction of episodes that have *at least* i successful subtasks
        logging.info(f"{i}: {np.sum(np.array(episode_solved_subtasks) >= i) / len(episode_solved_subtasks) * 100}%")

    logging.info("\n Per subtask avg success:")
    for key in per_subtask_success:
        logging.info(f"{key}: \t\t\t {np.mean(per_subtask_success[key]) * 100}%")
    logging.info("#####################################################")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
