import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set to the GPU you want to use, or leave empty for CPU

builder = tfds.builder_from_directory(
    # path to the `droid` directory (not its parent)
    builder_dir="<path_to_droid_dataset_tfds_files>",
)
ds = builder.as_dataset(split="train", shuffle_files=False)
tf.data.experimental.ignore_errors(ds)

keep_ranges_path = "<path_to_where_to_save_the_json>"
keep_ranges_map = {}
if os.path.exists(keep_ranges_path):
    with open(keep_ranges_path, "r") as f:
        keep_ranges_map = json.load(f)

min_idle_len = 7 # If more than this number of consecutive idle frames, filter all of them out
min_non_idle_len = 16 # If fewer than this number of consecutive non-idle frames, filter all of them out

for ep_idx, ep in enumerate(tqdm(ds)):
    recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
    file_path = ep["episode_metadata"]["file_path"].numpy().decode()

    key = f"{recording_folderpath}--{file_path}"
    if key in keep_ranges_map:
        continue
    
    joint_velocities = []
    for step in ep["steps"]:
        joint_velocities.append(step["action_dict"]["joint_velocity"].numpy())
    joint_velocities = np.array(joint_velocities)

    is_idle_array = np.hstack([np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)])

    # Get all idle ranges of length at least 7
    padded = np.concatenate([[False], is_idle_array, [False]])

    diff = np.diff(padded.astype(int))
    true_starts = np.where(diff == 1)[0]  # +1 transitions
    true_ends   = np.where(diff == -1)[0]  # -1 transitions

    true_segment_masks = (true_ends - true_starts) >= min_idle_len
    true_starts = true_starts[true_segment_masks]
    true_ends = true_ends[true_segment_masks]

    keep_mask = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(true_starts, true_ends):
        keep_mask[start:end] = False

    # Get all non-idle ranges of at least 16
    padded = np.concatenate([[False], keep_mask, [False]])

    diff = np.diff(padded.astype(int))
    true_starts = np.where(diff == 1)[0]  # +1 transitions
    true_ends   = np.where(diff == -1)[0]  # -1 transitions

    true_segment_masks = (true_ends - true_starts) >= min_non_idle_len
    true_starts = true_starts[true_segment_masks]
    true_ends = true_ends[true_segment_masks]

    keep_ranges_map[key] = []
    for start, end in zip(true_starts, true_ends):
        keep_ranges_map[key].append((int(start), int(end)))

    if ep_idx % 1000 == 0:
        with open(keep_ranges_path, "w") as f:
            json.dump(keep_ranges_map, f)

print("Done!")
with open(keep_ranges_path, "w") as f:
    json.dump(keep_ranges_map, f)