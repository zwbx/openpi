from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.simpler_policy import make_simpler_example
from openpi.shared import download

# config = _config.get_config("pi05_droid")
config_name = _config.get_config("pi05_simpler")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
checkpoint_dir = "/opt/tiger/openpi/pi05_simpler_28000"

# Create a trained policy.
print(f"Loading policy from {checkpoint_dir}")
policy = policy_config.create_trained_policy(config_name, checkpoint_dir)
print(f"Policy loaded successfully!")


# Run inference on a dummy example.
print(f"Running inference on a dummy example.")
example = make_simpler_example()
print(f"Example created successfully!")

print(f"Running inference on the example.")
action_chunk = policy.infer(example)["actions"]
print(f"Inference completed!")
print(action_chunk)
print(f"Inference completed successfully!")