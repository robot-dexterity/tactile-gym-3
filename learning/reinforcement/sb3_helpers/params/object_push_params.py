from learning.reinforcement.sb3_helpers.params.default_params import env_args
from learning.reinforcement.sb3_helpers.params.default_params import rl_params_ppo
from learning.reinforcement.sb3_helpers.params.default_params import ppo_params
from learning.reinforcement.sb3_helpers.params.default_params import rl_params_sac
from learning.reinforcement.sb3_helpers.params.default_params import sac_params


env_args["env_params"]["max_steps"] = 1000
env_args["env_params"]["observation_mode"] = "oracle"
# env_args["env_params"]["observation_mode"] = "tactile_and_feature"
# env_args["env_params"]["observation_mode"] = "visual_and_feature"
# env_args["env_params"]["observation_mode"] = "visuotactile_and_feature"

env_args["robot_arm_params"]["control_mode"] = "tcp_velocity_control"
env_args["robot_arm_params"]["control_dofs"] = ["y", "Rz"]

env_args["tactile_sensor_params"]["type"] = "right_angle_tactip"
# env_args["tactile_sensor_params"]["type"] = "right_angle_digit"
# env_args["tactile_sensor_params"]["type"] = "right_angle_digitac"

rl_params_ppo["env_id"] = "object_push-v0"
rl_params_ppo["total_timesteps"] = int(1e6)
ppo_params["learning_rate"] = 3e-4

rl_params_sac["env_id"] = "object_push-v0"
rl_params_sac["total_timesteps"] = int(1e6)
sac_params["learning_rate"] = 3e-4
