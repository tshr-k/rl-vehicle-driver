import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from environment.car_model import KinematicBicycleModel
from environment.lts_env import RewardCoefficients

# Define environment parameters
TRACK_CSV_PATH = "data/tracks/MexicoCity.csv"
NUM_RANGEFINDERS = 20
NUM_FUTURE_CURVATURES = 20

REWARD_COEFFICIENTS = RewardCoefficients(
    progress=+5.0,
    parallel_velocity=0.1,
    survival=+0.1,
    collision=-100.0,
)

# Register and create base environment
gym.register(
    id="LTS-v0",
    entry_point="environment.lts_env:LtsEnv",
)

car_model = KinematicBicycleModel()

env_train = gym.make(
    "LTS-v0",
    render_mode=None,
    track_csv_path=TRACK_CSV_PATH,
    num_rangefinders=NUM_RANGEFINDERS,
    num_future_curvatures=NUM_FUTURE_CURVATURES,
    reward_coefficients=REWARD_COEFFICIENTS,
    car_model=car_model,
    follow_car=True,
)


# Define the agent
agent = PPO("MultiInputPolicy", env_train, verbose=1)
agent.learn(total_timesteps=10_000)

# Convert the environment to a vectorized environment
vec_env_train = agent.get_env()
if vec_env_train is None:
    raise ValueError("The model's environment is None. Please check the model's environment.")

# Normalize observations but not rewards 
vec_env_train = VecNormalize(vec_env_train, norm_obs=True, norm_reward=False)


print("Training complete. Running the trained agent...")


# Create a new environment with rendering enabled for evaluation
env_eval = gym.make(
    "LTS-v0",
    render_mode="human",
    track_csv_path=TRACK_CSV_PATH,
    num_rangefinders=NUM_RANGEFINDERS,
    num_future_curvatures=NUM_FUTURE_CURVATURES,
    car_model=car_model,
    follow_car=False,
)

# Run the trained agent
obs, _ = env_eval.reset()
while True:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_eval.step(action)

    print(info)

    if terminated or truncated:
        obs, _ = env_eval.reset()

