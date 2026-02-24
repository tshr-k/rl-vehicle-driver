import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from environment.car_model import KinematicBicycleModel
from environment.lts_env import ResetOptions, RewardCoefficients

# Define environment parameters
TRACK_CSV_PATH = "data/tracks/MexicoCity.csv"
NUM_RANGEFINDERS = 6
NUM_FUTURE_CURVATURES = 20
REWARD_COEFFICIENTS = RewardCoefficients(
    progress=+5.0,
    parallel_velocity=0.1,
    survival=+0.1,
    collision=-100.0,
)
RESET_OPTIONS = ResetOptions(
    randomise_start=True,
    randomise_track=False,
)

# Register the base environment
gym.register(
    id="LTS-v0",
    entry_point="environment.lts_env:LtsEnv",
)

# Initialise the car physics model
car_model = KinematicBicycleModel()


def make_env():
    # Create the training environment
    return gym.make(
        "LTS-v0",
        render_mode=None,
        track_csv_path=TRACK_CSV_PATH,
        num_rangefinders=NUM_RANGEFINDERS,
        num_future_curvatures=NUM_FUTURE_CURVATURES,
        reward_coefficients=REWARD_COEFFICIENTS,
        car_model=car_model,
        follow_car=False,
        reset_options=RESET_OPTIONS,
    )


print("Starting training...")

if __name__ == "__main__":
    # Initialise the parallel environment
    NUM_PROCESSES = 16
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_PROCESSES)])

    # Normalize observations and rewards for stable training

    # UNCOMMENT AFTER TESTING WITHOUT NORMALIZATION
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Define the agent
    agent = PPO("MultiInputPolicy", vec_env, verbose=1)
    agent.learn(total_timesteps=10_000)

    print("Training complete")
    vec_env.close()


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
