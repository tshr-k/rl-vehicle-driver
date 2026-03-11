import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecVideoRecorder, VecMonitor
from wandb.integration.sb3 import WandbCallback

import wandb
from environment.car_model import KinematicBicycleModel
from environment.lts_env import ResetOptions, RewardCoefficients

# Define environment parameters
TRACK_CSV_PATH = "data/tracks/MexicoCity.csv"
NUM_RANGEFINDERS = 6
NUM_FUTURE_CURVATURES = 20
REWARD_COEFFICIENTS = RewardCoefficients(
    progress=+5.0,
    parallel_velocity=0.1,
    survival=+0.2,
    collision=-200.0,
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


def make_env() -> gym.Env:
    # Create the training environment
    return gym.make(
        "LTS-v0",
        render_mode="rgb_array",
        track_csv_path=TRACK_CSV_PATH,
        num_rangefinders=NUM_RANGEFINDERS,
        num_future_curvatures=NUM_FUTURE_CURVATURES,
        reward_coefficients=REWARD_COEFFICIENTS,
        car_model=car_model,
        follow_car=False,
        reset_options=RESET_OPTIONS,
    )


if __name__ == "__main__":
    # Get a representation of the car model for logging, excluding private attributes
    car_model_repr = {
        "model": car_model.__class__.__name__,
        **{k: v for k, v in car_model.__dict__.items() if not k.startswith("_")},
    }

    # Initialise Weights & Biases for experiment tracking
    config = {
        "total_timesteps": 100,
        "track_csv_path": TRACK_CSV_PATH,
        "num_rangefinders": NUM_RANGEFINDERS,
        "num_future_curvatures": NUM_FUTURE_CURVATURES,
        "reward_coefficients": REWARD_COEFFICIENTS.__dict__,
        "reset_options": RESET_OPTIONS.__dict__,
        "car_model": car_model_repr,
    }

    run = wandb.init(
        project="LTS",
        name="PPO Training Run",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    print("Starting training...")
    # Initialise the parallel environment
    NUM_PROCESSES = 16
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_PROCESSES)])

    # Normalize observations and rewards for stable training
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Monitor the environment to log episode rewards and lengths
    vec_env = VecMonitor(vec_env)

    # Record agent performance videos
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder=f"videos/{run.id}",
        record_video_trigger=lambda x: x % 10_000 == 0,
        video_length=200,
        name_prefix="ppo_lts_agent",
    )

    # Define the agent
    agent = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=f"runs/{run.id}")
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=0,
            model_save_path=f"models/{run.id}",
            model_save_freq=10_000,
            gradient_save_freq=1_000,
        ),
    )

    print("Training complete")

    vec_env.close()
    wandb.finish()



    # agent.save("ppo_lts_agent")

    # Required for correct evaluation
    # vec_env.save("vec_normalize.pkl")
