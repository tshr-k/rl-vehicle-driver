import os
from enum import Enum
from pathlib import Path

import typer

# Suppress gym, TensorFlow and gRPC warnings for cleaner output
os.environ["GYM_IGNORE_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"


# Initialise a virtual display in headless environments like Colab/SageMaker
if not os.environ.get("DISPLAY"):
    from pyvirtualdisplay.display import Display

    display = Display(visible=False, size=(1400, 900))
    display.start()


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    VecVideoRecorder,
)
from wandb.integration.sb3 import WandbCallback

import wandb
from environment.car_model import KinematicBicycleModel
from environment.lts_env import ResetOptions, RewardCoefficients

# Define absolute paths to avoid folder recursion issues
PROJECT_ROOT = Path("/content/rl-lap-time-simulator").resolve()
VIDEO_DIR = str(PROJECT_ROOT / "videos")
LOG_DIR = str(PROJECT_ROOT / "logs")
MODEL_DIR = str(PROJECT_ROOT / "models")


# Define options for validating CLI arguments
class CarModelOption(str, Enum):
    kinematic_bicycle = "kinematic_bicycle"


class TrackOption(str, Enum):
    Austin = "Austin.csv"
    BrandsHatch = "BrandsHatch.csv"
    Budapest = "Budapest.csv"
    Catalunya = "Catalunya.csv"
    Hockenheim = "Hockenheim.csv"
    IMS = "IMS.csv"
    Melbourne = "Melbourne.csv"
    MexicoCity = "MexicoCity.csv"
    Montreal = "Montreal.csv"
    Monza = "Monza.csv"
    MoscowRaceway = "MoscowRaceway.csv"
    Norisring = "Norisring.csv"
    Nuerburgring = "Nuerburgring.csv"
    Oschersleben = "Oschersleben.csv"
    Sakhir = "Sakhir.csv"
    SaoPaulo = "SaoPaulo.csv"
    Sepang = "Sepang.csv"
    Shanghai = "Shanghai.csv"
    Silverstone = "Silverstone.csv"
    Sochi = "Sochi.csv"
    Spa = "Spa.csv"
    Spielberg = "Spielberg.csv"
    Suzuka = "Suzuka.csv"
    YasMarina = "YasMarina.csv"
    Zandvoort = "Zandvoort.csv"


# Map car model names to their corresponding classes
CAR_MODELS = {
    CarModelOption.kinematic_bicycle.value: KinematicBicycleModel,
}


# Define environment parameters
TRACKS_DIR = PROJECT_ROOT / "data" / "tracks"
NUM_RANGEFINDERS = 6
NUM_FUTURE_CURVATURES = 25
RESET_OPTIONS = ResetOptions(
    randomise_start=True,
    randomise_track=False,
)

# Register the base environment
gym.register(
    id="LTS-v0",
    entry_point="environment.lts_env:LtsEnv",
)


def make_env(config: dict) -> gym.Env:
    # Initialise the car physics model from the selected option
    car_model_cls = CAR_MODELS[config["car_model"]]
    car_model = car_model_cls()

    reward_coefficients = RewardCoefficients(
        progress=config["reward_progress"],
        parallel_velocity=config["reward_parallel_velocity"],
        survival=config["reward_survival"],
        collision=config["reward_collision"],
    )

    reset_options = ResetOptions(
        randomise_start=config["reset_randomise_start"],
        randomise_track=config["reset_randomise_track"],
    )

    # Create the training environment
    return gym.make(
        "LTS-v0",
        render_mode="rgb_array",
        track_csv_path=config["track_csv_path"],
        num_rangefinders=config["num_rangefinders"],
        num_future_curvatures=config["num_future_curvatures"],
        reward_coefficients=reward_coefficients,
        car_model=car_model,
        follow_car=False,
        reset_options=reset_options,
    )


def main(
    # Training parameters
    total_timesteps: int = 100,
    model_save_freq: int = 99,
    gradient_save_freq: int = 49,
    video_save_freq: int = 49,
    # Environment parameters
    num_processes: int = 4,
    track: TrackOption = TrackOption.MexicoCity,
    num_rangefinders: int = NUM_RANGEFINDERS,
    num_future_curvatures: int = NUM_FUTURE_CURVATURES,
    # Reward coefficients (flattened)
    reward_progress: float = 5.0,
    reward_parallel_velocity: float = 0.1,
    reward_survival: float = 0.2,
    reward_collision: float = -200.0,
    # Reset options (flattened)
    reset_randomise_start: bool = RESET_OPTIONS.randomise_start,
    reset_randomise_track: bool = RESET_OPTIONS.randomise_track,
    # Car model selection
    car_model: CarModelOption = CarModelOption.kinematic_bicycle,
) -> None:

    track_csv_path = str(TRACKS_DIR / track.value)

    # Initialise Weights & Biases for experiment tracking
    config = {
        # Training parameters
        "total_timesteps": total_timesteps,
        "model_save_freq": model_save_freq,
        "gradient_save_freq": gradient_save_freq,
        "video_save_freq": video_save_freq,
        # Environment parameters
        "num_processes": num_processes,
        "track_csv_path": track_csv_path,
        "num_rangefinders": num_rangefinders,
        "num_future_curvatures": num_future_curvatures,
        # Reward coefficients (flattened)
        "reward_progress": reward_progress,
        "reward_parallel_velocity": reward_parallel_velocity,
        "reward_survival": reward_survival,
        "reward_collision": reward_collision,
        # Reset options (flattened)
        "reset_randomise_start": reset_randomise_start,
        "reset_randomise_track": reset_randomise_track,
        # Car model selection
        "car_model": car_model.value,
    }

    run = wandb.init(
        project="LTS",
        name="PPO Training Run",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # Create directories for videos, logs and models if they don't exist
    for directory in [VIDEO_DIR, LOG_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)

    print("Starting training...")
    # Initialise the parallel environment
    NUM_PROCESSES = config["num_processes"]
    vec_env = SubprocVecEnv([lambda: make_env(config) for _ in range(NUM_PROCESSES)])

    # Normalize observations and rewards for stable training
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Monitor the environment to log episode rewards and lengths
    vec_env = VecMonitor(vec_env)

    vec_env = VecVideoRecorder(
        vec_env,
        video_folder=VIDEO_DIR,
        record_video_trigger=lambda x: x % config["video_save_freq"] == 0,
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
            model_save_freq=config["model_save_freq"],
            gradient_save_freq=config["gradient_save_freq"],
        ),
    )

    print("Training complete")

    vec_env.close()
    wandb.finish()

    # agent.save("ppo_lts_agent")

    # Required for correct evaluation
    # vec_env.save("vec_normalize.pkl")


if __name__ == "__main__":
    typer.run(main)
