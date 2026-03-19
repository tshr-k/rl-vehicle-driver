import os
import platform
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import typer

# Suppress gym, TensorFlow and gRPC warnings for cleaner output
os.environ["GYM_IGNORE_WARNINGS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"


# Initialise a virtual display in headless environments like Colab/SageMaker
if (platform.system() == "Linux") and (not os.environ.get("DISPLAY")):
    from pyvirtualdisplay.display import Display

    display = Display(visible=False, size=(1400, 900))
    display.start()


import gymnasium as gym
from gymnasium.wrappers import TimeLimit
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


# Define absolute paths using a dynamic project root finder
def get_project_root() -> Path:
    """Return the project root depending on the execution environment."""

    # 1. Check for Colab first
    if os.path.exists("/content"):
        return Path("/content/rl-lap-time-simulator")

    # 2. Check if we are in a script (.py)
    try:
        return Path(__file__).resolve().parent
    except NameError:
        # 3. Fallback for local Jupyter Notebooks
        return Path.cwd().resolve()


PROJECT_ROOT = get_project_root()
VIDEO_DIR = PROJECT_ROOT / "videos"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"


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


def make_env(config: Dict[str, Any]) -> gym.Env:
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

    track_csv_path = str(TRACKS_DIR / config["track_name"])

    # Create the training environment
    env = gym.make(
        "LTS-v0",
        render_mode=config["render_mode"],
        track_csv_path=track_csv_path,
        num_rangefinders=config["num_rangefinders"],
        num_future_curvatures=config["num_future_curvatures"],
        reward_coefficients=reward_coefficients,
        car_model=car_model,
        follow_car=config["follow_car"],
        reset_options=reset_options,
    )

    # Enforce maximum episode length
    env = TimeLimit(env, max_episode_steps=config["max_episode_steps"])

    return env


class VideoLoggerCallback(WandbCallback):
    def __init__(
        self,
        video_path: Path,
        config: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.video_path = Path(video_path)
        self.video_log_freq: int = config["video_save_freq"]
        self.render_mode: str | None = config["render_mode"]

    def _on_step(self) -> bool:  # type: ignore[override]
        if self.render_mode != "rgb_array":
            return True  # Skip video logging if not using rgb_array rendering

        # Check every X steps (matches record_video_trigger frequency)
        if self.video_log_freq > 0 and self.n_calls % self.video_log_freq == 0:
            files = list(self.video_path.glob("*.mp4"))
            if files:
                latest_video = max(files, key=os.path.getmtime)
                wandb.log({"video": wandb.Video(str(latest_video), format="mp4")})
        return True

    def _on_training_end(self) -> None:  # Triggered even by KeyboardInterrupt
        # Save model locally
        final_model_path = MODEL_DIR / "ppo_lts_final_model"
        vec_normalize_path = MODEL_DIR / "vec_normalize.pkl"

        self.model.save(final_model_path)
        self.training_env: VecNormalize
        self.training_env.save(str(vec_normalize_path))

        # Sync to W&B
        wandb.save(vec_normalize_path, base_path=str(MODEL_DIR))
        wandb.save(final_model_path.with_suffix(".zip"), base_path=str(MODEL_DIR))

        # Clean up
        self.training_env.close()
        wandb.finish()


def main(
    # Training parameters
    total_timesteps: int = 10_000,
    model_save_freq: int = 2_000,
    gradient_save_freq: int = 500,
    video_save_freq: int = 1_000,
    video_length: int = 300,
    # Environment parameters
    max_episode_steps: int = 1_000,
    num_processes: int = 4,
    track: TrackOption = TrackOption.MexicoCity,
    num_rangefinders: int = NUM_RANGEFINDERS,
    num_future_curvatures: int = NUM_FUTURE_CURVATURES,
    render_mode: str | None = None,
    follow_car: bool = False,
    # Reward coefficients (flattened)
    reward_progress: float = 5.0,
    reward_parallel_velocity: float = 0.1,
    reward_survival: float = 0.0,
    reward_collision: float = -20.0,
    # Reset options (flattened)
    reset_randomise_start: bool = RESET_OPTIONS.randomise_start,
    reset_randomise_track: bool = RESET_OPTIONS.randomise_track,
    # Car model selection
    car_model: CarModelOption = CarModelOption.kinematic_bicycle,
) -> None:

    # Store only the track *name* in the config
    track_name = track.value

    # Initialise Weights & Biases for experiment tracking
    config = {
        # Training parameters
        "total_timesteps": total_timesteps,
        "model_save_freq": model_save_freq,
        "gradient_save_freq": gradient_save_freq,
        "video_save_freq": video_save_freq,
        "video_length": video_length,
        # Environment parameters
        "max_episode_steps": max_episode_steps,
        "num_processes": num_processes,
        "track_name": track_name,
        "num_rangefinders": num_rangefinders,
        "num_future_curvatures": num_future_curvatures,
        "render_mode": render_mode,
        "follow_car": follow_car,
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
        monitor_gym=False,  # We will log videos manually via the callback
        save_code=True,
    )

    # Create directories for videos, logs and models if they don't exist
    for directory in [VIDEO_DIR, LOG_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)

    print("Starting training...")
    if render_mode == "rgb_array":
        print(
            "Video recording enabled. Videos will be saved every "
            f"{video_save_freq} steps and logged to W&B."
        )
    else:
        print("Video recording disabled. To enable, set --render-mode to 'rgb_array'.")

    # Initialise the parallel environment
    NUM_PROCESSES = config["num_processes"]
    vec_env = SubprocVecEnv([lambda: make_env(config) for _ in range(NUM_PROCESSES)])

    # Save videos locally only when the environment supports rgb_array rendering
    if config["render_mode"] == "rgb_array":
        vec_env = VecVideoRecorder(
            vec_env,
            video_folder=str(VIDEO_DIR),
            record_video_trigger=lambda x: x % config["video_save_freq"] == 0,
            video_length=config["video_length"],
            name_prefix="ppo_lts_agent",
        )

    # Monitor the environment to log episode rewards, episode lengths and end-of-episode info
    vec_env = VecMonitor(vec_env, info_keywords=("speed", "progress", "terminated", "truncated"))

    # Normalize observations and rewards for stable training
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Define the agent
    agent = PPO(
        "MultiInputPolicy", vec_env, verbose=1, n_steps=256, tensorboard_log=f"runs/{run.id}"
    )
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=VideoLoggerCallback(
            video_path=VIDEO_DIR,
            config=config,
            verbose=0,
            model_save_path=f"models/{run.id}",
            model_save_freq=config["model_save_freq"],
            gradient_save_freq=config["gradient_save_freq"],
        ),
    )

    print("Training complete")

    # Clean up happens in the callback's _on_training_end method


if __name__ == "__main__":
    typer.run(main)
