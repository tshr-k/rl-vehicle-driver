import os

import typer
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wandb
from train import make_env

load_dotenv()


def main(run_id: str, render_mode: str | None = None, follow_car: bool = False):
    # Initialise the WandB API
    wandb_api = wandb.Api()
    run = wandb_api.run(f"{os.environ['WANDB_USERNAME']}/LTS/{run_id}")

    # Load the model weights and configuration from the WandB run
    model_path = (
        run.file("ppo_lts_final_model.zip").download(root=f"./models/{run_id}", replace=True).name
    )
    vec_norm_path = (
        run.file("vec_normalize.pkl").download(root=f"./models/{run_id}", replace=True).name
    )
    config = run.config

    # Override the the training run's render settings for evaluation
    config["render_mode"] = render_mode
    config["follow_car"] = follow_car

    # Create a dummy vectorized environment
    env_eval = DummyVecEnv([lambda: make_env(config)])

    # Load the VecNormalize statistics and weights of the trained agent
    print("Loading the trained agent...")
    env_eval = VecNormalize.load(vec_norm_path, env_eval)
    env_eval.training = False
    env_eval.norm_reward = False
    agent = PPO.load(model_path, env=env_eval)

    # Run the trained agent
    print("Running the trained agent...")
    obs = env_eval.reset()

    while True:
        action, _ = agent.predict(obs, deterministic=False)  # type: ignore
        obs, reward, done, info = env_eval.step(action)
        print(info)

        if done:
            obs = env_eval.reset()


if __name__ == "__main__":
    typer.run(main)
