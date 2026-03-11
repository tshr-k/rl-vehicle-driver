import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train import NUM_RANGEFINDERS, RESET_OPTIONS, TRACK_CSV_PATH, NUM_FUTURE_CURVATURES, car_model

def make_env() -> gym.Env:
    # Create the evaluation environment
    return gym.make(
        "LTS-v0",
        render_mode="human",
        track_csv_path=TRACK_CSV_PATH,
        num_rangefinders=NUM_RANGEFINDERS,
        num_future_curvatures=NUM_FUTURE_CURVATURES,
        car_model=car_model,
        follow_car=False,
        reset_options=RESET_OPTIONS,
    )

# Create a dummy vectorized environment
env_eval = DummyVecEnv([make_env])

# Load the VecNormalize statistics and weights of the trained agent
env_eval = VecNormalize.load("vec_normalize.pkl", env_eval)
env_eval.training = False
env_eval.norm_reward = False

agent = PPO.load("ppo_lts_agent", env=env_eval)

# Run the trained agent
obs = env_eval.reset()

while True:
    action, _ = agent.predict(obs, deterministic=True) # type: ignore
    obs, reward, done, info = env_eval.step(action)
    print(info)

    if done:
        obs = env_eval.reset()
