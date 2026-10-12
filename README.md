# RL Lap Time Simulator

A reinforcement learning project where PPO agents learn to drive fast laps around real-world race tracks using only first-person observations (like a human driver). I built it from scratch with Gymnasium and Stable-Baselines3 to explore environment design, reward engineering and policy optimisation.

## Features

- PPO agents trained in parallel across multiple processes with observation and reward normalisation
- Multi-input policy over a dictionary observation of local kinematics, rangefinder rays to the track boundaries and upcoming track curvature
- Continuous `[throttle, steering]` action space
- Configurable reward shaping for track progress, velocity along the track, survival and collisions
- Randomised start positions and episode time limits for more varied training experience
- Experiment tracking, model checkpoints and episode videos logged to Weights & Biases
- Custom Gymnasium environment covering 25 real-world circuits, with a kinematic bicycle car model including drag, downforce and tyre grip limits
- Pygame rendering, with a manual driving mode for testing the environment yourself

## Setup

Requires Python 3.11+ and a [Weights & Biases](https://wandb.ai) account.

```sh
pip install -r requirements.txt
wandb login
```

To evaluate runs, create a `.env` file in the project root containing your W&B username:

```sh
WANDB_USERNAME=your-username
```

## Running

### Driving manually

```sh
python play.py
```

Use the arrow keys to accelerate and steer, and space to brake.

### Training

```sh
python train.py --total-timesteps 1000000 --track Monza.csv
```

Training hyperparameters, environment settings, reward coefficients and the track are all configurable from the command line. Run `python train.py --help` to see every option. Pass `--render-mode rgb_array` to record episode videos and log them to W&B.

### Evaluating

```sh
python eval.py <wandb-run-id> --render-mode human
```

This downloads the trained model and normalisation statistics from the given W&B run and watches the agent drive. Add `--follow-car` to keep the camera centred on the car.

## Acknowledgements

Track data is from the [TUMFTM racetrack database](https://github.com/TUMFTM/racetrack-database).
