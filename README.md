# RL Vehicle Driver

A reinforcement learning project where an agent given only first-person observations learns to drive around a 2D racetrack environment. I built it from scratch (including vehicle dynamics) with Gymnasium and Stable-Baselines3 to explore environment design, reward engineering and reinforcement learning algorithms.

## Try it out

### Setup

Requires Python 3.11+ and a [Weights & Biases](https://wandb.ai) account.

```sh
pip install -r requirements.txt
wandb login
```

To evaluate runs, create a `.env` file in the project root containing your W&B username:

```sh
WANDB_USERNAME=your-username
```

### Running

#### Driving manually

```sh
python play.py
```

Use the arrow keys to accelerate and steer, and space to brake.

#### Training

```sh
python train.py --total-timesteps 1000000 --track Monza.csv
```

Training hyperparameters, environment settings, reward coefficients and the track are all configurable from the command line. Run `python train.py --help` to see every option. Pass `--render-mode rgb_array` to record episode videos and log them to W&B.

#### Evaluating

```sh
python eval.py <wandb-run-id> --render-mode human
```

This downloads the trained model and normalisation statistics from the given W&B run and watches the agent drive. Add `--follow-car` to keep the camera centred on the car.

## Acknowledgements

Track data is from the [TUMFTM racetrack database](https://github.com/TUMFTM/racetrack-database). The choice of observations given to the agent is inspired by [a Sony paper](https://ieeexplore.ieee.org/abstract/document/9372847).