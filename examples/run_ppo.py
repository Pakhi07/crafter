import argparse

import crafter
import stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


from policy_stability_callback import PolicyStabilityCallback


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

env = crafter.Env()
    
# The Recorder wrapper is great for saving stats and videos
env = crafter.Recorder(
    env,
    args.outdir, # Save recordings in the same log directory
    save_stats=True,
    save_episode=False,
    save_video=False,
)

# Vectorize the environment for Stable-Baselines3
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

# Instantiate our custom callback
stability_callback = PolicyStabilityCallback(log_interval=4096)

# Setup the PPO model, adding the crucial tensorboard_log argument
model = stable_baselines3.PPO(
    'CnnPolicy',
    env,
    verbose=1,
    tensorboard_log=args.outdir  # This tells SB3 where to save logs
)

# Train the model, passing the callback
print(f"Starting training. Logs will be saved")
model.learn(
    total_timesteps=int(args.steps),
    callback=stability_callback
)
print("Training finished.")