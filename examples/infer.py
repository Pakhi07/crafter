import argparse
import crafter
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
# No external wrappers needed for this version

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .zip model file.')
    parser.add_argument('--env', type=str, default='crafter', help='Should be "crafter" for this script.')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run for evaluation.')
    parser.add_argument('--outdir', type=str, default='logdir/evaluation', help='Directory to save evaluation stats.')
    parser.add_argument('--max_steps', type=int, default=2000, help='Max steps per evaluation episode.')
    args = parser.parse_args()

    # --- 1. Set up the environment ---
    print("Setting up standard Crafter for evaluation.")
    env = crafter.Env()
        
    # --- 2. Add the necessary wrappers ---
    env = crafter.Recorder(
        env,
        args.outdir,
        save_stats=True,
        save_video=False,
        save_episode=False
    )
    # The DummyVecEnv and VecTransposeImage are still needed for SB3
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    # --- 3. Load the pre-trained model ---
    print(f"Loading model from: {args.model_path}")
    model = PPO.load(args.model_path)
    
    # --- 4. Run the evaluation loop with corrected timeout logic ---
    obs = env.reset()
    episodes_ran = 0
    steps_this_episode = 0

    while episodes_ran < args.episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        steps_this_episode += 1

        # Check if the episode ended naturally (done flag) or via timeout
        timed_out = steps_this_episode >= args.max_steps
        
        if dones[0] or timed_out:
            episodes_ran += 1
            print(f"Episode {episodes_ran}/{args.episodes} finished (Reason: {'Timeout' if timed_out else 'Done'}).")
            
            # If it was a timeout, we need to manually reset the environment
            # This ensures the Recorder logs the episode before starting the next one.
            if timed_out:
                obs = env.reset()

            steps_this_episode = 0 # Reset step counter for the new episode
    
    env.close()
    print(f"\nEvaluation finished. Achievement stats saved in {args.outdir}/stats.jsonl")

if __name__ == '__main__':
    main()