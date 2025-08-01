import argparse
import json
import os
from collections import defaultdict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import crafter


def compute_metrics(actions, positions, rewards, healths, visited_states, action_names, achievements, death_count):
    positions = np.array(positions)
    exploration_variance = np.var(positions, axis=0).sum() if len(positions) > 1 else 0

    state_counts = np.array(list(visited_states.values()))
    state_probs = state_counts / (state_counts.sum() + 1e-10)
    state_entropy = -np.sum(state_probs * np.log2(state_probs + 1e-10))

    action_counts = np.bincount(actions, minlength=len(action_names))
    action_probs = action_counts / (action_counts.sum() + 1e-10)
    action_entropy = -np.sum(action_probs * np.log2(action_probs + 1e-10))

    reward_mean = np.mean(rewards)
    health_mean = np.mean(healths) if healths else 0

    return {
        'exploration_variance': exploration_variance,
        'state_entropy': state_entropy,
        'action_entropy': action_entropy,
        'reward_mean': reward_mean,
        'health_mean': health_mean,
        'total_zombie_defeated': achievements['defeat_zombie'],
        'total_skeleton_defeated': achievements['defeat_skeleton'],
        'total_wake_ups': achievements['wake_up'],
        'total_stones_placed': achievements['place_stone'],
        'total_deaths': death_count,
        'total_episodes': achievements['episodes'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--env', type=str, default='crafter', choices=['crafter', 'homeostatic'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--outdir', type=str, default='logdir/evaluation')
    parser.add_argument('--max_steps', type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Initialize environment
    env = crafter.Env() if args.env == 'crafter' else __import__('homeostatic_crafter').Env()
    env = crafter.Recorder(
        env,
        args.outdir,
        save_stats=True,
        save_video=False,
        save_episode=False
    )
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    model = PPO.load(args.model_path)

    visited_states = defaultdict(int)
    actions, positions, rewards, healths = [], [], [], []
    achievements = defaultdict(int)
    place_stone_actions = 0
    death_count = 0
    episodes_ran = 0
    steps_this_episode = 0
    obs = env.reset()
    action_names = env.get_attr('action_names')[0]

    per_episode_metrics = []

    while episodes_ran < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info = info[0]
        steps_this_episode += 1

        actions.append(action[0])
        rewards.append(reward[0])
        healths.append(info.get('player_health', 0))
        positions.append(info.get('player_pos', (0, 0)))

        state_key = str(obs[0].tobytes())
        visited_states[state_key] += 1

        action_name = action_names[action[0]]
        if action_name == 'place_stone':
            achievements['place_stone'] += 1

        if info.get('discount') == 0.0:
            death_count += 1

        for key in ['defeat_zombie', 'defeat_skeleton', 'wake_up']:
            if info.get('achievements', {}).get(key, 0) > achievements.get(key, 0):
                achievements[key] += 1

        if done[0] or steps_this_episode >= args.max_steps:
            episodes_ran += 1
            steps_this_episode = 0
            achievements['episodes'] += 1
            print(f"Episode {episodes_ran}/{args.episodes} finished.")

            # Store per-episode metrics
            episode_metrics = compute_metrics(
                actions, positions, rewards, healths, visited_states, action_names, achievements.copy(), death_count
            )
            episode_metrics['episode'] = episodes_ran
            per_episode_metrics.append(episode_metrics)

            # Clear per-episode buffers
            actions.clear()
            positions.clear()
            rewards.clear()
            healths.clear()
            visited_states.clear()

            obs = env.reset()

    env.close()

    # Save final cumulative metrics
    final_metrics = compute_metrics(
        actions, positions, rewards, healths, visited_states, action_names, achievements, death_count
    )
    final_metrics = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                     for k, v in final_metrics.items()}
    with open(os.path.join(args.outdir, 'inference_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Save per-episode metrics
    with open(os.path.join(args.outdir, 'inference_episode_metrics.jsonl'), 'w') as f:
        for m in per_episode_metrics:
            m_clean = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in m.items()}
            json.dump(m_clean, f)
            f.write('\n')

    print(f"\nSaved final metrics to: {args.outdir}/inference_metrics.json")
    print(f"Saved per-episode metrics to: {args.outdir}/inference_episode_metrics.jsonl")


if __name__ == '__main__':
    main()
