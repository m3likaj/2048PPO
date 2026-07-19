"""Evaluation: mirrors pufferlib/ocean/g2048/eval.py :: evaluate().

Original protocol (https://github.com/kywch/PufferLib/tree/simple-2048,
MIT License): 4096 envs, scaffolding_ratio = 0, greedy-free stochastic
policy rollouts; stats are per-flush episode means (flushed every
log_interval=128 ticks), aggregated weighted by episode count n.
The printed block matches the original, including the caveat that
Max columns are maxima over flush-means, not single-episode maxima.

Usage:
  python eval.py --checkpoint experiments/<run>/latest.pt
  python eval.py --checkpoint latest.pt --min-episodes 20000
"""

import argparse

import torch

from config import CONFIG
from g2048_env import G2048TorchEnv
from g2048_policy import make_policy, sample_logits


@torch.no_grad()
def evaluate(checkpoint, num_envs=4096, min_episodes=5000, device=None,
             seed=CONFIG['seed']):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    env = G2048TorchEnv(num_envs=num_envs, scaffolding_ratio=0.0,
                        seed=seed, device=device,
                        log_interval=CONFIG['log_interval'])

    policy = make_policy(CONFIG['hidden_size'], CONFIG['rnn_input_size'],
                         CONFIG['rnn_hidden_size']).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt['policy'] if 'policy' in ckpt else ckpt
    policy.load_state_dict(state)
    policy.eval()

    obs = env.reset(seed=seed)
    h = torch.zeros(num_envs, policy.hidden_size, device=device)
    c = torch.zeros(num_envs, policy.hidden_size, device=device)

    stats = {k: [] for k in ['n', 'episode_length', 'score', 'merge_score',
                             'reached_16384', 'reached_32768',
                             'reached_65536', 'reached_131072']}
    episodes = 0
    step = 0
    while episodes < min_episodes:
        state = dict(lstm_h=h, lstm_c=c)
        logits, _value = policy.forward_eval(obs, state)
        h, c = state['lstm_h'], state['lstm_c']
        action, _lp, _ent = sample_logits(logits)
        obs, _r, terminal, _t, info = env.step(action)
        # LSTM state carries across auto-resets, matching training collection.
        if info is not None:
            for k in stats:
                stats[k].append(info[k])
            episodes += info['n']
        step += 1
        if step % 5000 == 0:
            print(f'  ... {episodes:,.0f} episodes after {step:,} steps')

    num_episodes = sum(stats['n'])

    def wmean(key):
        return sum(n * v for n, v in zip(stats['n'], stats[key])) / num_episodes

    episode_lengths = wmean('episode_length')
    max_tiles = wmean('score')
    merge_scores = wmean('merge_score')
    reached_16384 = wmean('reached_16384')
    reached_32768 = wmean('reached_32768')
    reached_65536 = wmean('reached_65536')
    reached_131072 = wmean('reached_131072')

    print(f"Num episodes: {int(num_episodes)}")
    print(f"Max tile avg: {max_tiles:.1f}")
    print(f"Episode length -- Avg: {episode_lengths:.1f}, "
          f"Max: {max(stats['episode_length']):.1f}")
    print(f"Merge score -- Avg: {merge_scores:.1f}, "
          f"Max: {max(stats['merge_score']):.1f}")
    print(f"Reached 16384 prob: {reached_16384*100:.2f} %")
    print(f"Reached 32768 prob: {reached_32768*100:.2f} %")
    print(f"Reached 65536 prob: {reached_65536*100:.2f} %")
    print(f"Reached 131072 prob: {reached_131072*100:.2f} %")

    # Reference numbers from the original branch (embeddings + new reward):
    #   Reached 32768 prob: 84.88 %   Reached 65536 prob: 33.96 %
    return dict(n=num_episodes, r32k=reached_32768, r65k=reached_65536)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--min-episodes', type=int, default=5000)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=CONFIG['seed'])
    a = p.parse_args()
    evaluate(a.checkpoint, a.num_envs, a.min_episodes, a.device, a.seed)
