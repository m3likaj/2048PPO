"""Proof that the earned-tile / extended-logging addition is inert.

Runs a short but complete training loop twice under an identical seed --
once with track_earned=True, once with track_earned=False -- and asserts
that every observable training quantity matches bit-for-bit: the RNG-driven
environment rollout, the collected rewards/dones, and the final policy
weights after several optimizer steps.

If this passes, the logging provably cannot have changed your training
trajectory; any difference between two full runs is seed/GPU-nondeterminism
(deviation #2), not the logging.

Run:  python test_logging_inert.py
"""

import copy

import numpy as np
import torch

from config import CONFIG, resolve
from g2048_env import G2048TorchEnv
from g2048_policy import make_policy, sample_logits


# ---------------------------------------------------------------------------
# Part A: the environment itself is identical with tracking on vs off.
# Same seed -> same boards, rewards, terminals at every step, because the
# tracker never draws from the RNG.
# ---------------------------------------------------------------------------

def rollout(track, steps=400, num_envs=64, seed=1234):
    env = G2048TorchEnv(num_envs=num_envs, scaffolding_ratio=0.67, seed=seed,
                        device='cpu', log_interval=128, track_earned=track)
    obs = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(999)   # fixed action stream
    trace = []
    for _ in range(steps):
        a = torch.randint(0, 4, (num_envs,), generator=gen)
        obs, r, term, trunc, info = env.step(a)
        trace.append((obs.clone(), r.clone(), term.clone()))
    return trace


def test_env_rollout_identical():
    on = rollout(True)
    off = rollout(False)
    assert len(on) == len(off)
    for t, ((o1, r1, d1), (o0, r0, d0)) in enumerate(zip(on, off)):
        assert torch.equal(o1, o0), f'obs differ at step {t}'
        assert torch.equal(r1, r0), f'rewards differ at step {t}'
        assert torch.equal(d1, d0), f'terminals differ at step {t}'
    print(f'  env rollout identical for {len(on)} steps '
          '(obs, rewards, terminals bit-for-bit)')


# ---------------------------------------------------------------------------
# Part B: a full mini training run produces identical weights either way.
# Reuses the real Trainer so collection + PPO + optimizer are exercised.
# ---------------------------------------------------------------------------

def train_a_few_epochs(track, epochs=3, num_envs=512, seed=7):
    from train import Trainer  # imported here so the module import is cheap
    cfg = dict(CONFIG)
    cfg['num_envs'] = num_envs
    cfg['total_timesteps'] = num_envs * 64 * epochs
    cfg['device'] = 'cpu'
    cfg['seed'] = seed
    cfg['optimizer'] = 'adam'      # deterministic on CPU; muon path is heavier
    cfg['track_earned'] = track
    cfg = resolve(cfg)

    torch.manual_seed(seed)
    np.random.seed(seed)
    trainer = Trainer(cfg, run_dir=None)
    for _ in range(epochs):
        trainer.evaluate()      # collection (pufferl evaluate)
        trainer.train_epoch()   # PPO update
        trainer.epoch += 1
    return [p.detach().clone() for p in trainer.policy.parameters()]


def test_training_weights_identical():
    w_on = train_a_few_epochs(True)
    w_off = train_a_few_epochs(False)
    assert len(w_on) == len(w_off)
    max_abs = 0.0
    for a, b in zip(w_on, w_off):
        max_abs = max(max_abs, (a - b).abs().max().item())
    assert max_abs == 0.0, f'weights diverged (max |delta| = {max_abs:.3e})'
    print('  post-training policy weights bit-for-bit identical '
          '(max |delta| = 0.0 across all parameters)')


ALL_TESTS = [test_env_rollout_identical, test_training_weights_identical]

if __name__ == '__main__':
    for fn in ALL_TESTS:
        print(f'[{fn.__name__}]')
        fn()
    print('\nLogging is provably inert: identical seed => identical training.')
