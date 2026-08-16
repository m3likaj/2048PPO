"""Training loop: faithful single-file replication of PuffeRL for simple-2048.

Replicates pufferlib/pufferl.py from
https://github.com/kywch/PufferLib/tree/simple-2048
(MIT License, Copyright (c) 2022 PufferAI; branch by Kyoung Whan Choe),
launched as `puffer train puffer_g2048` with g2048.ini.

Replicated exactly (see README for the full mapping):
  * Buffer layout [segments, horizon] with segments == total agents; each
    epoch collects exactly bptt_horizon=64 steps per agent.
  * Storage convention: at slot t we store (obs_t, action_t, logprob_t,
    value_t) together with the reward/terminal RECEIVED WITH obs_t, i.e. the
    consequence of action_{t-1}. The vtrace kernel indexes rewards[t+1]
    accordingly.
  * LSTM hidden state zeroed at the start of every collection phase (and
    zero-initialized per minibatch in training), never at episode bounds.
  * Rewards clamped to [-1, 1] at collection time.
  * Advantages recomputed from the FULL buffer before every minibatch using
    the live `ratio` and `values` buffers; values[idx] overwritten with the
    new value predictions after each minibatch.
  * Prioritized segment sampling: p ~ |adv|^prio_alpha (multinomial without
    replacement), importance correction (segments * p)^-beta with
    beta = b0 + (1-b0) * alpha * epoch / total_epochs.
  * PPO clip / clipped value / entropy losses with the exact g2048.ini
    coefficients; grad-norm clip 0.5; one optimizer step per epoch's 32
    minibatches with accumulate_minibatches = 1.
  * heavyball ForeachMuon(lr, betas=(0.99, 0.9999), eps=1e-4,
    heavyball_momentum=True); CosineAnnealingLR(T_max = total_timesteps //
    batch_size, eta_min = 0.15 * lr) stepped once per epoch.

Usage:
  python train.py                     # full original run (6.77B steps)
  python train.py --total-timesteps 1e9 --num-envs 4096   # scaled down
  python train.py --resume experiments/<run>/latest.pt    # continue
"""

import argparse
import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch

from config import CONFIG, resolve
from g2048_env import G2048TorchEnv
from g2048_policy import make_policy, sample_logits
from puffer_advantage import compute_puff_advantage


def make_optimizer(policy, cfg):
    """pufferl.py optimizer block, verbatim behavior."""
    if cfg['optimizer'] == 'adam':
        return torch.optim.Adam(
            policy.parameters(),
            lr=cfg['learning_rate'],
            betas=(cfg['adam_beta1'], cfg['adam_beta2']),
            eps=cfg['adam_eps'],
        )
    if cfg['optimizer'] == 'muon':
        import heavyball
        from heavyball import ForeachMuon
        warnings.filterwarnings(action='ignore', category=UserWarning,
                                module=r'heavyball.*')
        heavyball.utils.compile_mode = 'default'
        # heavyball_momentum=True (heavyball >= 2.1.1) recovers the
        # heavyball-1.7.2 behaviour the original hyperparameters were swept on.
        return ForeachMuon(
            policy.parameters(),
            lr=cfg['learning_rate'],
            betas=(cfg['adam_beta1'], cfg['adam_beta2']),
            eps=cfg['adam_eps'],
            heavyball_momentum=True,
        )
    raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")


class Trainer:
    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        self.run_dir = run_dir
        device = torch.device(cfg['device'])
        self.device = device

        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.deterministic = cfg['torch_deterministic']
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

        self.env = G2048TorchEnv(
            num_envs=cfg['num_envs'],
            scaffolding_ratio=cfg['scaffolding_ratio'],
            seed=cfg['seed'],
            device=cfg['device'],
            log_interval=cfg['log_interval'],
        )

        self.policy = make_policy(
            hidden_size=cfg['hidden_size'],
            rnn_input_size=cfg['rnn_input_size'],
            rnn_hidden_size=cfg['rnn_hidden_size'],
        ).to(device)
        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f'Policy parameters: {n_params:,}')

        self.optimizer = make_optimizer(self.policy, cfg)
        eta_min = cfg['learning_rate'] * cfg['min_lr_ratio']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['total_epochs'], eta_min=eta_min)

        S, H = cfg['segments'], cfg['bptt_horizon']
        self.observations = torch.zeros(S, H, 16, dtype=torch.uint8, device=device)
        self.actions = torch.zeros(S, H, dtype=torch.int32, device=device)
        self.values = torch.zeros(S, H, device=device)
        self.logprobs = torch.zeros(S, H, device=device)
        self.rewards = torch.zeros(S, H, device=device)
        self.terminals = torch.zeros(S, H, device=device)
        self.ratio = torch.ones(S, H, device=device)

        h = self.policy.hidden_size
        self.lstm_h = torch.zeros(cfg['num_envs'], h, device=device)
        self.lstm_c = torch.zeros(cfg['num_envs'], h, device=device)

        self.epoch = 0
        self.global_step = 0
        self.stats = defaultdict(list)
        self.last_stats = {}
        # Extended earned/scaffold tracking (addition, logging-only):
        # window sums since the last CSV row, and cumulative sums for the
        # 20-report progress file. Keys come from the env's xlog.
        self.xwin = defaultdict(float)
        self.xcum = defaultdict(float)
        self.reports_written = 0
        self.run_started = time.time()

        # First recv: reset obs with zero reward/terminal (async_reset).
        self.obs = self.env.reset(seed=cfg['seed'])
        self.prev_reward = torch.zeros(cfg['num_envs'], device=device)
        self.prev_done = torch.zeros(cfg['num_envs'], device=device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        """pufferl.PuffeRL.evaluate(): collect horizon steps per agent."""
        cfg = self.cfg
        self.lstm_h.zero_()
        self.lstm_c.zero_()

        for l in range(cfg['bptt_horizon']):
            o_device = self.obs
            r = self.prev_reward
            d = self.prev_done
            self.global_step += cfg['num_envs']

            state = dict(lstm_h=self.lstm_h, lstm_c=self.lstm_c)
            logits, value = self.policy.forward_eval(o_device, state)
            action, logprob, _ = sample_logits(logits)
            r = torch.clamp(r, -1, 1)

            self.lstm_h = state['lstm_h']
            self.lstm_c = state['lstm_c']

            self.observations[:, l] = o_device
            self.actions[:, l] = action
            self.logprobs[:, l] = logprob
            self.rewards[:, l] = r
            self.terminals[:, l] = d
            self.values[:, l] = value.flatten()

            obs, reward, terminal, _trunc, info = self.env.step(action)
            self.obs = obs
            self.prev_reward = reward
            self.prev_done = terminal.float()

            if info is not None:
                xs = info.pop('xstats', None)
                if xs is not None:
                    for k, v in xs.items():
                        self.xwin[k] += v
                        self.xcum[k] += v
                for k, v in info.items():
                    self.stats[k].append(v)

    # ------------------------------------------------------------------
    def train_epoch(self):
        """pufferl.PuffeRL.train(): one epoch over the collected batch."""
        cfg = self.cfg
        losses = defaultdict(float)

        b0 = cfg['prio_beta0']
        a = cfg['prio_alpha']
        clip_coef = cfg['clip_coef']
        vf_clip = cfg['vf_clip_coef']
        anneal_beta = b0 + (1 - b0) * a * self.epoch / cfg['total_epochs']
        self.ratio[:] = 1

        S = cfg['segments']
        advantages = None
        for mb in range(cfg['total_minibatches']):
            advantages = torch.zeros(self.values.shape, device=self.device)
            advantages = compute_puff_advantage(
                self.values, self.rewards, self.terminals, self.ratio,
                advantages, cfg['gamma'], cfg['gae_lambda'],
                cfg['vtrace_rho_clip'], cfg['vtrace_c_clip'])

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv ** a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, cfg['minibatch_segments'])
            mb_prio = (S * prio_probs[idx, None]) ** -anneal_beta

            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
            logits, newvalue = self.policy(mb_obs, state)
            _actions, newlogprob, entropy_t = sample_logits(logits, action=mb_actions)

            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            # Weight advantages by priority and normalize
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy_t.mean()

            loss = (pg_loss + cfg['vf_coef'] * v_loss
                    - cfg['ent_coef'] * entropy_loss)

            self.values[idx] = newvalue.detach().float()

            losses['policy_loss'] += pg_loss.item() / cfg['total_minibatches']
            losses['value_loss'] += v_loss.item() / cfg['total_minibatches']
            losses['entropy'] += entropy_loss.item() / cfg['total_minibatches']
            losses['old_approx_kl'] += old_approx_kl.item() / cfg['total_minibatches']
            losses['approx_kl'] += approx_kl.item() / cfg['total_minibatches']
            losses['clipfrac'] += clipfrac.item() / cfg['total_minibatches']
            losses['importance'] += ratio.mean().item() / cfg['total_minibatches']

            loss.backward()
            if (mb + 1) % cfg['accumulate_minibatches'] == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               cfg['max_grad_norm'])
                self.optimizer.step()
                self.optimizer.zero_grad()

        if cfg['anneal_lr']:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        losses['explained_variance'] = (
            float('nan') if var_y == 0
            else (1 - (y_true - y_pred).var() / var_y).item())

        self.epoch += 1
        return losses

    # ------------------------------------------------------------------
    def mean_stats(self):
        out = {}
        for k, v in self.stats.items():
            if k == 'n':
                out[k] = float(np.sum(v))
            else:
                ns = self.stats.get('n', [1] * len(v))
                tot_n = max(float(np.sum(ns)), 1.0)
                out[k] = float(np.sum([a * b for a, b in zip(v, ns)]) / tot_n)
        if out:
            self.last_stats = out
        self.stats = defaultdict(list)
        return self.last_stats

    def write_report(self, report_path, losses):
        """Append one human-readable progress line (of ~20 total per run).

        Cumulative earned/reached milestone percentages come WITH raw counts
        so tiny rates (e.g. 0.002%) are never rounded away.
        """
        C = self.xcum
        n_norm, n_scaf = C['xn_norm'], C['xn_scaf']
        n_all = n_norm + n_scaf
        elapsed = time.time() - self.run_started

        def cnt(prefix, m):
            return C[f'{prefix}{m}_norm'] + C[f'{prefix}{m}_scaf']

        def pct(x):
            return 100.0 * x / max(n_all, 1.0)

        avg_max_norm = C['xsum_maxtile_norm'] / max(n_norm, 1.0)
        avg_earned_all = ((C['xsum_earnedmax_norm'] + C['xsum_earnedmax_scaf'])
                          / max(n_all, 1.0))
        merge_score_norm = C['xsum_mergescore_norm'] / max(n_norm, 1.0)
        ep_len_all = ((C['xsum_len_norm'] + C['xsum_len_scaf'])
                      / max(n_all, 1.0))
        ev = losses.get('explained_variance', float('nan'))

        self.reports_written += 1
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            f"[{ts}] report {self.reports_written:2d} | "
            f"epoch {self.epoch}/{self.cfg['total_epochs']} | "
            f"step {self.global_step:.4e} | elapsed {elapsed/3600:.2f}h",
            f"  episodes {n_all:,.0f} (normal {n_norm:,.0f} / "
            f"scaffold {n_scaf:,.0f}) | avg ep len {ep_len_all:.1f}",
            f"  score(merge, normal eps) {merge_score_norm:,.1f} | "
            f"EV {ev:.4f} | AvgMaxTile(normal) {avg_max_norm:,.1f} | "
            f"AvgEarnedMaxTile(all) {avg_earned_all:,.1f}",
        ]
        for label, m in (('8k', 13), ('16k', 14), ('32k', 15),
                         ('65k', 16), ('131k', 17)):
            e, r = cnt('xe', m), cnt('xr', m)
            lines.append(
                f"  {label:>4}: earned {pct(e):8.4f}% (n={e:,.0f})   "
                f"reached {pct(r):8.4f}% (n={r:,.0f})   "
                f"scaffold-only {pct(r - e):8.4f}% (n={r - e:,.0f})")
        block = '\n'.join(lines) + '\n'
        with open(report_path, 'a') as f:
            f.write(block)
        print(block, end='', flush=True)

    def checkpoint(self, name):
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, name)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'xcum': dict(self.xcum),
            'reports_written': self.reports_written,
            'config': {k: v for k, v in self.cfg.items()},
        }, path)
        # latest.pt duplicates the full state so --resume (added for
        # Kaggle session limits) can always pick up exactly where it left off.
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'xcum': dict(self.xcum),
            'reports_written': self.reports_written,
            'config': {k: v for k, v in self.cfg.items()},
        }, os.path.join(self.run_dir, 'latest.pt'))
        return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--num-envs', type=int, default=CONFIG['num_envs'])
    p.add_argument('--total-timesteps', type=float,
                   default=CONFIG['total_timesteps'])
    p.add_argument('--scaffolding-ratio', type=float,
                   default=CONFIG['scaffolding_ratio'])
    p.add_argument('--seed', type=int, default=CONFIG['seed'])
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--optimizer', type=str, default=CONFIG['optimizer'],
                   choices=['muon', 'adam'],
                   help='muon = original (heavyball ForeachMuon). '
                        'adam = fallback if heavyball is unavailable.')
    p.add_argument('--data-dir', type=str, default=CONFIG['data_dir'])
    p.add_argument('--checkpoint-interval', type=int,
                   default=CONFIG['checkpoint_interval'])
    p.add_argument('--load-model-path', type=str, default=None,
                   help='Load policy weights only (mirrors PufferLib).')
    p.add_argument('--resume', type=str, default=None,
                   help='Full resume: policy + optimizer + scheduler + epoch.')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='g2048-recreation')
    p.add_argument('--tag', type=str, default='pg2048')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    cfg['num_envs'] = args.num_envs
    cfg['total_timesteps'] = int(args.total_timesteps)
    cfg['scaffolding_ratio'] = args.scaffolding_ratio
    cfg['seed'] = args.seed
    cfg['device'] = args.device
    cfg['optimizer'] = args.optimizer
    cfg['checkpoint_interval'] = args.checkpoint_interval
    cfg = resolve(cfg)

    run_name = f"g2048_{args.tag}_seed{cfg['seed']}_{int(time.time())}"
    run_dir = os.path.join(args.data_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({k: str(v) for k, v in cfg.items()}, f, indent=2)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                               config=cfg)

    trainer = Trainer(cfg, run_dir)

    if args.load_model_path:
        ckpt = torch.load(args.load_model_path, map_location=cfg['device'],
                          weights_only=False)
        state = ckpt['policy'] if 'policy' in ckpt else ckpt
        trainer.policy.load_state_dict(state)
        print(f'Loaded policy weights from {args.load_model_path}')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg['device'],
                          weights_only=False)
        trainer.policy.load_state_dict(ckpt['policy'])
        if 'optimizer' in ckpt:
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
            trainer.scheduler.load_state_dict(ckpt['scheduler'])
            trainer.epoch = ckpt['epoch']
            trainer.global_step = ckpt.get('global_step',
                                           trainer.epoch * cfg['batch_size'])
            for k, v in ckpt.get('xcum', {}).items():
                trainer.xcum[k] = v
            trainer.reports_written = ckpt.get('reports_written', 0)
        else:
            print('WARNING: checkpoint has no optimizer state '
                  '(policy-only file); counters start from 0. '
                  'Use --load-model-path for policy-only warm starts.')
        print(f"Resumed from {args.resume} at epoch {trainer.epoch}")

    print(f"batch_size={cfg['batch_size']:,}  segments={cfg['segments']:,}  "
          f"minibatches/epoch={cfg['total_minibatches']}  "
          f"total_epochs={cfg['total_epochs']:,}")

    last_time, last_step = time.time(), trainer.global_step
    csv_path = os.path.join(run_dir, 'log.csv')
    csv_header_written = os.path.exists(csv_path)
    report_path = os.path.join(run_dir, 'report.txt')
    # ~20 evenly spaced progress reports over the whole schedule.
    report_interval = max(1, cfg['total_epochs'] // 20)

    xkeys = trainer.env._xlog_keys
    last_report_epoch = -1

    while trainer.global_step < cfg['total_timesteps']:
        trainer.evaluate()
        losses = trainer.train_epoch()

        if trainer.epoch % max(1, cfg['checkpoint_interval'] // 20) == 0 or \
                trainer.epoch <= 3:
            now = time.time()
            sps = (trainer.global_step - last_step) / max(now - last_time, 1e-9)
            last_time, last_step = now, trainer.global_step
            stats = trainer.mean_stats()
            lr = trainer.scheduler.get_last_lr()[0]
            line = {
                'epoch': trainer.epoch,
                'global_step': trainer.global_step,
                'sps': int(sps),
                'lr': lr,
                **{f'loss/{k}': v for k, v in losses.items()},
                **{f'env/{k}': v for k, v in stats.items()},
                **{f'x/{k}': trainer.xwin.get(k, 0.0) for k in xkeys},
            }
            trainer.xwin = defaultdict(float)
            msg = (f"epoch {trainer.epoch:6d} | step {trainer.global_step:.3e} "
                   f"| SPS {int(sps):8,d} | lr {lr:.2e}")
            if stats:
                msg += (f" | score {stats.get('score', 0):9.1f}"
                        f" | ep_len {stats.get('episode_length', 0):8.1f}"
                        f" | 32k {100*stats.get('reached_32768', 0):5.2f}%"
                        f" | 65k {100*stats.get('reached_65536', 0):5.2f}%")
            print(msg, flush=True)
            if wandb_run is not None:
                wandb_run.log(line, step=trainer.global_step)
            import csv as _csv
            loss_keys = ['policy_loss', 'value_loss', 'entropy',
                         'old_approx_kl', 'approx_kl', 'clipfrac',
                         'importance', 'explained_variance']
            env_keys = ['perf', 'score', 'merge_score', 'episode_return',
                        'episode_length', 'lifetime_max_tile',
                        'reached_16384', 'reached_32768', 'reached_65536',
                        'reached_131072', 'n']
            fieldnames = (['epoch', 'global_step', 'sps', 'lr']
                          + [f'loss/{k}' for k in loss_keys]
                          + [f'env/{k}' for k in env_keys]
                          + [f'x/{k}' for k in xkeys])
            with open(csv_path, 'a', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=fieldnames, restval='',
                                    extrasaction='ignore')
                if not csv_header_written:
                    w.writeheader()
                    csv_header_written = True
                w.writerow(line)

        if trainer.epoch % report_interval == 0:
            trainer.write_report(report_path, losses)
            last_report_epoch = trainer.epoch

        if trainer.epoch % cfg['checkpoint_interval'] == 0:
            path = trainer.checkpoint(f'model_{trainer.epoch:06d}.pt')
            print(f'checkpoint -> {path}', flush=True)

    if last_report_epoch != trainer.epoch:
        trainer.write_report(report_path, losses)   # final report
    path = trainer.checkpoint(f'model_{trainer.epoch:06d}_final.pt')
    print(f'Training complete. Final checkpoint -> {path}')
    try:
        from plot_training import make_plots
        outs = make_plots(csv_path, os.path.join(run_dir, 'plots'))
        print(f'Wrote {len(outs)} plots -> {os.path.join(run_dir, "plots")}')
    except Exception as e:      # plotting must never kill a finished run
        print(f'(plots skipped: {e}; run `python plot_training.py '
              f'--csv {csv_path}` later)')
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
