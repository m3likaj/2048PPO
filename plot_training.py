"""Training-curve plots from log.csv (addition, logging-only).

Produces per-milestone earned/reached/scaffold-only percentage curves plus
loss, explained-variance, learning-rate, throughput, score, max-tile, and
episode-length figures -- everything needed to write up a run.

Definitions (see README "Earned-tile tracking"):
  earned    = episode created a tile >= milestone BY MERGING (any merge
              result counts as earned, even if one parent was scaffolded).
  reached   = episode max tile >= milestone (original metric's definition).
  scaffold-only = reached but not earned (the milestone tile was only ever
              a scaffold placement).
Percentages are per CSV-row window over ALL episodes (normal + scaffold);
dashed lines show the normal-episode-only rate, comparable to the original
reached_* logs.

Usage:
  python plot_training.py --csv experiments/<run>/log.csv
  python plot_training.py --csv .../log.csv --out .../plots
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MILESTONES = [(13, '8192'), (14, '16384'), (15, '32768'),
              (16, '65536'), (17, '131072')]


def _f(row, key):
    v = row.get(key, '')
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        return [dict(r) for r in csv.DictReader(f)]


def _finite(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys)
           if y == y and y not in (float('inf'), float('-inf'))]
    return ([p[0] for p in pts], [p[1] for p in pts])


def _save(fig, outdir, name, paths):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)


def make_plots(csv_path, outdir):
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f'No rows in {csv_path}')
    os.makedirs(outdir, exist_ok=True)
    paths = []
    ep = [_f(r, 'epoch') for r in rows]

    # Window episode counts
    n_norm = [_f(r, 'x/xn_norm') for r in rows]
    n_scaf = [_f(r, 'x/xn_scaf') for r in rows]
    n_all = [a + b for a, b in zip(n_norm, n_scaf)]

    def rate(num, den):
        return [100.0 * x / d if d and d == d and x == x else float('nan')
                for x, d in zip(num, den)]

    # --- One figure per milestone: earned vs reached vs scaffold-only ---
    for m, label in MILESTONES:
        e_all = [_f(r, f'x/xe{m}_norm') + _f(r, f'x/xe{m}_scaf') for r in rows]
        r_all = [_f(r, f'x/xr{m}_norm') + _f(r, f'x/xr{m}_scaf') for r in rows]
        r_norm = [_f(r, f'x/xr{m}_norm') for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(*_finite(ep, rate(e_all, n_all)), label='earned (all eps)',
                color='tab:green', lw=2)
        ax.plot(*_finite(ep, rate(r_all, n_all)), label='reached (all eps)',
                color='tab:blue', lw=2)
        ax.plot(*_finite(ep, rate([r - e for r, e in zip(r_all, e_all)],
                                  n_all)),
                label='scaffold-only (reached, not earned)',
                color='tab:orange', lw=2)
        ax.plot(*_finite(ep, rate(r_norm, n_norm)),
                label='reached (normal eps only, original metric)',
                color='tab:blue', ls='--', lw=1.2)
        ax.set_xlabel('epoch'); ax.set_ylabel('% of episodes')
        ax.set_title(f'Milestone {label}: earned vs scaffolded')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        _save(fig, outdir, f'milestone_{label}.png', paths)

    # --- Losses ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(
            axes, ['loss/policy_loss', 'loss/value_loss', 'loss/entropy'],
            ['Policy loss', 'Value loss', 'Entropy']):
        ax.plot(*_finite(ep, [_f(r, key) for r in rows]), lw=1.5)
        ax.set_xlabel('epoch'); ax.set_title(title); ax.grid(alpha=0.3)
    _save(fig, outdir, 'losses.png', paths)

    # --- Explained variance ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(*_finite(ep, [_f(r, 'loss/explained_variance') for r in rows]),
            lw=1.5, color='tab:purple')
    ax.set_xlabel('epoch'); ax.set_ylabel('EV')
    ax.set_title('Explained variance'); ax.grid(alpha=0.3)
    _save(fig, outdir, 'explained_variance.png', paths)

    # --- Score / max tile ---
    avg_max_norm = [x / n if n else float('nan')
                    for x, n in zip([_f(r, 'x/xsum_maxtile_norm') for r in rows],
                                    n_norm)]
    avg_earned_all = [(a + b) / n if n else float('nan')
                      for a, b, n in zip(
                          [_f(r, 'x/xsum_earnedmax_norm') for r in rows],
                          [_f(r, 'x/xsum_earnedmax_scaf') for r in rows],
                          n_all)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(*_finite(ep, avg_max_norm), label='avg max tile (normal eps)',
            color='tab:blue', lw=2)
    ax.plot(*_finite(ep, avg_earned_all),
            label='avg EARNED max tile (all eps)', color='tab:green', lw=2)
    ax.set_xlabel('epoch'); ax.set_ylabel('tile value'); ax.set_yscale('log', base=2)
    ax.set_title('Average max tile'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, outdir, 'avg_max_tile.png', paths)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(*_finite(ep, [_f(r, 'env/merge_score') for r in rows]),
            lw=1.5, color='tab:red')
    ax.set_xlabel('epoch'); ax.set_ylabel('game score')
    ax.set_title('Merge score (normal eps)'); ax.grid(alpha=0.3)
    _save(fig, outdir, 'merge_score.png', paths)

    # --- Episode length by class ---
    len_norm = [x / n if n else float('nan')
                for x, n in zip([_f(r, 'x/xsum_len_norm') for r in rows], n_norm)]
    len_scaf = [x / n if n else float('nan')
                for x, n in zip([_f(r, 'x/xsum_len_scaf') for r in rows], n_scaf)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(*_finite(ep, len_norm), label='normal eps', color='tab:blue', lw=2)
    ax.plot(*_finite(ep, len_scaf), label='scaffold eps', color='tab:orange', lw=2)
    ax.set_xlabel('epoch'); ax.set_ylabel('steps')
    ax.set_title('Average episode length'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, outdir, 'episode_length.png', paths)

    # --- LR + throughput ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(*_finite(ep, [_f(r, 'lr') for r in rows]), lw=1.5)
    axes[0].set_title('Learning rate (cosine)'); axes[0].set_xlabel('epoch')
    axes[0].grid(alpha=0.3)
    axes[1].plot(*_finite(ep, [_f(r, 'sps') for r in rows]), lw=1.5,
                 color='tab:gray')
    axes[1].set_title('Steps / second'); axes[1].set_xlabel('epoch')
    axes[1].grid(alpha=0.3)
    _save(fig, outdir, 'lr_and_sps.png', paths)

    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True)
    p.add_argument('--out', type=str, default=None,
                   help='default: <csv dir>/plots')
    a = p.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.csv), 'plots')
    for path in make_plots(a.csv, out):
        print(path)
