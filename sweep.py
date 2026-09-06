"""Encoder sweep runner: attention configs vs MLP, identical training conditions.

Two modes:

  COMPARISON sweep (default): mlp / attention / mlp_matched, one fixed
  attention config. Used for the main attention-vs-MLP comparison.

  WIDTH sweep (--mode width): multiple attention configs (d_model 32/64/96
  + depth variant), attention-only arms. Used to find the best attention
  config before the full-budget head-to-head. You already have d_model=64
  from the comparison sweep, so it is included for reference but skipped if
  already done.

In both modes:
  - Every arm runs at the same budget, num_envs, scaffolding, reward, seed.
  - Each arm has a stable directory name -> fully resumable.
  - --session-hours stops cleanly before Kaggle kills the session.
  - Re-running the same command skips DONE arms and resumes partial ones.
  - Failures are recorded and the sweep continues.

Typical use:

  # comparison sweep (account 1, seed 43 after seed 42 is done)
  python sweep.py --mode comparison --seeds 43 --session-hours 11

  # width sweep (account 2, running in parallel)
  python sweep.py --mode width --seeds 42 --session-hours 11

  # reports
  python sweep.py --report --mode comparison --seeds 42 43
  python sweep.py --report --mode width --seeds 42
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

# Comparison sweep arms (fixed attention config)
COMPARISON_ARMS = ['mlp', 'attention', 'mlp_matched']

# Width sweep: attention-only configs to compare
# name -> (d_model, n_layers, n_heads)
WIDTH_CONFIGS = {
    'attn_d32_l1':  (32,  1, 4),
    'attn_d64_l1':  (64,  1, 4),   # already run in comparison sweep
    'attn_d96_l1':  (96,  1, 8),
    'attn_d64_l2':  (64,  2, 4),   # depth variant
}


def config_tag(name, d_model=64, n_layers=1, n_heads=4):
    """Stable string for directory naming."""
    return name


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def run_name_comparison(encoder, seed, budget):
    return f'sweep_{encoder}_seed{seed}_b{budget:.0e}'.replace('+', '')


def run_name_width(cfg_name, seed, budget):
    return f'width_{cfg_name}_seed{seed}_b{budget:.0e}'.replace('+', '')


def run_dir_path(data_dir, name):
    return os.path.join(data_dir, name)


def is_done(d):
    return os.path.exists(os.path.join(d, 'DONE'))


def latest_ckpt(d):
    c = os.path.join(d, 'latest.pt')
    return c if os.path.exists(c) else None


# ---------------------------------------------------------------------------
# Kaggle inter-session seeding
# ---------------------------------------------------------------------------

def seed_from_prior_outputs(data_dir, pattern='sweep_*'):
    if not os.path.isdir('/kaggle/input'):
        return
    import shutil
    for src in glob.glob(f'/kaggle/input/**/{pattern}', recursive=True):
        if not os.path.isdir(src):
            continue
        dst = os.path.join(data_dir, os.path.basename(src))
        if os.path.exists(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, '*')):
            if os.path.isfile(f):
                import shutil as sh
                sh.copy2(f, dst)
        print(f'  seeded {os.path.basename(src)} from prior output')


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch(args, run_name, encoder, seed, hours_left,
           d_model=64, n_layers=1, n_heads=4):
    d = run_dir_path(args.data_dir, run_name)
    cmd = [sys.executable, 'train.py',
           '--encoder', encoder,
           '--seed', str(seed),
           '--num-envs', str(args.num_envs),
           '--total-timesteps', str(int(args.budget)),
           '--data-dir', args.data_dir,
           '--run-name', run_name,
           '--max-hours', f'{hours_left:.3f}']
    if encoder in ('attention', 'mlp_matched'):
        cmd += ['--d-model', str(d_model),
                '--n-layers', str(n_layers),
                '--n-heads', str(n_heads)]
    ck = latest_ckpt(d)
    if ck:
        cmd += ['--resume', ck]
        print(f'\n=== RESUME {run_name} ===')
    else:
        print(f'\n=== START  {run_name} ===')
    print('   ', ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = (time.time() - t0) / 3600
    print(f'--- {run_name}: rc={rc}, {elapsed:.2f}h used')
    if rc != 0:
        with open(os.path.join(d, 'FAILED'), 'a') as f:
            f.write(f'rc={rc} after {elapsed:.2f}h at {time.ctime()}\n')
        hint = ''
        if rc == -9:
            hint = ('  (OOM -- try smaller --d-model or fewer --n-layers)')
        print(f'!!! {run_name} FAILED rc={rc}{hint}')
    return rc, elapsed


# ---------------------------------------------------------------------------
# Summarise a run from its log.csv
# ---------------------------------------------------------------------------

MILESTONES = [(13, '8k'), (14, '16k'), (15, '32k'), (16, '65k')]


def summarize_run(d):
    import csv
    path = os.path.join(d, 'log.csv')
    if not os.path.exists(path):
        return None
    keys = ['xn_norm', 'xn_scaf']
    for m, _ in MILESTONES:
        keys += [f'xe{m}_norm', f'xe{m}_scaf',
                 f'xr{m}_norm', f'xr{m}_scaf']
    tot = {k: 0.0 for k in keys}
    first16k_step = None
    last_epoch = 0
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for k in keys:
                v = row.get('x/' + k, '')
                if v not in ('', None):
                    tot[k] += float(v)
            if first16k_step is None:
                e16 = (float(row.get('x/xe14_norm') or 0)
                       + float(row.get('x/xe14_scaf') or 0))
                if e16 > 0:
                    first16k_step = float(row.get('global_step') or 0)
            last_epoch = int(float(row.get('epoch') or 0))
    n = tot['xn_norm'] + tot['xn_scaf']
    if n == 0:
        return None
    out = {'episodes': n, 'epoch': last_epoch, 'done': is_done(d),
           'first_earned_16k_step': first16k_step}
    for m, label in MILESTONES:
        e = tot[f'xe{m}_norm'] + tot[f'xe{m}_scaf']
        r = tot[f'xr{m}_norm'] + tot[f'xr{m}_scaf']
        out[f'earned_{label}'] = 100 * e / n
        out[f'earned_{label}_n'] = e
        out[f'reached_{label}'] = 100 * r / n
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_comparison(args):
    import statistics
    rows = {}
    for enc in COMPARISON_ARMS:
        for seed in args.seeds:
            name = run_name_comparison(enc, seed, args.budget)
            d = run_dir_path(args.data_dir, name)
            s = summarize_run(d)
            if s:
                rows.setdefault(enc, []).append((seed, s))

    if not rows:
        print('No comparison sweep runs found in', args.data_dir)
        return

    print(f'\nComparison sweep @ {args.budget:.2e} steps, '
          f'num_envs={args.num_envs}\n')
    hdr = (f"{'arm':15s} {'seed':>5s} {'epoch':>7s} {'eps':>10s} "
           f"{'earn8k%':>9s} {'earn16k%':>9s} {'earn32k%':>9s} "
           f"{'16k_n':>7s} {'done':>5s}")
    print(hdr); print('-' * len(hdr))
    for enc in COMPARISON_ARMS:
        for seed, s in sorted(rows.get(enc, [])):
            print(f"{enc:15s} {seed:5d} {s['epoch']:7d} "
                  f"{s['episodes']:10,.0f} "
                  f"{s['earned_8k']:9.3f} {s['earned_16k']:9.3f} "
                  f"{s['earned_32k']:9.3f} {s['earned_16k_n']:7,.0f} "
                  f"{'Y' if s['done'] else '-':>5s}")

    print('\nMean +/- stdev across seeds (complete runs only):')
    base16k = None
    for enc in COMPARISON_ARMS:
        done = [s for _, s in rows.get(enc, []) if s['done']]
        if not done:
            print(f'  {enc:15s} (no completed runs)')
            continue
        vals8  = [s['earned_8k']  for s in done]
        vals16 = [s['earned_16k'] for s in done]
        mu8  = sum(vals8)  / len(vals8)
        mu16 = sum(vals16) / len(vals16)
        sd8  = (sum((x-mu8)**2  for x in vals8)  / max(len(vals8)-1,1))**0.5
        sd16 = (sum((x-mu16)**2 for x in vals16) / max(len(vals16)-1,1))**0.5
        print(f'  {enc:15s} n={len(done)} | '
              f'earn8k {mu8:.3f}+/-{sd8:.3f}  earn16k {mu16:.4f}+/-{sd16:.4f}')
        if enc == 'mlp':
            base16k = mu16
        elif base16k and mu16:
            print(f'  {"":15s} vs mlp earn16k: {100*(mu16-base16k)/base16k:+.1f}%')

    _print_failures(args.data_dir, 'sweep_*')


def report_width(args):
    rows = {}
    for cfg_name in WIDTH_CONFIGS:
        for seed in args.seeds:
            name = run_name_width(cfg_name, seed, args.budget)
            d = run_dir_path(args.data_dir, name)
            s = summarize_run(d)
            if s:
                rows.setdefault(cfg_name, []).append((seed, s))

    if not rows:
        print('No width sweep runs found in', args.data_dir)
        return

    print(f'\nWidth sweep @ {args.budget:.2e} steps, '
          f'num_envs={args.num_envs}\n')
    print(f"{'config':16s} {'d':>5s} {'L':>3s} {'H':>3s} "
          f"{'seed':>5s} {'epoch':>7s} {'earn8k%':>9s} "
          f"{'earn16k%':>9s} {'earn32k%':>9s} {'done':>5s}")
    print('-' * 78)
    for cfg_name, (dm, nl, nh) in WIDTH_CONFIGS.items():
        for seed, s in sorted(rows.get(cfg_name, [])):
            print(f"{cfg_name:16s} {dm:5d} {nl:3d} {nh:3d} "
                  f"{seed:5d} {s['epoch']:7d} "
                  f"{s['earned_8k']:9.3f} {s['earned_16k']:9.4f} "
                  f"{s['earned_32k']:9.4f} "
                  f"{'Y' if s['done'] else '-':>5s}")

    print('\nRanking by earned_16k (complete runs only):')
    ranked = []
    for cfg_name in WIDTH_CONFIGS:
        done = [s for _, s in rows.get(cfg_name, []) if s['done']]
        if done:
            mu = sum(s['earned_16k'] for s in done) / len(done)
            ranked.append((mu, cfg_name, len(done)))
    for mu, name, n in sorted(ranked, reverse=True):
        dm, nl, nh = WIDTH_CONFIGS[name]
        print(f'  {name:16s} d={dm} L={nl} H={nh}  '
              f'earn16k {mu:.4f}%  (n={n} seed{"s" if n>1 else ""})')

    _print_failures(args.data_dir, 'width_*')


def _print_failures(data_dir, pattern):
    failed = [os.path.basename(d) for d in
              glob.glob(os.path.join(data_dir, pattern))
              if os.path.exists(os.path.join(d, 'FAILED'))
              and not os.path.exists(os.path.join(d, 'DONE'))]
    if failed:
        print('\n!!! FAILED arms: ' + ', '.join(failed))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['comparison', 'width'],
                   default='comparison',
                   help='comparison: mlp vs attention vs mlp_matched. '
                        'width: attention configs at different d_model/depth.')
    p.add_argument('--budget', type=float, default=1e9)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--data-dir', type=str, default='experiments_sweep')
    p.add_argument('--session-hours', type=float, default=11.0)
    p.add_argument('--report', action='store_true')
    # attention hyperparams (comparison mode only)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=1)
    p.add_argument('--n-heads', type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.report:
        if args.mode == 'comparison':
            report_comparison(args)
        else:
            report_width(args)
        return

    seed_from_prior_outputs(args.data_dir,
                            'sweep_*' if args.mode == 'comparison'
                            else 'width_*')

    started = time.time()

    if args.mode == 'comparison':
        queue = [(run_name_comparison(enc, sd, args.budget),
                  enc, sd, args.d_model, args.n_layers, args.n_heads)
                 for sd in args.seeds for enc in COMPARISON_ARMS]
    else:
        # Width sweep: seed-major over configs (excluding d64_l1 if already
        # done in the comparison sweep -- check by looking for its DONE file).
        queue = []
        for sd in args.seeds:
            for cfg_name, (dm, nl, nh) in WIDTH_CONFIGS.items():
                name = run_name_width(cfg_name, sd, args.budget)
                queue.append((name, 'attention', sd, dm, nl, nh))

    for name, encoder, seed, dm, nl, nh in queue:
        d = run_dir_path(args.data_dir, name)

        # For width sweep: check if this config was already done in the
        # comparison sweep directory (d64_l1 = the current baseline).
        if args.mode == 'width' and dm == 64 and nl == 1:
            comp_name = run_name_comparison('attention', seed, args.budget)
            comp_dir  = run_dir_path(args.data_dir, comp_name)
            if is_done(comp_dir):
                print(f'skip (done in comparison sweep): {name} '
                      f'-> using {comp_name}')
                # symlink so the report can find it
                if not os.path.exists(d):
                    try:
                        os.symlink(comp_dir, d)
                    except Exception:
                        pass
                continue

        if is_done(d):
            print(f'skip (done): {name}')
            continue

        left = args.session_hours - (time.time() - started) / 3600
        if left < 0.2:
            print(f'\nSession budget exhausted. Re-run next session.')
            break

        launch(args, name, encoder, seed, left, dm, nl, nh)

    with open(os.path.join(args.data_dir, 'sweep_state.json'), 'w') as f:
        json.dump({'mode': args.mode, 'budget': args.budget,
                   'seeds': args.seeds, 'num_envs': args.num_envs,
                   'updated': time.time()}, f, indent=2)

    if args.mode == 'comparison':
        report_comparison(args)
    else:
        report_width(args)


if __name__ == '__main__':
    main()
