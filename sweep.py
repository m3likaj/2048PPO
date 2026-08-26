"""Encoder sweep runner: attention vs MLP, held at identical training conditions.

Runs every (encoder, seed) arm at the SAME budget, num_envs, scaffolding,
reward and hyperparameters -- the encoder and the seed are the only things
that vary. Designed for Kaggle: each arm gets a stable run directory name, so
the sweep is fully resumable. Re-running this script skips finished arms,
resumes partial ones, and starts the next one, until the session's time
budget runs out.

Typical use (Kaggle, one cell per session):
    python sweep.py --budget 1e9 --seeds 42 43 --session-hours 11

Then, once arms have finished:
    python sweep.py --report

Metrics: the comparison is judged on EARNED milestone rates (capability the
agent built itself), never on episode_return -- see README.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ARMS = ['mlp', 'attention', 'mlp_matched']


def run_name(encoder, seed, budget):
    return f'sweep_{encoder}_seed{seed}_b{budget:.0e}'.replace('+', '')


def run_dir(data_dir, encoder, seed, budget):
    return os.path.join(data_dir, run_name(encoder, seed, budget))


def is_done(d):
    return os.path.exists(os.path.join(d, 'DONE'))


def latest_ckpt(d):
    c = os.path.join(d, 'latest.pt')
    return c if os.path.exists(c) else None


def seed_from_prior_outputs(data_dir):
    """Kaggle: a committed run starts with an empty /kaggle/working. Copy any
    prior sweep runs out of the read-only /kaggle/input mounts so arms resume
    instead of restarting."""
    if not os.path.isdir('/kaggle/input'):
        return
    import shutil
    for src in glob.glob('/kaggle/input/**/sweep_*', recursive=True):
        if not os.path.isdir(src):
            continue
        dst = os.path.join(data_dir, os.path.basename(src))
        if os.path.exists(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(src, '*')):
            if os.path.isfile(f):
                shutil.copy2(f, dst)
        print(f'  seeded {os.path.basename(src)} from prior output')


def launch(args, encoder, seed, hours_left):
    d = run_dir(args.data_dir, encoder, seed, args.budget)
    name = os.path.basename(d)
    cmd = [sys.executable, 'train.py',
           '--encoder', encoder,
           '--seed', str(seed),
           '--num-envs', str(args.num_envs),
           '--total-timesteps', str(int(args.budget)),
           '--data-dir', args.data_dir,
           '--run-name', name,
           '--max-hours', f'{hours_left:.3f}']
    if encoder == 'attention':
        cmd += ['--d-model', str(args.d_model),
                '--n-layers', str(args.n_layers),
                '--n-heads', str(args.n_heads)]
    ck = latest_ckpt(d)
    if ck:
        cmd += ['--resume', ck]
        print(f'\n=== RESUME {name} ===')
    else:
        print(f'\n=== START  {name} ===')
    print('   ', ' '.join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    return rc, (time.time() - t0) / 3600


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

MILESTONES = [(13, '8k'), (14, '16k'), (15, '32k'), (16, '65k')]


def summarize_run(d):
    """Cumulative earned/reached rates over a run's whole log.csv."""
    import csv
    path = os.path.join(d, 'log.csv')
    if not os.path.exists(path):
        return None
    keys = ['xn_norm', 'xn_scaf']
    for m, _ in MILESTONES:
        keys += [f'xe{m}_norm', f'xe{m}_scaf', f'xr{m}_norm', f'xr{m}_scaf']
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


def report(args):
    import statistics
    rows = {}
    for enc in ARMS:
        for seed in args.seeds:
            d = run_dir(args.data_dir, enc, seed, args.budget)
            s = summarize_run(d)
            if s:
                rows.setdefault(enc, []).append((seed, s))

    if not rows:
        print('No sweep runs found in', args.data_dir)
        return

    print(f'\nEncoder sweep @ budget {args.budget:.2e} steps, '
          f'num_envs={args.num_envs}')
    print('Judged on EARNED milestone rates (capability built by the agent).\n')
    hdr = (f"{'arm':13s} {'seed':>5s} {'epoch':>7s} {'eps':>10s} "
           f"{'earn8k%':>9s} {'earn16k%':>9s} {'earn32k%':>9s} "
           f"{'16k n':>8s} {'done':>5s}")
    print(hdr); print('-' * len(hdr))
    for enc in ARMS:
        for seed, s in sorted(rows.get(enc, [])):
            print(f"{enc:13s} {seed:5d} {s['epoch']:7d} {s['episodes']:10,.0f} "
                  f"{s['earned_8k']:9.3f} {s['earned_16k']:9.3f} "
                  f"{s['earned_32k']:9.3f} {s['earned_16k_n']:8,.0f} "
                  f"{'Y' if s['done'] else '-':>5s}")

    print('\nPer-arm mean +/- stdev across seeds (complete runs only):')
    base = None
    for enc in ARMS:
        done = [s for _, s in rows.get(enc, []) if s['done']]
        if not done:
            print(f'  {enc:13s} (no completed runs yet)')
            continue
        line = f'  {enc:13s} n_seeds={len(done)}'
        for _, label in MILESTONES[:3]:
            vals = [s[f'earned_{label}'] for s in done]
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else float('nan')
            line += f' | earned_{label} {mu:6.3f}+/-{sd:5.3f}'
            if enc == 'mlp' and label == '16k':
                base = mu
        print(line)
        if base and enc != 'mlp':
            mu = statistics.mean([s['earned_16k'] for s in done])
            print(f'{"":15s} vs mlp on earned_16k: {100*(mu-base)/base:+.1f}%')

    failed = [os.path.basename(d) for d in
              glob.glob(os.path.join(args.data_dir, 'sweep_*'))
              if os.path.exists(os.path.join(d, 'FAILED'))
              and not os.path.exists(os.path.join(d, 'DONE'))]
    if failed:
        print('\n!!! FAILED arms (see their FAILED file): '
              + ', '.join(failed))

    if len(args.seeds) < 2:
        print('\nNOTE: one seed per arm cannot separate a real effect from '
              'seed noise (GPU RNG is not reproducible -- deviation #2). '
              'Add seeds before drawing conclusions.')
    print('\nNext: run eval.py (scaffolding off) on each arm\'s final '
          'checkpoint for unassisted milestone rates.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=float, default=1e9,
                   help='total_timesteps per arm (a COMPLETE cosine schedule '
                        'is shaped to this, so short runs are still valid).')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--arms', type=str, nargs='+', default=ARMS, choices=ARMS)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--data-dir', type=str, default='experiments_sweep')
    p.add_argument('--session-hours', type=float, default=11.0,
                   help='Wall-clock budget for THIS invocation (Kaggle kills '
                        'sessions at 12h; leave headroom).')
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=1)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--report', action='store_true', help='Only print results.')
    args = p.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    if args.report:
        report(args)
        return

    seed_from_prior_outputs(args.data_dir)

    # Seed-major order: finish one seed across all arms before starting the
    # next, so an interrupted sweep still yields a complete comparison.
    queue = [(enc, sd) for sd in args.seeds for enc in args.arms]
    started = time.time()
    for enc, sd in queue:
        d = run_dir(args.data_dir, enc, sd, args.budget)
        if is_done(d):
            print(f'skip (done): {os.path.basename(d)}')
            continue
        left = args.session_hours - (time.time() - started) / 3600
        if left < 0.2:
            print(f'\nSession budget exhausted; {os.path.basename(d)} not '
                  f'started. Re-run this cell next session to continue.')
            break
        rc, used = launch(args, enc, sd, left)
        print(f'--- {os.path.basename(d)}: rc={rc}, {used:.2f}h used')
        if rc != 0:
            # Record and CONTINUE: a systematic failure (e.g. attention OOM)
            # fails fast on every arm and wastes nothing, while a transient
            # one shouldn't cost you the rest of the sweep. Failures are
            # surfaced loudly in the report.
            with open(os.path.join(d, 'FAILED'), 'a') as f:
                f.write(f'rc={rc} after {used:.2f}h at {time.ctime()}\n')
            hint = ''
            if rc == -9:
                hint = ('  (killed -- likely out of memory. Attention needs '
                        '~2GB more than the MLP at the 32,768-row minibatch; '
                        'try a smaller --d-model.)')
            print(f'!!! {os.path.basename(d)} FAILED rc={rc}{hint}')

    with open(os.path.join(args.data_dir, 'sweep_state.json'), 'w') as f:
        json.dump({'budget': args.budget, 'seeds': args.seeds,
                   'arms': args.arms, 'num_envs': args.num_envs,
                   'updated': time.time()}, f, indent=2)
    report(args)


if __name__ == '__main__':
    main()
