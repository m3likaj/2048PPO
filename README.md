# simple-2048 RL — faithful PyTorch recreation

A single-GPU PyTorch recreation of **Kyoung Whan Choe (kywch)**'s
*"Less is More: 2048 agents"* setup, ported line-by-line from his
`simple-2048` branch of PufferLib so it runs on **Kaggle** or an **HPC
cluster** with nothing but `torch`, `numpy`, and `heavyball`.

- Blog posts: [Less is More: 2048 agents (2026-01)](https://kywch.github.io/blog/2026/01/less-is-more-2048-agents/) and [Curriculum learning: 2048 & Tetris (2025-12)](https://kywch.github.io/blog/2025/12/curriculum-learning-2048-tetris/)
- Original source: <https://github.com/kywch/PufferLib/tree/simple-2048> (MIT License, Copyright (c) 2022 PufferAI). The license and the original `g2048.h` / `g2048.ini` are included verbatim in `reference/` so you can diff every constant yourself.
- Credits: Kyoung Whan Choe (environment, reward design, curriculum, training recipe), Joseph Suarez / PufferAI (PufferLib trainer, LSTM wrapper, vtrace kernel), David Rubinstein (value+position embedding network idea).

The reference result to beat (original run `1v5kls7l`, embeddings + new
reward): **84.88 %** of episodes reach 32768, **33.96 %** reach 65536,
max-tile average ≈ 40 981, with a ~3.47 M-parameter network.

## Files

| File | Recreates | Original source |
|---|---|---|
| `g2048_env.py` | 2048 environment, rewards, scaffolding curriculum, logging | `pufferlib/ocean/g2048/g2048.h`, `g2048.py` |
| `g2048_policy.py` | embedding policy + LSTM wrapper + sampling | `pufferlib/ocean/torch.py::G2048`, `pufferlib/models.py::LSTMWrapper`, `pufferlib/pytorch.py` |
| `puffer_advantage.py` | vtrace-GAE advantage kernel | `pufferlib/extensions/pufferlib.cpp::puff_advantage_row` |
| `train.py` | PuffeRL collection + PPO training loop | `pufferlib/pufferl.py` |
| `eval.py` | evaluation protocol & printout | `pufferlib/ocean/g2048/eval.py` |
| `config.py` | every hyperparameter, with per-value source comments | `pufferlib/config/ocean/g2048.ini` + `config/default.ini` |
| `test_exactness.py` | 14 tests proving the port matches the C semantics | — |
| `reference/` | verbatim `g2048.h`, `g2048.ini`, `LICENSE` from the branch | — |
| `plot_training.py` | milestone (earned/scaffolded), loss, EV, and run plots | — |
| `kaggle_2048.ipynb` | ready-to-run Kaggle notebook | — |
| `run_hpc.sbatch` | single-GPU SLURM job script | — |

## Every variable, verbatim

All values live in `config.py` with a comment naming the ini file each one
comes from. Highlights (nothing altered):

- Env: `scaffolding_ratio=0.67`, merge reward `0.05 + pow15[exp-6]*0.03` for
  tiles ≥128, invalid move `-0.05`, game over `-1.0`, `BASE_MAX_TICKS=1000`
  with the dynamic `max(1000*max(1, lifetime-8), score/4)` limit, 10 % chance
  a spawned tile is a 4, and the exact two-branch scaffolding curriculum
  keyed on `lifetime_max_tile` (< 14: one tile `max(12+U{0..4}, lifetime)`;
  ≥ 14: the four base-14/15 cases).
- Policy: `Embedding(18,3)` value + `Embedding(16,3)` position embeddings
  (`embed_dim = ceil(33**0.25) = 3`), encoder 48→1024→512→512 with GELU,
  decoder/value heads with `std=0.01` / `std=1.0` output init,
  `LSTM(512,512)` wrapper — 3 466 859 parameters, and the wrapper's
  init-order quirk (policy params re-orthogonalized before the LSTM exists)
  is preserved on purpose.
- Training: `total_timesteps=6 767 676 767`, 16 384 agents ×
  `bptt_horizon=64` → batch 1 048 576, minibatch 32 768 (32 minibatches,
  `update_epochs=1`), `lr=0.000467` cosine-annealed to 15 %,
  `gamma=0.99567`, `gae_lambda=0.67`, `clip_coef=0.067`,
  `vf_clip_coef=0.167`, `vf_coef=2.0`, `ent_coef=0.0267`,
  `max_grad_norm=0.5`, vtrace `rho=1.1` / `c=2.0`, priority sampling
  `alpha=0.8` / `beta0=0.1`, heavyball **ForeachMuon**
  `(betas=(0.99, 0.9999), eps=1e-4, heavyball_momentum=True)`, seed 42.
- Trainer mechanics: rewards clamped to [-1,1] at collection; reward/done
  stored with the *following* observation (so vtrace reads `rewards[t+1]`);
  LSTM state zeroed each collection phase and `None` (zeros) per training
  minibatch; advantages recomputed from the full buffer before **every**
  minibatch with the live `ratio`/`values`; `values[idx]` overwritten after
  each minibatch; `advantages[:, -1]` stays 0.

`python test_exactness.py` proves all of the above: all 104 976 possible
rows are checked against an independently written 2048 implementation, the
vectorized move path is checked against a direct transliteration of the C
`move()`, and the advantage kernel is checked against a scalar
transliteration of `puff_advantage_row`.

## Quick start

```bash
pip install -r requirements.txt
python test_exactness.py          # ~2 min, all 14 must pass

# Full original run (needs a big GPU and days of wall clock):
python train.py

# Kaggle-sized run (T4/P100), resumable across sessions:
python train.py --num-envs 4096 --total-timesteps 1e9 \
    --data-dir /kaggle/working/experiments
python train.py --resume /kaggle/working/experiments/<run>/latest.pt

# Evaluate a checkpoint (original protocol: scaffolding off):
python eval.py --checkpoint experiments/<run>/latest.pt --min-episodes 5000
```

Wall-clock expectations are hardware-dependent; as rough orientation, the
full 6.77 B-step schedule is a multi-day job on a Kaggle T4 but fits
comfortably in a day-class job on a modern data-center GPU. `--num-envs`
and `--total-timesteps` scale the run down without touching any other
hyperparameter (batch size auto-resolves to `num_envs × 64` exactly as
PuffeRL does). Kaggle sessions are time-limited, so train in chunks with
`--resume`; checkpoints land in `--data-dir` every `checkpoint_interval`
(200) epochs and in `latest.pt` (full state).

## Earned-tile tracking, reports, and plots (addition)

The original `add_log` **skips scaffolding episodes**, so it cannot tell you
whether a milestone tile was built by the agent or handed to it. This port
adds a logging-only sidecar that fixes that without touching training:

- **Earned rule.** `earned_max_tile` is the largest exponent **created by a
  merge** during the episode. Any merge result counts as earned — including
  when one parent was a scaffolded placement (earned 8192 + scaffolded 8192
  → the 16384 is earned). Only tiles that exist purely because scaffolding
  placed them are un-earned. Implemented via a merge-max column in the row
  table: zero RNG consumption, zero reward/observation change, and the
  original log stream (`perf`, `score`, `reached_*`, lifetime updates) is
  byte-identical.
- **Extended log.** Every finished episode (normal *and* scaffold) is
  tallied separately: counts, per-milestone `reached` (max tile ≥ m, the
  original definition) and `earned` (earned max ≥ m) for
  8192/16384/32768/65536/131072, plus max-tile, earned-max-tile, episode
  length, and merge-score sums per class. Raw sums land in `log.csv` as
  `x/...` columns.
- **`report.txt`** — ~20 evenly spaced human-readable progress blocks per
  run (`total_epochs // 20`), each with timestamp, epoch/step/elapsed,
  cumulative episode counts (normal/scaffold), merge score, EV, average max
  tile, average *earned* max tile, and per-milestone earned / reached /
  scaffold-only percentages **with raw counts** so a 0.002 % rate is never
  rounded to zero. Counters persist through `--resume`.
- **`plot_training.py`** — `python plot_training.py --csv <run>/log.csv`
  (also auto-runs at training end) produces per-milestone
  earned-vs-reached-vs-scaffold-only curves (with the original
  normal-episodes-only reached rate dashed for comparability), loss curves,
  explained variance, average max tile (log₂ scale, earned vs raw), merge
  score, episode length by class, learning rate, and throughput — the full
  set you'd want for a write-up.

Definition note: "reached" keeps the original `max_tile` semantics
(recomputed on valid moves), so an episode terminated purely by invalid
moves logs `max_tile = 0` exactly as the C code would.

## Deviations from the original (complete list)

Everything not listed here is a faithful recreation. These are the only
differences, all forced by the port target or disclosed conveniences:

1. **Substrate.** The C environment (`g2048.h`) is replaced by a
   PyTorch-vectorized environment driven by a precomputed table of all
   18⁴ = 104 976 row transitions, generated by a direct transliteration of
   the C `slide_and_merge`. Mechanics, rewards, curriculum, tick limits and
   logging are bit-identical (proven by tests); only the execution substrate
   differs.
2. **RNG streams.** C `rand()` sequences cannot be reproduced in torch; the
   port uses a seeded `torch.Generator` with identical *distributions*
   (uniform cell choice, 10 % fours, curriculum draws). Individual episodes
   therefore differ; aggregate behavior does not. (The author's runs were
   robust across seeds.)
3. **Vectorization topology.** Original: 4 worker processes × 4096 C envs =
   16 384 agents. Port: one process, 16 384 GPU-vectorized envs. Agent
   count, batch math, and the trainer's view of the data are identical.
4. **Optimizer packaging.** `heavyball` is pinned to **2.1.4** because 3.x
   removed `ForeachMuon`. The optimizer call itself is verbatim. A
   `--optimizer adam` fallback flag exists but the default is the original
   muon.
5. **`--resume` (addition).** The original only supports policy-only
   `load_model_path` (also provided here). Because Kaggle kills sessions
   after ~9–12 h, `latest.pt` stores full state (optimizer, scheduler,
   epoch, step) and `--resume` restores it.
6. **AMP context.** The original wraps forward passes in
   `torch.amp.autocast(dtype=float32)` — a no-op at float32 precision. The
   port simply runs float32 without the context (numerically identical).
7. **Seeding.** The original commented out `torch.manual_seed`/np seeding;
   the port enables them for reproducibility of the *port's* runs. (Given
   deviation 2, this cannot and does not chase C-run bit-equality.)
8. **Unreachable-state definition.** Merging two 131072 tiles (exp 17→18)
   would index past the C `pow15_table[12]` (undefined behavior). That state
   is unreachable in play, but the exhaustive row table must define it, so
   the table is extended with its own generating formula, `12^1.5 = 41.57`.
9. **CLI conveniences.** `--num-envs`, `--total-timesteps`,
   `--scaffolding-ratio`, `--seed`, `--device`, `--data-dir`,
   `--checkpoint-interval`, `--wandb`, `--tag` — all defaulting to the
   original values; passing nothing reproduces the original configuration.
10. **Logging.** The curses dashboard is replaced by plain prints +
   `log.csv` (+ optional wandb). Logged quantities and their definitions
   (per-episode means over 128-step flush windows, weighted by `n`) match
   the original.
11. **Earned-tile tracking (user-requested addition).** Merge-provenance
   tracker, all-episode extended log split normal/scaffold, `report.txt`
   (~20 progress blocks), and `plot_training.py`. Logging-only: no RNG,
   reward, observation, or optimizer-path change; the original log stream
   is unchanged (verified by tests).

## Plugging in a transformer later

The clean swap point is `G2048Policy.encode_observations` /
`G2048Policy.encoder` in `g2048_policy.py`. The contract: take the
`[B, 16]` uint8 board, return a `[B, 512]` encoding; `value_embed` +
`pos_embed` already give you 16 tokens of dim 3 to feed a transformer over
tile positions — replace the flatten+MLP with your encoder and leave
`decode_actions`, `LSTMWrapper`, the trainer, and every hyperparameter
untouched for a controlled comparison. (If you later want the transformer
to replace the LSTM over *time* as well, that means swapping `LSTMWrapper`
and the `forward`/`forward_eval` state handling — a bigger change; do the
encoder swap first.)

## License

MIT, same as the original (see `reference/LICENSE`). Please cite kywch's
blog posts and PufferLib if you build on this.
