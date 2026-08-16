"""Exactness tests for the simple-2048 port.

Verifies the PyTorch port against the semantics of the original C sources
(g2048.h, pufferlib.cpp) from https://github.com/kywch/PufferLib/tree/simple-2048
(MIT License, Copyright (c) 2022 PufferAI).

Run:  python test_exactness.py            (plain runner, no pytest needed)
      python -m pytest test_exactness.py  (also works)
"""

import math

import numpy as np
import torch

from config import CONFIG, resolve
from g2048_env import (BASE_MAX_TICKS, GAME_OVER_PENALTY, INVALID_MOVE_PENALTY,
                       MERGE_BASE_REWARD, MERGE_REWARD_SCALE, POW15_TABLE,
                       G2048TorchEnv, build_left_tables, slide_and_merge_row)
from g2048_policy import make_policy, sample_logits
from puffer_advantage import compute_puff_advantage

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# 1. Row mechanics: independent implementation cross-check (all 18^4 rows)
# ---------------------------------------------------------------------------

def independent_merge(row):
    """A deliberately different-style 2048 row implementation.

    Written from the rules, not from the C code: compact non-zeros, then
    merge equal neighbors left-to-right (each tile merges at most once),
    then pad. Used as an independent oracle for the transliteration.
    (Uses the same 13-entry pow15 table; entry 12 covers the play-unreachable
    exp-18 merge that would be out-of-bounds in the C table.)
    """
    vals = [v for v in row if v != 0]
    out, reward, score = [], 0.0, 0
    merged_max = 0
    i = 0
    while i < len(vals):
        if i + 1 < len(vals) and vals[i] == vals[i + 1]:
            new = vals[i] + 1
            out.append(new)
            merged_max = max(merged_max, new)
            score += 2 ** new
            if new <= 6:
                reward += MERGE_BASE_REWARD
            else:
                reward += MERGE_BASE_REWARD + POW15_TABLE[new - 6] * MERGE_REWARD_SCALE
            i += 2
        else:
            out.append(vals[i])
            i += 1
    out += [0] * (4 - len(out))
    moved = out != list(row)
    return out, reward, score, moved, merged_max


def test_all_rows_vs_independent_oracle():
    rows, rewards, scores, moved, merge_max = build_left_tables()
    n = 18 ** 4
    for idx in range(n):
        r = idx
        c3 = r % 18; r //= 18
        c2 = r % 18; r //= 18
        c1 = r % 18; r //= 18
        row = [r, c1, c2, c3]
        o_row, o_rew, o_sc, o_mv, o_mm = independent_merge(row)
        assert list(rows[idx]) == o_row, (row, list(rows[idx]), o_row)
        assert abs(float(rewards[idx]) - np.float32(o_rew)) < 1e-6, (row, rewards[idx], o_rew)
        assert int(scores[idx]) == o_sc, (row, scores[idx], o_sc)
        assert bool(moved[idx]) == o_mv, (row, moved[idx], o_mv)
        assert int(merge_max[idx]) == o_mm, (row, merge_max[idx], o_mm)
    print(f"  all {n} rows match the independent oracle "
          "(incl. merged-max for the earned tracker)")


def test_known_rows():
    cases = [
        # (input, expected_row, expected_reward, expected_score, moved)
        ([1, 1, 2, 2], [2, 3, 0, 0], 2 * MERGE_BASE_REWARD, 4 + 8, True),
        ([1, 1, 1, 1], [2, 2, 0, 0], 2 * MERGE_BASE_REWARD, 8, True),
        ([0, 1, 0, 1], [2, 0, 0, 0], MERGE_BASE_REWARD, 4, True),
        ([1, 2, 3, 4], [1, 2, 3, 4], 0.0, 0, False),
        ([1, 2, 1, 0], [1, 2, 1, 0], 0.0, 0, False),
        ([0, 0, 0, 1], [1, 0, 0, 0], 0.0, 0, True),
        ([2, 2, 3, 0], [3, 3, 0, 0], MERGE_BASE_REWARD, 8, True),
        # 64+64 -> 128 (exp 7): first pow15 bonus tier
        ([6, 6, 0, 0], [7, 0, 0, 0],
         MERGE_BASE_REWARD + POW15_TABLE[1] * MERGE_REWARD_SCALE, 128, True),
        # 512+512 -> 1024 (exp 10): 0.05 + 8.0*0.03 = 0.29
        ([9, 9, 0, 0], [10, 0, 0, 0], 0.29, 1024, True),
        # 32768+32768 -> 65536 (exp 16): 0.05 + 31.62*0.03 = 0.9986
        ([15, 15, 0, 0], [16, 0, 0, 0], 0.05 + 31.62 * 0.03, 65536, True),
    ]
    for row, e_row, e_rew, e_sc, e_mv in cases:
        o_row, o_rew, o_sc, o_mv = slide_and_merge_row(row)
        assert o_row == e_row, (row, o_row, e_row)
        assert abs(o_rew - e_rew) < 1e-6, (row, o_rew, e_rew)
        assert o_sc == e_sc and o_mv == e_mv, (row,)
    print("  hand-verified merge/reward/score cases pass "
          "(incl. 0.08 @128, 0.29 @1024, 0.9986 @65536)")


# ---------------------------------------------------------------------------
# 2. Orientation machinery vs a direct transliteration of C move()
# ---------------------------------------------------------------------------

def reference_move(board, direction):
    """Direct transliteration of move() from g2048.h on a 4x4 python grid."""
    g = [row[:] for row in board]
    moved, reward, score = False, 0.0, 0
    merged = []
    for line in range(4):
        if direction == 0:      # UP:    temp[i] = grid[i][col]
            temp = [g[i][line] for i in range(4)]
        elif direction == 1:    # DOWN:  temp[i] = grid[SIZE-1-i][col]
            temp = [g[3 - i][line] for i in range(4)]
        elif direction == 2:    # LEFT:  temp[i] = grid[row][i]
            temp = [g[line][i] for i in range(4)]
        else:                   # RIGHT: temp[i] = grid[row][SIZE-1-i]
            temp = [g[line][3 - i] for i in range(4)]
        new, rew, sc, mv = slide_and_merge_row(temp, merged)
        reward += rew
        score += sc
        moved |= mv
        for i in range(4):
            if direction == 0:
                g[i][line] = new[i]
            elif direction == 1:
                g[3 - i][line] = new[i]
            elif direction == 2:
                g[line][i] = new[i]
            else:
                g[line][3 - i] = new[i]
    return g, reward, score, moved, (max(merged) if merged else 0)


def test_env_moves_match_reference():
    rng = np.random.default_rng(123)
    n_boards = 200
    for direction in range(4):
        boards = rng.integers(0, 12, size=(n_boards, 4, 4)).astype(np.uint8)
        # sprinkle zeros so slides occur
        mask = rng.random((n_boards, 4, 4)) < 0.4
        boards[mask] = 0

        env = G2048TorchEnv(num_envs=n_boards, scaffolding_ratio=0.0, seed=7)
        env.boards = torch.as_tensor(boards.copy())
        actions = torch.full((n_boards,), direction)
        moved, reward, score, merge_max = env._apply_moves(actions)

        for b in range(n_boards):
            ref_g, ref_rew, ref_sc, ref_mv, ref_mm = reference_move(
                boards[b].tolist(), direction)
            assert env.boards[b].tolist() == ref_g, (direction, b)
            assert abs(float(reward[b]) - np.float32(ref_rew)) < 1e-5, (direction, b)
            assert int(score[b]) == ref_sc, (direction, b)
            assert bool(moved[b]) == ref_mv, (direction, b)
            assert int(merge_max[b]) == ref_mm, (direction, b)
    print("  vectorized moves match the C move() transliteration "
          "for all 4 directions (200 random boards each, incl. merged-max)")


# ---------------------------------------------------------------------------
# 3. c_step semantics
# ---------------------------------------------------------------------------

def _fixed_env(boards, **kw):
    env = G2048TorchEnv(num_envs=boards.shape[0], scaffolding_ratio=0.0,
                        seed=kw.pop('seed', 11), **kw)
    env.reset()
    env.boards = torch.as_tensor(boards)
    env.max_tile = env.boards.amax(dim=(1, 2))
    return env


def test_invalid_move():
    board = np.array([[[1, 2, 3, 4]] * 4], dtype=np.uint8)  # LEFT is a no-op
    env = _fixed_env(board.copy())
    tick0 = env.tick.clone()
    met0 = env.max_episode_ticks.clone()
    obs, r, term, trunc, _ = env.step(torch.tensor([2]))  # LEFT
    assert float(r[0]) == np.float32(INVALID_MOVE_PENALTY)
    assert env.boards[0].numpy().tolist() == board[0].tolist()   # obs unchanged
    assert int(env.tick[0]) == int(tick0[0]) + 1                 # tick++ always
    assert int(env.moves_made[0]) == 0
    assert int(env.max_episode_ticks[0]) == int(met0[0])         # no limit update
    assert int(term[0]) == 0
    print("  invalid move: -0.05, board/limit unchanged, tick++, no spawn")


def test_valid_move_spawn_score_and_limit():
    board = np.zeros((1, 4, 4), dtype=np.uint8)
    board[0, 0] = [1, 1, 2, 2]        # LEFT -> [2,3,0,0], reward 0.10, score 12
    env = _fixed_env(board.copy())
    env.lifetime_max_tile[:] = 12     # tick_multiplier = 4
    obs, r, term, trunc, _ = env.step(torch.tensor([2]))
    assert abs(float(r[0]) - 0.10) < 1e-6
    assert int(env.score[0]) == 12
    assert env.boards[0, 0].tolist()[:2] == [2, 3]
    # exactly one spawned tile (exp 1 or 2) somewhere on the rest of the board
    rest = env.boards[0].flatten().tolist()
    spawned = [v for i, v in enumerate(rest) if v != 0 and i not in (0, 1)]
    assert len(spawned) == 1 and spawned[0] in (1, 2)
    # max_tile recomputed from post-move pre-spawn grid
    assert int(env.max_tile[0]) == 3
    # dynamic limit: max(1000 * max(1, 12-8), score//4) = max(4000, 3) = 4000
    assert int(env.max_episode_ticks[0]) == 4000
    print("  valid move: table reward/score applied, 1 tile spawned, "
          "max_tile recomputed, dynamic tick limit = max(1000*mult, score//4)")


def test_game_over_and_autoreset():
    stuck = np.array([[[1, 2, 1, 2],
                       [2, 1, 2, 1],
                       [1, 2, 1, 2],
                       [2, 1, 2, 1]]], dtype=np.uint8)
    env = _fixed_env(stuck.copy())
    obs, r, term, trunc, _ = env.step(torch.tensor([0]))
    # invalid (-0.05) then game-over penalty added on top (-1.0)
    assert abs(float(r[0]) - (INVALID_MOVE_PENALTY + GAME_OVER_PENALTY)) < 1e-6
    assert int(term[0]) == 1 and int(trunc[0]) == 0
    # auto-reset: fresh board with exactly two starting tiles (exp 1 or 2)
    vals = [v for v in env.boards[0].flatten().tolist() if v != 0]
    assert len(vals) == 2 and all(v in (1, 2) for v in vals)
    assert int(env.tick[0]) == 0 and int(env.score[0]) == 0
    assert int(env.max_episode_ticks[0]) == BASE_MAX_TICKS
    print("  stuck board: reward -1.05 (invalid + game-over), terminal, "
          "auto-reset to 2 fresh tiles")


def test_tick_limit_termination():
    board = np.array([[[1, 2, 3, 4]] * 4], dtype=np.uint8)
    env = _fixed_env(board.copy())
    env.tick[:] = BASE_MAX_TICKS - 1   # next (invalid) step hits the limit
    obs, r, term, trunc, _ = env.step(torch.tensor([2]))
    assert int(term[0]) == 1
    assert abs(float(r[0]) - INVALID_MOVE_PENALTY) < 1e-6  # no game-over penalty
    print("  tick >= max_episode_ticks terminates without the -1.0 penalty")


def test_spawn_distribution():
    env = G2048TorchEnv(num_envs=1, scaffolding_ratio=0.0, seed=3)
    tiles = env._get_new_tile(200_000)
    frac4 = float((tiles == 2).float().mean())
    assert abs(frac4 - 0.10) < 0.01, frac4
    print(f"  spawn distribution: P(tile=4) = {frac4:.4f} (expected 0.10)")


# ---------------------------------------------------------------------------
# 4. Scaffolding curriculum + logging rules
# ---------------------------------------------------------------------------

def test_scaffolding_branch_low():
    env = G2048TorchEnv(num_envs=4096, scaffolding_ratio=1.0, seed=5)
    env.reset()  # lifetime = 0 -> branch 1
    assert bool(env.is_scaffolding.all())
    counts = (env.boards.view(4096, 16) != 0).sum(dim=1)
    assert bool((counts == 1).all())
    vals = env.boards.view(4096, 16).amax(dim=1)
    assert int(vals.min()) >= 12 and int(vals.max()) <= 16
    seen = set(vals.unique().tolist())
    assert seen == {12, 13, 14, 15, 16}, seen
    print("  scaffolding lifetime<14: one tile, exp = 12 + U{0..4}")


def _scaffold_multisets(lifetime, seed):
    env = G2048TorchEnv(num_envs=4096, scaffolding_ratio=1.0, seed=seed)
    env.reset()
    env.lifetime_max_tile[:] = lifetime
    env._reset_envs(torch.arange(4096))
    out = set()
    for b in range(4096):
        vals = tuple(sorted(v for v in env.boards[b].flatten().tolist() if v != 0))
        out.add(vals)
    return out


def test_scaffolding_branch_high():
    # lifetime = 14 -> base 14: {[14], [15], [14,13], [15,14]}
    got = _scaffold_multisets(14, seed=6)
    assert got == {(14,), (15,), (13, 14), (14, 15)}, got
    # lifetime = 16 -> base 15: {[15], [16], [15,14], [16,15]}
    got = _scaffold_multisets(16, seed=7)
    assert got == {(15,), (16,), (14, 15), (15, 16)}, got
    print("  scaffolding lifetime>=14: exact 4-case curriculum for base 14/15")


def test_add_log_skips_scaffolding_and_updates_lifetime():
    # Scaffolding episodes: never logged, lifetime never updated by them.
    env = G2048TorchEnv(num_envs=8, scaffolding_ratio=1.0, seed=9,
                        log_interval=10**9)
    env.reset()
    env.tick[:] = 10**7  # force terminal via tick limit on next step
    env.step(torch.zeros(8, dtype=torch.long))
    logs = env.pop_logs()
    # Original stream stays empty for scaffolding episodes (no 'n' key);
    # only the extended sidecar sees them.
    assert logs is not None and 'n' not in logs
    assert logs['xstats']['xn_scaf'] == 8 and logs['xstats']['xn_norm'] == 0
    assert int(env.lifetime_max_tile.max()) == 0

    # Non-scaffolding episodes: logged, lifetime updated to episode max tile.
    env = G2048TorchEnv(num_envs=8, scaffolding_ratio=0.0, seed=9,
                        log_interval=10**9)
    env.reset()
    env.boards[:] = 0
    env.boards[:, 0, 0] = 14          # pretend we reached 16384
    env.max_tile[:] = 14
    env.tick[:] = 10**7
    env.step(torch.zeros(8, dtype=torch.long))
    logs = env.pop_logs()
    assert logs is not None and logs['n'] == 8
    assert int(env.lifetime_max_tile.min()) == 14
    assert abs(logs['score'] - 2 ** 14) < 1e-6          # score logged as 2^max
    assert abs(logs['reached_16384'] - 1.0) < 1e-9
    assert abs(logs['reached_32768'] - 0.0) < 1e-9
    perf = min(0.8 * 2 ** 14 / 65536.0, 1.0)
    assert abs(logs['perf'] - perf) < 1e-9
    print("  add_log: scaffolding episodes skipped; lifetime/perf/score/"
          "reached_* logged exactly for real episodes")


def test_earned_tracking_rule():
    """User rule: ANY merge result is earned, even with a scaffolded parent.
    e.g. earned 8192 + scaffolded 8192 -> the 16384 is earned."""
    board = np.zeros((1, 4, 4), dtype=np.uint8)
    board[0, 0] = [13, 13, 0, 0]      # two 8192s (one imagined scaffolded)
    env = _fixed_env(board.copy())
    env.is_scaffolding[:] = True       # provenance is irrelevant to the rule
    env.step(torch.tensor([2]))        # LEFT: 8192+8192 -> 16384
    assert int(env.earned_max_tile[0]) == 14
    # A later smaller merge must not lower it
    env.boards[0] = 0
    env.boards[0, 1] = torch.tensor([1, 1, 0, 0], dtype=torch.uint8)
    env.step(torch.tensor([2]))
    assert int(env.earned_max_tile[0]) == 14
    # Invalid move leaves it untouched
    env.boards[0] = torch.tensor([[1, 2, 3, 4]] * 4, dtype=torch.uint8)
    env.step(torch.tensor([2]))
    assert int(env.earned_max_tile[0]) == 14
    # Reset clears it
    env._reset_envs(torch.tensor([0]))
    assert int(env.earned_max_tile[0]) == 0
    print("  earned rule: merge => earned (scaffolded parent ok); "
          "monotone; invalid move no-op; cleared on reset")


def test_xlog_split_and_counts():
    """Extended log: ALL episodes tallied, split normal/scaffold, with
    earned vs reached milestones; original log stream unaffected."""
    env = G2048TorchEnv(num_envs=2, scaffolding_ratio=0.0, seed=21,
                        log_interval=10**9)
    env.reset()
    # env 0: scaffold-style episode, max tile 15 all from placement (earned 0)
    # env 1: normal episode that EARNED a 14
    env.is_scaffolding[:] = torch.tensor([True, False])
    env.max_tile[:] = torch.tensor([15, 14])
    env.earned_max_tile[:] = torch.tensor([0, 14])
    env.score[:] = torch.tensor([100, 200])
    # Stuck full board: the UP move is invalid (max_tile untouched) and
    # game-over terminates both episodes immediately.
    stuck = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5],
                          [3, 4, 5, 6], [4, 5, 6, 7]], dtype=torch.uint8)
    env.boards[0] = stuck
    env.boards[1] = stuck
    env.step(torch.zeros(2, dtype=torch.long))

    logs = env.pop_logs()
    # Original stream: only the normal episode (scaffold skipped, verbatim)
    assert logs['n'] == 1 and abs(logs['score'] - 2 ** 14) < 1e-6
    X = logs['xstats']
    assert X['xn_scaf'] == 1 and X['xn_norm'] == 1
    # scaffold episode: reached 32768 without earning anything
    assert X['xr15_scaf'] == 1 and X['xe15_scaf'] == 0
    assert X['xr13_scaf'] == 1 and X['xe13_scaf'] == 0
    # normal episode: earned == reached at 16384, nothing at 32768
    assert X['xr14_norm'] == 1 and X['xe14_norm'] == 1
    assert X['xr15_norm'] == 0 and X['xe15_norm'] == 0
    assert X['xsum_maxtile_scaf'] == 2 ** 15
    assert X['xsum_earnedmax_norm'] == 2 ** 14
    assert X['xsum_mergescore_norm'] == 200
    # accumulators cleared after pop
    assert env.pop_logs() is None
    print("  xlog: all episodes tallied, normal/scaffold split, "
          "earned vs reached separated; original stream untouched")



# ---------------------------------------------------------------------------
# 5. Policy: parameter count, forward paths, sampling helpers
# ---------------------------------------------------------------------------

def test_policy_params_and_paths():
    torch.manual_seed(42)
    policy = make_policy(CONFIG['hidden_size'], CONFIG['rnn_input_size'],
                         CONFIG['rnn_hidden_size'])
    n_params = sum(p.numel() for p in policy.parameters())
    # Exact architecture arithmetic:
    #   embeddings: 18*3 + 16*3
    #   encoder:    48*1024+1024 + 1024*512+512 + 512*512+512
    #   decoder:    512*512+512 + 512*4+4
    #   value:      512*512+512 + 512*1+1
    #   lstm:       4*(512*512)*2 + 4*512*2   (LSTMCell shares these tensors)
    expected = (18 * 3 + 16 * 3
                + 48 * 1024 + 1024 + 1024 * 512 + 512 + 512 * 512 + 512
                + 512 * 512 + 512 + 512 * 4 + 4
                + 512 * 512 + 512 + 512 * 1 + 1
                + 2 * 4 * 512 * 512 + 2 * 4 * 512)
    assert n_params == expected, (n_params, expected)

    # forward (TT=1) must equal forward_eval given the same LSTM state
    B = 5
    obs = torch.randint(0, 18, (B, 16), dtype=torch.uint8)
    h0 = torch.randn(B, 512)
    c0 = torch.randn(B, 512)
    s_eval = dict(lstm_h=h0.clone(), lstm_c=c0.clone())
    logits_e, values_e = policy.forward_eval(obs, s_eval)
    s_train = dict(lstm_h=h0.clone().unsqueeze(0), lstm_c=c0.clone().unsqueeze(0))
    logits_t, values_t = policy.forward(obs.unsqueeze(1), s_train)
    assert torch.allclose(logits_e, logits_t, atol=1e-5)
    assert torch.allclose(values_e.flatten(), values_t.flatten(), atol=1e-5)
    assert torch.allclose(s_eval['lstm_h'], s_train['lstm_h'][0], atol=1e-5)
    assert torch.allclose(s_eval['lstm_c'], s_train['lstm_c'][0], atol=1e-5)

    # sample_logits logprob/entropy vs torch.distributions.Categorical
    logits = torch.randn(B, 4)
    a, lp, ent = sample_logits(logits)
    dist = torch.distributions.Categorical(logits=logits)
    assert torch.allclose(lp, dist.log_prob(a), atol=1e-6)
    assert torch.allclose(ent, dist.entropy(), atol=1e-6)
    a2, lp2, _ = sample_logits(logits, action=a)
    assert torch.equal(a2.flatten().long(), a.flatten().long())
    assert torch.allclose(lp2.flatten(), lp.flatten(), atol=1e-6)
    print(f"  policy: {n_params:,} params (exact arithmetic match, "
          f"~{n_params/1e6:.2f}M); forward==forward_eval @TT=1; "
          "sample_logits matches Categorical")


# ---------------------------------------------------------------------------
# 6. Advantage kernel vs scalar C transliteration
# ---------------------------------------------------------------------------

def reference_advantage_row(values, rewards, dones, importance,
                            gamma, lam, rho_clip, c_clip):
    """Scalar transliteration of puff_advantage_row (pufferlib.cpp)."""
    horizon = len(values)
    adv = [0.0] * horizon
    lastpufferlam = 0.0
    for t in range(horizon - 2, -1, -1):
        nextnonterminal = 1.0 - dones[t + 1]
        rho_t = min(importance[t], rho_clip)
        c_t = min(importance[t], c_clip)
        delta = rho_t * (rewards[t + 1] + gamma * values[t + 1] * nextnonterminal
                         - values[t])
        lastpufferlam = delta + gamma * lam * c_t * lastpufferlam * nextnonterminal
        adv[t] = lastpufferlam
    return adv


def test_advantage_matches_c_kernel():
    rng = np.random.default_rng(0)
    S, H = 64, 64
    values = rng.normal(size=(S, H)).astype(np.float32)
    rewards = rng.normal(size=(S, H)).astype(np.float32)
    dones = (rng.random((S, H)) < 0.05).astype(np.float32)
    imp = np.exp(rng.normal(scale=0.3, size=(S, H))).astype(np.float32)

    adv = torch.zeros(S, H)
    compute_puff_advantage(torch.as_tensor(values), torch.as_tensor(rewards),
                           torch.as_tensor(dones), torch.as_tensor(imp), adv,
                           CONFIG['gamma'], CONFIG['gae_lambda'],
                           CONFIG['vtrace_rho_clip'], CONFIG['vtrace_c_clip'])
    for s in range(S):
        ref = reference_advantage_row(values[s], rewards[s], dones[s], imp[s],
                                      CONFIG['gamma'], CONFIG['gae_lambda'],
                                      CONFIG['vtrace_rho_clip'],
                                      CONFIG['vtrace_c_clip'])
        assert np.allclose(adv[s].numpy(), ref, atol=1e-4), s
        assert adv[s, -1] == 0.0
    print("  vectorized advantage == scalar puff_advantage_row transliteration "
          "(64x64 random data, advantages[:, -1] stays 0)")


# ---------------------------------------------------------------------------
# 7. Config resolution replicates PuffeRL batch math
# ---------------------------------------------------------------------------

def test_config_resolution():
    cfg = resolve(CONFIG)
    assert cfg['batch_size'] == 16384 * 64 == 1_048_576
    assert cfg['segments'] == 16384
    assert cfg['minibatch_size_eff'] == 32768
    assert cfg['accumulate_minibatches'] == 1
    assert cfg['total_minibatches'] == 32
    assert cfg['minibatch_segments'] == 512
    assert cfg['total_epochs'] == 6_767_676_767 // 1_048_576 == 6454
    lr, ratio = cfg['learning_rate'], cfg['min_lr_ratio']
    assert abs(lr - 0.000467) < 1e-12 and abs(ratio * lr - 0.15 * 0.000467) < 1e-12
    print("  batch math: 1,048,576 batch / 32,768 minibatch / 32 minibatches "
          "/ 16,384 segments / 6,454 epochs (matches PuffeRL resolution)")


# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_known_rows,
    test_all_rows_vs_independent_oracle,
    test_env_moves_match_reference,
    test_invalid_move,
    test_valid_move_spawn_score_and_limit,
    test_game_over_and_autoreset,
    test_tick_limit_termination,
    test_spawn_distribution,
    test_scaffolding_branch_low,
    test_scaffolding_branch_high,
    test_add_log_skips_scaffolding_and_updates_lifetime,
    test_earned_tracking_rule,
    test_xlog_split_and_counts,
    test_policy_params_and_paths,
    test_advantage_matches_c_kernel,
    test_config_resolution,
]

if __name__ == '__main__':
    for fn in ALL_TESTS:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\nAll {len(ALL_TESTS)} exactness tests passed.")
