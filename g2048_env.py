"""Vectorized PyTorch port of the simple-2048 environment.

This is a line-faithful port of pufferlib/ocean/g2048/g2048.h from
https://github.com/kywch/PufferLib/tree/simple-2048
(MIT License, Copyright (c) 2022 PufferAI; simple-2048 by Kyoung Whan Choe).

Every game rule, reward constant, curriculum branch, tick limit, and logging
rule below mirrors the C source. Comments cite the original function names.
The only substrate difference: the C scalar loop is replaced by a precomputed
row-transition table (generated with a direct transliteration of the C
`slide_and_merge`), so thousands of boards step in parallel on GPU/CPU.
RNG streams necessarily differ from C `rand()` (documented in README).

Observation: 16 uint8 cell exponents (0=empty, k means tile 2^k), Box(0, 18).
Actions: Discrete(4); 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (C: actions[0]+1).
"""

import numpy as np
import torch

# --- Constants from g2048.h ---
SIZE = 4
BASE_MAX_TICKS = 1000

MERGE_BASE_REWARD = 0.05
MERGE_REWARD_SCALE = 0.03
INVALID_MOVE_PENALTY = -0.05
GAME_OVER_PENALTY = -1.0

# Pow 1.5 lookup table for tiles 128+ (index = exponent - 6). Entries 0..11
# are verbatim from g2048.h (pow15_table[12], values = index^1.5 rounded).
# Entry 12 (= 12^1.5 = 41.57) is the formula's continuation, needed only for
# a merge of two 131072 tiles -> 262144 (exp 18). That state is unreachable
# in play (131072 is the 4x4 theoretical max; the C table would index out of
# bounds there), but our exhaustive row table must still define it.
POW15_TABLE = [
    0.0, 1.0, 2.83, 5.20, 8.0, 11.18, 14.70, 18.52, 22.63, 27.0, 31.62, 36.48,
    41.57,
]

NUM_TILE_VALUES = 18  # exponents 0..17 (empty .. 131072)


def slide_and_merge_row(row, merged_out=None):
    """Direct transliteration of slide_and_merge() from g2048.h.

    Takes a length-4 list of exponents; returns (new_row, reward,
    score_increase, moved). Slide direction is toward index 0 (LEFT).

    `merged_out`: optional list; every exponent CREATED by a merge in this
    row is appended to it. Pure bookkeeping for the earned-tile tracker —
    the game logic above/below this line is untouched C transliteration.
    """
    row = list(row)
    moved = False
    reward = 0.0
    score_increase = 0

    # Single pass: slide
    write_pos = 0
    for read_pos in range(SIZE):
        if row[read_pos] != 0:
            if write_pos != read_pos:
                row[write_pos] = row[read_pos]
                row[read_pos] = 0
                moved = True
            write_pos += 1

    # Merge pass
    i = 0
    while i < SIZE - 1:
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i] += 1
            if merged_out is not None:
                merged_out.append(row[i])
            # Tiles 2-64 (exp 1-6): base reward only
            # Tiles 128+ (exp 7+): base + pow1.5 scaled bonus
            if row[i] <= 6:
                reward += MERGE_BASE_REWARD
            else:
                reward += MERGE_BASE_REWARD + POW15_TABLE[row[i] - 6] * MERGE_REWARD_SCALE
            score_increase += 1 << row[i]
            # Shift remaining elements left
            for j in range(i + 1, SIZE - 1):
                row[j] = row[j + 1]
            row[SIZE - 1] = 0
            moved = True
        i += 1

    return row, reward, score_increase, moved


_POW = np.array([NUM_TILE_VALUES ** 3, NUM_TILE_VALUES ** 2, NUM_TILE_VALUES, 1],
                dtype=np.int64)


def build_left_tables():
    """Enumerate all 18^4 rows through slide_and_merge_row (LEFT direction).

    Returns (rows, rewards, scores, moved, merge_max) where merge_max[idx]
    is the largest exponent CREATED by a merge in that row (0 if none) —
    used only by the earned-tile tracker, never by game logic.
    """
    n = NUM_TILE_VALUES ** 4
    rows = np.zeros((n, 4), dtype=np.uint8)
    rewards = np.zeros(n, dtype=np.float32)
    scores = np.zeros(n, dtype=np.int32)
    moved = np.zeros(n, dtype=bool)
    merge_max = np.zeros(n, dtype=np.uint8)
    for idx in range(n):
        r = idx
        c3 = r % 18; r //= 18
        c2 = r % 18; r //= 18
        c1 = r % 18; r //= 18
        c0 = r
        merged = []
        new_row, rew, sc, mv = slide_and_merge_row([c0, c1, c2, c3], merged)
        rows[idx] = new_row
        rewards[idx] = np.float32(rew)
        scores[idx] = sc
        moved[idx] = mv
        merge_max[idx] = max(merged) if merged else 0
    return rows, rewards, scores, moved, merge_max


class G2048TorchEnv:
    """GPU/CPU vectorized 2048 matching g2048.h semantics.

    Mirrors the PufferEnv interface used in training: step(actions) returns
    (observations, rewards, terminals, truncations) and auto-resets done envs
    (c_step -> add_log -> c_reset). pop_logs() mirrors binding.vec_log().
    """

    def __init__(self, num_envs, scaffolding_ratio=0.0, seed=0,
                 device='cpu', log_interval=128):
        self.num_envs = num_envs
        self.scaffolding_ratio = float(scaffolding_ratio)
        self.device = torch.device(device)
        self.log_interval = log_interval
        self.single_observation_shape = (16,)
        self.num_actions = 4

        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        self.gen = g

        rows, rewards, scores, moved, merge_max = build_left_tables()
        d = self.device
        self.t_rows = torch.as_tensor(rows, device=d)             # [18^4, 4] uint8
        self.t_rewards = torch.as_tensor(rewards, device=d)       # float32
        self.t_scores = torch.as_tensor(scores, device=d)         # int32
        self.t_moved = torch.as_tensor(moved, device=d)           # bool
        self.t_merge_max = torch.as_tensor(merge_max, device=d)   # uint8 (earned tracker)
        self.pow_w = torch.as_tensor(_POW, device=d)              # base-18 packing

        N = num_envs
        self.boards = torch.zeros((N, 4, 4), dtype=torch.uint8, device=d)
        self.score = torch.zeros(N, dtype=torch.int64, device=d)
        self.tick = torch.zeros(N, dtype=torch.int32, device=d)
        self.episode_reward = torch.zeros(N, dtype=torch.float32, device=d)
        self.moves_made = torch.zeros(N, dtype=torch.int32, device=d)
        self.max_episode_ticks = torch.full((N,), BASE_MAX_TICKS, dtype=torch.int64, device=d)
        self.max_tile = torch.zeros(N, dtype=torch.uint8, device=d)        # episode max
        self.lifetime_max_tile = torch.zeros(N, dtype=torch.uint8, device=d)  # init() once
        self.is_scaffolding = torch.zeros(N, dtype=torch.bool, device=d)
        # Earned-tile tracker (addition, logging-only): largest exponent
        # CREATED BY A MERGE this episode. Scaffold-placed tiles never set
        # it; any merge result counts as earned, even when one parent was a
        # scaffolded tile (e.g. earned 8192 + scaffolded 8192 -> earned 16384).
        self.earned_max_tile = torch.zeros(N, dtype=torch.uint8, device=d)

        # add_log accumulators (Log struct)
        self._log_keys = ['perf', 'score', 'merge_score', 'episode_return',
                          'episode_length', 'lifetime_max_tile',
                          'reached_16384', 'reached_32768', 'reached_65536',
                          'reached_131072', 'n']
        self._log = {k: torch.zeros((), dtype=torch.float64, device=d)
                     for k in self._log_keys}

        # Extended log (addition, logging-only): raw SUMS over ALL episodes,
        # split normal ('norm') / scaffolding ('scaf'). The original add_log
        # above stays untouched and still skips scaffolding episodes.
        self.XMILESTONES = (13, 14, 15, 16, 17)   # 8192 .. 131072
        xkeys = []
        for cls in ('norm', 'scaf'):
            xkeys += [f'xn_{cls}', f'xsum_maxtile_{cls}',
                      f'xsum_earnedmax_{cls}', f'xsum_len_{cls}',
                      f'xsum_mergescore_{cls}']
            for m in self.XMILESTONES:
                xkeys += [f'xr{m}_{cls}', f'xe{m}_{cls}']
        self._xlog_keys = xkeys
        self._xlog = {k: torch.zeros((), dtype=torch.float64, device=d)
                      for k in xkeys}
        self._step_count = 0

    # ------------------------------------------------------------------
    def _rand_uniform(self, n):
        return torch.rand(n, generator=self.gen, device=self.device)

    def _get_new_tile(self, n):
        # get_new_tile(): 10% chance of exponent 2 (tile 4), 90% exponent 1 (tile 2)
        r = torch.randint(0, 10, (n,), generator=self.gen, device=self.device)
        return torch.where(r == 0, 2, 1).to(torch.uint8)

    def _place_tile_at_random_cell(self, env_idx, tiles):
        """place_tile_at_random_cell(): uniform over empty cells; no-op if full."""
        if env_idx.numel() == 0:
            return
        flat = self.boards.view(self.num_envs, 16)
        sub = flat[env_idx]                                   # [n, 16]
        empty = sub == 0
        k = empty.sum(dim=1)                                  # empty_count
        has_empty = k > 0
        if not bool(has_empty.any()):
            return
        env_idx = env_idx[has_empty]
        sub = sub[has_empty]
        empty = empty[has_empty]
        k = k[has_empty]
        tiles = tiles[has_empty]
        # target = rand() % empty_count  (uniform over 0..k-1)
        u = self._rand_uniform(env_idx.numel())
        target = torch.clamp((u * k).long(), max=(k - 1).long())
        csum = empty.long().cumsum(dim=1)
        pick = empty & (csum == (target + 1).unsqueeze(1))    # exactly one True
        pos = pick.long().argmax(dim=1)
        flat[env_idx, pos] = tiles

    # ------------------------------------------------------------------
    def _set_scaffolding_curriculum(self, env_idx):
        """set_scaffolding_curriculum(): verbatim branch structure."""
        if env_idx.numel() == 0:
            return
        life = self.lifetime_max_tile[env_idx].long()
        low = life < 14

        # Branch 1: lifetime < 14 -> one tile, exp = max(12 + rand()%5, lifetime)
        idx_lo = env_idx[low]
        if idx_lo.numel() > 0:
            cur = torch.randint(0, 5, (idx_lo.numel(),), generator=self.gen,
                                device=self.device)
            high_tile = torch.maximum(12 + cur, self.lifetime_max_tile[idx_lo].long())
            self._place_tile_at_random_cell(idx_lo, high_tile.to(torch.uint8))

        # Branch 2: base = 15 if lifetime >= 16 else 14; rand()%4 cases
        idx_hi = env_idx[~low]
        if idx_hi.numel() > 0:
            base = torch.where(self.lifetime_max_tile[idx_hi] >= 16, 15, 14).long()
            cur = torch.randint(0, 4, (idx_hi.numel(),), generator=self.gen,
                                device=self.device)
            # First placement: c0 -> base, c1 -> base+1, c2 -> base, c3 -> base+1
            first = torch.where((cur == 1) | (cur == 3), base + 1, base)
            self._place_tile_at_random_cell(idx_hi, first.to(torch.uint8))
            # Second placement only for c2 (base-1) and c3 (base)
            two = (cur == 2) | (cur == 3)
            idx2 = idx_hi[two]
            if idx2.numel() > 0:
                second = torch.where(cur[two] == 2, base[two] - 1, base[two])
                self._place_tile_at_random_cell(idx2, second.to(torch.uint8))

    def _reset_envs(self, env_idx):
        """c_reset(): zero state, draw scaffolding, place tiles, refresh obs."""
        if env_idx.numel() == 0:
            return
        self.boards[env_idx] = 0
        self.score[env_idx] = 0
        self.tick[env_idx] = 0
        self.episode_reward[env_idx] = 0.0
        self.moves_made[env_idx] = 0
        self.max_episode_ticks[env_idx] = BASE_MAX_TICKS
        self.max_tile[env_idx] = 0
        self.earned_max_tile[env_idx] = 0

        u = self._rand_uniform(env_idx.numel())
        scaff = u < self.scaffolding_ratio
        self.is_scaffolding[env_idx] = scaff

        self._set_scaffolding_curriculum(env_idx[scaff])

        idx_norm = env_idx[~scaff]
        if idx_norm.numel() > 0:
            # Add two random tiles at the start (value drawn per placement)
            self._place_tile_at_random_cell(idx_norm, self._get_new_tile(idx_norm.numel()))
            self._place_tile_at_random_cell(idx_norm, self._get_new_tile(idx_norm.numel()))

    def reset(self, seed=None):
        if seed is not None:
            self.gen.manual_seed(int(seed))
        all_idx = torch.arange(self.num_envs, device=self.device)
        # init(): lifetime_max_tile cleared only at construction/reset-all
        self.lifetime_max_tile.zero_()
        self._reset_envs(all_idx)
        return self.observations()

    def observations(self):
        # update_observations(): memcpy of the row-major grid
        return self.boards.view(self.num_envs, 16)

    # ------------------------------------------------------------------
    def _oriented_rows(self, action_mask_boards, direction):
        """Extract rows so a LEFT slide reproduces C move() for `direction`.
        UP:   temp[i] = grid[i][col]        -> transpose
        DOWN: temp[i] = grid[SIZE-1-i][col] -> transpose + flip
        LEFT: temp[i] = grid[row][i]        -> identity
        RIGHT:temp[i] = grid[row][SIZE-1-i] -> flip
        """
        b = action_mask_boards
        if direction == 0:      # UP
            return b.transpose(1, 2).contiguous()
        if direction == 1:      # DOWN
            return b.transpose(1, 2).flip(2).contiguous()
        if direction == 2:      # LEFT
            return b.contiguous()
        return b.flip(2).contiguous()  # RIGHT

    def _restore_orientation(self, rows, direction):
        if direction == 0:
            return rows.transpose(1, 2)
        if direction == 1:
            return rows.flip(2).transpose(1, 2)
        if direction == 2:
            return rows
        return rows.flip(2)

    def _apply_moves(self, actions):
        """move(): returns (moved, reward, score_gain, merge_max) and updates
        boards. merge_max is the largest exponent created by a merge this
        step (0 if none) -- consumed only by the earned-tile tracker."""
        N = self.num_envs
        moved = torch.zeros(N, dtype=torch.bool, device=self.device)
        reward = torch.zeros(N, dtype=torch.float32, device=self.device)
        score_gain = torch.zeros(N, dtype=torch.int64, device=self.device)
        merge_max = torch.zeros(N, dtype=torch.uint8, device=self.device)

        for direction in range(4):
            sel = (actions == direction).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            rows = self._oriented_rows(self.boards[sel], direction)   # [n,4,4]
            idx = (rows.long() * self.pow_w.view(1, 1, 4)).sum(dim=2)  # [n,4]
            new_rows = self.t_rows[idx]                                # [n,4,4]
            reward[sel] = self.t_rewards[idx].sum(dim=1)
            score_gain[sel] = self.t_scores[idx].long().sum(dim=1)
            moved[sel] = self.t_moved[idx].any(dim=1)
            merge_max[sel] = self.t_merge_max[idx].amax(dim=1)
            self.boards[sel] = self._restore_orientation(new_rows, direction)

        return moved, reward, score_gain, merge_max

    def _is_game_over(self):
        """is_game_over(): board full and no adjacent equal pair."""
        b = self.boards
        full = ~(b == 0).any(dim=(1, 2))
        eq_h = (b[:, :, :3] == b[:, :, 1:]).any(dim=(1, 2))
        eq_v = (b[:, :3, :] == b[:, 1:, :]).any(dim=(1, 2))
        return full & ~eq_h & ~eq_v

    def _add_log(self, env_idx):
        """add_log(): skip scaffolding episodes; update lifetime best first."""
        if env_idx.numel() == 0:
            return
        keep = ~self.is_scaffolding[env_idx]
        env_idx = env_idx[keep]
        if env_idx.numel() == 0:
            return
        mt = self.max_tile[env_idx].long()
        life = torch.maximum(self.lifetime_max_tile[env_idx].long(), mt)
        self.lifetime_max_tile[env_idx] = life.to(torch.uint8)

        two_pow_mt = (1 << mt).double()
        perf = torch.clamp(0.8 * two_pow_mt / 65536.0, max=1.0)

        L = self._log
        L['perf'] += perf.sum()
        L['score'] += two_pow_mt.sum()
        L['merge_score'] += self.score[env_idx].double().sum()
        L['episode_length'] += self.tick[env_idx].double().sum()
        L['episode_return'] += self.episode_reward[env_idx].double().sum()
        L['lifetime_max_tile'] += (1 << life).double().sum()
        L['reached_16384'] += (mt >= 14).double().sum()
        L['reached_32768'] += (mt >= 15).double().sum()
        L['reached_65536'] += (mt >= 16).double().sum()
        L['reached_131072'] += (mt >= 17).double().sum()
        L['n'] += env_idx.numel()

    def _add_xlog(self, env_idx):
        """Extended log (addition): tallies for ALL finished episodes, split
        normal/scaffolding, with earned milestones. Raw sums, no means."""
        if env_idx.numel() == 0:
            return
        scaf = self.is_scaffolding[env_idx]
        for cls, mask in (('norm', ~scaf), ('scaf', scaf)):
            idx = env_idx[mask]
            if idx.numel() == 0:
                continue
            mt = self.max_tile[idx].long()
            emt = self.earned_max_tile[idx].long()
            X = self._xlog
            X[f'xn_{cls}'] += idx.numel()
            X[f'xsum_maxtile_{cls}'] += (1 << mt).double().sum()
            X[f'xsum_earnedmax_{cls}'] += (1 << emt).double().sum()
            X[f'xsum_len_{cls}'] += self.tick[idx].double().sum()
            X[f'xsum_mergescore_{cls}'] += self.score[idx].double().sum()
            for m in self.XMILESTONES:
                X[f'xr{m}_{cls}'] += (mt >= m).double().sum()
                X[f'xe{m}_{cls}'] += (emt >= m).double().sum()

    def pop_logs(self):
        """binding.vec_log(): mean per episode over the accumulation window.
        Original keys are untouched; extended raw sums ride along under
        'xstats' (addition, logging-only)."""
        n = self._log['n'].item()
        xn = (self._xlog['xn_norm'] + self._xlog['xn_scaf']).item()
        if n == 0 and xn == 0:
            return None
        out = {}
        if n > 0:
            out = {k: (self._log[k].item() / n)
                   for k in self._log_keys if k != 'n'}
            out['n'] = n
            for k in self._log:
                self._log[k].zero_()
        if xn > 0:
            out['xstats'] = {k: self._xlog[k].item() for k in self._xlog_keys}
            for k in self._xlog:
                self._xlog[k].zero_()
        return out if out else None

    # ------------------------------------------------------------------
    def step(self, actions):
        """c_step(): move -> spawn -> tick limits -> terminal -> auto-reset."""
        actions = actions.to(self.device).long().view(-1)
        moved, reward, score_gain, merge_max = self._apply_moves(actions)
        self.tick += 1
        # Earned tracker: any merge result counts as earned this episode.
        self.earned_max_tile = torch.maximum(self.earned_max_tile, merge_max)

        midx = moved.nonzero(as_tuple=True)[0]
        if midx.numel() > 0:
            self.moves_made[midx] += 1
            # update_stats(): recompute max tile from post-move, pre-spawn grid
            self.max_tile[midx] = self.boards[midx].amax(dim=(1, 2))
            self._place_tile_at_random_cell(midx, self._get_new_tile(midx.numel()))
            self.score[midx] += score_gain[midx]
            # Dynamic tick limit (practically no limit for competent agents)
            tick_mult = torch.clamp(self.lifetime_max_tile[midx].long() - 8, min=1)
            self.max_episode_ticks[midx] = torch.maximum(
                BASE_MAX_TICKS * tick_mult, self.score[midx] // 4)

        reward = torch.where(moved, reward,
                             torch.full_like(reward, INVALID_MOVE_PENALTY))

        game_over = self._is_game_over()
        max_ticks_reached = self.tick.long() >= self.max_episode_ticks
        terminals = game_over | max_ticks_reached

        # Game over penalty overrides other rewards (added on top, as in C)
        reward = reward + game_over.float() * GAME_OVER_PENALTY

        self.episode_reward += reward

        done_idx = terminals.nonzero(as_tuple=True)[0]
        self._add_log(done_idx)
        self._add_xlog(done_idx)
        self._reset_envs(done_idx)

        self._step_count += 1
        info = None
        if self._step_count % self.log_interval == 0:
            info = self.pop_logs()

        truncations = torch.zeros_like(terminals)
        return (self.observations(), reward, terminals.to(torch.uint8),
                truncations.to(torch.uint8), info)
