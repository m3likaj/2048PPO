"""Exact port of the simple-2048 policy stack.

Sources (https://github.com/kywch/PufferLib/tree/simple-2048, MIT License,
Copyright (c) 2022 PufferAI):
  - class G2048        : pufferlib/ocean/torch.py
  - class LSTMWrapper  : pufferlib/models.py  (rnn_name = Recurrent -> this)
  - layer_init         : pufferlib/pytorch.py (CleanRL default init)
  - sample_logits/log_prob/entropy : pufferlib/pytorch.py (discrete path)

Network idea credit: David Rubinstein (value + position embeddings).

Faithfulness notes, preserved on purpose:
  * embed_dim = ceil(33 ** 0.25) = 3.
  * LSTMWrapper re-initializes ALL wrapped-policy parameters (including the
    two Embedding tables and the layer_init'd Linears) with orthogonal gain
    1.0 / zero bias BEFORE creating the LSTM, exactly as the original does.
    The nn.LSTM itself therefore keeps PyTorch's default uniform init.
  * The inference path uses an LSTMCell whose weights are tied to the LSTM.
"""

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization (pufferlib/pytorch.py)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class G2048Policy(nn.Module):
    """pufferlib/ocean/torch.py :: class G2048 (verbatim structure)."""

    def __init__(self, num_actions=4, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.embed_dim = int(np.ceil(33 ** 0.25))          # = 3
        self.num_grid_cell = 4 * 4
        self.num_obs = self.num_grid_cell * self.embed_dim  # 48

        self.value_embed = torch.nn.Embedding(18, self.embed_dim)
        self.pos_embed = torch.nn.Embedding(self.num_grid_cell, self.embed_dim)

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            layer_init(nn.Linear(self.num_obs, 2 * hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(2 * hidden_size, hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )

        num_atns = num_actions
        self.decoder = torch.nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_size, num_atns), std=0.01),
        )
        self.value = torch.nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        value_obs = self.value_embed(observations.long())
        pos_obs = self.pos_embed.weight.expand(*value_obs.shape)
        grid_obs = (value_obs + pos_obs).flatten(1)
        return self.encoder(grid_obs)

    def decode_actions(self, hidden):
        logits = self.decoder(hidden)
        values = self.value(hidden)
        return logits, values


class LSTMWrapper(nn.Module):
    """pufferlib/models.py :: class LSTMWrapper (verbatim logic)."""

    def __init__(self, policy, obs_shape=(16,), input_size=512, hidden_size=512):
        super().__init__()
        self.obs_shape = tuple(obs_shape)

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        # NOTE: runs BEFORE self.lstm exists -> re-inits only the wrapped
        # policy's parameters, exactly as in the original.
        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

    def forward_eval(self, observations, state):
        """Inference: single step with the tied LSTMCell."""
        hidden = self.policy.encode_observations(observations, state=state)
        h = state['lstm_h']
        c = state['lstm_c']

        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0]
            lstm_state = (h, c)
        else:
            lstm_state = None

        hidden, c = self.cell(hidden, lstm_state)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        """Training: time-batched LSTM over bptt_horizon."""
        x = observations
        lstm_h = state['lstm_h']
        lstm_c = state['lstm_c']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B * TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B * TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)
        hidden = hidden.transpose(0, 1)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B * TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        state['hidden'] = hidden
        state['lstm_h'] = lstm_h.detach()
        state['lstm_c'] = lstm_c.detach()
        return logits, values


# ----- pufferlib/pytorch.py sampling helpers (discrete path, verbatim) -----

def logits_to_probs(logits):
    return torch.nn.functional.softmax(logits, dim=-1)


def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)


def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)


def sample_logits(logits, action=None):
    """Discrete branch of pufferlib.pytorch.sample_logits."""
    logits = logits.unsqueeze(0)
    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(logits)

    if action is None:
        probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
        action = torch.multinomial(
            probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()
        action = action.reshape(probs.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    logprob = log_prob(normalized_logits, action)
    logits_entropy = entropy(normalized_logits).sum(0)

    return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)


# ---------------------------------------------------------------------------
# Attention encoder (ADDITION -- experiment only; `--encoder mlp` is untouched)
#
# Swaps ONLY G2048Policy.encoder / encode_observations. Everything downstream
# is frozen: decode_actions, the value head, LSTMWrapper, the trainer, and
# every hyperparameter. The tokens fed to attention are choe's own
# value_embed + pos_embed (16 tokens of dim 3), so the only thing that changes
# is how those 16 tokens are MIXED: flatten+MLP vs self-attention.
#
# Positional information is NOT optional here (self-attention is permutation
# invariant and 2048 is entirely about geometry); it comes from the existing
# pos_embed, exactly as in the baseline.
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Standard pre-LN transformer block over the 16 board cells."""

    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class G2048AttentionPolicy(G2048Policy):
    """G2048Policy with the flatten+MLP encoder replaced by self-attention.

    Sequence length is 16, so no efficiency-oriented attention variant is
    relevant; this is plain multi-head self-attention.
    """

    def __init__(self, num_actions=4, hidden_size=512, d_model=64,
                 n_layers=1, n_heads=4, ff_mult=4):
        super().__init__(num_actions=num_actions, hidden_size=hidden_size)
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads

        # 3-dim tokens are very narrow for multi-head attention -> project up.
        self.token_proj = layer_init(nn.Linear(self.embed_dim, d_model))
        self.blocks = nn.ModuleList(
            [AttentionBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)])
        # Flatten the 16 attended tokens back to the 512-dim interface the
        # LSTM expects. Trailing GELU matches the baseline encoder's output
        # activation, so the LSTM sees comparable statistics.
        self.out_proj = layer_init(nn.Linear(self.num_grid_cell * d_model,
                                             hidden_size))
        self.out_act = nn.GELU()
        del self.encoder      # the MLP encoder is replaced, not stacked

    def encode_observations(self, observations, state=None):
        value_obs = self.value_embed(observations.long())         # [B, 16, 3]
        pos_obs = self.pos_embed.weight.expand(*value_obs.shape)
        x = self.token_proj(value_obs + pos_obs)                  # [B, 16, d]
        for blk in self.blocks:
            x = blk(x)
        return self.out_act(self.out_proj(x.flatten(1)))          # [B, 512]


class G2048MatchedMLPPolicy(G2048Policy):
    """Baseline architecture with the first encoder layer resized so the total
    parameter count matches a given target (the attention model).

    This is the capacity control: if the attention model wins, this says
    whether it won because of attention or because of parameter count.
    """

    def __init__(self, num_actions=4, hidden_size=512, hidden1=None,
                 target_params=None):
        super().__init__(num_actions=num_actions, hidden_size=hidden_size)
        if hidden1 is None:
            assert target_params is not None, 'need hidden1 or target_params'
            hidden1 = solve_matched_hidden1(target_params, hidden_size,
                                            self.num_obs)
        self.hidden1 = hidden1
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            layer_init(nn.Linear(self.num_obs, hidden1)),
            nn.GELU(),
            layer_init(nn.Linear(hidden1, hidden_size)),
            nn.GELU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )


def solve_matched_hidden1(target_total_params, hidden_size=512, num_obs=48):
    """Pick encoder width h1 so the FULL policy hits target_total_params.

    Encoder params(h1) = (num_obs+1)*h1 + (h1+1)*hidden + (hidden+1)*hidden,
    and everything outside the encoder is fixed, so this is linear in h1.
    """
    fixed = count_params(make_policy(hidden_size=hidden_size)) - \
        encoder_params_mlp(1024, hidden_size, num_obs)
    per_h1 = (num_obs + 1) + hidden_size
    const = hidden_size * hidden_size + hidden_size + hidden_size
    h1 = round((target_total_params - fixed - const) / per_h1)
    return max(1, h1)


def encoder_params_mlp(h1, hidden_size=512, num_obs=48):
    return ((num_obs + 1) * h1 + (h1 + 1) * hidden_size
            + (hidden_size + 1) * hidden_size)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def make_policy(hidden_size=512, rnn_input_size=512, rnn_hidden_size=512,
                encoder='mlp', d_model=64, n_layers=1, n_heads=4,
                matched_hidden1=None):
    """policy_name = G2048, rnn_name = Recurrent (g2048.ini [base]).

    encoder='mlp' is the faithful baseline and the default -- the attention
    options are an additive experiment and change nothing when unused.
    """
    if encoder == 'mlp':
        policy = G2048Policy(num_actions=4, hidden_size=hidden_size)
    elif encoder == 'attention':
        policy = G2048AttentionPolicy(
            num_actions=4, hidden_size=hidden_size, d_model=d_model,
            n_layers=n_layers, n_heads=n_heads)
    elif encoder == 'mlp_matched':
        if matched_hidden1 is None:
            target = count_params(make_policy(
                hidden_size, rnn_input_size, rnn_hidden_size,
                encoder='attention', d_model=d_model, n_layers=n_layers,
                n_heads=n_heads))
            matched_hidden1 = solve_matched_hidden1(target, hidden_size)
        policy = G2048MatchedMLPPolicy(num_actions=4, hidden_size=hidden_size,
                                       hidden1=matched_hidden1)
    else:
        raise ValueError(f'Unknown encoder: {encoder}')
    return LSTMWrapper(policy, obs_shape=(16,), input_size=rnn_input_size,
                       hidden_size=rnn_hidden_size)
