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


def make_policy(hidden_size=512, rnn_input_size=512, rnn_hidden_size=512):
    """policy_name = G2048, rnn_name = Recurrent (g2048.ini [base])."""
    policy = G2048Policy(num_actions=4, hidden_size=hidden_size)
    return LSTMWrapper(policy, obs_shape=(16,), input_size=rnn_input_size,
                       hidden_size=rnn_hidden_size)
