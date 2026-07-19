"""PyTorch port of compute_puff_advantage.

Original: pufferlib/extensions/pufferlib.cpp :: puff_advantage_row
(https://github.com/kywch/PufferLib/tree/simple-2048, MIT License,
Copyright (c) 2022 PufferAI).

The C kernel walks each [horizon] row backward:

    lastpufferlam = 0
    for t = horizon-2 .. 0:
        nextnonterminal = 1 - dones[t+1]
        rho_t  = min(importance[t], rho_clip)
        c_t    = min(importance[t], c_clip)
        delta  = rho_t * (rewards[t+1] + gamma * values[t+1] * nextnonterminal
                          - values[t])
        lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam
                        * nextnonterminal
        advantages[t] = lastpufferlam

advantages[horizon-1] is left at its initial value (the trainer passes a
fresh zero tensor every minibatch, so it stays 0). This port vectorizes the
recursion over the segment dimension; per-element math is identical.
"""

import torch


def compute_puff_advantage(values, rewards, terminals, ratio, advantages,
                           gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    horizon = values.shape[1]
    last = torch.zeros(values.shape[0], device=values.device,
                       dtype=values.dtype)
    for t in range(horizon - 2, -1, -1):
        nextnonterminal = 1.0 - terminals[:, t + 1]
        rho_t = torch.clamp(ratio[:, t], max=vtrace_rho_clip)
        c_t = torch.clamp(ratio[:, t], max=vtrace_c_clip)
        delta = rho_t * (rewards[:, t + 1]
                         + gamma * values[:, t + 1] * nextnonterminal
                         - values[:, t])
        last = delta + gamma * gae_lambda * c_t * last * nextnonterminal
        advantages[:, t] = last
    return advantages
