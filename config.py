"""Exact hyperparameters for the simple-2048 recreation.

Every value below is taken verbatim from kywch's simple-2048 branch of PufferLib
(MIT License, Copyright (c) 2022 PufferAI; branch author Kyoung Whan Choe):
  - pufferlib/config/ocean/g2048.ini   (env / policy / train overrides)
  - pufferlib/config/default.ini       (PufferLib trainer defaults not overridden)
Source: https://github.com/kywch/PufferLib/tree/simple-2048

Nothing here has been changed from the original. If a value came from
default.ini (because g2048.ini does not override it), the comment says so.
"""

CONFIG = dict(
    # ---------------- environment (g2048.ini [env] + [vec]) ----------------
    # Original runs 4 vectorized C worker processes x 4096 envs each = 16384 agents.
    # This port is a single-process GPU-vectorized env with the same agent count.
    num_envs=16384,                 # [vec] num_envs=4 * [env] num_envs=4096
    scaffolding_ratio=0.67,         # g2048.ini [env]
    log_interval=128,               # G2048.__init__ default (g2048.py)

    # ---------------- policy (g2048.ini [policy] / [rnn]) ------------------
    hidden_size=512,                # g2048.ini [policy] hidden_size
    rnn_input_size=512,             # g2048.ini [rnn] input_size
    rnn_hidden_size=512,            # g2048.ini [rnn] hidden_size
    use_rnn=True,                   # rnn_name = Recurrent

    # ---------------- training (g2048.ini [train]) -------------------------
    total_timesteps=6_767_676_767,  # g2048.ini
    anneal_lr=True,                 # g2048.ini
    min_lr_ratio=0.15,              # g2048.ini
    batch_size='auto',              # g2048.ini -> resolves to num_envs * bptt_horizon
    bptt_horizon=64,                # g2048.ini
    minibatch_size=32768,           # g2048.ini
    clip_coef=0.067,                # g2048.ini
    ent_coef=0.0267,                # g2048.ini
    gae_lambda=0.67,                # g2048.ini
    gamma=0.99567,                  # g2048.ini
    vf_clip_coef=0.167,             # g2048.ini
    vf_coef=2.0,                    # g2048.ini
    learning_rate=0.000467,         # g2048.ini
    max_grad_norm=0.5,              # g2048.ini
    adam_beta1=0.99,                # g2048.ini
    adam_beta2=0.9999,              # g2048.ini
    adam_eps=0.0001,                # g2048.ini
    prio_alpha=0.8,                 # g2048.ini
    prio_beta0=0.1,                 # g2048.ini
    vtrace_c_clip=2.0,              # g2048.ini
    vtrace_rho_clip=1.1,            # g2048.ini

    # -------- trainer defaults inherited from default.ini [train] ----------
    seed=42,                        # default.ini
    torch_deterministic=True,       # default.ini
    device='cuda',                  # default.ini
    optimizer='muon',               # default.ini (heavyball ForeachMuon)
    precision='float32',            # default.ini
    update_epochs=1,                # default.ini
    checkpoint_interval=200,        # default.ini (epochs)
    data_dir='experiments',         # default.ini
    max_minibatch_size=32768,       # default.ini (grad accumulation threshold)
    compile=False,                  # default.ini
)


def resolve(cfg):
    """Replicates PuffeRL.__init__ batch-size resolution (pufferl.py)."""
    cfg = dict(cfg)
    total_agents = cfg['num_envs']
    if cfg['batch_size'] == 'auto':
        cfg['batch_size'] = total_agents * cfg['bptt_horizon']
    horizon = cfg['bptt_horizon']
    batch_size = cfg['batch_size']
    segments = batch_size // horizon
    assert total_agents <= segments, (total_agents, segments)
    cfg['segments'] = segments

    minibatch_size = min(cfg['minibatch_size'], cfg['max_minibatch_size'])
    cfg['accumulate_minibatches'] = max(1, cfg['minibatch_size'] // cfg['max_minibatch_size'])
    cfg['minibatch_size_eff'] = minibatch_size
    cfg['total_minibatches'] = int(cfg['update_epochs'] * batch_size / minibatch_size)
    cfg['minibatch_segments'] = minibatch_size // horizon
    assert cfg['minibatch_segments'] * horizon == minibatch_size
    cfg['total_epochs'] = cfg['total_timesteps'] // batch_size
    return cfg
