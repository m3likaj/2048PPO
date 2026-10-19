"""Watch a trained 2048 agent play — generates a self-contained HTML replay.

Loads a checkpoint, plays games with scaffolding ON or OFF, records every
board state, and writes an interactive HTML page (zen2048-style) you can open
in a browser to animate / step through the agent's play. Multiple games and
both scaffold settings can go in one page.

Usage:
  # unassisted play (scaffold off) — the real test of learned skill
  python play.py --checkpoint experiments/full_d64l2_seed42/latest.pt \
      --scaffolding-ratio 0 --num-games 3 --out play_noscaffold.html

  # with the training curriculum's scaffolding on (board starts seeded)
  python play.py --checkpoint experiments/full_d64l2_seed42/latest.pt \
      --scaffolding-ratio 0.67 --num-games 3 --out play_scaffold.html

  # both in one page
  python play.py --checkpoint <ckpt> --both --num-games 3 --out play_both.html

  # compare several models in ONE viewer (switch via the dropdown)
  python play.py --checkpoint mlp/latest.pt matched/latest.pt attn/latest.pt \
      --both --num-games 2 --out play_compare.html
"""

import argparse
import html
import json
import os

import torch

from config import CONFIG
from g2048_env import G2048TorchEnv
from g2048_policy import make_policy, sample_logits

ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']


def load_policy(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
    policy = make_policy(CONFIG['hidden_size'], CONFIG['rnn_input_size'],
                         CONFIG['rnn_hidden_size'],
                         encoder=cfg.get('encoder', 'mlp'),
                         d_model=cfg.get('d_model', 64),
                         n_layers=cfg.get('n_layers', 1),
                         n_heads=cfg.get('n_heads', 4)).to(device)
    state = ckpt['policy'] if 'policy' in ckpt else ckpt
    policy.load_state_dict(state)
    policy.eval()
    enc = cfg.get('encoder', 'mlp')
    if enc == 'mlp':
        desc = 'MLP'
    else:
        desc = (f"{enc} d{cfg.get('d_model')} "
                f"L{cfg.get('n_layers')} H{cfg.get('n_heads')}")
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    return policy, desc, epoch


@torch.no_grad()
def play_game(policy, device, scaffolding_ratio, seed, max_steps, greedy):
    env = G2048TorchEnv(num_envs=1, scaffolding_ratio=scaffolding_ratio,
                        seed=seed, device=device, log_interval=10**9)
    obs = env.reset(seed=seed)
    h = torch.zeros(1, policy.hidden_size, device=device)
    c = torch.zeros(1, policy.hidden_size, device=device)
    is_scaf = bool(env.is_scaffolding[0].item())
    frames = []

    def exp(v):
        return (1 << v) if v > 0 else 0

    def snap():
        board = env.boards[0].to('cpu').numpy().astype(int).reshape(-1).tolist()
        mt = int(env.max_tile[0].item())
        em = int(env.earned_max_tile[0].item())
        frames.append({
            'b': board,                        # 16 exponents (row-major)
            's': int(env.score[0].item()),
            'mt': exp(mt),
            'em': exp(em),
            'a': None,
        })

    snap()
    for _ in range(max_steps):
        state = dict(lstm_h=h, lstm_c=c)
        logits, _v = policy.forward_eval(obs, state)
        h, c = state['lstm_h'], state['lstm_c']
        if greedy:
            action = logits.argmax(-1).int().view(-1)
        else:
            action, _lp, _e = sample_logits(logits)
            action = action.view(-1)
        a = int(action[0].item())
        frames[-1]['a'] = ACTION_NAMES[a]
        obs2, _r, term, _tr, _info = env.step(action)
        if bool(term[0].item()):
            break
        obs = obs2
        snap()

    return {
        'scaffold_episode': is_scaf,
        'scaffolding_ratio': scaffolding_ratio,
        'seed': seed,
        'frames': frames,
        'final_score': frames[-1]['s'],
        'final_max_tile': max(f['mt'] for f in frames),
        'final_earned_max': max(f['em'] for f in frames),
        'moves': len(frames) - 1,
    }


def write_html(games, out_path, title):
    data = json.dumps(games, separators=(',', ':'))
    page = HTML_TEMPLATE.replace('__TITLE__', html.escape(title)) \
                        .replace('__DATA__', data)
    with open(out_path, 'w') as f:
        f.write(page)
    return out_path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#faf8ef; --fg:#776e65; --board:#bbada0; --cell:rgba(238,228,218,.35);
    --panel:#fff; --line:#e7e2d9;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --bg:#1c1a17; --fg:#e8e2d6; --board:#4a433d; --cell:rgba(255,255,255,.06);
    --panel:#272420; --line:#3a352f;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font-family:'Segoe UI',system-ui,-apple-system,Arial,sans-serif;
    display:flex;flex-direction:column;align-items:center;padding:16px;gap:14px}
  h1{font-size:1.25rem;margin:.2rem 0;text-align:center}
  .sub{font-size:.8rem;opacity:.65;text-align:center;margin-top:-8px}
  .wrap{display:flex;flex-wrap:wrap;gap:18px;justify-content:center;align-items:flex-start;width:100%;max-width:760px}
  .boardbox{background:var(--board);border-radius:10px;padding:10px;
    width:min(92vw,380px);aspect-ratio:1/1}
  .grid{display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(4,1fr);
    gap:10px;width:100%;height:100%}
  .tile{display:flex;align-items:center;justify-content:center;border-radius:6px;
    background:var(--cell);font-weight:700;font-size:clamp(14px,5vw,30px);
    transition:background .12s, color .12s; color:#776e65}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:14px 16px;min-width:210px;font-size:.92rem}
  .stat{display:flex;justify-content:space-between;gap:12px;padding:3px 0}
  .stat b{font-variant-numeric:tabular-nums}
  .badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.72rem;
    font-weight:700;margin-left:6px}
  .scaf{background:#f2b179;color:#4a2f14}
  .earn{background:#9fd3a3;color:#17341a}
  .controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:center;
    width:100%;max-width:420px}
  button,select{font:inherit;color:var(--fg);background:var(--panel);
    border:1px solid var(--line);border-radius:8px;padding:7px 11px;cursor:pointer}
  button:hover{filter:brightness(1.05)}
  input[type=range]{width:100%}
  .row{display:flex;gap:8px;align-items:center;width:100%;max-width:420px}
  .row label{font-size:.8rem;opacity:.7;white-space:nowrap}
  .mv{font-size:.8rem;opacity:.7;text-align:center}
</style></head>
<body>
  <h1>__TITLE__</h1>
  <div class="sub" id="meta"></div>

  <div class="row">
    <label>Game</label>
    <select id="gamesel"></select>
  </div>

  <div class="wrap">
    <div class="boardbox"><div class="grid" id="grid"></div></div>
    <div class="panel">
      <div class="stat"><span>Score</span><b id="score">0</b></div>
      <div class="stat"><span>Max tile</span><b id="maxtile">0</b></div>
      <div class="stat"><span>Earned max</span><b id="earned">0</b></div>
      <div class="stat"><span>Move</span><b id="movei">0</b></div>
      <div class="stat"><span>Last action</span><b id="action">—</b></div>
      <div class="stat"><span>Episode</span><b id="eptype">—</b></div>
      <div id="note" style="font-size:.78rem;opacity:.7;margin-top:8px"></div>
    </div>
  </div>

  <div class="controls">
    <button id="back">⏮</button>
    <button id="play">▶ Play</button>
    <button id="fwd">⏭</button>
    <button id="restart">↺</button>
  </div>
  <div class="row">
    <label>Speed</label>
    <input type="range" id="speed" min="1" max="60" value="12">
  </div>
  <div class="row">
    <input type="range" id="scrub" min="0" max="0" value="0">
  </div>
  <div class="mv" id="posn"></div>

<script>
const GAMES = __DATA__;
const COLORS = {
  0:'',2:'#eee4da',4:'#ede0c8',8:'#f2b179',16:'#f59563',32:'#f67c5f',
  64:'#f65e3b',128:'#edcf72',256:'#edcc61',512:'#edc850',1024:'#edc53f',
  2048:'#edc22e',4096:'#3c3a32',8192:'#3c3a32',16384:'#1f6f43',
  32768:'#196f3d',65536:'#0b5345',131072:'#512e5f'};
const val = e => e>0 ? (1<<e) : 0;

let gi=0, fi=0, playing=false, timer=null;
const grid=document.getElementById('grid');
const cells=[];
for(let i=0;i<16;i++){const d=document.createElement('div');d.className='tile';grid.appendChild(d);cells.push(d);}

const sel=document.getElementById('gamesel');
GAMES.forEach((g,i)=>{const o=document.createElement('option');o.value=i;
  o.textContent=`${g.model?g.model+' · ':''}${g.scaffold_episode?'scaffold':'no scaffold'} · `
    +`max ${g.final_max_tile} · ${g.moves} moves`;sel.appendChild(o);});

function colorFor(v){
  if(v>=4096) return COLORS[Math.min(v,131072)]||'#3c3a32';
  return COLORS[v]||'#eee4da';
}
function render(){
  const g=GAMES[gi], f=g.frames[fi];
  for(let i=0;i<16;i++){
    const e=f.b[i], v=val(e);
    const c=cells[i];
    c.textContent = v? v : '';
    c.style.background = v? colorFor(v) : 'var(--cell)';
    c.style.color = v>=8 ? '#f9f6f2' : '#776e65';
    c.style.fontSize = v>=1024 ? 'clamp(11px,3.6vw,22px)' : '';
  }
  document.getElementById('score').textContent=f.s.toLocaleString();
  document.getElementById('maxtile').textContent=f.mt;
  document.getElementById('earned').textContent=f.em;
  document.getElementById('movei').textContent=f.move!==undefined?f.move:fi;
  document.getElementById('action').textContent=f.a||'—';
  const scaf=g.scaffold_episode;
  document.getElementById('eptype').innerHTML = scaf
    ? 'seeded <span class="badge scaf">SCAFFOLD</span>'
    : 'empty start <span class="badge earn">UNASSISTED</span>';
  document.getElementById('note').textContent = scaf
    ? 'Board was pre-seeded with high tiles by the curriculum. "Earned max" shows the largest tile the agent built by merging.'
    : 'Agent started from an empty board — every tile here was earned.';
  document.getElementById('scrub').value=fi;
  document.getElementById('posn').textContent=`frame ${fi+1} / ${g.frames.length}`;
}
function loadGame(i){
  gi=i; fi=0; sel.value=i;
  const g=GAMES[gi];
  document.getElementById('scrub').max=g.frames.length-1;
  document.getElementById('meta').textContent =
    `${g.model?g.model+(g.epoch!==undefined?' (epoch '+g.epoch+')':'')+' · ':''}`
    +`seed ${g.seed} · scaffolding_ratio ${g.scaffolding_ratio} · `
    +`final score ${g.final_score.toLocaleString()} · earned max ${g.final_earned_max}`;
  render();
}
function step(d){
  const n=GAMES[gi].frames.length;
  fi=Math.max(0,Math.min(n-1,fi+d));
  render();
  if(fi>=n-1) stop();
}
function play(){ if(playing){stop();return;}
  playing=true; document.getElementById('play').textContent='⏸ Pause';
  const fps=+document.getElementById('speed').value;
  timer=setInterval(()=>step(1), 1000/fps);
}
function stop(){ playing=false; document.getElementById('play').textContent='▶ Play';
  if(timer){clearInterval(timer);timer=null;} }

document.getElementById('fwd').onclick=()=>{stop();step(1);};
document.getElementById('back').onclick=()=>{stop();step(-1);};
document.getElementById('play').onclick=play;
document.getElementById('restart').onclick=()=>{stop();fi=0;render();};
document.getElementById('speed').oninput=()=>{if(playing){stop();play();}};
document.getElementById('scrub').oninput=e=>{stop();fi=+e.target.value;render();};
sel.onchange=e=>{stop();loadGame(+e.target.value);};
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){stop();step(1);}
  else if(e.key==='ArrowLeft'){stop();step(-1);}
  else if(e.key===' '){e.preventDefault();play();}
});
loadGame(0);
</script>
</body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, nargs='+',
                   help='One or more checkpoints. Pass several to compare '
                        'models in one viewer (switch via the dropdown).')
    p.add_argument('--scaffolding-ratio', type=float, default=0.0,
                   help='0 = unassisted (empty board). 0.67 = training curriculum.')
    p.add_argument('--both', action='store_true',
                   help='Record games at scaffolding 0 AND 0.67 in one page.')
    p.add_argument('--num-games', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=4000,
                   help='Cap per game (competent agents play very long).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--greedy', action='store_true',
                   help='Argmax actions instead of sampling (can loop; default samples).')
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', type=str, default='play.html')
    a = p.parse_args()

    ratios = [0.0, 0.67] if a.both else [a.scaffolding_ratio]

    games = []
    descs = []
    for ckpt_path in a.checkpoint:
        policy, desc, epoch = load_policy(ckpt_path, a.device)
        descs.append(desc)
        print(f'=== {desc} (epoch {epoch}) — {ckpt_path} ===')
        for ratio in ratios:
            for g in range(a.num_games):
                seed = a.seed + g + int(ratio * 1000)
                game = play_game(policy, a.device, ratio, seed,
                                 a.max_steps, a.greedy)
                game['model'] = desc
                game['epoch'] = epoch
                games.append(game)
                tag = 'scaffold' if ratio > 0 else 'no-scaffold'
                print(f'  {tag} seed {seed}: {game["moves"]} moves, '
                      f'score {game["final_score"]:,}, max {game["final_max_tile"]}, '
                      f'earned max {game["final_earned_max"]}')

    if len(a.checkpoint) == 1:
        title = f'2048 agent replay — {descs[0]}'
    else:
        title = f'2048 agent replay — {len(a.checkpoint)} models compared'
    write_html(games, a.out, title)
    print(f'\nWrote {a.out} — open it in a browser (or display inline on Kaggle).')


if __name__ == '__main__':
    main()
