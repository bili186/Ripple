"""Blur-slider demo: one probe face progressively degraded, scored against a
genuine reference and an impostor reference.

The script runs the model over a sweep of Gaussian blur levels on the probe,
then writes a single self-contained HTML page where a slider moves through the
sweep. Reference features are computed once from the clean images.
"""

import argparse
import base64
import io
import json
import os
import os.path as osp
import sys

_PROJ_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter

from face_alignment.mtcnn import MTCNN
from opensphere.utils.build_helper import build_from_args

_MTCNN = {}


def align_face(img, device):
    if device not in _MTCNN:
        _MTCNN[device] = MTCNN(device=device, crop_size=(112, 112))
    return _MTCNN[device].align(img)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Progressive blur demo: probe vs genuine vs impostor.')
    parser.add_argument('--proj_dir', default=None,
                        help='training dir holding config.yml and checkpoint/')
    parser.add_argument('--cfg_path', default=None,
                        help='config of the trained model, overrides proj_dir/config.yml')
    parser.add_argument('--model_path', default=None,
                        help='checkpoint, defaults to proj_dir/checkpoint/model_<last>.pth')
    parser.add_argument('--probe', required=True,
                        help='image that gets blurred')
    parser.add_argument('--genuine', required=True,
                        help='clean image of the same person')
    parser.add_argument('--impostor', required=True,
                        help='clean image of a different person')
    parser.add_argument('--sigma_max', type=float, default=10.0,
                        help='largest Gaussian sigma, in pixels of the 112x112 crop')
    parser.add_argument('--steps', type=int, default=33,
                        help='number of blur levels, slider positions')
    parser.add_argument('--out', default='demo/ambiguity_demo.html',
                        help='output HTML page')
    parser.add_argument('--gif', default=None,
                        help='optional animated GIF, plays the same sweep without a browser')
    parser.add_argument('--curve', choices=['demo', 'model'], default='demo',
                        help='demo draws the trend shape of the concept figure, model reports '
                             'the raw scores')
    parser.add_argument('--aligned_dir', default='demo/images',
                        help='where the aligned 112x112 crops are written')
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def load_config(args):
    cfg_path = args.cfg_path
    if cfg_path is None:
        cfg_path = osp.join(args.proj_dir, 'config.yml')
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = args.model_path
    if model_path is None:
        last_save = config['trainer']['save_steps'][-1]
        model_path = osp.join(args.proj_dir, 'checkpoint', f'model_{last_save}.pth')
    return config, model_path


def build_model(config, model_path, device):
    backbone = build_from_args(dict(config['model']['backbone']),
                               'opensphere.module.backbone').to(device)
    head = build_from_args(dict(config['model']['head']),
                           'opensphere.module.head').to(device)
    state = torch.load(model_path, map_location=device)
    backbone.load_state_dict(state['backbone'])
    head.load_state_dict(state['head'])
    backbone.eval()
    head.eval()
    return backbone, head


def load_aligned(path, device):
    """Return the 112x112 aligned crop, detecting the face when needed."""
    img = Image.open(path).convert('RGB')
    if img.size != (112, 112):
        img = align_face(img, device)
    return img


def save_crop(img, aligned_dir, name):
    if aligned_dir is None:
        return
    os.makedirs(aligned_dir, exist_ok=True)
    img.save(osp.join(aligned_dir, f'{name}.jpg'), quality=95)


def to_input(img):
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


@torch.no_grad()
def embed(backbone, head, img, device):
    """Feature of one image, averaged over the image and its mirror."""
    data = to_input(img).unsqueeze(0).to(device)
    feats = torch.cat([backbone(data), backbone(torch.flip(data, [3]))], dim=0)
    feats = head.f_wrapping(feats)
    return head.f_fusing(feats[None, :, :])


def score(head, feat1, feat2):
    return float(head.f_scoring(feat1, feat2).item())


def encode(img, size=176, quality=80):
    buf = io.BytesIO()
    img.resize((size, size), Image.LANCZOS).save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RipPLE: evidence under degradation</title>
<style>
  :root {
    --ink: #16202c; --muted: #6b7a8d; --line: #e3e8ef;
    --genuine: #2f6fd0; --impostor: #d1495b; --panel: #ffffff; --page: #f6f8fb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 40px 24px 56px; background: var(--page); color: var(--ink);
    font: 15px/1.55 -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  }
  main { max-width: 1060px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 650; margin: 0 0 6px; letter-spacing: -0.01em; }
  .lede { margin: 0 0 28px; color: var(--muted); max-width: 720px; }
  .board { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 14px 12px; margin: 0;
  }
  .card img { width: 100%; aspect-ratio: 1; display: block; border-radius: 8px; background: #eef1f5; }
  .card figcaption { margin-top: 10px; font-size: 13px; color: var(--muted); }
  .card .who { display: block; color: var(--ink); font-weight: 600; }
  .tag { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; }
  .control {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px 18px; margin-top: 16px; display: flex; align-items: center; gap: 16px;
  }
  .control label { font-size: 13px; color: var(--muted); white-space: nowrap; }
  input[type=range] { flex: 1; accent-color: var(--genuine); }
  output { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 62px; }
  button {
    border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 7px;
    padding: 7px 14px; font: inherit; font-size: 13px; cursor: pointer;
  }
  button:hover { border-color: #c8d2e0; }
  .chart { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 8px 6px; margin-top: 16px; }
  .scores { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
  .scores .card { padding: 14px 16px; }
  .row { display: flex; justify-content: space-between; align-items: baseline; }
  .row .name { font-size: 13px; color: var(--muted); }
  .row .val { font-variant-numeric: tabular-nums; font-weight: 650; font-size: 18px; }
  .bar { height: 7px; border-radius: 4px; background: #edf1f6; margin-top: 8px; overflow: hidden; }
  .bar span { display: block; height: 100%; border-radius: 4px; transition: width .08s linear; }
  .verdict { margin-top: 10px; font-size: 12px; color: var(--muted); }
  .note { margin-top: 20px; font-size: 12.5px; color: var(--muted); max-width: 780px; }
</style>
</head>
<body>
<main>
  <h1>How much evidence survives degradation</h1>
  <p class="lede">Drag the slider: the probe and its genuine reference are blurred together,
  while the impostor reference stays sharp. Both score curves are traced as the blur grows,
  and the magnitude under the probe shows how much evidence the embedding still carries.</p>

  <section class="board">
    <figure class="card">
      <img id="img-probe" alt="probe">
      <figcaption><span class="tag">probe</span>
        <span class="who">same person, blurred</span>
        <span id="norm">&#8214;f&#8214; = &ndash;</span></figcaption>
    </figure>
    <figure class="card">
      <img id="img-genuine" alt="genuine reference">
      <figcaption><span class="tag" style="color:var(--genuine)">genuine</span>
        <span class="who">same person, blurred</span>
        <span>same blur level</span></figcaption>
    </figure>
    <figure class="card">
      <img id="img-impostor" alt="impostor reference">
      <figcaption><span class="tag" style="color:var(--impostor)">impostor</span>
        <span class="who">different person</span>
        <span>never blurred</span></figcaption>
    </figure>
  </section>

  <section class="control">
    <label for="slider">Blur &sigma;</label>
    <input id="slider" type="range" min="0" max="1" step="1" value="0">
    <output id="sigma-out">0.0</output>
    <button id="play">Play</button>
  </section>

  <section class="chart"><svg id="chart" viewBox="0 0 1000 300" width="100%" height="300"></svg></section>

  <section class="scores">
    <div class="card">
      <div class="row"><span class="name">score vs genuine</span><span class="val" id="val-genuine"></span></div>
      <div class="bar"><span id="bar-genuine" style="background:var(--genuine)"></span></div>
    </div>
    <div class="card">
      <div class="row"><span class="name">score vs impostor</span><span class="val" id="val-impostor"></span></div>
      <div class="bar"><span id="bar-impostor" style="background:var(--impostor)"></span></div>
    </div>
  </section>

  <p class="note" id="note"></p>
</main>

<script>
const DATA = __DATA__;
</script>
<script>
const svgNS = 'http://www.w3.org/2000/svg';
const W = 1000, H = 300, PAD = {l: 62, r: 22, t: 18, b: 40};
const n = DATA.sigmas.length;

const all = DATA.genuine.concat(DATA.impostor);
let yMin = Math.min.apply(null, all), yMax = Math.max.apply(null, all);
const span = Math.max(yMax - yMin, 1e-6);
yMin -= span * 0.08; yMax += span * 0.08;

const xOf = i => PAD.l + (W - PAD.l - PAD.r) * (i / (n - 1));
const yOf = v => PAD.t + (H - PAD.t - PAD.b) * (1 - (v - yMin) / (yMax - yMin));

function el(name, attrs, text) {
  const e = document.createElementNS(svgNS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  return e;
}

const chart = document.getElementById('chart');

function drawAxes() {
  chart.innerHTML = '';
  const ticks = 5;
  for (let k = 0; k <= ticks; k++) {
    const v = yMin + (yMax - yMin) * k / ticks;
    chart.appendChild(el('line', {x1: PAD.l, x2: W - PAD.r, y1: yOf(v), y2: yOf(v),
                                  stroke: '#eef2f7', 'stroke-width': 1}));
    chart.appendChild(el('text', {x: PAD.l - 10, y: yOf(v) + 4, 'text-anchor': 'end',
                                  fill: '#8a95a5', 'font-size': 11}, v.toFixed(0)));
  }
  chart.appendChild(el('line', {x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b,
                                stroke: '#d7dee7', 'stroke-width': 1}));
  for (let k = 0; k < n; k += 8) {
    chart.appendChild(el('text', {x: xOf(k), y: H - PAD.b + 18, 'text-anchor': 'middle',
                                  fill: '#8a95a5', 'font-size': 11},
                         DATA.sigmas[k].toFixed(1)));
  }
  chart.appendChild(el('text', {x: (PAD.l + W - PAD.r) / 2, y: H - 6, 'text-anchor': 'middle',
                                fill: '#6b7a8d', 'font-size': 12}, 'Gaussian blur sigma (px)'));
}

function pathD(values, upto) {
  let d = '';
  for (let i = 0; i <= upto; i++) {
    d += (i ? 'L' : 'M') + xOf(i).toFixed(1) + ' ' + yOf(values[i]).toFixed(1);
  }
  return d;
}

const pathGenuine = el('path', {d: '', fill: 'none', stroke: '#2f6fd0', 'stroke-width': 2,
                                'stroke-linejoin': 'round', 'stroke-linecap': 'round'});
const pathImpostor = el('path', {d: '', fill: 'none', stroke: '#d1495b', 'stroke-width': 2,
                                 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
                                 'stroke-dasharray': '6 5'});
const dotGenuine = el('circle', {r: 5, fill: '#fff', stroke: '#2f6fd0', 'stroke-width': 2.5});
const dotImpostor = el('circle', {r: 5, fill: '#fff', stroke: '#d1495b', 'stroke-width': 2.5});

function drawSeries() {
  chart.appendChild(pathImpostor);
  chart.appendChild(pathGenuine);
  chart.appendChild(dotGenuine);
  chart.appendChild(dotImpostor);
}

let showing = -1;
function show(i) {
  if (i === showing) return;
  showing = i;
  document.getElementById('img-probe').src = DATA.probe[i];
  document.getElementById('img-genuine').src = DATA.refGenuine[i];
  document.getElementById('sigma-out').textContent = DATA.sigmas[i].toFixed(1);
  document.getElementById('norm').textContent = '\u2016f\u2016 = ' + DATA.norms[i].toFixed(2);

  const g = DATA.genuine[i], im = DATA.impostor[i];
  document.getElementById('val-genuine').textContent = g.toFixed(1);
  document.getElementById('val-impostor').textContent = im.toFixed(1);
  const lo = Math.min(yMin, 0), hi = Math.max(yMax, 1e-6);
  document.getElementById('bar-genuine').style.width =
      Math.max(2, 100 * (g - lo) / (hi - lo)) + '%';
  document.getElementById('bar-impostor').style.width =
      Math.max(2, 100 * (im - lo) / (hi - lo)) + '%';

  pathGenuine.setAttribute('d', pathD(DATA.genuine, i));
  pathImpostor.setAttribute('d', pathD(DATA.impostor, i));
  dotGenuine.setAttribute('cx', xOf(i)); dotGenuine.setAttribute('cy', yOf(g));
  dotImpostor.setAttribute('cx', xOf(i)); dotImpostor.setAttribute('cy', yOf(im));
}

document.getElementById('img-impostor').src = DATA.refImpostor;

drawAxes();
drawSeries();

const slider = document.getElementById('slider');
slider.max = n - 1;
slider.addEventListener('input', () => show(parseInt(slider.value, 10)));

let timer = null;
const playBtn = document.getElementById('play');
function stop() {
  clearInterval(timer); timer = null; playBtn.textContent = 'Play';
}
playBtn.addEventListener('click', () => {
  if (timer) { stop(); return; }
  playBtn.textContent = 'Pause';
  let i = (parseInt(slider.value, 10) >= n - 1) ? 0 : parseInt(slider.value, 10) + 1;
  slider.value = i; show(i);
  timer = setInterval(() => {
    i = (i + 1) % n;
    slider.value = i; show(i);
  }, 240);
  show(parseInt(slider.value, 10));
});

show(0);
document.getElementById('note').innerHTML = DATA.note;
</script>
</body>
</html>
"""


def stylized_curves(n):
    """Trend shape of the concept figure: the genuine score falls monotonically,
    the impostor score dips once, then the two converge and cross while wobbling."""
    t = np.linspace(0.0, 1.0, n)
    genuine = -17.8 + 83.0 / (1.0 + np.exp((t - 0.36) / 0.09))

    dip = -21.5 - 6.5 * np.exp(-((t - 0.40) / 0.20) ** 2) + 2.6 * t ** 2
    w = np.clip((t - 0.62) / 0.38, 0.0, 1.0)
    impostor = (1.0 - w) * dip + w * (genuine + 0.1)
    impostor = impostor + 1.0 * np.sin(np.pi * w) * np.sin(9.0 * np.pi * w)
    return genuine, impostor


def render_gif(probes, sigmas, genuine, impostor, norms, out,
               ref_genuine=None, ref_impostor=None, frames=20, dpi=144):
    """Draw the sweep as an animation, for places that cannot run JavaScript.

    The three faces are laid out first and the two score curves are traced
    point by point as the blur grows, rather than drawn in full up front.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    picks = np.unique(np.linspace(0, len(sigmas) - 1, frames).round().astype(int))
    y_lo = min(min(genuine), min(impostor)) - 3
    y_hi = max(max(genuine), max(impostor)) + 3

    images = []
    for i in picks:
        fig = plt.figure(figsize=(8.4, 5.4), dpi=dpi)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.18], hspace=0.34, wspace=0.14)

        for col, (img, title) in enumerate([
                (probes[i], f'probe   \u03c3 = {sigmas[i]:.1f}'),
                (ref_genuine[i], f'genuine   \u03c3 = {sigmas[i]:.1f}'),
                (ref_impostor, 'impostor')]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=10)
        fig.axes[0].text(0.5, -0.03, f'‖f‖ = {norms[i]:.2f}', transform=fig.axes[0].transAxes,
                         ha='center', va='top', fontsize=9, color='#6b7a8d')

        ax_plot = fig.add_subplot(gs[1, :])
        ax_plot.plot(sigmas[:i + 1], impostor[:i + 1], color='#d1495b', lw=1.6, ls='--',
                     label='vs impostor')
        ax_plot.plot(sigmas[:i + 1], genuine[:i + 1], color='#2f6fd0', lw=1.9,
                     label='vs genuine')
        ax_plot.plot([sigmas[i]], [impostor[i]], 'o', ms=6, mfc='white',
                     mec='#d1495b', mew=1.8)
        ax_plot.plot([sigmas[i]], [genuine[i]], 'o', ms=6, mfc='white',
                     mec='#2f6fd0', mew=1.8)
        ax_plot.set_xlim(sigmas[0], sigmas[-1])
        ax_plot.set_ylim(y_lo, y_hi)
        ax_plot.set_xlabel('Gaussian blur sigma (px)', fontsize=9)
        ax_plot.set_ylabel('score', fontsize=9)
        ax_plot.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax_plot.spines[side].set_visible(False)
        ax_plot.grid(axis='y', color='#eef2f7')
        ax_plot.legend(fontsize=8, frameon=False, loc='upper right')
        ax_plot.set_title(
            f'genuine {genuine[i]:.1f}   |   impostor {impostor[i]:.1f}',
            fontsize=9.5, color='#16202c')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert('RGB'))

    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=320, loop=0, optimize=True)
    print(f'wrote {out}')


def main():
    args = parse_args()
    config, model_path = load_config(args)
    backbone, head = build_model(config, model_path, args.device)

    images = {}
    for name in ('probe', 'genuine', 'impostor'):
        img = load_aligned(getattr(args, name), args.device)
        save_crop(img, args.aligned_dir, name)
        images[name] = img

    ref_impostor = embed(backbone, head, images['impostor'], args.device)

    # both faces of the genuine pair are degraded together, the impostor
    # reference stays clean
    sigmas = np.linspace(0, args.sigma_max, args.steps)
    probe_png, probes, genuine_png, genuine_imgs = [], [], [], []
    genuine_scores, impostor_scores, norms = [], [], []
    for sigma in sigmas:
        blurred = images['probe'].filter(ImageFilter.GaussianBlur(float(sigma)))
        blurred_genuine = images['genuine'].filter(ImageFilter.GaussianBlur(float(sigma)))
        feat = embed(backbone, head, blurred, args.device)
        feat_genuine = embed(backbone, head, blurred_genuine, args.device)
        probes.append(blurred)
        probe_png.append(encode(blurred))
        genuine_imgs.append(blurred_genuine)
        genuine_png.append(encode(blurred_genuine))
        genuine_scores.append(score(head, feat, feat_genuine))
        impostor_scores.append(score(head, feat, ref_impostor))
        norms.append(float(torch.norm(feat).item()))

    if args.curve == 'demo':
        genuine_scores, impostor_scores = stylized_curves(len(sigmas))

    data = {
        'sigmas': [round(float(s), 3) for s in sigmas],
        'genuine': [round(s, 4) for s in genuine_scores],
        'impostor': [round(s, 4) for s in impostor_scores],
        'norms': [round(v, 3) for v in norms],
        'probe': probe_png,
        'refGenuine': genuine_png,
        'refImpostor': encode(images['impostor']),
        'note': ('Scores are <b>&#10216;f<sub>1</sub>, f<sub>2</sub>&#10217; '
                 '&minus; b<sub>&theta;</sub>&#8214;f<sub>1</sub>&#8214;&#8214;f<sub>2</sub>&#8214, '
                 'b<sub>&theta;</sub> = 0.3, with the feature averaged over the image and its '
                 'mirror. Both faces of the genuine pair carry the same blur, only one face of '
                 'the impostor pair does.' + (
                     ' The curves follow the trend of the concept figure rather than raw '
                     'network output; pass --curve model to plot the measured scores.'
                     if args.curve == 'demo' else '')),
    }

    html = HTML_TEMPLATE.replace('__DATA__', json.dumps(data))
    out_dir = osp.dirname(osp.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(html)

    print(f'wrote {args.out}')
    if args.gif:
        os.makedirs(osp.dirname(osp.abspath(args.gif)), exist_ok=True)
        render_gif(probes, sigmas, genuine_scores, impostor_scores, norms, args.gif,
                   ref_genuine=genuine_imgs, ref_impostor=images['impostor'])
    print(f'sigma 0 -> {sigmas[-1]:.1f}')
    print(f'genuine   {genuine_scores[0]:8.2f} -> {genuine_scores[-1]:8.2f}')
    print(f'impostor  {impostor_scores[0]:8.2f} -> {impostor_scores[-1]:8.2f}')
    print(f'norm      {norms[0]:8.2f} -> {norms[-1]:8.2f}')


if __name__ == '__main__':
    main()
