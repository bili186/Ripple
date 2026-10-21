#!/usr/bin/env python3
"""Image-level quality filtering on IJB-B/IJB-C with the embedding norm
(ours) or the PFE confidence, followed by template re-aggregation and
rescoring of the protocol pairs. Corrupted images are excluded.

Inputs are the npz files written by dump_ijb_image_feats.py; see --help.
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

FILTER_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.6, 0.7, 0.8, 0.9]
B_THETA = 0.3


def tar_at_far(score, labels, far, tau=None):
    """Return (TAR, tau). Calibrate tau on the negatives when it is None."""
    neg = score[labels == 0]
    pos = score[labels == 1]
    if neg.size == 0 or pos.size == 0:
        return float('nan'), tau
    if tau is None:
        tau = float(np.quantile(neg, 1.0 - far))
    return float((pos > tau).mean() * 100.0), tau


def build_groups(tmpl_id_img, tmpl_ids):
    """Group image indices by template. Returns a list of index arrays."""
    order = np.argsort(tmpl_id_img, kind='stable')
    ids = tmpl_id_img[order]
    st = np.searchsorted(ids, tmpl_ids)
    en = np.searchsorted(ids, tmpl_ids, side='right')
    return [order[st[i]:en[i]] for i in range(len(tmpl_ids))]


def media_weights(media):
    """Media-dedup weights: images of the same media share its mass equally."""
    u, c = np.unique(media, return_counts=True)
    mp = dict(zip(u.tolist(), c.tolist()))
    w = np.array([1.0 / mp[a] for a in media.tolist()], np.float64)
    return w / len(u)


def aggregate_global(feats, media, keep_mask, groups):
    """Aggregate templates from the images kept by keep_mask.

    Returns (tmpl_feat (T, D), valid (T,)). A template with no kept image
    is marked invalid.
    """
    T, D = len(groups), feats.shape[1]
    out = np.zeros((T, D), np.float32)
    valid = np.zeros(T, bool)
    for ti, idxs in enumerate(groups):
        kept = idxs[keep_mask[idxs]]
        if kept.size == 0:
            continue
        out[ti] = (media_weights(media[kept])[:, None] * feats[kept]).sum(0)
        valid[ti] = True
    return out, valid


def aggregate_within(feats, media, groups, quality, fr, eligible=None):
    """Drop the lowest fr fraction of each template, then aggregate.

    When eligible is given, the ratio is computed over the eligible images
    inside the template only.
    """
    T, D = len(groups), feats.shape[1]
    out = np.zeros((T, D), np.float32)
    valid = np.zeros(T, bool)
    for ti, idxs in enumerate(groups):
        if eligible is not None:
            idxs = idxs[eligible[idxs]]
        n = idxs.size
        if n == 0:
            continue
        n_keep = max(1, int(round((1.0 - fr) * n)))
        sel = idxs[np.argsort(-quality[idxs])[:n_keep]] if n_keep < n else idxs
        out[ti] = (media_weights(media[sel])[:, None] * feats[sel]).sum(0)
        valid[ti] = True
    return out, valid


def scores_ours(tf, valid, tmpl_ids, pairs, b_theta):
    """Score S = g * (cos - b), g = ||x1|| * ||x2||. Returns (score, label)."""
    row = {t: i for i, t in enumerate(tmpl_ids.tolist()) if valid[i]}
    t0, t1, labels = pairs[:, 0].astype(np.int64), \
        pairs[:, 1].astype(np.int64), pairs[:, 2].astype(np.int64)
    ok = np.array([(a in row) and (b in row)
                   for a, b in zip(t0.tolist(), t1.tolist())])
    i0 = np.array([row[t] for t in t0[ok].tolist()], np.int64)
    i1 = np.array([row[t] for t in t1[ok].tolist()], np.int64)
    x, y = tf[i0], tf[i1]
    nx = np.linalg.norm(x, axis=1)
    ny = np.linalg.norm(y, axis=1)
    cos = (x * y).sum(1) / (nx * ny + 1e-12)
    g = nx * ny
    return g * (cos - b_theta), labels[ok]


def run_ours_global(feats, r_img, timg, mimg, tmpl_ids, pairs, far, b_theta,
                    eligible=None, fixed_tau=False):
    """Global filtering on the embedding norm r_img."""
    groups = build_groups(timg, tmpl_ids)
    quality = r_img.astype(np.float64)
    if eligible is None:
        eligible = np.ones(feats.shape[0], bool)
    pool = np.flatnonzero(eligible)
    tars, taus = [], []
    tau = None
    for fr in FILTER_RATES:
        cut = min(int(round(pool.size * fr)), pool.size)
        drop = pool[np.argpartition(quality[pool], cut)[:cut]] if cut > 0 else \
            np.zeros(0, np.int64)
        keep = eligible.copy()
        keep[drop] = False
        tf, valid = aggregate_global(feats, mimg, keep, groups)
        s, lab = scores_ours(tf, valid, tmpl_ids, pairs, b_theta)
        tar, tau = tar_at_far(s, lab, far, tau if fixed_tau else None)
        tars.append(tar)
        taus.append(tau)
    return tars, taus


def run_ours_within(feats, r_img, timg, mimg, tmpl_ids, pairs, far, b_theta,
                    eligible=None, fixed_tau=False):
    """Within-template filtering on the embedding norm r_img."""
    groups = build_groups(timg, tmpl_ids)
    quality = r_img.astype(np.float64)
    tars, taus = [], []
    tau = None
    for fr in FILTER_RATES:
        tf, valid = aggregate_within(feats, mimg, groups, quality, fr, eligible)
        s, lab = scores_ours(tf, valid, tmpl_ids, pairs, b_theta)
        tar, tau = tar_at_far(s, lab, far, tau if fixed_tau else None)
        tars.append(tar)
        taus.append(tau)
    return tars, taus


def aggregate_pfe(mu, sigma_sq):
    """Official PFE template aggregation."""
    attention = 1.0 / sigma_sq
    attention = attention / attention.sum(axis=0, keepdims=True)
    mu_new = (mu * attention).sum(axis=0)
    sigma_sq_new = sigma_sq.min(axis=0)
    mu_new = mu_new / (1e-8 + np.linalg.norm(mu_new))
    return mu_new, sigma_sq_new


def pfe_quality(sigma_sq):
    """Per-image confidence q = -sum(log sigma^2)."""
    ss = np.maximum(sigma_sq, 1e-12)
    return -np.log(ss).sum(axis=1)


def scores_pfe(mu, sigma_sq, valid, tmpl_ids, pairs):
    """PFE MLS score: -sum_d[(mu1 - mu2)^2 / (s1 + s2) + log(s1 + s2)]."""
    row = {t: i for i, t in enumerate(tmpl_ids.tolist()) if valid[i]}
    t0, t1, labels = pairs[:, 0].astype(np.int64), \
        pairs[:, 1].astype(np.int64), pairs[:, 2].astype(np.int64)
    ok = np.array([(a in row) and (b in row)
                   for a, b in zip(t0.tolist(), t1.tolist())])
    i0 = np.array([row[t] for t in t0[ok].tolist()], np.int64)
    i1 = np.array([row[t] for t in t1[ok].tolist()], np.int64)
    m0, m1 = mu[i0], mu[i1]
    s0, s1 = sigma_sq[i0], sigma_sq[i1]
    ss = np.maximum(s0 + s1, 1e-12)
    mls = -(((m0 - m1) ** 2) / ss).sum(axis=1) - np.log(ss).sum(axis=1)
    return mls, labels[ok]


def _aggregate_pfe_groups(mu, sigma_sq, groups, keep_mask):
    T, D = len(groups), mu.shape[1]
    mus = np.zeros((T, D), np.float32)
    sgs = np.zeros((T, D), np.float32)
    valid = np.zeros(T, bool)
    for ti, idxs in enumerate(groups):
        kept = idxs[keep_mask[idxs]]
        if kept.size == 0:
            continue
        mus[ti], sgs[ti] = aggregate_pfe(mu[kept], sigma_sq[kept])
        valid[ti] = True
    return mus, sgs, valid


def run_pfe_global(mu, sigma_sq, img_tmpl, tmpl_ids, pairs, far,
                   eligible=None, fixed_tau=False):
    """Global filtering on the PFE confidence."""
    q = pfe_quality(sigma_sq)
    if eligible is None:
        eligible = np.ones(mu.shape[0], bool)
    pool = np.flatnonzero(eligible)
    groups = build_groups(img_tmpl, tmpl_ids)
    tars, taus = [], []
    tau = None
    for fr in FILTER_RATES:
        cut = min(int(round(pool.size * fr)), pool.size)
        drop = pool[np.argpartition(q[pool], cut)[:cut]] if cut > 0 else \
            np.zeros(0, np.int64)
        keep = eligible.copy()
        keep[drop] = False
        mus, sgs, valid = _aggregate_pfe_groups(mu, sigma_sq, groups, keep)
        s, lab = scores_pfe(mus, sgs, valid, tmpl_ids, pairs)
        tar, tau = tar_at_far(s, lab, far, tau if fixed_tau else None)
        tars.append(tar)
        taus.append(tau)
    return tars, taus


def run_pfe_within(mu, sigma_sq, img_tmpl, tmpl_ids, pairs, far,
                   eligible=None, fixed_tau=False):
    """Within-template filtering on the PFE confidence."""
    q = pfe_quality(sigma_sq)
    groups = build_groups(img_tmpl, tmpl_ids)
    T, D = len(groups), mu.shape[1]
    tars, taus = [], []
    tau = None
    for fr in FILTER_RATES:
        mus = np.zeros((T, D), np.float32)
        sgs = np.zeros((T, D), np.float32)
        valid = np.zeros(T, bool)
        for ti, idxs in enumerate(groups):
            if eligible is not None:
                idxs = idxs[eligible[idxs]]
            n = idxs.size
            if n == 0:
                continue
            n_keep = max(1, int(round((1.0 - fr) * n)))
            sel = idxs[np.argsort(-q[idxs])[:n_keep]] if n_keep < n else idxs
            mus[ti], sgs[ti] = aggregate_pfe(mu[sel], sigma_sq[sel])
            valid[ti] = True
        s, lab = scores_pfe(mus, sgs, valid, tmpl_ids, pairs)
        tar, tau = tar_at_far(s, lab, far, tau if fixed_tau else None)
        tars.append(tar)
        taus.append(tau)
    return tars, taus


def run_ds(ds, feats_dir, pfe_dir, far, b_theta, mode, sim_name, fixed_tau):
    sim_path = osp.join(feats_dir, f'img_feats_{sim_name}_{ds}.npz')
    if not osp.exists(sim_path):
        print(f'[skip] {ds}: missing {sim_path}')
        return None
    sim = np.load(sim_path, allow_pickle=True)
    pfe_path = osp.join(pfe_dir, f'img_feats_pfe_sphere64_{ds}.npz')
    if not osp.exists(pfe_path):
        print(f'[skip] {ds}: missing PFE npz')
        return None
    pfe = np.load(pfe_path, allow_pickle=True)
    if 'img_sigma' not in pfe:
        print(f'[skip] {ds}: PFE npz has no per-image img_sigma')
        return None

    feats = sim['feats'].astype(np.float32)
    r_img = sim['r_img'].astype(np.float64)
    timg = sim['tmpl_id_img'].astype(np.int64)
    mimg = sim['media_id_img'].astype(np.int64)
    tmpl_ids = sim['tmpl_ids'].astype(np.int64)
    pairs = sim['pairs'].astype(np.int64)

    # Align PFE images to the ours image order.
    sn = np.asarray(sim['img_name'])
    pn = np.asarray(pfe['img_name'])
    if np.array_equal(sn, pn):
        p_mu = pfe['img_mu'].astype(np.float32)
        p_sg = pfe['img_sigma'].astype(np.float32)
        p_tmpl = pfe['img_tmpl'].astype(np.int64)
        p_corrupt = pfe['corrupt_img'].astype(bool) \
            if 'corrupt_img' in pfe else np.zeros(pn.size, bool)
    else:
        idx = {n: i for i, n in enumerate(pn.tolist())}
        perm = np.array([idx[n] for n in sn.tolist()], np.int64)
        p_mu = pfe['img_mu'][perm].astype(np.float32)
        p_sg = pfe['img_sigma'][perm].astype(np.float32)
        p_tmpl = pfe['img_tmpl'][perm].astype(np.int64)
        p_corrupt = pfe['corrupt_img'][perm].astype(bool) \
            if 'corrupt_img' in pfe else np.zeros(perm.size, bool)
    p_tmpl_ids = pfe['tmpl_ids'].astype(np.int64)

    # Keep clean images only on both sides.
    sim_corrupt = sim['corrupt_img'].astype(bool) \
        if 'corrupt_img' in sim else np.zeros(feats.shape[0], bool)
    ours_elig = ~sim_corrupt
    pfe_elig = ~p_corrupt

    res = {'ours': {}, 'pfe': {}, 'ours_tau': {}, 'pfe_tau': {}}
    if mode == 'global':
        res['ours'][ds], res['ours_tau'][ds] = run_ours_global(
            feats, r_img, timg, mimg, tmpl_ids, pairs, far, b_theta,
            ours_elig, fixed_tau)
        res['pfe'][ds], res['pfe_tau'][ds] = run_pfe_global(
            p_mu, p_sg, p_tmpl, p_tmpl_ids, pairs, far, pfe_elig, fixed_tau)
    else:
        res['ours'][ds], res['ours_tau'][ds] = run_ours_within(
            feats, r_img, timg, mimg, tmpl_ids, pairs, far, b_theta,
            ours_elig, fixed_tau)
        res['pfe'][ds], res['pfe_tau'][ds] = run_pfe_within(
            p_mu, p_sg, p_tmpl, p_tmpl_ids, pairs, far, pfe_elig, fixed_tau)
    print(f'  [{ds}] mode={mode}')
    print('    ours: ' + ' '.join(f'{v:.2f}' if v == v else ' nan'
                                  for v in res['ours'][ds]), flush=True)
    print('    pfe : ' + ' '.join(f'{v:.2f}' if v == v else ' nan'
                                  for v in res['pfe'][ds]), flush=True)
    return res


def plot_curves(json_path, out_path, ours_label, pfe_label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    with open(json_path) as f:
        data = json.load(f)
    fr = np.array(data['filter_rates']) * 100.0
    far = data['far']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, ds, tag in zip(axes, ['ijbb', 'ijbc'],
                           ['(a) IJB-B', '(b) IJB-C']):
        if ds not in data.get('ours', {}) or ds not in data.get('pfe', {}):
            continue
        ax.plot(fr, data['ours'][ds], '-o', color='#D62728', lw=2.2, ms=6,
                label=ours_label)
        ax.plot(fr, data['pfe'][ds], '-s', color='#1F77B4', lw=2.2, ms=6,
                label=pfe_label)
        ax.set_xlabel('filter-out rate (%)')
        ax.set_ylabel(f'TAR@FAR={far:g} (%)')
        ax.set_xlim(-4, 94)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.grid(True, axis='y', color='0.85', lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(tag, pad=8)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.spines['left'].set_linewidth(1.8)
        ax.spines['bottom'].set_linewidth(1.8)
        ax.tick_params(width=1.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feats_dir', default='data/img_feats',
                   help='directory with the ours per-image npz files')
    p.add_argument('--pfe_dir', default=None,
                   help='directory with the PFE per-image npz files '
                        '(defaults to --feats_dir)')
    p.add_argument('--sim_name', default='ripple',
                   help='ours prefix: img_feats_{sim_name}_{ijbb,ijbc}.npz')
    p.add_argument('--out_dir', default='picture/reject_baselines')
    p.add_argument('--out_prefix', default=None,
                   help='output prefix (default: filter_image_quality_{mode})')
    p.add_argument('--far', type=float, default=1e-3)
    p.add_argument('--b_theta', type=float, default=B_THETA)
    p.add_argument('--filter_mode', default='global',
                   choices=['global', 'within'],
                   help='global: drop the worst images across templates; '
                        'within: drop them inside each template')
    p.add_argument('--fixed_tau', action='store_true',
                   help='calibrate tau once at fr=0 and reuse it '
                        '(default: re-estimate at every rate)')
    p.add_argument('--ours_label', default='confidence (ours, ripple)')
    p.add_argument('--pfe_label', default='PFE $\\sigma$ (MLS)')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pfe_dir = args.pfe_dir or args.feats_dir
    prefix = args.out_prefix or f'filter_image_quality_{args.filter_mode}'

    out = {'filter_rates': FILTER_RATES, 'far': args.far,
           'mode': args.filter_mode, 'sim_name': args.sim_name,
           'clean_only': True,
           'fixed_tau': args.fixed_tau,
           'ours': {}, 'pfe': {}, 'ours_tau': {}, 'pfe_tau': {}}
    for ds in ['ijbb', 'ijbc']:
        r = run_ds(ds, args.feats_dir, pfe_dir, args.far, args.b_theta,
                   args.filter_mode, args.sim_name, args.fixed_tau)
        if r is not None:
            for k in ['ours', 'pfe', 'ours_tau', 'pfe_tau']:
                out[k][ds] = r[k][ds]

    json_path = osp.join(args.out_dir, f'{prefix}.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[saved] {json_path}')

    if args.plot:
        plot_curves(json_path, osp.join(args.out_dir, f'{prefix}.png'),
                    args.ours_label, args.pfe_label)


if __name__ == '__main__':
    main()
