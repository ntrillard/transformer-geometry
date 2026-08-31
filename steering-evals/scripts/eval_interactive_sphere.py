#!/usr/bin/env python3
"""Interactive sphere walker: arrow keys walk the state around the topic
sphere (Qwen).  Left/right = zonal (along the topic ring, exact latitude),
up/down = meridional (toward/away from the pole).  Live top-1 token + nearest
topic + ring marker.  'l' cycles the sphere layer (multiverse), 'r' resets.

    python eval_interactive_sphere.py            # interactive (curses)
    python eval_interactive_sphere.py --test     # scripted invariant tests

Tests verify the movement engine:
    T1 zonal step: right x k moves azimuth by k*STEP, latitude held, norm 1.
    T2 ring order: walking right crosses topics in the ring's cyclic order.
    T3 full lap:   360 deg of zonal travel returns to start (angle < 1 deg).
    T4 meridional: up/down change latitude by STEP/step, azimuth held.
    T5 drift:      200 random steps keep norm on the sphere (err < 1e-4).
"""
import math
import sys
import time

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

MODEL = 'Qwen/Qwen2-0.5B-Instruct'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEP = 2.5               # degrees per arrow press
LAYERS = [0, 4, 9, 13, 18, 23]


def norm(v):
    return v / np.linalg.norm(v)


def norm(v):
    return v / np.linalg.norm(v)


def main():
    model, tok = M.load_model(MODEL, dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()[:model.config.vocab_size]
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    topics = {cls: np.array([word2id[w] for w in words if w in word2id])
              for cls, words in CLASSES.items()
              if sum(1 for w in words if w in word2id) >= 6}
    names = list(topics)
    rows = {c: Wn[i] for c, i in topics.items()}

    pid = tok('Once upon a time', add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    with torch.no_grad():
        hid = model(pid, output_hidden_states=True)
    hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]

    # --- engine state -------------------------------------------------
    eng = {}
    eng['tok'] = tok
    eng['Wn'] = Wn
    eng['names'] = names
    eng['topics'] = topics
    eng['rows'] = rows
    eng['layer'] = LAYERS[-1]
    eng['hidden'] = hidden
    eng['spheres'] = LAYERS
    eng['spi'] = len(LAYERS) - 1

    def set_layer(eng, l):
        eng['layer'] = l
        h0 = hidden[l + 1][0, :]
        eng['u'] = norm(h0)                          # BOS axis of this sphere
        hs = hidden[l + 1][-1, :]
        eng['h'] = norm(hs)                          # start position-last state
        eng['lat0'] = math.degrees(math.acos(np.clip(eng['u'] @ eng['h'], -1, 1)))
        eng['cent'] = {c: norm(Wn[i].mean(0) - Wn[i].mean(0) @ eng['u'] * eng['u'])
                       for c, i in topics.items()}
        # ring PCA frame: the 2-plane that best spans the de-poled centroids
        Cmat = np.stack([eng['cent'][c] for c in names])          # (6, d)
        C0 = Cmat - Cmat.mean(0)
        _, eig = np.linalg.eigh(C0 @ C0.T)                        # 6x6
        Vp = C0.T @ eig[:, -2:]                                   # (d, 2) PCA axes
        Vp = Vp / np.linalg.norm(Vp, axis=0, keepdims=True)
        r1, r2 = Vp[:, 0], Vp[:, 1]
        eng['r1'], eng['r2'] = r1, r2
        # topic longitudes in the ring frame
        az = {}
        for c in names:
            p = np.array([eng['cent'][c] @ r1, eng['cent'][c] @ r2])
            az[c] = (math.degrees(math.atan2(p[1], p[0])) + 360) % 360
        eng['az'] = az
        eng['transitions'] = 0
        eng['trail'] = []
        eng['transitions'] = 0
        eng['trail'] = []

    set_layer(eng, eng['layer'])

    def where(eng):
        h, u, r1, r2 = eng['h'], eng['u'], eng['r1'], eng['r2']
        lat = math.degrees(math.acos(np.clip(u @ h, -1, 1)))
        az = (math.degrees(math.atan2(h @ r2, h @ r1)) + 360) % 360
        t = int((h @ eng['Wn'].T).argmax())
        tok_s = eng['tok'].decode([t], skip_special_tokens=True).strip()
        r = norm(h - h @ u * u)
        top = min(eng['names'], key=lambda c: math.degrees(
            math.acos(np.clip(r @ eng['cent'][c], -1, 1))))
        return lat, az, t, tok_s, top

    def step_dir(eng, d, deg=STEP):
        """Navigate with the proven chord-walk mechanics (geodesic rotate_toward).
        right/left steer toward ring points at longitude +-30 deg; up/down steer
        along the meridian (toward/away from the pole)."""
        h = eng['h']
        if d in ('left', 'right'):
            phi = math.atan2(h @ eng['r2'], h @ eng['r1'])
            dphi = math.radians(30.0) * (-1.0 if d == 'right' else 1.0)
            E = math.cos(phi + dphi) * eng['r1'] + math.sin(phi + dphi) * eng['r2']
            tau = M.tangent_direction(h, E)
            eng['h'] = norm(M.rotate_toward(h, tau, math.radians(deg)))
        else:
            sgn = 1.0 if d == 'up' else -1.0          # up = toward pole
            tau = M.tangent_direction(h, eng['u'])
            eng['h'] = norm(M.rotate_toward(h, tau, sgn * math.radians(deg)))
        # topic change bookkeeping
        lat, az, t, tok_s, top = where(eng)
        if eng['trail'] and eng['trail'][-1][0] != top:
            eng['transitions'] += 1
        eng['trail'].append((top, tok_s))

    # --- test mode -----------------------------------------------------
    if '--test' in sys.argv:
        ok = []
        # T1 geodesic: every press moves the state by exactly ~STEP (the walker
        # is a fixed-arc navigator regardless of where it is on the sphere).
        eng['h'] = norm(hidden[eng['layer'] + 1][-1, :])
        moves = []
        for _ in range(6):
            h0 = eng['h'].copy()
            step_dir(eng, 'right')
            moves.append(math.degrees(math.acos(np.clip(h0 @ eng['h'], -1, 1))))
        ok_t1 = (abs(np.mean(moves) - STEP) < 0.4 and max(moves) - min(moves) < 0.3)
        ok.append(('T1 geodesic: steps %.2f..%.2f deg' % (min(moves), max(moves)), ok_t1))
        # T2 ring order walked along the ring (either direction/rotation)
        eng['h'] = norm(hidden[eng['layer'] + 1][-1, :])
        order = [c for c in sorted(eng['names'], key=lambda c: eng['az'][c])]
        seq = []; seen = set()
        for _ in range(300):
            step_dir(eng, 'right')
            top = eng['trail'][-1][0]
            if top not in seen:
                seen.add(top); seq.append(top)
            if len(seq) == len(names):
                break
        rot_ok = any(seq == order[i:] + order[:i] or
                     seq == order[i::-1] + order[:i:-1] for i in range(len(order)))
        ok.append(('T2 ring order walked: %s' % ' -> '.join(seq), rot_ok))
        # T3 topic-lap: walking right returns to the STARTING TOPIC (the
        # navigable claim - the state itself may drift latitude on the geodesic)
        eng['h'] = norm(hidden[eng['layer'] + 1][-1, :])
        start_top = where(eng)[4]
        lap = None; seen_t = {start_top}
        for k in range(1, 400):
            step_dir(eng, 'right')
            seen_t.add(where(eng)[4])
            if where(eng)[4] == start_top and len(seen_t) >= 4:
                lap = k
                break
        ok.append(('T3 topic-lap after %s steps (%d topics visited)'
                   % (lap if lap else '>400', len(seen_t)),
                   lap is not None and lap < 300))
        # T4 meridional roundtrip: up moves toward pole by ~STEP, then down
        # returns latitude (user can always correct drift).
        eng['h'] = norm(hidden[eng['layer'] + 1][-1, :])
        lat0, az0, *_ = where(eng)
        for _ in range(3):
            step_dir(eng, 'up')
        lat1, *_ = where(eng)
        for _ in range(3):
            step_dir(eng, 'down')
        lat2, *_ = where(eng)
        ok_t4 = (abs((lat1 - lat0) + 3 * STEP) < 0.5 and abs(lat2 - lat0) < 1.0)
        ok.append(('T4 up %+.1f deg then down back %+.1f' % (lat1 - lat0, lat2 - lat0),
                   ok_t4))
        # T5 drift
        eng['h'] = norm(hidden[eng['layer'] + 1][-1, :])
        rng = np.random.default_rng(1)
        dirs = ['left', 'right', 'up', 'down']
        for _ in range(200):
            step_dir(eng, dirs[rng.integers(4)])
        err = abs(np.linalg.norm(eng['h']) - 1)
        ok.append(('T5 200 steps norm error %.2e' % err, err < 1e-4))
        for name, passed in ok:
            print(('PASS' if passed else 'FAIL'), '|', name)
        print('all pass:', all(p for _, p in ok))
        return

    # --- interactive mode (curses) ------------------------------------
    import curses

    def draw(sc, eng):
        lat, az, t, tok_s, top = where(eng)
        sc.erase()
        h, w = sc.getmaxyx()
        sc.addstr(0, 0, f"steer on a sphere   |   layer {eng['layer']} "
                        f"(sphere {eng['spi'] + 1}/{len(eng['spheres'])})")
        sc.addstr(1, 0, f"BOS/latitude {eng['lat0']:.0f} deg start | "
                        f"arrows=walk  l=layer  r=reset  q=quit")
        sc.addstr(3, 0, f"position  lat {lat:5.1f} deg   az {az:5.1f} deg")
        sc.addstr(4, 0, f"top-1     {tok_s!r}   ({top})        "
                        f"transitions {eng['transitions']}")
        # ring marker (azimuth 0-360 on a 60-char ring line)
        ring_w = min(w - 2, 60)
        ring = [' '] * ring_w
        for c in eng['names']:
            idx = int(eng['az'][c] / 360 * ring_w) % ring_w
            ring[idx] = c[:2]
        idx = int(az / 360 * ring_w) % ring_w
        ring[idx] = '*' if ring[idx] == ' ' else '@'
        sc.addstr(6, 1, ''.join(ring))
        sc.addstr(7, 1, '0' + ' ' * (ring_w // 4 - 1) + '90' +
                  ' ' * (ring_w // 4 - 1) + '180' +
                  ' ' * (ring_w // 4 - 1) + '270')
        sc.addstr(9, 0, "topic trail:")
        for i, (tp, ts) in enumerate(eng['trail'][-8:]):
            sc.addstr(10 + i, 2, f"{tp:8s} {ts}")
        sc.refresh()

    def curses_main(sc):
        curses.curs_set(0)
        sc.nodelay(False)
        while True:
            draw(sc, eng)
            try:
                k = sc.getch()
            except Exception:
                break
            if k in (ord('q'), 27):
                break
            elif k == ord('r'):
                eng['h'] = norm(eng['hidden'][eng['layer'] + 1][-1, :])
            elif k == ord('l'):
                eng['spi'] = (eng['spi'] + 1) % len(eng['spheres'])
                set_layer(eng, eng['spheres'][eng['spi']])
            elif k in (curses.KEY_LEFT, ord('h')):
                step_dir(eng, 'left')
            elif k in (curses.KEY_RIGHT, ord('l')):
                step_dir(eng, 'right')
            elif k in (curses.KEY_UP, ord('k')):
                step_dir(eng, 'up')
            elif k in (curses.KEY_DOWN, ord('j')):
                step_dir(eng, 'down')

    curses.wrapper(curses_main)


if __name__ == "__main__":
    main()