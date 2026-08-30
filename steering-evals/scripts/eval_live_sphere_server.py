#!/usr/bin/env python3
"""LIVE interactive sphere server: the running LLM drives the 3js browser UI.

Loads the model once (Qwen2-0.5B by default, weights cached), computes each
layer sphere's basis (BOS axis u, ring frame r1/r2, start state h0, topic
azimuths + member words, ring token samples), and serves:

    GET  /                      -> web/sphere_walker.html (3js page)
    GET  /api/layers            -> model + per-layer sphere descriptors
    GET  /api/state             -> current state (lat, az, top-1, topic, layer)
    POST /api/move  {"dir":..}  -> move the state (left/right/up/down), returns
                                   new state (real GPU argmax for the top-1 token)
    POST /api/reset {"layer":..}-> switch sphere layer + reset state

Every move is computed from the ACTUAL model geometry (validated formulas:
right/left steer toward ring azimuth +-30 deg, up/down along the meridian).

Run:    python3 eval_live_sphere_server.py [--model M] [--port P]   (bind 127.0.0.1)
Test:   curl localhost:8790/api/state ; curl -X POST localhost:8790/api/move -d '{"dir":"right"}'
"""
import argparse
import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

import steering_geometry_test as M
from eval_nb_quick import CLASSES

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
STEP = 2.5                       # degrees per arrow press
LAYERS = [0, 4, 9, 13, 18, 23]   # sphere layers (spread across depth)
HERE = Path(__file__).resolve().parent
WEB = HERE.parent / 'web'
PROMPT = 'Once upon a time'


def norm(v):
    return v / np.linalg.norm(v)


class Engine:
    """Holds the live model + current navigation state."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tok = M.load_model(model_name, dtype='fp16')
        W = self.model.lm_head.weight.detach().cpu().float().numpy()
        self.W = W[:self.model.config.vocab_size]
        self.Wn = self.W / np.linalg.norm(self.W, axis=1, keepdims=True)
        # topic rows
        word2id = {}
        for w in sorted({x for c in CLASSES.values() for x in c}):
            ids = self.tok(' ' + w, add_special_tokens=False).input_ids
            if len(ids) == 1:
                word2id[w] = int(ids[0])
        self.topics = {cls: np.array([word2id[w] for w in words if w in word2id])
                       for cls, words in CLASSES.items()
                       if sum(1 for w in words if w in word2id) >= 6}
        self.names = list(self.topics)
        self.members = {cls: [w for w in words if w in word2id]
                        for cls, words in CLASSES.items()}
        # forward through the prompt once (all layers)
        pid = self.tok(PROMPT, add_special_tokens=False,
                       return_tensors='pt').input_ids.to(DEV)
        with torch.no_grad():
            hid = self.model(pid, output_hidden_states=True)
        self.hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]
        # build the layer-sphere descriptors + state
        self.spheres = []
        for l in LAYERS:
            self.spheres.append(self._make_sphere(l))
        self.spi = len(LAYERS) - 1          # start on the readout sphere
        self._load_sphere(self.spi)
        self.transitions = 0
        self.trail = []

    def _make_sphere(self, l):
        u = norm(self.hidden[l + 1][0, :])
        h0 = norm(self.hidden[l + 1][-1, :])
        cent = {c: norm(self.Wn[i].mean(0) - (self.Wn[i].mean(0) @ u) * u)
                for c, i in self.topics.items()}
        C0 = np.stack(list(cent.values())) - np.stack(list(cent.values())).mean(0)
        _, eig = np.linalg.eigh(C0 @ C0.T)
        Vp = C0.T @ eig[:, -2:]
        r1, r2 = Vp[:, 0], Vp[:, 1]
        r1, r2 = norm(r1), norm(r2)
        az = {}
        for c in self.names:
            p = np.array([cent[c] @ r1, cent[c] @ r2])
            az[c] = (math.degrees(math.atan2(p[1], p[0])) + 360) % 360
        # ring token samples every 10 deg az (from the readout head)
        ring = []
        Wt = torch.as_tensor(self.W, device=DEV)
        for a in range(0, 360, 10):
            d = math.cos(math.radians(a)) * r1 + math.sin(math.radians(a)) * r2
            tid = int((Wt @ torch.as_tensor(d, device=DEV)).argmax())
            ring.append({'az': a, 'token': self.tok.decode([tid], skip_special_tokens=True).strip()})
        return {'layer': l, 'u': u, 'r1': r1, 'r2': r2, 'h0': h0,
                'cent': cent, 'az': az, 'ring': ring}

    def _load_sphere(self, spi):
        s = self.spheres[spi]
        self.spi = spi
        self.u, self.r1, self.r2, self.h0 = s['u'], s['r1'], s['r2'], s['h0']
        self.cent, self.az, self.ring = s['cent'], s['az'], s['ring']
        self.h = self.h0.copy()
        self.transitions = 0
        self.trail = []

    # --- geometry ------------------------------------------------------
    def _state_of(self):
        h, u, r1, r2 = self.h, self.u, self.r1, self.r2
        lat = math.degrees(math.acos(np.clip(u @ h, -1, 1)))
        az = (math.degrees(math.atan2(h @ r2, h @ r1)) + 360) % 360
        # top-1 token from the LIVE head (GPU argmax)
        with torch.no_grad():
            top = int((torch.as_tensor(h, device=DEV) @
                       torch.as_tensor(self.Wn, device=DEV).T).argmax())
        tok_s = self.tok.decode([top], skip_special_tokens=True).strip()
        hp = h - (h @ u) * u
        hp = hp / np.linalg.norm(hp)
        topic = min(self.names, key=lambda c: math.degrees(
            math.acos(np.clip(hp @ self.cent[c], -1, 1))))
        return {'layer': self.spheres[self.spi]['layer'], 'lat': round(lat, 1),
                'az': round(az, 1), 'token': tok_s, 'topic': topic,
                'transitions': self.transitions,
                'trail': self.trail[-64:]}

    def move(self, d):
        h = self.h
        if d in ('left', 'right'):
            phi = math.atan2(h @ self.r2, h @ self.r1)
            dphi = math.radians(30.0) * (-1.0 if d == 'right' else 1.0)
            E = math.cos(phi + dphi) * self.r1 + math.sin(phi + dphi) * self.r2
            tau = M.tangent_direction(h, E)
            self.h = norm(M.rotate_toward(h, tau, math.radians(STEP)))
        else:
            sgn = 1.0 if d == 'up' else -1.0          # up = toward pole
            tau = M.tangent_direction(h, self.u)
            self.h = norm(M.rotate_toward(h, tau, sgn * math.radians(STEP)))
        st = self._state_of()
        self.trail.append((st['az'], 90 - st['lat']))   # (az, polar)
        return st

    # --- JSON serialization for the browser ---------------------------
    def layers_json(self):
        return {'model': self.model_name,
                'layers': [{'layer': s['layer'],
                            'azimuths': {c: round(s['az'][c], 1) for c in self.names},
                            'members': self.members,
                            'ring': s['ring']} for s in self.spheres]}


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        eng = self.engine
        if self.path in ('/', '/index.html'):
            p = WEB / 'sphere_walker.html'
            if p.exists():
                body = p.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send({'error': 'web/sphere_walker.html missing'}, 404)
        elif self.path == '/api/layers':
            self._send(eng.layers_json())
        elif self.path == '/api/state':
            self._send(eng._state_of())
        else:
            self._send({'error': 'not found'}, 404)

    def do_POST(self):
        eng = self.engine
        ln = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(ln) or b'{}')
        except Exception:
            body = {}
        if self.path == '/api/move':
            d = body.get('dir', 'right')
            if d not in ('left', 'right', 'up', 'down'):
                return self._send({'error': 'bad dir'}, 400)
            return self._send(eng.move(d))
        if self.path == '/api/reset':
            if 'layer' in body:
                try:
                    spi = LAYERS.index(int(body['layer']))
                except ValueError:
                    return self._send({'error': 'bad layer'}, 400)
                eng._load_sphere(spi)
            else:
                eng._load_sphere(eng.spi)
            return self._send(eng._state_of())
        self._send({'error': 'not found'}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2-0.5B-Instruct')
    ap.add_argument('--port', type=int, default=8790)
    a = ap.parse_args()
    print(f"loading {a.model} ...")
    Handler.engine = Engine(a.model)
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    print(f"LIVE sphere ready:  http://127.0.0.1:{a.port}/")
    print("  arrow keys: walk the sphere   l: change layer   r: reset")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()