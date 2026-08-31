#!/usr/bin/env python3
"""LIVE interactive sphere server: the running LLM drives the 3js browser UI.

Loads the model once (Qwen2-0.5B by default, weights cached), computes each
layer sphere's basis (BOS axis u, ring frame r1/r2, start state h0, topic
azimuths + member words, ring token samples), and serves:

    GET  /                      -> web/sphere_walker.html (3js page)
    GET  /api/layers            -> model + prompt + per-layer sphere descriptors
    GET  /api/state             -> current state (lat, az, top-1, topic, layer)
    POST /api/move  {"dir":..}  -> move the state (left/right/up/down), returns
                                   new state (real GPU argmax for the top-1 token)
    POST /api/reset {"layer":..}-> switch sphere layer + reset state
    POST /api/prompt {"prompt":..}-> re-seed with a new prompt (rebuilds spheres)

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
        self.model, self.tok = M.load_model(model_name, dtype='fp16',
                                            attn_implementation='eager')   # eager: needed for output_attentions
        W = self.model.lm_head.weight.detach().cpu().float().numpy()
        self.W = W[:self.model.config.vocab_size]
        self.Wn = self.W / np.linalg.norm(self.W, axis=1, keepdims=True)
        self.Wt = torch.as_tensor(self.W, device=DEV)       # cached on GPU
        self.Wnt = torch.as_tensor(self.Wn, device=DEV)     # normalized head, cached
        import threading
        self._lock = threading.Lock()   # serialize engine access (transformers is not thread-safe)
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
        # forward through the prompt once (all layers) + build sphere descriptors
        self.prompt = None
        self.generation = ''
        self.base_prompt = PROMPT
        self.gen_mode = True
        self.set_prompt(PROMPT)

    def set_prompt(self, text, max_tokens=128):
        """Re-seed: run the model on a NEW base prompt, rebuild every layer sphere.
        The ring geometry (topic az + ring tokens) is context-independent (it lives in
        the LM head), so only the pole u and start state h0 re-derive; we rebuild them
        here for a clean full reset and reset the growing generation."""
        text = (text or PROMPT).strip() or PROMPT
        self.base_prompt = text
        self.generation = ''
        pid = self.tok(text, add_special_tokens=False,
                       return_tensors='pt').input_ids[:, :max_tokens].to(DEV)
        with torch.no_grad():
            hid = self.model(pid, output_hidden_states=True, output_attentions=True)
        self.hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]
        self.attns = [a[0].cpu().float().numpy() for a in hid.attentions]   # [heads, seq, seq] per layer
        self.spheres = [self._make_sphere(l) for l in LAYERS]
        self.spi = len(LAYERS) - 1          # start on the readout sphere
        self._load_sphere(self.spi)
        self.prompt = text

    def _step_context(self):
        """Forward the base prompt + generation (cheap: only u and h0 re-derive;
        the r1/r2/ring geometry is context-independent and stays cached)."""
        text = (self.base_prompt + ' ' + self.generation).strip()[:1024]
        pid = self.tok(text, add_special_tokens=False,
                       return_tensors='pt').input_ids[:, -128:].to(DEV)
        with torch.no_grad():
            hid = self.model(pid, output_hidden_states=True, output_attentions=True)
        self.hidden = [h[0].cpu().float().numpy() for h in hid.hidden_states]
        self.attns = [a[0].cpu().float().numpy() for a in hid.attentions]
        for s in self.spheres:
            l = s['layer']
            s['u'] = norm(self.hidden[l + 1][0, :])
            s['h0'] = norm(self.hidden[l + 1][-1, :])
            self._foci_for(s)      # attention changes with context
        self._load_sphere(self.spi)   # fresh start on the (updated) sphere
        return self._state_of()

    def commit_token(self):
        """Append the current top-1 token to the generation and re-seed."""
        tok_s = self._state_of()['token']
        if not tok_s or tok_s == '<unk>':
            return self._state_of()
        self.generation = (self.generation + ' ' + tok_s).strip()
        return self._step_context()
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
            tid = int((self.Wt @ torch.as_tensor(d.astype(np.float32), device=DEV)).argmax())
            ring.append({'az': a, 'token': self.tok.decode([tid], skip_special_tokens=True).strip()})
        sp = {'layer': l, 'u': u, 'r1': r1, 'r2': r2, 'h0': h0,
              'cent': cent, 'az': az, 'ring': ring}
        self._foci_for(sp)   # attention-derived mesh-refinement field
        return sp

    def _load_sphere(self, spi):
        s = self.spheres[spi]
        self.spi = spi
        self.u, self.r1, self.r2, self.h0 = s['u'], s['r1'], s['r2'], s['h0']
        self.cent, self.az, self.ring = s['cent'], s['az'], s['ring']
        self.h = self.h0.copy()
        self.transitions = 0
        self.trail = []

    def _foci_for(self, s):
        """Map this layer's last-token attention onto the sphere surface: the
        weighted centroid plus the strongest individual context foci. The mesh
        should refine (get denser) where these land."""
        l = s['layer']
        r1, r2, u = s['r1'], s['r2'], s['u']
        H = self.hidden[l + 1]                    # [seq, dim] residual stream
        hat = H / np.linalg.norm(H, axis=1, keepdims=True)
        try:
            w = self.attns[l][:, -1, :].mean(0)   # [seq] last-token -> positions, heads avg
        except (AttributeError, IndexError, TypeError):
            s['foci'] = []
            s['attn_entropy'] = 1.0
            return
        w = np.clip(w, 0, None)
        w = w / (w.sum() or 1.0)
        azs = np.degrees(np.arctan2(hat @ r2, hat @ r1)) % 360
        lats = np.degrees(np.arccos(np.clip(hat @ u, -1, 1)))
        if w.shape[0] != len(azs) or len(w) < 2:   # 1-token prompts: entropy undefined
            s['foci'] = []
            s['attn_entropy'] = 1.0
            return
        ent = float(-(w * np.log(w + 1e-12)).sum() / math.log(len(w)))
        if not math.isfinite(ent):
            ent = 1.0
        ent = float(-(w * np.log(w + 1e-12)).sum() / math.log(len(w)))
        cent = norm(np.tensordot(w, hat, axes=1))
        foci = [{'az': round((math.degrees(math.atan2(cent @ r2, cent @ r1)) + 360) % 360, 1),
                 'lat': round(math.degrees(math.acos(np.clip(cent @ u, -1, 1))), 1),
                 'w': 1.0, 'kind': 'focus'}]
        order = np.argsort(-w)[:4]                # strongest context tokens
        for j in order:
            if j == len(w) - 1:
                continue                          # skip self-attention
            foci.append({'az': round(float(azs[j]), 1),
                         'lat': round(float(lats[j]), 1),
                         'w': float(w[j]), 'kind': 'token'})
        s['foci'] = foci
        s['attn_entropy'] = round(ent, 3)
    # --- geometry ------------------------------------------------------
    def _state_of(self):
        h, u, r1, r2 = self.h, self.u, self.r1, self.r2
        lat = math.degrees(math.acos(np.clip(u @ h, -1, 1)))
        az = (math.degrees(math.atan2(h @ r2, h @ r1)) + 360) % 360
        # top-1 token from the LIVE head (GPU argmax)
        with torch.no_grad():
            top = int((torch.as_tensor(h, device=DEV, dtype=torch.float32) @ self.Wnt.T).argmax())
        tok_s = self.tok.decode([top], skip_special_tokens=True).strip()
        hp = h - (h @ u) * u
        hp = hp / np.linalg.norm(hp)
        topic = min(self.names, key=lambda c: math.degrees(
            math.acos(np.clip(hp @ self.cent[c], -1, 1))))
        # dynamic ring readout: the next-token field around the CURRENT state
        ringNow = []
        for a in range(0, 360, 30):
            ringNow.append({'az': a, 'token': self._token_at_az(a)})
        s = self.spheres[self.spi]
        return {'layer': s['layer'], 'lat': round(lat, 1),
                'az': round(az, 1), 'token': tok_s, 'topic': topic,
                'transitions': self.transitions,
                'trail': self.trail[-64:],
                'generation': self.generation, 'gen_mode': self.gen_mode,
                'prompt_seed': self.base_prompt,
                'ringNow': ringNow,
                'foci': s.get('foci', []),
                'attn_entropy': s.get('attn_entropy', 1.0)}

    def _token_at_az(self, a):
        """Top-1 token if you steered the state toward azimuth a (live head)."""
        d = math.cos(math.radians(a)) * self.r1 + math.sin(math.radians(a)) * self.r2
        tau = M.tangent_direction(self.h, d)
        v = norm(M.rotate_toward(self.h, tau, 0.2))
        tid = int((torch.as_tensor(v, device=DEV, dtype=torch.float32) @ self.Wnt.T).argmax())
        tok = self.tok.decode([tid], skip_special_tokens=True).strip()
        return tok or '\u00b7'

    def ringat(self, az):
        """Token at an arbitrary azimuth (for hover-peek)."""
        return {'az': round(float(az) % 360, 1), 'token': self._token_at_az(float(az) % 360)}

    def topk(self, k=12):
        """Top-k next-token candidates at the current state (GPU, decoded)."""
        with torch.no_grad():
            logits = torch.as_tensor(self.h, device=DEV, dtype=torch.float32) @ self.Wnt.T
            ids = torch.topk(logits, min(k, logits.shape[-1])).indices.cpu().tolist()
        toks = []
        for i in ids:
            t = self.tok.decode([i], skip_special_tokens=True).strip()
            if t and t != '<unk>' and t not in [x['token'] for x in toks]:
                toks.append({'id': int(i), 'token': t})
        return toks[:k]

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
        # moves append the top-1 token to the generation INSTANTLY (no re-seed);
        # choosing/re-seeding happens on /api/choose
        if self.gen_mode and st['token'] and st['token'] != '<unk>':
            self.generation = (self.generation + ' ' + st['token']).strip()
            st['generation'] = self.generation
        return st

    def go(self, target_az, tol=3.0, max_steps=90):
        """Auto-walk: steer the state's azimuth toward target_az (shortest way)."""
        target_az = float(target_az) % 360
        for _ in range(max_steps):
            cur = (math.degrees(math.atan2(self.h @ self.r2, self.h @ self.r1)) + 360) % 360
            d = (target_az - cur + 180.0) % 360.0 - 180.0
            if abs(d) <= tol:
                break
            self.move('right' if d < 0 else 'left')
        return self._state_of()

    # --- JSON serialization for the browser ---------------------------
    def layers_json(self):
        return {'model': self.model_name, 'prompt': self.prompt, 'generation': self.generation,
                'layers': [{'layer': s['layer'],
                            'azimuths': {c: round(s['az'][c], 1) for c in self.names},
                            'members': self.members,
                            'ring': s['ring']} for s in self.spheres]}


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *a):
        pass

    @staticmethod
    def _clean(o):
        """Recursively replace non-finite floats (nan/inf) with None so the
        JSON is always strict-parser safe on the browser side."""
        if isinstance(o, dict):
            return {k: Handler._clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [Handler._clean(x) for x in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    def _send(self, obj, code=200):
        try:
            body = json.dumps(Handler._clean(obj)).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass          # client vanished mid-write; never poison the next request

    def do_GET(self):
        with self.engine._lock:
            self._do_GET()

    def _do_GET(self):
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
        elif self.path == '/api/layers':
            self._send(eng.layers_json())
        elif self.path == '/api/state':
            self._send(eng._state_of())
        elif self.path.startswith('/models/'):
            name = Path(self.path.split('?')[0]).name
            p = WEB / 'models' / name
            if p.exists():
                ctype = 'model/gltf-binary' if p.suffix == '.glb' else 'text/plain'
                body = p.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send({'error': 'model missing: ' + name}, 404)
        else:
            self._send({'error': 'not found'}, 404)

    def do_POST(self):
        with self.engine._lock:
            self._do_POST()

    def _do_POST(self):
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
        if self.path == '/api/gen':
            mode = bool(body.get('mode', True))
            eng.gen_mode = mode
            return self._send(eng._state_of())
        if self.path == '/api/go':
            try:
                az = float(body.get('az'))
            except (TypeError, ValueError):
                return self._send({'error': 'bad az'}, 400)
            return self._send(eng.go(az))
        if self.path == '/api/prompt':
            text = (body.get('prompt') or '').strip()
            if not text:
                return self._send({'error': 'empty prompt'}, 400)
            eng.set_prompt(text)
            return self._send({'prompt': eng.prompt, 'state': eng._state_of()})
        if self.path == '/api/reset':
            if 'layer' in body:
                try:
                    spi = LAYERS.index(int(body['layer']))
                except ValueError:
                    return self._send({'error': 'bad layer'}, 400)
            else:
                spi = eng.spi
            eng._load_sphere(spi)   # reset position on the sphere
            eng.generation = ''      # fresh start
            return self._send(eng._state_of())   # NOTE: no re-seed -> instant
        if self.path == '/api/topk':
            try:
                k = int(body.get('k', 12))
            except (TypeError, ValueError):
                k = 12
            return self._send({'tokens': eng.topk(k)})
        if self.path == '/api/choose':
            tid = body.get('token')
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                return self._send({'error': 'bad token'}, 400)
            tok_s = eng.tok.decode([tid], skip_special_tokens=True).strip()
            if not tok_s or tok_s == '<unk>':
                return self._send({'error': 'empty token'}, 400)
            eng.generation = (eng.generation + ' ' + tok_s).strip()
            st = eng._step_context()
            return self._send(st)
        if self.path == '/api/ringat':
            try:
                az = float(body.get('az'))
            except (TypeError, ValueError):
                return self._send({'error': 'bad az'}, 400)
            return self._send(eng.ringat(az))
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
    print("  arrow keys: walk the sphere   l: change layer   r: reset   prompt box: re-seed")
    print("  arrow keys: walk the sphere   l: change layer   r: reset")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()