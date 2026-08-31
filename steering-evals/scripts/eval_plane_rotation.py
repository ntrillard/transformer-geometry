#!/usr/bin/env python3
"""eval_plane_rotation.py — BIG LEAP #4: trace the 2D rotation arc.

Follows 05d9225 (universal 3D plane: axis1=sink, axis2=native). Trace the
state's coordinates in the (sink, native) basis per layer -> the arc the
model draws. Hypothesis: plunge along sink (monotone negative native-coord
growth), then at the readout ROTATE onto the native axis at reduced scale.

One forward, 27 captures, ~4s. Per layer: project v_l onto sink-hat and
native-hat -> (a_l, b_l). Print the arc + the angle from the sink axis.
The READOUT collapsed state should sit at (small a, large b relative to
its norm) = the rotation.

Run: timeout 60 python3 -u eval_plane_rotation.py  # GEMMA-3-1B
"""
import sys
import time

import numpy as np
import torch

import steering_geometry_test as SGT

MODEL = 'google/gemma-3-1b-it'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PROMPT = 'For dinner I made'


def main():
    t0 = time.time()
    model, tok = SGT.load_model(MODEL, dtype='fp16')
    lm_w = model.lm_head.weight
    NL = model.config.num_hidden_layers

    ids = tok(PROMPT, add_special_tokens=False,
              return_tensors='pt').input_ids.to(DEV)
    caps = {}

    def mk(li):
        def h(m, i, o):
            caps[li] = o[0, -1, :].float()
        return h

    hooks = [model.model.layers[li].register_forward_hook(mk(li))
             for li in range(NL)]
    hooks.append(model.model.norm.register_forward_hook(mk('f')))
    with torch.no_grad():
        L0 = model(ids).logits[0, -1].float()
    for h in hooks:
        h.remove()
    native = int(L0.argmax())
    Wn = lm_w[native].detach().float().cpu().numpy().astype(np.float32)
    vf = caps['f'].cpu().numpy()

    V = np.stack([caps[li].cpu().numpy() for li in range(NL)] +
                 [caps['f'].cpu().numpy()])

    # basis: sink dir (v22-v18), native dir (lm_head row of native)
    sink = V[22] - V[18]
    sink_h = sink / (np.linalg.norm(sink) + 1e-12)
    nat_h = Wn / (np.linalg.norm(Wn) + 1e-12)
    # orthogonalize native against sink (2D orthonormal basis)
    nat_h2 = nat_h - (nat_h @ sink_h) * sink_h
    nat_h2 = nat_h2 / (np.linalg.norm(nat_h2) + 1e-12)

    print(f"[{MODEL}] {PROMPT!r} native={tok.decode([native])!r}")
    print(f"  basis: sink_h (L22-L18), native_h (off-sink part)")
    print(f"  {'layer':>6} {'a(sink)':>9} {'b(native)':>10} {'angle':>7} "
          f"{'norm':>8}", flush=True)
    for li in range(NL + 1):
        v = V[li]
        a = float(v @ sink_h)
        b = float(v @ nat_h2)
        nrm = float(np.linalg.norm(v))
        ang = float(np.arctan2(b, a) * 180 / np.pi)
        tag = ' final' if li == NL else f' L{li:>2}'
        print(f"{tag:>7} {a:>+9.1f} {b:>+10.1f} {ang:>+7.1f} {nrm:>8.1f}",
              flush=True)

    # the readout rotation: angle(sink->native) sweep of the LAST layers
    print(f"\n  arc summary: L18->final angle "
          f"{np.arctan2(V[18] @ nat_h2, V[18] @ sink_h) * 180 / np.pi:+.1f} "
          f"-> {np.arctan2(V[-1] @ nat_h2, V[-1] @ sink_h) * 180 / np.pi:+.1f} deg")
    print(f"  native in-plane frac = "
          f"{np.linalg.norm((sink_h @ Wn) * sink_h + (nat_h2 @ Wn) * nat_h2) / (np.linalg.norm(Wn) + 1e-12):.3f}")
    print(f"[{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()