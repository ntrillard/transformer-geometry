"""multiverse_authorship.py — FAST: is the two-phase authorship pattern
(attention plants the topic mid-stream, deep MLPs write it) universal across
topics, and WHICH attention heads do the planting?

Phase A: project the SAME captured residual deltas (1 forward, prompt
'For dinner I made') onto all 6 topic chord centers + 3 random unit vectors:
plant-attn = sum(attn·C, layers 5..16), write = sum(mlp·C, 17..23), final spike.

Phase B: manual Qwen2 GQA attention on the plant layers (5,6,7) -> per-head
residual contribution projected on the food direction. Validated against the
hook-captured attn delta (rel-err).

~10s.  Run: python3 multiverse_authorship.py"""
import math

import numpy as np
import torch

import steering_geometry_test as M
from eval_chord_steering import CLASSES
from multiverse_lab import chord_summary

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    model, tok = M.load_model('Qwen/Qwen2-0.5B-Instruct', dtype='fp16')
    W = model.lm_head.weight.detach().cpu().float().numpy()
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    NL = model.config.num_hidden_layers

    word2id = {}
    for w in sorted({x for c in CLASSES.values() for x in c}):
        ids = tok(' ' + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            word2id[w] = int(ids[0])
    avail = {c: [w for w in words if w in word2id] for c, words in CLASSES.items()}
    Cs = {}
    for cls, words in avail.items():
        ids = np.array([word2id[w] for w in words[:5]])
        C, _, _ = chord_summary(ids, Wn)
        Cs[cls] = C
    rng = np.random.default_rng(0)
    for k in range(3):
        Cs[f'rand{k}'] = rng.standard_normal(W.shape[1])
        Cs[f'rand{k}'] /= np.linalg.norm(Cs[f'rand{k}'])

    def capture(prompt):
        deltas = {'attn': [], 'mlp': []}
        handles = []
        for l in range(NL):
            lay = model.model.layers[l]
            handles.append(lay.self_attn.register_forward_hook(
                lambda m, i, o, k='attn': deltas[k].append(
                    (o[0] if isinstance(o, tuple) else o).detach().float())))
            handles.append(lay.mlp.register_forward_hook(
                lambda m, i, o, k='mlp': deltas[k].append(o.detach().float())))
        pid = tok(prompt, add_special_tokens=False,
                  return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            hid = model(pid, output_hidden_states=True)
        for h in handles:
            h.remove()
        hs = [h[0, -1].float().cpu() for h in hid.hidden_states]
        ad = torch.stack([d[0, -1] for d in deltas['attn']]).cpu().float().numpy()
        md = torch.stack([d[0, -1] for d in deltas['mlp']]).cpu().float().numpy()
        return hs, ad, md

    print("== PHASE A: per-target authorship (prompt 'For dinner I made') ==")
    hs, ad, md = capture("For dinner I made")
    print(f"{'target':>8} {'plant(5-16)':>12} {'write(17-23)':>12} "
          f"{'finalMLP':>9} {'h23pre·C':>9}")
    print(f"{'':>8} {'attn|mlp':>16} {'attn|mlp':>16}")
    for name, C in Cs.items():
        pa = ad[5:17] @ C            # plant-phase attn
        pm = md[5:17] @ C
        wa = ad[17:24] @ C           # write-phase attn
        wm = md[17:24] @ C
        fm = md[23] @ C              # final-layer mlp spike
        hpre = hs[23] + ad[23] + md[23]
        print(f"{name:>8} {pa.sum():+7.2f}|{pm.sum():+7.2f} "
              f"{wa.sum():+7.2f}|{wm.sum():+7.2f} {fm:+9.2f} {float(hpre @ C):+9.2f}")

    # second prompt control
    print("\n== PHASE A2: prompt control ('The recipe calls for'), food/color/rand ==")
    hs2, ad2, md2 = capture("The recipe calls for")
    for name in ('food', 'color', 'rand0'):
        C = Cs[name]
        pa = ad2[5:17] @ C
        pm = md2[5:17] @ C
        wm = md2[17:24] @ C
        print(f"{name:>8} plant attn {pa.sum():+7.2f} mlp {pm.sum():+7.2f}  "
              f"write mlp {wm.sum():+7.2f}")

    # ---- PHASE B: head-level contributors on the plant layers ----
    print("\n== PHASE B: per-head planting of food (layers 5-7) ==")
    C = Cs['food']
    for l in (5, 6, 7):
        attn = model.model.layers[l].self_attn
        hdim = model.config.hidden_size // model.config.num_attention_heads
        nh = model.config.num_attention_heads
        nkv = model.config.num_key_value_heads
        # fresh forward keeping EVERY token for the plant layers
        pidf = tok("For dinner I made", add_special_tokens=False,
                   return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            hf = model(pidf, output_hidden_states=True).hidden_states
        x = hf[l][0].float().to(DEV)   # [seq, dim] full tokens
        with torch.no_grad():
            q = attn.q_proj(x.to(model.dtype)).float()     # [seq, nh*hdim]
            k = attn.k_proj(x.to(model.dtype)).float()
            v = attn.v_proj(x.to(model.dtype)).float()
        seq = x.shape[0]
        q = q.view(seq, nh, hdim)      # [seq, nh, hdim]
        k0 = k.view(seq, nkv, hdim)    # [seq, nkv, hdim]
        v0 = v.view(seq, nkv, hdim)
        # --- manual RoPE (half-split, GPT-NeoX style, matches Qwen2) ---
        inv = 1.0 / (10000 ** (torch.arange(0, hdim, 2, device=DEV).float() / hdim))
        freqs = torch.outer(torch.arange(seq, device=DEV).float(), inv)   # [seq, hdim//2]
        cos_c, sin_c = freqs.cos(), freqs.sin()   # shared cos/sin for both halves (Qwen style)

        def rope(xx):                     # xx [seq, H, hdim] -> rotated
            a, b = xx[..., :hdim // 2], xx[..., hdim // 2:]
            return torch.cat((a * cos_c[:, None, :] - b * sin_c[:, None, :],
                              b * cos_c[:, None, :] + a * sin_c[:, None, :]), -1)
        q = rope(q)
        k0 = rope(k0)
        # GQA: expand kv heads across query groups
        k = k0.repeat_interleave(nh // nkv, dim=1).transpose(0, 1)   # [nh, seq, hdim]
        v = v0.repeat_interleave(nh // nkv, dim=1).transpose(0, 1)
        q = q.transpose(0, 1)
        sc = q @ k.transpose(-2, -1) / math.sqrt(hdim)
        mask = torch.triu(torch.full((seq, seq), float('-inf'), device=DEV), 1)
        P = torch.softmax(sc + mask, dim=-1)    # [nh, seq, seq]
        perh = torch.einsum('hij,hjd->hid', P, v)   # [nh, seq, hdim]
        Wo = attn.o_proj.weight
        # residual contrib per head = Wo[:, h*hdim:(h+1)*hdim] @ ctx[h, -1]
        contribs = torch.stack([
            Wo[:, h * hdim:(h + 1) * hdim].float() @ perh[h, -1].float() for h in range(nh)])
        contribC = (contribs @ torch.as_tensor(C, device=DEV)).detach().cpu().numpy()
        # validation vs hook-captured attn delta
        capL = ad[l]
        rel = float(np.linalg.norm(contribs.sum(0).detach().cpu().numpy() - capL)) / float(
            np.linalg.norm(capL))
        top = np.argsort(-contribC)[:4]
        print(f"  L{l} (rel-err {rel:.3f}) top heads: " +
              ", ".join(f"h{h}:{contribC[h]:+.3f}" for h in top))
        # uniform-attention control: unchanged top heads => contributions are set
        # by the head's WRITE direction (o_proj @ value), not its reading
        Pu = torch.full_like(P, 1.0 / seq)
        perh_u = torch.einsum('hij,hjd->hid', Pu, v)
        cU = torch.stack([
            Wo[:, h * hdim:(h + 1) * hdim].float() @ perh_u[h, -1].float()
            for h in range(nh)]).detach().cpu().numpy() @ C
        top_u = np.argsort(-cU)[:4]
        same = set(top[:4]) == set(top_u[:4])
        print(f"  L{l} uniform-attn top: " +
              ", ".join(f"h{h}:{cU[h]:+.3f}" for h in top_u) +
              f"  same-as-real={same}")


if __name__ == "__main__":
    main()