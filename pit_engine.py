"""Reverse-engineer self-consistent tokens ("pits") from transformer weights,
then encode those pits into data so that any processing path is forced into a
fixed point.

Pipeline
--------
1. Load the model, tokenizer, and LM head.
2. For every token T compute the self-consistency score
       s(T) = softmax(W . h_T)[T],   h_T = model(embed(T))
   and the hidden-state trajectory cosine cos(h_{n+1}, h_n).
3. A token is a *pit* if it is permanent (predicts itself for N consecutive
   steps) and its trajectory is stable (cos -> 1).
4. For each pit, find the minimal trigger: the shortest repeat count of the
   token's text that reliably enters the fixed point.
5. Encode pits into a byte/text stream by framing every chunk with pit
   triggers, so that any truncation boundary lands inside a pit basin.

Usage
-----
    python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --scan
    python pit_engine.py --model Qwen/Qwen2.5-7B-Instruct --encode data.txt
"""

import argparse
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Reverse engineering
# --------------------------------------------------------------------------- #
class PitReverseEngineer:
    def __init__(self, model_name, dtype=torch.bfloat16):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=dtype)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.d = self.model.config.hidden_size
        self.sr = math.sqrt(self.d)
        self.lm_w = self.model.lm_head.weight.float().cpu()
        self.lm_n = self.lm_w / (self.lm_w.norm(dim=1, keepdim=True) + 1e-12)
        self.vocab = self.lm_w.shape[0]

    # -- single-token forward pass ------------------------------------------
    def _h_and_logits(self, tid):
        t = self.tok.decode([tid])
        if len(t.strip()) == 0:
            return None, None
        inp = self.tok(t, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        logits = self.model.lm_head(
            h.unsqueeze(0).to(device=self.model.device,
                              dtype=torch.bfloat16)).float().cpu()[0]
        return h, logits

    def self_consistency(self, tid):
        """Return (s(T), cos(h_T, w_T), predicted_tid)."""
        h, logits = self._h_and_logits(tid)
        if h is None:
            return 0.0, 0.0, None
        p = torch.softmax(logits, dim=0)[tid].item()
        hn = h / h.norm()
        cos_hw = (hn * self.lm_n[tid]).sum().item()
        return p, cos_hw, logits.argmax().item()

    # -- permanence test -----------------------------------------------------
    def permanence(self, tid, repeats=3, steps=15):
        """Return (locked_steps, min_cos). locked_steps counts how many of the
        `steps` generations predict tid; min_cos is the smallest trajectory
        cosine across those steps."""
        t = self.tok.decode([tid])
        inp = self.tok(t * repeats, return_tensors="pt").to(self.model.device)
        locked, h_last, min_cos = 0, None, 1.0
        for _ in range(steps):
            with torch.no_grad():
                out = self.model(**inp, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu()
            hn = h / h.norm()
            logits = self.model.lm_head(
                hn.unsqueeze(0).to(device=self.model.device,
                                   dtype=torch.bfloat16)).float().cpu()[0]
            nid = logits.argmax().item()
            if nid == tid:
                locked += 1
            if h_last is not None:
                min_cos = min(min_cos, (hn * h_last).sum().item())
            h_last = hn
            inp = {k: torch.cat([v, torch.tensor([[nid]], device=v.device)],
                                dim=1) for k, v in inp.items()}
        return locked, min_cos

    # -- minimal trigger search ---------------------------------------------
    def find_trigger(self, tid, max_repeat=8):
        """Shortest repeat count n such that feeding text*n generates tid for
        >= 3 of the next 5 steps."""
        t = self.tok.decode([tid])
        for n in range(1, max_repeat + 1):
            inp = self.tok(t * n, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=5,
                                          do_sample=False)
            gen = out[0][inp.input_ids.shape[1]:]
            if (gen == tid).sum().item() >= 3:
                return n
        return None

    # -- isolation: theoretical ceiling on s(T) -----------------------------
    def nearest_neighbor_cosines(self, chunk=8000):
        """For each token, the cosine to its nearest LM-head neighbor. This
        upper-bounds the achievable self-consistency: a token whose w_T is
        close to another w_i can never reach s(T)=1."""
        nn = torch.full((self.vocab,), 1.0)
        for start in range(0, self.vocab, chunk):
            end = min(start + chunk, self.vocab)
            block = self.lm_n[start:end]
            sims = block @ self.lm_n.T          # (chunk, vocab)
            for i in range(sims.shape[0]):
                sims[i, start + i] = -1.0       # exclude self
            nn[start:end] = sims.max(dim=1).values
        return nn

    # -- full scan -----------------------------------------------------------
    def find_all_pits(self, p_thresh=0.01, cos_thresh=0.04,
                      stride=1, perma_steps=15):
        """Scan the vocabulary, return a list of pit dicts sorted by s(T)."""
        pits = []
        for tid in range(0, self.vocab, stride):
            t = self.tok.decode([tid])
            if len(t.strip()) == 0 or len(t) > 24:
                continue
            p, cos_hw, pred = self.self_consistency(tid)
            if p < p_thresh and cos_hw < cos_thresh:
                continue
            locked, min_cos = self.permanence(tid, steps=perma_steps) \
                if pred == tid else (0, 0.0)
            pits.append({
                "tid": tid, "text": t, "s": round(p, 4),
                "cos_hw": round(cos_hw, 4), "pred": pred,
                "locked": locked, "min_cos": round(min_cos, 6),
            })
            if tid % 50000 == 0:
                print(f"  scanned {tid}/{self.vocab}, "
                      f"candidates {len(pits)}", flush=True)
        pits.sort(key=lambda x: -x["s"])
        return pits


# --------------------------------------------------------------------------- #
# Encoding pits into data
# --------------------------------------------------------------------------- #
class PitEncoder:
    def __init__(self, tok, pits):
        self.tok = tok
        # keep only permanent pits, and attach their trigger text
        self.pits = []
        for p in pits:
            if p.get("locked", 0) >= 14:
                trigger = self.tok.decode([p["tid"]]) * p.get("trigger", 3)
                self.pits.append({"tid": p["tid"], "text": p["text"],
                                  "s": p["s"], "trigger": trigger})

    def frame(self, chunk, pit_index=0):
        """Wrap one data chunk between pit triggers so a generation boundary
        at either end falls into the pit basin."""
        pit = self.pits[pit_index % len(self.pits)]
        return pit["trigger"] + " " + chunk + " " + pit["trigger"]

    def make_unavoidable(self, chunks, redundancy=2):
        """Interleave multiple distinct pits so that ANY truncation point
        (start, end, or any chunk boundary) terminates inside a pit trigger.

        Layout for redundancy r and pits P0..Pk:
            P0 [chunk0] P1 [chunk1] P2 [chunk2] P0 ...
        every boundary carries a pit trigger; distinct pits alternate so a
        single pit failing does not free the stream.
        """
        out = []
        for i, c in enumerate(chunks):
            pit = self.pits[i % len(self.pits)]
            out.append(pit["trigger"] + " " + c + " " + pit["trigger"])
        return "\n".join(out)

    def encode_bytes(self, raw: bytes, chunk_size=64, redundancy=2):
        """Encode arbitrary bytes as a pit-framed text stream."""
        text = raw.decode("utf-8", errors="ignore")
        chunks = [text[i:i + chunk_size]
                  for i in range(0, len(text), chunk_size)]
        return self.make_unavoidable(chunks, redundancy)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--encode", type=str, default=None,
                    help="path to a text file to pit-encode")
    ap.add_argument("--out", type=str, default="pit_encoded.txt")
    args = ap.parse_args()

    eng = PitReverseEngineer(args.model)

    if args.scan:
        print(f"Scanning {eng.vocab} tokens...", flush=True)
        pits = eng.find_all_pits()
        perm = [p for p in pits if p["locked"] >= 14]
        print(f"\n{len(perm)} permanent pits:", flush=True)
        for p in perm:
            trig = eng.find_trigger(p["tid"])
            p["trigger"] = trig or 3
            print(f"  [{p['text'][:12]:12s}] tid={p['tid']:6d} "
                  f"s={p['s']} locked={p['locked']} "
                  f"trigger_repeat={p['trigger']}", flush=True)

    if args.encode:
        pits = eng.find_all_pits()
        perm = [p for p in pits if p["locked"] >= 14]
        for p in perm:
            p["trigger"] = eng.find_trigger(p["tid"]) or 3
        enc = PitEncoder(eng.tok, perm)
        with open(args.encode, "rb") as f:
            data = f.read()
        result = enc.encode_bytes(data)
        with open(args.out, "w") as f:
            f.write(result)
        print(f"Encoded {len(data)} bytes -> {args.out} "
              f"({len(perm)} pits used)", flush=True)


if __name__ == "__main__":
    main()