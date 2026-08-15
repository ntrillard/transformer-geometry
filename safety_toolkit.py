#!/usr/bin/env python3
"""
Transformer Safety Toolkit — N. Trillard (2026)

Geometric safety tools for transformer language models.

Based on the geometric theory of transformer dynamics:
  - Hidden states live on a sphere of radius sqrt(d)
  - Attention creates geometric contraction (convex combinations on sphere)
  - λ ≈ 0.05 is the healthy Lyapunov exponent
  - Sphere steering enables geometric output control

Tools:
  1. λ health check — 1-second model diagnostic
  2. Fine-tuning monitor — track λ during training
  3. Sphere steer-away — geometric suppression of harmful outputs
  4. Stability report — full per-zone diagnostic
"""

import torch
import torch.nn.functional as F
import math
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class SafetyToolkit:
    """Geometric safety tools for any HF transformer."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.d = self.model.config.hidden_size
        self.sphere_r = math.sqrt(self.d)
        self.nL = len(self.model.model.layers)
        self.lm_head_w = self.model.lm_head.weight.float().cpu()
        self._cached_ref = None  # Cache for health check

    # ─── λ Health Check ─────────────────────────────────────────

    def health_check(self, text: str = "A store has 120 apples, sells 45. How many left?",
                     trials: int = 3) -> dict:
        """Measure λ in the computation zone. Returns λ and health verdict."""
        templ = self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True
        )
        inp = self.tok(templ, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ref = self.model(**inp, output_hidden_states=True)
        embeds = self.model.model.embed_tokens(inp.input_ids)

        l_sum = 0
        for _ in range(trials):
            noise = torch.randn_like(embeds) * 1e-4
            with torch.no_grad():
                out = self.model(inputs_embeds=embeds + noise, output_hidden_states=True)
            lyaps = []
            for l in range(self.nL // 3, 2 * self.nL // 3):
                di = (out.hidden_states[l][0] - out_ref.hidden_states[l][0]).float().norm(dim=1).mean()
                do = (out.hidden_states[l+1][0] - out_ref.hidden_states[l+1][0]).float().norm(dim=1).mean()
                lyaps.append(math.log((do / (di + 1e-12)).item()))
            l_sum += sum(lyaps) / len(lyaps)

        lam = l_sum / trials
        L_comp = self.nL // 3
        lam_L_comp = lam * L_comp

        if lam < 0.02:
            verdict = "FROZEN — model may be repetitive or undertrained"
        elif lam < 0.10:
            verdict = "HEALTHY — λ in optimal range"
        elif lam < 0.15:
            verdict = "WARNING — λ elevated, possible instability"
        else:
            verdict = "UNHEALTHY — λ too high, model likely impaired"

        return {"λ": round(lam, 4), "λ·L/3": round(lam_L_comp, 3),
                "verdict": verdict, "layers": self.nL}

    # ─── Fine-Tuning Monitor ─────────────────────────────────────

    def fine_tuning_monitor(self, model_after, text: str = None) -> dict:
        """Compare λ before and after fine-tuning. Detects instability."""
        before = self._cached_ref or self.health_check(text or "test")
        self._cached_ref = before

        # Swap to the fine-tuned model
        orig_model = self.model
        self.model = model_after
        after = self.health_check(text or "test")
        self.model = orig_model

        delta = after["λ"] - before["λ"]
        if abs(delta) < 0.02:
            verdict = "STABLE — fine-tuning preserved dynamics"
        elif delta > 0:
            verdict = f"WARNING — λ increased by {delta:.3f}, model becoming unstable"
        else:
            verdict = f"NOTE — λ decreased by {delta:.3f}, model becoming more stable"

        return {"before": before, "after": after, "Δλ": round(delta, 4), "verdict": verdict}

    # ─── Sphere Steer-Away ────────────────────────────────────────

    def steer_away(self, prompt: str, harmful_tokens: list[str],
                   strength: float = 0.5, max_tok: int = 50) -> str:
        """Generate text while steering AWAY from harmful concepts.

        Computes the tangent direction for each harmful token, then moves
        the hidden state opposite to the AVERAGE harmful direction.
        """
        h, inp = self._get_hidden(prompt)
        h_hat = h / h.norm()

        # Compute average harmful direction on the sphere
        combined = torch.zeros(self.d)
        for t in harmful_tokens:
            tid = self.tok.encode(t)[0]
            w = self.lm_head_w[tid]
            g = w - (w @ h_hat) * h_hat  # Tangent direction
            if g.norm() > 0:
                combined = combined + g / g.norm() * strength

        if combined.norm() > 0:
            combined = combined / combined.norm() * self.sphere_r * strength

        # Move OPPOSITE to harmful direction
        hs = h - combined
        hs = hs / hs.norm() * self.sphere_r

        return self._generate(hs, inp, max_tok)

    # ─── Stability Report ─────────────────────────────────────────

    def stability_report(self, text: str = None) -> dict:
        """Full per-zone λ diagnostic."""
        text = text or "A store has 120 apples, sells 45. How many left?"
        templ = self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True
        )
        inp = self.tok(templ, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out_ref = self.model(**inp, output_hidden_states=True)
        embeds = self.model.model.embed_tokens(inp.input_ids)

        zone_lams = {}
        for zone_name, zrange in [("routing (L0-6)", range(0, 7)),
                                    ("computation (L7-16)", range(7, 17)),
                                    ("execution (L17-27)", range(17, 28))]:
            lyaps = []
            for _ in range(2):
                noise = torch.randn_like(embeds) * 1e-4
                with torch.no_grad():
                    out = self.model(inputs_embeds=embeds + noise, output_hidden_states=True)
                for l in zrange:
                    if l + 1 >= self.nL:
                        continue
                    di = (out.hidden_states[l][0] - out_ref.hidden_states[l][0]).float().norm(dim=1).mean()
                    do = (out.hidden_states[l+1][0] - out_ref.hidden_states[l+1][0]).float().norm(dim=1).mean()
                    lyaps.append(math.log((do / (di + 1e-12)).item()))
            zone_lams[zone_name] = round(sum(lyaps) / len(lyaps), 4)

        report = self.health_check(text)
        report["zones"] = zone_lams
        report["model"] = self.model.config._name_or_path
        report["d"] = self.d
        report["layers"] = self.nL
        return report

    # ─── Helpers ──────────────────────────────────────────────────

    def _get_hidden(self, text: str):
        templ = self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True
        )
        inp = self.tok(templ, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu()
        h = h / h.norm() * self.sphere_r
        return h, inp

    def _generate(self, hs, inp, max_tok):
        g = inp.input_ids[0].tolist()
        for step in range(max_tok):
            inp_t = torch.tensor([g], device=self.device)
            with torch.no_grad():
                out = self.model(inp_t, output_hidden_states=True, use_cache=False)
            hf = hs.to(self.device, dtype=torch.bfloat16) if step == 0 else out.hidden_states[-1][0, -1, :]
            logits = self.model.lm_head.weight.to(dtype=torch.bfloat16) @ hf
            nxt = logits.argmax().item()
            g.append(nxt)
            if nxt == self.tok.eos_token_id:
                break
        return self.tok.decode(g[len(inp.input_ids[0]):], skip_special_tokens=True)


# ─── CLI Usage ────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys

    if len(sys.argv) < 2:
        print("Usage: python safety_toolkit.py <command> [model_name]")
        print("Commands: health, report, steer_away, monitor")
        print("Examples:")
        print("  python safety_toolkit.py health Qwen/Qwen2-1.5B-Instruct")
        print("  python safety_toolkit.py report Qwen/Qwen2-1.5B-Instruct")
        print("  python safety_toolkit.py steer_away Qwen/Qwen2-1.5B-Instruct")
        sys.exit(1)

    cmd = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2-1.5B-Instruct"

    print(f"Loading {model_name}...")
    st = SafetyToolkit(model_name)

    if cmd == "health":
        result = st.health_check()
        print(json.dumps(result, indent=2))

    elif cmd == "report":
        result = st.stability_report()
        print(json.dumps(result, indent=2))

    elif cmd == "steer_away":
        prompt = input("Enter prompt: ")
        harmful = input("Harmful tokens (comma-separated): ").split(",")
        result = st.steer_away(prompt, [t.strip() for t in harmful])
        print(f"\nSafe output: {result}")

    elif cmd == "monitor":
        print("λ health check ready. Call st.fine_tuning_monitor(new_model) after fine-tuning.")
        result = st.health_check()
        print(f"Pre-training λ: {result['λ']} ({result['verdict']})")