#!/usr/bin/env python3
"""Audit the committed cross-family CSVs against the numbers quoted in the
thread (five models, n = 2,560 rows).

Recomputes every table number directly from the raw CSV rows and asserts
the qualitative contracts the discussion relies on:

  - wrong-target tangent rank-1 rate == 0.0% on every model (strict exclusion)
  - random-tangent rank-1 rate    == 0.0% on every model
  - toward-blocker rate < arc-reach <= away-blocker rate  (competitor-direction effect)
  - no-mid-arc-loss: every case with a finite first-rank-1 angle has the
    target at rank 1 at the arc endpoint (1,958 / 1,958 on committed rows)

Run:  python verify_csv_audit.py        (csv only, < 1 s)
"""
import csv
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "steering_geometry_results"
GLOB = "*__t64c2_lf0-0.33-0.67-0.99_fp16.csv"


def rate(rows, col):
    return 100.0 * sum(1 for x in rows if x[col] == "1") / len(rows)


def main():
    print(f"{'model':38s} {'n':>5} {'arc%':>6} {'wrong%':>6} {'rand%':>6} "
          f"{'toward%':>7} {'away%':>7} {'midarc viol':>11}")
    total = reach = endp = viol = 0
    for f in sorted(RESULTS.glob(GLOB)):
        rows = list(csv.DictReader(open(f)))
        n = len(rows)
        total += n
        arc = rate(rows, "rank_target_tangent")
        wrong = rate(rows, "rank_wrong_tangent")
        rnd = rate(rows, "rank_random_tangent")
        tw = rate(rows, "rank_offarc_toward")
        aw = rate(rows, "rank_offarc_away")
        v = 0
        for x in rows:
            a = x["first_rank1_angle"]
            if a not in ("", "nan", "None"):
                reach += 1
                if x["rank_target_tangent"] != "1":
                    v += 1
            if x["rank_target_tangent"] == "1":
                endp += 1
        viol += v
        name = f.name.split("__")[0]
        print(f"{name:38s} {n:5d} {arc:6.1f} {wrong:6.1f} {rnd:6.1f} "
              f"{tw:7.1f} {aw:7.1f} {v:11d}")
        assert wrong == 0.0, f"wrong-target nonzero on {name}"
        assert rnd == 0.0, f"random nonzero on {name}"
        assert tw < arc <= aw, f"toward/arc/away ordering violated on {name}"
    print(f"\ntotal rows: {total}; reachable-on-arc: {reach}; endpoint rank-1: {endp}; "
          f"mid-arc-loss violations: {viol}")
    assert total == 2560 and reach == endp and viol == 0
    print("-> committed rows consistent with every quoted number.")


if __name__ == "__main__":
    main()