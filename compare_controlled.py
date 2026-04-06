"""Print controlled-vs-uncontrolled comparison from saved JSON files."""
import json

with open("results.json") as f:
    prev = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
with open("results_controlled.json") as f:
    ctrl = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

prev_fg = [v["forgetting_pct"] for v in prev.values()]
ctrl_fg = [v["forgetting_pct"] for v in ctrl.values()]

prev_ranking = sorted(prev, key=lambda k: prev[k]["forgetting_pct"])
ctrl_ranking = sorted(ctrl, key=lambda k: ctrl[k]["forgetting_pct"])

prev_spread = max(prev_fg) - min(prev_fg)
ctrl_spread = max(ctrl_fg) - min(ctrl_fg)

min_ctrl = min(ctrl, key=lambda k: ctrl[k]["forgetting_pct"])
max_ctrl = max(ctrl, key=lambda k: ctrl[k]["forgetting_pct"])

print("=" * 64)
print("COMPARISON: Uncontrolled vs Token-Controlled (50k tokens)")
print("=" * 64)
print(f"\n{'Pair':<22} {'Uncontrolled':>14} {'Controlled':>12}  Rank match?")
print("-" * 64)
all_pairs = sorted(set(list(prev.keys()) + list(ctrl.keys())))
for k in all_pairs:
    p  = prev.get(k, {}).get("forgetting_pct", float("nan"))
    c  = ctrl.get(k, {}).get("forgetting_pct", float("nan"))
    pr = prev_ranking.index(k) + 1 if k in prev_ranking else "?"
    cr = ctrl_ranking.index(k) + 1 if k in ctrl_ranking else "?"
    match = "OK" if pr == cr else "NO"
    print(f"{k:<22} {p:>+13.2f}% {c:>+11.2f}%  #{pr}->{cr} {match}")

ranking_stable = prev_ranking == ctrl_ranking
math_lit_key   = "math->literature"
# handle arrow variants in key names
ml_key = next((k for k in ctrl if "math" in k and "lit" in k), None)
still_highest = (max_ctrl == ml_key) if ml_key else False

print(f"\nPrev spread:  {prev_spread:.2f} pp")
print(f"Ctrl spread:  {ctrl_spread:.2f} pp  (token-equalized)")
print(f"Lowest  forgetting: {min_ctrl}  ({ctrl[min_ctrl]['forgetting_pct']:+.2f}%)")
print(f"Highest forgetting: {max_ctrl}  ({ctrl[max_ctrl]['forgetting_pct']:+.2f}%)")
print(f"Ranking stable: {'YES' if ranking_stable else 'NO -- some reordering'}")
print(f"Spread >10pp:   {'YES' if ctrl_spread > 10 else 'NO'}")
print(f"math->literature still highest: {'YES' if still_highest else 'NO'}")

if ctrl_spread > 10 and ranking_stable:
    verdict = "CONFIRMED: ranking stable + spread >10pp. Signal is real, not a data-volume artifact."
elif ctrl_spread > 10 and not ranking_stable:
    verdict = "MIXED: spread >10pp but ranking shifted. Signal real but details noisy."
elif ctrl_spread > 3:
    verdict = "WEAK: ranking held but spread <10pp. Partial confound possible."
else:
    verdict = "INCONCLUSIVE: signal collapsed under token control."

print(f"\nVerdict: {verdict}")
