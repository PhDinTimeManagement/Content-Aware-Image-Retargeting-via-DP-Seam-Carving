import csv
from collections import defaultdict
import matplotlib.pyplot as plt

CSV_PATH = "output/eval_w300_test.csv"
OUT_PNG  = "output/plot_ratio_box.png"

ratios = defaultdict(list)

with open(CSV_PATH, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        method = row["method"]
        ratios[method].append(float(row["ratio"]))

methods = ["seam", "resize", "crop"]
data = [ratios[m] for m in methods]

plt.figure()
plt.boxplot(data, labels=methods)
plt.ylabel("avgE_out / avgE_in (ratio)")
plt.title("Energy ratio distribution on BSDS500 test split")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print("Saved:", OUT_PNG)
