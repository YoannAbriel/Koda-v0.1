"""
Generate comparison charts from bench_results/*.json files.
Produces:
- results.csv
- bar charts per task (bar_hellaswag.png, etc.)
- radar chart (radar.png)
- scaling plot: avg score vs training compute (scaling.png)
- markdown summary (summary.md)
"""
import json
import glob
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path("/opt/yoann-test/bench_results")
OUT_DIR = Path("/opt/yoann-test/bench_charts")
OUT_DIR.mkdir(exist_ok=True)

# Known training tokens (for scaling plot) — from papers/model cards
TRAIN_TOKENS = {
    "kodalite-1.3b": 1.64e9,
    "gpt2-124m": 40e9,
    "gpt2-medium-355m": 40e9,
    "gpt2-large-774m": 40e9,
    "gpt2-xl-1.5b": 40e9,
    "pythia-1b": 300e9,
    "pythia-1.4b": 300e9,
    "opt-1.3b": 180e9,
    "tinyllama-1.1b": 3000e9,
}

TASK_ORDER = [
    "hellaswag", "arc_easy", "arc_challenge", "winogrande",
    "piqa", "boolq", "openbookqa", "lambada_openai",
]


def load_results():
    rows = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        name = Path(f).stem
        with open(f) as fp:
            data = json.load(fp)
        row = {
            "name": name,
            "model_id": data["model"],
            "params": data.get("params", 0),
            "train_tokens": TRAIN_TOKENS.get(name, 0),
        }
        for task in TASK_ORDER:
            t = data["tasks"].get(task, {})
            row[task] = t.get("accuracy") if isinstance(t, dict) and "accuracy" in t else None
        if "lambada_openai" in data["tasks"] and "perplexity" in data["tasks"]["lambada_openai"]:
            row["lambada_ppl"] = data["tasks"]["lambada_openai"]["perplexity"]
        rows.append(row)
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        fieldnames = ["name", "model_id", "params", "train_tokens"] + TASK_ORDER + ["lambada_ppl"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def bar_charts(rows, out_dir):
    for task in TASK_ORDER:
        names = [r["name"] for r in rows if r.get(task) is not None]
        scores = [r[task] for r in rows if r.get(task) is not None]
        if not scores:
            continue
        # Sort by score, KodaLite highlighted
        pairs = sorted(zip(names, scores), key=lambda x: x[1])
        names_s, scores_s = zip(*pairs)
        colors = ["#ff6b6b" if "kodalite" in n else "#4dabf7" for n in names_s]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(names_s, scores_s, color=colors)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"{task}")
        ax.axvline(x=0.25 if task in ("hellaswag", "arc_easy", "arc_challenge", "openbookqa") else 0.5,
                   color="gray", linestyle="--", alpha=0.5, label="random")
        ax.legend(loc="lower right")
        for bar, s in zip(bars, scores_s):
            ax.text(s + 0.005, bar.get_y() + bar.get_height()/2, f"{s:.3f}",
                    va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{task}.png", dpi=120)
        plt.close()


def radar_chart(rows, out_dir):
    tasks = [t for t in TASK_ORDER if t != "lambada_openai"]  # skip lambada (has ppl)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)

    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rows):
        values = [r.get(t) or 0 for t in tasks]
        values += values[:1]
        is_koda = "kodalite" in r["name"]
        ax.plot(angles, values, color="red" if is_koda else cmap(i),
                linewidth=3 if is_koda else 1.5, label=r["name"])
        if is_koda:
            ax.fill(angles, values, color="red", alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)
    plt.title("Model comparison — zero-shot accuracy", y=1.08, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "radar.png", dpi=120, bbox_inches="tight")
    plt.close()


def scaling_plot(rows, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    tasks = [t for t in TASK_ORDER if t != "lambada_openai"]
    for r in rows:
        avg = np.mean([r[t] for t in tasks if r.get(t) is not None])
        tokens = r["train_tokens"]
        is_koda = "kodalite" in r["name"]
        ax.scatter(tokens, avg, s=200 if is_koda else 80,
                   c="red" if is_koda else "blue",
                   edgecolors="black", zorder=3)
        ax.annotate(r["name"], (tokens, avg),
                    xytext=(5, 5), textcoords="offset points", fontsize=9,
                    fontweight="bold" if is_koda else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens (log scale)")
    ax.set_ylabel("Average accuracy across 7 benchmarks")
    ax.set_title("Model performance vs training compute")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scaling.png", dpi=120)
    plt.close()


def markdown_summary(rows, path):
    tasks = [t for t in TASK_ORDER if t != "lambada_openai"]
    lines = ["# KodaLite-1.3B Benchmark Results", ""]
    lines.append("Zero-shot accuracy across standard LM evaluation tasks.")
    lines.append("")

    # Table header
    header = ["Model", "Params", "Tokens"] + [t[:8] for t in TASK_ORDER] + ["Avg"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in sorted(rows, key=lambda r: -np.mean([r[t] or 0 for t in tasks])):
        row_vals = [
            r["name"],
            f"{r['params']/1e9:.2f}B",
            f"{r['train_tokens']/1e9:.0f}B",
        ]
        for t in TASK_ORDER:
            v = r.get(t)
            row_vals.append(f"{v:.3f}" if v is not None else "—")
        avg = np.mean([r[t] for t in tasks if r.get(t) is not None])
        row_vals.append(f"**{avg:.3f}**")
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    koda = next((r for r in rows if "kodalite" in r["name"]), None)
    if koda:
        lines.append(f"- **KodaLite-1.3B**: {koda['params']/1e9:.2f}B params, trained on {koda['train_tokens']/1e9:.2f}B tokens.")
        lines.append(f"- The model is **~25× undertrained** compared to GPT-2 (40B tokens) and **~180× undertrained** vs Pythia.")
        lines.append(f"- Performance near random on most tasks is expected given the training budget.")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    rows = load_results()
    print(f"Loaded {len(rows)} model results")
    save_csv(rows, OUT_DIR / "results.csv")
    bar_charts(rows, OUT_DIR)
    radar_chart(rows, OUT_DIR)
    scaling_plot(rows, OUT_DIR)
    markdown_summary(rows, OUT_DIR / "summary.md")
    print(f"Charts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
