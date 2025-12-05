"""Utility for surfacing the top binary evaluation runs by metric."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import os
import matplotlib.pyplot as plt


RESULTS_PATH = Path(__file__).resolve().parent / "results" / "multitype_results.json"
TOP_K = 3
METRICS = ("Accuracy", "F1")


@dataclass(frozen=True)
class EvalRun:
    label: str
    metrics: Dict[str, float]

    @classmethod
    def from_entry(cls, entry: Dict) -> "EvalRun":
        label = f"{entry['model_name']} ({entry['prompt_version']})"
        return cls(label=label, metrics=entry.get("metrics", {}))


def load_runs(path: Path) -> List[EvalRun]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file at: {path}")

    with path.open() as fp:
        payload = json.load(fp)

    return [EvalRun.from_entry(entry) for entry in payload]


def top_k_runs(runs: List[EvalRun], metric: str, k: int) -> List[EvalRun]:
    filtered = [run for run in runs if metric in run.metrics]
    if not filtered:
        raise ValueError(f"Metric '{metric}' missing from all runs.")

    sorted_runs = sorted(
        filtered,
        key=lambda run: run.metrics[metric],
        reverse=True,
    )
    return sorted_runs[:k]


def plot_top_runs(runs: List[EvalRun], metrics: List[str], top_k: int) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        top_runs = top_k_runs(runs, metric, top_k)
        scores = [run.metrics[metric] for run in top_runs]
        labels = [run.label for run in top_runs]

        bars = ax.bar(range(len(scores)), scores, tick_label=[f"#{idx+1}" for idx in range(len(scores))])
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_xlabel("Rank")
        ax.set_title(f"Top {len(scores)} by {metric}")

        legend_labels = [f"{label}: {score:.3f}" for label, score in zip(labels, scores)]
        ax.legend(bars, legend_labels, loc="lower left", fontsize="small")

    fig.tight_layout()
    plt.show()
    PLOT_DIR="w4/qwen"
    save_path = os.path.join(PLOT_DIR, "best_multilabel_prompt.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot → {save_path}")


def main() -> None:
    runs = load_runs(RESULTS_PATH)

    print("Top runs:")
    for metric in METRICS:
        best = top_k_runs(runs, metric, TOP_K)
        for idx, run in enumerate(best, start=1):
            print(f"{metric} #{idx}: {run.label} -> {run.metrics[metric]:.4f}")

    plot_top_runs(runs, list(METRICS), TOP_K)


if __name__ == "__main__":
    main()
