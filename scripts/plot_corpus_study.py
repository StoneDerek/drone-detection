#!/usr/bin/env python3
"""Plot saved final metrics; this command never fits or scores a detector."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from drone_detector.common import read_json


def plot(report, output):
    results = read_json(report)["results"]
    families = ["spectral", "clap", "beats", "level-duration"]
    names = ["Spectral", "Frozen CLAP", "Frozen BEATs", "Level/duration"]
    colors = ["#4C78A8", "#E58B31", "#329A79", "#8A67AA"]
    sources = ["poland-uav-v1", "dronenoise-edzell-v3", "neaptide-v1"]
    labels = ["Poland\nnew site", "Edzell\nM1", "NEAPTIDE\nprocessed"]
    figure, (recall, fpr) = plt.subplots(1, 2, figsize=(11, 4.7), width_ratios=[1.35, 1])
    x = np.arange(3)
    width = 0.18
    for index, (family, label, color) in enumerate(zip(families, names, colors, strict=True)):
        bysource = results[family]["metrics"]["by_source"]
        values = [bysource[source]["positive_recall"] for source in sources]
        recall.bar(x + (index - 1.5) * width, values, width, color=color, label=label)
        negatives = bysource["urbansound8k"]
        fpr.bar(index - 0.18, negatives["negative_clip_FPR"], 0.34, color=color)
        fpr.bar(
            index + 0.18,
            negatives["macro_negative_group_FPR"],
            0.34,
            facecolor="white",
            edgecolor=color,
            hatch="////",
        )
    recall.set_xticks(x, labels)
    recall.set_title("Held-out positive crop recall", loc="left", weight="bold")
    recall.set_ylabel("Positive crops classified as drone")
    fpr.set_xticks(range(4), ["Spectral", "CLAP", "BEATs", "Level/\nduration"])
    fpr.set_title("Held-out UrbanSound8K FPR", loc="left", weight="bold")
    fpr.set_ylabel("Negative-label crops classified as drone")
    fpr.axhline(0.01, color="#444444", linestyle=":", linewidth=1)
    fpr.legend(
        handles=[
            Patch(facecolor="#999999", label="Clip FPR"),
            Patch(facecolor="white", edgecolor="#666666", hatch="////", label="Parent-mean FPR"),
            Line2D([0], [0], color="#444444", linestyle=":", label="1% development cutoff rule"),
        ],
        loc="upper right",
        fontsize=8,
        frameon=False,
    )
    for axis in [recall, fpr]:
        axis.set_ylim(0, 1.04)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    largest_fpr = max(
        results[f]["metrics"]["by_source"]["urbansound8k"][key]
        for f in families
        for key in ["negative_clip_FPR", "macro_negative_group_FPR"]
    )
    fpr.set_ylim(0, min(1.04, max(0.05, largest_fpr * 1.35)))
    figure.legend(
        *recall.get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.065),
    )
    figure.suptitle(
        "Frozen heads and one development-selected threshold per model",
        x=0.07,
        y=0.99,
        ha="left",
        fontsize=12,
    )
    figure.text(
        0.07,
        0.025,
        "Weak recording labels; fixed ≤3 s crops; few positive campaigns. "
        "FPR is a crop fraction, not false alarms/hour.",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.95), w_pad=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.output)


if __name__ == "__main__":
    main()
