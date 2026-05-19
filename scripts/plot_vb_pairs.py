#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "datasets" / "processed" / "mill_process_info.csv"
    output_dir = root / "eda" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        (1, 9),
        (2, 12),
        (3, 11),
        (4, 10),
        (5, 16),
        (6, 15),
        (7, 13),
        (8, 14),
    ]

    df = pd.read_csv(csv_path)
    df = df[df["case"].isin([case for pair in pairs for case in pair])].copy()
    df["VB"] = pd.to_numeric(df["VB"], errors="coerce")

    colors = plt.get_cmap("tab10").colors
    pair_to_color = {pair: colors[idx % len(colors)] for idx, pair in enumerate(pairs)}
    case_to_pair = {case: pair for pair in pairs for case in pair}
    case_to_marker = {}
    for pair in pairs:
        case_to_marker[pair[0]] = "o"
        case_to_marker[pair[1]] = "s"

    def plot_xy(x_col: str, x_label: str, output_name: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 6))
        for case in sorted(df["case"].unique()):
            case_df = df[df["case"] == case].dropna(subset=["VB", x_col])
            if case_df.empty:
                continue
            case_df = case_df.sort_values(x_col)
            pair = case_to_pair[case]
            color = pair_to_color[pair]
            marker = case_to_marker[case]
            ax.plot(
                case_df[x_col],
                case_df["VB"],
                marker=marker,
                color=color,
                linewidth=1.5,
                markersize=5,
                label=f"case {case} (pair {pair[0]}-{pair[1]})",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("VB")
        ax.set_title(f"{x_label} vs VB by case (paired colors)")
        ax.grid(True, alpha=0.3)
        ax.legend(
            ncol=2,
            fontsize=8,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        plt.subplots_adjust(right=0.75)
        output_path = output_dir / output_name
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    plot_xy("time", "time", "vb_time_by_case_pairs.png")
    plot_xy("run", "run_index", "vb_run_by_case_pairs.png")


if __name__ == "__main__":
    main()
