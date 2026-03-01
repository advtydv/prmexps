from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from prm.rubric import StepScore


def render_step_heatmap(
    scores: list[StepScore],
    title: str = "Step-Level Rubric Scores",
) -> Figure:
    """Render a color-coded heatmap: rows = steps, columns = criteria."""
    if not scores:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No scores to display", ha="center", va="center")
        return fig

    criteria_names = list(scores[0].scores.keys())
    n_steps = len(scores)
    n_criteria = len(criteria_names)

    matrix = np.zeros((n_steps, n_criteria))
    for i, step_score in enumerate(scores):
        for j, name in enumerate(criteria_names):
            val = step_score.scores.get(name)
            matrix[i, j] = val if val is not None else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n_criteria * 2), max(4, n_steps * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n_criteria))
    ax.set_xticklabels(criteria_names, rotation=45, ha="right")
    step_labels = [f"Step {s.step_index + 1}" for s in scores]
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(step_labels)

    for i in range(n_steps):
        for j in range(n_criteria):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label="Score")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def render_comparison(
    correct_scores: list[StepScore],
    incorrect_scores: list[StepScore],
    correct_title: str = "Correct Chain",
    incorrect_title: str = "Corrupted Chain",
) -> Figure:
    """Render two heatmaps side by side for comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, max(len(correct_scores), len(incorrect_scores)) * 0.8)))

    for ax, step_scores, title in [
        (ax1, correct_scores, correct_title),
        (ax2, incorrect_scores, incorrect_title),
    ]:
        if not step_scores:
            ax.text(0.5, 0.5, "No scores", ha="center", va="center")
            ax.set_title(title)
            continue

        criteria_names = list(step_scores[0].scores.keys())
        n_steps = len(step_scores)
        n_criteria = len(criteria_names)

        matrix = np.zeros((n_steps, n_criteria))
        for i, ss in enumerate(step_scores):
            for j, name in enumerate(criteria_names):
                val = ss.scores.get(name)
                matrix[i, j] = val if val is not None else 0.0

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_criteria))
        ax.set_xticklabels(criteria_names, rotation=45, ha="right")
        ax.set_yticks(range(n_steps))
        ax.set_yticklabels([f"Step {s.step_index + 1}" for s in step_scores])

        for i in range(n_steps):
            for j in range(n_criteria):
                val = matrix[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        ax.set_title(title)

    fig.colorbar(im, ax=[ax1, ax2], label="Score", shrink=0.8)
    fig.suptitle("Process Reward Model: Correct vs. Corrupted", fontsize=14)
    fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.4)
    return fig


def render_aggregate(
    scores: list[StepScore],
    title: str = "Mean Score by Criterion",
) -> Figure:
    """Bar chart of average score per criterion across all steps."""
    if not scores:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No scores to display", ha="center", va="center")
        return fig

    criteria_names = list(scores[0].scores.keys())
    means: list[float] = []
    for name in criteria_names:
        vals = [s.scores[name] for s in scores if s.scores.get(name) is not None]
        means.append(sum(vals) / len(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(criteria_names) * 1.5), 4))
    colors = ["#2ecc71" if m >= 0.7 else "#f39c12" if m >= 0.4 else "#e74c3c" for m in means]
    bars = ax.bar(criteria_names, means, color=colors, edgecolor="black", linewidth=0.5)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean Score")
    ax.set_title(title)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="Good threshold")
    ax.legend()
    fig.tight_layout()
    return fig
