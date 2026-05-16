#!/usr/bin/env python3
"""Generate the two conceptual figures for the LLM psychometrics paper."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


COLORS = {
    "ink": "#172033",
    "muted": "#5B6475",
    "line": "#CBD5E1",
    "blue": "#2563EB",
    "blue_soft": "#EAF2FF",
    "green": "#059669",
    "green_soft": "#E7F7EF",
    "red": "#DC2626",
    "red_soft": "#FDECEC",
    "amber": "#D97706",
    "amber_soft": "#FFF4DF",
    "purple": "#7C3AED",
    "purple_soft": "#F1EAFF",
    "gray_soft": "#F8FAFC",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body="",
    fc="#FFFFFF",
    ec=None,
    title_color=None,
    lw=1.6,
    radius=0.05,
    title_size=10.5,
    body_size=8.2,
    title_y=0.74,
    body_y=0.36,
    body_linespacing=1.18,
):
    ec = ec or COLORS["line"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=title_color or ec,
        fontweight="bold",
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * body_y,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["ink"],
            linespacing=body_linespacing,
        )
    return patch


def arrow(ax, start, end, color=None, rad=0.0, lw=1.8, ms=14):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            color=color or COLORS["muted"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def draw_intro(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.75, 7.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Vertical spine
    ax.plot([0.5, 0.5], [0.145, 0.93], color="#D6DEE9", lw=2.2, zorder=0)

    steps = [
        (
            0.865,
            "Common practice",
            "Administer human personality questionnaires to LLMs, compute scores, and interpret them as profiles.",
            COLORS["blue_soft"],
            COLORS["blue"],
            "1",
        ),
        (
            0.675,
            "Missing prior check",
            "A score is meaningful only if reliability, factor structure, validity, and invariance checks hold.",
            COLORS["amber_soft"],
            COLORS["amber"],
            "2",
        ),
        (
            0.485,
            "Our central question",
            "When applied to LLMs, do these instruments measure stable traits or response process artifacts?",
            COLORS["purple_soft"],
            COLORS["purple"],
            "3",
        ),
        (
            0.295,
            "Mechanistic answer",
            "Agreement bias and alignment shaped refusals can dominate the observed personality scores.",
            COLORS["red_soft"],
            COLORS["red"],
            "4",
        ),
    ]

    for cy, title, body, fc, ec, num in steps:
        box(
            ax,
            0.155,
            cy - 0.070,
            0.75,
            0.14,
            title,
            wrap(body, 43),
            fc=fc,
            ec=ec,
            title_color=ec,
            title_size=10.2,
            body_size=8.0,
            title_y=0.73,
            body_y=0.36,
            lw=1.25,
            radius=0.035,
        )
        circ = Circle((0.075, cy), 0.029, fc=ec, ec="white", lw=1.2)
        ax.add_patch(circ)
        ax.text(0.075, cy, num, ha="center", va="center", color="white", fontsize=11.4, fontweight="bold")
        ax.plot([0.105, 0.155], [cy, cy], color=ec, lw=1.2)

    for y1, y2 in [(0.795, 0.745), (0.605, 0.555), (0.415, 0.365)]:
        arrow(ax, (0.5, y1), (0.5, y2), color="#94A3B8", lw=1.1, ms=9)

    # Bottom claim panel
    box(
        ax,
        0.10,
        0.055,
        0.80,
        0.085,
        "Claim tested",
        wrap("Observed scores reflect response styles more than LLM personality traits.", 50),
        fc=COLORS["gray_soft"],
        ec=COLORS["ink"],
        title_color=COLORS["ink"],
        title_size=8.8,
        body_size=7.3,
        title_y=0.70,
        body_y=0.33,
        lw=1.15,
        radius=0.03,
    )

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_research_question_concept.{ext}", dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_method(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.955,
        "Methodology: validate the measurement before interpreting the profile",
        ha="center",
        va="top",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["ink"],
    )

    xs = [0.035, 0.225, 0.415, 0.605, 0.795]
    w = 0.145
    y = 0.50
    h = 0.255

    cards = [
        ("Instruments", "221 items\nIPIP, SD3,\nZKPQ, EPQR", COLORS["blue_soft"], COLORS["blue"]),
        ("Administration", "18 LLMs\n17 prompts\nDefault + MBTI", COLORS["green_soft"], COLORS["green"]),
        ("Response Matrix", "67,626 responses\nnormalized by scale", COLORS["amber_soft"], COLORS["amber"]),
        ("Validation Checks", "Reliability\nfactor structure\nvalidity + invariance", COLORS["purple_soft"], COLORS["purple"]),
        ("Diagnosis", "Separate traits from\nitems, prompts,\nand response bias", COLORS["red_soft"], COLORS["red"]),
    ]

    centers = []
    for x, (title, body, fc, ec) in zip(xs, cards):
        box(
            ax,
            x,
            y,
            w,
            h,
            title,
            body,
            fc=fc,
            ec=ec,
            title_color=ec,
            radius=0.028,
            title_size=9.0,
            body_size=7.3,
            title_y=0.71,
            body_y=0.34,
            lw=1.35,
        )
        centers.append((x + w / 2, y + h / 2))

    for i in range(len(centers) - 1):
        arrow(ax, (xs[i] + w + 0.012, y + h / 2), (xs[i + 1] - 0.012, y + h / 2), color="#64748B", lw=1.35, ms=10)

    # Diagnostics band
    band = FancyBboxPatch(
        (0.055, 0.17),
        0.89,
        0.19,
        boxstyle="round,pad=0.014,rounding_size=0.026",
        fc="#FBFDFF",
        ec="#D8E2EF",
        lw=1.2,
    )
    ax.add_patch(band)
    ax.text(0.5, 0.327, "Validation outputs", ha="center", va="center", fontsize=9.5, fontweight="bold", color=COLORS["ink"])

    output_items = [
        ("Cronbach's α", "Do items cohere?"),
        ("EFA factors", "Does Big Five survive?"),
        ("PIR", "Forward vs reverse consistency"),
        ("Variance split", "Model vs item vs prompt"),
        ("Convergent validity", "Do scales agree?"),
        ("Persona stress test", "Does role play restore validity?"),
    ]
    oxs = [0.085, 0.235, 0.385, 0.535, 0.685, 0.835]
    for x, (title, desc) in zip(oxs, output_items):
        ax.text(x, 0.265, title, ha="center", va="center", fontsize=7.6, color=COLORS["ink"], fontweight="bold")
        ax.text(x, 0.220, wrap(desc, 18), ha="center", va="center", fontsize=6.4, color=COLORS["muted"], linespacing=1.10)

    # Curved arrow from checks to validation band
    arrow(ax, (0.677, 0.50), (0.677, 0.37), color=COLORS["purple"], rad=0.0, lw=1.15, ms=9)
    arrow(ax, (0.868, 0.50), (0.868, 0.37), color=COLORS["red"], rad=0.0, lw=1.15, ms=9)

    # Final interpretation strip
    final = FancyBboxPatch(
        (0.22, 0.045),
        0.56,
        0.065,
        boxstyle="round,pad=0.012,rounding_size=0.024",
        fc="#172033",
        ec="#172033",
        lw=1.0,
    )
    ax.add_patch(final)
    ax.text(
        0.5,
        0.078,
        "Only after these checks can a score be treated as evidence of personality.",
        ha="center",
        va="center",
        color="white",
        fontsize=8.4,
        fontweight="bold",
    )

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig2_methodology_pipeline.{ext}", dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    draw_intro(args.out_dir)
    draw_method(args.out_dir)


if __name__ == "__main__":
    main()
