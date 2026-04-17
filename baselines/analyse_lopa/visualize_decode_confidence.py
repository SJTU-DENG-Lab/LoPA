import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Edit Config Here
# =========================
INPUT_ROOT = Path("/home/chenkai/data/eval_all/analyse_lopa/evals_results_instruct")

STRATEGY_TO_SUBDIR = {
    "left_to_right": "gsm8k-ns4-len256-temp0.2-limit100-diffsteps256-dtypebfloat16-topp09-left_to_right",
    "right_to_left": "gsm8k-ns4-len256-temp0.2-limit100-diffsteps256-dtypebfloat16-topp09-right_to_left",
    "confidence_max": "gsm8k-ns4-len256-temp0.2-limit100-diffsteps256-dtypebfloat16-topp09-confidence_max",
    "random": "gsm8k-ns4-len256-temp0.2-limit100-diffsteps256-dtypebfloat16-topp09-random",
    "oracle": "gsm8k-ns4-len256-temp0.2-limit100-diffsteps256-dtypebfloat16-topp09-oracle",
}

# Metric: exact_match,flexible-extract
STRATEGY_TO_SCORE = {
    "left_to_right": 0.80,
    "right_to_left": 0.23,
    "confidence_max": 0.71,
    "random": 0.46,
    "oracle": 0.43,
}

# 保持原始输出文件名
OUTPUT_PATH = INPUT_ROOT / "decode_strategy_confidence.png"

FIGURE_TITLE = "Average Remaining-Mask Confidence by Decode Strategy"
FIGURE_DPI = 300
FIGURE_SIZE = (10.2, 6.0)

# 沿用原始示例配色风格，并把原本最深的那条线改成橙色系
PALETTE = [
    "#274753",  # dark blue grey
    "#297270",  # deep teal
    "#299d8f",  # teal
    "#8ab07c",  # sage green
    "#e7c66b",  # mustard yellow
    "#f3a361",  # sandy orange
    "#e66d50",  # terracotta red
]

STRATEGY_COLORS = {
    "left_to_right": "#e66d50",   # 红橙
    "right_to_left": "#297270",   # 改回原本示例配色中的绿色/青绿色
    "confidence_max": "#8ab07c",  # 改回原本示例配色中的绿色
    "random": "#e7c66b",          # 原本示例配色中的黄色
    "oracle": "#f3a361",          # 把原本最深的那条线改成橙色系
}

BACKGROUND_COLOR = "#fbfbf8"
GRID_COLOR = "#cfcfcf"
SPINE_COLOR = "#c8c8c8"
TEXT_COLOR = "#333333"


def build_strategy_dirs(input_root: Path, strategy_to_subdir: Dict[str, str]) -> Dict[str, Path]:
    strategy_dirs: Dict[str, Path] = {}
    for strategy_name, subdir in strategy_to_subdir.items():
        strategy_path = Path(subdir)
        if not strategy_path.is_absolute():
            strategy_path = input_root / strategy_path
        strategy_dirs[strategy_name] = strategy_path.resolve()
    return strategy_dirs


def collect_step_traces(strategy_dir: Path) -> List[List[float]]:
    traces: List[List[float]] = []
    rank_files = sorted(strategy_dir.glob("rank_*.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No rank_*.jsonl files found in {strategy_dir}")

    for rank_file in rank_files:
        with rank_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                trace = record.get("step_mean_max_prob")
                if trace is None:
                    raise KeyError(f"Missing 'step_mean_max_prob' in {rank_file}:{line_num}")
                if not isinstance(trace, list):
                    raise TypeError(f"'step_mean_max_prob' must be a list in {rank_file}:{line_num}")
                traces.append([float(x) for x in trace])

    if not traces:
        raise ValueError(f"No valid traces found in {strategy_dir}")

    return traces


def aggregate_statistics(
    traces: List[List[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(trace) for trace in traces)

    xs = []
    means = []
    q25s = []
    q75s = []
    counts = []

    for idx in range(max_len):
        vals = [trace[idx] for trace in traces if idx < len(trace)]
        if not vals:
            continue

        arr = np.asarray(vals, dtype=float)
        xs.append(idx)
        means.append(np.mean(arr))
        q25s.append(np.percentile(arr, 25))
        q75s.append(np.percentile(arr, 75))
        counts.append(len(arr))

    xs = np.asarray(xs)
    means = np.asarray(means)
    q25s = np.asarray(q25s)
    q75s = np.asarray(q75s)
    counts = np.asarray(counts)

    # 截断尾部样本极少的部分，避免末端噪声太大
    valid_mask = counts > max(1, int(len(traces) * 0.01))

    return xs[valid_mask], means[valid_mask], q25s[valid_mask], q75s[valid_mask], counts[valid_mask]


def setup_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.facecolor": BACKGROUND_COLOR,
        "figure.facecolor": "white",
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 1.0,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": True,
    })


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(1.0)

    ax.grid(True, axis="both", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.75)
    ax.set_axisbelow(True)


def plot_curves(strategy_to_dir: Dict[str, Path], output_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    style_axes(ax)

    strategy_order = [
        "left_to_right",
        "right_to_left",
        "confidence_max",
        "random",
        "oracle",
    ]

    max_x = 0
    max_y = 0

    for strategy_name in strategy_order:
        if strategy_name not in strategy_to_dir:
            continue

        strategy_dir = strategy_to_dir[strategy_name]
        if not strategy_dir.exists():
            raise FileNotFoundError(f"Strategy directory does not exist: {strategy_name} -> {strategy_dir}")

        traces = collect_step_traces(strategy_dir)
        xs, ys, q25, q75, counts = aggregate_statistics(traces)

        if len(xs) == 0:
            continue

        color = STRATEGY_COLORS.get(strategy_name, PALETTE[0])
        score = STRATEGY_TO_SCORE.get(strategy_name)

        max_x = max(max_x, int(xs[-1]))
        max_y = max(max_y, float(np.max(q75)))

        # 各条曲线自己的混色 IQR
        ax.fill_between(
            xs,
            q25,
            q75,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )

        # 主曲线，score 放到图例中，写成 Score=71% 的形式
        mark_interval = max(1, len(xs) // 18)
        if score is not None:
            label = f"{strategy_name} (Score={int(round(score * 100))}%)"
        else:
            label = strategy_name

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgewidth=0,
            markevery=mark_interval,
            label=label,
            zorder=3,
        )

        print(
            f"{strategy_name}: {len(traces)} samples, {len(xs)} plotted steps, "
            f"last-step sample count={counts[-1] if len(counts) else 0}, "
            f"exact_match,flexible-extract={score if score is not None else 'N/A'}"
        )

    ax.set_title(FIGURE_TITLE, pad=12)
    ax.set_xlabel("Decoding Step")
    ax.set_ylabel("Average Remaining-Mask Confidence")

    ax.set_xlim(left=0, right=max_x * 1.05 if max_x > 0 else 1)
    ax.set_ylim(bottom=0, top=max_y * 1.08 if max_y > 0 else 1)

    # 保持原本布局，图例放在偏右下角区域
    legend = ax.legend(
        loc="center right",
        bbox_to_anchor=(0.98, 0.32),
        fontsize=10.5,
        borderpad=0.5,
        handlelength=2.0,
        handletextpad=0.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d0d0d0")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存原始 png 名称
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")

    # 同步保存 pdf
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    input_root = INPUT_ROOT.resolve()
    output_path = OUTPUT_PATH.resolve()
    strategy_to_dir = build_strategy_dirs(input_root, STRATEGY_TO_SUBDIR)

    print(f"Input root: {input_root}")
    print(f"Output path: {output_path}")
    print(f"PDF path: {output_path.with_suffix('.pdf')}")
    for strategy_name, strategy_dir in strategy_to_dir.items():
        print(f"{strategy_name}: {strategy_dir}")

    plot_curves(strategy_to_dir, output_path)
    print(f"Saved figure to {output_path}")
    print(f"Saved figure to {output_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()