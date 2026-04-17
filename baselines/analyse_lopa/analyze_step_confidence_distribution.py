import json
from collections import Counter, defaultdict
from html import escape
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

# Edit these paths and hyperparameters directly before running.
INPUT_DIR = Path(
    "/home/chenkai/data/eval_all/analyse_lopa/evals_results_instruct_perposition/"
    "gsm8k-ns4-len256-temp0.2-limit100-diffsteps256-dtypebfloat16-topp09-confidence_max-per-position"
)
OUTPUT_DIR = INPUT_DIR / "analysis_step_confidence"
TOKENIZER_PATH = Path("/home/chenkai/data/models/Dream-v0-Instruct-7B")

# 需要分析的 step 索引。这里使用 1-based step 编号，例如 [1, 10, 20]。
ANALYSIS_STEPS = [1, 50, 100]

# 直方图的分桶数。
HIST_BINS = 20

# 统计 token 时使用的置信度区间边界，必须覆盖 [0, 1]。
# 例如 [0.0, 0.2, 0.5, 0.8, 1.0] 表示分成 4 个区间：
# [0.0,0.2), [0.2,0.5), [0.5,0.8), [0.8,1.0]
TOKEN_CONFIDENCE_INTERVALS = [0.0, 1 / 3, 2 / 3, 1.0]

# 每个置信度区间保留出现频率最高的多少个 token。
TOP_TOKENS_PER_INTERVAL = 20


@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def load_id_to_token_map(tokenizer_path: Path) -> Dict[int, str]:
    vocab_path = tokenizer_path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab.json under {tokenizer_path}")

    with vocab_path.open("r", encoding="utf-8") as fin:
        vocab = json.load(fin)

    id_to_token = {int(token_id): token_text for token_text, token_id in vocab.items()}

    added_tokens_path = tokenizer_path / "added_tokens.json"
    if added_tokens_path.exists():
        with added_tokens_path.open("r", encoding="utf-8") as fin:
            added_tokens = json.load(fin)
        for token_text, token_id in added_tokens.items():
            id_to_token[int(token_id)] = token_text

    return id_to_token


def decode_vocab_token(raw_token: str) -> str:
    if raw_token.startswith("<|") and raw_token.endswith("|>"):
        return raw_token

    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    try:
        return bytearray([byte_decoder[c] for c in raw_token]).decode("utf-8", errors="replace")
    except Exception:
        return raw_token


def load_step_items(input_dir: Path, target_steps: List[int]) -> Dict[int, List[dict]]:
    per_step_items = {step: [] for step in target_steps}
    rank_files = sorted(input_dir.glob("rank_*.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No rank_*.jsonl files found in {input_dir}")

    for rank_file in rank_files:
        with rank_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                record = json.loads(line)
                traces = record.get("step_remaining_mask_details", [])
                for step in target_steps:
                    step_index = step - 1
                    if 0 <= step_index < len(traces):
                        per_step_items[step].extend(traces[step_index])
    return per_step_items


def save_histogram(step: int, items: List[dict], bins: int, output_dir: Path) -> None:
    confidences = np.array([item["confidence"] for item in items], dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(confidences, bins=bin_edges)
    save_histogram_svg(step, counts.tolist(), bin_edges.tolist(), output_dir / f"step_{step}_confidence_hist.svg")

    histogram_summary = {
        "step": step,
        "num_positions": int(len(confidences)),
        "min_confidence": float(confidences.min()) if len(confidences) else None,
        "max_confidence": float(confidences.max()) if len(confidences) else None,
        "mean_confidence": float(confidences.mean()) if len(confidences) else None,
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }
    with (output_dir / f"step_{step}_confidence_hist.json").open("w", encoding="utf-8") as fout:
        json.dump(histogram_summary, fout, ensure_ascii=False, indent=2)


def save_histogram_svg(step: int, counts: List[int], bin_edges: List[float], output_path: Path) -> None:
    width = 1000
    height = 600
    margin_left = 90
    margin_right = 30
    margin_top = 60
    margin_bottom = 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_count = max(counts) if counts else 1
    max_count = max(max_count, 1)
    bar_gap = 2
    bar_width = chart_width / max(len(counts), 1)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">Step {step} Confidence Distribution</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="black" stroke-width="2"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="black" stroke-width="2"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="18" font-family="Arial">Confidence</text>',
        f'<text x="25" y="{height / 2}" text-anchor="middle" font-size="18" font-family="Arial" transform="rotate(-90 25 {height / 2})">Count of Mask Positions</text>',
    ]

    for tick in range(6):
        tick_value = tick / 5
        x = margin_left + chart_width * tick_value
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 8}" stroke="black" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 28}" text-anchor="middle" font-size="14" font-family="Arial">{tick_value:.1f}</text>'
        )

    for tick in range(6):
        tick_value = max_count * tick / 5
        y = height - margin_bottom - chart_height * tick / 5
        svg_parts.append(
            f'<line x1="{margin_left - 8}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="black" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#d0d0d0" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="14" font-family="Arial">{int(round(tick_value))}</text>'
        )

    for idx, count in enumerate(counts):
        bar_height = chart_height * (count / max_count)
        x = margin_left + idx * bar_width + bar_gap / 2
        y = height - margin_bottom - bar_height
        svg_parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - bar_gap, 1):.2f}" height="{bar_height:.2f}" fill="#4C78A8" stroke="#2F4B6C" stroke-width="0.6"/>'
        )

    for idx in range(0, len(bin_edges), max(len(bin_edges) // 5, 1)):
        edge = bin_edges[idx]
        x = margin_left + chart_width * edge
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 46}" text-anchor="middle" font-size="11" font-family="Arial">{edge:.2f}</text>'
        )
    if bin_edges:
        svg_parts.append(
            f'<text x="{width - margin_right:.2f}" y="{height - margin_bottom + 46}" text-anchor="middle" font-size="11" font-family="Arial">{bin_edges[-1]:.2f}</text>'
        )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def get_interval_label(lower: float, upper: float, is_last: bool) -> str:
    right_bracket = "]" if is_last else ")"
    return f"[{lower:.4f}, {upper:.4f}{right_bracket}"


def save_top_tokens_by_interval(
    step: int,
    items: List[dict],
    top_k: int,
    id_to_token: Dict[int, str],
    interval_edges: List[float],
    output_dir: Path,
) -> None:
    coarse_bins = len(interval_edges) - 1
    interval_token_counts = defaultdict(Counter)
    interval_totals = Counter()

    for item in items:
        confidence = float(item["confidence"])
        token_id = int(item["top_token_id"])
        interval_index = int(np.searchsorted(interval_edges, confidence, side="right") - 1)
        interval_index = min(max(interval_index, 0), coarse_bins - 1)
        interval_token_counts[interval_index][token_id] += 1
        interval_totals[interval_index] += 1

    result = {
        "step": step,
        "num_intervals": coarse_bins,
        "interval_edges": [float(x) for x in interval_edges.tolist()],
        "intervals": [],
    }

    csv_lines = [
        "step,interval_index,interval_label,interval_total_count,rank,token_id,token_text_decoded,token_text_raw,token_count,token_ratio_within_interval"
    ]
    for interval_index in range(coarse_bins):
        lower = float(interval_edges[interval_index])
        upper = float(interval_edges[interval_index + 1])
        total = int(interval_totals[interval_index])
        label = get_interval_label(lower, upper, interval_index == coarse_bins - 1)
        top_tokens = []
        if total > 0:
            for rank, (token_id, count) in enumerate(interval_token_counts[interval_index].most_common(top_k), start=1):
                raw_token_text = id_to_token.get(token_id, f"<UNK_ID_{token_id}>")
                decoded_token_text = decode_vocab_token(raw_token_text)
                ratio = count / total
                top_tokens.append(
                    {
                        "rank": rank,
                        "token_id": int(token_id),
                        "token_text_decoded": decoded_token_text,
                        "token_text_raw": raw_token_text,
                        "token_count": int(count),
                        "token_ratio_within_interval": float(ratio),
                    }
                )
                escaped_decoded = escape(decoded_token_text).replace(",", "\\,").replace("\n", "\\n")
                escaped_raw = escape(raw_token_text).replace(",", "\\,").replace("\n", "\\n")
                csv_lines.append(
                    f"{step},{interval_index},{label},{total},{rank},{token_id},{escaped_decoded},{escaped_raw},{count},{ratio:.8f}"
                )

        result["intervals"].append(
            {
                "interval_index": interval_index,
                "interval_label": label,
                "lower": lower,
                "upper": upper,
                "interval_total_count": total,
                "top_tokens": top_tokens,
            }
        )

    with (output_dir / f"step_{step}_top_tokens_by_interval.json").open("w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    with (output_dir / f"step_{step}_top_tokens_by_interval.csv").open("w", encoding="utf-8") as fout:
        fout.write("\n".join(csv_lines) + "\n")


def main() -> None:
    output_dir = OUTPUT_DIR
    target_steps = ANALYSIS_STEPS
    output_dir.mkdir(parents=True, exist_ok=True)
    interval_edges = np.array(TOKEN_CONFIDENCE_INTERVALS, dtype=np.float64)
    if interval_edges.ndim != 1 or len(interval_edges) < 2:
        raise ValueError("TOKEN_CONFIDENCE_INTERVALS must contain at least two boundaries.")
    if abs(float(interval_edges[0]) - 0.0) > 1e-8 or abs(float(interval_edges[-1]) - 1.0) > 1e-8:
        raise ValueError("TOKEN_CONFIDENCE_INTERVALS must start at 0.0 and end at 1.0.")
    if np.any(np.diff(interval_edges) <= 0):
        raise ValueError("TOKEN_CONFIDENCE_INTERVALS must be strictly increasing.")

    id_to_token = load_id_to_token_map(TOKENIZER_PATH)
    per_step_items = load_step_items(INPUT_DIR, target_steps)

    summary = {"input_dir": str(INPUT_DIR), "steps": []}
    for step in target_steps:
        items = per_step_items[step]
        if not items:
            summary["steps"].append({"step": step, "num_positions": 0, "status": "no_data"})
            continue

        save_histogram(step, items, HIST_BINS, output_dir)
        save_top_tokens_by_interval(
            step,
            items,
            TOP_TOKENS_PER_INTERVAL,
            id_to_token,
            interval_edges,
            output_dir,
        )
        summary["steps"].append({"step": step, "num_positions": len(items), "status": "ok"})

    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
