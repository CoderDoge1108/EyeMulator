"""Distribution analysis over the distilled human-attention priors.

Reproduces the human-side descriptive statistics reported under "RQ1:
Artifact Distillation" in the paper: fitted Beta parameters per semantic
label, posterior mean salience E[theta_s] = alpha_s / (alpha_s + beta_s),
and the most frequent monogram / bigram / trigram transitions over
programmer fixation sequences.

Standard library only. Matplotlib is imported lazily and is needed only
when ``--plot`` is passed.

Example
-------
    python example/analyze_human_attention.py --priors priors/combined --top 10
    python example/analyze_human_attention.py --priors priors/reading  --plot beta_reading.pdf
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_priors(prior_dir: Path) -> Dict[str, object]:
    """Load the five JSON files produced by the distillation step."""
    prior_dir = Path(prior_dir)
    return {
        "beta": _load_json(prior_dir / "beta_distribution.json"),
        "monogram": _load_json(prior_dir / "monogram_counts.json"),
        "bigram": _load_json(prior_dir / "bigram_counts.json"),
        "trigram": _load_json(prior_dir / "trigram_counts.json"),
        "indexed_ngrams": _load_json(prior_dir / "indexed_ngrams.json"),
    }


def beta_summary(beta_rows: List[dict]) -> List[dict]:
    """Augment each row with posterior mean and sort by it."""
    out = []
    for row in beta_rows:
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        posterior_mean = alpha / (alpha + beta)
        out.append(
            {
                "semantic_label": row["semantic_label"],
                "alpha": alpha,
                "beta": beta,
                "posterior_mean": posterior_mean,
                "saccades_count": row.get("saccades_count"),
                "word_count": row.get("word_count"),
            }
        )
    out.sort(key=lambda r: r["posterior_mean"], reverse=True)
    return out


def _parse_ngram_key(key: str) -> Tuple[str, ...]:
    """Bigram / trigram JSON keys are stringified Python tuples.

    e.g. "('argument', 'function call')" -> ("argument", "function call").
    """
    try:
        value = ast.literal_eval(key)
        if isinstance(value, tuple):
            return tuple(str(v) for v in value)
    except (ValueError, SyntaxError):
        pass
    return (key,)


def top_ngrams(counts: Dict[str, int], k: int) -> List[Tuple[Tuple[str, ...], int]]:
    parsed = [(_parse_ngram_key(k_), int(v)) for k_, v in counts.items()]
    parsed.sort(key=lambda item: item[1], reverse=True)
    return parsed[:k]


def _format_table(rows: Iterable[Iterable[object]], header: Iterable[str]) -> str:
    rows = list(rows)
    header = list(header)
    cols = list(zip(header, *rows))
    widths = [max(len(str(cell)) for cell in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*header), fmt.format(*["-" * w for w in widths])]
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))
    return "\n".join(lines)


def print_beta_table(summary: List[dict], top_k: int) -> None:
    rows = [
        (
            r["semantic_label"],
            int(r["alpha"]),
            int(r["beta"]),
            f"{r['posterior_mean']:.3f}",
        )
        for r in summary[:top_k]
    ]
    print("\nTop-{} semantic labels by posterior salience E[theta_s]".format(top_k))
    print(_format_table(rows, ["semantic_label", "alpha", "beta", "E[theta_s]"]))


def print_ngram_table(title: str, items: List[Tuple[Tuple[str, ...], int]]) -> None:
    rows = [(" -> ".join(key), count) for key, count in items]
    print("\n" + title)
    print(_format_table(rows, ["sequence", "count"]))


def maybe_plot_beta_curves(summary: List[dict], top_k: int, out_path: Path) -> None:
    """Optional: plot the density of Beta(alpha_s, beta_s) for the top-k labels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for --plot; install with `pip install matplotlib`"
        ) from e

    xs = [i / 200 for i in range(1, 200)]

    def beta_pdf(x: float, a: float, b: float) -> float:
        log_B = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - log_B)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in summary[:top_k]:
        ys = [beta_pdf(x, r["alpha"], r["beta"]) for x in xs]
        ax.plot(xs, ys, label=r["semantic_label"])
    ax.set_xlabel("theta (attention probability)")
    ax.set_ylabel("density")
    ax.set_title("Beta density per semantic label")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"\nWrote plot to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priors",
        default="priors/combined",
        help="Path to a priors subfolder (combined / reading / writing).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many rows to show in each table.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional output path; if given, render Beta density curves there.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    priors = load_priors(Path(args.priors))
    summary = beta_summary(priors["beta"])

    print(f"Loaded priors from: {args.priors}")
    print(f"  semantic labels:   {len(summary)}")
    print(f"  unique monograms:  {len(priors['monogram'])}")
    print(f"  unique bigrams:    {len(priors['bigram'])}")
    print(f"  unique trigrams:   {len(priors['trigram'])}")

    print_beta_table(summary, args.top)

    semantic_labels = {row["semantic_label"] for row in priors["beta"]}
    monogram_items = [
        (k, int(v)) for k, v in priors["monogram"].items() if k in semantic_labels
    ]
    monogram_items.sort(key=lambda kv: kv[1], reverse=True)
    print_ngram_table(
        f"Top-{args.top} monograms (semantic labels by fixation count)",
        [((k,), v) for k, v in monogram_items[: args.top]],
    )
    print_ngram_table(
        f"Top-{args.top} bigrams",
        top_ngrams(priors["bigram"], args.top),
    )
    print_ngram_table(
        f"Top-{args.top} trigrams",
        top_ngrams(priors["trigram"], args.top),
    )

    if args.plot:
        maybe_plot_beta_curves(summary, args.top, Path(args.plot))


if __name__ == "__main__":
    main()
