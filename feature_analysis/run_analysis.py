#!/usr/bin/env python3
"""Run the standalone V2X textual feature analysis.

From the repository root:

    python -m feature_analysis.run_analysis

Reads data/processed/{train,val,test}.csv only. Writes artifacts under
feature_analysis/output/. Does not modify models, training, or the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_analysis.analyze import (  # noqa: E402
    class_feature_stats,
    important_tfidf_terms,
    write_findings,
)
from feature_analysis.extract import build_feature_matrix, fit_tfidf  # noqa: E402
from feature_analysis.load_messages import TEXT_COLUMN, load_messages  # noqa: E402
from feature_analysis.visualize import (  # noqa: E402
    plot_class_mean_heatmap,
    plot_feature_boxplots,
    plot_tfidf_terms,
)

OUTPUT = ROOT / "feature_analysis" / "output"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    messages = load_messages()
    feature_frame = build_feature_matrix(messages)
    stats = class_feature_stats(feature_frame)
    vectorizer, matrix = fit_tfidf(messages[TEXT_COLUMN].tolist())
    terms = important_tfidf_terms(
        messages[TEXT_COLUMN].tolist(),
        messages["Message Type"].tolist(),
        vectorizer,
        matrix,
    )

    feature_path = OUTPUT / "feature_matrix.csv"
    stats_path = OUTPUT / "class_feature_stats.csv"
    terms_path = OUTPUT / "tfidf_top_terms.csv"
    findings_path = OUTPUT / "findings.txt"
    feature_frame.to_csv(feature_path, index=False)
    stats.to_csv(stats_path, index=False)
    terms.to_csv(terms_path, index=False)

    # Dense TF-IDF is a separate table so the linguistic matrix stays readable.
    tfidf_frame = pd_tfidf(vectorizer, matrix, messages)
    tfidf_path = OUTPUT / "tfidf_matrix.csv"
    tfidf_frame.to_csv(tfidf_path, index=False)

    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_feature_boxplots(feature_frame, figures)
    plot_class_mean_heatmap(stats, figures)
    plot_tfidf_terms(terms, figures)

    write_findings(
        findings_path,
        n_messages=len(messages),
        feature_frame=feature_frame,
        stats=stats,
        terms=terms,
        n_tfidf_terms=len(vectorizer.get_feature_names_out()),
    )
    print(f"Wrote {feature_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {terms_path}")
    print(f"Wrote {tfidf_path}")
    print(f"Wrote {findings_path}")
    print(f"Wrote figures in {figures}")


def pd_tfidf(vectorizer, matrix, messages):
    import pandas as pd

    dense = matrix.toarray()
    columns = list(vectorizer.get_feature_names_out())
    frame = pd.DataFrame(dense, columns=columns)
    meta = messages[["Message ID", "Message Type"]].reset_index(drop=True)
    return pd.concat([meta, frame], axis=1)


if __name__ == "__main__":
    main()
