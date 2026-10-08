"""EDA sub-package."""

from scintilla.eda.summary import (
    dataset_summary,
    expressed_genes,
    mean_expression_by_group,
    sample_counts,
)

__all__ = ["dataset_summary", "expressed_genes", "sample_counts", "mean_expression_by_group"]
