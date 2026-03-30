"""Differential expression sub-package."""

from scintilla.differential_expression.wilcoxon import wilcoxon_de
from scintilla.differential_expression.ttest import ttest_de
from scintilla.differential_expression.pseudobulk import pseudobulk_de
from scintilla.differential_expression.permutation import permutation_de
from scintilla.differential_expression.rank_genes import rank_genes_groups
from scintilla.differential_expression.utils import volcano_plot_data, filter_de_genes
from scintilla.differential_expression.benchmark import benchmark_de_methods

__all__ = [
    "wilcoxon_de",
    "ttest_de",
    "pseudobulk_de",
    "permutation_de",
    "rank_genes_groups",
    "volcano_plot_data",
    "filter_de_genes",
    "benchmark_de_methods",
]
