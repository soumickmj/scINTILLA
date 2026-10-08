"""Dimensionality reduction sub-package."""

from scintilla.dimensionality_reduction.umap import run_umap
from scintilla.dimensionality_reduction.tsne import run_tsne
from scintilla.dimensionality_reduction.diffusion_map import run_diffusion_map
from scintilla.dimensionality_reduction.force_directed import run_force_directed
from scintilla.dimensionality_reduction.benchmark import benchmark_embeddings

__all__ = [
    "run_umap",
    "run_tsne",
    "run_diffusion_map",
    "run_force_directed",
    "benchmark_embeddings",
]
