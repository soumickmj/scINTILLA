"""Diffusion map dimensionality reduction."""

from __future__ import annotations

import anndata as ad

from scintilla.config import RANDOM_SEED


def run_diffusion_map(
    adata: ad.AnnData,
    n_comps: int = 10,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Run diffusion map embedding via scanpy.

    Parameters
    ----------
    adata:
        AnnData.
    n_comps:
        Number of diffusion components.
    use_rep:
        Representation to use.

    Returns
    -------
    AnnData with X_diffmap in obsm.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("scanpy is required. Install with: pip install scanpy") from exc

    adata = adata.copy()
    if use_rep not in adata.obsm:
        sc.tl.pca(adata, random_state=random_state)
        use_rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_state)
    sc.tl.diffmap(adata, n_comps=n_comps, random_state=random_state)
    return adata
