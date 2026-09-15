"""t-SNE dimensionality reduction."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scintilla.config import RANDOM_SEED


def run_tsne(
    adata: ad.AnnData,
    n_components: int = 2,
    perplexity: float = 30.0,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
) -> ad.AnnData:
    """Run t-SNE embedding.

    Parameters
    ----------
    adata:
        AnnData.
    n_components:
        Number of t-SNE dimensions.
    perplexity:
        t-SNE perplexity.
    use_rep:
        Representation to use.

    Returns
    -------
    AnnData with X_tsne in obsm.
    """
    try:
        import scanpy as sc  # noqa: PLC0415
        adata = adata.copy()
        if use_rep not in adata.obsm:
            sc.tl.pca(adata, random_state=random_state)
            use_rep = "X_pca"
        sc.tl.tsne(
            adata, use_rep=use_rep, perplexity=perplexity, n_pcs=None,
            random_state=random_state,
        )
        return adata
    except Exception:
        from sklearn.manifold import TSNE  # noqa: PLC0415
        adata = adata.copy()
        if use_rep in adata.obsm:
            X = adata.obsm[use_rep]
        else:
            X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        perp = min(perplexity, X.shape[0] - 1)
        tsne = TSNE(n_components=n_components, perplexity=perp, random_state=random_state)
        adata.obsm["X_tsne"] = tsne.fit_transform(X.astype(np.float64))
        return adata
