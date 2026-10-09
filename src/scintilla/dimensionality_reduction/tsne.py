"""t-SNE dimensionality reduction."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np

from scintilla._compat import finish, prepare, record_params
from scintilla.config import RANDOM_SEED
from scintilla.dimensionality_reduction._common import get_representation, stand_in


def run_tsne(
    adata: ad.AnnData,
    n_components: int = 2,
    perplexity: float = 30.0,
    use_rep: str = "X_pca",
    random_state: int = RANDOM_SEED,
    *,
    layer: Optional[str] = None,
    key_added: str = "X_tsne",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Run t-SNE and store the embedding in ``adata.obsm[key_added]``.

    scanpy's implementation is used when available; otherwise scikit-learn's
    :class:`~sklearn.manifold.TSNE` (the fallback honours ``n_components``).

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Number of t-SNE dimensions.
    perplexity
        t-SNE perplexity (capped at ``n_obs - 1`` by the scikit-learn fallback).
    use_rep
        Representation in ``adata.obsm`` to embed.  When it is absent a temporary PCA
        of ``X`` is used (and not stored).
    random_state
        Random seed.
    layer
        Layer used for the temporary PCA when ``use_rep`` is absent.
    key_added
        Key of the embedding in ``adata.obsm``.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    adata, give_back = prepare(adata, copy=copy)
    rep = get_representation(adata, use_rep, random_state, layer)
    try:
        import scanpy as sc

        tmp = stand_in(rep, adata.obs_names)
        sc.tl.tsne(tmp, use_rep="rep", perplexity=perplexity, n_pcs=None, random_state=random_state)
        embedding = tmp.obsm["X_tsne"]
    except ImportError:
        from sklearn.manifold import TSNE

        perp = min(perplexity, rep.shape[0] - 1)
        embedding = TSNE(n_components=n_components, perplexity=perp, random_state=random_state).fit_transform(
            rep.astype(np.float64)
        )
    adata.obsm[key_added] = embedding
    record_params(
        adata, key_added, n_components=n_components, perplexity=perplexity, use_rep=use_rep,
        random_state=random_state, key_added=key_added,
    )
    return finish(adata, give_back)
