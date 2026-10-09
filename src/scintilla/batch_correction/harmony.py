"""Harmony batch correction."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np

from scintilla._compat import finish, prepare, record_params
from scintilla.config import RANDOM_SEED
from scintilla.dimensionality_reduction._common import get_representation


def harmony_correct(
    adata: ad.AnnData,
    batch_key: str,
    n_components: int = 30,
    random_state: int = RANDOM_SEED,
    *,
    use_rep: str = "X_pca",
    key_added: str = "X_pca_harmony",
    copy: bool = False,
) -> Optional[ad.AnnData]:
    """Apply Harmony to a PCA embedding and store it in ``adata.obsm[key_added]``.

    Parameters
    ----------
    adata
        Annotated data matrix with the embedding ``use_rep`` in ``obsm``.  When it is
        absent a temporary PCA of ``X`` is used (and not stored).
    batch_key
        Column in ``adata.obs`` with batch labels.
    n_components
        Number of embedding dimensions to correct.
    random_state
        Random seed for Harmony's optimisation.
    use_rep
        Embedding to correct.
    key_added
        Key of the corrected embedding in ``adata.obsm``.
    copy
        Return a modified copy instead of modifying ``adata`` in place.

    Returns
    -------
    anndata.AnnData or None
        ``None`` when working in place; the modified copy when ``copy=True``.
    """
    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError(
            "harmonypy is required for Harmony correction. "
            "Install with: pip install harmonypy"
        ) from exc

    adata, give_back = prepare(adata, copy=copy)
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Column '{batch_key}' not found in obs.")
    X_pca = get_representation(adata, use_rep, random_state)[:, :n_components]
    ho = harmonypy.run_harmony(
        X_pca, adata.obs, batch_key, random_state=random_state,
    )

    # harmonypy < 2.0 returns Z_corr as (n_components, n_cells); 2.0 returns
    # (n_cells, n_components).  Orient by the caller's own shape instead of
    # assuming either, so obsm never receives a transposed embedding.
    Z_corr = np.asarray(ho.Z_corr)
    expected = X_pca.shape
    if Z_corr.shape == expected:
        embedding = Z_corr
    elif Z_corr.shape == expected[::-1]:
        embedding = Z_corr.T
    else:
        raise ValueError(
            "harmonypy returned a corrected embedding of unexpected shape "
            f"{Z_corr.shape}; expected {expected} or {expected[::-1]}"
        )
    adata.obsm[key_added] = embedding
    record_params(
        adata, key_added, batch_key=batch_key, n_components=n_components, random_state=random_state,
        use_rep=use_rep, key_added=key_added,
    )
    return finish(adata, give_back)
