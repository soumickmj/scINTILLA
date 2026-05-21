"""Visualisation utilities for supervised analysis results."""

from __future__ import annotations

from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _save_and_close(fig: plt.Figure, save: Optional[str]) -> plt.Figure:
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_classifier_comparison(
    results: dict,
    cmap: str = "RdYlGn",
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> Dict[str, plt.Figure]:
    """Create a suite of comparison plots from supervised_analysis results.

    Parameters
    ----------
    results:
        The dict returned by :func:`supervised_analysis`.
    cmap:
        Colormap for heatmaps.
    save:
        If provided, saves each figure as ``<save>_<plot_name>.png``.
    figsize:
        Optional base figure size.

    Returns
    -------
    Dict mapping plot name to its matplotlib Figure.
    """
    all_results = results["all_results"]
    best_name = results.get("best_model_name")
    figs: Dict[str, plt.Figure] = {}

    # ── 1. Build a metrics DataFrame ────────────────────────────────
    scalar_keys = [
        "accuracy", "precision", "recall", "f1",
        "fdr", "fnr", "auroc", "auprc",
        "mcc", "cohen_kappa", "balanced_accuracy",
    ]
    rows = []
    for name, res in all_results.items():
        m = res["metrics"]
        row = {"model": name}
        for k in scalar_keys:
            row[k] = m.get(k, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")

    # ── 2. Bar chart: core metrics side-by-side ─────────────────────
    core = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    core_df = df[core]
    n_models = len(core_df)
    fig1, ax1 = plt.subplots(figsize=figsize or (max(8, n_models * 1.2), 5))
    core_df.plot.bar(ax=ax1, width=0.8)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.tick_params(axis="x", rotation=45)
    _save_path = f"{save}_core_metrics_bar.png" if save else None
    figs["core_metrics_bar"] = _save_and_close(fig1, _save_path)

    # ── 3. Heatmap: all scalar metrics ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=figsize or (max(8, len(scalar_keys) * 0.9), max(4, n_models * 0.6)))
    sns.heatmap(
        df[scalar_keys].astype(float), annot=True, fmt=".3f",
        cmap=cmap, vmin=0, vmax=1, linewidths=0.5, ax=ax2,
    )
    _save_path = f"{save}_metrics_heatmap.png" if save else None
    figs["metrics_heatmap"] = _save_and_close(fig2, _save_path)

    # ── 4. Radar / spider chart of core metrics ─────────────────────
    categories = core
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3 = plt.subplots(figsize=figsize or (7, 7), subplot_kw=dict(polar=True))
    for model_name in df.index:
        values = df.loc[model_name, categories].values.tolist()
        values += values[:1]
        ax3.plot(angles, values, linewidth=1.5, label=model_name)
        ax3.fill(angles, values, alpha=0.05)
    ax3.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    _save_path = f"{save}_radar.png" if save else None
    figs["radar"] = _save_and_close(fig3, _save_path)

    # ── 5. Error-rate bar chart (FDR & FNR) ─────────────────────────
    err_df = df[["fdr", "fnr"]]
    fig4, ax4 = plt.subplots(figsize=figsize or (max(8, n_models * 1.2), 4))
    err_df.plot.bar(ax=ax4, color=["#e74c3c", "#e67e22"], width=0.7)
    ax4.set_ylabel("Rate")
    ax4.set_ylim(0, max(0.5, err_df.max().max() * 1.2))
    ax4.legend(["FDR", "FNR"], fontsize=9)
    ax4.tick_params(axis="x", rotation=45)
    _save_path = f"{save}_error_rates.png" if save else None
    figs["error_rates"] = _save_and_close(fig4, _save_path)

    # ── 6. Ranked bar chart by F1 ──────────────────────────────────
    ranked = df["f1"].sort_values(ascending=True)
    colours = ["#2ecc71" if m == best_name else "steelblue" for m in ranked.index]
    fig5, ax5 = plt.subplots(figsize=figsize or (8, max(3, n_models * 0.5)))
    ax5.barh(range(len(ranked)), ranked.values, color=colours)
    ax5.set_yticks(range(len(ranked)))
    ax5.set_yticklabels(ranked.index)
    ax5.set_xlabel("Macro F1")
    ax5.set_xlim(0, 1.05)
    _save_path = f"{save}_f1_ranking.png" if save else None
    figs["f1_ranking"] = _save_and_close(fig5, _save_path)

    # ── 7. Confusion matrices ──────────────────────────────────────
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    fig6, axes = plt.subplots(n_rows, n_cols,
                              figsize=figsize or (5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, (name, res) in enumerate(all_results.items()):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        cm = res["metrics"].get("confusion_matrix")
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                        cbar=False)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    for idx in range(n_models, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)
    _save_path = f"{save}_confusion_matrices.png" if save else None
    figs["confusion_matrices"] = _save_and_close(fig6, _save_path)

    # ── 8. MCC & Cohen's Kappa comparison ──────────────────────────
    mk_df = df[["mcc", "cohen_kappa"]].dropna()
    if not mk_df.empty:
        fig7, ax7 = plt.subplots(figsize=figsize or (max(8, n_models * 1.2), 4))
        mk_df.plot.bar(ax=ax7, color=["#3498db", "#9b59b6"], width=0.7)
        ax7.set_ylabel("Score")
        ax7.legend(fontsize=9)
        ax7.tick_params(axis="x", rotation=45)
        _save_path = f"{save}_mcc_kappa.png" if save else None
        figs["mcc_kappa"] = _save_and_close(fig7, _save_path)

    return figs


def compute_label_quality_score(
    adata: ad.AnnData,
    cell_type_col: str = "CellType",
    confusion_cols: Optional[List[str]] = None,
    obs_key: str = "scintilla_label_quality",
    supervised_weight: float = 2.0,
    unsupervised_weight: float = 1.0,
    force: bool = False,
) -> pd.Series:
    """Compute a per-cell label quality score and store it in ``adata.obs``.

    The score is computed per cell type (from confusion and consistency
    metrics), then mapped back to every cell.  If ``adata.obs[obs_key]``
    already exists the computation is skipped unless *force* is True.

    Supervised metrics (``pred_agreement``, ``pred_entropy``,
    ``pred_avg_confidence``) receive *supervised_weight* while unsupervised
    confusion columns receive *unsupervised_weight*.  The final score is
    the weighted average across all metrics.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data object.
    cell_type_col : str
        Column in ``adata.obs`` with cell-type labels.
    confusion_cols : list of str, optional
        Columns with confusion scores.  Auto-detected when *None*.
    obs_key : str
        Column name used to store the score in ``adata.obs``.
    supervised_weight : float
        Weight for supervised metrics (agreement, entropy, confidence).
    unsupervised_weight : float
        Weight for unsupervised confusion columns.
    force : bool
        Recompute even if the score already exists.

    Returns
    -------
    pd.Series
        Per-cell-type quality score (index = cell type, sorted ascending).
    """
    if obs_key in adata.obs.columns and not force:
        quality_map = adata.obs.groupby(cell_type_col)[obs_key].first()
        return quality_map.sort_values(ascending=True)

    obs = adata.obs

    # Auto-detect confusion columns
    if confusion_cols is None:
        confusion_cols = sorted([c for c in obs.columns if c.startswith("scintilla_top") and c.endswith("_confusion")])

    # Gather all metric columns
    consistency_cols = []
    for c in ["pred_agreement", "pred_entropy", "pred_avg_confidence"]:
        if c in obs.columns:
            consistency_cols.append(c)

    all_metric_cols = confusion_cols + consistency_cols
    if not all_metric_cols:
        raise ValueError("No confusion or consistency columns found in adata.obs.")

    # Group by cell type and compute means
    grouped = obs.groupby(cell_type_col)[all_metric_cols].mean()

    # Normalise each metric to [0, 1] for composite scoring
    normed = grouped.copy()
    for col in normed.columns:
        vals = normed[col]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normed[col] = (vals - vmin) / (vmax - vmin)
        else:
            normed[col] = 0.0

    # Compute quality score: high = good label quality
    # Supervised metrics get higher weight than unsupervised confusion
    quality = pd.Series(0.0, index=normed.index)
    total_weight = 0.0

    for col in confusion_cols:
        quality += unsupervised_weight * (1.0 - normed[col])
        total_weight += unsupervised_weight
    if "pred_entropy" in normed.columns:
        quality += supervised_weight * (1.0 - normed["pred_entropy"])
        total_weight += supervised_weight
    if "pred_agreement" in normed.columns:
        quality += supervised_weight * normed["pred_agreement"]
        total_weight += supervised_weight
    if "pred_avg_confidence" in normed.columns:
        quality += supervised_weight * normed["pred_avg_confidence"]
        total_weight += supervised_weight

    quality /= total_weight

    # Map per-cell-type score back to every cell in obs
    adata.obs[obs_key] = adata.obs[cell_type_col].map(quality).astype(float)

    return quality.sort_values(ascending=True)


def plot_celltype_label_quality(
    adata: ad.AnnData,
    cell_type_col: str = "CellType",
    confusion_cols: Optional[List[str]] = None,
    cmap: str = "coolwarm",
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
    obs_key: str = "scintilla_label_quality",
    supervised_weight: float = 2.0,
    unsupervised_weight: float = 1.0,
    force: bool = False,
) -> plt.Figure:
    """Rank cell types by label quality using confusion and consistency metrics.

    Computes a composite label quality score and stores it in
    ``adata.obs[obs_key]`` (skipped if already present, unless *force*).
    """
    quality = compute_label_quality_score(
        adata,
        cell_type_col=cell_type_col,
        confusion_cols=confusion_cols,
        obs_key=obs_key,
        supervised_weight=supervised_weight,
        unsupervised_weight=unsupervised_weight,
        force=force,
    )

    # Rebuild grouped metrics for the heatmap panel
    obs = adata.obs
    if confusion_cols is None:
        confusion_cols = sorted([c for c in obs.columns if c.startswith("scintilla_top") and c.endswith("_confusion")])
    consistency_cols = [c for c in ["pred_agreement", "pred_entropy", "pred_avg_confidence"] if c in obs.columns]
    all_metric_cols = confusion_cols + consistency_cols
    grouped = obs.groupby(cell_type_col)[all_metric_cols].mean()

    # ── Plot ────────────────────────────────────────────────────────
    n_types = len(quality)
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize or (14, max(4, n_types * 0.4)),
        gridspec_kw={"width_ratios": [1, 1.5]},
    )

    # Left: composite quality score bar chart
    colors = plt.cm.RdYlGn(quality.values)
    ax1.barh(range(n_types), quality.values, color=colors)
    ax1.set_yticks(range(n_types))
    ax1.set_yticklabels(quality.index, fontsize=9)
    ax1.set_xlabel("Label Quality Score")
    ax1.set_xlim(0, 1.05)
    ax1.axvline(x=0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Right: heatmap of raw mean metrics per cell type
    heatmap_data = grouped.loc[quality.index[::-1]]
    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap=cmap,
        linewidths=0.5, ax=ax2, cbar_kws={"shrink": 0.8},
    )
    ax2.set_ylabel("")

    return _save_and_close(fig, save)


def plot_metric_correlation(
    adata: ad.AnnData,
    metric_x: str,
    metric_y: str,
    cell_type_col: str = "CellType",
    label_dots: bool = True,
    cmap: str = "tab20",
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Scatter plot of two metrics aggregated per cell type with correlation.

    Each point is one cell type (mean of the metric across its cells).
    """
    from scipy.stats import pearsonr, spearmanr

    grouped = adata.obs.groupby(cell_type_col)[[metric_x, metric_y]].mean()
    x = grouped[metric_x].values
    y = grouped[metric_y].values
    labels = grouped.index.tolist()

    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=figsize or (10, 7))

    colourmap = plt.cm.get_cmap(cmap, len(labels))
    for i, label in enumerate(labels):
        ax.scatter(x[i], y[i], color=colourmap(i), s=80, edgecolors="black",
                   linewidths=0.5, zorder=3, label=label)

    # Use adjustText to avoid overlapping labels if available
    if label_dots:
        try:
            from adjustText import adjust_text
            texts = []
            for i, label in enumerate(labels):
                texts.append(ax.text(x[i], y[i], label, fontsize=7))
            adjust_text(texts, x=x, y=y, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except ImportError:
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]), fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")

    # Regression line
    z = np.polyfit(x, y, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p_fit(x_line), color="red", linewidth=1.5, linestyle="--",
            alpha=0.7, zorder=1)

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)

    # Show correlation in a text box instead of title
    ax.text(0.02, 0.98,
            f"Pearson r={r_p:.3f} (p={p_p:.2e})\nSpearman ρ={r_s:.3f} (p={p_s:.2e})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Legend outside the plot
    ax.legend(fontsize=7, markerscale=0.8, bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0, framealpha=0.8)

    return _save_and_close(fig, save)


def plot_celltype_confusion(
    adata: ad.AnnData,
    cell_type_col: str = "CellType",
    cell_types: Optional[List[str]] = None,
    min_count: int = 0,
    normalise: bool = True,
    cmap: str = "Reds",
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Show per-cell-type prediction confusion across classifiers (off-diagonal only).

    Parameters
    ----------
    min_count : int
        Minimum raw misclassification count for a cell to be shown.
        Cells below this threshold are zeroed out *before* normalisation,
        so they won't appear in the heatmap.  Default 0 (no filtering).
    """
    obs = adata.obs
    pred_cols = sorted([c for c in obs.columns
                        if c.startswith("pred_") and not c.endswith("_confidence")
                        and c not in ("pred_consensus", "pred_agreement",
                                      "pred_entropy", "pred_avg_confidence")])

    if not pred_cols:
        raise ValueError("No pred_* columns found. Run supervised_analysis with check_consistency=True first.")

    all_types = sorted(obs[cell_type_col].unique().astype(str))
    if cell_types is None:
        cell_types = all_types

    # Build raw count matrix: rows = true type, cols = predicted type
    conf = pd.DataFrame(0.0, index=cell_types, columns=all_types)
    for ct in cell_types:
        mask = obs[cell_type_col].astype(str) == ct
        sub = obs.loc[mask, pred_cols]
        counts = {}
        for col in pred_cols:
            for v in sub[col].astype(str):
                counts[v] = counts.get(v, 0) + 1
        for pred_type, count in counts.items():
            if pred_type in conf.columns:
                conf.loc[ct, pred_type] = count

    # Zero out the diagonal (correct predictions)
    for ct in cell_types:
        if ct in conf.columns:
            conf.loc[ct, ct] = 0.0

    # Filter out cells below the minimum count threshold
    if min_count > 0:
        conf = conf.where(conf >= min_count, 0.0)

    # Normalise per row: of all misclassifications, what fraction goes where
    if normalise:
        row_sums = conf.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # avoid division by zero
        conf = conf.div(row_sums, axis=0)

    # Drop columns and rows with no confusion
    conf = conf.loc[:, (conf > 0).any(axis=0)]
    conf = conf.loc[(conf > 0).any(axis=1), :]

    if conf.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No confusion detected", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return _save_and_close(fig, save)

    n_r, n_c = conf.shape
    fig, ax = plt.subplots(figsize=figsize or (max(8, n_c * 0.8), max(6, n_r * 0.6)))
    sns.heatmap(
        conf, annot=True, fmt=".2f" if normalise else ".0f",
        cmap=cmap, linewidths=0.5, ax=ax,
        vmin=0, vmax=None,
        cbar_kws={"label": "Fraction" if normalise else "Count"},
    )
    ax.set_xlabel("Predicted Cell Type")
    ax.set_ylabel("True Cell Type")

    return _save_and_close(fig, save)


def plot_celltype_confusion_network(
    adata: ad.AnnData,
    cell_type_col: str = "CellType",
    cell_types: Optional[List[str]] = None,
    min_confusion: int = 1,
    cmap: str = "tab20",
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Network graph of cell-type confusion from classifier predictions.

    Nodes are cell types; edges connect cell types that are confused
    with each other.  Edge thickness is proportional to the total number
    of misclassifications between the two cell types (summed in both
    directions).  Node size scales with total confusion count.

    Parameters
    ----------
    adata : ad.AnnData
        Must contain ``pred_*`` columns from ``supervised_analysis``.
    cell_type_col : str
        Column with ground-truth cell-type labels.
    cell_types : list of str, optional
        Subset of cell types to include.  Default: all.
    min_confusion : int
        Minimum total confusion count for an edge to be drawn.
    cmap : str
        Matplotlib colormap for node colours.
    save : str, optional
        Path to save the figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    plt.Figure
    """
    import networkx as nx

    obs = adata.obs
    pred_cols = sorted([c for c in obs.columns
                        if c.startswith("pred_") and not c.endswith("_confidence")
                        and c not in ("pred_consensus", "pred_agreement",
                                      "pred_entropy", "pred_avg_confidence")])

    if not pred_cols:
        raise ValueError("No pred_* columns found. Run supervised_analysis with check_consistency=True first.")

    all_types = sorted(obs[cell_type_col].unique().astype(str))
    if cell_types is None:
        cell_types = all_types

    # Build raw confusion count matrix (not normalised, not zeroed diagonal yet)
    conf = pd.DataFrame(0, index=cell_types, columns=all_types)
    for ct in cell_types:
        mask = obs[cell_type_col].astype(str) == ct
        sub = obs.loc[mask, pred_cols]
        counts: Dict[str, int] = {}
        for col in pred_cols:
            for v in sub[col].astype(str):
                counts[v] = counts.get(v, 0) + 1
        for pred_type, count in counts.items():
            if pred_type in conf.columns:
                conf.loc[ct, pred_type] = count

    # Zero diagonal
    for ct in cell_types:
        if ct in conf.columns:
            conf.loc[ct, ct] = 0

    # Collect all cell types involved in confusion (selected + their confused partners)
    involved_types = set(cell_types)
    for ct in cell_types:
        for other in all_types:
            if other == ct:
                continue
            weight = int(conf.loc[ct, other]) if other in conf.columns else 0
            if weight >= min_confusion:
                involved_types.add(other)

    # Also build confusion for the reverse direction (other -> selected cell types)
    # so edges are symmetric even when only a subset is selected
    conf_reverse = pd.DataFrame(0, index=list(involved_types - set(cell_types)), columns=all_types)
    for ct in conf_reverse.index:
        mask = obs[cell_type_col].astype(str) == ct
        sub = obs.loc[mask, pred_cols]
        counts: Dict[str, int] = {}
        for col in pred_cols:
            for v in sub[col].astype(str):
                counts[v] = counts.get(v, 0) + 1
        for pred_type, count in counts.items():
            if pred_type in conf_reverse.columns:
                conf_reverse.loc[ct, pred_type] = count
        if ct in conf_reverse.columns:
            conf_reverse.loc[ct, ct] = 0

    # Build symmetric edge weights between all involved types
    G = nx.Graph()
    involved_list = sorted(involved_types)
    for ct in involved_list:
        G.add_node(ct)

    for i, ct_a in enumerate(involved_list):
        for ct_b in involved_list[i + 1:]:
            weight = 0
            if ct_a in conf.index and ct_b in conf.columns:
                weight += int(conf.loc[ct_a, ct_b])
            if ct_a in conf_reverse.index and ct_b in conf_reverse.columns:
                weight += int(conf_reverse.loc[ct_a, ct_b])
            if ct_b in conf.index and ct_a in conf.columns:
                weight += int(conf.loc[ct_b, ct_a])
            if ct_b in conf_reverse.index and ct_a in conf_reverse.columns:
                weight += int(conf_reverse.loc[ct_b, ct_a])
            if weight >= min_confusion:
                G.add_edge(ct_a, ct_b, weight=weight)

    # Remove isolated nodes (no confusion above threshold)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if G.number_of_edges() == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No confusion above threshold", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return _save_and_close(fig, save)

    # Layout
    pos = nx.spring_layout(G, k=2.0 / max(np.sqrt(G.number_of_nodes()), 1), seed=42)

    # Node sizes: proportional to total confusion involving that node
    node_confusion = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    max_node_conf = max(node_confusion.values()) if node_confusion else 1
    node_sizes = [300 + 2000 * (node_confusion[n] / max_node_conf) for n in G.nodes()]

    # Edge widths: proportional to weight
    edge_weights = [d["weight"] for _, _, d in G.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 6.0 * (w / max_weight) for w in edge_weights]

    # Node colours from colourmap
    colourmap = plt.cm.get_cmap(cmap)
    node_list = list(G.nodes())
    node_colours = [colourmap(i / max(len(node_list) - 1, 1)) for i in range(len(node_list))]
    # Draw
    n_nodes = G.number_of_nodes()
    fig, ax = plt.subplots(figsize=figsize or (max(10, n_nodes * 0.6), max(8, n_nodes * 0.5)))

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.5, edge_color="#888888")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colours, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    # Edge labels with counts
    edge_labels = {(u, v): str(d["weight"]) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7, font_color="#444444")

    ax.set_axis_off()
    ax.set_title("Cell-Type Confusion Network", fontsize=13, fontweight="bold")

    return _save_and_close(fig, save)
