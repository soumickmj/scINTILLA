"""Shared plumbing for the plotting API (``scintilla.pl``)."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _save(out: Any, save: str) -> None:
    path = Path(save)
    if isinstance(out, Figure):
        out.savefig(path, dpi=300, bbox_inches="tight")
    elif isinstance(out, dict):
        for key, fig in out.items():
            if isinstance(fig, Figure):
                fig.savefig(path.with_name(f"{path.stem}_{key}{path.suffix}"), dpi=300, bbox_inches="tight")
    elif hasattr(out, "figure"):
        out.figure.savefig(path, dpi=300, bbox_inches="tight")


def plot_api(func: Callable) -> Callable:
    """Give a plotting function the scanpy-style ``show`` and ``save`` arguments.

    The function itself is unchanged.  The wrapper returns what the function returned
    (a figure, axes or dict of figures) and never closes it: whether to display or
    close a figure is the caller's decision.  Functions that already define ``show``
    or ``save`` are left alone.

    Parameters
    ----------
    func
        Plotting function returning a figure.
    """
    signature = inspect.signature(func)
    has_save = "save" in signature.parameters
    has_show = "show" in signature.parameters
    if has_save and has_show:
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, show: Optional[bool] = None, save: Optional[str] = None, **kwargs: Any) -> Any:
        if has_save and save is not None:
            kwargs["save"] = save
            save = None
        out = func(*args, **kwargs)
        if save is not None:
            _save(out, save)
        if show:
            plt.show()
        return out

    params = list(signature.parameters.values())
    extra = []
    if not has_show:
        extra.append(inspect.Parameter("show", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[bool]))
    if not has_save:
        extra.append(inspect.Parameter("save", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=Optional[str]))
    var_kw = [p for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
    others = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
    if has_save:
        # ``save`` already exists on the wrapped function, keep its own definition.
        wrapper.__signature__ = signature.replace(parameters=others + extra + var_kw)
    else:
        wrapper.__signature__ = signature.replace(parameters=others + extra + var_kw)
    wrapper.__doc__ = (func.__doc__ or "") + (
        "\n\n    Other Parameters\n    ----------------\n"
        "    show\n        Call :func:`matplotlib.pyplot.show` after drawing.\n"
        "    save\n        Path to save the figure to (dpi 300); a dict of figures is saved with the key as suffix.\n"
    )
    return wrapper
