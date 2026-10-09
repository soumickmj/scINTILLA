"""Logging for scintilla.

Library code never calls :func:`print`.  Progress and diagnostic messages go
through the ``scintilla`` logger, whose level is controlled by
:data:`scintilla.settings.verbosity` or, for a single call, by the ``verbose``
argument that the pipeline functions accept.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger("scintilla")

_HANDLER_NAME = "scintilla-default"


def _install_default_handler() -> None:
    """Attach a plain stderr handler once, so INFO messages are visible on request."""
    if any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False


_install_default_handler()


def resolve_verbose(verbose: bool | None = None, config: Any = None) -> bool:
    """Resolve a ``verbose`` argument to a bool.

    Order of precedence: the explicit argument, then ``config.verbose``, then
    whether the logger currently emits INFO messages.
    """
    if verbose is None and config is not None:
        verbose = getattr(config, "verbose", None)
    if verbose is None:
        return logger.isEnabledFor(logging.INFO)
    return bool(verbose)


@contextmanager
def verbosity_override(verbose: bool | None) -> Iterator[None]:
    """Temporarily force the logger to INFO (``True``) or WARNING (``False``)."""
    if verbose is None:
        yield
        return
    previous = logger.level
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


def verbosity_aware(func: F) -> F:
    """Make a function honour its ``verbose`` (and ``config.verbose``) argument.

    The wrapped function still receives ``verbose`` unchanged; the decorator
    only sets the logger level for the duration of the call.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        verbose = bound.arguments.get("verbose")
        config = bound.arguments.get("config")
        if verbose is None and config is not None:
            verbose = getattr(config, "verbose", None)
        with verbosity_override(verbose):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
