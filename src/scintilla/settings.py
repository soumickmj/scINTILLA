"""Global settings, in the style of ``scanpy.settings``.

Example
-------
>>> import scintilla as si
>>> si.settings.verbosity = "info"   # show progress messages
"""

from __future__ import annotations

import logging

from scintilla._logging import logger

_LEVELS = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "hint": logging.INFO,
    "debug": logging.DEBUG,
}


class Settings:
    """Run-wide options.

    Seeds are deliberately *not* a global setting: pass ``random_state`` to the
    call (or ``AnalysisConfig.random_seed`` to a pipeline), so every result can be
    traced to the arguments that produced it.

    Attributes
    ----------
    verbosity
        One of ``"error"``, ``"warning"`` (default), ``"info"`` or ``"debug"``.
        Controls the ``scintilla`` logger.  Library code is silent at the
        default level; set ``"info"`` to see progress messages.
    dense_warning_gb
        Algorithms that cannot work on sparse data densify the matrix they are given. A
        :class:`UserWarning` is emitted before a dense copy larger than this many GiB
        (default 4) is allocated.
    """

    def __init__(self) -> None:
        self.dense_warning_gb: float = 4.0

    @property
    def verbosity(self) -> str:
        """Name of the active level: ``"error"``, ``"warning"``, ``"info"`` or ``"debug"``."""
        level = logger.level
        for name in ("debug", "info", "warning", "error"):
            if level <= _LEVELS[name]:
                return name
        return "error"

    @verbosity.setter
    def verbosity(self, value: str | int) -> None:
        if isinstance(value, str):
            try:
                level = _LEVELS[value.lower()]
            except KeyError:
                raise ValueError(f"verbosity must be one of {sorted(_LEVELS)}, got {value!r}") from None
        else:
            level = int(value)
        logger.setLevel(level)

    def __repr__(self) -> str:
        """Return a short description of the settings."""
        return f"Settings(verbosity={self.verbosity!r})"


settings = Settings()
