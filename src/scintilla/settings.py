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
    """

    @property
    def verbosity(self) -> str:
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
        return f"Settings(verbosity={self.verbosity!r})"


settings = Settings()
