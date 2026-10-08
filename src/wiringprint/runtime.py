"""Live access to notebook variables from shared code.

The notebooks used to define their helper functions at top level, so those functions silently read notebook
variables such as ``DEVICE`` or ``DTYPE`` as globals. When a helper moves into this package it must keep reading
the *notebook's* variable (not a stale copy), so call ``wiringprint.bind(globals())`` once near the top of a
notebook. Shared code then reads ``_nb.DEVICE`` etc. from that namespace at call time, exactly like a global
lookup did before. A missing variable raises ``NameError`` just as before.
"""
from __future__ import annotations


class _Late:
    """Sentinel default: 'use the notebook variable' (replaces a default such as ``device=DEVICE``).

    Python evaluates default values when ``def`` runs. In a notebook that was the moment the cell executed;
    in a package it would be import time, before the notebook has defined anything. The variable is therefore
    read when the function is *called* (checked: none of the notebooks reassigns it after the def)."""

    __slots__ = ()

    def __repr__(self):
        return "<notebook default>"


LATE = _Late()
_ns = None


def bind(namespace):
    """Tell shared code which namespace holds the notebook variables. Call as ``bind(globals())``."""
    global _ns
    _ns = namespace
    return namespace


class _Ambient:
    def __getattr__(self, name):
        if _ns is None:
            raise RuntimeError("wiringprint.bind(globals()) has not been called in this notebook")
        try:
            return _ns[name]
        except KeyError:
            raise NameError(f"name {name!r} is not defined in the notebook (shared code in wiringprint expects it)") from None

    def __dir__(self):
        return sorted(_ns or {})


ambient = _Ambient()
