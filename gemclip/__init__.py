"""Gem Clip package facades.

This package provides lightweight facades to group core, UI, and infra
modules without moving existing implementation files yet. It enables
incremental refactors while giving callers a stable import surface.
"""

from .core import *  # re-export core conveniences
from .ui import *    # re-export UI entry points
from .infra import * # re-export infra accessors

