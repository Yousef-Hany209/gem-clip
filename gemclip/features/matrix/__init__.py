"""Matrix feature facade.

Provides a stable import for the matrix batch processor UI without moving
the large implementation file yet.
"""
from .window import MatrixBatchProcessorWindow

__all__ = ["MatrixBatchProcessorWindow"]
