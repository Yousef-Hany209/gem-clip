"""Matrix feature window class (transition shim + subclass).

We subclass the legacy window to provide a stable import path and allow
incremental overrides/migrations without breaking callers.
"""

from matrix_batch_processor import MatrixBatchProcessorWindow as _LegacyMatrixWindow


class MatrixBatchProcessorWindow(_LegacyMatrixWindow):
    """Subclass of the legacy window.

    Currently no overrides; migration proceeds by moving/overriding
    methods here step-by-step.
    """

    pass


__all__ = ["MatrixBatchProcessorWindow"]
