class DataConfigError(ValueError):
    """Raised when a versioned data configuration violates the frozen protocol."""


class DataValidationError(RuntimeError):
    """Raised when raw data or a generated manifest fails a blocking check."""
