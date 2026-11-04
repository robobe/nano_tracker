"""Public exceptions raised by NanoTracker."""


class NanoTrackerError(Exception):
    """Base class for errors raised by the NanoTracker library."""


class ModelNotFoundError(NanoTrackerError, FileNotFoundError):
    """A model directory or required model file is missing."""


class EngineUnavailableError(NanoTrackerError, RuntimeError):
    """The selected inference engine or required runtime is unavailable."""


class InvalidImageError(NanoTrackerError, ValueError):
    """An image does not have the expected format."""


class TrackerNotInitializedError(NanoTrackerError, RuntimeError):
    """Tracking was requested before initialization."""


class InferenceError(NanoTrackerError, RuntimeError):
    """The inference runtime failed while executing a model."""
