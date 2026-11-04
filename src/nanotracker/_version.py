from importlib.metadata import version as distribution_version


def version() -> str:
    """Return the installed NanoTracker distribution version."""
    return distribution_version("nanotracker")
