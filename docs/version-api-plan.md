# NanoTracker version API plan

Expose the installed package version through a public `nanotracker.version()` function and `NanoTracker.version()` class method. Read the value from installed distribution metadata, which is built from the version in `pyproject.toml`, rather than maintaining another version constant or parsing TOML at runtime.

Add `--version` to the `nanotracker-demo` command so it prints the version and exits. Normal demo startup prints a `NanoTracker <version>` banner before opening the video.

Acceptance: both Python APIs return the same distribution version, `nanotracker-demo --version` exits without opening a video, and regular demo launch prints the banner.
