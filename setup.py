import os
import shutil
import sys
from setuptools import setup


def _rust_extensions():
    if not os.environ.get("BUILD_RUST"):
        return []

    if shutil.which("cargo") is None:
        print("WARNING: BUILD_RUST=1 set but cargo not found. Skipping Rust build.")
        return []

    from setuptools_rust import RustBin

    if sys.platform.startswith("linux"):
        print(
            "Note: building prompt-box on Linux requires X11 development headers.\n"
            "If compilation fails, install them first:\n"
            "  Debian/Ubuntu: sudo apt-get install libxcb-dev\n"
            "  Fedora/RHEL:   sudo dnf install libxcb-devel\n"
            "  Arch Linux:    sudo pacman -S libxcb\n"
            "prompt-box also requires xclip or xsel at runtime for clipboard support."
        )

    return [RustBin("prompt-box", "claude-prompt-box/Cargo.toml")]


setup(rust_extensions=_rust_extensions())
