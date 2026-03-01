import shutil
import sys
from setuptools import setup
from setuptools_rust import RustBin


def _rust_extensions():
    if shutil.which("cargo") is None:
        return []

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
