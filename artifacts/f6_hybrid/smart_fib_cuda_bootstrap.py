"""Prepare the CUDA Toolkit DLL search order before PyTorch/Numba imports."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


DEFAULT_CUDA_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9")


def configure_cuda_toolkit(cuda_home: str | Path | None = None) -> Path | None:
    """Expose the toolkit and preload its nvJitLink before PyTorch loads DLLs."""
    home = Path(cuda_home or os.environ.get("CUDA_HOME", DEFAULT_CUDA_HOME))
    if not home.exists():
        return None
    os.environ.setdefault("CUDA_HOME", str(home))
    os.environ.setdefault("CUDA_PATH", str(home))
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for path in (home / "bin", home / "nvvm" / "bin"):
        text = str(path)
        if text not in path_parts:
            path_parts.insert(0, text)
    os.environ["PATH"] = os.pathsep.join(path_parts)

    nvjitlink = home / "bin" / "nvJitLink_120_0.dll"
    if nvjitlink.exists():
        try:
            ctypes.WinDLL(str(nvjitlink))
        except OSError:
            pass
    return home


configure_cuda_toolkit()
