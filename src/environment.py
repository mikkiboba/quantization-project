from __future__ import annotations

import platform
import subprocess
from typing import Any

import torch


def _get_macos_version() -> str:
    """
    Return the macOS version.
    """

    try:
        result = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "unknown"


def _get_hardware_info() -> dict[str, str]:
    """
    Get non-sensitive hardware information from system_profiler.

    We deliberately do NOT collect:
        - serial number
        - hardware UUID
        - provisioning UDID
        - activation information
    """

    info = {
        "Machine": "unknown",
        "Model identifier": "unknown",
        "Chip": "unknown",
        "CPU cores": "unknown",
        "Memory": "unknown",
    }

    try:
        result = subprocess.run(
            [
                "system_profiler",
                "SPHardwareDataType",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.splitlines():

            line = line.strip()

            if line.startswith("Model Name:"):
                info["Machine"] = (
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith("Model Identifier:"):
                info["Model identifier"] = (
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith("Chip:"):
                info["Chip"] = (
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith(
                "Total Number of Cores:"
            ):
                info["CPU cores"] = (
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith("Memory:"):
                info["Memory"] = (
                    line.split(":", 1)[1].strip()
                )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        pass

    return info


def get_environment_info() -> dict[str, Any]:
    """
    Collect reproducibility metadata for the benchmark.
    """

    hardware = _get_hardware_info()

    return {
        # Hardware.
        "Machine": hardware["Machine"],
        "Model identifier": hardware[
            "Model identifier"
        ],
        "Chip": hardware["Chip"],
        "CPU cores": hardware["CPU cores"],
        "System memory": hardware["Memory"],

        # Operating system.
        "Operating system": platform.platform(),
        "macOS version": _get_macos_version(),

        # Python/runtime.
        "Python version": platform.python_version(),
        "Python architecture": platform.machine(),

        # PyTorch.
        "PyTorch version": torch.__version__,
        "MPS available": torch.backends.mps.is_available(),

        # Device.
        "CUDA available": torch.cuda.is_available(),
    }