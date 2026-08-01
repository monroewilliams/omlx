# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Unified hardware detection for Apple Silicon.

Single source of truth for:
- Chip identification (M1, M2, M3, M4 series)
- Memory detection (total, available, max working set)
- MLX availability checks
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

# Default fallback value for all memory functions (conservative)
DEFAULT_MEMORY_BYTES = 8 * 1024 * 1024 * 1024  # 8GB

# Absolute paths to the macOS system tools we shell out to. They live in
# /usr/sbin, which is not on PATH in some headless launchd contexts (e.g.
# `brew services`). Looking them up by name there raises FileNotFoundError and
# silently degrades detection (chip falls back to M1, gpu/uuid to None), so we
# always invoke them by absolute path. See issue #1322.
_SYSCTL = "/usr/sbin/sysctl"
_SYSTEM_PROFILER = "/usr/sbin/system_profiler"
_IOREG = "/usr/sbin/ioreg"


@dataclass
class HardwareInfo:
    """Hardware information for Apple Silicon."""

    chip_name: str
    total_memory_gb: float
    max_working_set_bytes: int
    mlx_device_name: Optional[str] = None


# =============================================================================
# Core Detection Functions
# =============================================================================


def get_chip_name() -> str:
    """
    Get Apple Silicon chip name via sysctl.

    Returns:
        Chip name (e.g., "Apple M4 Pro") or "Apple Silicon" as fallback.
    """
    try:
        result = subprocess.run(
            [_SYSCTL, "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "Apple Silicon"


def get_total_memory_bytes() -> int:
    """
    Get total unified memory in bytes.

    Fallback chain:
    1. sysctl hw.memsize (most reliable)
    2. mlx.metal.device_info()["memory_size"]
    3. DEFAULT_MEMORY_BYTES (8GB)

    Returns:
        Total memory in bytes.
    """
    # Primary: sysctl
    try:
        result = subprocess.run(
            [_SYSCTL, "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        pass

    # Fallback: MLX Metal
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                device_info = mx.device_info()
                if "memory_size" in device_info:
                    return int(device_info["memory_size"])
        except Exception:
            pass

    # Last resort: default
    logger.warning(f"Using default memory size: {DEFAULT_MEMORY_BYTES // (1024**3)} GB")
    return DEFAULT_MEMORY_BYTES


def get_total_memory_gb() -> float:
    """Get total unified memory in GB."""
    return get_total_memory_bytes() / (1024**3)


def get_max_working_set_bytes() -> int:
    """
    Get max_recommended_working_set_size from MLX Metal.

    Fallback chain:
    1. mlx.metal.device_info()["max_recommended_working_set_size"]
    2. total system memory * 0.75
    3. DEFAULT_MEMORY_BYTES (8GB)

    Returns:
        Maximum working set size in bytes.
    """
    # Primary: MLX Metal
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                device_info = mx.device_info()
                max_working_set = device_info.get("max_recommended_working_set_size", 0)
                if max_working_set > 0:
                    return max_working_set
        except Exception:
            pass

    total_ram = get_total_memory_bytes()
    if total_ram > 0:
        return int(total_ram * 0.75)

    # Last resort: default
    logger.warning(
        f"Using default max working set: {DEFAULT_MEMORY_BYTES // (1024**3)} GB"
    )
    return DEFAULT_MEMORY_BYTES


def get_mlx_device_name() -> Optional[str]:
    """Get raw device name from MLX Metal API."""
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                device_info = mx.device_info()
                return device_info.get("device_name")
        except Exception:
            pass
    return None


def detect_hardware() -> HardwareInfo:
    """
    Detect Apple Silicon hardware and return complete info.

    Returns:
        HardwareInfo with all hardware specifications.
    """
    return HardwareInfo(
        chip_name=get_chip_name(),
        total_memory_gb=get_total_memory_gb(),
        max_working_set_bytes=get_max_working_set_bytes(),
        mlx_device_name=get_mlx_device_name(),
    )


# =============================================================================
# MLX Availability Checks
# =============================================================================


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (arm64 macOS)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_mlx_available() -> bool:
    """Check if MLX is available and working."""
    if not is_apple_silicon():
        return False
    if not HAS_MLX:
        return False

    try:
        # Verify we can actually use MLX
        _ = mx.array([1.0, 2.0, 3.0])
        return True
    except Exception:
        return False


# =============================================================================
# Version Information
# =============================================================================


def get_mlx_version() -> str:
    """Get MLX version string."""
    try:
        import mlx

        return getattr(mlx, "__version__", "Unknown")
    except Exception:
        return "Unknown"


def get_mlx_lm_version() -> str:
    """Get mlx-lm version string."""
    try:
        import mlx_lm

        return getattr(mlx_lm, "__version__", "Unknown")
    except Exception:
        return "Unknown"


def get_mlx_vlm_version() -> str:
    """Get mlx-vlm version string."""
    try:
        import mlx_vlm

        return getattr(mlx_vlm, "__version__", "Unknown")
    except Exception:
        return "Unknown"


# =============================================================================
# Benchmark / omlx.ai Integration
# =============================================================================

_OWNER_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def get_gpu_core_count() -> Optional[int]:
    """Get GPU core count via system_profiler."""
    try:
        result = subprocess.run(
            [_SYSTEM_PROFILER, "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "Total Number of Cores" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    return int(match.group(1))
    except Exception:
        pass
    return None


def get_io_platform_uuid() -> Optional[str]:
    """Get IOPlatformUUID from ioreg (unique per device)."""
    try:
        result = subprocess.run(
            [_IOREG, "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', line)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None


def parse_chip_info(chip_string: str) -> tuple[str, str]:
    """Parse chip name and variant from sysctl brand string.

    Args:
        chip_string: e.g. "Apple M4 Pro", "Apple M3 Max", "Apple M2"

    Returns:
        (chip_name, chip_variant) e.g. ("M4", "Pro"), ("M3", "Max"), ("M2", "")
    """
    match = re.search(r"M(\d+)\s*(Pro|Max|Ultra)?", chip_string)
    if not match:
        return ("M1", "")
    chip_name = f"M{match.group(1)}"
    chip_variant = match.group(2) or ""
    return (chip_name, chip_variant)


def compute_owner_hash(
    uuid: str, chip_name: str, gpu_cores: Optional[int], memory_gb: int
) -> str:
    """Compute owner_hash for omlx.ai benchmark submissions.

    Format: SHA-256(uuid + chip_name + gpu_cores + memory_gb) + verify_char
    The verify_char is ALPHABET[sum(charCodes of hash) % 36].

    Returns:
        Full owner_hash including verify character.
    """
    raw = f"{uuid}{chip_name}{gpu_cores}{memory_gb}"
    hash_hex = hashlib.sha256(raw.encode()).hexdigest()
    verify_sum = sum(ord(c) for c in hash_hex)
    verify_char = _OWNER_HASH_ALPHABET[verify_sum % 36]
    return hash_hex + verify_char


def get_os_version() -> str:
    """Get macOS version string (e.g. 'macOS 15.2')."""
    try:
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            return f"macOS {mac_ver}"
    except Exception:
        pass
    return "macOS"


# =============================================================================
# Utility Functions
# =============================================================================


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string (e.g., '16.00 GB')."""
    if bytes_value >= 1024**3:
        return f"{bytes_value / 1024**3:.2f} GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / 1024**2:.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} B"


# =============================================================================
# Compute weight estimation — ALL NUMBERS BELOW ARE MADE UP PLACEHOLDERS
# =============================================================================
# These values feed into `allocate_layers()` as a multiplier on ram_fraction
# to decide how many layers each node gets. They have NOT been benchmarked.
#
# TODO: Replace with real per-chip benchmarks (e.g. time-to-first-token for
# a standard layer on each Mac, or Geekbench 6 single-core GPU scores).
# The current numbers are guesses based on rough generation-overlap logic.
#
# Design notes (current heuristic, not backed by data):
#   - Normalized so the "most powerful current chip" = 1.0 (hardcoded
#     reference of M5 Max 40-core). Replace that reference with a real chip
#     once the cluster is finalized.
#   - Tier multiplier (Ultra/Max/Pro/Air/Base) is guessed: Ultra gets 2x
#     single-die because it's two dies, but the exact ratio is unknown.
#   - No M4 Ultra exists — this code path would be hit for future chips.

_GEN_UPFIFT: dict[str, float] = {  # ALL MADE UP
    "M1": 1.0,
    "M2": 1.15,
    "M3": 1.20,
    "M4": 1.25,
    "M5": 1.30,
}

_GPU_PER_CORE_UPFIT: dict[str, float] = {  # ALL MADE UP
    "M1": 1.0,
    "M2": 1.15,
    "M3": 1.20,
    "M4": 1.25,
    "M5": 1.30,
}


def get_compute_weight(chip_name: str, gpu_cores: int | None = None) -> float:
    """Estimate normalized compute weight for a chip.

    WARNING: All numbers in this function are made-up placeholders. See the
    module-level comments above.

    Args:
        chip_name: Full chip name, e.g. "M4 Pro", "M3 Max".
        gpu_cores: GPU core count from ``get_gpu_core_count()``.

    Returns:
        Normalized compute weight in [0.3, 1.0]. (Garbage output until
        replaced with real benchmarks.)
    """
    import re as _re

    if gpu_cores and gpu_cores > 0:
        gen_match = _re.search(r"(M\d+)", chip_name, _re.IGNORECASE)
        gen = gen_match.group(1).upper() if gen_match else "M1"

        upfit = _GPU_PER_CORE_UPFIT.get(gen, 1.0)
        raw_weight = gpu_cores * upfit

        # Reference is M5 Max 40-core — also a guess
        REFERENCE = 40.0 * _GPU_PER_CORE_UPFIT.get("M5", 1.3)
        return min(raw_weight / REFERENCE, 1.0)

    # No core count — fall back to generation-based default (also guessed)
    gen_match = _re.search(r"(M\d+)", chip_name, _re.IGNORECASE)
    gen = gen_match.group(1).upper() if gen_match else "M1"
    gen_upfit = _GEN_UPFIFT.get(gen, 1.0)

    tier_mult = 1.0
    for word in ["Ultra", "Max", "Pro", "Air", "Base"]:
        if word.lower() in chip_name.lower():
            tier_mult = {"Ultra": 1.0, "Max": 0.9, "Pro": 0.75, "Air": 0.55, "Base": 0.45}.get(word, 1.0)
            break

    if "Ultra" in chip_name.upper():
        tier_mult *= 2.0

    return min(gen_upfit * tier_mult / 1.3, 1.0)


def build_distributed_node_info(
    node_id: str,
) -> dict[str, Any]:
    """Build the node info dict for distributed /api/distributed-node-info.

    Args:
        node_id: This node's identifier (from DiscoveryService).

    Returns:
        Dict with keys suitable for layer allocation: node_id,
        ram_total_gb, ram_available_gb, chip_model, compute_weight.
    """
    from ..utils.psutil_compat import virtual_memory

    chip_string = get_chip_name()
    chip_name, chip_variant = parse_chip_info(chip_string)
    gpu_cores = get_gpu_core_count()

    vm = virtual_memory()
    ram_total_gb = round(vm.total / (1024**3), 1)
    ram_available_gb = round(vm.available / (1024**3), 1)

    return {
        "node_id": node_id,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "chip_model": f"Apple {chip_string}" if chip_string else "Unknown",
        "compute_weight": get_compute_weight(chip_name, gpu_cores),
    }
