# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opensora spatiotemporal VAE loader for Marey.

Isolated here so both the legacy monolithic pipeline and the new
`MareyVaePipeline` stage share the same import-path bootstrap and
error-handling story.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import types
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def opensora_logging_guard():
    """Temporarily lower the root logger level while importing opensora.

    ``moonvalley_ai/common.logging_utils.get_logger`` raises ``ValueError``
    when the root logger's level is ``>= logging.ERROR`` (40) — a hydra
    misconfiguration check. Several opensora modules call ``get_logger`` at
    import time, so importing from a vllm-omni process (which runs at
    ERROR/CRITICAL) trips that guard.
    """
    root = logging.getLogger()
    saved = root.level
    if saved >= logging.ERROR:
        root.setLevel(logging.WARNING - 1)
    try:
        yield
    finally:
        root.setLevel(saved)


def resolve_moonvalley_dir() -> str:
    """Locate the ``moonvalley_ai`` directory.

    Resolution order:
        1. ``MOONVALLEY_AI_PATH`` environment variable.
        2. Derived from an editable ``opensora`` install, if importable.
        3. ``/workspace/moonvalley_ai`` (default Docker layout).
        4. Sibling of the vllm-omni repo root.
        5. Inside the vllm-omni repo root (legacy layout).
    """

    def _is_valid(candidate: Path) -> bool:
        return (candidate / "open_sora" / "opensora" / "models" / "vae").is_dir()

    env_root = os.environ.get("MOONVALLEY_AI_PATH")
    if env_root:
        candidate = Path(env_root).resolve()
        if _is_valid(candidate):
            return str(candidate)
        raise RuntimeError(
            f"MOONVALLEY_AI_PATH={env_root} does not contain open_sora/opensora/models/vae"
        )

    candidates: list[Path] = []
    try:
        import opensora as _os_pkg  # noqa: F401 (import for path discovery)
        candidates.append(Path(_os_pkg.__file__).resolve().parents[2])
    except ImportError as e:
        logger.debug("opensora not importable (%s); falling back to path-based lookup", e)
    except Exception:
        logger.warning(
            "opensora is importable but raised during import; "
            "falling back to path-based lookup",
            exc_info=True,
        )

    # this file lives at <repo>/vllm_omni/diffusion/models/marey/vae_loader.py,
    # so parents[4] is the repo root.
    repo_root = Path(__file__).resolve().parents[4]
    candidates.extend([
        Path("/workspace/moonvalley_ai"),
        repo_root.parent / "moonvalley_ai",
        repo_root / "moonvalley_ai",
    ])

    for candidate in candidates:
        if _is_valid(candidate):
            return str(candidate.resolve())

    raise RuntimeError(
        "Could not locate moonvalley_ai. Set MOONVALLEY_AI_PATH to the "
        "directory containing open_sora/opensora/models/vae, or place "
        "moonvalley_ai alongside the vllm-omni repo."
    )


def setup_opensora_imports() -> None:
    """Prepare sys.modules so the opensora VAE can be imported."""
    moonvalley_dir = resolve_moonvalley_dir()
    logger.info("Resolved moonvalley_ai path: %s", moonvalley_dir)
    opensora_pkg_root = str(Path(moonvalley_dir) / "open_sora")
    for candidate in (opensora_pkg_root, moonvalley_dir):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    for mod_name in (
        "opensora.models",
        "opensora.datasets",
        "opensora.datasets.utils",
        "opensora.datasets.video_transforms",
        "opensora.datasets.datasets",
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            stub.__path__ = []
            stub.__package__ = mod_name
            sys.modules[mod_name] = stub

    opensora_models_path = str(Path(opensora_pkg_root) / "opensora" / "models")
    sys.modules["opensora.models"].__path__ = [opensora_models_path]


def load_vae(
    vae_config: dict,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[nn.Module | None, str | None]:
    """Load the opensora spatiotemporal VAE from config.

    Returns ``(vae_module, error_msg)``. On success ``error_msg`` is None;
    on failure ``vae_module`` is None and ``error_msg`` describes the cause.
    """
    vae_path = vae_config.get("cp_path", "")
    if not os.path.exists(vae_path):
        return None, f"VAE checkpoint not found at {vae_path}."

    try:
        setup_opensora_imports()
        logger.info("Setup Opensora imports")
        with opensora_logging_guard():
            from opensora.models.vae.vae_adapters import PretrainedSpatioTemporalVAETokenizer
        logger.info(
            "Import Opensora, loading VAE from %s with vae_config: %s",
            vae_path,
            vae_config,
        )

        vae = PretrainedSpatioTemporalVAETokenizer(
            cp_path=vae_path,
            strict_loading=vae_config.get("strict_loading", False),
            extra_kwargs=vae_config.get("extra_kwargs", {"no_losses": True}),
            scaling_factor=vae_config.get("scaling_factor", 1.0),
            bias_factor=vae_config.get("bias_factor", 0.0),
            frame_chunk_len=vae_config.get("frame_chunk_len"),
            max_batch_size=vae_config.get("max_batch_size"),
            reuse_as_spatial_vae=vae_config.get("reuse_as_spatial_vae", False),
            extra_context_and_drop_strategy=vae_config.get("extra_context_and_drop_strategy", False),
            enable_sequence_parallelism=False,
        )
        vae = vae.to(device, dtype).eval()
        logger.info(
            "Loaded opensora VAE (out_channels=%s, downsample=%s)",
            vae.out_channels,
            vae.downsample_factors,
        )
        return vae, None
    except Exception as e:
        return None, f"Could not load opensora VAE from {vae_path}: {type(e).__name__}: {e}"
