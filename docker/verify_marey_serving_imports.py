#!/usr/bin/env python3
"""Verify the serving image can import the Marey/OpenSora VAE path directly."""

from __future__ import annotations

import os
import sys


def _main() -> int:
    print("MOONVALLEY_AI_ROOT=", os.environ.get("MOONVALLEY_AI_ROOT", "<unset>"))
    print("PYTHONPATH=", os.environ.get("PYTHONPATH", "<unset>"))
    print("sys.path[:4]=", sys.path[:4])

    import common  # noqa: F401
    import opensora  # noqa: F401
    from opensora.models.vae.vae_adapters import PretrainedSpatioTemporalVAETokenizer  # noqa: F401

    print("Marey serving imports verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
