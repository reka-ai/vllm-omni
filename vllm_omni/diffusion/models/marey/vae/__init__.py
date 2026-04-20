# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""In-tree port of the Marey (opensora/moonvalley_ai) spatiotemporal VAE.

The public entry point is :class:`SpatioTemporalVAETokenizer`, which preserves
the surface that ``pipeline_marey_vae`` relies on (``decode``, ``out_channels``,
``downsample_factors``, ``frame_chunk_len``, ...). Sequence parallelism is
wired through vllm-omni's own ``sp_shard`` / ``sp_gather`` primitives.
"""

from .tokenizer import SpatioTemporalVAETokenizer
from .two_stage import TwoStageVAE

__all__ = ["SpatioTemporalVAETokenizer", "TwoStageVAE"]
