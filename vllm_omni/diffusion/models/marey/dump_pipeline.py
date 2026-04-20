# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug subclass of MareyPipeline with dump/load instrumentation.

See ``vllm_omni/diffusion/debug/dump.py`` for the model-agnostic mixin and
``vllm_omni/diffusion/models/marey/pipeline_marey.py`` for the production
pipeline. Pre/post process functions are re-exported so the registry's
``_load_process_func`` lookup resolves correctly when ``DumpMareyPipeline``
is selected via ``--model-class-name``.
"""

from vllm_omni.diffusion.debug.dump import DumpMixin

from .pipeline_marey import (
    MareyPipeline,
    get_marey_post_process_func,  # noqa: F401  (re-exported for registry)
    get_marey_pre_process_func,  # noqa: F401  (re-exported for registry)
)


class DumpMareyPipeline(DumpMixin, MareyPipeline):
    """MareyPipeline with dump/load instrumentation.

    Behaviour-identical to ``MareyPipeline`` when none of these env vars
    are set:

      - ``MAREY_DUMP_DIR``
      - ``MAREY_LOAD_INITIAL_NOISE``
      - ``MAREY_LOAD_STEP_NOISE_DIR``

    See ``DumpMixin`` for details.
    """

    pass
