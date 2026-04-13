# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

logger = init_logger(__name__)

MAREY_PIPELINE_CLASS_NAME = "MareyPipeline"
MAREY_PIPELINE_CONFIG_NAME = "config.yaml"


def configure_local_marey_pipeline(od_config: OmniDiffusionConfig) -> bool:
    """Short-circuit HF config probing for a local Marey checkpoint.

    Marey is configured directly from ``config.yaml`` in the checkpoint
    directory, so it should not be forced through the standard
    HF/diffusers ``config.json`` or ``transformer/config.json`` layout.

    Returns ``True`` when Marey was fully configured here and callers should
    skip the standard HF/diffusers config resolution path.
    """

    if od_config.model is None or od_config.model_class_name is None:
        return False

    if od_config.model_class_name != MAREY_PIPELINE_CLASS_NAME:
        return False

    config_path = os.path.join(od_config.model, MAREY_PIPELINE_CONFIG_NAME)
    if not os.path.isfile(config_path):
        raise ValueError(
            f"{od_config.model_class_name} requires a local checkpoint directory "
            f"containing '{MAREY_PIPELINE_CONFIG_NAME}': {od_config.model}"
        )

    od_config.tf_model_config = TransformerConfig()
    od_config.update_multimodal_support()
    logger.info("Using local %s for %s", MAREY_PIPELINE_CONFIG_NAME, od_config.model_class_name)
    return True
