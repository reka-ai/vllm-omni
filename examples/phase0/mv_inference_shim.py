#!/usr/bin/env python3
"""Phase 0 torchrun entrypoint for moonvalley_ai Marey inference.

Problem: `main` branch does not expose `--offload-diffusion`/`--offload-vae`/
`--offload-text-encoder` typer options (those were added by the commit
9486523b9 that lives on `marey-serving-comparison`). Without offload, 30B
weights don't fit on 8x80GB GPUs with `num-seq-parallel-splits=8`.

Solution: monkey-patch `MareyInference.__init__` to force all three offload
flags True regardless of how the underlying code exposes them. Both branches
end up running with the same memory strategy, preserving within-codebase
Phase 0 parity.

Usage: invoked as the torchrun target by `run_moonvalley.sh`. Forwards its
argv to marey_inference's typer app. Drop after Phase 0 closes along with
the temp latent-dump hook.
"""

import sys


def _patch_and_run() -> None:
    import marey_inference

    # Patch 1 — force offload kwargs on so the 30B model fits on 8x80GB.
    # Main's CLI doesn't expose --offload-*; its MareyInference.__init__
    # accepts them internally.
    _orig_init = marey_inference.MareyInference.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["offload_diffusion"] = True
        kwargs["offload_vae"] = True
        kwargs["offload_text_encoder"] = True
        return _orig_init(self, *args, **kwargs)

    marey_inference.MareyInference.__init__ = _patched_init

    # Patch 2 — main's CLI at marey_inference.py:infer() still passes
    # first_frame_img_path / control_condition_video_path / control_type /
    # lora_path into MareyInference.infer(), but that method no longer
    # accepts those kwargs (they were replaced with structured dicts like
    # image_references/controls/loras). Strip them silently so pure
    # text-to-video keeps working across branches.
    _STALE_INFER_KWARGS = (
        "first_frame_img_path",
        "control_condition_video_path",
        "control_type",
        "lora_path",
    )
    _orig_infer = marey_inference.MareyInference.infer

    def _patched_infer(self, *args, **kwargs):
        for k in _STALE_INFER_KWARGS:
            kwargs.pop(k, None)
        return _orig_infer(self, *args, **kwargs)

    marey_inference.MareyInference.infer = _patched_infer

    marey_inference.app()


if __name__ == "__main__":
    _patch_and_run()
