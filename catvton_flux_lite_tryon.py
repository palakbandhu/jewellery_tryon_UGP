"""
catvton_flux_lite_tryon.py -- alternative Stage 3'/4' for the landmark-guided
jewellery try-on pipeline.

Replaces the current Stage 3 (rough affine warp) + Stage 4 (FLUX.1-Fill-dev +
ControlNet-Canny + IP-Adapter) with a single FLUX.1-Fill-dev pipeline plus a
small LoRA, conditioned by spatial concatenation instead of ControlNet
residuals / IP-Adapter cross-attention. See
similar_solutions_and_alternatives.md, section 2, for the full rationale.

Drop this file next to pipeline/landmark_guided_tryon.py and import Stage 0,
Stage 1, Stage 2 and Stage 5 from there unchanged:

    from landmark_guided_tryon import (
        extract_jewellery_asset,          # Stage 0
        anchor_ring, anchor_bracelet,      # Stage 1
        anchor_earring, anchor_necklace,
        build_inpaint_region_mask,        # Stage 2
        roi_box_from_mask,
        lab_color_match, maybe_upscale,   # Stage 5
    )

Everything below is un-run -- no CUDA hardware was available while writing
it. Syntax-checked only (see the bottom of this file). Treat the "What's
unverified" section at the end as load-bearing, not boilerplate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

try:
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
    from transformers import BitsAndBytesConfig as TransformersBnbConfig
except ImportError as e:
    raise ImportError(
        "This module needs diffusers>=0.32 (FluxFillPipeline + LoRA loading) "
        "and a recent transformers/bitsandbytes/peft, same as the main "
        "pipeline's Stage 4. pip install -U diffusers transformers "
        "bitsandbytes peft accelerate"
    ) from e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"

# 37.4M-parameter LoRA, trained on VITON-HD (clothing), released alongside
# https://github.com/nftblackmagic/catvton-flux
# UNVERIFIED for jewellery: this is the one real bet in this file. If
# results look garment-like / ignore fine reference detail, this LoRA not
# generalizing to jewellery-scale regions is the first thing to suspect --
# see "What's unverified" at the bottom.
LORA_REPO = "xiaozaa/catvton-flux-lora-alpha"

# catvton-flux's own reported VITON-HD settings; not tuned for jewellery.
DEFAULT_GUIDANCE_SCALE = 30.0
DEFAULT_STEPS = 30

CATEGORY_PROMPTS = {
    "ring": "a photo of a hand wearing a ring, photorealistic, same lighting",
    "bracelet": "a photo of a wrist wearing a bracelet, photorealistic, same lighting",
    "earring": "a photo of an ear wearing an earring, photorealistic, same lighting",
    "necklace": "a photo of a neck wearing a necklace, photorealistic, same lighting",
}


# ---------------------------------------------------------------------------
# Stage 3' -- side-by-side canvas (replaces warp_and_place)
# ---------------------------------------------------------------------------

@dataclass
class ConcatCanvas:
    canvas: Image.Image        # RGB, width = person_w + asset_w
    canvas_mask: Image.Image   # L, same size; 255 = inpaint, 0 = keep
    person_width: int          # pixel offset where the person half ends,
                                # needed to crop the result back out


def _letterbox_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize img to exactly target_h tall, preserving aspect ratio."""
    if img.height == 0:
        raise ValueError("jewellery_asset has zero height -- check Stage 0 output")
    scale = target_h / img.height
    new_w = max(1, round(img.width * scale))
    return img.resize((new_w, target_h), Image.LANCZOS)


def build_concat_canvas(
    person_crop: Image.Image,
    jewellery_asset: Image.Image,
    stage2_mask: np.ndarray,
) -> ConcatCanvas:
    """Stage 3' -- build the [person | jewellery] canvas and matching mask.

    person_crop:      padded ROI crop around the Stage 2 mask (RGB). Use the
                       existing roi_box_from_mask() from landmark_guided_tryon.py
                       to compute this crop -- unchanged from the current pipeline.
    jewellery_asset:   Stage 0's RGBA cutout (extract_jewellery_asset() output),
                        unchanged from the current pipeline.
    stage2_mask:       single-channel (H, W) uint8 array from
                        build_inpaint_region_mask(), same shape as person_crop.

    No affine warp happens here -- unlike the current Stage 3, this stage does
    not try to place the asset at the right scale/rotation itself. The model
    infers placement from the mask shape during denoising, the way CatVTON
    does for garments. This means your landmark-driven capsule mask from
    Stage 2 is still doing real work: it's what constrains placement to a
    small, correctly-shaped region instead of leaving the model to solve an
    unconstrained "where does this jewellery go" problem.
    """
    h, w = person_crop.height, person_crop.width
    if stage2_mask.shape[:2] != (h, w):
        raise ValueError(
            f"stage2_mask shape {stage2_mask.shape[:2]} does not match "
            f"person_crop size {(h, w)} -- pass the mask for the same crop"
        )

    asset_rgb = jewellery_asset.convert("RGBA")
    # flatten the RGBA cutout onto white before letterboxing -- Fill's image
    # input is RGB, and an unflattened alpha channel would otherwise leave
    # transparent pixels as black, which the model has no reason to treat as
    # "ignore this"
    bg = Image.new("RGBA", asset_rgb.size, (255, 255, 255, 255))
    asset_flat = Image.alpha_composite(bg, asset_rgb).convert("RGB")
    asset_letterboxed = _letterbox_to_height(asset_flat, h)

    canvas = Image.new("RGB", (w + asset_letterboxed.width, h))
    canvas.paste(person_crop.convert("RGB"), (0, 0))
    canvas.paste(asset_letterboxed, (w, 0))

    mask = Image.new("L", canvas.size, 0)
    mask.paste(Image.fromarray(stage2_mask.astype(np.uint8)), (0, 0))
    # right half (the reference asset) stays all-zero: Fill's mask-conditioned
    # denoising guarantees zero-mask pixels return identical to the input, so
    # this is what preserves fine reference detail -- no separate pixel-restore
    # step needed, same reasoning your existing Stage 4/orchestration already
    # relies on for the person side.

    return ConcatCanvas(canvas=canvas, canvas_mask=mask, person_width=w)


# ---------------------------------------------------------------------------
# Stage 4' -- single Fill pipeline + LoRA (replaces run_conditioned_inpainting)
# ---------------------------------------------------------------------------

def load_pipeline(quantize: bool = True) -> FluxFillPipeline:
    """Load FLUX.1-Fill-dev + the catvton-flux LoRA.

    quantize=True applies the same NF4 quantization pattern your existing
    Stage 4 already uses for the Fill transformer, so this fits alongside
    the current pipeline's VRAM budget. There is no ControlNet in this
    version, so the model_cpu_offload_seq exclusion your walkthrough
    documents for the controlnet does not apply here -- enable_model_cpu_offload()
    can hook every registered module.
    """
    if quantize:
        transformer = FluxTransformer2DModel.from_pretrained(
            FILL_REPO,
            subfolder="transformer",
            quantization_config=DiffusersBnbConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxFillPipeline.from_pretrained(
            FILL_REPO, transformer=transformer, torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxFillPipeline.from_pretrained(FILL_REPO, torch_dtype=torch.bfloat16)

    pipe.load_lora_weights(LORA_REPO)
    pipe.enable_model_cpu_offload()
    return pipe


def run_concat_conditioned_inpainting(
    concat: ConcatCanvas,
    category: str,
    pipe: FluxFillPipeline,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
) -> Image.Image:
    """Stage 4' -- run Fill over the concatenated canvas, crop back to the
    person half.

    Returns an RGB image the same size as the original person_crop passed
    into build_concat_canvas() -- drop this straight into the existing
    Stage 5 (lab_color_match / maybe_upscale) unchanged.
    """
    prompt = CATEGORY_PROMPTS.get(category)
    if prompt is None:
        raise ValueError(
            f"no prompt configured for category={category!r}; add one to "
            f"CATEGORY_PROMPTS (see the maang_tikka/anklet extension for "
            f"the pattern if you're adding a new category)"
        )

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=concat.canvas,
        mask_image=concat.canvas_mask,
        height=concat.canvas.height,
        width=concat.canvas.width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    return result.crop((0, 0, concat.person_width, concat.canvas.height))


# ---------------------------------------------------------------------------
# Orchestration -- mirrors run_conditioned_inpainting() in the main pipeline
# ---------------------------------------------------------------------------

def run_lite_tryon(
    person_crop: Image.Image,
    jewellery_asset: Image.Image,
    stage2_mask: np.ndarray,
    category: str,
    pipe: FluxFillPipeline,
    **kwargs,
) -> Image.Image:
    """One-call replacement for {Stage 3 + Stage 4} in the main pipeline.
    Paste the returned crop back into the full portrait at the same ROI
    offset the main pipeline's orchestration already computes -- that part
    of run_conditioned_inpainting() in landmark_guided_tryon.py is unchanged.
    """
    concat = build_concat_canvas(person_crop, jewellery_asset, stage2_mask)
    return run_concat_conditioned_inpainting(concat, category, pipe, **kwargs)


if __name__ == "__main__":
    # Smoke test: exercises build_concat_canvas() with synthetic data only --
    # no model load, no GPU. Confirms the canvas/mask geometry is internally
    # consistent before you spend a Colab session on the real thing.
    person = Image.new("RGB", (256, 256), (200, 180, 160))
    asset = Image.new("RGBA", (120, 80), (255, 215, 0, 255))
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:160, 90:170] = 255

    concat = build_concat_canvas(person, asset, mask)
    assert concat.canvas.height == 256
    assert concat.canvas.width == 256 + round(120 * 256 / 80)
    assert concat.canvas_mask.size == concat.canvas.size
    assert concat.person_width == 256
    print(f"OK -- canvas {concat.canvas.size}, mask {concat.canvas_mask.size}, "
          f"person_width={concat.person_width}", file=sys.stderr)
