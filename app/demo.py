"""Web demo: upload an image -> Real/Fake, confidence, mask, overlay with boxes, area.

Run (Colab or local, after ``pip install -e ".[demo]"``)::

    python app/demo.py --checkpoint RUN/best_dice.pt [--share]

Several checkpoints may be given (e.g. RGB, Noise, DCT, SegFormer); the user
chooses one in the interface to compare models on the same image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from forgery_localization.inference import (
    Predictor, load_thresholds, mask_image, render_overlay, report_text,
)


def load_predictors(checkpoints: list[str], device: str) -> dict[str, Predictor]:
    predictors = {}
    for checkpoint in map(Path, checkpoints):
        thresholds_path = checkpoint.parent / "thresholds.json"
        thresholds = load_thresholds(thresholds_path if thresholds_path.is_file() else None)
        predictor = Predictor(checkpoint, thresholds=thresholds, device=device)
        label = predictor.name
        if thresholds["calibrated_on"] is None:
            label += " (uncalibrated)"
        predictors[label] = predictor
    return predictors


def build_interface(predictors: dict[str, Predictor]):
    import gradio as gr

    def analyse(image: Image.Image, model_name: str):
        if image is None:
            raise gr.Error("Please upload an image.")
        predictor = predictors[model_name]
        probabilities, mask, summary = predictor.predict(image)
        heatmap = Image.fromarray((probabilities * 255).astype(np.uint8))
        return (
            render_overlay(image, mask, summary),
            mask_image(mask),
            heatmap,
            report_text(summary, predictor.name),
            summary.to_dict(),
        )

    names = list(predictors)
    with gr.Blocks(title="Image Forgery Detection and Localization") as demo:
        gr.Markdown("## Image Forgery Detection and Localization\n"
                    "Upload an image to get a Real/Fake decision, a manipulation mask, "
                    "bounding boxes and the suspicious area.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Input image")
                model = gr.Dropdown(names, value=names[0], label="Model")
                button = gr.Button("Analyse", variant="primary")
            with gr.Column():
                report = gr.Markdown()
                overlay = gr.Image(label="Overlay (red = suspected, yellow = boxes)")
        with gr.Row():
            mask = gr.Image(label="Binary mask")
            heatmap = gr.Image(label="Probability map")
        details = gr.JSON(label="Raw output")
        button.click(analyse, [image, model], [overlay, mask, heatmap, report, details])
    return demo


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--share", action="store_true", help="Public Gradio link (needed on Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    demo = build_interface(load_predictors(args.checkpoint, args.device))
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
