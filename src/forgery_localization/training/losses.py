"""Planned binary segmentation objective: BCEWithLogitsLoss plus Dice loss."""

# TODO: Combine BCE on raw logits with Dice on sigmoid probabilities.
# TODO: Specify smoothing and behavior for empty authentic-image masks.
