"""Dummy image classifier placeholder used by the Gradio UI."""

from __future__ import annotations


def predict_disease(image_path: str | None) -> tuple[str, int]:
    """Return a hardcoded skin disease prediction and confidence."""
    if not image_path:
        raise ValueError("Image input is required for prediction.")
    return "Atopic dermatitis (eczema)", 87
