"""Gradio UI that predicts a disease from image and generates a local report."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.agent import generate_structured_report, google_maps_search_url
from predict_disease import predict_disease

DISCLAIMER = """
**Informational only — not medical advice, diagnosis, or treatment.**  
Consult a qualified clinician. For emergencies, use local emergency services.
"""

DEFAULT_LLM_MODEL = "sonar-pro"


def _format_report_markdown(payload: dict[str, Any]) -> str:
    model_used = payload.get("model_used", "—")
    maps_url = payload.get("maps_search_url", "")
    rep = payload.get("report") or {}

    parts: list[str] = [
        "### Report metadata",
        f"- **Model used:** `{model_used}`",
    ]
    if maps_url:
        parts.append(f"- **Google Maps (near you):** [Open map search]({maps_url})")

    parts.extend(
        [
            "",
            "### Summary",
            rep.get("summary", "—"),
            "",
            "### Symptom ↔ ML prediction",
            rep.get("consistency_check", "—"),
            "",
            "### Limits of the ML output",
            rep.get("ml_limitations", "—"),
            "",
            "### Precautions",
            rep.get("precautions", "—"),
            "",
            "### When to seek urgent care",
        ]
    )
    for line in rep.get("urgent_care") or []:
        parts.append(f"- {line}")
    if not rep.get("urgent_care"):
        parts.append("- —")

    parts.extend(["", "### Nearby facilities & links"])
    facs = rep.get("nearby_facilities") or []
    if not facs:
        parts.append("- None returned.")
    else:
        for i, f in enumerate(facs, 1):
            name = f.get("name", "—")
            area = f.get("area_or_city", "—")
            rel = f.get("relevance", "—")
            url = (f.get("url") or "").strip()
            parts.append(f"**{i}. {name}** — *{area}*")
            parts.append(f"{rel}")
            if url:
                parts.append(f"[Link]({url})")
            parts.append("")

    parts.extend(
        [
            "### Disclaimer",
            rep.get("disclaimer", "—"),
        ]
    )
    return "\n".join(parts).strip()


LOADING_MD = """
<div align="center">

### Generating report…

Please wait — this can take up to a minute while the model runs.

</div>
""".strip()


def submit_report(
    image: str | None,
    symptoms: str,
    duration: str,
    patient_name: str,
    gender: str,
    age: float | None,
    location: str,
) -> Iterator[tuple[str, gr.Button]]:
    yield LOADING_MD, gr.Button(value="Generating report...", interactive=False)

    if not image:
        yield "**Error:** Please upload an image.", gr.Button(
            value="Generate report", interactive=True
        )
        return
    if not (symptoms or "").strip():
        yield "**Error:** Please enter symptoms.", gr.Button(
            value="Generate report", interactive=True
        )
        return
    if not (location or "").strip():
        yield "**Error:** Please enter a location.", gr.Button(
            value="Generate report", interactive=True
        )
        return

    try:
        ml_condition, ml_confidence = predict_disease(image)
    except Exception as e:
        yield f"**Prediction error:** {e}", gr.Button(
            value="Generate report", interactive=True
        )
        return

    fields: dict[str, Any] = {
        "symptoms": symptoms.strip(),
        "location": location.strip(),
        "ml_condition": ml_condition,
        "ml_confidence": int(ml_confidence),
        "duration": duration.strip() or None,
        "patient_name": patient_name.strip() or None,
        "gender": gender if gender != "Prefer not to say" else None,
        "age": int(age) if age is not None else None,
    }
    fields = {k: v for k, v in fields.items() if v is not None}

    try:
        model_used, report = generate_structured_report(fields, model=DEFAULT_LLM_MODEL)
        maps_search_url = google_maps_search_url(fields["location"])
        data = {
            "model_used": model_used,
            "maps_search_url": maps_search_url,
            "report": report.model_dump(),
        }
    except Exception as e:
        yield f"**Report generation error:** {type(e).__name__}: {e}", gr.Button(
            value="Generate report", interactive=True
        )
        return

    yield _format_report_markdown(data), gr.Button(value="Generate report", interactive=True)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Skin report assistant") as demo:
        gr.Markdown(
            "# Skin condition report assistant\n"
            "Upload an image and fill the form below to generate an informational report."
        )
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Skin image",
                    type="filepath",
                )

                gr.Markdown("### Symptoms")
                symptoms = gr.Textbox(
                    label="Symptoms",
                    placeholder="Describe your symptoms…",
                    lines=5,
                )
                duration = gr.Textbox(
                    label="Duration",
                    placeholder="e.g., About 3 weeks",
                    lines=1,
                )

                gr.Markdown("### Personal details")
                patient_name = gr.Textbox(label="Name (optional)", lines=1)
                gender = gr.Dropdown(
                    choices=["Prefer not to say", "Female", "Male", "Non-binary", "Other"],
                    value="Male",
                    label="Gender",
                )
                age = gr.Number(label="Age", precision=0, value=30)
                location = gr.Textbox(
                    label="Location",
                    placeholder="e.g., Delhi, India",
                    lines=2,
                )

                submit = gr.Button("Generate report", variant="primary", interactive=True)

            with gr.Column(scale=1):
                gr.Markdown("### Report")
                report_md = gr.Markdown()

        submit.click(
            fn=submit_report,
            inputs=[
                image,
                symptoms,
                duration,
                patient_name,
                gender,
                age,
                location,
            ],
            outputs=[report_md, submit],
            show_progress="full",
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
