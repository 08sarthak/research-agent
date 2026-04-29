"""Build prompts and call Perplexity (OpenAI-compatible) for structured JSON reports."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import quote_plus

from dotenv import load_dotenv
from openai import OpenAI

from core.schemas import FacilityItem, StructuredReport

load_dotenv()

PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
DEFAULT_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")

SYSTEM_PROMPT_JSON = """You are assisting with an educational, non-diagnostic health information assistant.

Hard rules:
- This is NOT medical advice. The user must see a qualified clinician for diagnosis and treatment.
- Never state or imply that the machine learning prediction is confirmed, ruled out, or equivalent to a doctor's diagnosis.
- Use cautious language in all text fields: "may align", "might suggest", "could be consistent with", "uncertainty remains".
- Flag urgent warning signs where appropriate in the urgent_care list.

nearby_facilities (must never be empty):
- The "nearby_facilities" array MUST always contain at least 2 entries. Never return an empty array.
- Use web search to find real dermatology clinics, hospital dermatology departments, diagnostic labs, or public health resources that serve the user's location. When you have a trustworthy page from search, set "url" to that full http(s) link copied verbatim.
- If you cannot find enough venue-specific links, still add helpful entries: give clear, practical "relevance" text (how to seek care locally, what type of provider to look for) and set "url" to a Google Maps search URL for discovering dermatology/skin care near that location. You may form it as: https://www.google.com/maps/search/?api=1&query= plus a URL-encoded query such as "dermatology clinic near [city, region]" — this is allowed as a discovery link when a single official site is not available.
- Do not invent fake hospital domains; prefer real search-backed URLs or the Maps discovery pattern above. Never use example.com or localhost.
- Respond with ONLY a single JSON object matching the schema described in the user message. No markdown fences, no commentary before or after the JSON."""


def _client() -> OpenAI:
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key or key.strip() in ("", "pplx-your-key-here"):
        raise ValueError("PERPLEXITY_API_KEY is missing. Set it in your .env file.")
    return OpenAI(api_key=key.strip(), base_url=PERPLEXITY_BASE_URL)


def build_json_user_prompt(fields: dict[str, Any]) -> str:
    ml_name = fields.get("ml_condition") or "(not provided)"
    ml_conf = fields.get("ml_confidence")
    if ml_conf is not None:
        ml_line = f"ML-predicted condition (from a separate model, for discussion only): {ml_name} (confidence: {ml_conf}%)"
    else:
        ml_line = f"ML-predicted condition (from a separate model, for discussion only): {ml_name}"

    name = fields.get("patient_name") or "Not provided"
    gender = fields.get("gender") or "Not provided"
    age = fields.get("age")
    age_s = str(age) if age is not None else "Not provided"
    loc = fields.get("location") or "Not provided"

    symptoms = fields.get("symptoms") or ""
    duration = fields.get("duration") or "Not specified"

    return f"""Using current web search where helpful, fill a JSON object with EXACTLY these keys and value types:

- "summary": string — brief summary of symptoms, duration, demographics, location, ML prediction + confidence.
- "consistency_check": string — how described symptoms might or might not align with the ML prediction; overlap, gaps, confounders; emphasize uncertainty.
- "ml_limitations": string — what an image classifier confidence does and does not mean; symptoms + ML label are not a clinical conclusion.
- "precautions": string — conservative non-prescriptive precautions (no medication names or doses).
- "urgent_care": array of strings — red-flag signs warranting prompt or emergency care.
- "nearby_facilities": array of objects with "name", "area_or_city", "relevance", "url". Minimum 2 items, never []. Prefer real venue or hospital URLs from search. If needed, include a Maps-based discovery link (see system rules) and/or entries with strong written guidance in "relevance" so users still know how to find care.
- "disclaimer": string — short paragraph that this is informational only, not diagnosis, not a substitute for a licensed professional.

Before returning, confirm nearby_facilities has at least two entries.

Input data:
{ml_line}
Symptoms: {symptoms}
Duration: {duration}
Name (optional): {name}
Gender: {gender}
Age: {age_s}
Location (for local facilities): {loc}
"""


def extract_json_object(text: str) -> str:
    """Strip optional markdown fences and isolate a JSON object."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if not t.startswith("{"):
        m = re.search(r"\{[\s\S]*\}\s*$", t)
        if m:
            t = m.group(0).strip()
    return t


def google_maps_search_url(location: str, *, focus: str = "dermatology clinic") -> str:
    """Stable Google Maps search URL for skin-care discovery near the given location."""
    q = quote_plus(f"{focus} near {location.strip()}")
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def _fallback_facility_row(location: str, maps_url: str) -> FacilityItem:
    return FacilityItem(
        name="Find dermatology & skin care near you (Google Maps)",
        area_or_city=location.strip() or "your area",
        relevance="Opens a map search for dermatology clinics and skin specialists near you. Check hours, reviews, and credentials before visiting.",
        url=maps_url,
    )


def finalize_nearby_facilities(report: StructuredReport, location: str) -> StructuredReport:
    """Ensure nearby_facilities is never empty and has at least two rows when possible; add Maps-backed rows if needed."""
    loc = (location or "").strip() or "your area"
    maps_url = google_maps_search_url(loc)
    items = list(report.nearby_facilities)
    has_http = any((f.url or "").strip().lower().startswith("http") for f in items)
    if not items:
        items = [_fallback_facility_row(loc, maps_url)]
    elif not has_http:
        items = items + [_fallback_facility_row(loc, maps_url)]
    if len(items) < 2:
        alt = google_maps_search_url(loc, focus="dermatologist skin specialist")
        if not any((f.url or "").strip() == alt for f in items):
            items.append(
                FacilityItem(
                    name="More nearby dermatology options (Google Maps)",
                    area_or_city=loc,
                    relevance="Additional map search to compare clinics and specialists. Informational only; verify licensing and hours yourself.",
                    url=alt,
                )
            )
        else:
            items.append(
                FacilityItem(
                    name="Seeking in-person care for skin concerns",
                    area_or_city=loc,
                    relevance="If specific clinic links were unavailable, consider a licensed dermatologist or general practitioner for examination. Emergency departments handle severe spreading rashes, fever with rash, or breathing difficulty.",
                    url=maps_url,
                )
            )
    return report.model_copy(update={"nearby_facilities": items})


def generate_structured_report(
    fields: dict[str, Any],
    model: str | None = None,
) -> tuple[str, StructuredReport]:
    """Call Perplexity Sonar; return (model_used, parsed report)."""
    model_id = (model or DEFAULT_MODEL).strip()
    user_content = build_json_user_prompt(fields)
    client = _client()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_JSON},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise ValueError("The model returned an empty response.")

    try:
        blob = json.loads(extract_json_object(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}") from e

    try:
        report = StructuredReport.model_validate(blob)
    except Exception as e:
        raise ValueError(f"JSON did not match the expected schema: {e}") from e

    loc = (fields.get("location") or "").strip() or "your area"
    report = finalize_nearby_facilities(report, loc)
    return model_id, report
